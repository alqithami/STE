#!/usr/bin/env python3
"""Combine disjoint verified collection shards; never combine inference statistics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'frozen'))
from ste.common import atomic_json, finish_unit, sha, unit_complete, write_rows
from run_experiment import source_fingerprint, validate_config


def check_shard_coverage(locks, expected):
    seen = set()
    for lock in locks:
        ids = lock['collection_ids']
        if not ids or len(ids) != len(set(ids)) or any(x not in expected for x in ids):
            raise RuntimeError('Malformed or out-of-range shard collection identifiers')
        overlap = seen.intersection(ids)
        if overlap:
            raise RuntimeError('Duplicated independent collection IDs: ' + repr(sorted(overlap)))
        seen.update(ids)
    if seen != set(expected):
        raise RuntimeError('Full inference requires all collections; missing ' + repr(sorted(set(expected) - seen)))
    return sorted(seen)


def compatible_runtime(left, right):
    fields = ('torch', 'numpy', 'scipy', 'pandas', 'cuda_runtime', 'gpu_name', 'gpu_capability')
    return all(left.get(k) == right.get(k) for k in fields)


def locate_results(path):
    path = Path(path).resolve()
    if path.is_dir():
        choices = [path] if (path / 'RUN_LOCK.json').is_file() else list(path.rglob('RUN_LOCK.json'))
        if len(choices) != 1:
            raise RuntimeError('Expected exactly one shard results root: ' + str(path))
        return choices[0] if choices[0].is_dir() else choices[0].parent
    raise RuntimeError('Missing extracted results directory: ' + str(path))


def _merge_directories(inputs, out):
    roots = [locate_results(p) for p in inputs]
    locks, protocols, markers = [], [], []
    for root in roots:
        lock = json.loads((root / 'RUN_LOCK.json').read_text())
        protocol = json.loads((root / 'PROTOCOL_LOCK.json').read_text())
        cfg = protocol['config']
        validate_config(cfg)
        if cfg.get('smoke') or cfg.get('pilot') or cfg['learning']['collections'] != 12:
            raise RuntimeError('Software QA and timing trials cannot be merged into a scientific run')
        if protocol['source_sha256'] != source_fingerprint():
            raise RuntimeError('Delivered source changed after shard launch')
        if {k: lock[k] for k in ('config', 'supplied_config', 'source_sha256')} != protocol:
            raise RuntimeError('Shard run identity differs from its global protocol')
        marker_path = root / ('SHARD_COMPLETE.json' if (root / 'SHARD_COMPLETE.json').is_file() else 'RUN_COMPLETE.json')
        item = json.loads(marker_path.read_text())
        if not unit_complete(root / 'learning'):
            raise RuntimeError('Incomplete collection shard: ' + str(root))
        if item['run_lock_sha256'] != sha(root / 'RUN_LOCK.json') or item['learning_complete_sha256'] != sha(root / 'learning' / 'COMPLETE.json'):
            raise RuntimeError('Shard completion identity/hash mismatch')
        if item['collection_ids'] != lock['collection_ids']:
            raise RuntimeError('Shard marker and run lock IDs differ')
        if not str(lock['runtime'].get('device', '')).startswith('cuda'):
            raise RuntimeError('Scientific shards must have run on CUDA')
        actual = sorted(int(p.name.split('_')[-1]) for p in (root / 'learning').glob('collection_*') if p.is_dir())
        if actual != sorted(lock['collection_ids']):
            raise RuntimeError('Unexpected or missing collection directories in shard')
        locks.append(lock); protocols.append(protocol); markers.append(item)
    if any(protocol != protocols[0] for protocol in protocols):
        raise RuntimeError('Shards do not share one immutable protocol/config/source identity')
    if any(not compatible_runtime(locks[0]['runtime'], lock['runtime']) for lock in locks):
        raise RuntimeError('Production shards require identical numerical versions, GPU model and capability')
    cfg = protocols[0]['config']
    ids = check_shard_coverage(locks, list(range(cfg['learning']['collections'])))
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Use a new empty merge output; existing run data are never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    learning = out / 'learning'; learning.mkdir()
    provenance = []
    for index, (root, lock, marker) in enumerate(zip(roots, locks, markers)):
        evidence = out / 'SHARDS' / f'shard_{index:02d}'; evidence.mkdir(parents=True)
        for name in ('RUN_LOCK.json', 'PROTOCOL_LOCK.json', 'ENVIRONMENT.json', 'SHARD_COMPLETE.json', 'RUN_COMPLETE.json', 'COMMON_WARMUP.json'):
            if (root / name).is_file(): shutil.copy2(root / name, evidence / name)
        for c in lock['collection_ids']:
            cp = root / 'learning' / f'collection_{c:02d}'
            if not unit_complete(cp): raise RuntimeError('Collection failed recursive hash verification')
            shutil.copytree(cp, learning / cp.name)
        provenance.append({'shard_index': index, 'collection_ids': lock['collection_ids'],
                           'run_lock_sha256': sha(root / 'RUN_LOCK.json'),
                           'learning_complete_sha256': sha(root / 'learning' / 'COMPLETE.json'),
                           'runtime': lock['runtime'], 'source_path_not_retained': True})
    lock = {**protocols[0], 'runtime': locks[0]['runtime'], 'collection_ids': ids}
    atomic_json(out / 'RUN_LOCK.json', lock)
    atomic_json(out / 'PROTOCOL_LOCK.json', protocols[0])
    atomic_json(out / 'CONFIG.json', cfg)
    atomic_json(out / 'FROZEN_CONFIG.json', protocols[0]['supplied_config'])
    atomic_json(out / 'ENVIRONMENT.json', {'merged_verified_shards': provenance, 'scientific_training_device': 'cuda', 'merging_is_not_training': True})
    atomic_json(out / 'MERGE_PROVENANCE.json', {'collection_ids': ids, 'disjoint': True, 'all_twelve': True, 'shards': provenance,
                 'inference_policy': 'No shard-level tests are pooled. Analysis recomputes all twelve independent collection means.'})
    dataset_rows = [{'path': str(p.relative_to(out)), 'sha256': sha(p)} for p in sorted(learning.glob('collection_*/datasets/*.npz'))]
    write_rows(learning / 'DATASET_HASHES.csv', dataset_rows)
    finish_unit(learning, [learning / 'DATASET_HASHES.csv', *sorted(learning.glob('collection_*/COMPLETE.json'))])
    timings = [json.loads(p.read_text()) for p in learning.glob('collection_*/uc/init_*/*_lambda_*/timing.json')]
    selection_seconds = sum(json.loads(p.read_text())['charged_seconds_total'] for p in learning.glob('collection_*/uc/init_*/SELECTION_COST.json'))
    item = {'release': cfg['release'], 'complete': True, 'smoke': False, 'pilot': False, 'scientific_run': True,
            'collections': len(ids), 'collection_ids': ids, 'global_collections': len(ids), 'full_global_run': True,
            'initializations': cfg['learning']['initializations'], 'candidate_count': len(timings),
            'dataset_count': len(dataset_rows), 'all_candidates_within_fixed_tolerance': all(t['budget_valid'] for t in timings),
            'charged_candidate_seconds_total': sum(t['charged_seconds'] for t in timings),
            'charged_final_selection_seconds_total': selection_seconds,
            'charged_complete_system_seconds_total': sum(t['charged_seconds'] for t in timings) + selection_seconds,
            'allocated_candidate_seconds_total': sum(t['budget_seconds'] for t in timings),
            'learning_complete_sha256': sha(learning / 'COMPLETE.json'), 'run_lock_sha256': sha(out / 'RUN_LOCK.json'),
            'merged_shards': len(roots), 'formal_inference': 'Requires complete independent analysis and fixed ±2% budget audit.'}
    atomic_json(out / 'RUN_COMPLETE.json', item)
    print(json.dumps({'status': 'GLOBAL_TRAINING_MERGED_NOT_YET_ANALYZED', 'collections': ids, 'results': str(out)}), flush=True)
    return item


def merge(inputs, out):
    """Verify archives before safe extraction, then copy complete collection units."""
    from verify_archive import verify
    with tempfile.TemporaryDirectory(prefix='ste-harduc-verified-merge-') as temporary:
        roots = []
        for index, item in enumerate(inputs):
            item = Path(item).resolve()
            if item.is_file():
                checked = verify(item)
                if not checked['shard']:
                    raise RuntimeError('Merge archive inputs must be completed shards, not full or review archives')
                destination = Path(temporary) / f'shard_{index:02d}'
                destination.mkdir()
                with zipfile.ZipFile(item) as archive:
                    for info in archive.infolist():
                        if (info.external_attr >> 16) & 0o170000 == 0o120000:
                            raise RuntimeError('Archive symlinks are not allowed')
                    archive.extractall(destination)
                roots.append(destination / 'run')
            else:
                roots.append(item)
        return _merge_directories(roots, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    merge(args.inputs, args.out)
