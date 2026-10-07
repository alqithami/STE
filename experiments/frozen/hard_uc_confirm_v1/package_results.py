#!/usr/bin/env python3
"""Package immutable, independently replayable result archives and upload chunks."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'frozen'))
from ste.common import unit_complete
from analyze import atomic_json, effective_config
from verify_archive import verify, verify_outroot, verify_parts, FULL_MARKER, SMOKE_MARKER, SHARD_MARKER, sha

MAX_PART_BYTES = 190 * 1024 * 1024


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def selected_checkpoints(results):
    selected = set()
    for path in sorted((results / 'learning').glob('collection_*/uc/init_*/CHOICES_BEFORE_TEST.json')):
        for choice in json.loads(path.read_text()).values():
            checkpoint = (results / choice['checkpoint']).resolve()
            if not checkpoint.is_relative_to(results.resolve()) or not checkpoint.is_file():
                raise RuntimeError('Selected checkpoint is missing or escapes run root')
            selected.add(checkpoint)
    return selected


def payload_files(code, results, review):
    selected = selected_checkpoints(results)
    items, omissions = [], []
    for p in sorted(code.rglob('*')):
        if not p.is_file() or any(part in ('.venv', '__pycache__', '.git') for part in p.relative_to(code).parts) or p.suffix in ('.pyc', '.tmp', '.zip'):
            continue
        if p.is_symlink(): raise RuntimeError('Refusing a source symlink: ' + str(p))
        items.append((p, 'code/' + p.relative_to(code).as_posix()))
    for p in sorted(results.rglob('*')):
        if not p.is_file() or p.name in ('RUN.lock', 'RUNNING.lock', 'ARCHIVES_READY.json', 'SHARD_ARCHIVES_READY.json') or p.suffix in ('.tmp', '.zip', '.sha256'):
            continue
        if any(part in ('__pycache__', '.venv') for part in p.relative_to(results).parts): continue
        if p.is_symlink(): raise RuntimeError('Refusing a result symlink: ' + str(p))
        member = 'run/' + p.relative_to(results).as_posix()
        omit = review and ((p.suffix == '.pt' and p.name != 'initial.pt' and p.resolve() not in selected) or (p.name == 'train.npz' and p.parent.name == 'datasets') or p.name == 'history.json')
        if omit:
            omissions.append({'path': member, 'bytes': p.stat().st_size, 'sha256': sha(p)})
        else:
            items.append((p, member))
    return items, omissions


def create_archive(path, code, results, cfg, constraints, review, shard=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    items, omissions = payload_files(code, results, review)
    lock = json.loads((results / 'RUN_LOCK.json').read_text())
    manifest = {'release': 'STE_HardUC_Confirm_v1', 'smoke': bool(cfg.get('smoke', False)), 'shard': bool(shard),
                'review': bool(review), 'collection_ids': lock['collection_ids'],
                'classification': 'SOFTWARE QA; NOT SCIENTIFIC EVIDENCE' if cfg.get('smoke') else ('completed subset; no global inference' if shard else 'complete prospective follow-up; completion does not imply superiority'),
                'primary_inference_valid': bool(constraints.get('primary_inference_valid', False)),
                'budget_valid': constraints.get('budget_valid'),
                'log_policy': 'Logs are the exact bytes captured during packaging. Subsequent messages are outside this immutable snapshot.',
                'review_omissions': omissions, 'files': []}
    with temp.open('w+b') as handle:
        with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for source, member in items:
                data = source.read_bytes()
                manifest['files'].append({'path': member, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
                z.writestr(member, data)
            z.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path); fsync_directory(path.parent)
    checksum = path.with_name(path.name + '.sha256')
    checksum.write_text(sha(path) + '  ' + path.name + '\n')
    result = verify(path)
    return {**result, 'name': path.name, 'checksum_name': checksum.name}


def split_archive(path, max_bytes=MAX_PART_BYTES):
    """Upload-size chunks are transport only; they never create independent results."""
    path = Path(path)
    if max_bytes <= 0: raise ValueError('Part byte limit must be positive')
    parts = []
    with path.open('rb') as source:
        index = 1
        while True:
            data = source.read(max_bytes)
            if not data: break
            target = path.with_name(path.name + f'.part{index:03d}')
            temporary = target.with_name(target.name + '.tmp')
            with temporary.open('wb') as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(temporary, target)
            parts.append({'name': target.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            index += 1
    manifest = path.with_name(path.stem + '_PARTS.json')
    atomic_json(manifest, {'archive_name': path.name, 'archive_bytes': path.stat().st_size, 'archive_sha256': sha(path),
                           'maximum_part_bytes': max_bytes, 'parts': parts,
                           'instruction': 'Upload every part and this JSON; reassemble in numerical order and verify the archive SHA256 before reading results.'})
    verify_parts(manifest)
    return {'manifest_name': manifest.name, 'parts': parts}


def package(results, outroot, config, shard=False):
    results, outroot = Path(results).resolve(), Path(outroot).resolve()
    if outroot.is_relative_to(results):
        raise RuntimeError('Archive directory must not be inside the run result directory')
    cfg = effective_config(results, config)
    smoke = bool(cfg.get('smoke', False))
    lock = json.loads((results / 'RUN_LOCK.json').read_text())
    replay = json.loads((results / 'SELECTED_CHECKPOINT_REPLAY.json').read_text())
    expected_units = len(lock['collection_ids']) * cfg['learning']['initializations'] * len(cfg['learning']['objectives']) * len(cfg['learning']['test_sizes'])
    if replay.get('status') != 'STE_HARDUC_SELECTED_CHECKPOINT_REPLAY_VERIFIED' or replay.get('collections') != lock['collection_ids'] or replay.get('prediction_units') != expected_units or bool(replay.get('smoke')) != smoke:
        raise RuntimeError('Every selected checkpoint must replay successfully before packaging')
    outroot.mkdir(parents=True, exist_ok=True)
    if shard:
        if not (results / 'SHARD_COMPLETE.json').is_file() or not unit_complete(results / 'learning'):
            raise RuntimeError('All selected collection units must complete before shard packaging')
        marker_path = outroot / 'SHARD_ARCHIVES_READY.json'; marker_path.unlink(missing_ok=True)
        label = '-'.join(str(value) for value in lock['collection_ids'])
        prefix = 'STE_HardUC_Confirm_v1' + ('_SMOKE' if smoke else '') + '_SHARD_' + label
        archive = create_archive(outroot / (prefix + '.zip'), CODE, results, cfg, {}, False, True)
        marker = {'status': SHARD_MARKER, 'smoke': smoke, 'collection_ids': lock['collection_ids'], 'archive': archive,
                  'integrity_complete': True, 'primary_inference_valid': False, 'budget_valid': None}
        if archive['bytes'] > MAX_PART_BYTES: marker['transport'] = split_archive(outroot / archive['name'])
        atomic_json(marker_path, marker)
        result = verify_outroot(outroot, shard=True)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print(SHARD_MARKER, flush=True)
        print('Share completed shard: ' + str(outroot / archive['name']), flush=True)
        return result
    if not (results / 'RUN_COMPLETE.json').is_file() or not unit_complete(results / 'analysis'):
        raise RuntimeError('Full run and verified analysis must complete before full packaging')
    constraints = json.loads((results / 'analysis' / 'INTERPRETATION_CONSTRAINTS.json').read_text())
    if smoke != bool(constraints['smoke']): raise RuntimeError('Analysis classification differs from effective run')
    if not smoke and (cfg['learning']['collections'] != 12 or cfg['learning']['initializations'] != 3):
        raise RuntimeError('Refusing complete-study archive for an incomplete prospective protocol')
    (outroot / 'ARCHIVES_READY.json').unlink(missing_ok=True)
    prefix = 'STE_HardUC_Confirm_v1_SMOKE' if smoke else 'STE_HardUC_Confirm_v1'
    full = create_archive(outroot / (prefix + '_RESULTS.zip'), CODE, results, cfg, constraints, False)
    review = create_archive(outroot / (prefix + '_REVIEW.zip'), CODE, results, cfg, constraints, True)
    marker = {'status': SMOKE_MARKER if smoke else FULL_MARKER, 'smoke': smoke, 'full': full, 'review': review,
              'budget_valid': bool(constraints['budget_valid']), 'primary_inference_valid': bool(constraints['primary_inference_valid']),
              'integrity_complete': True, 'scientific_notice': 'Integrity and completion do not imply superiority or valid equal-budget inference.'}
    if review['bytes'] > MAX_PART_BYTES: marker['review_transport'] = split_archive(outroot / review['name'])
    atomic_json(outroot / 'ARCHIVES_READY.json', marker); fsync_directory(outroot)
    checked = verify_outroot(outroot)
    print(json.dumps(checked, indent=2, sort_keys=True), flush=True)
    print(marker['status'], flush=True)
    print('Share: ' + str(outroot / review['name']), flush=True)
    if 'review_transport' in marker: print('If upload is too large, share all transport parts and ' + str(outroot / marker['review_transport']['manifest_name']), flush=True)
    print('Keep full raw archive: ' + str(outroot / full['name']), flush=True)
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--outroot', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--shard', action='store_true')
    args = parser.parse_args()
    package(args.results, args.outroot, args.config, args.shard)
