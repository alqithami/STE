#!/usr/bin/env python3
"""Package only audited outputs, retaining originals and verifying every ZIP byte."""
from __future__ import annotations
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

# These are the same non-payload files excluded by the archive audit. Markers
# are checked through their explicit hash chain below, rather than self-hashed.
AUDIT_EXCLUDED_NAMES = {
    'RUNNING.lock', 'run.log', 'FIT_COMPLETE.json', 'COMPLETE.json',
    'AUDIT.json', 'FAILED.json',
}

def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_audited_results(out):
    """Require the audit chain and exactly the files covered by that audit."""
    out = Path(out)
    from run_experiment import source_hashes, SOFTWARE_RELEASE
    lock=json.loads((out/'RUN_LOCK.json').read_text())
    if lock.get('config',{}).get('release') != SOFTWARE_RELEASE:
        raise RuntimeError('Corrected v2 packaging refuses historical v1 release')
    if lock.get('source_sha256') != source_hashes():
        raise RuntimeError('Current source differs from audited run lock; original source required')
    complete = json.loads((out/'COMPLETE.json').read_text())
    fit_path, audit_path = out/'FIT_COMPLETE.json', out/'AUDIT.json'
    fit = json.loads(fit_path.read_text())
    audit = json.loads(audit_path.read_text())
    if complete.get('status') != 'COMPLETE' or audit.get('status') != 'AUDIT_COMPLETE':
        raise RuntimeError('Audited COMPLETE/AUDIT status required')
    if fit.get('status') != 'FIT_COMPLETE_AWAITING_AUDIT':
        raise RuntimeError('Completed fitting marker required')
    for field, path in (('FIT_COMPLETE_sha256', fit_path), ('AUDIT_sha256', audit_path)):
        if complete.get(field) != digest(path.read_bytes()):
            raise RuntimeError('Changed audit-chain marker: '+path.name)
    for field in ('coverage_verified', 'source_hashes_verified', 'data_hashes_verified',
                  'checkpoint_hashes_verified', 'selected_replay_verified'):
        if audit.get(field) is not True or complete.get(field) is not True:
            raise RuntimeError('Required audit verification missing: '+field)
    for field, value in audit.items():
        if field != 'status' and complete.get(field) != value:
            raise RuntimeError('COMPLETE disagrees with AUDIT: '+field)
    actual = {path.relative_to(out).as_posix() for path in out.rglob('*')
              if path.is_file() and path.name not in AUDIT_EXCLUDED_NAMES}
    hashes = fit.get('hashes')
    if not isinstance(hashes, dict) or actual != set(hashes):
        raise RuntimeError('Unhashed or missing audited output artifact')
    for name, expected in hashes.items():
        if digest((out/name).read_bytes()) != expected:
            raise RuntimeError('Changed audited payload: '+name)
    return complete

def make_archive(destination, payload, classification):
    destination = Path(destination)
    if destination.exists() or destination.with_suffix(destination.suffix + '.sha256').exists():
        raise RuntimeError('Existing archive refused; preserve evidence and use a new destination')
    manifest = {'classification': classification,
                'payload_sha256': {name: digest(data) for name, data in payload.items()}}
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(payload.items()):
            z.writestr(name, data)
        z.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP CRC failed')
        for name, expected in manifest['payload_sha256'].items():
            if digest(z.read(name)) != expected:
                raise RuntimeError('Archive payload hash failed: ' + name)
    value = digest(destination.read_bytes())
    with destination.with_suffix(destination.suffix + '.sha256').open('x') as handle:
        handle.write(value+'  '+destination.name+'\n')
    return {'file': str(destination), 'bytes': destination.stat().st_size, 'sha256': value,
            'payload_files': len(payload)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); out = a.out.resolve()
    complete = validate_audited_results(out)
    full, review = {}, {}
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path.name in {'RUNNING.lock','run.log'} or path.suffix in {'.zip','.sha256'}:
            continue
        name='results/'+path.relative_to(out).as_posix()
        data=path.read_bytes(); full[name]=data
        if path.suffix not in {'.pt','.npz'}:
            review[name]=data
        elif path.suffix == '.npz' and 'prediction' in str(path.relative_to(out)).lower():
            with np.load(path) as arrays:
                compact={k:arrays[k] for k in arrays.files if k not in {'memberships','component_ids'}}
            b=io.BytesIO();np.savez_compressed(b,**compact)
            review['compact_'+name]=b.getvalue()
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in {'.pyc','.zip'}:
            continue
        if any(part in {'validation','output','tmp'} for part in path.relative_to(HERE).parts):
            continue
        name='code/'+path.relative_to(HERE).as_posix(); data=path.read_bytes()
        full[name]=data
        if path.suffix in {'.md','.json','.py','.sh','.txt','.tex'} and 'reference' not in path.relative_to(HERE).parts:
            review[name]=data
    review['REVIEW_SCOPE.txt']=(
        'The full archive retains selected checkpoints and raw tournament membership samples.\n'
        'Compact review predictions omit membership draw arrays and component IDs; they are derived copies.\n'
        'Archive integrity and saved-state replay do not constitute independent retraining or population validation.\n'
        'Pilot and smoke results are not production findings.\n').encode()
    from run_experiment import SOFTWARE_RELEASE
    cfg=json.loads((out/'CONFIG.json').read_text())
    if cfg.get('release') != SOFTWARE_RELEASE:
        raise RuntimeError('Corrected v2 packager refuses historical v1 output; use frozen v1 packager')
    base=out.parent/(SOFTWARE_RELEASE.replace('-','_')+'_'+out.name)
    records=[make_archive(base.with_name(base.name+'_FULL.zip'),full,complete),
             make_archive(base.with_name(base.name+'_REVIEW.zip'),review,complete)]
    print(json.dumps({'status':'ARCHIVES_VERIFIED','archives':records},indent=2))

if __name__=='__main__':
    main()
