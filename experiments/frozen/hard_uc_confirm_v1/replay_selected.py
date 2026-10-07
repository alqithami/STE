#!/usr/bin/env python3
"""Independently replay every selected checkpoint against archived held-out inputs.

Replay verifies implementation and artifact provenance; it is not a new training
experiment or an extra inferential endpoint. CPU replay is allowed for archived
GPU results and is labeled explicitly.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'frozen'))
from ste.models import PairModel
from ste.operators import hard_core
from ste.common import configure_torch, seed_for
from analyze import atomic_json, digest, effective_config
from run_experiment import predict


def replay(results, config, report=None, device_name='cpu'):
    results = Path(results).resolve()
    cfg = effective_config(results, config)
    root_marker = results / ('RUN_COMPLETE.json' if (results / 'RUN_COMPLETE.json').is_file() else 'SHARD_COMPLETE.json')
    if not root_marker.is_file(): raise RuntimeError('No completed full run or completed shard')
    lock = json.loads((results / 'RUN_LOCK.json').read_text())
    if device_name.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA replay requested but unavailable')
    device = torch.device(device_name)
    configure_torch(seed_for(cfg['seed'], 'independent-selected-state-replay'), device)
    started = time.perf_counter(); rows = []
    totals = {'P': 0., 'direct': 0., 'structural': 0.}
    for c in lock['collection_ids']:
        cp = results / 'learning' / f'collection_{c:02d}'
        for init in range(cfg['learning']['initializations']):
            ip = cp / 'uc' / f'init_{init:02d}'
            choices_path = ip / 'CHOICES_BEFORE_TEST.json'
            choices = json.loads(choices_path.read_text())
            freeze = json.loads((ip / 'SELECTION_FROZEN.json').read_text())
            if freeze['choices_sha256'] != digest(choices_path): raise RuntimeError('Changed choice bytes')
            for key, choice in choices.items():
                objective = choice['objective']
                checkpoint = (results / choice['checkpoint']).resolve()
                if not checkpoint.is_relative_to(results) or not checkpoint.is_file(): raise RuntimeError('Missing or unsafe selected state')
                model = PairModel(cfg['learning']['hidden'], cfg['learning']['systems'][objective]['training_objective'] == 'relational_aux').to(device)
                state = torch.load(checkpoint, map_location=device, weights_only=True)
                if state['objective'] != objective or state['checkpoint_index'] != choice['checkpoint_index'] or state['weight'] != choice['weight']:
                    raise RuntimeError('Selected state metadata differs from development-only choice')
                model.load_state_dict(state['model'], strict=True)
                for n in cfg['learning']['test_sizes']:
                    raw_path = cp / 'datasets' / f'test_n{n}.npz'
                    with np.load(raw_path, allow_pickle=False) as z: data = {name: z[name] for name in z.files}
                    predpath = ip / 'predictions' / f"{objective}_l{choice['weight']:g}_ck{choice['checkpoint_index']:02d}_n{n}.npz"
                    marker = json.loads(predpath.with_suffix(predpath.suffix + '.sha256.json').read_text())
                    if marker['sha256'] != digest(predpath) or marker['checkpoint_sha256'] != digest(checkpoint) or marker['choices_sha256'] != digest(choices_path):
                        raise RuntimeError('Selected prediction/state hash chain failed')
                    with np.load(predpath, allow_pickle=False) as z: saved = {name: z[name] for name in z.files}
                    for field, raw_field in [('y','y_uc'), ('case_id','case_id'), ('family','family')]:
                        if not np.array_equal(saved[field], data[raw_field]): raise RuntimeError('Prediction alignment differs from raw held-out split')
                    computed = predict(model, data, cfg, device)
                    errors = {}
                    for field in ('P', 'direct', 'structural'):
                        if computed[field].shape != saved[field].shape or not np.isfinite(saved[field]).all(): raise RuntimeError('Invalid replay shape or nonfinite output')
                        error = float(np.max(np.abs(computed[field].astype(np.float64)-saved[field].astype(np.float64))))
                        tolerance = 5e-5 if field == 'structural' else 1e-5
                        if not np.allclose(computed[field], saved[field], atol=tolerance, rtol=1e-5):
                            raise RuntimeError(f'Selected checkpoint replay disagrees: {objective}, c{c}, init{init}, n{n}, {field}, max_error={error}')
                        errors[field] = error; totals[field] = max(totals[field], error)
                    decoded_saved = np.array([hard_core(matrix > .5, 'uc') for matrix in saved['P']])
                    if not np.array_equal(decoded_saved, saved['hard_uc']) or not np.array_equal(computed['hard_uc'], saved['hard_uc']):
                        raise RuntimeError('Hard-UC selected checkpoint replay changes a deployed set; inspect near-majority ties')
                    rows.append({'collection': c, 'initialization': init, 'system': objective, 'n': n,
                                 'cases': len(saved['y']), 'checkpoint': str(checkpoint.relative_to(results)),
                                 'checkpoint_sha256': digest(checkpoint), 'prediction_sha256': digest(predpath),
                                 'maximum_absolute_errors': errors, 'hard_uc_exact_match': True})
                    print(f'REPLAY_OK collection={c} init={init} system={objective} n={n} cases={len(saved["y"])}', flush=True)
    summary = {'status': 'STE_HARDUC_SELECTED_CHECKPOINT_REPLAY_VERIFIED', 'device': str(device),
               'smoke': bool(cfg.get('smoke')), 'collections': lock['collection_ids'], 'prediction_units': len(rows),
               'case_outputs': sum(row['cases'] for row in rows), 'maximum_absolute_errors': totals,
               'seconds': time.perf_counter()-started, 'units': rows,
               'notice': 'Replay is artifact/software validation; it is not new training, independent replication, or additional statistical evidence.'}
    target = Path(report) if report else results / 'SELECTED_CHECKPOINT_REPLAY.json'
    atomic_json(target, summary)
    print(json.dumps({key:value for key,value in summary.items() if key != 'units'}, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=CODE / 'config.json')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda','cuda:0'])
    args = parser.parse_args()
    replay(args.results, args.config, args.report, args.device)
