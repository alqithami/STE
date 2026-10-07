#!/usr/bin/env python3
"""Fixed synthetic certificate diagnostics; never a human-data coverage claim."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from stepc.certificates import simultaneous_hoeffding_intervals, certificate_bounds
from stepc.posterior import exact_uc


def model(n, family, seed):
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(n, 1)
    P = np.full((n, n), .5)
    signs = rng.choice([-1, 1], len(i))
    gap = .15 if family == 'uniform_margin' else .01
    P[i, j] = .5 + gap * signs
    P[j, i] = 1 - P[i, j]
    if family == 'condorcet_weak_rest':
        P[0, 1:] = .85
        P[1:, 0] = .15
    elif family == 'cycle_core_weak_rest':
        P[:3, 3:] = .85
        P[3:, :3] = .15
        for a, b in ((0, 1), (1, 2), (2, 0)):
            P[a, b], P[b, a] = .85, .15
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    from run_experiment import validate_future_output_path
    args.out=validate_future_output_path(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    seed, delta = 20261007, .05
    repetitions = 8 if args.smoke else 256
    sizes = [12] if args.smoke else [12, 24]
    pair_counts = [64, 256] if args.smoke else [16, 64, 256, 1024]
    families = ['condorcet_weak_rest', 'cycle_core_weak_rest', 'uniform_margin']
    started = time.perf_counter()
    rows, raw = [], []
    for fi, family in enumerate(families):
        for n in sizes:
            P = model(n, family, seed + fi * 1000 + n)
            target = exact_uc(P > .5)
            ii, jj = np.triu_indices(n, 1)
            for m in pair_counts:
                rng = np.random.default_rng(seed + fi * 100000 + n * 1000 + m)
                counts = rng.binomial(m, P[ii, jj], size=(repetitions, len(ii)))
                full_orientation, exact_certificate, covered, violations = 0, 0, 0, 0
                widths = []
                for rep in range(repetitions):
                    W = np.zeros((n, n), dtype=np.int64)
                    W[ii, jj], W[jj, ii] = counts[rep], m - counts[rep]
                    intervals = simultaneous_hoeffding_intervals(W, delta)
                    bounds = certificate_bounds(intervals)
                    envelope = bool(np.all(P[ii, jj] >= intervals.lower[ii, jj])
                                    and np.all(P[ii, jj] <= intervals.upper[ii, jj]))
                    sound = bool(np.all(~bounds.inner | target) and np.all(~target | bounds.outer))
                    certain = bool(np.array_equal(bounds.inner, bounds.outer))
                    all_edges = bool(np.all((intervals.known | intervals.known.T)[ii, jj]))
                    if envelope and not sound:
                        raise AssertionError('A certificate failed on a covered probability matrix')
                    full_orientation += all_edges
                    exact_certificate += certain
                    covered += envelope
                    violations += not sound
                    widths.append(int(np.sum(bounds.outer) - np.sum(bounds.inner)))
                    raw.append({'family': family, 'n': n, 'count_per_pair': m, 'replicate': rep,
                                'simultaneous_interval_contains_P': envelope,
                                'certificate_contains_true_UC': sound,
                                'certificate_is_exact': certain, 'all_edges_oriented': all_edges,
                                'inner_size': int(bounds.inner.sum()), 'outer_size': int(bounds.outer.sum()),
                                'target_size': int(target.sum())})
                tag = f'{family}_n{n}_m{m}'
                np.savez_compressed(args.out / (tag + '.npz'), P=P, true_uc=target, upper_wins=counts,
                                    upper_i=ii, upper_j=jj, sample_size=np.array(m))
                rows.append({'family': family, 'n': n, 'count_per_pair': m, 'repetitions': repetitions,
                             'interval_coverage_fraction': covered / repetitions,
                             'uc_envelope_coverage_fraction': 1 - violations / repetitions,
                             'exact_certificate_fraction': exact_certificate / repetitions,
                             'all_edges_oriented_fraction': full_orientation / repetitions,
                             'mean_unresolved_memberships': float(np.mean(widths)),
                             'target_size': int(target.sum())})
                print(json.dumps(rows[-1]), flush=True)
    for name, values in [('summary.csv', rows), ('raw_metrics.csv', raw)]:
        with (args.out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    report = {'classification': 'software_qa' if args.smoke else 'fixed_synthetic_certificate_diagnostic',
              'seed': seed, 'delta_per_simulated_case': delta, 'repetitions': repetitions,
              'elapsed_seconds': time.perf_counter() - started, 'payload_sha256': hashes,
              'assumptions': 'Fixed counts; independent Bernoulli observations within each unordered pair.',
              'limits': 'Illustrative fixed families; not human-cohort coverage or a comparison of learned systems.'}
    (args.out / 'DIAGNOSTICS_COMPLETE.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'DIAGNOSTICS_COMPLETE', **report}), flush=True)


if __name__ == '__main__':
    main()
