#!/usr/bin/env python3
"""Independent replay of the fixed CPU diagnostic; imports no stepc code.

Rebuilds generator probabilities/counts from their declared seeds, validates
every original payload SHA256, reconstructs intervals with scalar math, checks
true UC by independent cover and adjacency-list traversal oracles, and rebuilds
all replicate metrics and summaries.  An explicit possible-completion
construction independently checks the outer set on every replicate.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FAMILIES = ("condorcet_weak_rest", "cycle_core_weak_rest", "uniform_margin")
SIZES = (12, 24)
BUDGETS = (16, 64, 256, 1024)
SEED, DELTA, REPETITIONS = 20261007, .05, 256


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability_fixture(n, family, fi):
    rng = np.random.default_rng(SEED + fi * 1000 + n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    signs = rng.choice([-1, 1], len(pairs))
    P = np.full((n, n), .5)
    margin = .15 if family == "uniform_margin" else .01
    for (i, j), sign in zip(pairs, signs):
        P[i, j] = .5 + margin * sign
        P[j, i] = 1 - P[i, j]
    if family == "condorcet_weak_rest":
        for opponent in range(1, n):
            P[0, opponent], P[opponent, 0] = .85, .15
    elif family == "cycle_core_weak_rest":
        for core in range(3):
            for rest in range(3, n):
                P[core, rest], P[rest, core] = .85, .15
        for a, b in ((0, 1), (1, 2), (2, 0)):
            P[a, b], P[b, a] = .85, .15
    return P, pairs


def out_sets(adjacency):
    return [set(np.flatnonzero(row).tolist()) for row in adjacency]


def uc_by_cover(adjacency):
    out = out_sets(adjacency)
    return {v for v in range(len(out))
            if not any(v in out[u] and out[v].issubset(out[u])
                       for u in range(len(out)) if u != v)}


def uc_by_two_step_traversal(adjacency):
    out = out_sets(adjacency)
    vertices = set(range(len(out)))
    result = set()
    for v in vertices:
        reach = {v} | out[v]
        for child in out[v]:
            reach.update(out[child])
        if reach == vertices:
            result.add(v)
    return result


def possible_uc_by_reinforcement(known):
    # Orient all missing root edges outward, then every admissible edge from
    # that maximal out-neighborhood to the root's known in-neighbors outward.
    # These choices affect distinct edges and maximize two-step reachability.
    n = len(known)
    possible = set()
    for v in range(n):
        maximal_out = {w for w in range(n) if w != v and not known[w, v]}
        known_in = {u for u in range(n) if known[u, v]}
        if all(any(not known[u, w] for w in maximal_out) for u in known_in):
            possible.add(v)
    return possible


def independent_intervals(counts, m, n, pairs):
    radius = math.sqrt(math.log(2 * len(pairs) / DELTA) / (2 * m))
    lower, upper = np.zeros((n, n)), np.ones((n, n))
    known = np.zeros((n, n), dtype=bool)
    W = np.zeros((n, n), dtype=np.int64)
    for count, (i, j) in zip(counts, pairs):
        W[i, j], W[j, i] = int(count), m - int(count)
        lo, hi = max(0., int(count) / m - radius), min(1., int(count) / m + radius)
        lower[i, j], upper[i, j] = lo, hi
        lower[j, i], upper[j, i] = 1 - hi, 1 - lo
        known[i, j], known[j, i] = lo > .5, hi < .5
    np.fill_diagonal(lower, .5)
    np.fill_diagonal(upper, .5)
    assert np.array_equal(W + W.T, m * (1 - np.eye(n, dtype=np.int64)))
    assert not np.any(known & known.T)
    return lower, upper, known, W, radius


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_row(expected, observed):
    assert set(expected) == set(observed), (set(expected), set(observed))
    for key, value in expected.items():
        if isinstance(value, bool):
            assert observed[key] == str(value), (key, value, observed[key])
        elif isinstance(value, str):
            assert observed[key] == value, (key, value, observed[key])
        elif isinstance(value, int):
            assert int(observed[key]) == value, (key, value, observed[key])
        else:
            assert math.isclose(float(observed[key]), value, rel_tol=0, abs_tol=1e-14), (key, value, observed[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("validation/certificate_diagnostics_full"))
    ap.add_argument("--audit", type=Path, default=Path("validation/CERTIFICATE_DIAGNOSTIC_AUDIT.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    complete = json.loads((args.out / "DIAGNOSTICS_COMPLETE.json").read_text())
    assert complete["classification"] == "fixed_synthetic_certificate_diagnostic"
    assert complete["seed"] == SEED and complete["repetitions"] == REPETITIONS
    assert complete["delta_per_simulated_case"] == DELTA
    expected_names = {f"{family}_n{n}_m{m}.npz"
                      for family in FAMILIES for n in SIZES for m in BUDGETS}
    expected_names |= {"summary.csv", "raw_metrics.csv"}
    assert set(complete["payload_sha256"]) == expected_names
    verified_hashes = {}
    for name, expected_hash in complete["payload_sha256"].items():
        actual = digest(args.out / name)
        assert actual == expected_hash, (name, expected_hash, actual)
        verified_hashes[name] = actual
    raw = read_csv(args.out / "raw_metrics.csv")
    summary = read_csv(args.out / "summary.csv")
    assert len(raw) == 24 * REPETITIONS and len(summary) == 24
    reconstructed, cell_reports = [], []
    total_interval_failures = total_uc_failures = total_exact_truth_recovery = 0
    total_known_edge_conflicts = total_probability_matrices = total_replayed_count_entries = 0
    total_individual_comparisons = 0
    for fi, family in enumerate(FAMILIES):
        for n in SIZES:
            expected_P, pairs = probability_fixture(n, family, fi)
            truth = uc_by_cover(expected_P > .5)
            assert truth == uc_by_two_step_traversal(expected_P > .5)
            total_probability_matrices += 1
            for m in BUDGETS:
                tag = f"{family}_n{n}_m{m}"
                with np.load(args.out / (tag + ".npz"), allow_pickle=False) as archive:
                    assert set(archive.files) == {"P", "true_uc", "upper_wins", "upper_i", "upper_j", "sample_size"}
                    P, saved_uc, counts = archive["P"], archive["true_uc"], archive["upper_wins"]
                    assert np.array_equal(P, expected_P), tag
                    assert set(np.flatnonzero(saved_uc).tolist()) == truth, tag
                    assert saved_uc.dtype == np.bool_
                    assert np.array_equal(archive["upper_i"], [i for i, j in pairs])
                    assert np.array_equal(archive["upper_j"], [j for i, j in pairs])
                    assert int(archive["sample_size"]) == m
                assert counts.shape == (REPETITIONS, len(pairs))
                assert np.issubdtype(counts.dtype, np.integer) and np.all((0 <= counts) & (counts <= m))
                probabilities = np.array([P[i, j] for i, j in pairs])
                rng = np.random.default_rng(SEED + fi * 100000 + n * 1000 + m)
                replayed = rng.binomial(m, probabilities, size=(REPETITIONS, len(pairs)))
                assert np.array_equal(counts, replayed), tag
                total_replayed_count_entries += counts.size
                total_individual_comparisons += int(counts.size) * m
                metrics, widths, conflict_replicates = [], [], 0
                for rep, count_vector in enumerate(counts):
                    lower, upper, known, W, radius = independent_intervals(count_vector, m, n, pairs)
                    inner = uc_by_two_step_traversal(known)
                    outer = possible_uc_by_reinforcement(known)
                    # Cover-based nonmember witness reconstruction is a second
                    # check against the independent possible-completion method.
                    out = out_sets(known)
                    cover_outer = {v for v in range(n) if not any(
                        v in out[u] and all(v in out[w] or w in out[u]
                                           for w in range(n) if w not in (u, v))
                        for u in range(n) if u != v)}
                    assert outer == cover_outer, (tag, rep)
                    covered = all(lower[i, j] <= P[i, j] <= upper[i, j] for i, j in pairs)
                    sound = inner.issubset(truth) and truth.issubset(outer)
                    exact = inner == outer
                    all_oriented = all(known[i, j] or known[j, i] for i, j in pairs)
                    conflicts = int(np.sum(known & ~(P > .5)))
                    assert not covered or (sound and conflicts == 0), (tag, rep)
                    conflict_replicates += conflicts > 0
                    total_known_edge_conflicts += conflicts
                    row = {"family": family, "n": n, "count_per_pair": m, "replicate": rep,
                           "simultaneous_interval_contains_P": covered,
                           "certificate_contains_true_UC": sound, "certificate_is_exact": exact,
                           "all_edges_oriented": all_oriented,
                           "inner_size": len(inner), "outer_size": len(outer), "target_size": len(truth)}
                    check_row(row, raw[len(reconstructed)])
                    metrics.append(row)
                    widths.append(len(outer) - len(inner))
                    reconstructed.append({**row, "inner_vertices": " ".join(map(str, sorted(inner))),
                                          "outer_vertices": " ".join(map(str, sorted(outer))),
                                          "true_uc_vertices": " ".join(map(str, sorted(truth))),
                                          "known_edge_conflicts": conflicts, "radius": radius})
                counts_metric = {key: sum(row[key] for row in metrics) for key in
                                 ("simultaneous_interval_contains_P", "certificate_contains_true_UC",
                                  "certificate_is_exact", "all_edges_oriented")}
                expected_summary = {"family": family, "n": n, "count_per_pair": m,
                                    "repetitions": REPETITIONS,
                                    "interval_coverage_fraction": counts_metric["simultaneous_interval_contains_P"] / REPETITIONS,
                                    "uc_envelope_coverage_fraction": counts_metric["certificate_contains_true_UC"] / REPETITIONS,
                                    "exact_certificate_fraction": counts_metric["certificate_is_exact"] / REPETITIONS,
                                    "all_edges_oriented_fraction": counts_metric["all_edges_oriented"] / REPETITIONS,
                                    "mean_unresolved_memberships": sum(widths) / REPETITIONS,
                                    "target_size": len(truth)}
                check_row(expected_summary, summary[len(cell_reports)])
                interval_failures = REPETITIONS - counts_metric["simultaneous_interval_contains_P"]
                uc_failures = REPETITIONS - counts_metric["certificate_contains_true_UC"]
                exact_truth = sum(row["certificate_is_exact"] and row["certificate_contains_true_UC"] for row in metrics)
                total_interval_failures += interval_failures
                total_uc_failures += uc_failures
                total_exact_truth_recovery += exact_truth
                cell_reports.append({**expected_summary, "interval_failure_count": interval_failures,
                                     "uc_envelope_failure_count": uc_failures,
                                     "exact_and_correct_count": exact_truth,
                                     "replicates_with_confident_wrong_edge": conflict_replicates})
    reconstruction_path = args.out / "INDEPENDENT_RECONSTRUCTION.csv"
    with reconstruction_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reconstructed[0]))
        writer.writeheader()
        writer.writerows(reconstructed)
    source_hashes = {name: digest(root / name) for name in
                     ("run_diagnostics.py", "audit_diagnostics.py", "stepc/certificates.py", "stepc/posterior.py")}
    extra_hashes = {name: digest(args.out / name) for name in
                    ("DIAGNOSTICS_COMPLETE.json", "INDEPENDENT_RECONSTRUCTION.csv")}
    stdout = args.out / "RUN_STDOUT.log"
    if stdout.is_file():
        extra_hashes[stdout.name] = digest(stdout)
    result = {
        "status": "PASS", "classification": "independent_fixed_synthetic_certificate_diagnostic_audit",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {"command": "python run_diagnostics.py --out " + str(args.out),
                      "cpu_only": True, "diagnostic_elapsed_seconds": complete["elapsed_seconds"],
                      "audit_elapsed_seconds": time.perf_counter() - started,
                      "python_version": sys.version, "numpy_version": np.__version__, "platform": platform.platform()},
        "design": {"seed": SEED, "delta_per_case": DELTA, "families": list(FAMILIES),
                   "sizes": list(SIZES), "counts_per_pair": list(BUDGETS), "cells": len(cell_reports),
                   "replicates_per_cell": REPETITIONS, "total_replicates": len(reconstructed),
                   "distinct_fixed_probability_matrices": total_probability_matrices,
                   "replayed_binomial_count_entries": total_replayed_count_entries,
                   "represented_bernoulli_comparisons": total_individual_comparisons},
        "verification": {"original_payload_hashes_verified": len(verified_hashes),
                         "generator_probability_matrices_match_seed": True,
                         "all_count_arrays_match_independent_seed_replay": True,
                         "all_directed_count_totals_reconstructed": True,
                         "saved_true_uc_matches_independent_cover_and_traversal": True,
                         "uc_outer_matches_independent_reinforced_completion_and_cover_witness": True,
                         "all_raw_metric_rows_match": True, "all_summary_rows_match": True,
                         "covered_implies_sound_and_no_confident_wrong_edges": True},
        "observed_totals": {"interval_envelope_failures": total_interval_failures,
                            "uc_envelope_failures": total_uc_failures,
                            "confident_wrong_edges": total_known_edge_conflicts,
                            "exact_and_correct_uc_replicates": total_exact_truth_recovery},
        "source_sha256": source_hashes, "verified_original_payload_sha256": verified_hashes,
        "additional_output_sha256": extra_hashes, "cell_results": cell_reports,
        "independence": "Audit imports no stepc code or run_diagnostics; scalar radii, adjacency-list traversal, set-cover UC, and possible-completion reinforcement are rebuilt independently.",
        "limits": ["One fixed P for each of six family/n settings; budgets use independent, nonnested seeded batches.",
                   "Delta=.05 is simultaneous across unordered pairs within one case, not across all 6144 cases or all budgets.",
                   "Observed rates are conditional fixed-design diagnostics, not population coverage, human-cohort inference, learned-method comparison, or an anytime stopping claim.",
                   "Exact certificate alone does not imply truth recovery outside the interval event; audit checks exact AND sound separately.",
                   "No exhaustive enumeration of n12/n24 completions was attempted; independent cover/traversal and constructive completion methods check every saved case."]}
    results_note = root / "RESULTS_CERTIFICATES.md"
    if results_note.is_file():
        result["results_note_sha256"] = digest(results_note)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "cells": len(cell_reports), "replicates": len(reconstructed),
                      "hashes_verified": len(verified_hashes), "observed_totals": result["observed_totals"],
                      "audit": str(args.audit)}))


if __name__ == "__main__":
    main()
