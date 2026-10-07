"""Auditable subsamples of the bundled, previously observed human profiles.

No latent preferences or ballots are generated. Every sample is a multivariate
hypergeometric draw from the observed ballot-type multiplicities. The ranks of
omitted alternatives stay -1, and comparisons involving an omitted alternative
remain unobserved. Full-profile arrays are labels, never prediction features.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_ORDER = ("00007", "00019", "00021", "00022", "00067", "00073")
DEFAULT_SEED = 20261007
DEFAULT_FRACTIONS = (0.05, 0.10, 0.25)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *parts], separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf8")).digest()[:8], "little")


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf8")
    os.replace(temp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def parse_profile(path: str | Path) -> dict[str, Any]:
    """Read PrefLib ordinal records, retaining explicit ties and omitted ranks.

    This reproduces the frozen parser's conventions, while validating complete
    token consumption to avoid silently accepting malformed ballot records.
    Alternative order is the ascending order of the declared alternative IDs.
    """
    meta: dict[str, str] = {}
    records: list[tuple[int, list[list[int]]]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if ":" in line:
                key, value = line[1:].split(":", 1)
                meta[key.strip()] = value.strip()
            continue
        if ":" not in line:
            raise ValueError(f"Missing multiplicity separator on line {line_number}")
        mult_text, order = line.split(":", 1)
        try:
            mult = int(mult_text.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid multiplicity on line {line_number}") from exc
        if mult < 1:
            raise ValueError("Nonpositive ballot multiplicity")
        tokens = re.findall(r"\{[^{}]*\}|[+-]?\d+", order)
        residual = re.sub(r"\{[^{}]*\}|[+-]?\d+", "", order)
        if residual.strip(" ,\t") or not tokens:
            raise ValueError(f"Malformed ranking on line {line_number}")
        groups = []
        for token in tokens:
            if token.startswith("{"):
                inner = token[1:-1]
                if not re.fullmatch(r"\s*[+-]?\d+\s*(?:,\s*[+-]?\d+\s*)*", inner):
                    raise ValueError(f"Malformed tied rank on line {line_number}")
            groups.append([int(x) for x in re.findall(r"[+-]?\d+", token)])
        records.append((mult, groups))
    if meta.get("DATA TYPE") not in ("soc", "soi", "toc", "toi"):
        raise ValueError("Unsupported profile type")
    try:
        n, voters = int(meta["NUMBER ALTERNATIVES"]), int(meta["NUMBER VOTERS"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Missing or invalid profile size metadata") from exc
    if n < 1 or voters < 1:
        raise ValueError("Profile sizes must be positive")
    ids = sorted(int(key.rsplit(" ", 1)[-1]) for key in meta if key.startswith("ALTERNATIVE NAME "))
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("Missing or duplicate alternative IDs")
    index = {alternative: i for i, alternative in enumerate(ids)}
    ranks, multiplicity = [], []
    for mult, groups in records:
        row = np.full(n, -1, dtype=np.int16)
        seen: set[int] = set()
        for rank, group in enumerate(groups):
            for alternative in group:
                if alternative in seen or alternative not in index:
                    raise ValueError("Invalid or repeated alternative in ballot")
                seen.add(alternative)
                row[index[alternative]] = rank
        ranks.append(row)
        multiplicity.append(mult)
    counts = np.asarray(multiplicity, dtype=np.int64)
    ranks_array = np.asarray(ranks, dtype=np.int16).reshape((-1, n))
    if int(counts.sum()) != voters:
        raise ValueError("Voter count differs from ballot multiplicities")
    if "NUMBER UNIQUE ORDERS" in meta and len(records) != int(meta["NUMBER UNIQUE ORDERS"]):
        raise ValueError("Order count mismatch")
    return {"meta": meta, "ranks": ranks_array, "multiplicity": counts,
            "n": n, "voters": voters, "alternative_ids": ids}


def ballot_counts(profile: dict[str, Any], multiplicity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count only comparisons for which both alternatives were explicitly ranked."""
    ranks, n = profile["ranks"], int(profile["n"])
    mult = np.asarray(multiplicity)
    if mult.shape != (len(ranks),) or not np.issubdtype(mult.dtype, np.integer) or np.any(mult < 0):
        raise ValueError("Invalid sampled ballot multiplicities")
    if np.any(mult > profile["multiplicity"]):
        raise ValueError("Sample exceeds observed ballot multiplicities")
    mult = mult.astype(np.int64, copy=False)
    W = np.zeros((n, n), dtype=np.int64)
    T = np.zeros_like(W)
    for a in range(n):
        for b in range(a + 1, n):
            seen = (ranks[:, a] >= 0) & (ranks[:, b] >= 0)
            W[a, b] = mult[seen & (ranks[:, a] < ranks[:, b])].sum()
            W[b, a] = mult[seen & (ranks[:, b] < ranks[:, a])].sum()
            T[a, b] = T[b, a] = mult[seen & (ranks[:, a] == ranks[:, b])].sum()
    return W, T


def nested_subsamples(profile: dict[str, Any], fractions: list[float] | tuple[float, ...],
                      rng: np.random.Generator) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fractions = sorted(float(value) for value in fractions)
    if len(set(fractions)) != len(fractions) or not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("Sampling fractions must be unique and in (0, 1]")
    remaining = profile["multiplicity"].copy()
    used = np.zeros_like(remaining)
    previous = 0
    samples = {}
    for fraction in fractions:
        take = max(1, int(np.floor(fraction * profile["voters"])))
        if take > previous:
            draw = rng.multivariate_hypergeometric(remaining, take - previous)
            used += draw
            remaining -= draw
        previous = take
        W, T = ballot_counts(profile, used)
        samples[fraction] = W, T, used.copy()
    return samples


def uncovered_set(A: np.ndarray) -> np.ndarray:
    """Cover-relation oracle, valid for strict tournaments and partial relations."""
    A = np.asarray(A, dtype=bool).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency must be square")
    np.fill_diagonal(A, False)
    y = np.ones(len(A), dtype=bool)
    for a in range(len(A)):
        for c in range(len(A)):
            if c != a and A[c, a] and not np.any(A[a] & ~A[c]):
                y[a] = False
                break
    return y


def uncovered_set_independent(A: np.ndarray) -> np.ndarray:
    """Independent Python set-inclusion formulation of the reference UC."""
    A = np.asarray(A, dtype=bool)
    wins = [{j for j in range(len(A)) if j != i and A[i, j]} for i in range(len(A))]
    return np.asarray([not any(i in wins[j] and wins[i].issubset(wins[j])
                              for j in range(len(A)) if j != i)
                       for i in range(len(A))], dtype=bool)


def source_folds(sources: list[str] | tuple[str, ...] = SOURCE_ORDER) -> list[dict[str, Any]]:
    """Cyclic next-source development, with the other four sources training."""
    sources = tuple(sources)
    if len(sources) != 6 or len(set(sources)) != 6:
        raise ValueError("Exactly six distinct sources are required")
    return [{"fold_id": i, "fold": i, "test_source": source,
             "dev_source": sources[(i + 1) % len(sources)],
             "development_source": sources[(i + 1) % len(sources)],
             "train_sources": [s for s in sources if s not in (source, sources[(i + 1) % len(sources)])]}
            for i, source in enumerate(sources)]


def _data_config(config: dict[str, Any]) -> tuple[int, list[float], int]:
    data = config.get("data", config.get("real", {}))
    seed = int(config.get("seed", DEFAULT_SEED))
    if seed != DEFAULT_SEED:
        raise ValueError("This exploratory extension uses the fixed seed 20261007")
    fractions = list(data.get("fractions", DEFAULT_FRACTIONS))
    replicates = int(data.get("replicates", data.get("resamples", 8)))
    if replicates < 1:
        raise ValueError("Replicates must be positive")
    return seed, fractions, replicates


def _load_profiles(base: str | Path) -> list[dict[str, Any]]:
    base = Path(base)
    manifest_path = base / "reference" / "data" / "ELIGIBLE_PROFILES.csv"
    with manifest_path.open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36 or tuple(dict.fromkeys(row["source"] for row in rows)) != SOURCE_ORDER:
        raise ValueError("The fixed original 36-profile, six-source cohort was altered")
    profiles = []
    for row in rows:
        path = base / "reference" / "data" / "profiles" / row["file"]
        digest = sha256(path)
        if digest != row["sha256"]:
            raise ValueError("Bundled raw profile checksum mismatch: " + row["file"])
        profile = parse_profile(path)
        W, T = ballot_counts(profile, profile["multiplicity"])
        n = profile["n"]
        upper = np.triu_indices(n, 1)
        if not np.all((W + W.T)[upper] > 0) or np.any(W[upper] == W.T[upper]):
            raise ValueError("Cohort full-profile majority is not a complete strict tournament")
        A = W > W.T
        y = uncovered_set(A)
        if not np.array_equal(y, uncovered_set_independent(A)):
            raise RuntimeError("Full-profile UC oracles disagree: " + row["file"])
        # A third strict-tournament characterization provides a useful extra check.
        two_step = (A | ((A.astype(np.int64) @ A.astype(np.int64)) > 0) | np.eye(n, dtype=bool)).all(axis=1)
        if not np.array_equal(y, two_step):
            raise RuntimeError("Full-profile UC two-step oracle disagreement")
        if (n, profile["voters"], int(y.sum())) != (int(row["n"]), int(row["voters"]), int(row["uc_size"])):
            raise ValueError("Raw profile differs from frozen cohort metadata: " + row["file"])
        profiles.append({**profile, "source": row["source"], "source_name": row["source_name"],
                         "profile": row["file"], "sha256": digest, "fullW": W,
                         "fullT": T, "fullA": A, "y": y})
    sizes = [int(profile["y"].sum()) for profile in profiles]
    if sum(size == 1 for size in sizes) != 33 or sum(1 < size < profile["n"] for size, profile in zip(sizes, profiles)) != 3:
        raise ValueError("Frozen UC composition changed")
    return profiles


def _cases_from_profiles(profiles: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    seed, fractions, replicates = _data_config(config)
    cases = []
    for profile in profiles:
        for replicate in range(replicates):
            rng = np.random.default_rng(seed_for(seed, "observed-voter-subsample", profile["profile"], replicate))
            for fraction, (W, T, mult) in nested_subsamples(profile, fractions, rng).items():
                case_id = f"{profile['profile']}/r{replicate:02d}/f{fraction:g}"
                y = profile["y"]
                cases.append({"id": case_id, "case_id": case_id, "profile_id": profile["profile"],
                              "profile": profile["profile"], "file": profile["profile"],
                              "source": profile["source"], "n": profile["n"],
                              "fraction": fraction, "replicate": replicate, "resample": replicate,
                              "voters_sampled": int(mult.sum()), "voters_full": profile["voters"],
                              "W": W, "T": T, "observedW": W, "observedT": T,
                              "sampled_multiplicity": mult, "fullW": profile["fullW"],
                              "fullT": profile["fullT"], "fullA": profile["fullA"],
                              "y": y, "fulltruthUC": y, "uc_size": int(y.sum()),
                              "core_class": "singleton" if y.sum() == 1 else "selective_non_singleton",
                              "raw_sha256": profile["sha256"]})
    return cases


def get_profile_cases(base: str | Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return observed inputs and explicitly separated full-profile reference labels."""
    return _cases_from_profiles(_load_profiles(base), config)


def observed_view(case: dict[str, Any]) -> dict[str, Any]:
    """Prediction input without full counts, target membership, or target cardinality."""
    allowed = ("id", "case_id", "profile_id", "profile", "file", "source", "n", "fraction",
               "replicate", "resample", "voters_sampled", "W", "T", "observedW", "observedT")
    return {key: case[key] for key in allowed if key in case}


def fold_split(cases: list[dict[str, Any]], fold: dict[str, Any]) -> dict[str, Any]:
    """Training/development labels are available; held-out labels are evaluation-only."""
    train_sources = set(fold["train_sources"])
    dev_source, test_source = fold["dev_source"], fold["test_source"]
    if dev_source == test_source or dev_source in train_sources or test_source in train_sources:
        raise ValueError("Overlapping source folds")
    test = [case for case in cases if case["source"] == test_source]
    return {"train": [case for case in cases if case["source"] in train_sources],
            "dev": [case for case in cases if case["source"] == dev_source],
            "test": [observed_view(case) for case in test],
            "test_truth": {case["case_id"]: {"y": case["y"], "uc_size": case["uc_size"],
                                            "core_class": case["core_class"]} for case in test}}


def prepare_data(base: str | Path, config: dict[str, Any], out: str | Path) -> dict[str, Any]:
    """Save exact sample multiplicities, counts, raw checksums, and fixed source folds.

    A resume with changed inputs or sampling settings fails. Per-profile NPZs are
    numeric-only and readable with allow_pickle=False. Test full labels in these
    archives are evaluation artifacts; use observed_view for prediction inputs.
    """
    base, out = Path(base), Path(out)
    profiles = _load_profiles(base)
    cases = _cases_from_profiles(profiles, config)
    folds = source_folds()
    for fold in folds:
        for role, sources in (("train", set(fold["train_sources"])), ("dev", {fold["dev_source"]}), ("test", {fold["test_source"]})):
            fold[role + "_ids"] = [case["id"] for case in cases if case["source"] in sources]
    seed, fractions, replicates = _data_config(config)
    identity = {"seed": seed, "fractions": sorted(float(f) for f in fractions), "replicates": replicates,
                "raw_sha256": {profile["profile"]: profile["sha256"] for profile in profiles},
                "numpy_version": np.__version__, "sampler": "numpy.multivariate_hypergeometric"}
    identity_path = out / "DATA_PREPARATION.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing data preparation uses different sampling inputs or runtime")
    out.mkdir(parents=True, exist_ok=True)
    profile_rows = []
    sample_rows = []
    for profile in profiles:
        sample_cases = [case for case in cases if case["profile"] == profile["profile"]]
        relative_path = "data/profiles/" + Path(profile["profile"]).stem + ".npz"
        path = out / relative_path
        arrays = {"W": np.asarray([case["W"] for case in sample_cases]),
                  "T": np.asarray([case["T"] for case in sample_cases]),
                  "sampled_multiplicity": np.asarray([case["sampled_multiplicity"] for case in sample_cases]),
                  "ballot_multiplicity": np.asarray([case["sampled_multiplicity"] for case in sample_cases]),
                  "original_multiplicity": profile["multiplicity"], "ranks": profile["ranks"],
                  "alternative_ids": np.asarray(profile["alternative_ids"], dtype=np.int64),
                  "fullW": profile["fullW"], "fullT": profile["fullT"], "fullA": profile["fullA"],
                  "y": profile["y"], "fraction": np.asarray([case["fraction"] for case in sample_cases]),
                  "resample": np.asarray([case["resample"] for case in sample_cases], dtype=np.int64)}
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if set(old.files) != set(arrays) or any(not np.array_equal(old[key], value) for key, value in arrays.items()):
                    raise ValueError("Existing sampled counts differ: " + str(path))
        else:
            _atomic_npz(path, **arrays)
        profile_rows.append({"source": profile["source"], "source_name": profile["source_name"],
                             "profile": profile["profile"], "n": profile["n"], "voters": profile["voters"],
                             "unique_orders": len(profile["multiplicity"]), "uc_size": int(profile["y"].sum()),
                             "raw_sha256": profile["sha256"], "counts_npz": relative_path,
                             "counts_npz_sha256": sha256(path), "cases": len(sample_cases)})
        for index, case in enumerate(sample_cases):
            sample_rows.append({"case_id": case["case_id"], "source": case["source"],
                                "profile": case["profile"], "replicate": case["replicate"],
                                "fraction": case["fraction"], "n": case["n"],
                                "voters_sampled": case["voters_sampled"], "counts_npz": relative_path,
                                "array_index": index})
    manifest = {"cohort": "original previously observed 36 genuine PrefLib profiles", "profiles": profile_rows,
                "profile_count": len(profiles), "case_count": len(cases), "source_order": list(SOURCE_ORDER),
                "singleton_profiles": 33, "selective_non_singleton_profiles": 3,
                "ballots_generated": 0, "sampling": identity,
                "unranked_rule": "comparisons involving an unranked alternative remain unobserved",
                "tie_rule": "explicit tied ranks increment symmetric T; T never gives either alternative a win",
                "truth": "UC of finite full-profile strict empirical majority tournament W > W.T",
                "reference_use": "full counts, majority orientations and UC are train/dev labels or held-out evaluation targets only"}
    _atomic_json(out / "DATA_PROFILE_MANIFEST.json", manifest)
    _atomic_json(out / "SOURCE_FOLDS.json", {"source_order": list(SOURCE_ORDER), "folds": folds,
                                            "prediction_inputs": "observed W and T only; no full-profile reference arrays",
                                            "main_fraction": 0.1, "aggregation": "resamples then profiles then sources equally"})
    temp_csv = out / "DATA_COUNTS_MANIFEST.csv.tmp"
    with temp_csv.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    os.replace(temp_csv, out / "DATA_COUNTS_MANIFEST.csv")
    _atomic_json(identity_path, identity)
    return {"cases": cases, "folds": folds, "manifest": manifest}
