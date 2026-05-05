#!/usr/bin/env python3
"""Download and standardize public real-data inputs for the STE NeurIPS suite.

This script focuses on datasets that already contain pairwise human/model
preferences. Execution-agent datasets such as AgentBench/WebArena/OSWorld/SWE-bench
provide tasks/environments, but usually not a universal per-agent per-task results
CSV. For those, the STE suite expects a user-supplied scorelog exported from an
agent run or a leaderboard with per-task outcomes.

Outputs are standardized into the schemas consumed by ``neurips_suite.py``:

Pairwise preference schema:
    model_a, model_b, winner, category

Score-log schema:
    environment, agent, task_id, score, success, status
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml


PAIRWISE_DATASETS: Dict[str, Dict[str, object]] = {
    "chatbot_arena_33k": {
        "repo_id": "lmsys/chatbot_arena_conversations",
        "config": None,
        "split": "train",
        "output": "chatbot_arena_33k.csv",
        "gated": True,
        "default_download": True,
        "note": "Requires accepting the Hugging Face dataset conditions and providing HF_TOKEN.",
    },
    "arena_human_preference_55k": {
        "repo_id": "lmarena-ai/arena-human-preference-55k",
        "config": None,
        "split": "train",
        "output": "arena_human_preference_55k.csv",
        "gated": False,
        "default_download": True,
        "note": "Human preference battles; usually has model_a/model_b and winner indicator columns.",
    },
    # This repository is large and its schema has changed over time. We keep it off by default,
    # but the converter can try to standardize it if the loaded split exposes pairwise winner columns.
    "arena_hard_auto": {
        "repo_id": "lmarena-ai/arena-hard-auto",
        "config": None,
        "split": "train",
        "output": "arena_hard_auto_pairwise.csv",
        "gated": False,
        "default_download": False,
        "note": "Automatic judge data; large download; treat separately from human preferences. Schema may need manual conversion.",
    },
}

MANIFEST_PATH = Path("configs/real_datasets_manifest_template.yaml")

MODEL_A_CANDIDATES = [
    "model_a", "model_a_name", "model_a_id", "model_a_public_name", "model_a_private_name",
    "model_a_anony", "model_a_anonymous", "left_model", "answer_a_model",
]
MODEL_B_CANDIDATES = [
    "model_b", "model_b_name", "model_b_id", "model_b_public_name", "model_b_private_name",
    "model_b_anony", "model_b_anonymous", "right_model", "answer_b_model",
]
WINNER_CANDIDATES = [
    "winner", "label", "preference", "preference_label", "winner_model", "chosen", "vote",
]
CATEGORY_CANDIDATES = [
    "category", "cluster", "language", "arena", "turn", "domain", "benchmark", "dataset",
]


def _pick_column(columns: Iterable[str], candidates: List[str], required: bool = True) -> Optional[str]:
    cols = list(columns)
    lowered = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates}. Available columns: {cols}")
    return None


def _truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) > 0.5
    return str(value).strip().lower() in {"1", "true", "yes", "y", "prefer", "preferred", "win", "winner"}


def _standardize_winner(row: pd.Series, model_a: str, model_b: str, winner_col: Optional[str]) -> str:
    a = str(row[model_a])
    b = str(row[model_b])

    # Common Chatbot Arena / Kaggle style one-hot flags.
    for col in ["winner_model_a", "winner_a", "prefer_a", "label_model_a"]:
        if col in row.index and _truthy(row[col]):
            return a
    for col in ["winner_model_b", "winner_b", "prefer_b", "label_model_b"]:
        if col in row.index and _truthy(row[col]):
            return b
    for col in ["winner_tie", "winner_tie_both_bad", "tie", "both_bad"]:
        if col in row.index and _truthy(row[col]):
            return "tie"

    if winner_col is None:
        return "tie"
    raw = str(row[winner_col]).strip()
    low = raw.lower()
    if raw == a or low in {"model_a", "a", "left", "answer_a", "response_a", "winner_a", "prefer_a", "1"}:
        return a
    if raw == b or low in {"model_b", "b", "right", "answer_b", "response_b", "winner_b", "prefer_b", "0"}:
        return b
    if low in {"tie", "draw", "both", "both_bad", "tie (bothbad)", "tie (both bad)", "no_preference", "nan", "none"}:
        return "tie"
    # Last resort: preserve the raw winner. The STE runner will ignore unrecognized outcomes.
    return raw


def standardize_pairwise(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    model_a = _pick_column(df.columns, MODEL_A_CANDIDATES, required=True)
    model_b = _pick_column(df.columns, MODEL_B_CANDIDATES, required=True)
    winner_col = _pick_column(df.columns, WINNER_CANDIDATES, required=False)
    category_col = _pick_column(df.columns, CATEGORY_CANDIDATES, required=False)

    out = pd.DataFrame()
    out["model_a"] = df[model_a].astype(str)
    out["model_b"] = df[model_b].astype(str)
    out["winner"] = df.apply(lambda row: _standardize_winner(row, model_a, model_b, winner_col), axis=1)
    if category_col is not None:
        out["category"] = df[category_col].fillna("global").astype(str)
    else:
        out["category"] = "global"
    out.insert(0, "source_dataset", dataset_name)
    # Drop rows with missing or self-comparison models.
    out = out[(out["model_a"].notna()) & (out["model_b"].notna()) & (out["model_a"] != out["model_b"])]
    return out


def load_hf_dataset(repo_id: str, split: str, config: Optional[str], token: Optional[str]):
    try:
        from datasets import load_dataset  # imported lazily so --list works without datasets installed
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face downloader requires the optional dependency 'datasets'. "
            "Install with: pip install datasets huggingface_hub pyarrow"
        ) from exc

    kwargs = {"split": split}
    if token:
        kwargs["token"] = token
    if config:
        return load_dataset(repo_id, config, **kwargs)
    return load_dataset(repo_id, **kwargs)


def download_pairwise_dataset(name: str, out_dir: Path, token: Optional[str], overwrite: bool = False) -> Path:
    spec = PAIRWISE_DATASETS[name]
    out_path = out_dir / str(spec["output"])
    if out_path.exists() and not overwrite:
        print(f"[download] {name}: {out_path} exists; skipping. Use --overwrite to redownload.")
        return out_path

    print(f"[download] loading {name} from {spec['repo_id']} split={spec['split']}")
    ds = load_hf_dataset(
        repo_id=str(spec["repo_id"]),
        split=str(spec.get("split", "train")),
        config=spec.get("config"),
        token=token,
    )
    df = ds.to_pandas()
    print(f"[download] raw columns for {name}: {list(df.columns)}")
    standardized = standardize_pairwise(df, dataset_name=name)
    out_dir.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(out_path, index=False)
    print(f"[download] wrote {out_path} with {len(standardized):,} rows")
    return out_path


def write_execution_templates(out_dir: Path, overwrite: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = {
        # Use the canonical STE score-log schema in all templates so both
        # `scorelog` and manifest-driven `real-suite` commands work directly.
        "agentbench_scores.csv": "environment,agent,task_id,score,success,status\ndbbench-std,agent_a,task_001,1,1,success\ndbbench-std,agent_b,task_001,0,0,failed\n",
        "webarena_scores.csv": "environment,agent,task_id,score,success,status\nwebarena-shopping,agent_a,task_001,1,1,success\nwebarena-shopping,agent_b,task_001,0,0,failed\n",
        "osworld_scores.csv": "environment,agent,task_id,score,success,status\nosworld-spreadsheet,agent_a,task_001,1,1,success\nosworld-spreadsheet,agent_b,task_001,0,0,failed\n",
        "swebench_verified_scores.csv": "environment,agent,task_id,score,success,status\nswebench-django,agent_a,django__django-12345,1,1,resolved\nswebench-django,agent_b,django__django-12345,0,0,failed\n",
    }
    for fname, text in templates.items():
        path = out_dir / fname
        if path.exists() and not overwrite:
            continue
        path.write_text(text, encoding="utf-8")
        print(f"[template] wrote {path}")


def update_manifest_for_downloads(manifest_in: Path, manifest_out: Path, data_dir: Path, downloaded: Dict[str, Path]) -> None:
    if not manifest_in.exists():
        print(f"[manifest] input manifest not found: {manifest_in}; skipping", file=sys.stderr)
        return
    with open(manifest_in, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    manifest["output_root"] = "outputs/real_suite_final"
    for ds in manifest.get("datasets", []):
        name = ds.get("name")
        if name == "chatbot_arena_33k_human" and "chatbot_arena_33k" in downloaded:
            ds["enabled"] = True
            ds["path"] = str(downloaded["chatbot_arena_33k"])
        elif name == "arena_human_preference_55k" and "arena_human_preference_55k" in downloaded:
            ds["enabled"] = True
            ds["path"] = str(downloaded["arena_human_preference_55k"])
        elif name == "arena_hard_auto_pairwise" and "arena_hard_auto" in downloaded:
            ds["enabled"] = True
            ds["path"] = str(downloaded["arena_hard_auto"])
        # Execution logs remain disabled until real per-agent per-task score logs are supplied.
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"[manifest] wrote {manifest_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and standardize public real datasets for STE.")
    ap.add_argument("--dataset", default="all", choices=["all", *PAIRWISE_DATASETS.keys()], help="Dataset to download.")
    ap.add_argument("--out-dir", default="data", help="Directory for standardized CSV files.")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token. Defaults to HF_TOKEN env var.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include-optional", action="store_true", help="When --dataset all, also download optional large/fragile targets such as Arena-Hard-Auto.")
    ap.add_argument("--list", action="store_true", help="List supported download targets and exit.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    ap.add_argument("--write-execution-templates", action="store_true", help="Write templates for AgentBench/WebArena/OSWorld/SWE-bench score logs.")
    ap.add_argument("--update-manifest", default="configs/real_datasets_manifest_downloaded.yaml", help="Write a manifest with successfully downloaded datasets enabled.")
    args = ap.parse_args()

    if args.list:
        for name, spec in PAIRWISE_DATASETS.items():
            gated = "gated" if spec.get("gated") else "public"
            default = "default" if spec.get("default_download", True) else "optional"
            print(f"{name}: {spec['repo_id']} ({gated}, {default}) -> {spec['output']} | {spec.get('note','')}")
        print("Execution-log datasets are not auto-downloaded as pairwise results; templates can be written with --write-execution-templates.")
        return

    out_dir = Path(args.out_dir)
    if args.dataset == "all":
        targets = [name for name, spec in PAIRWISE_DATASETS.items() if spec.get("default_download", True) or args.include_optional]
    else:
        targets = [args.dataset]
    if args.dry_run:
        print(f"[dry-run] would write standardized CSVs to {out_dir.resolve()}")
        for name in targets:
            spec = PAIRWISE_DATASETS[name]
            print(f"[dry-run] {name}: load_dataset({spec['repo_id']!r}, split={spec['split']!r}) -> {out_dir / spec['output']}")
        if args.write_execution_templates:
            print("[dry-run] would write execution score-log templates")
        return

    downloaded: Dict[str, Path] = {}
    for name in targets:
        try:
            downloaded[name] = download_pairwise_dataset(name, out_dir, token=args.hf_token, overwrite=args.overwrite)
        except Exception as exc:
            print(f"[download] FAILED for {name}: {exc}", file=sys.stderr)
            if PAIRWISE_DATASETS[name].get("gated") and not args.hf_token:
                print("[download] Hint: accept the dataset conditions on Hugging Face and set HF_TOKEN.", file=sys.stderr)
    if args.write_execution_templates:
        write_execution_templates(out_dir, overwrite=args.overwrite)
    if args.update_manifest:
        update_manifest_for_downloads(MANIFEST_PATH, Path(args.update_manifest), out_dir, downloaded)


if __name__ == "__main__":
    main()
