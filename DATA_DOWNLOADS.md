# Real dataset download notes for STE

The STE real suite accepts standardized CSV files. This package now includes
`scripts/download_real_datasets.py` to download and standardize public Hugging Face
preference datasets when access is available.

## Human pairwise datasets

Run:

```bash
export HF_TOKEN=hf_your_token_if_needed
python scripts/download_real_datasets.py \
  --dataset all \
  --out-dir data \
  --write-execution-templates \
  --update-manifest configs/real_datasets_manifest_downloaded.yaml
```

This creates standardized pairwise CSVs with schema:

```csv
source_dataset,model_a,model_b,winner,category
```

The main targets are:

- `lmsys/chatbot_arena_conversations` -> `data/chatbot_arena_33k.csv`.
  This dataset is gated: log in to Hugging Face, accept the dataset conditions,
  and set `HF_TOKEN` before running.
- `lmarena-ai/arena-human-preference-55k` -> `data/arena_human_preference_55k.csv`.
- `lmarena-ai/arena-hard-auto` -> `data/arena_hard_auto_pairwise.csv` when its current schema exposes pairwise winner columns. Treat this as LLM-as-judge, not human preference.

## Execution-agent datasets

AgentBench, WebArena, OSWorld, and SWE-bench provide tasks/environments, but the
STE tournament analysis needs per-agent per-task outcomes. The downloader writes
templates for these files:

```text
data/agentbench_scores.csv
data/webarena_scores.csv
data/osworld_scores.csv
data/swebench_verified_scores.csv
```

Replace the template rows with real evaluation logs from your agent runs or a
leaderboard export. Required columns are:

```csv
environment,agent,task_id,score,success,status
```

or benchmark-specific aliases defined in `configs/real_datasets_manifest_template.yaml`.
Once populated, set the corresponding dataset `enabled: true` in the manifest.

## Run the real suite

```bash
python -m ste_neurips.neurips_suite real-suite \
  --manifest configs/real_datasets_manifest_downloaded.yaml \
  --out outputs/real_suite_final \
  --bootstrap 1000 \
  --seed 0
```

Merged paper-level outputs appear in:

```text
outputs/real_suite_final/all_real_scores.csv
outputs/real_suite_final/all_real_selection_diagnostics.csv
outputs/real_suite_final/all_real_high_confidence_cycles.csv
outputs/real_suite_final/real_suite_manifest_results.csv
```
