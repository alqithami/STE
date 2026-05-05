# Real-dataset quickstart for STE

## 1. Prepare CSV files

The code accepts two schemas.

### Pairwise preference schema

```csv
model_a,model_b,winner,category
model-1,model-2,model-1,coding
model-1,model-3,tie,writing
```

The winner may be the exact name of `model_a`/`model_b`, or one of: `model_a`, `model_b`, `a`, `b`, `tie`, `tie (bothbad)`.

### Score-log schema

```csv
environment,agent,task_id,score,success,status
web,agent-1,task-001,1.0,1,success
web,agent-2,task-001,0.0,0,failed
```

Same-task records are converted into pairwise wins. If `score` is missing, `success` or `status` is used.

## 2. Run one dataset

Human preference / Arena-style:

```bash
bash scripts/run_arena_human_preferences.sh data/chatbot_arena_33k.csv outputs/chatbot_arena_33k
```

Agent/task score logs:

```bash
python -m ste_neurips.neurips_suite scorelog \
  --input data/osworld_scores.csv \
  --out outputs/osworld
```

## 3. Run multiple datasets

Edit `configs/real_datasets_manifest_template.yaml`, set `enabled: true`, and provide paths. Then run:

```bash
python -m ste_neurips.neurips_suite real-suite \
  --manifest configs/real_datasets_manifest_template.yaml \
  --out outputs/real_suite_final
```

Merged outputs appear as:

```text
outputs/real_suite_final/all_real_scores.csv
outputs/real_suite_final/all_real_selection_diagnostics.csv
outputs/real_suite_final/all_real_high_confidence_cycles.csv
outputs/real_suite_final/real_suite_manifest_results.csv
```

## 4. Paper outputs to report

- `real_arena_scores.csv`: per-agent scores by method.
- `real_arena_bootstrap.csv`: bootstrap top-1 and selected-set membership.
- `real_selection_diagnostics.csv`: selected-set external attack/error rates.
- `real_arena_high_confidence_cycles.csv`: concrete cycle witnesses.
- `real_arena_report.md`: human-readable summary.

## 5. Reviewer-safe wording

Use “diagnostic” for real data unless ground-truth core labels are known. Use “human preference” only for datasets whose labels are human votes. Use “LLM-as-judge” or “automatic judge” for Arena-Hard/MT-Bench-style outputs.
