# STE IJCAI'26 Experiment Pipeline

This folder (`ste/`) is a self-contained, runnable experiment pipeline intended for an IJCAI-grade empirical section.

## What this pipeline does

1. **Synthetic tournaments**
   - Generates a ground-truth probabilistic tournament `P` with a tunable transitive signal (BTL) and tunable cyclicity.
   - Samples pairwise comparisons (with optional noise + sparsity).
   - Estimates `P_hat` from the sampled comparisons (train/val/test split is supported).
   - Computes **Top Cycle** and **Uncovered Set** membership probabilities using STE operators.
   - Evaluates recovery, robustness, calibration, stability-vs-sparsity, ablations, and runtime scaling.

2. **Real-world (optional)**
   - Loads **Chatbot Arena** pairwise data from a CSV (you provide the file path).
   - Loads **AgentBench** pairwise data from a JSONL/CSV-like format (you provide the file path).
   - Computes STE membership probabilities and baseline scores.

3. **Paper assets**
   - Writes raw result CSVs into a timestamped `runs/<timestamp>/` directory.
   - Writes LaTeX tables to `paper_assets/tables/`.
   - Writes figure PDFs to `paper_assets/figs/`.

## Why earlier outputs can look “fake”

If the reachability operator is implemented as a **sum of path masses** (e.g., `D + D^2 + …`) it can **saturate** (most reachability entries approach 1) for moderate `n`, which makes Top Cycle membership probabilities nearly constant and close to 1.

Similarly, some uncovered-set relaxations based on a single “witness” can collapse cover scores toward 0, which makes `sigmoid(beta*(0.5-max_cover))` nearly constant and close to `sigmoid(beta*0.5)`.

This pipeline therefore defaults to:
- `ste.reachability_mode: max_product` (bounded, existence-style)
- `ste.uncovered_mode: lukasiewicz` (fuzzy implication form of covering)

Both are deterministic and avoid the most common degeneracies.

## Setup

### Python dependencies

Create a clean environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `torch` installation depends on your machine; if you already have `torch`, keep it.
- `trueskill` is optional. If you do not install it, the TrueSkill baseline is skipped.

## Sanity checks (do this first)

From inside the `ste/` directory:

```bash
python -m tests.sanity
```

Expected behavior:
- **Transitive tournament** → Top Cycle size = 1 and Uncovered size = 1.
- **3-cycle** → Top Cycle size = 3 and Uncovered size = 3.

If this fails, do **not** run large experiments.

## Auditing / anti-placeholder tooling

Two helper scripts are included to make it easy to validate that the code is
executing real computations end-to-end:

1) Static scan for common placeholder patterns:

```bash
python tools/static_placeholder_scan.py
```

2) One-run audit that saves intermediate matrices (D, R, cover) and full
arrays in a compressed NPZ so you can inspect them numerically:

```bash
python tools/audit_one_run.py --config configs/ste_master.yaml --n 20 --rho 0.4 --seed 42
```

The audit artifacts are written under `outputs/audit/...` and include a JSON
report with SHA256 hashes.

## Running experiments

### Quick smoke test

```bash
python run_all.py --quick
```

Quick mode reduces the number of seeds and bootstrap samples. It is a smoke test, not publishable.

### Full run (publishable settings)

Edit the config and then run:

```bash
python run_all.py --config configs/ste_master.yaml
```

To run only one experiment:

```bash
python run_all.py --exp core_recovery
python run_all.py --exp calibration
```

### Output locations

- Raw results: `outputs/runs/<timestamp>/...csv`
- Figures: `outputs/paper_assets/figs/*.pdf`
- Tables: `outputs/paper_assets/tables/*.tex`

The pipeline also copies the exact config used into the run directory:

- `outputs/runs/<timestamp>/config_used.yaml`

## Real-world experiments

### Chatbot Arena

1. Obtain/export the pairwise dataset you intend to use.
2. Point the config key `paths.chatbot_arena_file` to your CSV.
3. Enable `experiments.chatbot_arena_global` (and optionally `chatbot_arena_by_category`).

The loader expects columns (configurable):
- `model_a`, `model_b`, and `winner` (with values indicating whether A/B won; ties can be dropped or handled by policy).

### AgentBench

1. Point `paths.agentbench_file` to your file.
2. Enable `experiments.agentbench_per_environment`.

Because AgentBench formats can vary, check `data/agentbench.py` and adapt column names as needed.

## Reproducibility checklist

Before trusting results:
- Confirm `config_used.yaml` matches what you intended.
- Run `python -m tests.sanity`.
- Run at least two different seeds and verify results change in plausible ways.
- Increase `m_per_pair` and verify recovery metrics improve.
- Sweep `rho` and verify recovery degrades as cyclicity increases.

