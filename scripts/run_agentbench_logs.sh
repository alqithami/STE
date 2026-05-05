#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   bash scripts/run_agentbench_logs.sh /path/to/agentbench_scores.csv outputs/agentbench_full
# Required CSV columns: environment, agent, task_id, and one of score/success/status.
cd "$(dirname "$0")/.."
INPUT="${1:?provide agentbench CSV path}"
OUT="${2:-outputs/agentbench_full}"
python -m ste_neurips.neurips_suite agentbench \
  --input "$INPUT" \
  --out "$OUT" \
  --bootstrap 1000 \
  --cycle-min-count 5 \
  --cycle-confidence 0.90 \
  --methods ste_posterior_edge_uc,ste_posterior_edge_tc,ste_plugin_uc,ste_plugin_tc,hard_uc,hard_tc,winrate,btl,elo,trueskill,rank_centrality,hodge,pagerank,copeland,schulze,minimax,ranked_pairs,kemeny_local
