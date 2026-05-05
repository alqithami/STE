#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   bash scripts/run_arena_human_preferences.sh /path/to/arena.csv outputs/arena_full
# Required CSV columns by default: model_a, model_b, winner, optional category.
cd "$(dirname "$0")/.."
INPUT="${1:?provide arena CSV path}"
OUT="${2:-outputs/arena_full}"
python -m ste_neurips.neurips_suite real-arena \
  --input "$INPUT" \
  --out "$OUT" \
  --by-category \
  --bootstrap 1000 \
  --cycle-min-count 30 \
  --cycle-confidence 0.95 \
  --methods ste_posterior_edge_uc,ste_posterior_edge_tc,ste_plugin_uc,ste_plugin_tc,hard_uc,hard_tc,winrate,btl,elo,trueskill,rank_centrality,hodge,pagerank,copeland,schulze,minimax,ranked_pairs,kemeny_local
