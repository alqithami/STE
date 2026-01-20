"""Chatbot Arena-style pairwise data ingestion.

This loader is intentionally permissive: it supports multiple common schemas
for Arena-derived data.

Expected schema (defaults):
- model_a, model_b: strings
- winner: one of {"model_a","model_b","tie"} OR {"A","B","tie"} OR {1,0} etc.
- category (optional): task category

Output:
- comparisons: numpy array of shape (m,3) with (a_id,b_id,y)
  where y=1 means a beats b.
- id2model: list mapping id->name
- df_norm: normalized DataFrame with columns: a_id,b_id,y,category(optional)

Tie handling:
- tie_policy='drop': remove tie rows.
- tie_policy='split': represent a tie as *two* rows (a beats b) and (b beats a).
  This implements an unbiased 0.5/0.5 split without requiring fractional counts.

No placeholder data: this module only normalizes/encodes what is in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_df(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path)
    if path.endswith('.jsonl'):
        return pd.read_json(path, lines=True)
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def _infer_winner(row, a_col: str, b_col: str, winner_col: str) -> Optional[int]:
    """Return y in {0,1} or None for tie/unknown."""
    w = row[winner_col]

    # numeric encoding
    if isinstance(w, (int, np.integer)):
        if int(w) == 1:
            return 1
        if int(w) == 0:
            return 0
        return None

    if isinstance(w, float) and not np.isnan(w):
        if int(w) == 1:
            return 1
        if int(w) == 0:
            return 0
        return None

    if isinstance(w, str):
        wl = w.strip().lower()
        if wl in ['model_a', 'a', 'left', '1', 'win_a', 'winner_a']:
            return 1
        if wl in ['model_b', 'b', 'right', '0', 'win_b', 'winner_b']:
            return 0
        if wl in ['tie', 'draw', 'both', 'neither', 'equal']:
            return None

        # winner is literally the model name
        if w == row[a_col]:
            return 1
        if w == row[b_col]:
            return 0

    return None


@dataclass
class ArenaDataset:
    comparisons: np.ndarray
    id2model: List[str]
    df_norm: pd.DataFrame


def load_chatbot_arena_pairwise(
    path: str,
    model_a_col: str = 'model_a',
    model_b_col: str = 'model_b',
    winner_col: str = 'winner',
    category_col: Optional[str] = 'category',
    tie_policy: str = 'drop',  # 'drop' | 'split'
) -> ArenaDataset:
    df = _read_df(path)

    required = {model_a_col, model_b_col, winner_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

    # Build model index
    models = pd.unique(pd.concat([df[model_a_col], df[model_b_col]], axis=0)).tolist()
    models = [m for m in models if isinstance(m, str) and len(m) > 0]
    model2id: Dict[str, int] = {m: i for i, m in enumerate(sorted(set(models)))}
    id2model = [m for m, _ in sorted(model2id.items(), key=lambda kv: kv[1])]

    rows = []
    comps: List[Tuple[int, int, int]] = []

    for _, r in df.iterrows():
        a_name = r[model_a_col]
        b_name = r[model_b_col]
        if a_name not in model2id or b_name not in model2id:
            continue

        a_id = model2id[a_name]
        b_id = model2id[b_name]

        y = _infer_winner(r, model_a_col, model_b_col, winner_col)
        if y is None:
            if tie_policy == 'split':
                # Add both outcomes deterministically (0.5/0.5 tie split)
                for yy in (1, 0):
                    comps.append((a_id, b_id, int(yy)))
                    row = {'a_id': a_id, 'b_id': b_id, 'y': int(yy), 'is_tie': True}
                    if category_col and category_col in df.columns:
                        row['category'] = r[category_col]
                    rows.append(row)
            continue

        comps.append((a_id, b_id, int(y)))
        row = {'a_id': a_id, 'b_id': b_id, 'y': int(y), 'is_tie': False}
        if category_col and category_col in df.columns:
            row['category'] = r[category_col]
        rows.append(row)

    df_norm = pd.DataFrame(rows)
    comparisons = np.array(comps, dtype=np.int64)

    return ArenaDataset(comparisons=comparisons, id2model=id2model, df_norm=df_norm)
