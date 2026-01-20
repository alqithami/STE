"""AgentBench-style pairwise data ingestion.

This mirrors the Arena loader but includes an optional 'environment' field.

Expected schema (defaults):
- agent_a, agent_b: strings
- winner: one of {"agent_a","agent_b","tie"} or similar encodings
- environment (optional): string for grouping

Output:
- comparisons: numpy array (m,3) of (a_id,b_id,y) with y=1 -> a wins
- id2agent: list mapping id->name
- df_norm: normalized DataFrame (a_id,b_id,y,environment(optional))

Tie handling:
- tie_policy='drop': remove tie rows.
- tie_policy='split': represent tie as two rows (a beats b) and (b beats a).

No placeholder data: this module only encodes existing rows.
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
    w = row[winner_col]

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
        if wl in ['agent_a', 'a', 'left', '1', 'winner_a']:
            return 1
        if wl in ['agent_b', 'b', 'right', '0', 'winner_b']:
            return 0
        if wl in ['tie', 'draw', 'both', 'neither', 'equal']:
            return None

        # winner is literally the agent name
        if w == row[a_col]:
            return 1
        if w == row[b_col]:
            return 0

    return None


@dataclass
class AgentBenchDataset:
    comparisons: np.ndarray
    id2agent: List[str]
    df_norm: pd.DataFrame


def load_agentbench_pairwise(
    path: str,
    agent_a_col: str = 'agent_a',
    agent_b_col: str = 'agent_b',
    winner_col: str = 'winner',
    env_col: Optional[str] = 'environment',
    tie_policy: str = 'drop',
) -> AgentBenchDataset:
    df = _read_df(path)

    required = {agent_a_col, agent_b_col, winner_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

    agents = pd.unique(pd.concat([df[agent_a_col], df[agent_b_col]], axis=0)).tolist()
    agents = [a for a in agents if isinstance(a, str) and len(a) > 0]
    agent2id: Dict[str, int] = {a: i for i, a in enumerate(sorted(set(agents)))}
    id2agent = [a for a, _ in sorted(agent2id.items(), key=lambda kv: kv[1])]

    rows = []
    comps: List[Tuple[int, int, int]] = []

    for _, r in df.iterrows():
        a_name = r[agent_a_col]
        b_name = r[agent_b_col]
        if a_name not in agent2id or b_name not in agent2id:
            continue

        a_id = agent2id[a_name]
        b_id = agent2id[b_name]

        y = _infer_winner(r, agent_a_col, agent_b_col, winner_col)
        if y is None:
            if tie_policy == 'split':
                for yy in (1, 0):
                    comps.append((a_id, b_id, int(yy)))
                    row = {'a_id': a_id, 'b_id': b_id, 'y': int(yy), 'is_tie': True}
                    if env_col and env_col in df.columns:
                        row['environment'] = r[env_col]
                    rows.append(row)
            continue

        comps.append((a_id, b_id, int(y)))
        row = {'a_id': a_id, 'b_id': b_id, 'y': int(y), 'is_tie': False}
        if env_col and env_col in df.columns:
            row['environment'] = r[env_col]
        rows.append(row)

    df_norm = pd.DataFrame(rows)
    comparisons = np.array(comps, dtype=np.int64)

    return AgentBenchDataset(comparisons=comparisons, id2agent=id2agent, df_norm=df_norm)
