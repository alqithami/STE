"""YAML config loading utilities.

This is intentionally minimal: it supports the master config used by the
experiment runner (ste/run.py).

If PyYAML is not installed, install it via:
    pip install pyyaml
"""

from __future__ import annotations

from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
        ) from e

    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping/dict. Got {type(cfg)}")
    return cfg


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict b into dict a (without mutating inputs)."""
    out: Dict[str, Any] = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
