"""config_loader.py

Loads the two config files and merges them into a single dict:

  config.yaml         – DSP / training / model parameters (original, untouched)
  runtime_config.yaml – operational settings (inference paths, recording, LCD …)

runtime_config.yaml values take precedence over config.yaml on key collision.
Any key can be further overridden at call-time via the *overrides* argument.

Typical usage
-------------
    from config_loader import load_config
    cfg = load_config()                          # both defaults
    cfg = load_config(overrides={"inference": {"threshold": 0.15}})

    # Explicit paths (e.g. during training, only the base config is needed)
    cfg = load_config(base_path="config.yaml", runtime_path=None)
"""

import os
import yaml
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(
    base_path: Optional[str] = None,
    runtime_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge base config + runtime config, then apply overrides.

    Args:
        base_path:    Path to config.yaml.
                      Default: config.yaml next to this file.
        runtime_path: Path to runtime_config.yaml.
                      Default: runtime_config.yaml next to this file.
                      Pass ``None`` to skip (e.g. for training scripts).
        overrides:    Nested dict applied on top of everything else.
                      Example: {"inference": {"threshold": 0.20}}

    Returns:
        Merged config dict.
    """
    # ── base config (always required) ────────────────────────
    if base_path is None:
        base_path = os.path.join(_HERE, "config.yaml")
    cfg = _load_yaml(base_path)

    # ── runtime config (optional) ─────────────────────────────
    if runtime_path is None:
        default_runtime = os.path.join(_HERE, "runtime_config.yaml")
        if os.path.exists(default_runtime):
            runtime_path = default_runtime

    if runtime_path is not None:
        _deep_update(cfg, _load_yaml(runtime_path))

    # ── call-time overrides ───────────────────────────────────
    if overrides:
        _deep_update(cfg, overrides)

    return cfg


# ── helpers ──────────────────────────────────────────────────

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge *updates* into *base* in-place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
