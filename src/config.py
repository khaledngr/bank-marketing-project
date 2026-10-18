"""Shared access to config.yaml, so no script hard-codes a path, seed or cost."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SEED = CONFIG["seed"]


def path(name: str) -> Path:
    """Absolute path for a `paths:` entry, created on first use."""
    p = ROOT / CONFIG["paths"][name]
    p.mkdir(parents=True, exist_ok=True)
    return p
