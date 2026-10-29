"""Config loading and shared paths. config.yaml is the single source of truth."""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# yfinance emits noisy pandas deprecation warnings; they are not actionable here.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*inplace.*")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def rng(cfg: dict[str, Any], offset: int = 0) -> np.random.Generator:
    """Seeded generator; all randomness in the project goes through this."""
    return np.random.default_rng(int(cfg["seed"]) + offset)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
