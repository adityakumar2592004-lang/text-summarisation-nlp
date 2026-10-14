"""
src/utils.py
Shared utilities: config loading, logging setup, and seed setting.

All other modules import from here so that behaviour is consistent
across scripts, notebooks, and the Streamlit app.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and return the YAML config as a nested dictionary.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.  Defaults to ``config.yaml`` in the
        current working directory.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with a simple timestamped format.

    Parameters
    ----------
    level:
        Logging level string (e.g., ``"INFO"``, ``"DEBUG"``).

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=numeric_level,
    )
    # Suppress noisy third-party loggers
    for noisy in ("transformers", "datasets", "tokenizers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("summarisation")


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # optional import – may not be available in minimal envs
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Path helpers ──────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The (now existing) directory path.
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path


def save_dataframe(df, target_path: str | Path, **kwargs) -> Path:
    """Save a pandas DataFrame with graceful fallback if OS folder protection is enabled."""
    target_path = Path(target_path)
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, **kwargs)
        return target_path
    except OSError:
        fallback_path = Path.home() / ".cache" / "text-summarisation-nlp" / target_path.name
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fallback_path, **kwargs)
        logging.getLogger("summarisation").warning(
            "Controlled Folder Access prevented writing to %s; saved to %s instead.",
            target_path,
            fallback_path,
        )
        return fallback_path


def resolve_data_file(path: str | Path) -> Path:
    """Find a file either at the primary path or in the fallback cache folder."""
    p = Path(path)
    if p.exists():
        return p
    fallback = Path.home() / ".cache" / "text-summarisation-nlp" / p.name
    if fallback.exists():
        return fallback
    return p

