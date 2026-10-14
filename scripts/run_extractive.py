"""
scripts/run_extractive.py
Run both extractive summarisers (TF-IDF and TextRank) on the test set,
save predictions to CSV, and print a quick progress bar.

Usage (from project root):
    python scripts/run_extractive.py
    python scripts/run_extractive.py --quick-test   # 10 samples only
    python scripts/run_extractive.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_loader import DataLoader
from src.extractive import TFIDFSummarizer, TextRankSummarizer
from src.utils import load_config, set_seed, setup_logging, ensure_dir, save_dataframe

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(cfg_path: str = "config.yaml", quick_test: bool = False) -> None:
    """Run extractive summarisers on the test split and save predictions.

    Parameters
    ----------
    cfg_path:
        Path to the YAML config file.
    quick_test:
        If True, use the small quick_test subset.
    """
    cfg = load_config(cfg_path)
    setup_logging(cfg.get("log_level", "INFO"))
    set_seed(cfg.get("seed", 42))

    results_dir = ensure_dir(cfg["evaluation"]["results_dir"])

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    dl = DataLoader(cfg, quick_test=quick_test)
    splits = dl.load_data()
    test_texts, test_refs = dl.get_texts_and_summaries(splits["test"])
    logger.info("Test samples: %d", len(test_texts))

    # ── Initialise models ──────────────────────────────────────────────────
    models = {
        "TFIDFSummarizer": TFIDFSummarizer(cfg),
        "TextRankSummarizer": TextRankSummarizer(cfg),
    }

    # ── Generate predictions ───────────────────────────────────────────────
    all_preds: dict[str, list[str]] = {}

    for name, model in models.items():
        logger.info("Running %s on %d samples …", name, len(test_texts))
        preds = []
        for text in tqdm(test_texts, desc=name, unit="doc"):
            try:
                preds.append(model.summarize(text))
            except Exception as exc:
                logger.warning("Error in %s: %s — using empty string.", name, exc)
                preds.append("")
        all_preds[name] = preds
        logger.info("✓ %s done.", name)

    # ── Save to CSV ────────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "article": test_texts,
            "reference": test_refs,
            **all_preds,
        }
    )
    out_path = results_dir / "extractive_predictions.csv"
    saved_path = save_dataframe(df, out_path, index=False)
    logger.info("Predictions saved to %s", saved_path)
    print(f"\n✓ Extractive predictions saved to {saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run extractive summarisers")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()
    main(args.config, args.quick_test)
