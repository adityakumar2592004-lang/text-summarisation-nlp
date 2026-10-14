"""
scripts/run_abstractive.py
Run abstractive summarisers (BART and T5) on the test set.

If a fine-tuned checkpoint exists in models/, it is used; otherwise the
pre-trained baseline is used (with a clear log message).

Usage (from project root):
    python scripts/run_abstractive.py
    python scripts/run_abstractive.py --quick-test
    python scripts/run_abstractive.py --model bart    # BART only
    python scripts/run_abstractive.py --model t5      # T5 only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.abstractive import BARTSummarizer, T5Summarizer
from src.data_loader import DataLoader
from src.utils import load_config, set_seed, setup_logging, ensure_dir, save_dataframe

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(
    cfg_path: str = "config.yaml",
    quick_test: bool = False,
    model_choice: str = "both",
) -> None:
    """Run abstractive summarisers on the test split and save predictions.

    Parameters
    ----------
    cfg_path:
        Path to the YAML config file.
    quick_test:
        If True, use the small quick_test subset.
    model_choice:
        Which models to run: ``"bart"``, ``"t5"``, or ``"both"``.
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

    # ── Initialise selected models ─────────────────────────────────────────
    all_models: dict[str, object] = {}
    if model_choice in ("bart", "both"):
        all_models["BARTSummarizer"] = BARTSummarizer(cfg)
    if model_choice in ("t5", "both"):
        all_models["T5Summarizer"] = T5Summarizer(cfg)

    # ── Generate predictions ───────────────────────────────────────────────
    all_preds: dict[str, list[str]] = {}

    for name, model in all_models.items():
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
    out_path = results_dir / "abstractive_predictions.csv"
    saved_path = save_dataframe(df, out_path, index=False)
    logger.info("Predictions saved to %s", saved_path)
    print(f"\n✓ Abstractive predictions saved to {saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run abstractive summarisers")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument(
        "--model",
        choices=["bart", "t5", "both"],
        default="both",
        help="Which model(s) to run",
    )
    args = parser.parse_args()
    main(args.config, args.quick_test, args.model)
