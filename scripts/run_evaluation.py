"""
scripts/run_evaluation.py
Load saved prediction CSVs, compute ROUGE + BERTScore for all models,
print the comparison table, and save results/metrics.csv + results/comparison.png.

Usage (from project root):
    # Run AFTER run_extractive.py and run_abstractive.py
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --quick-test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluate import Evaluator
from src.utils import load_config, set_seed, setup_logging, resolve_data_file

import pandas as pd

logger = logging.getLogger(__name__)

# Mapping: prediction CSV column name → display name in the results table
MODEL_COLUMNS = {
    "TFIDFSummarizer": "TF-IDF",
    "TextRankSummarizer": "TextRank",
    "BARTSummarizer": "BART",
    "T5Summarizer": "T5",
}


def main(cfg_path: str = "config.yaml", quick_test: bool = False) -> None:
    """Load predictions, run evaluation, and save outputs.

    Parameters
    ----------
    cfg_path:
        Path to the YAML config file.
    quick_test:
        Unused here; kept for interface consistency.
    """
    cfg = load_config(cfg_path)
    setup_logging(cfg.get("log_level", "INFO"))
    set_seed(cfg.get("seed", 42))

    results_dir = Path(cfg["evaluation"]["results_dir"])

    # ── Load prediction CSVs ──────────────────────────────────────────────
    extractive_path = resolve_data_file(results_dir / "extractive_predictions.csv")
    abstractive_path = resolve_data_file(results_dir / "abstractive_predictions.csv")

    dfs: list[pd.DataFrame] = []
    for path in [extractive_path, abstractive_path]:
        if path.exists():
            dfs.append(pd.read_csv(path))
            logger.info("Loaded predictions from %s (%d rows)", path, len(dfs[-1]))
        else:
            logger.warning("Prediction file not found: %s — skipping.", path)

    if not dfs:
        logger.error(
            "No prediction files found in %s. "
            "Run run_extractive.py and run_abstractive.py first.",
            results_dir,
        )
        sys.exit(1)

    # Merge on common columns
    merged = dfs[0]
    for df in dfs[1:]:
        extra_cols = [c for c in df.columns if c not in merged.columns]
        for col in extra_cols:
            merged[col] = df[col].values

    references = merged["reference"].fillna("").tolist()
    articles = merged["article"].fillna("").tolist()

    # ── Build model_predictions dict ──────────────────────────────────────
    model_predictions: dict[str, list[str]] = {}
    for col, display_name in MODEL_COLUMNS.items():
        if col in merged.columns:
            model_predictions[display_name] = merged[col].fillna("").tolist()
        else:
            logger.warning("Column '%s' not found in predictions; skipping.", col)

    if not model_predictions:
        logger.error("No model prediction columns found. Exiting.")
        sys.exit(1)

    # ── Evaluate ──────────────────────────────────────────────────────────
    evaluator = Evaluator(cfg)
    metrics_df = evaluator.compare_models(model_predictions, references)

    # ── Print markdown table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (ROUGE + BERTScore)")
    print("=" * 60)
    try:
        print(metrics_df.to_markdown(index=False))
    except Exception:
        print(metrics_df.to_string(index=False))
    print("=" * 60 + "\n")

    # ── Save outputs ───────────────────────────────────────────────────────
    evaluator.save_metrics_csv(metrics_df)
    evaluator.save_comparison_chart(metrics_df)
    evaluator.save_examples(articles, references, model_predictions)

    print(f"✓ Metrics  → {cfg['evaluation']['metrics_file']}")
    print(f"✓ Chart    → {cfg['evaluation']['chart_file']}")
    print(f"✓ Examples → {cfg['evaluation']['examples_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all saved model predictions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()
    main(args.config, args.quick_test)
