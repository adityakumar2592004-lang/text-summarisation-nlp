"""
src/evaluate.py
Evaluation utilities: ROUGE and BERTScore computation, comparison tables,
and bar chart generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Evaluator:
    """Compute ROUGE and BERTScore metrics for one or more models.

    Parameters
    ----------
    cfg:
        Full configuration dictionary from ``config.yaml``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.results_dir = Path(cfg["evaluation"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded metric objects
        self._rouge_scorer = None
        self._bertscore_scorer = None

    # ── ROUGE ────────────────────────────────────────────────────────────────

    def rouge(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

        Parameters
        ----------
        predictions:
            List of model-generated summaries.
        references:
            List of reference (gold) summaries.

        Returns
        -------
        dict
            Keys: ``rouge1``, ``rouge2``, ``rougeL`` – all F1 scores (0–1).
        """
        from rouge_score import rouge_scorer as rs

        scorer = rs.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        r1_list, r2_list, rl_list = [], [], []
        for pred, ref in zip(predictions, references):
            if not pred.strip() or not ref.strip():
                continue
            scores = scorer.score(ref, pred)
            r1_list.append(scores["rouge1"].fmeasure)
            r2_list.append(scores["rouge2"].fmeasure)
            rl_list.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": float(np.mean(r1_list)) if r1_list else 0.0,
            "rouge2": float(np.mean(r2_list)) if r2_list else 0.0,
            "rougeL": float(np.mean(rl_list)) if rl_list else 0.0,
        }

    # ── BERTScore ─────────────────────────────────────────────────────────────

    def bertscore(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        """Compute BERTScore F1 (using a lightweight distilbert model).

        BERTScore (Zhang et al., 2019) measures semantic similarity between
        hypothesis and reference using contextual embeddings.

        Parameters
        ----------
        predictions:
            List of model-generated summaries.
        references:
            List of reference summaries.

        Returns
        -------
        dict
            Keys: ``bertscore_p``, ``bertscore_r``, ``bertscore_f1``.
        """
        try:
            from bert_score import score as bert_score_fn

            bert_model = self.cfg["evaluation"]["bertscore_model"]
            logger.info("Computing BERTScore with model: %s", bert_model)

            # Filter empty pairs
            valid_preds, valid_refs = [], []
            for p, r in zip(predictions, references):
                if p.strip() and r.strip():
                    valid_preds.append(p)
                    valid_refs.append(r)

            if not valid_preds:
                return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

            P, R, F1 = bert_score_fn(
                valid_preds,
                valid_refs,
                model_type=bert_model,
                lang="en",
                nthreads=1,
                verbose=False,
            )
            return {
                "bertscore_p": float(P.mean()),
                "bertscore_r": float(R.mean()),
                "bertscore_f1": float(F1.mean()),
            }
        except Exception as exc:
            logger.error("BERTScore computation failed: %s", exc)
            return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

    # ── Compare all models ───────────────────────────────────────────────────

    def compare_models(
        self,
        model_predictions: dict[str, list[str]],
        references: list[str],
    ) -> pd.DataFrame:
        """Evaluate multiple models and return a comparison table.

        Parameters
        ----------
        model_predictions:
            Dictionary mapping model name → list of predictions.
        references:
            List of reference summaries (shared across all models).

        Returns
        -------
        pd.DataFrame
            Rows = models, Columns = metrics.
        """
        rows: list[dict] = []
        for model_name, preds in model_predictions.items():
            logger.info("Evaluating model: %s", model_name)
            rouge_scores = self.rouge(preds, references)
            bert_scores = self.bertscore(preds, references)
            rows.append(
                {
                    "Model": model_name,
                    "ROUGE-1": round(rouge_scores["rouge1"], 4),
                    "ROUGE-2": round(rouge_scores["rouge2"], 4),
                    "ROUGE-L": round(rouge_scores["rougeL"], 4),
                    "BERTScore-F1": round(bert_scores["bertscore_f1"], 4),
                }
            )

        df = pd.DataFrame(rows)
        try:
            logger.info("\nMetrics comparison:\n%s", df.to_markdown(index=False))
        except Exception:
            logger.info("\nMetrics comparison:\n%s", df.to_string(index=False))
        return df

    # ── Save helpers ──────────────────────────────────────────────────────────

    def save_metrics_csv(self, df: pd.DataFrame) -> None:
        """Save the metrics DataFrame to ``results/metrics.csv``."""
        from .utils import save_dataframe
        path = Path(self.cfg["evaluation"]["metrics_file"])
        saved = save_dataframe(df, path, index=False)
        logger.info("Metrics saved to %s", saved)

    def save_comparison_chart(self, df: pd.DataFrame) -> None:
        """Save a grouped bar chart of the metrics comparison."""
        metric_cols = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore-F1"]
        metric_cols = [c for c in metric_cols if c in df.columns]

        models = df["Model"].tolist()
        x = np.arange(len(models))
        width = 0.18
        n_metrics = len(metric_cols)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        for idx, (metric, color) in enumerate(zip(metric_cols, colors)):
            offset = (idx - n_metrics / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                df[metric].values,
                width,
                label=metric,
                color=color,
                alpha=0.85,
                edgecolor="white",
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.003,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Text Summarisation – Model Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylim(0, min(df[metric_cols].values.max() * 1.25, 1.0))
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        path = Path(self.cfg["evaluation"]["chart_file"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Chart saved to %s", path)
        except OSError:
            fallback = Path.home() / ".cache" / "text-summarisation-nlp" / path.name
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fallback, dpi=150, bbox_inches="tight")
            logger.info("Chart saved to %s", fallback)
        plt.close(fig)

    def save_examples(
        self,
        articles: list[str],
        references: list[str],
        model_predictions: dict[str, list[str]],
    ) -> None:
        """Save side-by-side example summaries to ``results/examples.md``."""
        n = self.cfg["evaluation"]["num_examples"]
        path = Path(self.cfg["evaluation"]["examples_file"])

        lines: list[str] = [
            "# Text Summarisation – Example Outputs\n",
            f"Showing the first **{n}** test-set examples.\n",
            "---\n",
        ]

        for i in range(min(n, len(articles))):
            snippet = articles[i][:500].strip().replace("\n", " ")
            lines.append(f"## Example {i + 1}\n")
            lines.append(f"**Article snippet** (first 500 chars):\n\n> {snippet}…\n")
            lines.append(f"\n**Reference summary:**\n\n> {references[i].strip()}\n")
            for model_name, preds in model_predictions.items():
                lines.append(f"\n**{model_name}:**\n\n> {preds[i].strip()}\n")
            lines.append("\n---\n")

        content = "\n".join(lines)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            logger.info("Examples saved to %s", path)
        except OSError:
            fallback = Path.home() / ".cache" / "text-summarisation-nlp" / path.name
            fallback.parent.mkdir(parents=True, exist_ok=True)
            with open(fallback, "w", encoding="utf-8") as fh:
                fh.write(content)
            logger.info("Examples saved to %s", fallback)
