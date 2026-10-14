"""
src/train.py
Fine-tuning script for BART and T5 using Hugging Face Seq2SeqTrainer.

Usage (from project root):
    python -m src.train --model bart
    python -m src.train --model t5
    python -m src.train --model both   # fine-tune both sequentially

Notes
-----
- Uses fp16 only if a CUDA GPU is detected; falls back to fp32 on CPU
  with a warning that training will be slow.
- The fine-tuned model is saved to ``models/bart_finetuned`` or
  ``models/t5_finetuned`` as specified in config.yaml.
- On CPU, set ``num_train_epochs: 1`` and batch size to 1 in config.yaml
  to keep training time manageable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_gpu() -> bool:
    """Return True if a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _make_tokenize_fn(tokenizer, text_col: str, summary_col: str,
                      prefix: str, max_input: int, max_target: int):
    """Return a dataset-map-compatible tokenisation function."""

    def tokenize_fn(batch):
        inputs = [prefix + doc for doc in batch[text_col]]
        targets = batch[summary_col]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target,
                truncation=True,
                padding="max_length",
            )

        # Replace padding token id (typically 0) with -100 so it is ignored
        # by the cross-entropy loss.
        label_ids = [
            [(-100 if tok == tokenizer.pad_token_id else tok) for tok in ids]
            for ids in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize_fn


def _compute_metrics_factory(tokenizer):
    """Return a compute_metrics function compatible with Seq2SeqTrainer."""
    from rouge_score import rouge_scorer as rs
    import numpy as np

    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Replace -100 in labels (ignored tokens) with pad_token_id
        labels = [
            [tok if tok != -100 else tokenizer.pad_token_id for tok in label]
            for label in labels
        ]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        r1, r2, rl = [], [], []
        for pred, ref in zip(decoded_preds, decoded_labels):
            if pred.strip() and ref.strip():
                s = scorer.score(ref, pred)
                r1.append(s["rouge1"].fmeasure)
                r2.append(s["rouge2"].fmeasure)
                rl.append(s["rougeL"].fmeasure)

        return {
            "rouge1": np.mean(r1) if r1 else 0.0,
            "rouge2": np.mean(r2) if r2 else 0.0,
            "rougeL": np.mean(rl) if rl else 0.0,
        }

    return compute_metrics


# ── Main training function ────────────────────────────────────────────────────

def fine_tune(
    cfg: dict[str, Any],
    model_key: str = "bart",
    quick_test: bool = False,
) -> None:
    """Fine-tune a single model (BART or T5) using Seq2SeqTrainer.

    Parameters
    ----------
    cfg:
        Configuration dictionary.
    model_key:
        Either ``"bart"`` or ``"t5"``.
    quick_test:
        If True, use tiny subset sizes for a fast end-to-end check.
    """
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from .data_loader import DataLoader
    from .preprocessing import Preprocessor
    from .utils import set_seed

    set_seed(cfg.get("seed", 42))
    has_gpu = _check_gpu()

    if not has_gpu:
        logger.warning(
            "⚠  No CUDA GPU detected. Training will be very slow on CPU. "
            "Consider using Google Colab (Free GPU) – see notebooks/colab_finetune.ipynb. "
            "Reducing num_train_epochs and batch size in config.yaml is strongly recommended."
        )

    # ── Config values ─────────────────────────────────────────────────────
    model_cfg = cfg["abstractive"][model_key]
    train_cfg = cfg["training"]
    pre_cfg = cfg["preprocessing"]

    model_name: str = model_cfg["model_name"]
    checkpoint_dir = Path(model_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg["abstractive"]["t5"].get("prefix", "") if model_key == "t5" else ""
    max_input = (
        pre_cfg["bart_max_input_length"]
        if model_key == "bart"
        else pre_cfg["max_input_length"]
    )
    max_target = pre_cfg["max_target_length"]

    text_col = cfg["dataset"]["text_column"]
    summary_col = cfg["dataset"]["summary_column"]

    # ── Load data ──────────────────────────────────────────────────────────
    dl = DataLoader(cfg, quick_test=quick_test)
    splits = dl.load_data()

    # ── Load tokenizer and model ───────────────────────────────────────────
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    logger.info("Loading model: %s", model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ── Tokenise datasets ──────────────────────────────────────────────────
    tokenize_fn = _make_tokenize_fn(
        tokenizer, text_col, summary_col, prefix, max_input, max_target
    )
    remove_cols = list(splits["train"].column_names)

    logger.info("Tokenising training split …")
    tokenised_train = splits["train"].map(
        tokenize_fn, batched=True, remove_columns=remove_cols
    )
    logger.info("Tokenising validation split …")
    tokenised_val = splits["validation"].map(
        tokenize_fn, batched=True, remove_columns=remove_cols
    )

    # ── Training arguments ────────────────────────────────────────────────
    use_fp16 = has_gpu and train_cfg.get("fp16", False)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=(
            train_cfg["per_device_train_batch_size"] if has_gpu else 1
        ),
        per_device_eval_batch_size=(
            train_cfg["per_device_eval_batch_size"] if has_gpu else 1
        ),
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        predict_with_generate=train_cfg["predict_with_generate"],
        fp16=use_fp16,
        generation_max_length=max_target,
        report_to="none",  # disable wandb / mlflow
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    compute_metrics = _compute_metrics_factory(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("Starting fine-tuning of %s …", model_name)
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    logger.info("Saving fine-tuned model to %s …", checkpoint_dir)
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    logger.info("✓ Fine-tuning complete. Model saved to %s", checkpoint_dir)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BART or T5 for summarisation")
    parser.add_argument(
        "--model",
        choices=["bart", "t5", "both"],
        default="both",
        help="Which model(s) to fine-tune (default: both)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use tiny dataset subset for a fast smoke test",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    # Set up logging and load config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import load_config, setup_logging

    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))

    models_to_train = ["bart", "t5"] if args.model == "both" else [args.model]
    for m in models_to_train:
        logger.info("=" * 60)
        logger.info("Fine-tuning: %s", m.upper())
        logger.info("=" * 60)
        fine_tune(cfg, model_key=m, quick_test=args.quick_test)
