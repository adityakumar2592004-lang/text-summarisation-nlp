"""
src/abstractive.py
Abstractive summarisation models using pre-trained Hugging Face transformers.

Two models are implemented:
- **BARTSummarizer**: facebook/bart-base (or bart-large-cnn for better quality)
- **T5Summarizer**: t5-small (or t5-base for better quality)

Both fall back to the original pre-trained weights if no fine-tuned checkpoint
is found in the ``models/`` directory.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from .preprocessing import Preprocessor
from .summarizer_base import Summarizer

logger = logging.getLogger(__name__)


# ── Abstract base for neural (abstractive) summarisers ────────────────────────

class AbstractiveSummarizer(Summarizer):
    """Base class for Hugging Face sequence-to-sequence summarisers.

    Loads the model and tokenizer lazily (on first call to :meth:`summarize`)
    so that importing this module is cheap.

    Parameters
    ----------
    cfg:
        Full configuration dictionary.
    model_name:
        Hugging Face hub model identifier.
    checkpoint_dir:
        Path to a local fine-tuned checkpoint.  If the directory does not
        exist or is empty, falls back to *model_name*.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        model_name: str,
        checkpoint_dir: str,
    ) -> None:
        super().__init__(cfg)
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.preprocessor = Preprocessor(cfg)

        self._tokenizer = None
        self._model = None
        self._device = None

    # ── Lazy model loading ───────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load tokenizer and model (called automatically on first use)."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Decide whether to load the fine-tuned checkpoint or the base model
        load_path = self._resolve_model_path()

        logger.info("Loading tokenizer from: %s", load_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            load_path, use_fast=True
        )

        logger.info("Loading model from: %s (device: %s)", load_path, self._device)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        self._model.to(self._device)
        self._model.eval()

    def _resolve_model_path(self) -> str:
        """Return the checkpoint path if it exists, else the hub model name."""
        if (
            self.checkpoint_dir.exists()
            and any(self.checkpoint_dir.iterdir())
        ):
            logger.info(
                "Fine-tuned checkpoint found at '%s'.", self.checkpoint_dir
            )
            return str(self.checkpoint_dir)
        else:
            logger.warning(
                "No fine-tuned checkpoint found at '%s'. "
                "Using pre-trained model '%s' as baseline. "
                "Run scripts/run_abstractive.py --train first to fine-tune.",
                self.checkpoint_dir,
                self.model_name,
            )
            return self.model_name

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def device(self):
        if self._device is None:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    # ── Generation ───────────────────────────────────────────────────────────

    def _generate(self, input_text: str, max_input_length: int) -> str:
        """Encode *input_text* and decode a beam-search summary.

        Parameters
        ----------
        input_text:
            Preprocessed text to summarise.
        max_input_length:
            Maximum number of encoder tokens.

        Returns
        -------
        str
            Generated summary string.
        """
        import torch

        gen_cfg = self.cfg["generation"]

        inputs = self.tokenizer(
            input_text,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                num_beams=gen_cfg["num_beams"],
                min_length=gen_cfg["min_length"],
                max_length=gen_cfg["max_length"],
                no_repeat_ngram_size=gen_cfg["no_repeat_ngram_size"],
                length_penalty=gen_cfg["length_penalty"],
                early_stopping=gen_cfg["early_stopping"],
            )

        summary = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return summary

    def summarize(self, text: str) -> str:
        raise NotImplementedError("Subclasses must implement summarize()")

    def summarize_batch(self, texts: list[str]) -> list[str]:
        """Summarise a list of texts efficiently using batched inference.

        Parameters
        ----------
        texts:
            List of document strings.

        Returns
        -------
        list[str]
            List of summary strings.
        """
        return [self.summarize(t) for t in texts]


# ── BART Summarizer ───────────────────────────────────────────────────────────

class BARTSummarizer(AbstractiveSummarizer):
    """Abstractive summariser backed by facebook/bart-base (or bart-large-cnn).

    BART (Lewis et al., 2019) is a denoising autoencoder pre-trained for
    sequence-to-sequence tasks.  ``facebook/bart-large-cnn`` is the version
    specifically fine-tuned on CNN/DailyMail and gives the best out-of-the-box
    quality.

    Parameters
    ----------
    cfg:
        Full configuration dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        bart_cfg = cfg["abstractive"]["bart"]
        super().__init__(
            cfg,
            model_name=bart_cfg["model_name"],
            checkpoint_dir=bart_cfg["checkpoint_dir"],
        )
        self.max_input_length: int = cfg["preprocessing"]["bart_max_input_length"]

    @property
    def name(self) -> str:
        return "BARTSummarizer"

    def summarize(self, text: str) -> str:
        """Generate a summary using BART.

        Parameters
        ----------
        text:
            Raw article text.

        Returns
        -------
        str
            Abstractive summary.
        """
        clean = self.preprocessor.clean_text(text)
        if not clean:
            return ""
        return self._generate(clean, self.max_input_length)


# ── T5 Summarizer ─────────────────────────────────────────────────────────────

class T5Summarizer(AbstractiveSummarizer):
    """Abstractive summariser backed by t5-small (or t5-base).

    T5 (Raffel et al., 2019) is a text-to-text transformer.  All tasks are
    framed as text generation; summarisation requires the prefix
    ``"summarize: "`` to be prepended to the input.

    Parameters
    ----------
    cfg:
        Full configuration dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        t5_cfg = cfg["abstractive"]["t5"]
        super().__init__(
            cfg,
            model_name=t5_cfg["model_name"],
            checkpoint_dir=t5_cfg["checkpoint_dir"],
        )
        self.prefix: str = t5_cfg.get("prefix", "summarize: ")
        self.max_input_length: int = cfg["preprocessing"]["max_input_length"]

    @property
    def name(self) -> str:
        return "T5Summarizer"

    def summarize(self, text: str) -> str:
        """Generate a summary using T5.

        Parameters
        ----------
        text:
            Raw article text.

        Returns
        -------
        str
            Abstractive summary.
        """
        clean = self.preprocessor.clean_text(text)
        if not clean:
            return ""
        # T5 requires the task prefix
        input_text = self.prefix + clean
        return self._generate(input_text, self.max_input_length)
