"""
src/preprocessing.py
Text cleaning, sentence tokenisation (NLTK), and subword tokenisation
(Hugging Face BPE) used across all models.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Any

import nltk
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize

logger = logging.getLogger(__name__)

# ── Ensure required NLTK resources are present ───────────────────────────────
def _download_nltk_data() -> None:
    """Silently download required NLTK corpora if not already cached."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", package)
            nltk.download(package, quiet=True)


_download_nltk_data()


class Preprocessor:
    """Cleans raw text and tokenises it at the sentence and word level.

    This class is intentionally stateless (no model weights) so that it can
    be used cheaply by both extractive and abstractive modules.

    Parameters
    ----------
    cfg:
        Full configuration dictionary from ``config.yaml``.
    """

    # Pattern to strip CNN-style bylines like "(CNN) --" or "(Reuters) --"
    _CNN_PREFIX = re.compile(r"^\s*\([^)]+\)\s*[-–—]+\s*")
    # Pattern to collapse multiple whitespace / newlines into a single space
    _WHITESPACE = re.compile(r"\s+")
    # Pattern to strip common HTML leftovers
    _HTML_TAGS = re.compile(r"<[^>]+>")
    # Odd characters: keep ASCII printable + common unicode punctuation
    _ODD_CHARS = re.compile(r"[^\x20-\x7E\u2018-\u2019\u201C\u201D\u2013\u2014]")

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    # ── Public API ───────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Apply a sequence of cleaning steps to raw article text.

        Steps (in order):
        1. Strip HTML tags.
        2. Remove CNN-style byline prefixes.
        3. Collapse whitespace.
        4. Remove odd / non-printable characters.

        Parameters
        ----------
        text:
            Raw article text.

        Returns
        -------
        str
            Cleaned text with normalised whitespace.
        """
        if not isinstance(text, str):
            return ""
        text = self._HTML_TAGS.sub(" ", text)
        text = self._CNN_PREFIX.sub("", text)
        text = self._ODD_CHARS.sub("", text)
        text = self._WHITESPACE.sub(" ", text)
        return text.strip()

    def sent_tokenize(self, text: str) -> list[str]:
        """Split text into a list of sentences using NLTK.

        Parameters
        ----------
        text:
            Cleaned (or raw) text.

        Returns
        -------
        list[str]
            Non-empty sentence strings.
        """
        sentences = nltk_sent_tokenize(text)
        # Drop empty or very short sentences (likely artefacts)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def word_tokenize(self, text: str) -> list[str]:
        """Tokenise text into words, lower-cased, no punctuation.

        Parameters
        ----------
        text:
            A single sentence or short paragraph.

        Returns
        -------
        list[str]
            Lower-cased word tokens without punctuation.
        """
        tokens = nltk_word_tokenize(text.lower())
        # Keep only alphabetic tokens (removes punctuation and numbers)
        return [t for t in tokens if t.isalpha()]

    def subword_tokenize(
        self,
        text: str,
        tokenizer,  # transformers PreTrainedTokenizer
        max_length: int | None = None,
        target_text: str | None = None,
        target_max_length: int | None = None,
    ) -> dict:
        """Tokenise a source text (and optionally a target) with a HF tokenizer.

        Handles T5's "summarize: " prefix automatically if it is already
        prepended by the caller.

        Parameters
        ----------
        text:
            Source article text.
        tokenizer:
            A Hugging Face ``PreTrainedTokenizer`` or ``PreTrainedTokenizerFast``.
        max_length:
            Maximum number of tokens for the encoder input.  Defaults to the
            tokenizer's ``model_max_length``.
        target_text:
            Reference summary text.  If provided, the labels are also encoded.
        target_max_length:
            Maximum number of tokens for the decoder target.

        Returns
        -------
        dict
            Tokenizer output dict with ``input_ids``, ``attention_mask``, and
            (if *target_text* is given) ``labels``.
        """
        max_length = max_length or tokenizer.model_max_length

        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        if target_text is not None:
            with tokenizer.as_target_tokenizer():
                target_encoding = tokenizer(
                    target_text,
                    max_length=target_max_length or 128,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            encoding["labels"] = target_encoding["input_ids"]

        return encoding

    def batch_clean(self, texts: list[str]) -> list[str]:
        """Apply :meth:`clean_text` to a list of texts.

        Parameters
        ----------
        texts:
            List of raw texts.

        Returns
        -------
        list[str]
            List of cleaned texts, same length as input.
        """
        return [self.clean_text(t) for t in texts]
