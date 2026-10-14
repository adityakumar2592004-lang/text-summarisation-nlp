"""
src/extractive.py
Extractive summarisation models.

Two algorithms are implemented:
- **TFIDFSummarizer**: scores sentences by mean TF-IDF weight.
- **TextRankSummarizer**: builds a cosine-similarity graph and runs PageRank.

Both pick the top-k sentences and return them in *original document order*
so the output reads naturally.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import Preprocessor
from .summarizer_base import Summarizer

logger = logging.getLogger(__name__)


# ── Base class for extractive methods ────────────────────────────────────────

class ExtractiveSummarizer(Summarizer):
    """Base class for extractive summarisers.

    Adds a shared :meth:`_select_sentences` helper that picks the top-k
    sentences (by score) and returns them in their original order.

    Parameters
    ----------
    cfg:
        Full configuration dictionary.
    top_k:
        Number of sentences to extract.  Defaults to ``cfg["extractive"]["top_k"]``.
    """

    def __init__(self, cfg: dict[str, Any], top_k: int | None = None) -> None:
        super().__init__(cfg)
        self.preprocessor = Preprocessor(cfg)
        self.top_k: int = top_k if top_k is not None else cfg["extractive"]["top_k"]

    def _select_sentences(
        self, sentences: list[str], scores: list[float]
    ) -> str:
        """Select top-k sentences by score and reassemble in original order.

        Parameters
        ----------
        sentences:
            All sentences from the document.
        scores:
            Score for each sentence (same length as *sentences*).

        Returns
        -------
        str
            The joined top-k sentences.
        """
        k = min(self.top_k, len(sentences))
        if k == 0:
            return ""
        # Get indices of top-k sentences sorted by score descending
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        # Restore original document order
        selected_indices = sorted(ranked_indices)
        return " ".join(sentences[i] for i in selected_indices)

    def summarize(self, text: str) -> str:
        raise NotImplementedError("Subclasses must implement summarize()")


# ── TF-IDF Summarizer ─────────────────────────────────────────────────────────

class TFIDFSummarizer(ExtractiveSummarizer):
    """Scores each sentence by the mean TF-IDF weight of its words.

    Algorithm
    ---------
    1. Clean and sentence-tokenise the document.
    2. Fit a TF-IDF vectoriser on all sentences.
    3. Score each sentence as the mean of its non-zero TF-IDF values.
    4. Select the top-k highest-scoring sentences (in original order).

    Parameters
    ----------
    cfg:
        Full configuration dictionary.
    top_k:
        Number of sentences to extract.
    """

    def __init__(self, cfg: dict[str, Any], top_k: int | None = None) -> None:
        super().__init__(cfg, top_k)
        self._model_name = "TF-IDF"

    @property
    def name(self) -> str:
        return "TFIDFSummarizer"

    def summarize(self, text: str) -> str:
        """Generate an extractive summary using TF-IDF sentence scoring.

        Parameters
        ----------
        text:
            Raw or cleaned article text.

        Returns
        -------
        str
            Extracted summary (top-k sentences joined by a space).
        """
        # Step 1: preprocess
        clean = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.sent_tokenize(clean)

        if len(sentences) <= self.top_k:
            # Document is too short; return as-is
            return " ".join(sentences)

        # Step 2: TF-IDF vectorisation
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # All sentences are stop-words or empty
            logger.warning("TF-IDF vectorisation failed; returning first sentences.")
            return " ".join(sentences[: self.top_k])

        # Step 3: score each sentence (mean of non-zero values)
        scores: list[float] = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray().flatten()
            non_zero = row[row > 0]
            scores.append(float(non_zero.mean()) if len(non_zero) > 0 else 0.0)

        # Step 4: select and return
        return self._select_sentences(sentences, scores)


# ── TextRank Summarizer ───────────────────────────────────────────────────────

class TextRankSummarizer(ExtractiveSummarizer):
    """Uses PageRank on a sentence-similarity graph (TextRank algorithm).

    Algorithm
    ---------
    1. Clean and sentence-tokenise the document.
    2. Represent each sentence as a TF-IDF vector.
    3. Build a fully-connected graph where edge weight = cosine similarity.
    4. Run NetworkX PageRank to score sentences.
    5. Select the top-k highest-scoring sentences (in original order).

    Reference: Mihalcea & Tarau (2004), "TextRank: Bringing Order into Texts."

    Parameters
    ----------
    cfg:
        Full configuration dictionary.
    top_k:
        Number of sentences to extract.
    damping:
        PageRank damping factor (default 0.85).
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        top_k: int | None = None,
        damping: float = 0.85,
    ) -> None:
        super().__init__(cfg, top_k)
        self.damping = damping

    @property
    def name(self) -> str:
        return "TextRankSummarizer"

    def summarize(self, text: str) -> str:
        """Generate an extractive summary using TextRank.

        Parameters
        ----------
        text:
            Raw or cleaned article text.

        Returns
        -------
        str
            Extracted summary (top-k sentences joined by a space).
        """
        # Step 1: preprocess
        clean = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.sent_tokenize(clean)

        if len(sentences) <= self.top_k:
            return " ".join(sentences)

        if len(sentences) < 2:
            return sentences[0] if sentences else ""

        # Step 2: TF-IDF vectorisation
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences).toarray()
        except ValueError:
            logger.warning("TextRank TF-IDF failed; falling back to first sentences.")
            return " ".join(sentences[: self.top_k])

        # Step 3: build similarity matrix (cosine similarity)
        sim_matrix = cosine_similarity(tfidf_matrix)
        # Remove self-similarity (diagonal)
        np.fill_diagonal(sim_matrix, 0.0)

        # Step 4: build graph and run PageRank
        graph = nx.from_numpy_array(sim_matrix)
        try:
            pr_scores = nx.pagerank(
                graph, alpha=self.damping, max_iter=200, tol=1.0e-6
            )
            scores = [pr_scores[i] for i in range(len(sentences))]
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge; using uniform scores.")
            scores = [1.0 / len(sentences)] * len(sentences)

        # Step 5: select and return
        return self._select_sentences(sentences, scores)
