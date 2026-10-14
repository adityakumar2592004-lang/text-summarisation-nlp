"""
src/summarizer_base.py
Abstract base class for all summarisation models.
Every concrete summariser (extractive or abstractive) must subclass this and
implement the ``summarize`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Summarizer(ABC):
    """Abstract interface for all summarisation models.

    Subclasses must implement :meth:`summarize`.
    They may optionally override :meth:`summarize_batch` for efficiency.

    Parameters
    ----------
    cfg:
        Full configuration dictionary from ``config.yaml``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    @abstractmethod
    def summarize(self, text: str) -> str:
        """Generate a summary for a single input text.

        Parameters
        ----------
        text:
            The document to summarise.

        Returns
        -------
        str
            Generated summary string.
        """
        ...

    def summarize_batch(self, texts: list[str]) -> list[str]:
        """Generate summaries for a list of texts.

        Default implementation loops over :meth:`summarize`.
        Override in subclasses that support true batch inference.

        Parameters
        ----------
        texts:
            List of document strings.

        Returns
        -------
        list[str]
            List of summary strings, same length as *texts*.
        """
        return [self.summarize(t) for t in texts]

    @property
    def name(self) -> str:
        """Human-readable model name (uses the class name by default)."""
        return self.__class__.__name__
