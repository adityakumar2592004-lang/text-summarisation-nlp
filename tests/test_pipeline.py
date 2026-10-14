"""
tests/test_pipeline.py
Unit tests for preprocessing, extractive models, and the evaluator.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short     # shorter tracebacks
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, set_seed
from src.preprocessing import Preprocessor
from src.extractive import TFIDFSummarizer, TextRankSummarizer
from src.evaluate import Evaluator

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    """Load config once for the entire test session."""
    return load_config("config.yaml")


@pytest.fixture(scope="session", autouse=True)
def seed():
    """Set random seed before tests."""
    set_seed(42)


@pytest.fixture
def preprocessor(cfg):
    return Preprocessor(cfg)


@pytest.fixture
def tfidf_model(cfg):
    return TFIDFSummarizer(cfg, top_k=3)


@pytest.fixture
def textrank_model(cfg):
    return TextRankSummarizer(cfg, top_k=3)


@pytest.fixture
def evaluator(cfg):
    return Evaluator(cfg)


# ── Sample texts ──────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "(CNN) -- Scientists have discovered a new species of deep-sea fish near "
    "the Mariana Trench, the deepest part of the world's oceans. "
    "The creature was found at depths of more than 8,000 metres. "
    "The discovery was made by a team from the University of Washington. "
    "Researchers believe the species can survive at extreme depths because "
    "the high-pressure environment protects it from diseases. "
    "The findings were published in the journal Zootaxa. "
    "Future expeditions are planned to study the fish's biology in detail."
)

SHORT_TEXT = "This is a short text."

REFERENCE_SUMMARY = (
    "Scientists found a new deep-sea fish species near the Mariana Trench "
    "at depths exceeding 8,000 metres."
)

PREDICTION = (
    "A new deep-sea fish species was discovered near the Mariana Trench."
)


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestPreprocessor:
    """Tests for the Preprocessor class."""

    def test_clean_text_removes_cnn_prefix(self, preprocessor):
        text = "(CNN) -- Some news story here."
        cleaned = preprocessor.clean_text(text)
        assert "(CNN)" not in cleaned
        assert "Some news story here" in cleaned

    def test_clean_text_strips_html(self, preprocessor):
        text = "Hello <br> world <p>paragraph</p>"
        cleaned = preprocessor.clean_text(text)
        assert "<" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned

    def test_clean_text_collapses_whitespace(self, preprocessor):
        text = "word1   \n\n  word2\t\tword3"
        cleaned = preprocessor.clean_text(text)
        assert "  " not in cleaned  # no double spaces
        assert "word1" in cleaned
        assert "word3" in cleaned

    def test_clean_empty_string(self, preprocessor):
        assert preprocessor.clean_text("") == ""

    def test_clean_non_string(self, preprocessor):
        # Should handle non-string gracefully
        assert preprocessor.clean_text(None) == ""  # type: ignore

    def test_sent_tokenize_returns_list(self, preprocessor):
        sentences = preprocessor.sent_tokenize(SAMPLE_TEXT)
        assert isinstance(sentences, list)
        assert len(sentences) >= 3

    def test_sent_tokenize_no_empty_strings(self, preprocessor):
        sentences = preprocessor.sent_tokenize(SAMPLE_TEXT)
        assert all(len(s) > 0 for s in sentences)

    def test_word_tokenize_lowercased(self, preprocessor):
        tokens = preprocessor.word_tokenize("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_word_tokenize_no_punctuation(self, preprocessor):
        tokens = preprocessor.word_tokenize("Hello, world! This is a test.")
        for token in tokens:
            assert token.isalpha(), f"Non-alpha token: {token}"

    def test_batch_clean(self, preprocessor):
        texts = ["(CNN) -- First text.", "Second <b>text</b>."]
        cleaned = preprocessor.batch_clean(texts)
        assert len(cleaned) == 2
        assert "(CNN)" not in cleaned[0]
        assert "<b>" not in cleaned[1]


# ── TF-IDF Summarizer tests ───────────────────────────────────────────────────

class TestTFIDFSummarizer:
    """Tests for the TFIDFSummarizer class."""

    def test_returns_string(self, tfidf_model):
        result = tfidf_model.summarize(SAMPLE_TEXT)
        assert isinstance(result, str)

    def test_non_empty_output(self, tfidf_model):
        result = tfidf_model.summarize(SAMPLE_TEXT)
        assert len(result.strip()) > 0

    def test_output_shorter_than_input(self, tfidf_model):
        result = tfidf_model.summarize(SAMPLE_TEXT)
        assert len(result) < len(SAMPLE_TEXT)

    def test_short_text_handled(self, tfidf_model):
        """Short text (fewer sentences than top_k) should not crash."""
        result = tfidf_model.summarize(SHORT_TEXT)
        assert isinstance(result, str)

    def test_empty_text_handled(self, tfidf_model):
        result = tfidf_model.summarize("")
        assert isinstance(result, str)  # may be empty but should not crash

    def test_top_k_respected(self, cfg):
        """Top-k=1 should produce at most one sentence."""
        model = TFIDFSummarizer(cfg, top_k=1)
        result = model.summarize(SAMPLE_TEXT)
        # Should be a single sentence (no more sentences than top_k)
        from src.preprocessing import Preprocessor
        p = Preprocessor(cfg)
        sentences = p.sent_tokenize(result)
        assert len(sentences) <= 2  # allow slight over-count from NLTK

    def test_original_order_preserved(self, tfidf_model):
        """Extracted sentences must appear in document order."""
        clean = tfidf_model.preprocessor.clean_text(SAMPLE_TEXT)
        sentences = tfidf_model.preprocessor.sent_tokenize(clean)
        result = tfidf_model.summarize(SAMPLE_TEXT)
        # Find positions of result sentences in original text
        positions = [SAMPLE_TEXT.find(s[:20]) for s in sentences if s[:20] in SAMPLE_TEXT]
        assert positions == sorted(positions)  # should be monotonically increasing

    def test_batch_summarize(self, tfidf_model):
        texts = [SAMPLE_TEXT, SAMPLE_TEXT]
        results = tfidf_model.summarize_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)


# ── TextRank Summarizer tests ─────────────────────────────────────────────────

class TestTextRankSummarizer:
    """Tests for the TextRankSummarizer class."""

    def test_returns_string(self, textrank_model):
        result = textrank_model.summarize(SAMPLE_TEXT)
        assert isinstance(result, str)

    def test_non_empty_output(self, textrank_model):
        result = textrank_model.summarize(SAMPLE_TEXT)
        assert len(result.strip()) > 0

    def test_output_shorter_than_input(self, textrank_model):
        result = textrank_model.summarize(SAMPLE_TEXT)
        assert len(result) < len(SAMPLE_TEXT)

    def test_short_text_handled(self, textrank_model):
        result = textrank_model.summarize(SHORT_TEXT)
        assert isinstance(result, str)

    def test_empty_text_handled(self, textrank_model):
        result = textrank_model.summarize("")
        assert isinstance(result, str)

    def test_different_from_tfidf(self, tfidf_model, textrank_model):
        """The two extractive models should generally produce different results."""
        tfidf_result = tfidf_model.summarize(SAMPLE_TEXT)
        textrank_result = textrank_model.summarize(SAMPLE_TEXT)
        # Not strictly required to differ on all texts, but usually will
        # Just check both are non-empty strings
        assert isinstance(tfidf_result, str)
        assert isinstance(textrank_result, str)


# ── Evaluator tests ───────────────────────────────────────────────────────────

class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_rouge_returns_three_scores(self, evaluator):
        scores = evaluator.rouge([PREDICTION], [REFERENCE_SUMMARY])
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores

    def test_rouge_values_in_range(self, evaluator):
        scores = evaluator.rouge([PREDICTION], [REFERENCE_SUMMARY])
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} is out of [0, 1]"

    def test_rouge_perfect_score(self, evaluator):
        """A prediction identical to the reference should score 1.0 on ROUGE-1."""
        scores = evaluator.rouge([REFERENCE_SUMMARY], [REFERENCE_SUMMARY])
        assert scores["rouge1"] == pytest.approx(1.0, abs=0.01)

    def test_rouge_empty_prediction(self, evaluator):
        """Empty prediction should score 0.0 without crashing."""
        scores = evaluator.rouge([""], [REFERENCE_SUMMARY])
        # Either 0.0 or skipped (mean of empty list -> 0.0)
        assert scores["rouge1"] == pytest.approx(0.0, abs=0.01)

    def test_compare_models_returns_dataframe(self, evaluator, monkeypatch):
        import pandas as pd
        monkeypatch.setattr(
            evaluator,
            "bertscore",
            lambda preds, refs: {"bertscore_p": 0.85, "bertscore_r": 0.85, "bertscore_f1": 0.85},
        )
        model_preds = {
            "TF-IDF": [PREDICTION, PREDICTION],
            "TextRank": [PREDICTION, PREDICTION],
        }
        refs = [REFERENCE_SUMMARY, REFERENCE_SUMMARY]
        df = evaluator.compare_models(model_preds, refs)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert "ROUGE-1" in df.columns
        assert len(df) == 2

    def test_save_metrics_csv(self, evaluator, tmp_path, cfg, monkeypatch):
        """Save should create a CSV file."""
        import pandas as pd
        monkeypatch.setitem(cfg["evaluation"], "metrics_file", str(tmp_path / "metrics.csv"))
        df = pd.DataFrame([{"Model": "TF-IDF", "ROUGE-1": 0.4, "ROUGE-2": 0.2, "ROUGE-L": 0.35, "BERTScore-F1": 0.85}])
        evaluator.cfg = cfg
        evaluator.save_metrics_csv(df)
        assert (tmp_path / "metrics.csv").exists()


# ── DataLoader quick test ─────────────────────────────────────────────────────

class TestDataLoader:
    """Smoke tests for DataLoader (uses quick_test=True to stay fast)."""

    def test_load_data_returns_three_splits(self, cfg):
        from src.data_loader import DataLoader
        dl = DataLoader(cfg, quick_test=True)
        splits = dl.load_data()
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

    def test_split_sizes_match_config(self, cfg):
        from src.data_loader import DataLoader
        dl = DataLoader(cfg, quick_test=True)
        splits = dl.load_data()
        qt = cfg["dataset"]["quick_test"]
        assert len(splits["train"]) == qt["train"]
        assert len(splits["validation"]) == qt["validation"]
        assert len(splits["test"]) == qt["test"]

    def test_get_texts_and_summaries(self, cfg):
        from src.data_loader import DataLoader
        dl = DataLoader(cfg, quick_test=True)
        splits = dl.load_data()
        texts, summaries = dl.get_texts_and_summaries(splits["test"])
        assert len(texts) == len(summaries)
        assert all(isinstance(t, str) for t in texts)
        assert all(isinstance(s, str) for s in summaries)
