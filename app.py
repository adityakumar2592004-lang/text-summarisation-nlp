"""
app.py – Streamlit web demo for Text Summarisation using NLP
=============================================================
Run with:
    streamlit run app.py

Features
--------
- Paste any text and get an extractive or abstractive summary.
- "Load sample article" button for quick demo.
- Sidebar controls: model choice, top-k / max-length, num_beams.
- Comparison view: all four models side by side.
- Optional ROUGE evaluation against a reference summary.
- Word count, compression ratio, and spinner while generating.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Make src importable when running from the project root ────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page configuration (must be the very first Streamlit call) ────────────────
st.set_page_config(
    page_title="Text Summarisation using NLP",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils import load_config, set_seed

# ── Sample article for demo ────────────────────────────────────────────────────
SAMPLE_ARTICLE = """(CNN) -- Scientists have discovered a new species of deep-sea fish
near the Mariana Trench, the deepest part of the world's oceans. The creature,
named Pseudoliparis swirei, was found at depths of more than 8,000 metres —
deeper than any fish has been recorded before. The discovery was made by a team
from the University of Washington and Newcastle University during an expedition
that deployed special pressure-resistant bait stations. The translucent, small
fish thrives in near-freezing temperatures and crushing pressures hundreds of
times greater than at the surface. Researchers believe the species can survive
at these extreme depths because the high-pressure environment protects it from
the diseases that affect fish at shallower depths. The findings, published in
the journal Zootaxa, could help scientists understand how life adapts to the
most extreme conditions on Earth. Lead researcher Dr. Thomas Linley said the
fish are surprisingly active at these depths, feeding on small crustaceans that
also live in the trench. The discovery underscores how little we know about the
deep ocean, which covers more than half the Earth's surface yet remains largely
unexplored. Future expeditions are planned to study the fish's biology in more
detail and to search for additional new species in other deep-sea trenches
around the world."""


# ── Config & seed ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_config():
    """Load config once and cache."""
    cfg = load_config("config.yaml")
    set_seed(cfg.get("seed", 42))
    return cfg


cfg = get_config()


# ── Model loading (cached so models are not reloaded on every interaction) ────

@st.cache_resource(show_spinner="Loading TF-IDF Summarizer …")
def get_tfidf_model():
    from src.extractive import TFIDFSummarizer
    return TFIDFSummarizer(cfg)


@st.cache_resource(show_spinner="Loading TextRank Summarizer …")
def get_textrank_model():
    from src.extractive import TextRankSummarizer
    return TextRankSummarizer(cfg)


@st.cache_resource(show_spinner="Loading BART model … (first load may take 1–2 min)")
def get_bart_model():
    from src.abstractive import BARTSummarizer
    return BARTSummarizer(cfg)


@st.cache_resource(show_spinner="Loading T5 model … (first load may take 1–2 min)")
def get_t5_model():
    from src.abstractive import T5Summarizer
    return T5Summarizer(cfg)


# ── Evaluation helper ─────────────────────────────────────────────────────────

@st.cache_resource
def get_evaluator():
    from src.evaluate import Evaluator
    return Evaluator(cfg)


def compute_rouge_live(prediction: str, reference: str) -> dict:
    """Compute ROUGE scores and return a display-ready dict."""
    evaluator = get_evaluator()
    return evaluator.rouge([prediction], [reference])


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Model")
    mode = st.selectbox(
        "Summarisation type",
        ["Extractive – TF-IDF", "Extractive – TextRank",
         "Abstractive – BART", "Abstractive – T5"],
    )

    st.subheader("Extractive controls")
    top_k = st.slider("Top-k sentences", min_value=1, max_value=10, value=3)

    st.subheader("Abstractive controls")
    max_len = st.slider("Max output length (tokens)", 50, 200, 128, step=10)
    num_beams = st.slider("Beam search width", 1, 8, 4)

    st.markdown("---")
    st.caption(
        "**Note:** BART and T5 load ~500 MB each. "
        "The first summary may take a few seconds."
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("📝 Text Summarisation using NLP")
st.markdown(
    "Compare **Extractive** (TF-IDF, TextRank) and **Abstractive** (BART, T5) "
    "summarisation models on any text. Scroll down to compare all models at once."
)

# Sample article button
col_btn, col_space = st.columns([1, 5])
with col_btn:
    load_sample = st.button("📰 Load sample article")

# Text input
default_text = SAMPLE_ARTICLE if load_sample else ""
input_text = st.text_area(
    "Paste your article or text here:",
    value=default_text,
    height=250,
    placeholder="Paste a news article, paper abstract, or any long text …",
)

# Validate input
if input_text and len(input_text.split()) < 20:
    st.warning("⚠️ Input is very short. Please provide at least 20 words for a meaningful summary.")
elif len(input_text) > 20_000:
    st.warning("⚠️ Input is very long and will be truncated to the model's maximum context window.")

summarize_btn = st.button("✨ Summarise", type="primary", disabled=not input_text.strip())

# ── Generate single summary ───────────────────────────────────────────────────
if summarize_btn and input_text.strip():
    # Override config generation params from sidebar
    gen_cfg = cfg["generation"].copy()
    gen_cfg["max_length"] = max_len
    gen_cfg["num_beams"] = num_beams
    cfg["generation"] = gen_cfg

    start_time = time.time()

    with st.spinner("Generating summary …"):
        try:
            if mode == "Extractive – TF-IDF":
                model = get_tfidf_model()
                model.top_k = top_k
                summary = model.summarize(input_text)

            elif mode == "Extractive – TextRank":
                model = get_textrank_model()
                model.top_k = top_k
                summary = model.summarize(input_text)

            elif mode == "Abstractive – BART":
                model = get_bart_model()
                summary = model.summarize(input_text)

            else:  # T5
                model = get_t5_model()
                summary = model.summarize(input_text)

        except Exception as e:
            st.error(f"❌ Error generating summary: {e}")
            summary = ""

    elapsed = time.time() - start_time

    if summary:
        st.subheader("📋 Summary")
        st.success(summary)

        # Stats
        input_words = len(input_text.split())
        summary_words = len(summary.split())
        compression = round((1 - summary_words / max(input_words, 1)) * 100, 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input words", input_words)
        c2.metric("Summary words", summary_words)
        c3.metric("Compression", f"{compression}%")
        c4.metric("Time", f"{elapsed:.1f}s")

        # Optional ROUGE evaluation
        with st.expander("📊 Evaluate against reference summary (optional)"):
            ref_text = st.text_area(
                "Paste reference summary:",
                height=80,
                placeholder="Paste the gold/reference summary here to compute ROUGE …",
            )
            if ref_text.strip():
                rouge = compute_rouge_live(summary, ref_text)
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("ROUGE-1", f"{rouge['rouge1']:.4f}")
                rc2.metric("ROUGE-2", f"{rouge['rouge2']:.4f}")
                rc3.metric("ROUGE-L", f"{rouge['rougeL']:.4f}")

# ── Compare all models ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Compare All Models Side by Side")
compare_btn = st.button(
    "⚡ Run All Models",
    disabled=not input_text.strip(),
    help="Generate summaries from all four models and display them side by side.",
)

if compare_btn and input_text.strip():
    models_to_run = {
        "TF-IDF": (get_tfidf_model, {}),
        "TextRank": (get_textrank_model, {}),
        "BART": (get_bart_model, {}),
        "T5": (get_t5_model, {}),
    }

    results: dict[str, str] = {}
    progress = st.progress(0, text="Running models …")

    for i, (name, (loader_fn, _)) in enumerate(models_to_run.items()):
        progress.progress((i) / len(models_to_run), text=f"Running {name} …")
        with st.spinner(f"Running {name} …"):
            try:
                model = loader_fn()
                if hasattr(model, "top_k"):
                    model.top_k = top_k
                results[name] = model.summarize(input_text)
            except Exception as e:
                results[name] = f"❌ Error: {e}"

    progress.progress(1.0, text="Done!")

    cols = st.columns(len(results))
    for col, (name, summary_text) in zip(cols, results.items()):
        with col:
            st.markdown(f"**{name}**")
            st.info(summary_text if summary_text else "*(empty)*")
            if summary_text and not summary_text.startswith("❌"):
                st.caption(f"{len(summary_text.split())} words")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "B.Tech (AI & ML) Minor Project · Text Summarisation using NLP · "
    "Models: TF-IDF · TextRank · BART · T5"
)
