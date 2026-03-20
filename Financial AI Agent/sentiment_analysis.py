"""
sentiment_analysis.py  –  Singleton sentiment pipeline.

The HuggingFace pipeline is expensive to load (~2-5 s).
We load it exactly ONCE at import time (in a background thread so startup
is non-blocking) and reuse the singleton for every subsequent call.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_pipeline = None
_pipeline_lock  = threading.Lock()
_pipeline_ready = threading.Event()


def _load_pipeline():
    global _pipeline
    try:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
        )
        logger.info("Sentiment pipeline loaded.")
    except Exception as e:
        logger.warning("Could not load sentiment pipeline: %s", e)
        _pipeline = None
    finally:
        _pipeline_ready.set()


# Kick off loading immediately at import time (background thread)
threading.Thread(target=_load_pipeline, daemon=True).start()


# A small rotating set of sample headlines keeps results varied without
# hitting the network every call.
_HEADLINES = [
    "Global stock markets show mixed investor sentiment amid economic uncertainty.",
    "Tech stocks surge as investors bet on AI-driven growth.",
    "Markets retreat on inflation concerns and rising interest rates.",
    "Analysts forecast steady growth for blue-chip equities.",
    "Emerging market volatility weighs on global risk appetite.",
]
_headline_idx = 0


def get_sentiment(wait_timeout: float = 5.0) -> str:
    """
    Return 'Positive', 'Negative', or 'Neutral'.

    Waits up to *wait_timeout* seconds for the pipeline to be ready before
    falling back to a heuristic.
    """
    global _headline_idx

    _pipeline_ready.wait(timeout=wait_timeout)

    headline = _HEADLINES[_headline_idx % len(_HEADLINES)]
    _headline_idx += 1

    if _pipeline is None:
        # Deterministic fallback: scan for obvious keywords
        hl = headline.lower()
        if any(w in hl for w in ("surge", "growth", "rally", "bull")):
            return "Positive"
        if any(w in hl for w in ("retreat", "volatility", "concerns", "decline")):
            return "Negative"
        return "Neutral"

    try:
        result = _pipeline(headline)[0]
        label  = result["label"].upper()
        if "POS" in label:
            return "Positive"
        if "NEG" in label:
            return "Negative"
        return "Neutral"
    except Exception as e:
        logger.warning("Sentiment inference failed: %s", e)
        return "Neutral"
