"""
Lazy-loaded ML models with graceful degradation.

Layers
------
1. Emotion/intensity — j-hartmann/emotion-english-distilroberta-base
   (7 emotions + neutral). Scores every dialogue/action beat once per run.
2. Embeddings — sentence-transformers/all-MiniLM-L6-v2 (384-dim).
   Used for hook-archetype matching, premise alignment and MMR diversity.
3. If transformers/torch are unavailable or KUKU_USE_ML=0, everything falls
   back to a transparent lexicon so the app still runs (degraded mode).

Swap any model by setting KUKU_EMOTION_MODEL / KUKU_EMBEDDING_MODEL.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

USE_ML = os.environ.get("KUKU_USE_ML", "1") != "0"
EMOTION_MODEL = os.environ.get("KUKU_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
EMBEDDING_MODEL = os.environ.get("KUKU_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# ---------------------------------------------------------------- lexicon fallback
_LEXICON = {
    "anger": ["slap", "fury", "furious", "angry", "rage", "scream", "hate", "strike", "hit", "shove", "kill", "murder", "bet", "liar", "lying", "traitor", "enough"],
    "disgust": ["disgust", "filthy", "wretched", "loath", "despise", "poison", "filth", "worm", "scum", "coward"],
    "fear": ["tremble", "afraid", "terror", "scared", "panic", "dread", "horror", "danger", "threat", "gasp", "shaking", "fear", "run"],
    "joy": ["love", "kiss", "embrace", "smile", "happy", "wedding", "joy", "darling", "marry", "hold you", "sweetheart"],
    "sadness": ["tears", "cry", "sob", "grief", "heartbroken", "pain", "mourn", "lost", "alone", "sacrifice", "dying"],
    "surprise": ["gasp", "shock", "stunned", "unbelievable", "what", "impossible", "secret", "revealed", "truth", "no"],
}

_WORD_RE = re.compile(r"[a-z']+")


def lexicon_emotion(text: str) -> dict:
    toks = _WORD_RE.findall(text.lower())
    if not toks:
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
    counts = {e: 0 for e in _LEXICON}
    for t in toks:
        for e, words in _LEXICON.items():
            if t in words:
                counts[e] += 1
    raw = {e: c / len(toks) * 6.0 for e, c in counts.items()}  # scaled
    total_nonneutral = sum(raw.values())
    # squash
    probs = {e: 1 - pow(2.718281828, -v) for e, v in raw.items()}
    s = sum(probs.values())
    neutral = max(0.05, 1.0 - s * 0.6)
    out = {e: probs[e] * (1 - neutral) for e in probs}
    out["neutral"] = neutral
    tot = sum(out.values())
    return {e: out[e] / tot for e in EMOTIONS}


@lru_cache(maxsize=1)
def _emotion_pipeline():
    if not USE_ML:
        return None
    try:
        from transformers import pipeline

        return pipeline("text-classification", model=EMOTION_MODEL, top_k=None, truncation=True, max_length=256)
    except Exception as exc:  # pragma: no cover - offline / low-memory
        print(f"[models] emotion model unavailable ({exc}); using lexicon fallback")
        return None


@lru_cache(maxsize=1)
def _embedder():
    if not USE_ML:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:  # pragma: no cover
        print(f"[models] embedding model unavailable ({exc}); embeddings disabled")
        return None


def emotion_scores(texts: list[str]) -> list[dict]:
    """Return [{emotion: prob}, ...] aligned with texts."""
    if not texts:
        return []
    clf = _emotion_pipeline()
    if clf is None:
        return [lexicon_emotion(t) for t in texts]
    try:
        results = clf([t if t.strip() else "." for t in texts], batch_size=32)
        out = []
        for res in results:
            d = {item["label"].lower(): float(item["score"]) for item in res}
            # normalise keys to our EMOTIONS set
            out.append({e: d.get(e, 0.0) for e in EMOTIONS})
        return out
    except Exception as exc:
        print(f"[models] emotion inference failed ({exc}); using lexicon fallback")
        return [lexicon_emotion(t) for t in texts]


def embed_texts(texts: list[str]):
    """Return numpy matrix [n, d] or None if embeddings unavailable."""
    model = _embedder()
    if model is None:
        return None
    try:
        return model.encode([t if t.strip() else "." for t in texts], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    except Exception as exc:
        print(f"[models] embedding failed ({exc})")
        return None


def ml_status() -> dict:
    """Cheap status: what is CONFIGURED. Models load lazily on first use."""
    return {
        "use_ml": USE_ML,
        "emotion_model": EMOTION_MODEL if USE_ML else "lexicon-fallback",
        "embedding_model": EMBEDDING_MODEL if USE_ML else "disabled",
        "emotion_available": None if USE_ML else True,   # None = will load on first run
        "embeddings_available": None if USE_ML else False,
    }


def ml_status_verified() -> dict:
    """Actually loads the models and reports availability (used by batch runs)."""
    return {
        "use_ml": USE_ML,
        "emotion_model": EMOTION_MODEL if USE_ML else "lexicon-fallback",
        "embedding_model": EMBEDDING_MODEL if USE_ML else "disabled",
        "emotion_available": _emotion_pipeline() is not None if USE_ML else True,
        "embeddings_available": _embedder() is not None if USE_ML else False,
    }
