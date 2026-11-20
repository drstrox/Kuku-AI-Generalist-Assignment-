"""
Feedback loop — learns *this editor's* taste from 👍/👎 votes.

Each vote stores the candidate's normalized feature vector. A balanced
LogisticRegression (standardized) distills votes into P(editor likes it).
The probability is blended into the final score with weight α = min(0.5, n/16),
so the model only starts to matter once there is real signal (≥6 votes).

Votes persist to data/feedback.jsonl (or KUKU_FEEDBACK_FILE). On ephemeral
hosts (Streamlit Cloud / HF Spaces free tier) the file resets on restart —
download it from the sidebar to keep it, and upload it to restore.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

FEATURE_KEYS = [
    "hook", "cliffhanger", "intensity", "emotional_swing",
    "premise_alignment", "pace", "dialogue_ratio", "episode_prior",
    "raw_pace_beats_per_s", "raw_dialogue_ratio", "duration_s",
]

FEEDBACK_FILE = Path(os.environ.get("KUKU_FEEDBACK_FILE", Path(__file__).resolve().parent.parent / "data" / "feedback.jsonl"))
MIN_VOTES = 6


def load_votes() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    votes = []
    with open(FEEDBACK_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    votes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return votes


def record_vote(features: dict, label: int, promo_id: str, episode: int, note: str = "") -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "promo_id": promo_id,
        "episode": episode,
        "label": int(label),
        "features": {k: float(features.get(k, 0.0)) for k in FEATURE_KEYS},
        "note": note[:280],
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


def clear_votes() -> None:
    if FEEDBACK_FILE.exists():
        FEEDBACK_FILE.unlink()


def _matrix(votes):
    X = np.array([[v["features"].get(k, 0.0) for k in FEATURE_KEYS] for v in votes], dtype=np.float32)
    y = np.array([v["label"] for v in votes], dtype=np.int32)
    return X, y


def fit_feedback_model(votes):
    """Return (model, scaler, n_used) or None if not enough signal/classes."""
    if len(votes) < MIN_VOTES:
        return None
    X, y = _matrix(votes)
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(scaler.transform(X), y)
    return model, scaler, len(votes)


def feedback_probabilities(candidates, votes=None) -> dict[int, float]:
    """Map id(candidate) -> P(like). Empty dict when model not trainable."""
    votes = load_votes() if votes is None else votes
    fitted = fit_feedback_model(votes)
    if fitted is None:
        return {}
    model, scaler, _ = fitted
    X = np.array([[c.features.get(k, 0.0) for k in FEATURE_KEYS] for c in candidates], dtype=np.float32)
    probs = model.predict_proba(scaler.transform(X))[:, 1]
    return {id(c): float(p) for c, p in zip(candidates, probs)}


def feedback_blend_weight(n_votes: int) -> float:
    return min(0.5, n_votes / 16.0)


def feedback_summary() -> dict:
    votes = load_votes()
    up = sum(1 for v in votes if v["label"] == 1)
    return {
        "n_votes": len(votes),
        "up": up,
        "down": len(votes) - up,
        "trainable": len(votes) >= MIN_VOTES and len({v["label"] for v in votes}) == 2,
        "blend_weight": feedback_blend_weight(len(votes)),
    }


def feature_importance() -> list[tuple[str, float]]:
    votes = load_votes()
    fitted = fit_feedback_model(votes)
    if fitted is None:
        return []
    model, _, _ = fitted
    return sorted(zip(FEATURE_KEYS, model.coef_[0]), key=lambda kv: -abs(kv[1]))
