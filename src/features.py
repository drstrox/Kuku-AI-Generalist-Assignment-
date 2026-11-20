"""
Candidate enumeration + feature extraction.

A candidate promo is a continuous [start, end) window anchored to beat
boundaries. Per-beat signals (emotion probs, embeddings, keywords) are
computed ONCE per episode, then O(window) aggregation is cheap, so we can
score every 20–60s window without re-running the models.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .models import embed_texts, emotion_scores
from .parser import Episode

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HOOK_KEYWORDS = [
    "slap", "kiss", "marry", "wedding", "divorce", "betray", "secret", "truth",
    "enough", "shut up", "get out", "whore", "bastard", "pregnant", "poison",
    "dead", "kill", "murder", "love", "leave", "liar", "lying", "steal", "stole",
    "heir", "duke", "scandal", "caught", "confess", "dare", "never forgive",
]
CLIFF_KEYWORDS = [
    "cliffhanger", "cut to black", "gasp", "what?", "no!", "unbelievable",
    "heart-stopping", "holding on", "black.",
]

AROUSAL = {"anger": 1.0, "disgust": 0.9, "fear": 0.95, "surprise": 0.85, "sadness": 0.55, "joy": 0.6, "neutral": 0.0}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _load_archetypes():
    with open(DATA_DIR / "hook_archetypes.json") as f:
        return json.load(f)["archetypes"]


@dataclass
class BeatSignals:
    intensity: np.ndarray      # [n_beats] arousal-weighted non-neutral emotion mass
    valence: np.ndarray        # [n_beats] joy minus negative emotions
    emotion_matrix: np.ndarray # [n_beats, 7]
    embeddings: Optional[np.ndarray]   # [n_beats, d] or None
    hook_sem: np.ndarray       # [n_beats, n_archetypes] cosine (0 if embeddings off)
    keyword_hit: np.ndarray    # [n_beats] 0..1
    cliff_hit: np.ndarray      # [n_beats] 0..1


@dataclass(eq=False)
class Candidate:
    episode: int
    start: float
    end: float
    features: dict
    hook_text: str
    hook_time: float
    hook_type: str
    cliff_text: str
    cliff_time: float
    window_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


_SIGNAL_CACHE: dict[str, "BeatSignals"] = {}


def _episode_cache_key(ep: Episode) -> str:
    if ep.source_path:
        return ep.source_path
    import hashlib
    h = hashlib.sha1()
    h.update(str(ep.episode_number).encode())
    for b in ep.beats:
        h.update(f"{b.start}:{b.end}:{b.text}".encode())
    return h.hexdigest()


def build_beat_signals(ep: Episode) -> BeatSignals:
    key = _episode_cache_key(ep)
    if key in _SIGNAL_CACHE:
        return _SIGNAL_CACHE[key]
    texts = [b.text for b in ep.beats]
    emo = emotion_scores(texts)
    emo_mat = np.array([[e[k] for k in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]] for e in emo], dtype=np.float32)
    # columns: 0 anger,1 disgust,2 fear,3 joy,4 neutral,5 sadness,6 surprise
    nonneutral = 1.0 - emo_mat[:, 4]
    arousal_w = np.array([AROUSAL[k] for k in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]], dtype=np.float32)
    intensity = (emo_mat * arousal_w).sum(axis=1)
    valence = emo_mat[:, 3] - (emo_mat[:, 0] + emo_mat[:, 1] + emo_mat[:, 2] + emo_mat[:, 5]) * 0.5

    archetypes = _load_archetypes()
    arch_prompts = [p for a in archetypes for p in a["prompts"]]
    embeddings = embed_texts(texts)
    arch_emb = embed_texts(arch_prompts) if embeddings is not None else None

    if embeddings is not None and arch_emb is not None:
        # group back: mean of prompt cosines per archetype
        cos = embeddings @ arch_emb.T  # [n_beats, n_prompts]
        idx = 0
        hook_sem = np.zeros((len(texts), len(archetypes)), dtype=np.float32)
        for ai, a in enumerate(archetypes):
            k = len(a["prompts"])
            hook_sem[:, ai] = cos[:, idx: idx + k].max(axis=1)
            idx += k
    else:
        # keyword-overlap fallback handled downstream via keyword_hit
        hook_sem = np.zeros((len(texts), len(archetypes)), dtype=np.float32)

    lower_texts = [t.lower() for t in texts]
    keyword_hit = np.array([clamp01(sum(1 for kw in HOOK_KEYWORDS if kw in t) / 2.0) for t in lower_texts], dtype=np.float32)
    cliff_hit = np.array([
        clamp01(sum(1 for kw in CLIFF_KEYWORDS if kw in t) / 2.0 + (0.35 if ("?" in t) else 0.0) + (0.35 if ("—" in t or "..." in t) else 0.0))
        for t in lower_texts
    ], dtype=np.float32)

    sig = BeatSignals(
        intensity=intensity,
        valence=valence,
        emotion_matrix=emo_mat,
        embeddings=embeddings,
        hook_sem=hook_sem,
        keyword_hit=keyword_hit,
        cliff_hit=cliff_hit,
    )
    _SIGNAL_CACHE[key] = sig
    return sig


def enumerate_candidates(ep: Episode, sig: BeatSignals, durations=None,
                         min_dur=18.0, max_dur=65.0) -> list[Candidate]:
    """
    Enumerate promo windows anchored to beat boundaries: every beat START is a
    candidate in-point and every beat END a candidate out-point with 18–65s
    between them. This lets windows land exactly on the script's own
    cliffhanger beats and cut-to-black endings instead of arbitrary fixed
    durations.
    """
    beats = ep.beats
    if not beats:
        return []
    archetypes = _load_archetypes()
    logline_emb = embed_texts([ep.logline or ep.title])
    logline_vec = logline_emb[0] if logline_emb is not None else None
    cliff_marker_idxs = {k for k, b in enumerate(beats)
                         if "cliffhanger" in b.text.lower() or "cut to black" in b.text.lower()}

    cands: list[Candidate] = []
    n = len(beats)

    for i in range(n):
        w_start = beats[i].start
        # walk forward ends until exceeding max_dur
        j = i
        while j < n and beats[j].end - w_start <= max_dur:
            duration = beats[j].end - w_start
            j += 1
            if duration < min_dur:
                continue
            w_end = w_start + duration
            idxs = list(range(i, j))
            if len(idxs) < 3:
                continue

            inten = sig.intensity[idxs]
            val = sig.valence[idxs]
            dialogue_mask = np.array([beats[k].is_dialogue for k in idxs])
            dial_time = sum(beats[k].duration for k in idxs if beats[k].is_dialogue)

            # ---- hook: beats overlapping the first 3.5s
            hook_idxs = [k for k in idxs if beats[k].start < w_start + 3.5] or [idxs[0]]
            best_hook_val, best_hook_beat, best_arch = -1.0, hook_idxs[0], archetypes[0]["label"]
            for k in hook_idxs:
                raw_sem = float(sig.hook_sem[k].max()) if sig.hook_sem.size else 0.0
                sem_e = clamp01((raw_sem - 0.18) / 0.45)   # subtract unrelated-sentence baseline
                kw = float(sig.keyword_hit[k])
                inten_b = clamp01(sig.intensity[k] * 1.35)
                dlg_bonus = 0.06 if beats[k].is_dialogue else 0.0
                hookiness = clamp01(0.50 * sem_e + 0.30 * kw + 0.20 * inten_b + dlg_bonus)
                if hookiness > best_hook_val:
                    best_hook_val = hookiness
                    best_hook_beat = k
                    if sig.hook_sem.size and sig.hook_sem[k].max() > 0:
                        best_arch = archetypes[int(sig.hook_sem[k].argmax())]["label"]
            # prefer a quotable dialogue line for the hook display when it is competitive
            if not beats[best_hook_beat].is_dialogue:
                dlg = [(k, 0.50 * clamp01((float(sig.hook_sem[k].max()) - 0.18) / 0.45) + 0.30 * float(sig.keyword_hit[k]) + 0.20 * clamp01(sig.intensity[k] * 1.35) + 0.06)
                       for k in hook_idxs if beats[k].is_dialogue]
                if dlg:
                    k_best, v_best = max(dlg, key=lambda kv: kv[1])
                    if v_best >= best_hook_val * 0.8:
                        best_hook_beat = k_best
            hook_score = clamp01(best_hook_val * 1.15)

            # ---- cliff: beats overlapping the final 4s
            cliff_idxs = [k for k in idxs if beats[k].end > w_end - 4.0] or [idxs[-1]]
            cliff_parts, best_cliff_beat, best_cliff_part = [], cliff_idxs[-1], -1.0
            for k in cliff_idxs:
                dlg_bonus = 0.05 if beats[k].is_dialogue else 0.0
                comp = clamp01(float(sig.cliff_hit[k]) * 0.75 + clamp01(sig.intensity[k] * 1.2) * 0.45 + dlg_bonus)
                cliff_parts.append(comp)
                if comp > best_cliff_part:
                    best_cliff_part = comp
                    best_cliff_beat = k
            if not beats[best_cliff_beat].is_dialogue:
                dlg = [(k, clamp01(float(sig.cliff_hit[k]) * 0.75 + clamp01(sig.intensity[k] * 1.2) * 0.45 + 0.05))
                       for k in cliff_idxs if beats[k].is_dialogue]
                if dlg:
                    k_best, v_best = max(dlg, key=lambda kv: kv[1])
                    if v_best >= best_cliff_part * 0.8:
                        best_cliff_beat = k_best
            cliff_score = clamp01(max(cliff_parts) * 0.8 + (0.25 if cliff_idxs and any("cliffhanger" in beats[k].text.lower() or "cut to black" in beats[k].text.lower() for k in cliff_idxs) else 0.0))

            # ---- intensity / swing / pace / dialogue
            intensity_score = clamp01(float(inten.mean()) * 1.35)
            swing_score = clamp01(float(val.std()) * 1.6)
            pace = len(idxs) / duration
            pace_score = clamp01(1.0 - abs(pace - 1.8) / 1.6)
            dialogue_ratio = dial_time / duration if duration else 0.0
            dialogue_score = clamp01(1.0 - abs(dialogue_ratio - 0.5) / 0.55)
            if dialogue_ratio < 0.08:
                dialogue_score *= 0.45  # talking-free promo is hard to follow without context

            # ---- premise alignment (self-containment proxy)
            if sig.embeddings is not None and logline_vec is not None:
                win_emb = sig.embeddings[idxs].mean(axis=0)
                win_emb = win_emb / (np.linalg.norm(win_emb) + 1e-8)
                premise = clamp01((float(win_emb @ logline_vec) - 0.35) / 0.35)
            else:
                win_emb = None
                premise = 0.5

            ep_prior = 1.0 if ep.episode_number <= 2 else (0.85 if ep.episode_number <= 8 else (0.7 if ep.episode_number <= 20 else 0.55))

            last_beat_cliff = (j - 1) in cliff_marker_idxs or (j >= n and float(sig.cliff_hit[j - 1]) >= 0.4)
            cliff_end_bonus = 0.35 if last_beat_cliff else (0.15 if j >= n else 0.0)
            cliff_sfx = 1.0 if any(b.kind == "sfx" and "cliffhanger" in b.text.lower() for b in (beats[k] for k in idxs)) else 0.0
            if cliff_end_bonus:
                cliff_score = clamp01(cliff_score + cliff_end_bonus)

            features = {
                "hook": round(hook_score, 4),
                "cliffhanger": round(cliff_score, 4),
                "intensity": round(intensity_score, 4),
                "emotional_swing": round(swing_score, 4),
                "premise_alignment": round(premise, 4),
                "pace": round(pace_score, 4),
                "dialogue_ratio": round(dialogue_score, 4),
                "episode_prior": round(ep_prior, 4),
                "cliff_sfx": round(cliff_sfx, 2),
                "duration_s": round(duration, 2),
                "n_beats": float(len(idxs)),
                "dialogue_seconds": round(dial_time, 2),
                "raw_pace_beats_per_s": round(pace, 3),
                "raw_dialogue_ratio": round(dialogue_ratio, 3),
            }

            def clean(t: str) -> str:
                return re.sub(r"\s+", " ", t).strip().strip('"').strip()

            cands.append(
                Candidate(
                    episode=ep.episode_number,
                    start=round(w_start, 2),
                    end=round(w_end, 2),
                    features=features,
                    hook_text=clean(beats[best_hook_beat].text)[:220],
                    hook_time=round(beats[best_hook_beat].start, 2),
                    hook_type=best_arch,
                    cliff_text=clean(beats[best_cliff_beat].text)[:220],
                    cliff_time=round(beats[best_cliff_beat].end, 2),
                    window_embedding=win_emb,
                )
            )
    return cands
