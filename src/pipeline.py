"""
End-to-end pipeline: episodes -> parsed -> features -> scored -> ranked promos.

Used by both the Streamlit app and the batch CLI. Deterministic given the
same inputs and model cache (seed is fixed).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import __version__
from .features import Candidate, build_beat_signals, enumerate_candidates
from .feedback import feedback_blend_weight, feedback_probabilities, load_votes
from .parser import Episode
from .scorer import (AUDIENCE_PRESETS, PLATFORM_PREFS, apply_archetype_boost, base_score,
                     mmr_rerank, non_max_suppression, resolved_weights, why_it_works)
from .validation import validate_promo


@dataclass
class RankedPromo:
    promo_id: str
    episode: int
    segments: list  # [{episode, start, end}]
    duration_s: float
    hook_text: str
    hook_time_s: float
    hook_type: str
    cliff_text: str
    cliff_time_s: float
    why_it_works: str
    scores: dict
    weights: dict
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "promo_id": self.promo_id,
            "rank": self.rank,
            "episode": self.episode,
            "segments": self.segments,
            "duration_s": self.duration_s,
            "hook_text": self.hook_text,
            "hook_time_s": self.hook_time_s,
            "hook_type": self.hook_type,
            "cliffhanger_text": self.cliff_text,
            "cliffhanger_time_s": self.cliff_time_s,
            "why_it_works": self.why_it_works,
            "scores": self.scores,
            "weights": self.weights,
        }


def _promo_id(ep: int, start: float, end: float, segments=None) -> str:
    raw = f"{ep}:{start}:{end}:{segments or ''}"
    return "promo_" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def analyze_episodes(episodes: list[Episode], audience: str, platform: str,
                     desired_duration=None, top_k: int = 12, max_pool: int = 400,
                     diversity_lambda: float = 0.72, use_feedback: bool = True,
                     on_episode=None, collect_stats: dict | None = None) -> list[RankedPromo]:
    weights = resolved_weights(audience, platform)
    lo, hi = PLATFORM_PREFS.get(platform, {}).get("duration_pref", (20, 60))
    if desired_duration:
        lo, hi = max(18.0, desired_duration[0]), min(65.0, desired_duration[1])

    all_cands: list[Candidate] = []
    for idx, ep in enumerate(episodes, start=1):
        if on_episode:
            on_episode(idx, len(episodes), ep)
        sig = build_beat_signals(ep)
        cands = enumerate_candidates(ep, sig, durations=(25, 35, 45, 55))
        # platform duration filter
        cands = [c for c in cands if lo - 6 <= c.duration <= hi + 6]
        all_cands.extend(cands)

    apply_archetype_boost(all_cands, audience)

    # pre-filter to most promising pool to keep MMR cheap
    for c in all_cands:
        c.features["base_score"] = round(base_score(c.features, weights), 4)
    all_cands.sort(key=lambda c: c.features["base_score"], reverse=True)
    all_cands = all_cands[:max_pool]

    # blend feedback-learned taste
    votes = load_votes() if use_feedback else []
    fb = feedback_probabilities(all_cands, votes) if votes else {}
    alpha = feedback_blend_weight(len(votes))
    for c in all_cands:
        base = c.features["base_score"]
        fbv = fb.get(id(c))
        if fbv is not None:
            total = (1 - alpha) * base + alpha * fbv
            c.features["feedback_taste"] = round(fbv, 4)
        else:
            total = base
            c.features["feedback_taste"] = None
        c.features["total_score"] = round(total, 4)
        c.features["_weights"] = weights

    kept = non_max_suppression(all_cands, overlap_thresh=0.45)
    # Rank beyond what we need, then keep only STRICT-valid promos (validation rejects, never silently accepts).
    mmr_pool = mmr_rerank(kept, top_k=min(len(kept), max(top_k * 4, 60)), lam=diversity_lambda)
    episodes_by_num = {e.episode_number: e for e in episodes}

    promos: list[RankedPromo] = []
    rejected: list[dict] = []
    for c in mmr_pool:
        if len(promos) >= top_k:
            break
        promo = RankedPromo(
            promo_id=_promo_id(c.episode, c.start, c.end),
            episode=c.episode,
            segments=[{"episode": c.episode, "start": c.start, "end": c.end}],
            duration_s=round(c.duration, 2),
            hook_text=c.hook_text,
            hook_time_s=c.hook_time,
            hook_type=c.hook_type,
            cliff_text=c.cliff_text,
            cliff_time_s=c.cliff_time,
            why_it_works=why_it_works(c, audience),
            scores={**{k: v for k, v in c.features.items() if k not in ("_weights",)},
                    "feedback_n_votes": len(votes)},
            weights=weights,
            rank=0,
        )
        res = validate_promo(promo, episodes_by_num)
        if res.ok:
            promo.rank = len(promos) + 1
            promos.append(promo)
        else:
            rejected.append({"promo_id": promo.promo_id, "episode": c.episode,
                             "start": c.start, "end": c.end,
                             "duration_s": round(c.duration, 2), "errors": res.errors})

    if collect_stats is not None:
        collect_stats["rejected"] = rejected
        collect_stats["considered"] = len(mmr_pool)
    return promos


def stitch_episode_promo(ep: Episode):
    """
    Build a validated stitched promo from one episode: ~9s strongest hook moment,
    ~8s peak-intensity confrontation and the ~9s scripted cliffhanger — the
    'joined moments' format the brief allows.

    Guarantees (validated before returning, else None):
      • chronological, non-overlapping segments; hook within the first 3s
      • total duration within [20, 60] s inclusive
      • ends on the scripted cliffhanger / cut-to-black
    """
    from .features import build_beat_signals
    from .validation import validate_promo

    sig = build_beat_signals(ep)
    beats = ep.beats
    if len(beats) < 9 or ep.duration < 46:
        return None

    # segment 3 — 9s ending at the scripted cliffhanger marker (or episode end)
    cliff_idx = None
    for i in range(len(beats) - 1, -1, -1):
        t = beats[i].text.lower()
        if "cliffhanger" in t or "cut to black" in t:
            cliff_idx = i
            break
    if cliff_idx is None:
        cliff_idx = len(beats) - 1
    s3_end = beats[cliff_idx].end
    s3_start = max(0.0, s3_end - 9.0)

    # segment 1 — 9s from the strongest hook beat (must end before s3)
    hook_scores = sig.hook_sem.max(axis=1) + 0.8 * sig.intensity if sig.hook_sem.size else sig.intensity
    hook_idx = int(np.argmax(hook_scores))
    s1_start = beats[hook_idx].start
    s1_end = min(s1_start + 9.0, s3_start)
    if s1_end - s1_start < 6.0:  # hook lands too deep in the episode — skip stitch
        return None

    # segment 2 — best mean-intensity 8s window inside [0, s3_start], not overlapping s1
    best_val, best_t = -1.0, None
    t = 0.0
    while t + 8.0 <= s3_start:
        overlaps_s1 = not (t + 8.0 <= s1_start or t >= s1_end)
        if overlaps_s1:
            t += 1.0
            continue
        idxs = [i for i, b in enumerate(beats) if b.start < t + 8.0 and b.end > t]
        if idxs:
            val = float(sig.intensity[idxs].mean())
            if val > best_val:
                best_val, best_t = val, t
        t += 1.0

    segments = []
    segments.append({"episode": ep.episode_number, "start": round(s1_start, 2), "end": round(s1_end, 2)})
    if best_t is not None and best_t + 8.0 <= s3_start + 0.01:
        segments.append({"episode": ep.episode_number, "start": round(best_t, 2), "end": round(best_t + 8.0, 2)})
    if s3_end - s3_start >= 4.0:
        segments.append({"episode": ep.episode_number, "start": round(s3_start, 2), "end": round(s3_end, 2)})

    # chronological order, drop overlapping / too-small pieces
    segments.sort(key=lambda s: s["start"])
    cleaned = []
    last_end = -1.0
    for s in segments:
        if s["start"] < last_end + 0.25 or s["end"] - s["start"] < 3.0:
            continue
        cleaned.append(s)
        last_end = s["end"]

    total = sum(s["end"] - s["start"] for s in cleaned)

    promo_dict = {
        "hook_text": beats[hook_idx].text[:220],
        "cliffhanger_text": beats[cliff_idx].text[:220],
        "scores": {"stitched": 1.0, "total_score": round(0.65 + 0.35 * min(1.0, total / 26.0), 3),
                   "cliffhanger": 1.0},
        "segments": cleaned,
        "hook_time_s": cleaned[0]["start"] if cleaned else 0.0,
    }
    res = validate_promo(promo_dict, {ep.episode_number: ep})
    if not res.ok:
        return None

    return RankedPromo(
        promo_id=_promo_id(ep.episode_number, cleaned[0]["start"], cleaned[-1]["end"], cleaned),
        episode=ep.episode_number,
        segments=cleaned,
        duration_s=round(total, 2),
        hook_text=beats[hook_idx].text[:220],
        hook_time_s=round(cleaned[0]["start"], 2),   # first cut IS the hook moment
        hook_type="Stitched highlight reel",
        cliff_text=beats[cliff_idx].text[:220],
        cliff_time_s=round(s3_end, 2),
        why_it_works="Stitched promo: opens on the episode's strongest hook, cuts to its peak confrontation, and lands on the scripted cliffhanger.",
        scores=promo_dict["scores"],
        weights={},
        rank=0,
    )


__all__ = ["analyze_episodes", "stitch_episode_promo", "RankedPromo", "__version__"]
