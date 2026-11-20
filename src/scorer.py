"""
Interpretable scoring, audience presets, MMR re-ranking and explanations.

score = Σ w_i · feature_i            (weights below; user presets adjust them)
final = (1 - α) · score + α · P_feedback   (α grows with #votes, max 0.5)

Everything is transparent: the breakdown panel in the UI shows the exact
contribution of each feature, which is what makes the ranking defensible in
an interview.
"""
from __future__ import annotations

import numpy as np

BASE_WEIGHTS = {
    "hook": 0.30,
    "cliffhanger": 0.28,
    "intensity": 0.12,
    "emotional_swing": 0.08,
    "premise_alignment": 0.08,
    "pace": 0.06,
    "dialogue_ratio": 0.04,
    "episode_prior": 0.04,
}

PLATFORM_PREFS = {
    "TikTok / Reels (fast, 20–35s)": {"duration_pref": (22, 35), "pace_boost": 0.04},
    "YouTube (45–60s, more context)": {"duration_pref": (40, 60), "pace_boost": -0.02},
    "Any platform": {"duration_pref": (20, 60), "pace_boost": 0.0},
}

AUDIENCE_PRESETS = {
    "General audience (balanced)": {
        "weights": {},
        "archetype_boost": {},
        "note": "Balanced across betrayal, romance, danger and status twists.",
    },
    "Romance lovers": {
        "weights": {"hook": 0.28, "cliffhanger": 0.24, "intensity": 0.10, "emotional_swing": 0.12, "premise_alignment": 0.10},
        "archetype_boost": {"Confession / forbidden love": 1.35, "Betrayal reveal": 1.15, "Power reversal": 1.1},
        "note": "Leans into confessions, reunions and heartbreak with sweeping stakes.",
    },
    "Revenge & justice": {
        "weights": {"hook": 0.32, "cliffhanger": 0.28, "intensity": 0.13, "emotional_swing": 0.07},
        "archetype_boost": {"Public humiliation": 1.3, "Slap / physical shock": 1.25, "Power reversal": 1.2, "Shocking line": 1.15},
        "note": "Leans into slaps, public takedowns and satisfying downfalls.",
    },
    "Suspense & thriller": {
        "weights": {"hook": 0.28, "cliffhanger": 0.33, "intensity": 0.12, "emotional_swing": 0.09},
        "archetype_boost": {"Secret exposed": 1.35, "Threat / mortal danger": 1.3, "Betrayal reveal": 1.15},
        "note": "Leans into poisons, secrets, threats and unresolved danger.",
    },
    "Family & betrayal drama": {
        "weights": {"hook": 0.30, "cliffhanger": 0.27, "premise_alignment": 0.10, "emotional_swing": 0.09},
        "archetype_boost": {"Betrayal reveal": 1.35, "Secret exposed": 1.2, "Public humiliation": 1.15},
        "note": "Leans into backstabbing, family lies and inheritance wars.",
    },
}


def resolved_weights(preset_name: str, platform_name: str) -> dict:
    w = dict(BASE_WEIGHTS)
    for k, v in AUDIENCE_PRESETS.get(preset_name, {}).get("weights", {}).items():
        w[k] = v
    w["pace"] = max(0.02, w["pace"] + PLATFORM_PREFS.get(platform_name, {}).get("pace_boost", 0.0))
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def base_score(candidate_features: dict, weights: dict) -> float:
    return float(sum(weights[k] * candidate_features.get(k, 0.0) for k in weights))


def apply_archetype_boost(candidates, preset_name: str):
    boosts = AUDIENCE_PRESETS.get(preset_name, {}).get("archetype_boost", {})
    if not boosts:
        return candidates
    for c in candidates:
        b = boosts.get(c.hook_type, 1.0)
        if b != 1.0:
            c.features["hook"] = round(min(1.0, c.features["hook"] * b), 4)
    return candidates


def non_max_suppression(candidates, overlap_thresh=0.45):
    """Keep highest-scoring candidates; drop ones overlapping a kept one."""
    kept = []
    for c in sorted(candidates, key=lambda c: c.features["total_score"], reverse=True):
        ok = True
        for k in kept:
            if k.episode != c.episode:
                continue
            inter = max(0.0, min(c.end, k.end) - max(c.start, k.start))
            union = min(c.duration, k.duration)
            if union > 0 and inter / union > overlap_thresh:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def mmr_rerank(candidates, top_k: int, lam: float = 0.72):
    """Maximal Marginal Relevance — avoids a feed full of near-identical cuts."""
    if not candidates:
        return []
    pool = list(candidates)          # indices shift as we pop; track embeddings alongside
    embs = [c.window_embedding for c in pool]
    selected = []
    selected_embs = []

    def sim_to_selected(e):
        if e is None:
            return 0.0
        sims = [float(np.dot(e, se)) for se in selected_embs if se is not None]
        return max(sims) if sims else 0.0

    while pool and len(selected) < top_k:
        best_idx, best_val = 0, -1e9
        for i, c in enumerate(pool):
            relevance = c.features["total_score"]
            diversity_penalty = sim_to_selected(embs[i]) if selected else 0.0
            val = lam * relevance - (1 - lam) * diversity_penalty
            if val > best_val:
                best_val, best_idx = val, i
        chosen = pool.pop(best_idx)
        selected.append(chosen)
        selected_embs.append(embs.pop(best_idx))
    return selected


def why_it_works(c, preset_name: str) -> str:
    f = c.features
    hook_desc = f"Opens on {c.hook_type.lower()} (\"{_trim(c.hook_text, 90)}\")" if c.hook_text else "Opens on a high-impact moment"
    cliff_desc = f"ends unresolved on \"{_trim(c.cliff_text, 80)}\"" if c.cliff_text else "ends before the answer"
    bits = []
    if f["hook"] >= 0.65:
        bits.append("scroll-stopping hook")
    elif f["hook"] >= 0.45:
        bits.append("solid opening hook")
    if f["cliffhanger"] >= 0.6:
        bits.append("hard cliffhanger cut")
    if f["intensity"] >= 0.6:
        bits.append("sustained emotional intensity")
    if f["emotional_swing"] >= 0.55:
        bits.append("sharp emotional whiplash")
    if f["premise_alignment"] >= 0.55:
        bits.append("carries the show's core premise (works without prior context)")
    if f.get("cliff_sfx", 0) > 0:
        bits.append("script explicitly scores a cliffhanger here")
    reason = ", ".join(bits[:3]) if bits else "best available balance of hook and unresolved tension"
    return f"{hook_desc} → {reason}; {cliff_desc}."


def _trim(t: str, n: int) -> str:
    t = t.replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


def score_breakdown(c) -> list[tuple[str, float, float]]:
    """(feature, feature_value, weight) rows for the UI."""
    rows = []
    mapping = [
        ("hook", "Hook strength (0–3.5s)"),
        ("cliffhanger", "Cliffhanger ending"),
        ("intensity", "Emotional intensity"),
        ("emotional_swing", "Emotional swing"),
        ("premise_alignment", "Self-contained premise"),
        ("pace", "Pace (cut density)"),
        ("dialogue_ratio", "Dialogue balance"),
        ("episode_prior", "Early-episode prior"),
    ]
    w = c.features.get("_weights", BASE_WEIGHTS)
    for key, label in mapping:
        rows.append((label, c.features.get(key, 0.0), w.get(key, 0.0)))
    return rows
