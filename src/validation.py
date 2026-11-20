"""
Strict final promo validation.

Every promo — whether produced by single-window ranking or by stitching — must
pass ALL of the following before it is shown to the editor or written to outputs:

  1. Duration: total across segments in [20.0, 60.0] seconds INCLUSIVE.
  2. Timestamps: finite, ≥ 0, start < end on every segment.
  3. Chronological order and NON-OVERLAPPING segments.
  4. Hook: the promoter's hook moment must fire within the first 3.0 s of the promo.
  5. Unresolved ending: the final beat must leave the viewer hanging —
     a scripted cliffhanger/cut-to-black marker, an unanswered question or
     interruption, a cut exactly at the episode end (which is the cliffhanger by
     construction in microdrama), or a high ML cliffhanger score with residual
     tension.

Candidates that fail are REJECTED (counted, never silently kept).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

MIN_DURATION_S = 20.0
MAX_DURATION_S = 60.0
HOOK_WINDOW_S = 3.0
CHRONO_EPS = 0.05   # tolerated float slack for segment ordering/overlap (50 ms)
EPS = 1e-6          # near-strict tolerance for duration bounds and the 3s hook window

UNRESOLVED_MARKERS = (
    "cliffhanger", "cut to black", "black.", "?", "—", "...", "…", "gasp",
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_duration_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "total_duration_s": round(self.total_duration_s, 2),
        }


def _seg_get(seg, key, default=None):
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _is_unresolved(promo: dict, episode_end: float | None) -> tuple[bool, str]:
    """Heuristics for 'the viewer is left hanging'. Returns (unresolved, evidence)."""
    cliff_text = (promo.get("cliffhanger_text") or promo.get("cliff_text") or "").lower()
    scores = promo.get("scores", {}) or {}
    cliff_score = float(scores.get("cliffhanger", 0.0) or 0.0)
    if episode_end is not None and promo.get("segments"):
        end_t = max(float(_seg_get(s, "end", 0.0)) for s in promo["segments"])
        if abs(end_t - episode_end) <= 0.5:
            return True, f"cuts at episode end ({end_t:.1f}s — the scripted cut-to-black)"
    for m in UNRESOLVED_MARKERS:
        if m.lower() in cliff_text:
            return True, f"unresolved marker {m!r} in final beat"
    if cliff_score >= 0.55:
        return True, f"ML cliffhanger score {cliff_score:.2f} ≥ 0.55 (unresolved tension)"
    return False, "no unresolved-ending evidence (no marker, no episode-end cut, cliffhanger < 0.55)"


def validate_promo(promo, episodes_by_num: dict | None = None) -> ValidationResult:
    """
    promo: RankedPromo, or dict shaped like RankedPromo.to_dict().
    episodes_by_num: optional {episode_number: Episode} for exact episode-end check.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if hasattr(promo, "to_dict"):
        promo = promo.to_dict()

    segments = promo.get("segments") or []
    if not segments:
        return ValidationResult(False, ["no segments"], [], 0.0)

    total = 0.0
    prev_end = -math.inf
    parsed = []
    for i, seg in enumerate(segments):
        ep = _seg_get(seg, "episode")
        try:
            start, end = float(_seg_get(seg, "start")), float(_seg_get(seg, "end"))
        except (TypeError, ValueError):
            errors.append(f"segment {i}: non-numeric timestamps")
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            errors.append(f"segment {i}: non-finite timestamps")
        if start < 0:
            errors.append(f"segment {i}: negative start ({start})")
        if end <= start:
            errors.append(f"segment {i}: end ({end}) must be > start ({start})")
        if ep is None:
            errors.append(f"segment {i}: missing episode number")
        if start < prev_end - CHRONO_EPS:
            errors.append(f"segment {i}: not chronological / overlaps previous (start {start} < prev end {prev_end})")
        prev_end = max(prev_end, end)
        total += max(0.0, end - start)
        parsed.append((ep, start, end))

    # 2/3: duration bounds (inclusive)
    if not (MIN_DURATION_S - EPS <= total <= MAX_DURATION_S + EPS):
        errors.append(f"total duration {total:.2f}s outside inclusive [{MIN_DURATION_S:.0f}, {MAX_DURATION_S:.0f}]s")
    for i, (ep, start, end) in enumerate(parsed):
        if end - start < 3.0 - EPS:
            warnings.append(f"segment {i} (ep {ep}) is very short: {end - start:.1f}s")

    # 4: hook within the first 3 seconds of the promo
    first_start = min((s for _, s, _ in parsed), default=None)
    hook_t = promo.get("hook_time_s", promo.get("hook_time"))
    if first_start is not None:
        if hook_t is None:
            errors.append("missing hook_time_s")
        else:
            hook_t = float(hook_t)
            if hook_t < first_start - EPS:
                errors.append(f"hook ({hook_t:.2f}s) fires BEFORE the promo starts ({first_start:.2f}s)")
            elif hook_t > first_start + HOOK_WINDOW_S + EPS:
                errors.append(f"hook ({hook_t:.2f}s) fires after the first {HOOK_WINDOW_S:.0f}s window (promo starts {first_start:.2f}s)")

    # 5: unresolved ending
    last_ep = parsed[-1][0] if parsed else None
    ep_end = None
    if episodes_by_num is not None and last_ep is not None:
        ep_obj = episodes_by_num.get(last_ep)
        ep_end = ep_obj.duration if ep_obj is not None else None
    unresolved, evidence = _is_unresolved(promo, ep_end)
    if not unresolved:
        errors.append(f"ending is not clearly unresolved — {evidence}")
    else:
        warnings.append(f"unresolved ending evidence: {evidence}")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, total_duration_s=total)


def filter_valid_promos(promos: list[dict], episodes_by_num: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Split promos into (valid, rejected[{promo, errors}])."""
    valid, rejected = [], []
    for p in promos:
        res = validate_promo(p, episodes_by_num)
        if res.ok:
            valid.append(p)
        else:
            rejected.append({"promo_id": p.get("promo_id"), "episode": p.get("episode"),
                             "duration_s": round(res.total_duration_s, 2), "errors": res.errors})
    return valid, rejected
