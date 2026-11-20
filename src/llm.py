"""
Swappable LLM layer — genuine contextual promo evaluation (OFF by default).

`evaluate_promo()` sends the candidate's verbatim script excerpt (timecoded) plus
the episode logline to an instruction-tuned LLM and asks for a structured JSON
judgment of four things a heuristic cannot see:

  hook_strength               — does the first ~3 s stop a scrolling viewer? (0–10)
  cliffhanger_quality         — is the cut unresolved, i.e. "must watch next"?  (0–10)
  standalone_comprehensibility— can a first-time viewer follow with no context? (0–10)
  narrative_coherence         — do the chosen moments hang together as a cut?   (0–10)

plus a verdict (post / maybe / reject) and a ≤30-word rationale.

Graceful degradation, never a crash:
  • provider not configured  -> heuristic-fallback (derived from the ML feature
                               scores, clearly labelled; no network call)
  • API/unreachable/bad JSON -> ok=False, scores=None, evaluator="error"

Nothing about the ML ranking depends on this module; it only *adds* a second
opinion when the user opts in and supplies a key. Swap models/providers purely
through environment variables:

  KUKU_LLM_PROVIDER = "huggingface" | "gemini" | "none"   (default "none")
  KUKU_LLM_MODEL    = e.g. "meta-llama/Llama-3.1-8B-Instruct" / "gemini-2.0-flash"
  HF_TOKEN          / GEMINI_API_KEY
  KUKU_LLM_TIMEOUT  = seconds (default 45)
"""
from __future__ import annotations

import json
import os
import re

import requests

PROVIDER = os.environ.get("KUKU_LLM_PROVIDER", "none").lower()
MODEL = os.environ.get("KUKU_LLM_MODEL", "")
TIMEOUT = float(os.environ.get("KUKU_LLM_TIMEOUT", "45"))

DIMENSIONS = ["hook_strength", "cliffhanger_quality", "standalone_comprehensibility", "narrative_coherence"]
VERDICTS = {"post", "maybe", "reject"}


def llm_available() -> tuple[bool, str]:
    if PROVIDER == "huggingface":
        return (bool(os.environ.get("HF_TOKEN")), f"huggingface · {MODEL or 'meta-llama/Llama-3.1-8B-Instruct'}")
    if PROVIDER == "gemini":
        return (bool(os.environ.get("GEMINI_API_KEY")), f"gemini · {MODEL or 'gemini-2.0-flash'}")
    return (False, "none (set KUKU_LLM_PROVIDER + key to enable)")


# ------------------------------------------------------------------ helpers
def _transcript_from_beats(segments: list[dict], beats, logline: str) -> str:
    lines = []
    for seg in segments:
        ep_label = f"EPISODE {seg.get('episode', '?')}"
        lines.append(f"— SEGMENT {ep_label} {seg['start']:.1f}s→{seg['end']:.1f}s —")
        for b in beats:
            if b.end > seg["start"] and b.start < seg["end"]:
                speaker = b.name if getattr(b, "is_dialogue", False) else (b.kind or "action").upper()
                text = re.sub(r"\s+", " ", (b.text or "")).strip()
                if text:
                    lines.append(f"[{b.start:6.1f}s] {speaker}: {text}")
    transcript = "\n".join(lines)
    if len(transcript) > 3000:
        transcript = transcript[:3000] + "\n…[truncated]"
    return transcript


def _heuristic_fallback(promo: dict) -> dict:
    s = promo.get("scores", {}) or {}
    def h(key, default=0.5):
        try:
            return float(s.get(key, default))
        except (TypeError, ValueError):
            return default
    scores = {
        "hook_strength": round(h("hook") * 10, 1),
        "cliffhanger_quality": round(h("cliffhanger") * 10, 1),
        "standalone_comprehensibility": round(h("premise_alignment") * 10, 1),
        "narrative_coherence": round((4 + 6 * h("dialogue_ratio")), 1),
    }
    mean = sum(scores.values()) / len(scores)
    return {
        "ok": True,
        "evaluator": "heuristic-fallback",
        "warning": "LLM not configured — scores derived from ML feature signals, not genuine contextual judgment.",
        "scores": scores,
        "verdict": "post" if mean >= 7 else ("maybe" if mean >= 5 else "reject"),
        "reason": "Heuristic estimate from ML features (hook/cliffhanger/premise/dialogue balance).",
    }


def _parse_and_validate(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object found in LLM response")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON is not an object")
    scores = {}
    for d in DIMENSIONS:
        if d not in data:
            raise ValueError(f"missing dimension {d!r}")
        v = float(data[d])
        if not (0.0 <= v <= 10.0):
            raise ValueError(f"dimension {d!r} out of range: {v}")
        scores[d] = round(v, 1)
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        mean = sum(scores.values()) / len(scores)
        verdict = "post" if mean >= 7 else ("maybe" if mean >= 5 else "reject")
    reason = str(data.get("reason", "")).strip()[:280]
    if not reason:
        raise ValueError("empty reason")
    return {"scores": scores, "verdict": verdict, "reason": reason}


def _call_llm(prompt: str) -> str:
    if PROVIDER == "huggingface":
        model = MODEL or "meta-llama/Llama-3.1-8B-Instruct"
        r = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 220, "temperature": 0.2},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if PROVIDER == "gemini":
        model = MODEL or "gemini-2.0-flash"
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={os.environ['GEMINI_API_KEY']}",
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 220, "temperature": 0.2}},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise ValueError(f"Unknown KUKU_LLM_PROVIDER: {PROVIDER!r}")


# ------------------------------------------------------------------ main API
def evaluate_promo(promo: dict, beats=None, logline: str = "") -> dict:
    """
    Return a dict:
      ok: bool
      evaluator: "huggingface"/"gemini"/"heuristic-fallback"/"error"
      scores: {hook_strength, cliffhanger_quality, standalone_comprehensibility,
               narrative_coherence} each 0–10  (None when evaluator == "error")
      verdict: "post" | "maybe" | "reject"
      reason: ≤280-char rationale
      warning/error when applicable
    Never raises.
    """
    ok, _ = llm_available()
    if not ok:
        return _heuristic_fallback(promo)

    segments = promo.get("segments", [])
    transcript = _transcript_from_beats(segments, beats, logline) if beats is not None else \
        f"Hook: {promo.get('hook_text', '')}\nEnd: {promo.get('cliffhanger_text', '')}"
    prompt = f"""You are a senior short-form video editor at a microdrama app (Kuku TV).
A viewer has NEVER seen this show. Judge this promo cut from ONLY the script excerpt below.

Show logline: {logline or "unknown"}
Promo segments (chronological, timecoded):
{transcript}

Score 0–10 (integers):
- hook_strength: stopping power of the FIRST ~3 seconds
- cliffhanger_quality: does the ending leave an open question the viewer must resolve?
- standalone_comprehensibility: clarity for a first-time viewer with zero context
- narrative_coherence: do the chosen beats work together as one cut?
Then one verdict (post | maybe | reject) and a reason of at most 30 words.

Respond with ONLY JSON: {{"hook_strength": 0, "cliffhanger_quality": 0, "standalone_comprehensibility": 0, "narrative_coherence": 0, "verdict": "post", "reason": "..."}}"""

    try:
        raw = _call_llm(prompt)
        parsed = _parse_and_validate(raw)
        return {"ok": True, "evaluator": PROVIDER, **parsed}
    except Exception as exc:
        return {"ok": False, "evaluator": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "scores": None, "verdict": "maybe",
                "reason": "LLM evaluation failed — ML ranking unaffected; see error."}
