"""LLM layer tests — structured output parsing, validation, graceful failure. No network."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

import src.llm as llm

PROMO = {
    "segments": [{"episode": 1, "start": 10.0, "end": 40.0}],
    "hook_text": "You are a terrible liar.",
    "cliffhanger_text": "He freezes — 'What are you—'",
    "scores": {"hook": 0.8, "cliffhanger": 0.9, "premise_alignment": 0.6, "dialogue_ratio": 0.7},
}


def test_parse_and_validate_happy_path():
    raw = '{"hook_strength": 9, "cliffhanger_quality": 10, "standalone_comprehensibility": 7, ' \
          '"narrative_coherence": 8, "verdict": "post", "reason": "Instant stop-power cold open."}'
    out = llm._parse_and_validate(raw)
    assert out["verdict"] == "post"
    assert out["scores"]["hook_strength"] == 9.0
    assert set(out["scores"]) == set(llm.DIMENSIONS)


def test_parse_rejects_out_of_range():
    raw = '{"hook_strength": 99, "cliffhanger_quality": 5, "standalone_comprehensibility": 5, ' \
          '"narrative_coherence": 5, "verdict": "post", "reason": "too high"}'
    with pytest.raises(ValueError):
        llm._parse_and_validate(raw)


def test_parse_rejects_missing_dimension():
    raw = '{"hook_strength": 5, "verdict": "post", "reason": "missing keys"}'
    with pytest.raises(ValueError):
        llm._parse_and_validate(raw)


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        llm._parse_and_validate("the promo is great, trust me")


def test_unconfigured_llm_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "none")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = llm.evaluate_promo(PROMO)
    assert out["ok"] is True
    assert out["evaluator"] == "heuristic-fallback"
    assert 0 <= min(out["scores"].values()) and max(out["scores"].values()) <= 10
    assert out["verdict"] in llm.VERDICTS


def test_llm_success_path(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(llm, "_call_llm", lambda prompt:
        '{"hook_strength": 8, "cliffhanger_quality": 9, "standalone_comprehensibility": 6, '
        '"narrative_coherence": 7, "verdict": "maybe", "reason": "Strong cold open; needs the Duchess named."}')
    out = llm.evaluate_promo(PROMO)
    assert out["ok"] is True and out["evaluator"] == "huggingface"
    assert out["scores"]["cliffhanger_quality"] == 9.0 and out["verdict"] == "maybe"


def test_llm_api_failure_returns_error_not_crash(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    def boom(prompt):
        raise RuntimeError("HTTP 503 from provider")

    monkeypatch.setattr(llm, "_call_llm", boom)
    out = llm.evaluate_promo(PROMO)
    assert out["ok"] is False and out["evaluator"] == "error"
    assert "RuntimeError" in out["error"] and out["scores"] is None


def test_llm_malformed_json_returns_error_not_crash(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_call_llm", lambda prompt: "not json at all")
    out = llm.evaluate_promo(PROMO)
    assert out["ok"] is False and out["evaluator"] == "error"


def test_llm_invalid_verdict_is_derived_not_crash(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_call_llm", lambda prompt:
        '{"hook_strength": 9, "cliffhanger_quality": 9, "standalone_comprehensibility": 9, '
        '"narrative_coherence": 9, "verdict": "yeet", "reason": "scores speak for themselves"}')
    out = llm.evaluate_promo(PROMO)
    assert out["ok"] is True and out["verdict"] == "post"
