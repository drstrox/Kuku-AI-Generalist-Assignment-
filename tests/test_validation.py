"""Strict promo-validation tests — duration boundaries, hook timing, segments, unresolved endings."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from src.validation import validate_promo, filter_valid_promos


def base_promo(**kw):
    p = {
        "promo_id": "t1", "episode": 1,
        "segments": [{"episode": 1, "start": 0.0, "end": 30.0}],
        "duration_s": 30.0,
        "hook_time_s": 0.0, "cliffhanger_text": "Someone gasps — unresolved…",
        "scores": {"cliffhanger": 0.8},
    }
    p.update(kw)
    return p


# ---- duration boundaries (inclusive 20–60) --------------------------------
@pytest.mark.parametrize("dur,expected_ok", [(19.99, False), (20.0, True), (25.0, True), (60.0, True), (60.01, False)])
def test_duration_bounds(dur, expected_ok):
    p = base_promo(segments=[{"episode": 1, "start": 0.0, "end": dur}], duration_s=dur)
    res = validate_promo(p)
    assert res.ok is expected_ok, res.errors


def test_total_duration_across_multiple_segments():
    # 10 + 12 = 22s total → valid (multi-segment must sum)
    p = base_promo(segments=[{"episode": 1, "start": 0.0, "end": 10.0},
                             {"episode": 1, "start": 30.0, "end": 42.0}],
                   hook_time_s=0.0)
    assert validate_promo(p).ok
    # 9 + 10 = 19s → invalid
    p = base_promo(segments=[{"episode": 1, "start": 0.0, "end": 9.0},
                             {"episode": 1, "start": 30.0, "end": 40.0}])
    assert not validate_promo(p).ok


# ---- timestamp validity ----------------------------------------------------
def test_negative_start_rejected():
    p = base_promo(segments=[{"episode": 1, "start": -1.0, "end": 30.0}], hook_time_s=0.0)
    assert not validate_promo(p).ok


def test_end_before_start_rejected():
    p = base_promo(segments=[{"episode": 1, "start": 30.0, "end": 10.0}])
    assert not validate_promo(p).ok


def test_nan_timestamps_rejected():
    p = base_promo(segments=[{"episode": 1, "start": float("nan"), "end": 30.0}])
    assert not validate_promo(p).ok


# ---- chronological / non-overlapping ----------------------------------------
def test_overlapping_segments_rejected():
    p = base_promo(segments=[{"episode": 1, "start": 0.0, "end": 15.0},
                             {"episode": 1, "start": 14.0, "end": 45.0}],
                   hook_time_s=0.0)
    assert not validate_promo(p).ok


def test_out_of_order_segments_rejected():
    p = base_promo(segments=[{"episode": 1, "start": 30.0, "end": 45.0},
                             {"episode": 1, "start": 0.0, "end": 25.0}],
                   hook_time_s=30.0)
    assert not validate_promo(p).ok


def test_back_to_back_segments_ok():
    p = base_promo(segments=[{"episode": 1, "start": 0.0, "end": 11.0},
                             {"episode": 1, "start": 11.0, "end": 25.0}],
                   hook_time_s=0.0)
    assert validate_promo(p).ok


# ---- hook timing ------------------------------------------------------------
def test_hook_within_first_3s_ok():
    p = base_promo(segments=[{"episode": 1, "start": 10.0, "end": 40.0}], hook_time_s=13.0)
    assert validate_promo(p).ok


def test_hook_after_first_3s_rejected():
    p = base_promo(segments=[{"episode": 1, "start": 10.0, "end": 40.0}], hook_time_s=13.01)
    assert not validate_promo(p).ok


def test_hook_before_promo_start_rejected():
    p = base_promo(segments=[{"episode": 1, "start": 10.0, "end": 40.0}], hook_time_s=9.5)
    assert not validate_promo(p).ok


def test_missing_hook_time_rejected():
    p = base_promo()
    del p["hook_time_s"]
    assert not validate_promo(p).ok


# ---- unresolved ending -------------------------------------------------------
def test_unresolved_via_marker_text():
    p = base_promo(cliffhanger_text="He freezes — 'What are you—'", scores={"cliffhanger": 0.1})
    assert validate_promo(p).ok


def test_unresolved_via_ml_score_only():
    p = base_promo(cliffhanger_text="A quiet look lingers.", scores={"cliffhanger": 0.7})
    assert validate_promo(p).ok


def test_unresolved_via_episode_end(tmp_path):
    class FakeEp:
        duration = 100.0
    p = base_promo(segments=[{"episode": 1, "start": 60.0, "end": 100.0}],
                   hook_time_s=60.0, cliffhanger_text="Nothing special happens.",
                   scores={"cliffhanger": 0.1})
    assert validate_promo(p, {1: FakeEp()}).ok


def test_resolved_ending_rejected():
    class FakeEp:
        duration = 100.0
    p = base_promo(segments=[{"episode": 1, "start": 10.0, "end": 40.0}],
                   hook_time_s=10.0, cliffhanger_text="They both smile. The end.",
                   scores={"cliffhanger": 0.1})
    assert not validate_promo(p, {1: FakeEp()}).ok


def test_filter_valid_promos_splits():
    good = base_promo()
    bad = base_promo(segments=[{"episode": 1, "start": 0.0, "end": 5.0}])
    valid, rejected = filter_valid_promos([good, bad])
    assert len(valid) == 1 and len(rejected) == 1 and rejected[0]["errors"]
