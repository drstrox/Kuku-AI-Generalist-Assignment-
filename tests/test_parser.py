"""Sanity tests for the screenplay parser and scoring invariants."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.parser import parse_screenplay_text, parse_timestamp

SAMPLE = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: Follows something.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. TEST – DAY
Scene start: 00:00.00
Scene end: 00:30.00

**ACTION:**
Start: (00:00.00) | End: (00:02.00)
A door slams.

**ADA (Angry):**
Start: (00:02.20) | End: (00:04.00)
"Get out."

**SOUND FX:**
Start: (00:04.20) | End: (00:06.00)
Low drone swells to an ominous cliffhanger.

--- SCRIPT END ---
"""


def test_parse_timestamp():
    assert parse_timestamp(None, "02", "18", "00") == 138.0
    assert parse_timestamp(None, "00", "04", "80") == 4.8
    assert parse_timestamp("01", "02", "03", "50") == 3723.5


def test_parse_structure():
    ep = parse_screenplay_text(SAMPLE)
    assert ep.episode_number == 7
    assert ep.logline == "A test logline."
    assert len(ep.scenes) == 1 and ep.scenes[0].end == 30.0
    assert len(ep.beats) == 3
    kinds = [b.kind for b in ep.beats]
    assert kinds == ["action", "character", "sfx"]
    ada = ep.beats[1]
    assert ada.name == "ADA" and ada.direction == "Angry" and ada.is_dialogue
    assert abs(ada.start - 2.2) < 1e-6


def test_timestamps_monotonic_after_sort():
    ep = parse_screenplay_text(SAMPLE)
    starts = [b.start for b in ep.beats]
    assert starts == sorted(starts)


def test_empty_body_does_not_crash():
    ep = parse_screenplay_text("EPISODE NUMBER: Episode 1\nEPISODE LOGLINE: x\nCONTINUITY NOTES: y\n")
    assert ep.beats == [] and ep.scenes == []
