"""End-to-end tests on the bundled ORIGINAL demo screenplay (no Kuku content)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from src.parser import parse_file, parse_screenplay_text
from src.pipeline import analyze_episodes, stitch_episode_promo
from src.validation import validate_promo

DEMO = pathlib.Path(__file__).resolve().parent.parent / "data" / "demo_episode.txt"

LONG_EPISODE = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A poisoned heiress finds allies in a dungeon while her enemies close in.
CONTINUITY NOTES: Original synthetic test script.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CASTLE DUNGEON - NIGHT
Scene start: 00:00.00
Scene end: 01:30.00

**ACTION:**
Start: (00:00.00) | End: (00:03.00)
A single torch drips wax. Rats scatter in the straw.

**DOWAGER:**
Start: (00:03.20) | End: (00:06.50)
"They will come for the ring before dawn."

**JULIANA (Trembling):**
Start: (00:07.00) | End: (00:10.00)
"Then I give it to you now. Before they take everything."

**CAMERA:**
Start: (00:10.00) | End: (00:13.00)
Tight close-up on the blood-red ruby as fingers close over it.

**SOUND FX:**
Start: (00:13.20) | End: (00:16.00)
Distant iron doors groaning open somewhere above.

**RUFUS (Low, Dangerous):**
Start: (00:16.20) | End: (00:19.50)
"You heard her. Before dawn."

**ACTION:**
Start: (00:19.80) | End: (00:23.00)
A shadow fills the cell door. Steel rings on stone.

**JULIANA:**
Start: (00:23.20) | End: (00:26.00)
"Rufus. You were supposed to be dead."

**RUFUS:**
Start: (00:26.30) | End: (00:29.00)
"Dead men keep no grudges. I keep mine."

**SOUND FX:**
Start: (00:29.20) | End: (00:33.00)
A long blade whispering out of a scabbard.

**JULIANA:**
Start: (00:33.20) | End: (00:36.50)
"If you kill me, the ring dies with me."

**RUFUS:**
Start: (00:37.00) | End: (00:40.00)
"Then we have an understanding."

**ACTION:**
Start: (00:40.20) | End: (00:44.00)
The torch flares. Figures pour into the corridor above.

**DOWAGER:**
Start: (00:44.20) | End: (00:47.50)
"Someone is coming. God help us."

**JULIANA:**
Start: (00:48.00) | End: (00:50.50)
"Then whatever happens — remember my mother's book."

**CAMERA:**
Start: (00:50.80) | End: (00:54.00)
Extreme close-up on Juliana's eye as keys jangle closer.

**SOUND FX:**
Start: (00:54.20) | End: (00:58.00)
A heavy key grinds into the lock — an ominous cliffhanger hold.

**ACTION:**
Start: (00:58.50) | End: (01:02.00)
The door swings inward on black darkness.

**RUFUS:**
Start: (01:02.20) | End: (01:05.00)
"Time to wake the dead."

**DOWAGER:**
Start: (01:05.20) | End: (01:08.00)
"Run, Juliana. Run!"

**ACTION:**
Start: (01:08.20) | End: (01:12.00)
Boots thunder down the stair. Cut to black.

**SOUND FX:**
Start: (01:12.20) | End: (01:30.00)
The low drone of strings rises to a gripping cliffhanger peak.

--- SCRIPT END ---
"""


@pytest.fixture(scope="module")
def demo_episode():
    return parse_file(str(DEMO))


def test_demo_parses(demo_episode):
    assert demo_episode.episode_number == 1
    assert len(demo_episode.beats) >= 10


def test_analyzed_promos_are_all_valid(demo_episode):
    eps_by_num = {demo_episode.episode_number: demo_episode}
    promos = analyze_episodes([demo_episode], "General audience (balanced)", "Any platform",
                              top_k=3, use_feedback=False)
    assert promos, "expected at least one valid promo for the demo episode"
    for p in promos:
        res = validate_promo(p.to_dict(), eps_by_num)
        assert res.ok, f"promo failed validation: {res.errors}"
        # chronological, non-overlapping segments
        segs = p.segments
        for i in range(1, len(segs)):
            assert segs[i]["start"] >= segs[i - 1]["end"]
        assert 20.0 <= p.duration_s <= 60.0


def test_stitched_promo_is_valid(demo_episode):
    ep = parse_screenplay_text(LONG_EPISODE, source_path="synthetic_long.txt")
    sp = stitch_episode_promo(ep)
    assert sp is not None, "synthetic long episode should support a stitched promo"
    res = validate_promo(sp.to_dict(), {ep.episode_number: ep})
    assert res.ok, res.errors
    assert len(sp.segments) >= 3
    starts = [s["start"] for s in sp.segments]
    assert starts == sorted(starts)


def test_analyze_rejects_nothing_silently(demo_episode):
    stats = {}
    promos = analyze_episodes([demo_episode], "General audience (balanced)", "Any platform",
                              top_k=5, use_feedback=False, collect_stats=stats)
    eps_by_num = {demo_episode.episode_number: demo_episode}
    for p in promos:
        assert validate_promo(p.to_dict(), eps_by_num).ok
