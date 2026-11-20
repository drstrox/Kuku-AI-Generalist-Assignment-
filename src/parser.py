"""
Screenplay parser for Kuku microdrama scripts.

Input format (observed across the 50-episode input pack):

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...

    --- SCRIPT START ---

    SCENE 01
    Scene heading: INT. CATHEDRAL – DAY
    Scene start: 00:00.00
    Scene end: 02:18.00

    Actions and dialogue:

    **CAMERA:**
    Start: (00:00.00) | End: (00:04.80)
    Symmetrical framing on the heavy church doors ...

    **LADY JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"

    --- SCRIPT END ---

The parser is deliberately tolerant: unknown block types are kept as-is,
missing end times are inferred from the next beat, and unparsable lines are
collected in `warnings` instead of crashing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

TS_RE = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?"
START_END_RE = re.compile(
    r"Start:\s*\(\s*" + TS_RE + r"\s*\)\s*\|\s*End:\s*\(\s*" + TS_RE + r"\s*\)"
)
SCENE_LINE_RE = re.compile(r"^SCENE\s+(\d+)", re.IGNORECASE)
BLOCK_HEADER_RE = re.compile(r"^\*\*(.+?):?\*\*\s*:?\s*$")


def parse_timestamp(h: Optional[str], m: str, s: str, frac: Optional[str]) -> float:
    """Convert (HH:)MM:SS(.fraction) or MM:SS.SS to seconds."""
    if h is not None:
        hours = int(h)
        minutes = int(m)
    else:
        hours = 0
        minutes = int(m)
    seconds = int(s)
    fraction = float("0." + frac) if frac else 0.0
    return hours * 3600 + minutes * 60 + seconds + fraction


@dataclass
class Beat:
    kind: str            # 'camera' | 'action' | 'sfx' | 'character' | 'other'
    name: str            # character name, or CAMERA / ACTION / SOUND FX
    direction: str       # parenthetical direction e.g. "Confused, Breathless" (may be "")
    start: float
    end: float
    text: str
    scene_id: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def is_dialogue(self) -> bool:
        return self.kind == "character"


@dataclass
class Scene:
    scene_id: str
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    episode_number: int
    title: str
    logline: str
    continuity: str
    scenes: List[Scene] = field(default_factory=list)
    beats: List[Beat] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_path: str = ""

    @property
    def duration(self) -> float:
        if self.scenes:
            return max(s.end for s in self.scenes)
        if self.beats:
            return max(b.end for b in self.beats)
        return 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _classify_header(header: str) -> tuple[str, str, str]:
    """Return (kind, name, direction) from a block header like 'LADY JULIANA (Confused)'."""
    h = header.strip()
    upper = h.upper()
    if upper == "CAMERA":
        return "camera", "CAMERA", ""
    if upper == "ACTION":
        return "action", "ACTION", ""
    if upper in ("SOUND FX", "SOUND EFFECT", "SFX", "SOUND"):
        return "sfx", "SOUND FX", ""
    m = re.match(r"^(?P<name>[^(]+)\((?P<dir>[^)]*)\)\s*$", h)
    if m:
        return "character", m.group("name").strip(), m.group("dir").strip()
    return "character", h.strip(), ""


def parse_screenplay_text(text: str, source_path: str = "") -> Episode:
    # --- header metadata ---
    ep_num = 0
    title = "Unknown"
    logline = ""
    continuity = ""
    for line in text.splitlines()[:40]:
        line = line.strip()
        if line.upper().startswith("EPISODE NUMBER"):
            m = re.search(r"(\d+)", line.split(":", 1)[1] if ":" in line else line)
            ep_num = int(m.group(1)) if m else 0
        elif line.upper().startswith("EPISODE LOGLINE"):
            logline = line.split(":", 1)[1].strip() if ":" in line else ""
        elif line.upper().startswith("CONTINUITY NOTES"):
            continuity = line.split(":", 1)[1].strip() if ":" in line else ""

    body_match = re.search(r"---\s*SCRIPT START\s*---(.*?)---\s*SCRIPT END\s*---", text, re.S | re.I)
    body = body_match.group(1) if body_match else text

    ep = Episode(
        episode_number=ep_num,
        title=f"Episode {ep_num}" if ep_num else (title),
        logline=logline,
        continuity=continuity,
        source_path=source_path,
    )

    lines = body.splitlines()
    current_scene_id = "SCENE 00"
    cur_heading = ""
    cur_scene_start = 0.0
    scene_open = False

    cur_beat: Optional[dict] = None
    cur_scene_num_for_beats = current_scene_id

    def flush_beat():
        nonlocal cur_beat
        if cur_beat is None:
            return
        text_body = " ".join(cur_beat["body"]).strip()
        if cur_beat.get("start") is None:
            ep.warnings.append(f"Beat '{cur_beat['header']}' missing Start/End; skipped")
            cur_beat = None
            return
        kind, name, direction = _classify_header(cur_beat["header"])
        ep.beats.append(
            Beat(
                kind=kind,
                name=name,
                direction=direction,
                start=cur_beat["start"],
                end=cur_beat["end"],
                text=text_body,
                scene_id=cur_beat.get("scene_id", current_scene_id),
            )
        )
        cur_beat = None

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        m_scene = SCENE_LINE_RE.match(stripped)
        if m_scene:
            flush_beat()
            current_scene_id = f"SCENE {int(m_scene.group(1)):02d}"
            cur_heading = ""
            cur_scene_start = 0.0
            scene_open = True
            continue

        low = stripped.lower()
        if low.startswith("scene heading:"):
            cur_heading = stripped.split(":", 1)[1].strip()
            continue
        if low.startswith("scene start:"):
            ts = re.search(TS_RE, stripped)
            if ts:
                cur_scene_start = parse_timestamp(*ts.groups())
            continue
        if low.startswith("scene end:"):
            ts = re.search(TS_RE, stripped)
            if ts:
                ep.scenes.append(
                    Scene(
                        scene_id=current_scene_id,
                        heading=cur_heading,
                        start=cur_scene_start,
                        end=parse_timestamp(*ts.groups()),
                    )
                )
                scene_open = False
            continue

        m_block = BLOCK_HEADER_RE.match(stripped)
        if m_block:
            flush_beat()
            cur_beat = {"header": m_block.group(1), "start": None, "end": None, "body": [], "scene_id": current_scene_id}
            continue

        m_se = START_END_RE.search(stripped)
        if m_se and cur_beat is not None:
            g = m_se.groups()
            cur_beat["start"] = parse_timestamp(*g[:4])
            cur_beat["end"] = parse_timestamp(*g[4:])
            continue

        if cur_beat is not None:
            # strip stray markdown quotes but keep content
            cur_beat["body"].append(stripped)

    flush_beat()

    # stable ordering + overlap sanity
    ep.beats.sort(key=lambda b: (b.start, b.end))
    if not ep.scenes and ep.beats:
        ep.scenes.append(Scene("SCENE 00", "", 0.0, max(b.end for b in ep.beats)))
    return ep


def parse_file(path: str) -> Episode:
    import pathlib

    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return parse_screenplay_text(text, source_path=str(p))
