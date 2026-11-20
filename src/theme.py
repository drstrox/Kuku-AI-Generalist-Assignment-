"""Shared theme: CSS injection, SVG helpers, formatting."""

CSS = """
<style>
:root{
  --bg:#0c0c12; --panel:#12121b; --panel2:#171724; --line:rgba(255,255,255,.08);
  --gold:#f2c069; --gold2:#e2725b; --violet:#8b5cf6; --ink:#e9e7f2; --dim:#9b98ad;
}
.stApp{ background: radial-gradient(1200px 600px at 70% -10%, rgba(139,92,246,.16), transparent 60%),
                radial-gradient(900px 500px at -10% 20%, rgba(242,192,105,.08), transparent 55%), var(--bg); }
header[data-testid="stHeader"]{ background: transparent; }
#MainMenu, footer{ visibility:hidden; }
.kuku-hero{ border:1px solid var(--line); border-radius:20px; padding:22px 26px; background:linear-gradient(135deg, rgba(139,92,246,.12), rgba(242,192,105,.05) 55%, transparent); margin-bottom:14px; }
.kuku-hero h1{ font-family: Georgia, 'Times New Roman', serif; font-weight:600; letter-spacing:.5px; font-size:2.05rem; margin:0; color:var(--ink);}
.kuku-hero h1 .accent{ background:linear-gradient(90deg,var(--gold),var(--gold2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.kuku-hero p{ color:var(--dim); margin:.35rem 0 0 0; font-size:.95rem;}
.chip{ display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); background:var(--panel2); color:var(--dim); border-radius:999px; padding:3px 11px; font-size:.74rem; margin-right:6px;}
.chip b{ color:var(--ink); font-weight:600;}
.chip.ok{ border-color:rgba(52,211,153,.35); color:#6ee7b7;}
.chip.warn{ border-color:rgba(251,191,36,.35); color:#fcd34d;}
.promo-card{ border:1px solid var(--line); border-radius:18px; background:var(--panel); padding:18px 20px; margin-bottom:14px; box-shadow:0 10px 30px rgba(0,0,0,.35); }
.promo-card:hover{ border-color:rgba(242,192,105,.35); }
.rank-badge{ font-family:Georgia,serif; font-size:1.35rem; color:#0c0c12; background:linear-gradient(135deg,var(--gold),var(--gold2)); border-radius:14px; width:44px;height:44px; display:flex;align-items:center;justify-content:center; font-weight:700;}
.promo-title{ font-family:Georgia,serif; font-size:1.05rem; color:var(--ink); margin:0;}
.hook-line{ border-left:3px solid var(--gold); padding:2px 10px; color:#d8d5e8; font-style:italic; margin:8px 0;}
.cliff-line{ border-left:3px solid var(--violet); padding:2px 10px; color:#c9c6de; font-style:italic; margin:8px 0;}
.meta{ color:var(--dim); font-size:.8rem; }
.score-num{ font-family:Georgia,serif; font-size:1.6rem; color:var(--gold); }
.section-label{ text-transform:uppercase; letter-spacing:.18em; font-size:.68rem; color:var(--dim); margin:10px 0 4px 0;}
.stitched{ border:1px dashed rgba(242,192,105,.5); border-radius:18px; padding:14px 18px; background:rgba(242,192,105,.05); margin-bottom:10px;}
.footer-note{ color:var(--dim); font-size:.78rem; border-top:1px solid var(--line); padding-top:10px; margin-top:18px;}
div[data-baseweb="tab"]{ font-size:.9rem; }
.stButton>button{ border-radius:12px; border:1px solid var(--line); background:var(--panel2); color:var(--ink); }
.stButton>button[kind="primary"]{ background:linear-gradient(135deg,var(--gold),var(--gold2)); color:#231303; border:none; font-weight:700; }
</style>
"""


def fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def fmt_clock(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:01d}:{s:04.1f}"


def seg_timeline_svg(episode_duration: float, segments: list, width: int = 560, height: int = 46) -> str:
    """Render episode timeline with highlighted promo segments."""
    width = max(width, 200)
    dur = max(episode_duration, max(s["end"] for s in segments), 1.0)
    bw = width - 20
    y = 18
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="10" y="{y}" width="{bw}" height="10" rx="5" fill="#23232f"/>',
    ]
    colors = ["#f2c069", "#e2725b", "#8b5cf6"]
    for i, s in enumerate(segments):
        x = 10 + bw * (s["start"] / dur)
        w = bw * ((s["end"] - s["start"]) / dur)
        parts.append(f'<rect x="{x:.1f}" y="{y-2}" width="{max(w,2):.1f}" height="14" rx="7" fill="{colors[i % 3]}" opacity="0.95"/>')
    parts.append(f'<text x="10" y="{height-2}" fill="#9b98ad" font-size="9" font-family="monospace">0:00</text>')
    parts.append(f'<text x="{width-10}" y="{height-2}" fill="#9b98ad" font-size="9" font-family="monospace" text-anchor="end">{fmt_clock(dur)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def contribution_bar_svg(rows: list, width: int = 560, height: int = 18) -> str:
    """Stacked horizontal bar of weight×value contributions per feature."""
    total = sum(v * w for _, v, w in rows) or 1e-9
    palette = ["#f2c069", "#e2725b", "#8b5cf6", "#34d399", "#60a5fa", "#f472b6", "#a3e635", "#94a3b8"]
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect x="0" y="0" width="{width}" height="{height}" rx="9" fill="#1b1b26"/>']
    x = 0.0
    for i, (_, v, w) in enumerate(rows):
        frac = (v * w) / total
        wpx = frac * width
        parts.append(f'<rect x="{x:.2f}" y="0" width="{wpx:.2f}" height="{height}" fill="{palette[i % len(palette)]}" opacity="0.85"/>')
        x += wpx
    parts.append("</svg>")
    return "".join(parts)
