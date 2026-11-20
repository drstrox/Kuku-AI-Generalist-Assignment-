# 🎬 Kuku Promo Finder

**ML-assisted promo-moment mining for microdrama screenplays.**
Upload a script. Get ranked 20–60s cut-ins with second-exact timecodes, a hook,
a cliffhanger, and a one-line rationale — every pick strict-validated before you see it.

[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-ff4b4b.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-38%20passing-2e9e6b.svg)](tests/)
[![No paid APIs](https://img.shields.io/badge/paid%20APIs-none-6d3ac9.svg)](#configuration)

</div>

![pipeline](pipeline.svg)

---

## Why this exists

Microdrama editors hand-pick promo clips — one episode at a time, by watching everything.
The selection criteria are repeatable, timestamp-exact and high-volume: a long target for ML-assisted drafting with a human approving before anything ships.

**Kuku Promo Finder** reads a screenplay (`.txt` / `.pdf` / `.docx`), scores every viable 18–65s beat-anchored window, and returns a diverse, non-overlapping, audience-tuned shortlist. It never edits video; the screenplay's own timestamps are the ground truth.

## Highlights

- **Whole-series search** — candidates enumerated across *all* episodes at once; de-duplicated
  (NMS), diversified (MMR, λ=0.72), ranked for the selected audience & platform.
- **Strict validation, always on** — every surfaced promo (ranked or stitched) must pass:
  duration **20–60 s inclusive** · finite, chronological, non-overlapping timestamps ·
  **hook within the first 3.0 s** · **unresolved ending**. Invalid candidates are rejected,
  logged and replaced — never silently kept.
- **Interpretable ranking** — a fully visible linear blend of 10 features (hook archetypes,
  ML emotion intensity/swing, premise alignment, pace, dialogue balance, episode prior).
  Every score is inspectable per-feature; no black-box ordering.
- **Taste learning** — 👍/👎 votes train a small logistic "taste model"
  (α = min(0.5, n/16)) that re-ranks on the next run. Offline, free, downloadable.
- **Optional LLM judgment** — swappable LLM scores each surfaced promo on hook strength,
  cliffhanger quality, standalone comprehensibility and narrative coherence as
  range-validated JSON. HF Inference or Gemini; heuristic fallback when unconfigured;
  clean error when the API fails; the ML ranking never depends on it.
- **No paid APIs** — local models (DistilRoBERTa emotion + MiniLM-L6 embeddings),
  CPU-friendly, free-tier deployable.

## Quickstart

**Prerequisites:** Python 3.10–3.13 · ~2 GB disk (models) · CPU is fine

```bash
git clone https://github.com/drstrox/Kuku-AI-Generalist-Assignment-
cd kuku-promo-finder
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

- Open http://localhost:8501
- **Upload** a screenplay or paste the text file 
- Pick audience & platform
- Click on RUN ML Analysis
- Ranked promo cards with copy-ready `HH:MM:SS.ss` timecodes.

### Batch mode — reproduce the raw outputs

```bash
python -m src.batch --input-dir "/path/to/50-episode-folder" \
    --out outputs --top-k 25 --seed 13
```

Produces `promos.json` · `promos.csv` · `eval_report.md` · `run_manifest.json`.

## Configuration

All models are swappable via environment variables — no code edits.

| Variable | Default | Meaning |
|---|---|---|
| `KUKU_USE_ML` | `1` | `0` → lexicon-only mode (no model downloads) |
| `KUKU_EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` | HF emotion classifier |
| `KUKU_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF sentence encoder |
| `KUKU_LLM_PROVIDER` | `none` | `none` · `huggingface` · `gemini` |
| `KUKU_LLM_MODEL` | provider default | HF: `meta-llama/Llama-3.1-8B-Instruct` · Gemini: `gemini-2.0-flash` |
| `KUKU_LLM_TIMEOUT` | `45` | LLM timeout (s) — on failure: clean error, app unaffected |

## Testing

```bash
pip install pytest
python -m pytest tests/ -q     # 38 passed: parser · validation · LLM · end-to-end
```

Validation suite covers duration boundaries (19.99 / 20.0 / 60.0 / 60.01 s), hook-window
shadows, multi-segment duration sums, timestamp finiteness, chronology, overlaps,
unresolved-ending detection, LLM output parsing/failure paths, and an end-to-end run on the
bundled original sample screenplay.

## What the outputs look like

```json
{
  "rank": 1, "episode": 27, "segments": [{"episode": 27, "start": 124.6, "end": 150.5}],
  "duration_s": 25.9, "hook_type": "Betrayal reveal",
  "hook_text": "She admits she has been giving the Duchess poison in secret!",
  "cliffhanger_time_s": 150.5, "scores": {"hook": 0.95, "cliffhanger": 1.00, "total_score": 0.82},
  "why_it_works": "…"
}
```

`eval_report.md` is split into **§A Automated** (strict-validation accounting with every
rejected candidate and why · cliffhanger agreement · baselines & ablations · auto-flagged
weak picks) and **§B Human** (builder-scored hook / cliffhanger / standalone ratings with
failure examples — explicitly judgment, not user research). Numbers: 18/20 top-20 promos
end within 2.5s of a scripted cliffhanger (21/25 overall) vs ~1/15 random; hook strength
~0.76 vs 0.36 random. No CTR claims — see **Honest limits** below.

## Project structure

```
app.py                  Streamlit UI (upload → prefs → analyze → results → feedback → how it works)
src/parser.py           screenplay → EPISODE/SCENE/BEAT structure (timestamp-exact)
src/features.py         per-beat ML signals + beat-anchored candidate enumeration
src/scorer.py           weights, audience/platform presets, NMS + MMR, explanations
src/pipeline.py         end-to-end ranked promos (+ strict validation, stitched reels)
src/validation.py       strict validator (20–60s inclusive · timestamps · hook ≤ 3s · unresolved ending)
src/feedback.py         👍/👎 → logistic taste model
src/llm.py              swappable LLM contextual evaluator (validated JSON + fallback)
src/batch.py            batch CLI → promos.json / .csv / eval_report.md / run_manifest.json
tests/                  parser · validation · LLM · end-to-end (pytest)
outputs/                eval_report.md · run_manifest.json · manual_review.json
data/demo_episode.txt   original sample screenplay (safe to publish)
docs/                   solution doc (2-page PDF) · architecture diagram · walkthrough script
```

## Honest limits

- It reads **scripts, not video** — no visual judgment, no cut/frame analysis.
- Premise alignment is a cosine proxy, correlated with plot-keyword density — not a
  comprehension test. Melodrama keywords over-fire without human review.
- Cliffhanger agreement saturates on this corpus (episodes literally mark cut-to-black);
  it validates endpoint placement, **not** CTR or watch time.
- Feedback persists only on writable disk (ephemeral on most free hosts).
- Builder-scored human review is one reviewer with the screenplay visible — n=1, ±1
  tolerance; not user research. A blind multi-editor study is the obvious next step.

## Confidentiality & academic integrity

The 50-episode input pack this was developed against is client/confidential material and is
**not** in this repository (`.gitignore`d; raw outputs submitted separately).
`data/demo_episode.txt` is an original sample written for this project.
Built with AI pair-programming; every modeling decision, weight and failure analysis was
reviewed and is defensible.

---

<div align="center">
<sub>Problem 1 · AI Generalist Assignment (IIT Indore on-campus placements) · 2026</sub>
</div>
