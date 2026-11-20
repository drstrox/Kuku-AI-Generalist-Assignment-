---
title: Kuku Promo Finder
emoji: 🎬
colorFrom: purple
colorTo: yellow
sdk: streamlit
sdk_version: 1.67.0
app_file: app.py
pinned: false
---

# 🎬 Kuku Promo Finder — Problem 1 (Find the Promos)

Reads a microdrama screenplay and finds the best **20–60 second promo moments** for
Instagram / TikTok / YouTube — ranked, timestamped, with a hook, a cliffhanger and a
one-line *why it works*. Built for the **Kuku — AI Generalist Assignment, Problem 1**.

> The full user workflow: **Upload screenplay → choose audience & promotional preferences →
> ML engine analyses → ranked promos with timestamps + explanations → feedback → regenerate.**

![pipeline](docs/pipeline.svg)

## What it does

1. **Finds promo moments across the whole series** — every 18–65s window anchored to
   beat boundaries is scored on 10 features; non-max suppression + MMR leave a diverse,
   non-overlapping top-N, and a **strict validator** then rejects anything outside
   20–60s (inclusive), with valid chronological non-overlapping timestamps, a hook inside
   the first 3.0s and an unresolved ending. Invalid candidates are replaced by the
   next-best valid ones — never silently kept. Each promo reports: episode, start/end
   timestamps of **every segment** (in order), the hook line, the cliffhanger line, and
   why it works.
2. **Ranks them** for a target audience (Romance / Revenge & justice / Suspense /
   Family-betrayal / General) and platform (TikTok-Reels vs YouTube), with fully visible,
   interpretable weights.
3. **Quality-checks itself** — `src/batch.py` reports cliffhanger agreement, promo–promo
   overlap, baselines & ablations, and flags weak picks. See `outputs/eval_report.md`
   (automated checks and builder-scored human review are kept in separate sections).
4. **Learns your taste** — 👍/👎 per recommendation trains a logistic **taste model**
   (α = min(0.5, n/16)) that re-ranks on the next run. Download/restore votes from the sidebar.
5. **Optional LLM judgment** — a swappable LLM scores each surfaced promo on hook strength,
   cliffhanger quality, standalone comprehensibility and narrative coherence (structured,
   validated JSON). Works with Hugging Face Inference or Gemini; heuristic fallback when
   unconfigured; clean error state if the API fails. The ML ranking never depends on it.

## ML inside

| Component | Model / method | Purpose |
|---|---|---|
| Emotion & intensity | `j-hartmann/emotion-english-distilroberta-base` | arousal-weighted intensity, emotional swing |
| Semantics | `sentence-transformers/all-MiniLM-L6-v2` | hook-archetype match, premise alignment, MMR diversity |
| Ranking | interpretable linear blend + NMS + MMR | transparent, interview-defensible ordering |
| Taste learning | sklearn `LogisticRegression` (standardized) | learns the editor’s preferences from 👍/👎 |
| Validation | deterministic rules (no ML) | 20–60s inclusive, timestamps, order, hook ≤ 3s, unresolved ending |
| LLM (optional) | HF Inference *or* Gemini (env-var swap) | contextual promo evaluation; off unless keys are set |

No paid APIs anywhere. Local models only; the LLM layer stays off unless keys are set.

## Setup & run locally

**Prerequisites:** Python 3.10–3.13, ~2 GB free disk (models), CPU is fine.

```bash
git clone <this-repo> && cd kuku-promo-finder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch CPU wheels (default index is huge CUDA builds; this saves ~2 GB):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip uninstall -y torch 2>/dev/null; pip install torch --index-url https://download.pytorch.org/whl/cpu
streamlit run app.py
```

First run downloads ~500 MB of models (cached afterwards in `~/.cache/huggingface`).
For a zero-ML tiny deploy: `KUKU_USE_ML=0 streamlit run app.py` (lexicon fallback, still works).

If the browser upload fails with **AxiosError 403** inside a sandboxed preview/iframe,
the XSRF handshake is blocked by the platform (not by your file). This repo ships
`.streamlit/config.toml` with `enableXsrfProtection = false` and `enableCORS = false`;
if you are on a strict iframe host you can also paste the screenplay text directly in the
Upload tab ("Paste screenplay instead") — it uses the same parser.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `KUKU_USE_ML` | `1` | `0` → lexicon fallback, no model download |
| `KUKU_EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` | HF emotion classifier |
| `KUKU_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF sentence encoder |
| `KUKU_DEV_PACK` | unset | absolute path to a confidential episode folder; adds a dev-only "load pack" button (never set in deploys) |
| `KUKU_LLM_PROVIDER` | `none` | `none` · `huggingface` · `gemini` |
| `KUKU_LLM_MODEL` | provider default | HF default `meta-llama/Llama-3.1-8B-Instruct` · Gemini default `gemini-2.0-flash` |
| `HF_TOKEN` | unset | required when `KUKU_LLM_PROVIDER=huggingface` (free HF Inference token) |
| `GEMINI_API_KEY` | unset | required when `KUKU_LLM_PROVIDER=gemini` (free Google AI Studio key) |
| `KUKU_LLM_TIMEOUT` | `45` | seconds before an LLM call is abandoned (clean error, app unaffected) |

Models are swappable by changing env vars only — no code edits.

## Running the pipeline (batch CLI, raw outputs)

```bash
python -m src.batch --input-dir "/path/to/50-episode-folder" \
    --out outputs --top-k 25 --seed 13 \
    [--audience "General audience (balanced)"] [--platform "Any platform"] \
    [--manual-review outputs/manual_review.json]
```

Produces `promos.json`, `promos.csv`, `eval_report.md`, `run_manifest.json`.
`outputs/` is gitignored; the input pack must NEVER be committed.

## Output formats

**`promos.json`** — list of promos, one object each:

```json
{
  "rank": 1, "promo_id": "promo_ab12cd34ef", "episode": 26,
  "segments": [{"episode": 26, "start": 132.4, "end": 187.1}],
  "duration_s": 54.7, "hook_type": "accusation",
  "hook_time_s": 132.4, "hook_text": "You are a terrible liar.",
  "cliffhanger_time_s": 187.1, "cliffhanger_text": "…he freezes —",
  "scores": {"hook": 0.83, "cliffhanger": 0.91, "total_score": 0.77, "...": "..."},
  "why_it_works": "Accusation cold open…", "weights": {"hook": 0.30, "...": "..."}
}
```

**`promos.csv`** — flat spreadsheet-friendly table (rank, episode, segments, duration_s,
hook, cliffhanger, component scores, why_it_works).

**`eval_report.md`** — §A automated evaluation (strict-validation accounting incl. rejected
candidates; cliffhanger agreement; overlap; baselines & ablation; auto-flagged weak picks)
and §B human evaluation (builder-scored hook/cliffhanger/standalone ratings + failure
examples; explicitly labelled as judgment, not user research).

**`run_manifest.json`** — run date, input dir, episode count/numbers, audience/platform,
top-k, seed, verified model availability, package versions, Python/platform, git commit,
weights, runtime, and the validation accounting (considered → rejected → surfaced).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AxiosError 403` on file upload | Sandboxed preview blocks the XSRF handshake — hard-refresh; or use the paste-text path; `.streamlit/config.toml` already relaxes XSRF/CORS |
| `torch` install pulls ~2.5 GB CUDA wheels | install the CPU build: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| First analysis takes minutes | one-time model download; subsequent runs reuse `~/.cache/huggingface` |
| `KUKU_USE_ML=0` gives weaker insights | expected — lexicon fallback has no emotion/semantic signal; use for deploy smoke-tests only |
| LLM button shows warning/error | set `KUKU_LLM_PROVIDER` + the matching key, or leave unset for heuristic fallback; the ML ranking never needs the LLM |
| Feedback "disappears" between sessions | free hosts have ephemeral disks; use the sidebar download/restore votes buttons |
| `No .txt episodes found` in batch | `--input-dir` must point at the folder containing `Episode *.txt` |

## Deploy free

**Hugging Face Spaces (recommended, free, 16 GB RAM)** — this README carries the Spaces
front-matter. New Space → SDK *Streamlit* → upload repo *without* the input pack → it
builds and runs. Cold start ≈ 1–2 min (model download); use persistent storage or accept
re-downloads.

**Streamlit Community Cloud (free, ~1 GB RAM)** — connect the GitHub repo → main file
`app.py` → deploy. Torch CPU fits, but keep `requirements.txt` as-is (CPU wheels). The
feedback file is ephemeral; use the sidebar download/restore.

**Vercel** — Vercel does not run Streamlit servers. It *can* host the static solution doc /
landing page; pair it with Spaces/Streamlit Cloud for the app itself.

**Colab** — open a notebook, `!pip install -r requirements.txt`,
`!streamlit run app.py & npx localtunnel --port 8501` (or `pyngrok`) to get a public URL;
upload episodes in the UI.

## Confidentiality & integrity

The 50-episode input pack is Kuku confidential content shared for this assignment. **Do not
commit it or publish raw outputs publicly** — submit them through the placement form as asked.
This repo is `.gitignore`d for exactly that. `data/demo_episode.txt` is an original sample
written for this project and is safe to publish. `outputs/manual_review.json` contains
paraphrased reviewer notes only — no screenplay excerpts — so it is safe to keep in git.

## Repo map

```
app.py                  Streamlit UI (6 tabs: upload → prefs → analyze → results → feedback → how it works)
src/parser.py           screenplay → EPISODE/SCENE/BEAT structure (timestamp-exact)
src/features.py         per-beat ML signals + beat-anchored candidate enumeration
src/scorer.py           weights, audience/platform presets, NMS + MMR, explanations
src/pipeline.py         end-to-end ranked promos (+ strict validation + stitched highlight reel)
src/validation.py       strict validator: 20–60s inclusive, timestamps, order, hook ≤ 3s, unresolved ending
src/feedback.py         👍/👎 → logistic taste model
src/llm.py              optional swappable LLM contextual evaluator (HF / Gemini; validated JSON + fallback)
src/batch.py            CLI batch run → promos.json / promos.csv / eval_report.md / run_manifest.json
tests/                  parser · validation · LLM · end-to-end suites (pytest)
outputs/                raw outputs from the full 50-episode input pack (gitignored)
data/demo_episode.txt   original sample screenplay
docs/                   solution doc + architecture diagram + walkthrough script
```

## Known limits (honest version)

See `docs/` — short list: it reads *scripts*, not video (no visual judgment); premise
alignment tracks plot-keyword density; cliffhanger agreement saturates on microdramas
(they literally mark cut-to-black); melodrama keywords can over-fire without human review;
feedback persists only per host (ephemeral on free tiers).
