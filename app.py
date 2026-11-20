"""
Kuku Promo Finder — find the best 20–60s promo moments in a microdrama screenplay.

Run:  streamlit run app.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import time

import streamlit as st

from src import __version__
from src.feedback import (FEATURE_KEYS, clear_votes, feature_importance, feedback_summary,
                          load_votes, record_vote)
from src.features import build_beat_signals  # noqa: F401  (kept for dev tooling)
from src.ingest import extract_text
from src.llm import evaluate_promo, llm_available
from src.models import ml_status
from src.parser import Episode, parse_file, parse_screenplay_text
from src.pipeline import analyze_episodes, stitch_episode_promo
from src.scorer import AUDIENCE_PRESETS, PLATFORM_PREFS, resolved_weights
from src.theme import CSS, contribution_bar_svg, fmt_clock, fmt_ts, seg_timeline_svg

st.set_page_config(page_title="Kuku Promo Finder", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- state
for key, default in [
    ("episodes", []), ("ep_durations", {}), ("promos", []), ("analysis_meta", None),
    ("why_overrides", {}), ("stitched", []), ("run_seconds", None), ("llm_evals", {}),
]:
    st.session_state.setdefault(key, default)


def parse_uploaded(files):
    episodes, durations = [], {}
    for f in files:
        try:
            text = extract_text(f.name, f.getvalue())
            ep = parse_screenplay_text(text, source_path=f.name)
            episodes.append(ep)
            durations[ep.episode_number] = ep.duration
        except Exception as exc:
            st.error(f"Could not parse {f.name}: {exc}")
    return episodes, durations


def run_analysis(audience, platform, top_k, lam, include_stitched, desired_duration=None):
    episodes = st.session_state.episodes
    prog = st.progress(0.0, text="Warming up models…")

    def on_episode(i, n, ep):
        prog.progress((i - 1) / n, text=f"ML analysis — {ep.title} ({i}/{n}) · emotion model + embeddings")

    t0 = time.time()
    stats = {}
    use_fb = True
    promos = analyze_episodes(episodes, audience, platform, desired_duration=desired_duration,
                              top_k=top_k, diversity_lambda=lam, use_feedback=use_fb,
                              on_episode=on_episode, collect_stats=stats)
    stitched = []
    if include_stitched:
        by_score = sorted(episodes, key=lambda e: e.episode_number)
        # stitch the 3 episodes that produced the best continuous promos
        best_eps = []
        for p in promos:
            if p.episode not in best_eps:
                best_eps.append(p.episode)
            if len(best_eps) >= 3:
                break
        for ep_num in best_eps:
            ep = next(e for e in episodes if e.episode_number == ep_num)
            sp = stitch_episode_promo(ep)
            if sp:
                stitched.append(sp)
    st.session_state.promos = [p.to_dict() for p in promos]
    st.session_state.stitched = [s.to_dict() for s in stitched]
    st.session_state.run_seconds = round(time.time() - t0, 1)
    st.session_state.analysis_meta = {
        "audience": audience, "platform": platform, "top_k": top_k, "lambda": lam,
        "weights": resolved_weights(audience, platform),
        "n_episodes": len(episodes),
        "rejected": stats.get("rejected", []),
        "considered": stats.get("considered", 0),
    }
    prog.progress(1.0, text="Done")
    time.sleep(0.3)
    prog.empty()


def promos_csv_bytes():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "episode", "segments", "duration_s", "hook_type", "hook_text",
                "cliffhanger_text", "total_score", "hook", "cliffhanger", "intensity",
                "emotional_swing", "premise_alignment", "why_it_works"])
    for p in st.session_state.promos:
        s = p["scores"]
        seg = " | ".join(f"E{x['episode']} {x['start']:.1f}s-{x['end']:.1f}s" for x in p["segments"])
        w.writerow([p["rank"], p["episode"], seg, p["duration_s"], p["hook_type"], p["hook_text"],
                    p["cliffhanger_text"], s.get("total_score"), s.get("hook"), s.get("cliffhanger"),
                    s.get("intensity"), s.get("emotional_swing"), s.get("premise_alignment"),
                    p["why_it_works"]])
    return buf.getvalue().encode("utf-8")


def cliff_agreement_of_current():
    markers = {}
    for ep in st.session_state.episodes:
        markers[ep.episode_number] = [b.end for b in ep.beats
                                      if "cliffhanger" in b.text.lower() or "cut to black" in b.text.lower()]
    hit = tot = 0
    for p in st.session_state.promos:
        ep = next((e for e in st.session_state.episodes if e.episode_number == p["episode"]), None)
        if ep is None:
            continue
        m = markers.get(p["episode"]) or [ep.duration]
        end_t = p["segments"][-1]["end"]
        tot += 1
        if any(abs(end_t - x) <= 2.5 for x in m):
            hit += 1
    return hit, tot


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🎬 Kuku Promo Finder")
    st.caption(f"v{__version__} · ML promo-moment detection for microdrama screenplays")
    status = ml_status()
    emo_state = status["emotion_available"]; emb_state = status["embeddings_available"]
    emo_label = "DistilRoBERTa" if emo_state in (True, None) else "lexicon fallback"
    emb_label = "MiniLM-L6" if emb_state in (True, None) else "off"
    emo_cls = "ok" if emo_state is not False else "warn"
    emb_cls = "ok" if emb_state is not False else "warn"
    chips = [f'<span class="chip {emo_cls}">emotion · <b>{emo_label}</b></span>',
             f'<span class="chip {emb_cls}">embeddings · <b>{emb_label}</b></span>',
             '<span class="chip">loads on first run</span>']
    st.markdown("".join(chips), unsafe_allow_html=True)

    ok_llm, llm_name = llm_available()
    st.markdown(f'<span class="chip {"ok" if ok_llm else "warn"}">LLM evaluator · <b>{llm_name}</b></span>', unsafe_allow_html=True)
    st.caption("Models load lazily on first run. Set KUKU_USE_ML=0 for lexicon-only (tiny) deploys; KUKU_EMOTION_MODEL / KUKU_EMBEDDING_MODEL to swap models; HF_TOKEN+KUKU_LLM_PROVIDER=huggingface (or GEMINI_API_KEY + gemini) for genuine LLM promo evaluation (hook/cliffhanger/standalone/coherence).")

    st.divider()
    st.markdown("**Your feedback signal**")
    fb = feedback_summary()
    st.markdown(f"👍 {fb['up']} · 👎 {fb['down']} · **{fb['n_votes']}** votes")
    if fb["trainable"]:
        st.success(f"Taste model active — blending weight α={fb['blend_weight']:.2f}")
    else:
        st.caption(f"Need ≥6 mixed votes to train taste model (α grows to 0.5 by 16 votes).")
    votes = load_votes()
    if votes:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download votes", json.dumps(votes, indent=2).encode(), "feedback.jsonl", "application/jsonl", use_container_width=True)
        with col2:
            if st.button("Clear votes", use_container_width=True):
                clear_votes()
                st.rerun()
    st.divider()
    if st.button("Reset session", use_container_width=True):
        for k in ["episodes", "ep_durations", "promos", "analysis_meta", "why_overrides", "stitched", "run_seconds", "llm_evals"]:
            st.session_state[k] = [] if k in ("episodes", "promos", "stitched") else ({} if k in ("ep_durations", "why_overrides", "llm_evals") else None)
        st.rerun()

# ---------------------------------------------------------------- hero
st.markdown(
    '<div class="kuku-hero"><h1>Kuku <span class="accent">Promo Finder</span></h1>'
    '<p>Emotion Transformer · MiniLM Embeddings · Logistic Regression for Taste Learning · MMR-based Diversity Ranking.</p>'
    '</div>', unsafe_allow_html=True)

tab_upload, tab_prefs, tab_run, tab_results, tab_feedback, tab_how = st.tabs(
    ["1 · Upload", "2 · Audience & platform", "3 · Analyze", "4 · Ranked promos", "5 · Feedback & regenerate", "How it works"])

# ============ TAB 1 — UPLOAD ============
with tab_upload:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Upload screenplay(s)")
        files = st.file_uploader("Drop .txt / .pdf / .docx files — select many to batch-rank across a whole series",
                                 type=["txt", "pdf", "docx"], accept_multiple_files=True)
        if files and st.button(f"Parse {len(files)} file(s)", type="primary"):
            with st.spinner("Parsing screenplays…"):
                eps, durs = parse_uploaded(files)
                st.session_state.episodes = eps
                st.session_state.ep_durations = durs
                st.session_state.promos = []
                st.session_state.stitched = []
            st.success(f"Parsed {len(eps)} episode(s).")

        with st.expander("Paste the script instead"):
            st.caption("Some preview sandboxes block file uploads (browser↔server POST). Pasting text runs the exact same parser.")
            paste_label = st.text_input("Episode label", "pasted_episode.txt")
            pasted = st.text_area("Paste screenplay text here", height=180,
                                  placeholder="EPISODE NUMBER: Episode 1\nEPISODE LOGLINE: …\n--- SCRIPT START ---\n…")
            if st.button("Parse pasted text"):
                if pasted.strip():
                    ep = parse_screenplay_text(pasted, source_path=paste_label)
                    st.session_state.episodes = st.session_state.episodes + [ep]
                    st.session_state.ep_durations = {**st.session_state.ep_durations, ep.episode_number: ep.duration}
                    st.session_state.promos = []
                    st.session_state.stitched = []
                    st.success(f"Parsed {ep.title}: {len(ep.beats)} beats, {fmt_clock(ep.duration)}.")
                else:
                    st.warning("Paste something first.")

        demo_col, pack_col = st.columns(2)
        with demo_col:
            if st.button("Load demo episode (original sample)", use_container_width=True):
                demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_episode.txt")
                ep = parse_file(demo_path)
                st.session_state.episodes = [ep]
                st.session_state.ep_durations = {0: ep.duration, ep.episode_number: ep.duration}
                st.session_state.promos = []
                st.session_state.stitched = []
                st.success("Loaded the bundled original demo script.")
        with pack_col:
            dev_pack = os.environ.get("KUKU_DEV_PACK")
            if dev_pack and os.path.isdir(dev_pack):
                if st.button("Load input pack (dev only)", use_container_width=True):
                    with st.spinner("Parsing the 50-episode input pack…"):
                        eps = [parse_file(os.path.join(dev_pack, f))
                               for f in sorted(os.listdir(dev_pack)) if f.endswith(".txt")]
                        st.session_state.episodes = eps
                        st.session_state.ep_durations = {e.episode_number: e.duration for e in eps}
                        st.session_state.promos = []
                        st.session_state.stitched = []
                    st.success(f"Loaded {len(eps)} episodes.")

    with c2:
        st.markdown("#### What we detected")
        if st.session_state.episodes:
            for ep in st.session_state.episodes[:12]:
                st.markdown(f"**{ep.title}** · {fmt_clock(ep.duration)} · {len(ep.scenes)} scene(s) · {len(ep.beats)} beats")
                if ep.logline:
                    st.caption(ep.logline[:160])
            if len(st.session_state.episodes) > 12:
                st.caption(f"… and {len(st.session_state.episodes) - 12} more.")
            total_words = sum(len(b.text.split()) for e in st.session_state.episodes for b in e.beats)
            st.markdown(f'<span class="chip ok">total · <b>{len(st.session_state.episodes)} episode(s)</b></span>'
                        f'<span class="chip">≈ <b>{total_words:,}</b> words</span>', unsafe_allow_html=True)
        else:
            st.info("Nothing parsed yet. Upload files, or hit “Load demo episode”.")

# ============ TAB 2 — PREFERENCES ============
with tab_prefs:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Who are you promoting to?")
        audience = st.selectbox("Target audience", list(AUDIENCE_PRESETS.keys()),
                                help="Re-weights the emotion mix and hook archetypes for this viewer.")
        st.caption(AUDIENCE_PRESETS[audience]["note"])
        platform = st.selectbox("Platform", list(PLATFORM_PREFS.keys()),
                                help="TikTok/Reels prefers 20–35s and faster cuts; YouTube tolerates 45–60s.")
        length = st.slider("Promo length sweet spot (seconds)", 20, 60, (22, 45),
                           help="Candidates near this range are preferred (still 18–65s hard bounds).")
    with c2:
        st.markdown("#### Ranking behaviour")
        top_k = st.slider("How many promos to surface", 3, 25, 10)
        lam = st.slider("Diversity of picks (MMR λ)", 0.5, 1.0, 0.72, 0.01,
                        help="1.0 = pure score. Lower = more varied picks across episodes/moments.")
        include_stitched = st.checkbox("Also build stitched “highlight reel” promos (join 2–3 moments per episode)",
                                       value=True,
                                       help="The brief allows joining moments. We join the strongest hook + peak confrontation + scripted cliffhanger, chronologically, ≤30s.")
        st.markdown("**Resolved weights for this audience/platform**")
        w = resolved_weights(audience, platform)
        st.dataframe(
            {"feature": list(w.keys()), "weight": [round(v, 3) for v in w.values()]},
            hide_index=True, use_container_width=True)
    st.session_state["prefs"] = dict(audience=audience, platform=platform, top_k=top_k,
                                     lam=lam, include_stitched=include_stitched, length=length)

# ============ TAB 3 — ANALYZE ============
with tab_run:
    prefs = st.session_state.get("prefs", dict(audience="General audience (balanced)", platform="Any platform",
                                               top_k=10, lam=0.72, include_stitched=True, length=(22, 45)))
    st.markdown("#### Run the ML pipeline")
    if not st.session_state.episodes:
        st.warning("Upload or load a screenplay first (Tab 1).")
    else:
        st.markdown(f"Ready to analyze **{len(st.session_state.episodes)}** episode(s) as **{prefs['audience']}** on **{prefs['platform']}**.")
        st.markdown("Each viable 18–65s window anchored to beat boundaries is scored on 10 features "
                    "(hook semantics + intensity, scripted-cliffhanger alignment, ML emotion intensity & swing, premise alignment vs the episode logline, pace, dialogue balance). "
                    "Then non-max suppression + MMR produce a diverse, non-overlapping top-N.")
        if st.button("▶  Run ML analysis", type="primary", use_container_width=True):
            run_analysis(prefs["audience"], prefs["platform"], prefs["top_k"], prefs["lam"],
                         prefs["include_stitched"], desired_duration=prefs["length"])
            st.success(f"Done in {st.session_state.run_seconds}s — see Tab 4 →")
        if st.session_state.analysis_meta:
            meta = st.session_state.analysis_meta
            rej = meta.get("rejected", [])
            st.markdown(f'<span class="chip">audience · <b>{meta["audience"]}</b></span>'
                        f'<span class="chip">platform · <b>{meta["platform"]}</b></span>'
                        f'<span class="chip">episodes · <b>{meta["n_episodes"]}</b></span>'
                        f'<span class="chip">runtime · <b>{st.session_state.run_seconds}s</b></span>'
                        f'<span class="chip ok">surfaced picks are strict-validated</span>'
                        f'<span class="chip warn">rejected (20–60s/hook≤3s/unresolved) · <b>{len(rej)}</b></span>', unsafe_allow_html=True)

# ============ TAB 4 — RESULTS ============
with tab_results:
    promos = st.session_state.promos
    st.markdown("#### Ranked promos")
    if not promos:
        st.info("No promos yet — run the analysis in Tab 3.")
    else:
        hit, tot = cliff_agreement_of_current()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Promos found", len(promos))
        m2.metric("End on scripted cliffhanger", f"{hit}/{tot}")
        m3.metric("Episodes covered", len({p["episode"] for p in promos}))
        m4.metric("Avg duration", f"{sum(p['duration_s'] for p in promos)/len(promos):.0f}s")

        dl1, dl2, _ = st.columns([1, 1, 2])
        dl1.download_button("⬇ Download promos.json", json.dumps(promos, indent=2, ensure_ascii=False).encode(),
                            "promos.json", "application/json", use_container_width=True)
        dl2.download_button("⬇ Download promos.csv", promos_csv_bytes(), "promos.csv",
                            "text/csv", use_container_width=True)

        for p in promos:
            s = p["scores"]
            segs = p["segments"]
            ep_dur = st.session_state.ep_durations.get(p["episode"], max(x["end"] for x in segs))
            with st.container():
                st.markdown('<div class="promo-card">', unsafe_allow_html=True)
                head_l, head_r = st.columns([1, 5])
                with head_l:
                    st.markdown(f'<div class="rank-badge">#{p["rank"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="score-num">{s.get("total_score", 0):.2f}</div><div class="meta">final score</div>', unsafe_allow_html=True)
                with head_r:
                    seg_str = "  ".join(f"E{x['episode']} {fmt_ts(x['start'])}→{fmt_ts(x['end'])}" for x in segs)
                    st.markdown(f'<p class="promo-title">Episode {p["episode"]} &nbsp;·&nbsp; {seg_str} &nbsp;·&nbsp; {p["duration_s"]:.0f}s</p>', unsafe_allow_html=True)
                    st.markdown(f'<span class="meta">hook type · <b>{p["hook_type"]}</b></span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="hook-line">HOOK @ {fmt_ts(p["hook_time_s"])} — “{p["hook_text"]}”</div>', unsafe_allow_html=True)
                    why = st.session_state.why_overrides.get(p["promo_id"], p["why_it_works"])
                    st.markdown(f'<div class="cliff-line">CLIFFHANGER @ {fmt_ts(p["cliffhanger_time_s"])} — “{p["cliffhanger_text"]}”</div>', unsafe_allow_html=True)
                    st.markdown(f'**Why it works:** {why}')
                    st.markdown(seg_timeline_svg(ep_dur, segs), unsafe_allow_html=True)

                    _eval = st.session_state.llm_evals.get(p["promo_id"])
                    if _eval:
                        if _eval.get("ok"):
                            sc = _eval["scores"]
                            ev = _eval.get("evaluator", "?")
                            st.markdown('<div class="section-label">LLM contextual evaluation · ' + ev.upper() + '</div>', unsafe_allow_html=True)
                            e1, e2, e3, e4 = st.columns(4)
                            e1.metric("Hook (0–3s)", f"{sc['hook_strength']:.0f}/10")
                            e2.metric("Cliffhanger", f"{sc['cliffhanger_quality']:.0f}/10")
                            e3.metric("Standalone", f"{sc['standalone_comprehensibility']:.0f}/10")
                            e4.metric("Coherence", f"{sc['narrative_coherence']:.0f}/10")
                            verdict = _eval.get("verdict", "maybe")
                            vcolor = {"post": "ok", "maybe": "warn", "reject": "warn"}.get(verdict, "warn")
                            st.markdown(f'<span class="chip {vcolor}">verdict · <b>{verdict.upper()}</b></span><span class="meta">{_eval.get("reason", "")}</span>', unsafe_allow_html=True)
                            if _eval.get("warning"):
                                st.caption(_eval["warning"])
                        else:
                            st.error(f"LLM evaluation failed ({_eval.get('evaluator')}): {_eval.get('error')} — ML ranking unaffected.")

                    b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
                    if b1.button("👍 Good pick", key=f"up_{p['promo_id']}"):
                        record_vote(s, 1, p["promo_id"], p["episode"])
                        st.toast("Vote recorded 👍")
                    if b2.button("👎 Bad pick", key=f"down_{p['promo_id']}"):
                        record_vote(s, 0, p["promo_id"], p["episode"])
                        st.toast("Vote recorded 👎")
                    if b3.button("🧠 LLM check", key=f"llm_{p['promo_id']}"):
                        with st.spinner("LLM is watching the promo…"):
                            ep_obj = next((e for e in st.session_state.episodes if e.episode_number == p["episode"]), None)
                            beats = ep_obj.beats if ep_obj else None
                            logline = ep_obj.logline if ep_obj else ""
                            st.session_state.llm_evals[p["promo_id"]] = evaluate_promo(p, beats=beats, logline=logline)
                        st.rerun()

                with st.expander("Score breakdown & raw output"):
                    from src.scorer import score_breakdown

                    class _C: pass
                    c = _C(); c.features = {**s, "_weights": p.get("weights", {})}
                    rows = score_breakdown(c)
                    st.markdown(contribution_bar_svg(rows), unsafe_allow_html=True)
                    for label, v, w in rows:
                        st.markdown(f'<span class="meta">{label}: <b>{v:.2f}</b> × weight {w:.2f} → +{v*w:.3f}</span>', unsafe_allow_html=True)
                    fbv = s.get("feedback_taste")
                    if fbv is not None:
                        st.markdown(f'<span class="chip ok">taste score · <b>{fbv:.2f}</b> (blended with α={min(0.5, s.get("feedback_n_votes", 0)/16.0):.2f})</span>', unsafe_allow_html=True)
                    st.code(json.dumps(p, indent=2, ensure_ascii=False), language="json")
                st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.stitched:
            st.markdown("#### Stitched “highlight reel” promos")
            st.caption("Joined moments from one episode — opens on the strongest hook, cuts to the peak confrontation, lands on the scripted cliffhanger.")
            for sp in st.session_state.stitched:
                seg_str = "  ".join(f"E{x['episode']} {fmt_ts(x['start'])}→{fmt_ts(x['end'])}" for x in sp["segments"])
                st.markdown(f'<div class="stitched"><b>Stitched reel · Episode {sp["episode"]}</b> · {seg_str} · {sp["duration_s"]:.0f}s total<br/>'
                            f'<span class="meta">opens on “{sp["hook_text"][:120]}” · ends on “{sp["cliffhanger_text"][:120]}”</span></div>',
                            unsafe_allow_html=True)

# ============ TAB 5 — FEEDBACK ============
with tab_feedback:
    st.markdown("#### Teach the ranker your taste")
    st.markdown("Every 👍/👎 on a promo card stores that promo’s feature vector. With ≥6 mixed votes a "
                "balanced logistic-regression **taste model** is trained; its P(like) is blended into the "
                "ranking with weight α = min(0.5, n/16). On ephemeral free-tier hosts the votes file resets "
                "on restart — download it from the sidebar to keep it.")
    fb = feedback_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Votes collected", fb["n_votes"])
    c2.metric("Blend weight α", f"{fb['blend_weight']:.2f}")
    c3.metric("Taste model", "trainable" if fb["trainable"] else "not yet")

    imp = feature_importance()
    if imp:
        st.markdown("**What the taste model learned** (logistic regression coefficients — positive = “you prefer more of this”)")
        st.bar_chart({k: float(v) for k, v in imp})
    else:
        st.caption("Vote on at least 6 promos (mix of 👍/👎) to see learned feature weights.")

    st.markdown("#### Regenerate recommendations with your feedback")
    if not st.session_state.episodes:
        st.warning("Nothing to analyze — load a screenplay first.")
    elif st.button("🔁 Re-rank everything with my votes", type="primary"):
        prefs = st.session_state.get("prefs", dict(audience="General audience (balanced)", platform="Any platform",
                                                   top_k=10, lam=0.72, include_stitched=True, length=(22, 45)))
        run_analysis(prefs["audience"], prefs["platform"], prefs["top_k"], prefs["lam"],
                     prefs["include_stitched"], desired_duration=prefs["length"])
        st.success("Re-ranked with feedback baked in — see Tab 4.")

# ============ TAB 6 — HOW IT WORKS ============
with tab_how:
    st.markdown("#### How it works (for the interviewer)")
    st.markdown("""
**Pipeline** — `parse → signals → enumerate → score → MMR → rank`

1. **Parse** — every `**BLOCK**` is timestamped; beats are typed `camera / action / sfx / character (emotion)`. Scene headings, logline, continuity notes and episode numbers are extracted deterministically. No regex guessing about *story*, only about *structure*.
2. **Per-beat ML signals (computed once per episode, cached)** — `emotion-english-distilroberta-base` gives 7-emotion probabilities per beat → arousal-weighted **intensity** and signed **valence** (for emotional swing). `all-MiniLM-L6-v2` embeds every beat → cosine to 16 hook-archetype prompts (betrayal reveal, slap/shock, secret exposed, power reversal…), plus an episode-logline vector for **premise alignment** (self-containment proxy).
3. **Candidate enumeration** — every window anchored to beat SMART: start at any beat boundary, end at any later beat boundary with 18–65s between them (~2–4k windows/episode). Beat anchoring means windows land *exactly* on the script’s cliffhanger beats and cut-to-blacks.
4. **Features** — hook (first 3.5s: archetype semantics + hook keywords + intensity, dialogue preferred for the quote), cliffhanger (final 4s: interruption punctuation, gasps, `cliffhanger`/`cut to black` markers, end-of-episode bonus), intensity, swing, premise alignment, pace (Gaussian around 1.8 cuts/s), dialogue balance, early-episode prior. Audience presets re-weight these & boost archetypes.
5. **Score & rank** — interpretable linear blend → non-max suppression (any two selected promos from one episode overlap ≤45%) → **MMR** re-ranking for feed diversity (λ slider).
6. **Taste learning** — votes → standardized logistic regression → blended with weight α=min(0.5, n/16). Downloadable/portable via sidebar.
7. **Strict validation** — before any promo is shown: 20–60s inclusive · valid timestamps · chronological non-overlapping segments · hook ≤ 3.0s · unresolved ending (scripted marker / episode-end cut / ML cliffhanger ≥ 0.55). Invalid candidates are rejected and replaced by the next-best valid ones.
8. **LLM contextual check (optional, off by default)** — per promo, an instruction-tuned LLM scores hook strength, cliffhanger quality, standalone comprehensibility and narrative coherence from the timecoded excerpt + logline; structured JSON, validated, with heuristic fallback when unconfigured and a clean error when the API fails. The ML ranking never depends on it.

**What ML does vs what rules do** — ML: emotion, semantics, taste, diversity, optional LLM judgment. Rules (deliberately): structure parsing, hard constraints (20–60s + validation), marker detection, overlap suppression. Marker detection *is* human-authored signal in the source material and we exploit it honestly — see eval_report.md.

**Honest limits** — see the solution doc. Short version: it cannot see the *actual video* (it reads scripts, not cuts), premise alignment correlates with plot-keyword density, and cliffhanger agreement saturates on scripts that mark cut-to-blacks (most microdramas). It under-scores dialogue-light visual gags and over-scores melodrama keywords.
""")
    st.markdown('<div class="footer-note">Kuku — AI Generalist Assignment · Problem 1 · Built with Streamlit + Hugging Face models · Deploys free on Streamlit Community Cloud or Hugging Face Spaces · Input pack is Kuku confidential content: do not publish this repo with the pack inside.</div>', unsafe_allow_html=True)
