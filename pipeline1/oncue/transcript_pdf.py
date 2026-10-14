"""Readable PDF transcripts with a class summary: `python -m oncue pdf [--class CID ...]`.

For Brady's own reading, never committed: the PDFs carry names and classmates' words, so they
are written under the (gitignored) work dir, `{data}/pdf/{split}/`.

Page 1 is the summary only:

* the class's Forum link (the full recording) and, per detected raise, its strip clip and
  frame sheet (audit/, grey view);
* my raises: onset on the video clock (re-timed), how long the hand stayed up, outcome,
  time to being called; class-wide votes;
* how much I spoke (Forum voice events, main room), my turns, fresh turns, my longest turn;
  talk time by role from the transcript words;
* lesson segments, polls, chat messages;
* clock sync: calibration status, offset, residuals, A/V lag, strip coverage.

From page 2 on: the transcript (Scribe, names mode), rebuilt for free from the saved
`*_asr.json` (no audio is sent anywhere).
"""

from __future__ import annotations

import logging
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from . import config, events, transcribe

log = logging.getLogger(__name__)

FORUM_CLASS_URL = "https://forum.minerva.edu/app/courses/{course}/sections/{section}/classes/{cid}"


def clock(t: float | None) -> str:
    if t is None or not np.isfinite(t):
        return "-"
    t = float(t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}" if t >= 3600 \
        else f"{int(t // 60)}:{t % 60:04.1f}"


def forum_url(klass: dict) -> str | None:
    from scribe_core.forum_events import kebab

    k = kebab(klass or {})
    sec = k.get("section") or {}
    course = (sec.get("course") or {}).get("id")
    if None in (course, sec.get("id"), k.get("id")):
        return None
    return FORUM_CLASS_URL.format(course=course, section=sec["id"], cid=k["id"])


# --------------------------------------------------------------------------- names transcript


class _SavedASR:
    """Replays a saved `*_asr.json` (like Scribe's CLI `--reuse-asr`): no audio, no cost."""

    accepts_url = False
    needs_audio = False

    def __init__(self, data: dict):
        self.data = data
        self.name = data.get("provider") or "saved"
        self.model = data.get("model")
        self.diarizes = bool(data.get("diarized"))

    def transcribe(self, **_):
        from scribe_core.asr_base import ASRResult

        d = self.data
        return ASRResult(words=d.get("words") or [], provider=self.name, model=self.model,
                         language=d.get("language"), duration=d.get("duration"),
                         diarized=self.diarizes, mode="saved",
                         meta={"source_res": d.get("source_res")})


def names_transcript(cid: int, tmp_dir: Path) -> dict:
    """Scribe's transcript of one class in names mode, rebuilt from the saved ASR."""
    import json

    from scribe_core.pipeline import TranscribeOptions, transcribe_bundle

    asr_path = transcribe.find_transcript(cid, "asr")
    if asr_path is None:
        raise FileNotFoundError(f"no *_asr.json for class {cid}")
    bundle = transcribe.load_bundle(cid)
    opts = TranscribeOptions(privacy="names", formats=("json",), debug_html=False,
                             save_asr=False, language=config.TRANSCRIBE_LANGUAGE)
    res = transcribe_bundle(bundle, out_dir=tmp_dir, provider=_SavedASR(
        json.loads(asr_path.read_text(encoding="utf-8"))), options=opts,
        me_id=events.load_me_id())
    for f in res["files"]:                         # keep nothing but the PDF
        Path(f).unlink(missing_ok=True)
    return res["transcripts"]["names"]


# --------------------------------------------------------------------------- raises in the transcript


ROW_MAX_CHARS = 2000      # a long monologue is split over several rows (a row cannot span pages)


def _chunks(text: str, n: int = ROW_MAX_CHARS) -> list[str]:
    """`text` in pieces of at most about `n` characters, cut at spaces."""
    text = str(text)
    out = []
    while len(text) > n:
        cut = text.rfind(" ", 0, n)
        cut = n if cut <= 0 else cut
        out.append(text[:cut])
        text = text[cut:].lstrip()
    return out + [text]


def raise_onset(r: dict) -> float:
    o = r.get("onset_retimed_s")
    return float(r["onset_s"]) if o is None or not np.isfinite(o) else float(o)


def raise_context(t: float, transcript: dict) -> str:
    """What was happening when the hand went up: whose line was being spoken, or how long it
    had been quiet and when speech resumed."""
    from scribe_core.schema import speaker_label

    utts = [{**u, "end": u.get("end", u["start"])} for u in transcript.get("utterances") or []
            if u.get("text")]
    cur = [u for u in utts if u["start"] <= t <= u["end"]]
    if cur:
        u = cur[-1]
        return f"while {speaker_label(transcript, u['speaker_id'])} was speaking " \
               f"({t - u['start']:.1f} s into the line)"
    prev = max((u["end"] for u in utts if u["end"] < t), default=None)
    nxt = min((u["start"] for u in utts if u["start"] > t), default=None)
    quiet = "" if prev is None else f"quiet for {t - prev:.1f} s"
    after = "" if nxt is None else f"speech resumed {nxt - t:.1f} s later"
    return "in a pause: " + ", ".join(x for x in (quiet, after) if x) if (quiet or after) \
        else "no speech nearby"


# --------------------------------------------------------------------------- summary


def _table(name: str, cid: int) -> pd.DataFrame | None:
    p = config.table_path(name)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df[df["class_id"] == int(cid)].reset_index(drop=True)


def summary(cid: int) -> dict:
    """Numbers for page 1 (all from the pipeline tables of the active work dir)."""
    s: dict = {"class_id": int(cid)}
    voice = _table("voice", cid)
    if voice is not None and len(voice):
        main = voice[voice["room_main"].astype(bool)]
        mine = main[main["role"] == config.ROLE_ME].sort_values("start_s")
        s["my_seconds"] = float((mine["end_s"] - mine["start_s"]).sum())
        s["my_events"] = len(mine)
        # turns: my events merged across gaps < TURN_MERGE_GAP_S; fresh after 30 s of my silence
        turns = []
        for a, b in zip(mine["start_s"], mine["end_s"], strict=True):
            if turns and a - turns[-1][1] < config.TURN_MERGE_GAP_S:
                turns[-1][1] = max(turns[-1][1], b)
            else:
                turns.append([a, b])
        s["my_turns"] = len(turns)
        s["my_fresh_turns"] = sum(1 for i, (a, _) in enumerate(turns)
                                  if i == 0 or a - turns[i - 1][1] >= config.FRESH_SILENCE_S)
        s["my_longest_turn_s"] = max((b - a for a, b in turns), default=0.0)
        s["my_first_speech_s"] = turns[0][0] if turns else None
        s["n_students_spoke"] = int(main.loc[main["role"] == config.ROLE_STUDENT,
                                             "spk_hash"].nunique())
    words = _table("words", cid)
    if words is not None and len(words):
        dur = (words["end_s"] - words["start_s"]).clip(lower=0)
        by = dur.groupby(words["role"]).sum()
        s["talk_by_role_s"] = {k: float(v) for k, v in by.items()}
        s["n_words"] = len(words)
    eps = _table("hand_episodes", cid)
    if eps is not None and len(eps):
        bids = eps[(eps["ep_type"] == config.EP_BID) & ~eps["left_censored"].astype(bool)]
        s["raises"] = bids.sort_values("onset_s").to_dict("records")
        s["n_votes"] = int((eps["ep_type"] == config.EP_VOTE).sum())
    seg = _table("segments", cid)
    s["segments"] = [] if seg is None else seg.sort_values("start_s")[
        ["start_s", "title_masked", "seg_type"]].to_dict("records")
    chat = _table("chat", cid)
    if chat is not None:
        s["n_chat"] = len(chat)
        s["n_my_chat"] = int((chat["role"] == config.ROLE_ME).sum())
    polls = _table("polls", cid)
    if polls is not None:
        s["n_polls"] = len(polls)
        s["n_my_poll_answers"] = int(polls["my_answer_text"].notna().sum())
    timing = _table(transcribe.TIMING_TABLE, cid)
    if timing is not None and len(timing):
        s["timing"] = timing.iloc[-1].to_dict()
    qc = _table("hand_qc", cid)
    if qc is not None and len(qc):
        s["hand_qc"] = qc.iloc[-1].to_dict()
    clips = config.AUDIT_DIR / str(int(cid)) / "raises.csv"
    s["clips"] = pd.read_csv(clips).to_dict("records") if clips.exists() else []
    return s


# --------------------------------------------------------------------------- PDF


def write_pdf(transcript: dict, summ: dict, path: Path, *, video_url: str | None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from scribe_core.schema import speaker_label

    st = getSampleStyleSheet()
    body = ParagraphStyle("B", parent=st["Normal"], fontSize=9.5, leading=12, wordWrap="CJK")
    small = ParagraphStyle("S", parent=body, fontSize=8.5, leading=10.5)
    grey = ParagraphStyle("G", parent=body, textColor=colors.grey)
    bold = ParagraphStyle("Bd", parent=body, fontName="Helvetica-Bold")
    head = ParagraphStyle("H", parent=body, fontName="Helvetica-Bold",
                          textColor=colors.whitesmoke)
    h3 = st["Heading3"]
    grid = TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4f4f")),
                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
                       ("LEFTPADDING", (0, 0), (-1, -1), 4),
                       ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                       ("TOPPADDING", (0, 0), (-1, -1), 2),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 2)])

    def P(text, style=body):
        return Paragraph(escape(str(text if text is not None else "")), style)

    def link(url, text):
        return f'<a href="{escape(url)}" color="blue">{escape(text)}</a>'

    def tbl(rows, widths):
        t = Table(rows, colWidths=[w * inch for w in widths], repeatRows=1)
        t.setStyle(grid)
        return t

    c = transcript["class"]
    el = [P(c.get("title"), st["Title"]),
          P(f"{c.get('course_code') or ''} · {c.get('date') or ''} · class {c.get('id')}",
            ParagraphStyle("c", parent=body, alignment=1)), Spacer(1, 6)]
    if video_url:
        el.append(Paragraph(f"<b>Video (Forum):</b> {link(video_url, video_url)}", body))
    clips = {int(r["ep_id"]): r for r in summ.get("clips") or []}
    if clips:
        el.append(Paragraph("<b>Strip clips of my raises (grey view, local files):</b> " +
                            ", ".join(link(Path(r["clip"]).resolve().as_uri(),
                                           clock(r["onset_s"]))
                                      for r in clips.values()), body))
    el.append(Spacer(1, 8))

    # my hand raises
    raises = summ.get("raises") or []
    el.append(P(f"My hand raises: {len(raises)}"
                + (f"  (class-wide votes: {summ['n_votes']})" if summ.get("n_votes") else ""), h3))
    if raises:
        rows = [[P(x, head) for x in ("#", "Hand up at", "Up for", "Outcome", "Called after",
                                       "What was happening", "Clip")]]
        for i, r in enumerate(raises, 1):
            onset = raise_onset(r)
            clip = clips.get(int(r["ep_id"]))
            rows.append([P(i, small), P(clock(onset), small), P(f"{r['dur_s']:.1f} s", small),
                         P(r.get("outcome"), small),
                         P("-" if not np.isfinite(r.get("call_latency_s") or np.nan)
                           else f"{r['call_latency_s']:.1f} s", small),
                         P(raise_context(onset, transcript), small),
                         Paragraph(link(Path(clip["clip"]).resolve().as_uri(), "clip")
                                   if clip else "-", small)])
        el.append(tbl(rows, [0.3, 0.8, 0.6, 0.75, 0.7, 2.25, 0.45]))
    el.append(Spacer(1, 8))

    # speaking
    rows = [[P("Speaking", head), P("", head)]]
    for label, key, fmt in (("Seconds I spoke (Forum mic, main room)", "my_seconds", "{:.0f} s"),
                            ("My turns (fresh: after 30 s of my silence)", None, None),
                            ("My longest turn", "my_longest_turn_s", "{:.0f} s"),
                            ("My first contribution at", "my_first_speech_s", "clock"),
                            ("Students who spoke", "n_students_spoke", "{}")):
        if key is None:
            val = f"{summ.get('my_turns', 0)} ({summ.get('my_fresh_turns', 0)} fresh)"
        elif summ.get(key) is None:
            val = "-"
        else:
            val = clock(summ[key]) if fmt == "clock" else fmt.format(summ[key])
        rows.append([P(label), P(val)])
    tr = summ.get("talk_by_role_s") or {}
    total = sum(tr.values()) or 1.0
    for role in (config.ROLE_PROF, config.ROLE_ME, config.ROLE_STUDENT, config.ROLE_UNK):
        if role in tr:
            rows.append([P(f"Talk time from the transcript: {role}"),
                         P(f"{tr[role] / 60:.1f} min ({100 * tr[role] / total:.0f}%)")])
    el.append(tbl(rows, [3.6, 2.4]))
    el.append(Spacer(1, 8))

    # class activity
    rows = [[P("Class activity", head), P("", head)],
            [P("Polls (answered by me)"),
             P(f"{summ.get('n_polls', 0)} ({summ.get('n_my_poll_answers', 0)})")],
            [P("Chat messages (mine)"), P(f"{summ.get('n_chat', 0)} ({summ.get('n_my_chat', 0)})")]]
    el.append(tbl(rows, [3.6, 2.4]))
    segs = summ.get("segments") or []
    if segs:
        el.append(Spacer(1, 6))
        rows = [[P("Segment starts", head), P("Title", head), P("Type", head)]]
        rows += [[P(clock(g["start_s"]), small), P(g["title_masked"], small),
                  P(g["seg_type"], small)] for g in segs]
        el.append(tbl(rows, [1.0, 4.0, 1.0]))
    el.append(Spacer(1, 8))

    # clock sync
    tm, qc = summ.get("timing") or {}, summ.get("hand_qc") or {}

    def num(d, k, f="{:.2f}"):
        v = d.get(k)
        return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)
    rows = [[P("Clock sync (all times above are on the video's clock)", head), P("", head)],
            [P("Forum → video calibration"),
             P(f"{tm.get('cal_status', '-')} / {tm.get('cal_method', '-')}, "
               f"{num(tm, 'cal_n_windows', '{:.0f}')} windows, offset {num(tm, 'cal_offset_s')} s")],
            [P("Residual median / p95"),
             P(f"{num(tm, 'resid_med_s')} / {num(tm, 'resid_p95_s')} s "
               f"({'pass' if tm.get('cal_qa_pass') else 'FAIL'})")],
            [P("A/V lag (ring vs words): all / mine"),
             P(f"{num(qc, 'av_lag_s')} / {num(qc, 'av_lag_brady_s')} s")],
            [P("Strip coverage (my tile found)"), P(num(qc, "coverage", "{:.0%}"))]]
    el.append(tbl(rows, [3.6, 2.4]))

    # transcript
    el += [PageBreak(), P("Transcript", h3),
           P("Rows marked HAND RAISED are my raises, at the moment the hand appeared "
             "(re-timed to the video frame), with what was happening then.", small),
           Spacer(1, 4)]
    rows = [[P("Time", head), P("Speaker", head), P("Contribution", head)]]
    items = [(u["start"], 1, u) for u in transcript.get("utterances") or [] if u.get("text")]
    items += [(raise_onset(r), 0, r) for r in raises]
    marks = []
    for _, kind, x in sorted(items, key=lambda z: (z[0], z[1])):
        if kind == 0:                                   # a raise
            o = raise_onset(x)
            lat = x.get("call_latency_s")
            called = "" if lat is None or not np.isfinite(lat) else f", called after {lat:.1f} s"
            rows.append([P(clock(o), bold), P("HAND RAISED", bold),
                         P(f"up {x['dur_s']:.0f} s, {x.get('outcome')}{called}. "
                           f"{raise_context(o, transcript)}.", bold)])
            marks.append(len(rows) - 1)
            continue
        style = grey if x.get("speaker_id") is None else body
        who = speaker_label(transcript, x["speaker_id"])
        for k, part in enumerate(_chunks(x["text"])):  # a table row must fit on one page
            rows.append([P(clock(x["start"]) if k == 0 else ""), P(who if k == 0 else "(cont.)",
                         style), P(part, style)])
    t = tbl(rows, [0.75, 1.6, 4.65])
    for i in marks:
        t.setStyle(TableStyle([("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e3efb5"))]))
    el.append(t)
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=letter, leftMargin=50, rightMargin=50, topMargin=50,
                      bottomMargin=50, title=str(c.get("title") or "")).build(el)
    return path


def make_pdf(cid: int, *, out_dir: Path | None = None) -> Path:
    import tempfile

    from .splits import split_of

    cid = int(cid)
    klass = events.read_raw(cid, "class") or {}
    with tempfile.TemporaryDirectory() as tmp:
        t = names_transcript(cid, Path(tmp))
    split = split_of().get(cid, "unsplit")
    base = Path(out_dir) if out_dir else config.TABLES_DIR.parent / "pdf" / split
    name = f"{t['class'].get('course_code') or 'class'}_{t['class'].get('date') or ''}_class{cid}.pdf"
    return write_pdf(t, summary(cid), base / name, video_url=forum_url(klass))


def cli(args) -> int:
    """`python -m oncue pdf [--class CID ... | --split S] [--work-dir DIR]`: classes with a saved
    `*_asr.json` by default."""
    config.load_env()
    ids = transcribe._class_ids(args)
    ok, bad = 0, 0
    for cid in ids:
        try:
            path = make_pdf(cid)
            ok += 1
            log.info("wrote %s", path)
        except Exception as exc:  # noqa: BLE001 - one bad class must not stop the batch
            bad += 1
            log.error("pdf class %s failed: %s", cid, transcribe.redact_error(
                f"{type(exc).__name__}: {exc}"))
    print(f"oncue pdf: {ok} written, {bad} failed (under {config.TABLES_DIR.parent / 'pdf'})")
    return 0 if not bad else 1
