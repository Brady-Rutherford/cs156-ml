"""Scribe transcription runs and the causal `words` table (spec §2.1, §4.1 A1-A2, §4.3, §13).

`transcribe_many` runs `scribe_core.pipeline.transcribe_class` per class with the options of
§13: privacy "ids", json + jsonl, English, no debug HTML, `*_asr.json` saved, the lowres
rendition and `strict_res`. The ASR provider is chosen per course from
`config.ASR_PROVIDER_BY_COURSE` (A8; local by default). `hand_log` is NEVER passed: the live
logger's key presses would become `hand-raise-key` events in the transcript (L26).

`build_words` / `words_table` rebuild the words from `*_asr.json` and the RAW Forum voice
events, mirroring `pipeline.transcribe_bundle` (D7):

    parse_voice_events -> calibrate -> shift      (Scribe, for the calibration and QA)
    events.parse_voice (unclamped, gap events dropped) -> Forum-causal role per word
    fuse() + smooth()                              (kept only as role_fused / how; A-attrib)

Forum-causal role (§4.1 A2): the main-room voice event with the most overlap with
[start - 0.25, end + 0.25] gives the speaker, mapped to prof / me / student (unk without an
event). The role is available at role_avail_s = max(end + eps_w, event start + eps_F); before
that the word's role reads unk. Scribe's cluster map and smooth() look at the whole file and
at later words, so they never feed `role`.

Text (§5.1, §5.3): names are masked in memory on Deepgram's cased tokens ([NAME]; Brady's
variants [ME]); `text_masked` keeps casing and punctuation (diagnostics, F-name, F-punct);
`text_norm` is lowercase without punctuation, with name tokens and one-word CALLON phrases
("unmute") deleted. It is decided per word, from that word alone: multi-word CALLON phrases
("go ahead") are removed later from joined text that is already available at t
(`causal.join_tokens`, `text.WindowBuilder`), never by looking ahead at a later word.

Class QA (§3, §4.3): calibration must be applied, status ok/piecewise, `method == "windowed"`
with >= 6 usable windows, max_residual_s <= 1.5 s, and the measured residual p95 <= 1.5 s and
median <= 0.5 s. A global-fallback fit (which reports max_residual_s = 0) never passes.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import re
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scribe_core.calibrate import calibrate, shift
from scribe_core.forum_events import (
    kebab,
    main_room_info,
    parse_voice_events,
    recording_timeline,
    timeline_duration,
    timing_precision,
)
from scribe_core.fuse import EDGE, calibration_words, cluster_to_user, fuse, smooth

from . import config, events, schemas

log = logging.getLogger(__name__)

# Bundle parts handed to Scribe. Post-class grading and staff analytics are left out on
# purpose (the transcript is never a feature source, but there is no reason to feed them).
# `listing` (our own sync file) only backs up the course code for the provider choice.
SCRIBE_BUNDLE_PARTS = ("class", "events", "polls", "poll_responses", "breakouts", "listing")
RUN_LOG_NAME = "transcribe_runs.csv"
TIMING_TABLE = "class_timing"


# =========================================================================== transcription


def provider_for(course_code: str | None, override: str | None = None,
                 mapping: dict[str, str] | None = None) -> str:
    """ASR provider for a course (A8). An explicit "deepgram" override is refused for a course
    whose consent does not cover third-party ASR; "local" is always allowed."""
    allowed = config.asr_provider(course_code, mapping)
    if override in (None, "", "auto"):
        return allowed
    if override not in config.ASR_PROVIDERS:
        raise ValueError(f"unknown ASR provider {override!r}")
    if override == "deepgram" and allowed != "deepgram":
        raise ValueError(f"course {course_code!r} is not cleared for third-party ASR "
                         "(config.ASR_PROVIDER_BY_COURSE); use local")
    return override


def scribe_has_strict_res() -> bool:
    from scribe_core.pipeline import TranscribeOptions

    return "strict_res" in {f.name for f in dataclasses.fields(TranscribeOptions)}


def transcribe_options(provider: str, salt: str, *, require_strict_res: bool = True,
                       workdir_root: Path | None = None):
    """Scribe `TranscribeOptions` exactly as §13 lists them."""
    from scribe_core.pipeline import TranscribeOptions

    kw = {"provider": provider, "privacy": config.TRANSCRIBE_PRIVACY,
          "formats": tuple(config.TRANSCRIBE_FORMATS), "language": config.TRANSCRIBE_LANGUAGE,
          "debug_html": False, "save_asr": True, "res": config.SRC_RES, "salt": salt,
          "workdir_root": workdir_root}
    if scribe_has_strict_res():
        kw["strict_res"] = True
    elif require_strict_res:
        raise RuntimeError(
            "This Scribe has no TranscribeOptions.strict_res (spec §2.1 patch), so a missing "
            f"{config.SRC_RES} rendition would silently fall back to the other one. Apply the "
            "Scribe patch, or pass require_strict_res=False / --allow-no-strict-res for a pilot.")
    else:
        kw["extra"] = {"strict_res": "unavailable"}
    return TranscribeOptions(**kw)


def load_bundle(class_id, *, raw_dir: Path | str | None = None,
                scribe_dir: Path | str | None = None) -> dict:
    """The `scribe sync` bundle for one class (raw JSON as synced; Scribe needs the roster for
    keyterms), restricted to SCRIBE_BUNDLE_PARTS, plus Brady's id for offline runs."""
    cdir = Path(raw_dir or config.RAW_DIR) / str(class_id)
    if not (cdir / config.RAW_SOURCE_FILES["class"]).exists():
        raise FileNotFoundError(f"no synced class.json for class {class_id} in {cdir}")
    bundle: dict = {"schema": "scribe.bundle/v1", "class_id": int(class_id)}
    for part in SCRIBE_BUNDLE_PARTS:
        path = cdir / config.RAW_SOURCE_FILES[part]
        bundle[part] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    bundle["me"] = {"id": events.load_me_id(scribe_dir)}
    return bundle


def course_code_of(bundle: dict) -> str | None:
    """Course code from class.json (section.course.course-code), else listing.json."""
    k = kebab(bundle.get("class") or {})
    section = k.get("section") if isinstance(k.get("section"), dict) else {}
    course = section.get("course") if isinstance(section.get("course"), dict) else {}
    listing = kebab(bundle.get("listing") or {})
    return course.get("course-code") or (listing.get("course-code")
                                         if isinstance(listing, dict) else None)


def _assert_no_hand_log(kwargs: dict) -> None:
    if kwargs.get("hand_log") is not None:
        raise AssertionError("hand_log must never reach a feature transcript (L26)")


def transcribe_many(class_ids: Iterable, workers: int = 1, provider: str | None = None, *,
                    out_dir: Path | str | None = None, raw_dir: Path | str | None = None,
                    scribe_dir: Path | str | None = None, salt: str | None = None,
                    force: bool = False, require_strict_res: bool = True,
                    provider_map: dict[str, str] | None = None,
                    transcribe_fn: Callable | None = None,
                    client_factory: Callable | None = None,
                    provider_factory: Callable[[str], object] | None = None,
                    workdir_root: Path | None = None,
                    run_log: Path | str | bool | None = None) -> list[dict]:
    """Transcribe classes with Scribe; returns one status row per class.

    * provider: None = per-course map (A8); "local" forces local; "deepgram" only for cleared
      courses (otherwise that class fails with an error row).
    * transcribe_fn / client_factory / provider_factory are injectable so tests stay offline:
      defaults are `scribe_core.pipeline.transcribe_class`, `ForumClient()` and None (Scribe
      builds the provider from `options.provider`). A None client runs Scribe offline on the
      synced bundle (needs a provider that replays saved ASR).
    * Local ASR jobs run one at a time (a lock), Deepgram jobs use up to `workers` threads.
    * Classes whose `*_asr.json` and `*_ids.json` already exist are skipped unless `force`.
    * run_log: CSV of the status rows (default data/logs/transcribe_runs.csv; False = none).
    """
    if transcribe_fn is None:
        from scribe_core.pipeline import transcribe_class as transcribe_fn
    if client_factory is None:
        def client_factory():
            from scribe_core.forum import ForumClient

            return ForumClient()
    out = Path(out_dir or config.TRANSCRIPTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    salt = salt if salt is not None else config.id_salt()
    local_lock = threading.Lock()

    def one(cid) -> dict:
        row = {"class_id": int(cid), "provider": None, "status": "error", "n_words": None,
               "calibration_status": None, "error": None}
        try:
            if not force and find_transcript(cid, "asr", out) and find_transcript(cid, "ids", out):
                row["status"] = "skipped"
                return row
            bundle = load_bundle(cid, raw_dir=raw_dir, scribe_dir=scribe_dir)
            name = provider_for(course_code_of(bundle), provider, provider_map)
            row["provider"] = name
            opts = transcribe_options(name, salt, require_strict_res=require_strict_res,
                                      workdir_root=workdir_root)
            kwargs = {"out_dir": out, "options": opts, "bundle": bundle,
                      "provider": provider_factory(name) if provider_factory else None,
                      "progress": _progress_logger(cid)}
            _assert_no_hand_log(kwargs)
            with local_lock if name == "local" else nullcontext():
                result = transcribe_fn(int(cid), client_factory(), **kwargs)
            summary = (result or {}).get("summary") or {}
            row.update(status="ok", n_words=summary.get("n_words"),
                       calibration_status=summary.get("calibration_status"))
        except Exception as exc:  # noqa: BLE001 - one failed class must not stop the batch
            row["error"] = f"{type(exc).__name__}: {exc}"
            log.error("transcribe class %s failed: %s", cid, row["error"])
        return row

    ids = [int(c) for c in class_ids]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        rows = list(pool.map(one, ids))
    if run_log is not False:
        _append_run_log(rows, Path(run_log) if run_log else config.LOGS_DIR / RUN_LOG_NAME)
    return rows


def _progress_logger(cid):
    def progress(stage: str, message: str, fraction: float | None = None) -> None:
        log.info("class %s [%s] %s", cid, stage, message)
    return progress


def _append_run_log(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["class_id"])
        if new:
            writer.writeheader()
        writer.writerows(rows)


# =========================================================================== reading outputs


def find_transcript(class_id, kind: str = "asr", transcripts_dir: Path | str | None = None
                    ) -> Path | None:
    """`{base}_asr.json` or `{base}_ids.json` for a class (Scribe's base ends in class{id})."""
    root = Path(transcripts_dir or config.TRANSCRIPTS_DIR)
    if not root.exists():
        return None
    rx = re.compile(rf"(?:^|_)class{int(class_id)}_{kind}\.json$")
    hits = sorted(p for p in root.glob(f"*class{int(class_id)}_{kind}.json") if rx.search(p.name))
    return hits[-1] if hits else None


_ASR_WORD_KEYS = ("start", "end", "text", "punctuated_word", "conf", "filler", "cluster",
                  "cluster_conf")


def read_asr(path_or_obj) -> dict:
    """`*_asr.json` restricted to words[] fields and run metadata (whitelist source "asr")."""
    obj = path_or_obj
    if not isinstance(obj, dict):
        obj = json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
    words = []
    for w in obj.get("words") or []:
        if isinstance(w, dict):
            words.append({k: w.get(k, w.get(k.replace("_", "-"))) for k in _ASR_WORD_KEYS})
    meta = {k: obj.get(k, obj.get(k.replace("_", "-")))
            for k in ("provider", "model", "language", "duration", "diarized", "source_res")}
    return {**meta, "words": words}


def read_transcript_meta(path_or_obj) -> dict:
    """The only parts of a Scribe transcript On Cue reads: `alignment` (calibration, n_words,
    timing precision) and `asr` (provider, source_res). Never `events` or `participation`."""
    obj = path_or_obj
    if not isinstance(obj, dict):
        obj = json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
    # explicit key reads (never .items()/iteration) so `events` is provably never touched
    align = obj["alignment"] if "alignment" in obj else {}  # noqa: SIM401
    asr = obj["asr"] if "asr" in obj else {}  # noqa: SIM401
    return {"calibration": align.get("calibration"), "n_words": align.get("n_words"),
            "timing_precision": align.get("forum_timing_precision"),
            "source_res": (asr or {}).get("source_res"), "provider": (asr or {}).get("provider")}


# =========================================================================== calibration, QA


@dataclass
class Calibration:
    cal: dict
    segs: list
    voice_raw: list[dict]       # Scribe parse_voice_events, main room (Forum video time)
    voice: list[dict]           # the same, shifted onto the master clock
    words: list[dict]           # ASR words sorted by (start, end), as the pipeline sorts them
    precision: str | None
    main_room: str | None


def calibrate_class(asr: dict, klass: dict, raw_events: list[dict]) -> Calibration:
    """Mirror of `pipeline.transcribe_bundle`'s alignment step (D7)."""
    klass = kebab(klass or {})
    evs = events.allowed_events(raw_events)
    segs = recording_timeline(klass)
    room, _ = main_room_info(evs)
    voice_raw = parse_voice_events(evs, segs, room_id=room)
    precision = timing_precision(voice_raw)
    words = sorted((w for w in asr["words"] if w.get("start") is not None
                    and w.get("end") is not None), key=lambda w: (w["start"], w["end"]))
    dur = max([asr.get("duration") or 0.0, timeline_duration(segs) or 0.0]
              + [w["end"] for w in words[-1:]])
    cal = calibrate(calibration_words(words, voice_raw), voice_raw, dur or None)
    return Calibration(cal, segs, voice_raw, shift(voice_raw, cal), words, precision, room)


def residuals(voice_starts, voice_ends, word_starts, *,
              search_s: float = config.RESID_SEARCH_S,
              pre_silence_s: float = config.RESID_PRE_SILENCE_S) -> np.ndarray:
    """|calibrated voice start - nearest ASR word start| within ±search_s, over main-room voice
    events preceded by >= pre_silence_s of Forum voice silence (§4.3). Events with no word
    start within the search window are left out (blips)."""
    vs = np.asarray(voice_starts, dtype=float)
    ve = np.asarray(voice_ends, dtype=float)
    ws = np.sort(np.asarray(word_starts, dtype=float))
    if not len(vs) or not len(ws):
        return np.array([])
    order = np.argsort(vs, kind="stable")
    vs, ve = vs[order], ve[order]
    prev_end = np.concatenate([[-np.inf], np.maximum.accumulate(ve)[:-1]])
    quiet = prev_end <= vs - pre_silence_s
    j = np.searchsorted(ws, vs)
    left = ws[np.clip(j - 1, 0, len(ws) - 1)]
    right = ws[np.clip(j, 0, len(ws) - 1)]
    near = np.minimum(np.abs(left - vs), np.abs(right - vs))
    ok = quiet & (near <= search_s)
    return near[ok]


def calibration_qa_row(cal: dict | None, resid: np.ndarray, precision: str | None) -> dict:
    """classes-table timing columns + the headline and sensitivity QA verdicts (§3, §4.3)."""
    cal = cal or {}
    med = float(np.median(resid)) if len(resid) else None
    p95 = float(np.percentile(resid, 95)) if len(resid) else None
    max_res = cal.get("max_residual_s")
    eps = config.eps_f(precision, p95 if p95 is not None else float("nan"),
                       max_res if max_res is not None else float("nan"))
    if not cal.get("applied"):
        eps = max(eps, config.EPS_F_FALLBACK_S)      # D5 fallback: uncalibrated clock
    ok, reason = config.calibration_qa(cal or None, med, p95)
    sens_ok, sens_reason = config.calibration_qa(cal or None, med, p95, sensitivity=True)
    return {
        "cal_status": cal.get("status"), "cal_applied": bool(cal.get("applied", False)),
        "cal_method": cal.get("method"), "cal_n_windows": len(cal.get("windows") or []),
        "cal_max_residual_s": float(max_res) if max_res is not None else np.nan,
        "cal_offset_s": float(cal.get("offset", 0.0) or 0.0),
        "cal_drift": float(cal.get("drift", 0.0) or 0.0),
        "cal_pieces_json": json.dumps(cal.get("pieces") or []),
        "resid_med_s": med if med is not None else np.nan,
        "resid_p95_s": p95 if p95 is not None else np.nan, "resid_n": len(resid),
        "eps_F_s": float(eps), "timing_precision": precision,
        "cal_qa_pass": bool(ok), "cal_qa_reason": reason,
        "cal_qa_sens_pass": bool(sens_ok), "cal_qa_sens_reason": sens_reason,
    }


def class_timing(calib: Calibration, *, class_id: int | None = None) -> dict:
    """Residuals + QA verdicts for one calibrated class."""
    resid = residuals([v["start"] for v in calib.voice], [v["end"] for v in calib.voice],
                      [w["start"] for w in calib.words])
    row = calibration_qa_row(calib.cal, resid, calib.precision)
    return {"class_id": None if class_id is None else int(class_id), **row}


def read_timing(class_id, *, tables_dir: Path | str | None = None) -> dict | None:
    """The `words` stage's timing row for a class, with the calibration rebuilt for the clock."""
    path = Path(tables_dir) / f"{TIMING_TABLE}.parquet" if tables_dir \
        else config.table_path(TIMING_TABLE)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df[df["class_id"] == int(class_id)]
    if not len(df):
        return None
    row = df.iloc[-1].to_dict()
    row["cal"] = {"applied": bool(row["cal_applied"]), "status": row["cal_status"],
                  "method": row["cal_method"], "pieces": json.loads(row["cal_pieces_json"])}
    if pd.isna(row.get("timing_precision")):
        row["timing_precision"] = None
    return row


# =========================================================================== roles, text


def forum_roles(w_start, w_end, v_start, v_end, *, edge: float = EDGE, chunk: int = 512
                ) -> np.ndarray:
    """Index of the voice event with the most overlap with [start - edge, end + edge] for each
    word (-1 when none overlaps). Ties go to the earlier event (events sorted by start)."""
    ws = np.asarray(w_start, dtype=float)
    we = np.maximum(np.asarray(w_end, dtype=float), ws)
    vs = np.asarray(v_start, dtype=float)
    ve = np.asarray(v_end, dtype=float)
    best = np.full(len(ws), -1, dtype=int)
    if not len(ws) or not len(vs):
        return best
    for a in range(0, len(ws), chunk):
        sl = slice(a, a + chunk)
        lo, hi = np.nanmin(ws[sl]) - edge, np.nanmax(we[sl]) + edge
        cand = np.flatnonzero((vs < hi) & (ve > lo))
        if not len(cand):
            continue
        ov = (np.minimum(we[sl, None] + edge, ve[None, cand])
              - np.maximum(ws[sl, None] - edge, vs[None, cand]))
        k = np.argmax(ov, axis=1)
        got = ov[np.arange(ov.shape[0]), k] > 0
        best[sl] = np.where(got, cand[k], -1)
    return best


_NORM_RE = re.compile(r"[^\w\s]")


def norm_token(token: str) -> str:
    """Lowercase, punctuation stripped (apostrophes too; the lexicons tolerate that)."""
    return _NORM_RE.sub("", str(token or "").lower()).strip()


def norm_word(token: str, is_name: bool) -> str:
    """Per-word `text_norm`: "" for a masked name or a one-word CALLON phrase, else the
    normalised token. Never depends on another word (no look-ahead)."""
    if is_name:
        return ""
    tok = norm_token(token)
    return "" if config.CALLON_RE.fullmatch(tok) else tok


# =========================================================================== words


@dataclass
class WordsBuild:
    words: pd.DataFrame
    timing: dict
    calibration: Calibration
    clock: events.VideoClock
    voice: pd.DataFrame          # events.parse_voice rows (raw ids; in memory only)
    roster: events.Roster


def build_words(class_id: int, asr: dict, klass: dict, raw_events: list[dict], *, me_id,
                salt: str | None = None, masker=None, transcript: dict | Path | None = None,
                probe: dict | None = None) -> WordsBuild:
    """The causal words table for one class from `*_asr.json` + raw Forum data (pure).

    `transcript` (Scribe's `*_ids.json`, dict or path) is optional and only used for the D7
    checks (word count, same calibration) through `read_transcript_meta`.
    """
    cid = int(class_id)
    events.require_me_id(me_id)
    salt = salt if salt is not None else config.id_salt()
    asr = read_asr(asr)
    klass_w = events.whitelist_filter("class", klass or {})
    evs = events.allowed_events(raw_events)
    calib = calibrate_class(asr, klass_w, evs)
    timing = class_timing(calib, class_id=cid)
    eps_f = timing["eps_F_s"]
    roster = events.roster_from_class(klass_w, me_id)
    clock = events.video_clock(probe, calib.cal, klass_w)
    voice = events.parse_voice(evs, clock, calib.main_room)
    main = voice[voice["room_main"] & voice["user_id"].notna()].reset_index(drop=True)

    w = calib.words
    ws = np.array([float(x["start"]) for x in w], dtype=float)
    we = np.array([float(x["end"]) for x in w], dtype=float)
    best = forum_roles(ws, we, main["start_s"].to_numpy(float), main["end_s"].to_numpy(float))
    has = best >= 0
    m_uid = np.append(main["user_id"].to_numpy(object), None)       # index -1 -> None
    m_start = np.append(main["start_s"].to_numpy(float), np.nan)
    uid = m_uid[best]
    ev_start = m_start[best]
    avail = np.maximum(we + config.EPS_W_S, np.where(has, ev_start + eps_f, -np.inf))
    # A12: Forum logs students' mics only, so a word heard while no main-room mic is on is the
    # professor's. That is known once a mic event covering the word would have shown up
    # (start + eps_F, start <= end + EDGE). No main-room events at all: nothing to go on.
    quiet = ~has & (len(main) > 0)
    uid = np.where(quiet, roster.teacher, uid)
    avail = np.where(quiet, np.maximum(we + config.EPS_W_S, we + EDGE + eps_f), avail)
    role = np.array([roster.role(u) for u in uid], dtype=object)

    # Scribe's fused attribution, kept only for A-attrib (never a feature).
    teacher = roster.teacher if len(calib.voice) else None
    mapping = (cluster_to_user(w, calib.voice, teacher=teacher) if asr.get("diarized")
               else {})
    fused = smooth(fuse(w, calib.voice, mapping, teacher=teacher))
    role_fused = [roster.role(f.get("user_id")) for f in fused]

    tokens = [str(x.get("punctuated_word") or x.get("text") or "") for x in w]
    # sentence-initial: first word, or after . ? ! Not at a speaker change: turn edges are
    # off by a fraction of a second, and a call-on name glued to the next turn ("... your
    # hand, Mary") would read as initial and escape masking. Punctuation alone over-masks.
    initial = np.ones(len(w), dtype=bool)
    initial[1:] = np.array([bool(events._SENT_END_RE.search(t)) for t in tokens[:-1]],
                           dtype=bool)
    masker = events.masker_for(roster, masker)
    masked, kinds = masker.mask_tokens(tokens, initial)
    norm = [norm_word(t, bool(k)) for t, k in zip(tokens, kinds, strict=True)]

    hashes = {u: events.spk_hash(u, salt) for u in set(uid) if u is not None}
    df = pd.DataFrame({
        "class_id": cid, "w_idx": np.arange(len(w)), "start_s": ws, "end_s": we,
        "text_masked": masked, "text_norm": norm,
        "has_qmark": [bool(re.search(r"\?[\"'”’)\]]*$", t)) for t in tokens],
        "conf": [np.nan if x.get("conf") is None else float(x["conf"]) for x in w],
        "filler": [bool(x.get("filler")) for x in w],
        "spk_hash": [hashes.get(u) for u in uid], "role": role,
        "role_event_start_s": ev_start, "role_avail_s": avail,
        "role_fused": role_fused, "how": [f.get("how") for f in fused],
    })
    df = schemas.coerce(df, "words")
    tmeta = read_transcript_meta(transcript) if transcript is not None else {}
    timing.update(n_words=len(df), n_words_transcript=tmeta.get("n_words"),
                  src_res=tmeta.get("source_res") or asr.get("source_res"),
                  asr_provider=asr.get("provider"),
                  cal_matches_transcript=_same_cal(calib.cal, tmeta.get("calibration")))
    if timing["n_words_transcript"] is not None and timing["n_words_transcript"] != len(df):
        log.warning("class %s: %d words rebuilt vs %s in the transcript (D7)", cid, len(df),
                    timing["n_words_transcript"])
    return WordsBuild(df, timing, calib, clock, voice, roster)


def _same_cal(a: dict | None, b: dict | None) -> bool | None:
    if not a or not b:
        return None
    keys = ("status", "method", "applied")
    if any(a.get(k) != b.get(k) for k in keys):
        return False
    return abs(float(a.get("offset", 0)) - float(b.get("offset", 0))) < 1e-6 and \
        abs(float(a.get("drift", 0)) - float(b.get("drift", 0))) < 1e-9


def load_words_build(class_id, *, raw_dir=None, transcripts_dir=None, scribe_dir=None,
                     salt: str | None = None, masker=None) -> WordsBuild:
    """`build_words` from disk: data/scribe/raw/{cid} + data/transcripts/*class{cid}_asr.json."""
    cid = int(class_id)
    asr_path = find_transcript(cid, "asr", transcripts_dir)
    if asr_path is None:
        raise FileNotFoundError(f"no *_asr.json for class {cid}; run `oncue transcribe` first")
    ids_path = find_transcript(cid, "ids", transcripts_dir)
    klass = events.read_raw(cid, "class", raw_dir)
    if klass is None:
        raise FileNotFoundError(f"no class.json for class {cid}")
    return build_words(cid, read_asr(asr_path), klass, events.read_raw(cid, "events", raw_dir)
                       or [], me_id=events.load_me_id(scribe_dir), salt=salt, masker=masker,
                       transcript=ids_path)


def words_table(class_id, **kwargs) -> pd.DataFrame:
    """Spec `words_table(cid)`: the words table of one class (see `load_words_build`)."""
    return load_words_build(class_id, **kwargs).words


def timing_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["class_id"] = df["class_id"].astype("int64")
    for c in ("cal_status", "cal_method", "timing_precision", "src_res", "asr_provider",
              "cal_qa_reason", "cal_qa_sens_reason"):
        if c in df:
            df[c] = df[c].astype("string")
    for c in ("n_words_transcript",):
        if c in df:
            df[c] = df[c].astype("Int64")
    if "cal_matches_transcript" in df:
        df["cal_matches_transcript"] = df["cal_matches_transcript"].astype("boolean")
    return df


# =========================================================================== CLI


def _class_ids(args) -> list[int]:
    if getattr(args, "class_ids", None) or getattr(args, "phase", None):
        return events._class_ids(args)
    root = config.TRANSCRIPTS_DIR
    ids = set()
    for p in root.glob("*_asr.json") if root.exists() else []:
        m = re.search(r"class(\d+)_asr\.json$", p.name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def cli_transcribe(args) -> int:
    """`python -m oncue transcribe --class CID ... [--provider local|deepgram] [--force]
    [--allow-no-strict-res] [--workers N]`."""
    p = argparse.ArgumentParser(prog="oncue transcribe")
    p.add_argument("--provider", choices=config.ASR_PROVIDERS, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-no-strict-res", action="store_true")
    opts = p.parse_args(args.extra)
    if not (getattr(args, "class_ids", None) or getattr(args, "phase", None)):
        print("oncue transcribe: pass --class CID (repeatable) or --phase P")
        return 2
    config.load_env()
    rows = transcribe_many(events._class_ids(args), args.workers, opts.provider,
                           force=opts.force, require_strict_res=not opts.allow_no_strict_res)
    bad = [r for r in rows if r["status"] == "error"]
    for r in bad:
        print(f"class {r['class_id']}: {r['error']}")
    print(f"oncue transcribe: {len(rows) - len(bad)} ok/skipped, {len(bad)} failed")
    return 0 if not bad else 1


def cli_words(args) -> int:
    """`python -m oncue words [--class CID ...]` -> words.parquet + class_timing.parquet."""
    config.load_env()
    ids = _class_ids(args)
    parts, timings, failed = [], [], []
    for cid in ids:
        try:
            b = load_words_build(cid)
        except Exception as exc:  # noqa: BLE001 - one bad class must not stop the batch
            log.error("class %s: %s", cid, exc)
            failed.append(cid)
            continue
        parts.append(b.words)
        timings.append(b.timing)
    done = [c for c in ids if c not in failed]
    if parts:
        events.write_table("words", pd.concat(parts, ignore_index=True), done)
        events.write_table(TIMING_TABLE, timing_frame(timings), done)
    n_pass = sum(bool(t["cal_qa_pass"]) for t in timings)
    print(f"oncue words: {len(done)} classes ({n_pass} pass calibration QA), "
          f"{len(failed)} failed")
    return 0 if not failed and ids else 1
