"""Does everything line up on one clock? `python -m oncue ingest-report [--work-dir DIR]`.

One row per ingested class (`{logs}/ingest_report.csv`), five checks:

A. transcript vs Forum: the calibration that maps Forum voice events onto the ASR words
   (method, windows, offset, drift, residual median / p95, QA verdict);
B. video vs transcript: the A/V lag, i.e. the teal speaker ring on the recording against
   the ASR words (`av_lag_s`, all tiles; `av_lag_brady_s`, Brady's tile vs his words), and
   the ring on Brady's tile against his own Forum voice events (`ring_vs_forum_s`);
C. the cut of the video: the recording's length (ffprobe), the Forum recording session's
   length and the ASR audio's length must agree;
D. raise -> speech: for raises that were called, the gap from the (re-timed) onset to
   Brady's next main-room speech, and its trend over the class (a drift means a clock
   problem);
E. URL vs upload: the ASR mode, to compare offsets between the two.

Only numbers: no text, no names, no other person's ids.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config, hands, transcribe

log = logging.getLogger(__name__)


def _read(name: str) -> pd.DataFrame | None:
    p = config.table_path(name)
    return pd.read_parquet(p) if p.exists() else None


def _asr_meta(cid: int) -> dict:
    """ASR mode and duration from the ids transcript (`asr` block only)."""
    p = transcribe.find_transcript(cid, "ids")
    if p is None:
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    asr = obj["asr"] if "asr" in obj else {}  # noqa: SIM401 - explicit key read only
    return {"asr_mode": asr.get("mode"), "asr_duration_s": asr.get("duration")}


def raise_to_speech(eps: pd.DataFrame, voice: pd.DataFrame, *, max_gap_s: float = 30.0
                    ) -> dict:
    """Gap from each called bid's onset to Brady's next main-room voice event start."""
    if eps is None or not len(eps) or voice is None or not len(voice):
        return {"n_called": 0, "gap_med_s": np.nan, "gap_iqr_s": np.nan,
                "gap_trend_s_per_h": np.nan}
    me = voice[(voice["role"] == config.ROLE_ME) & voice["room_main"].astype(bool)]
    starts = np.sort(me["start_s"].to_numpy(float))
    bids = eps[(eps["ep_type"] == config.EP_BID) & (eps["outcome"] == "called")]
    t, gaps = [], []
    for ep in bids.itertuples():
        o = float(ep.onset_retimed_s) if pd.notna(ep.onset_retimed_s) else float(ep.onset_s)
        j = np.searchsorted(starts, o)
        if j < len(starts) and starts[j] - o <= max_gap_s:
            t.append(o)
            gaps.append(starts[j] - o)
    g = np.array(gaps)
    trend = (float(np.polyfit(np.array(t) / 3600.0, g, 1)[0]) if len(g) >= 4 else np.nan)
    return {"n_called": len(g), "gap_med_s": float(np.median(g)) if len(g) else np.nan,
            "gap_iqr_s": float(np.subtract(*np.percentile(g, [75, 25]))) if len(g) else np.nan,
            "gap_trend_s_per_h": trend}


def ring_vs_forum(hf: pd.DataFrame, ht: pd.DataFrame, voice: pd.DataFrame) -> float:
    """Lag of the ring on Brady's tile behind his own Forum voice events (s)."""
    me = voice[(voice["role"] == config.ROLE_ME) & voice["room_main"].astype(bool)]
    if not len(me):
        return float("nan")
    border = hands.me_border(hf, ht)
    return hands.av_lag(hf, me[["start_s", "end_s"]], border=border.to_numpy(bool),
                        min_speech_s=config.AV_BRADY_MIN_SPEECH_S)


def report(class_ids) -> pd.DataFrame:
    classes = _read("classes")
    timing = _read(transcribe.TIMING_TABLE)
    qc = _read("hand_qc")
    eps_all = _read("hand_episodes")
    voice_all = _read("voice")
    hf_all = _read("hand_frames")
    rows = []
    for cid in (int(c) for c in class_ids):
        row: dict = {"class_id": cid}
        c = classes[classes["class_id"] == cid] if classes is not None else None
        if c is not None and len(c):
            r = c.iloc[0]
            row.update(course=r["course_code"], term=r["term"], split=r.get("split"),
                       forum_dur_s=float(r["T_end_s"]))
        row.update(_asr_meta(cid))
        info = hands._pass1_json(cid)
        media = info.get("media") or {}
        row["video_dur_s"] = media.get("duration")
        row["video_a_start_s"] = media.get("a_start")
        if timing is not None and (timing["class_id"] == cid).any():
            t = timing[timing["class_id"] == cid].iloc[-1]
            for k in ("cal_status", "cal_method", "cal_n_windows", "cal_offset_s", "cal_drift",
                      "resid_med_s", "resid_p95_s", "timing_precision", "cal_qa_pass",
                      "cal_qa_reason", "n_words"):
                row[k] = t.get(k)
        if qc is not None and (qc["class_id"] == cid).any():
            q = qc[qc["class_id"] == cid].iloc[-1]
            for k in ("av_lag_s", "av_lag_brady_s", "coverage", "tile_status",
                      "track_valid_share", "n_bids", "n_votes", "retime_failed_share"):
                row[k] = q.get(k)
        row["band_share"] = info.get("band_share")
        row["n_anchor_frames"] = info.get("n_anchor_frames")
        voice = voice_all[voice_all["class_id"] == cid] if voice_all is not None else None
        eps = eps_all[eps_all["class_id"] == cid] if eps_all is not None else None
        row.update(raise_to_speech(eps, voice))
        hf = hf_all[hf_all["class_id"] == cid] if hf_all is not None else None
        htp = hands.hand_tiles_path(cid)
        if hf is not None and len(hf) and htp.exists() and voice is not None:
            row["ring_vs_forum_s"] = ring_vs_forum(hf.reset_index(drop=True),
                                                   pd.read_parquet(htp), voice)
        rows.append(row)
    df = pd.DataFrame(rows)
    if {"video_dur_s", "forum_dur_s"} <= set(df):
        df["video_minus_forum_s"] = pd.to_numeric(df["video_dur_s"]) - df["forum_dur_s"]
    if {"video_dur_s", "asr_duration_s"} <= set(df):
        df["video_minus_asr_s"] = (pd.to_numeric(df["video_dur_s"])
                                   - pd.to_numeric(df["asr_duration_s"]))
    return df


def cli(args) -> int:
    """`python -m oncue ingest-report [--class CID ... | --split S] [--work-dir DIR]`: the
    classes of the last ingest run by default."""
    from . import events

    if getattr(args, "class_ids", None) or getattr(args, "split", None):
        ids = events._class_ids(args)
    else:
        p = config.LOGS_DIR / "ingest_classes.csv"
        if not p.exists():
            print("oncue ingest-report: no ingest_classes.csv; pass --class or --split")
            return 1
        ids = pd.read_csv(p)["class_id"].tolist()
    df = report(ids)
    out = config.LOGS_DIR / "ingest_report.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(df.round(3).to_string(index=False))
    print(f"wrote {out}")
    return 0
