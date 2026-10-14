"""Class-level QA (§3 "class QA", §2.1): decided before modelling, never a feature.

`oncue qa` merges what each stage measured about a class into `classes.parquet` and sets
`qa_pass` / `qa_reason`:

* `class_timing` (`oncue words`): calibration, residuals, eps_F, timing precision, the
  rendition the ASR actually used (`src_res`) and the ASR provider;
* `hand_qc` (`oncue track` / `episodes`): strip, tile status, coverage, baseline, A/V lag,
  bid / vote / contested counts;
* `pcw_status` (`oncue pcw`): the PCW status and template caveat.

The rules (§3, §2.1, §4.3). Every failing rule is listed in `qa_reason`, joined by "+", in
this order: rules no sensitivity population relaxes first, then `multi_session`, then the
calibration reason. So a reason starting with "cal_"/"resid_" means calibration was the
only failure (the cal-sensitivity population may re-admit it), and exactly
"multi_session" means the multi-session rule was (experiments._qa_override).

1. `no_asr`: no words-stage row, or no words;
2. `rendition` / `rendition_unknown`: the ASR rendition is not the one the scan reads
   (`config.SRC_RES`), or was not recorded;
3. `not_scanned` / `tile_<status>`: no detector QA, or `tile_status` not "located";
4. `av_lag` / `av_lag_missing`: |lag_c - median lag| > 0.5 s (median over the classes in
   the table with a measured lag), or no lag;
5. `multi_session`: more than one recording session, unless `multi_session_ok` (D14) has
   been set True by hand; it is never inferred;
6. calibration: `config.calibration_qa` (windowed, >= 6 windows, max residual, p95, median).

The median-lag sanity range (§2.1, [-0.25, 1.0] s) is reported, not enforced per class.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, schemas

log = logging.getLogger("oncue.qa")

TIMING_TABLE = "class_timing"          # transcribe.TIMING_TABLE
HAND_QC_TABLE = "hand_qc"
PCW_STATUS_TABLE = "pcw_status"        # pcw.STATUS_TABLE

# classes column <- source column, per source table
TIMING_COLUMNS = {c: c for c in (
    "cal_status", "cal_applied", "cal_method", "cal_n_windows", "cal_max_residual_s",
    "resid_med_s", "resid_p95_s", "eps_F_s", "timing_precision", "asr_provider")}
HAND_COLUMNS = {c: c for c in (
    "strip_y", "strip_h", "strip_found", "template_validated", "coverage", "tile_status",
    "down_baseline", "flagged", "av_lag_s", "av_lag_brady_s", "n_bids", "n_votes",
    "n_contested",
    # diagnostics only (never QA rules): A1 re-timing, name lookalikes, flagged-class audit
    "retime_failed_share", "g1_approximate", "n_lookalike_frames", "n_prebaseline_runs")}
PCW_COLUMNS = {"pcw_status": "pcw_status", "template_asof": "template_asof"}


def _merge(classes: pd.DataFrame, src: pd.DataFrame | None, columns: dict[str, str]
           ) -> pd.DataFrame:
    """Overwrite `columns` of classes from `src` (one row per class_id) where src has a row."""
    if src is None or not len(src):
        return classes
    src = src.drop_duplicates("class_id", keep="last").set_index("class_id")
    out = classes.copy()
    cid = out["class_id"].astype("int64")
    has = cid.isin(src.index)
    for dst, col in columns.items():
        if col not in src.columns:
            continue
        vals = cid.map(src[col])
        out[dst] = vals.where(has, out[dst]) if dst in out.columns else vals
    return out


def _cast(s: pd.Series, dtype: str) -> pd.Series:
    """Cast to a schema dtype, tolerating pd.NA / None in float columns."""
    if dtype.startswith("float"):
        obj = s.astype(object)
        return pd.to_numeric(obj.where(obj.notna(), np.nan), errors="coerce").astype(dtype)
    if dtype in ("any", "bytes"):
        return s
    return s.astype(object).where(s.notna(), None).astype(dtype)


def _coerce(classes: pd.DataFrame) -> pd.DataFrame:
    out = classes.copy()
    for c in schemas.schema("classes"):
        if c.name in out.columns:
            out[c.name] = _cast(out[c.name], c.dtype)
    return out


def median_lag(av_lag: pd.Series) -> float:
    lag = pd.to_numeric(av_lag, errors="coerce")
    lag = lag[np.isfinite(lag)]
    return float(lag.median()) if len(lag) else float("nan")


def class_reasons(row: dict, med_lag: float, *, timing: dict | None) -> list[str]:
    """Every failing QA rule for one merged classes row (see the module docstring)."""
    hard: list[str] = []
    n_words = (timing or {}).get("n_words")
    if timing is None or n_words is None or pd.isna(n_words) or not n_words:
        hard.append("no_asr")
    src = (timing or {}).get("src_res")
    if timing is not None:
        if src is None or pd.isna(src):
            hard.append("rendition_unknown")
        elif str(src) != config.SRC_RES:
            hard.append("rendition")
    status = row.get("tile_status")
    if status is None or pd.isna(status):
        hard.append("not_scanned")
    elif status != "located":
        hard.append(f"tile_{status}")
    lag = row.get("av_lag_s")
    if status is not None and not pd.isna(status):
        if lag is None or pd.isna(lag) or not np.isfinite(float(lag)):
            hard.append("av_lag_missing")
        elif not np.isfinite(med_lag) or not config.av_lag_ok(float(lag), med_lag):
            hard.append("av_lag")
    soft: list[str] = []
    ms_ok = row.get("multi_session_ok")
    ms_ok = False if ms_ok is None or pd.isna(ms_ok) else bool(ms_ok)
    if int(row.get("n_rec_sessions") or 0) > 1 and not ms_ok:
        soft.append("multi_session")
    if timing is not None:
        cal_reason = timing.get("cal_qa_reason")
        if not bool(timing.get("cal_qa_pass")):
            soft.append(str(cal_reason or "no_calibration"))
    return hard + soft


def apply(classes: pd.DataFrame, timing: pd.DataFrame | None = None,
          hand_qc: pd.DataFrame | None = None, pcw_status: pd.DataFrame | None = None
          ) -> pd.DataFrame:
    """classes with the stage columns merged and av_ok, multi_session_ok, qa_pass and
    qa_reason decided. Pure: no disk access."""
    out = _merge(classes, timing, TIMING_COLUMNS)
    if timing is not None and len(timing) and "src_res" in timing:
        known = timing.dropna(subset=["src_res"])
        out = _merge(out, known, {"src_res": "src_res"})        # the rendition ASR used
    out = _merge(out, hand_qc, HAND_COLUMNS)
    out = _merge(out, pcw_status, PCW_COLUMNS)
    med = median_lag(out["av_lag_s"]) if "av_lag_s" in out else float("nan")
    if np.isfinite(med) and not (config.AV_LAG_MEDIAN_RANGE_S[0] <= med
                                 <= config.AV_LAG_MEDIAN_RANGE_S[1]):
        log.warning("median A/V lag %.3f s is outside %s: stop and investigate (§2.1)",
                    med, config.AV_LAG_MEDIAN_RANGE_S)
    tim = {} if timing is None or not len(timing) else \
        timing.drop_duplicates("class_id", keep="last").set_index("class_id").to_dict("index")
    out = _coerce(out)
    multi = out["n_rec_sessions"].fillna(0).astype(int) > 1
    prior_ok = out["multi_session_ok"].astype("boolean") if "multi_session_ok" in out else \
        pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["multi_session_ok"] = prior_ok.fillna(pd.Series(~multi, index=out.index))
    lag = out["av_lag_s"].astype(float) if "av_lag_s" in out else \
        pd.Series(np.nan, index=out.index)
    av_ok = pd.Series((lag - med).abs() <= config.AV_LAG_TOL_S, dtype="boolean")
    out["av_ok"] = av_ok.mask(lag.isna())
    passes, reasons = [], []
    for row in out.to_dict("records"):
        why = class_reasons(row, med, timing=tim.get(int(row["class_id"])))
        passes.append(not why)
        reasons.append("+".join(why) if why else "ok")
    out["qa_pass"] = pd.array(passes, dtype="boolean")
    out["qa_reason"] = pd.array(reasons, dtype="string")
    return _coerce(out)


def summary(classes: pd.DataFrame) -> dict:
    """Counts only (no ids): QA pass / fail by first reason, dev vs vault."""
    first = classes["qa_reason"].astype(str).str.split("+").str[0]
    return {"classes": len(classes), "qa_pass": int(classes["qa_pass"].sum()),
            "qa_pass_dev": int((classes["qa_pass"] & ~classes["in_vault"]).sum()),
            "fail_by_first_reason": first[~classes["qa_pass"].astype(bool)]
            .value_counts().to_dict()}


CAL_REASON_PREFIXES = ("cal_", "resid_", "no_calibration")


def _reason_tokens(classes: pd.DataFrame) -> pd.Series:
    return classes["qa_reason"].astype(object).fillna("").astype(str).str.split("+")


def funnel_table(classes: pd.DataFrame, points: pd.DataFrame | None = None) -> pd.DataFrame:
    """The data funnel (notebook §1; `plots.fig_funnel`): cumulative rows synced -> recorded
    -> selected -> transcribed -> calibrated -> A/V lag ok -> scanned -> located -> QA pass,
    then dev / vault, with classes, bids (classes.n_bids) and at-risk points per row.
    Counts only, no ids."""
    c = classes.reset_index(drop=True)
    tokens = _reason_tokens(c) if "qa_reason" in c else pd.Series([[]] * len(c))

    def has(col: str) -> pd.Series:
        return c[col].notna() if col in c else pd.Series(False, index=c.index)
    cal_ok = ~tokens.map(lambda ts: any(t.startswith(CAL_REASON_PREFIXES) for t in ts))
    steps = [("synced", pd.Series(True, index=c.index)),
             ("recorded", c["n_rec_sessions"].fillna(0).astype(int) > 0),
             ("selected", has("phase")),
             ("transcribed", has("cal_status")),
             ("calibrated", cal_ok & has("cal_status")),
             ("av_lag_ok", c["av_ok"].fillna(False).astype(bool) if "av_ok" in c
              else pd.Series(False, index=c.index)),
             ("scanned", has("tile_status")),
             ("located", c.get("tile_status", pd.Series(index=c.index, dtype=object))
              .astype(object).eq("located")),
             ("qa_pass", c["qa_pass"].fillna(False).astype(bool) if "qa_pass" in c
              else pd.Series(False, index=c.index))]
    bids = pd.to_numeric(c.get("n_bids", pd.Series(0, index=c.index)), errors="coerce") \
        .fillna(0)
    risk = points.groupby("class_id")["at_risk"].sum() if points is not None and len(points) \
        else None
    rows, keep = [], pd.Series(True, index=c.index)
    vault = c["in_vault"].fillna(False).astype(bool)
    for name, mask in steps + [("dev", ~vault), ("vault", vault)]:
        m = (keep & mask.to_numpy()) if name not in ("dev", "vault") else (keep & mask)
        if name not in ("dev", "vault"):
            keep = m
        ids = c.loc[m, "class_id"]
        rows.append({"stage": name, "classes": int(m.sum()), "bids": int(bids[m].sum()),
                     "points_at_risk": (int(ids.map(risk).fillna(0).sum())
                                        if risk is not None else pd.NA)})
    return pd.DataFrame(rows).astype({"points_at_risk": "Int64"})


def _read(name: str) -> pd.DataFrame | None:
    path = config.table_path(name)
    return pd.read_parquet(path) if path.exists() else None


def cli(args) -> int:
    """`python -m oncue qa`: merge class_timing, hand_qc and pcw_status into classes.parquet
    and decide qa_pass / qa_reason. Run after `episodes` (and `pcw`), before `labels`."""
    classes = _read("classes")
    if classes is None:
        print("oncue qa: data/tables/classes.parquet is missing (run `oncue select`)")
        return 1
    out = apply(classes, _read(TIMING_TABLE), _read(HAND_QC_TABLE), _read(PCW_STATUS_TABLE))
    schemas.validate(out, "classes")
    out.to_parquet(config.table_path("classes"), index=False)
    if config.SPLIT_PATH.exists():                 # the step-3 split's classes (A11)
        from .splits import split_ids

        sel, what = out[out["class_id"].isin(split_ids("all"))], "split"
    else:
        sel, what = (out[out["phase"].notna()] if "phase" in out else out), "selected"
    s = summary(sel)
    print(f"oncue qa: {s['qa_pass']}/{s['classes']} {what} classes pass "
          f"({s['qa_pass_dev']} dev); failures by first reason {s['fail_by_first_reason']}")
    return 0
