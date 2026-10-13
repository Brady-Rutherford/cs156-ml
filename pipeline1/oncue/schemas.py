"""Column lists and dtypes for every On Cue table (spec §13, plus amendments A1-A9).

Tables live in `data/tables/{name}.parquet` (local, gitignored); `oof` and `results` also go
to the committed `results/` bundle with HMAC class ids. Times are master-clock seconds (the
media seconds of the rendition the ASR used). User ids are always hashed (`spk_hash`).

Conventions:
* label columns `y_*` are float32: 1.0 positive, 0.0 negative, NaN undefined;
* "Int64" / "boolean" are pandas nullable dtypes, for values a later stage fills in;
* `class_id` is the Forum class id (int64) locally; committed tables use the 12-hex HMAC.

Usage::

    from oncue import schemas
    schemas.validate(df, "words")            # raises SchemaError on missing columns
    df = schemas.empty("hand_episodes")      # typed, zero rows
    df = schemas.coerce(df, "voice")         # cast present columns to the schema dtypes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


class SchemaError(ValueError):
    """A table does not match its schema."""


@dataclass(frozen=True)
class Col:
    name: str
    dtype: str            # pandas dtype; "any" = unchecked, "bytes" = object holding bytes
    required: bool = True
    note: str = ""


def _cols(spec: str, dtype: str, *, required: bool = True) -> list[Col]:
    return [Col(n, dtype, required) for n in spec.split()]


_F, _I, _S, _B = "float64", "int64", "string", "bool"
_NI, _NB = "Int64", "boolean"   # nullable int / bool
_F32 = "float32"

TABLES: dict[str, tuple[Col, ...]] = {
    "classes": tuple(
        [Col("class_id", _I), Col("section_id", _I)]
        + _cols("course_code course_area term", _S)
        + _cols("semester_idx session_no", _NI)
        + [Col("date", _S, note="YYYY-MM-DD"), Col("weekday", _NI, note="0 = Monday")]
        + _cols("hour_local hour_utc", _F)
        + [Col("sched_start_utc", _S, note="ISO-8601"), Col("rec_offset_s", _F),
           Col("rec_segments_json", _S, note="[[wall_start_ms, wall_end_ms, video_start_s]]"),
           Col("n_rec_sessions", _I), Col("T_end_s", _F), Col("prof_hash", _S),
           Col("enrolled", _NI), Col("n_my_voice_main", _NI), Col("phase", _S),
           Col("in_vault", _B), Col("src_res", _S)]
        + _cols("a_start_s v_start_s url_lifetime_s", _F)
        + [Col("cal_status", _S), Col("cal_applied", _NB), Col("cal_method", _S),
           Col("cal_n_windows", _NI)]
        + _cols("cal_max_residual_s resid_med_s resid_p95_s eps_F_s", _F)
        + [Col("timing_precision", _S, note='"ms" | "s"')]
        + _cols("av_lag_s av_lag_brady_s", _F)
        + _cols("av_ok multi_session_ok", _NB)
        + [Col("era", _S), Col("strip_y", _F), Col("strip_h", _F), Col("strip_found", _NB),
           Col("template_validated", _NB), Col("coverage", _F),
           Col("tile_status", _S, note="|".join(config.TILE_STATUSES)),
           Col("down_baseline", _F), Col("flagged", _NB),
           Col("pcw_status", _S, note="|".join(config.PCW_STATUSES)),
           Col("template_asof", _S)]
        + _cols("n_bids n_votes n_contested", _NI)
        + [Col("qa_pass", _NB), Col("qa_reason", _S),
           Col("asr_provider", _S, required=False, note="A8: deepgram | local"),
           Col("n_sessions", _NI, required=False,
               note="size of the section's full schedule (ctx_session_frac)"),
           Col("run_order", _NI, required=False, note="select.assign_phases processing order"),
           Col("cv_group", _NI, required=False,
               note="L18: same-section same-date twins share one CV group (select.dedupe)"),
           Col("split", _S, required=False,
               note="step 3 (split.csv): train | validation | test | excluded")]
    ),
    "words": (
        Col("class_id", _I), Col("w_idx", _I), Col("start_s", _F), Col("end_s", _F),
        Col("text_masked", _S, note="cased, punctuated, [NAME]-masked; diagnostics only"),
        Col("text_norm", _S, note="lowercase, no punctuation, [name] and one-word CALLON "
            "deleted; multi-word CALLON phrases are removed from joined available text"),
        Col("has_qmark", _B, note="F-punct only"), Col("conf", _F), Col("filler", _B),
        Col("spk_hash", _S), Col("role", _S, note="prof|me|student|unk (Forum-causal)"),
        Col("role_event_start_s", _F), Col("role_avail_s", _F),
        Col("role_fused", _S, note="Scribe fuse() role; A-attrib only"),
        Col("how", _S, note="Scribe fuse() how; never a feature"),
    ),
    "frames": tuple(
        [Col("class_id", _I), Col("f_idx", _I)]
        + _cols("start_s end_s close_s", _F)
        + [Col("role", _S), Col("spk_hash", _S), Col("text_norm", _S), Col("n_words", _I),
           Col("emb_row", _NI, note="row in the embedding cache matrix")]
    ),
    "voice": (
        Col("class_id", _I), Col("spk_hash", _S), Col("role", _S),
        Col("start_s", _F, note="calibrated; start = timestamp - duration"),
        Col("end_s", _F, note="usable only once end <= t - eps_F"),
        Col("room_main", _B), Col("precise", _B), Col("word_bearing", _B),
        Col("first_word_avail_s", _F),
        Col("inferred", _B, False, note="professor turn built from words (A12); Forum never logs them"),
        Col("start_avail_s", _F, False, note="inferred rows only; Forum rows: start + eps_F"),
        Col("end_avail_s", _F, False, note="inferred rows only; Forum rows: end + eps_F"),
    ),
    "segments": (
        Col("class_id", _I), Col("seg_idx", _I), Col("start_s", _F),
        Col("title_masked", _S), Col("section_masked", _S),
        Col("seg_type", _S, note="|".join(config.SEG_TYPES) + "; NO end column"),
    ),
    "chat": (
        Col("class_id", _I), Col("t_s", _F), Col("spk_hash", _S), Col("role", _S),
        Col("text_masked", _S),
    ),
    "polls": (
        Col("class_id", _I), Col("poll_id", _S), Col("open_s", _F), Col("close_s", _F),
        Col("prompt_masked", _S), Col("my_answer_text", _S), Col("my_answer_avail_s", _F),
    ),
    "presence": (
        Col("class_id", _I), Col("spk_hash", _S), Col("role", _S),
        Col("enter_s", _F, note="enter-class-time on the master clock; avail + eps_F"),
    ),
    "pcw_blocks": (
        Col("class_id", _I), Col("block_uuid", _S), Col("block_order", _I),
        Col("prompt_text", _S), Col("answer_text", _S), Col("answer_words", _I),
        Col("has_code", _B), Col("n_files", _I),
    ),
    "hand_tiles": tuple(
        [Col("class_id", _I), Col("pts_s", _F), Col("tile_idx", _I), Col("x0", _I),
         Col("x1", _I)]
        + _cols("frac_hand_s1 frac_hand_s2 frac_tick_s1 frac_tick_s2 shape_s1 shape_s2 "
                "border ncc_me", _F32)
        + [Col("glyph_bits", "bytes", note="packed 12 x 48 binary glyph patch (72 bytes)")]
        + _cols("white_s1 white_s2", _F32, required=False)
    ),
    "hand_frames": (
        Col("class_id", _I), Col("t_s", _F), Col("n_tiles", _I), Col("tile_found", _B),
        Col("tile_idx", _NI), Col("ncc", _F32), Col("ncc_gap", _F32),
        Col("frac_hand_adj", _F32), Col("shape_ok", _B), Col("frac_tick", _F32),
        Col("n_others", _I), Col("border_any", _B),
        Col("raw_state", _S, note="UP|DOWN|NA; whole-class baseline (label side)"),
        Col("live_state", _S, note="UP|DOWN|NA; confirmed raw_state_live (causal)"),
        Col("raw_state_live", _S, required=False,
            note="UP|DOWN|NA on a running DOWN baseline of samples <= t (causal rules)"),
    ),
    "hand_episodes": (
        Col("class_id", _I), Col("ep_id", _I), Col("onset_s", _F),
        Col("onset_retimed_s", _F), Col("retime_failed", _NB), Col("offset_s", _F),
        Col("offset_live_s", _F), Col("dur_s", _F), Col("n_toggles", _I),
        Col("ep_type", _S, note="bid|vote|glitch"), Col("delta_n_others", _NI),
        Col("vote_lex", _B), Col("contested", _B), Col("left_censored", _B),
        Col("me_hand_eligible", _B), Col("outcome", _S, note="called|withdrawn|open"),
        Col("call_latency_s", _F), Col("seg_type", _S),
    ),
    "hand_log": (
        Col("class_id", _I), Col("press_wall_ms", _I), Col("press_s", _F),
        Col("matched_onset_s", _F), Col("lag_s", _F),
    ),
    "speech_onsets": tuple(
        [Col("class_id", _I)]
        + _cols("onset_s turn_end_s silence_before_s", _F)
        + [Col("fresh", _B), Col("hand_preceded", _NB)]
    ),
    "points": tuple(
        [Col("class_id", _I), Col("t_s", _F), Col("in_G", _B), Col("at_risk", _B),
         Col("excl_reason", _S, note="first rule that fires; <NA> when at risk"),
         Col("label_state", _S, note="|".join(config.LABEL_STATES) + " for the headline")]
        + [Col(c, _F32, note="1 / 0 / NaN = undefined") for c in config.LABEL_COLUMNS]
        + [Col("next_bid_s", _F, note="eval only"), Col("fold", _NI), Col("in_vault", _B)]
    ),
    "features": (
        Col("class_id", _I), Col("t_s", _F), Col("emb_row", _NI),
        # every other column must pass config.feature_allowed (checked by validate)
    ),
    "oof": (
        Col("class_id", "any", note="int locally; 12-hex HMAC in results/"),
        Col("t_s", _F), Col("fold", _I), Col("y", _F32), Col("p_raw", _F), Col("p_cal", _F),
        Col("inner", _B),
        Col("in_G", _B, required=False, note="points.in_G: event metrics from results/"),
    ),
    "results": tuple(
        _cols("condition tier model featureset split target", _S)
        + _cols("H g", _F)
        + [Col("fold", _NI, note="<NA> for pooled rows")]
        + _cols("n_classes n_onsets n_points", _NI)
        + _cols("base_rate ap ap_fold_mean ap_fold_sd ap_lift ap_within macro_ap ap_shift "
                "timing_gain ap_local timing_gain_local roc_auc brier ece recall_far01 "
                "recall_far05 realised_far01 realised_far05 tiw_far01 tiw_far05 "
                "eventlift_shift_far01 eventlift0_far01 lead_med_s pafroc", _F)
        # A9 additions (optional so early stages can write partial rows)
        + _cols("brier_skill recall_random_far01 recall_random_far05", _F, required=False)
        + _cols("ap_within_lo ap_within_hi", _F, required=False)
        # §8.5 class-bootstrap CIs (shared resamples) and the TG bases (kept classes only)
        + _cols("ap_lo ap_hi timing_gain_lo timing_gain_hi timing_gain_local_lo "
                "timing_gain_local_hi recall_far01_lo recall_far01_hi recall_far05_lo "
                "recall_far05_hi lead_med_s_lo lead_med_s_hi ap_shift_base ap_local_base",
                _F, required=False)
        + [Col("n_classes_shift_skipped", _NI, required=False,
               note="classes with < 400 defined points left out of ap_shift / timing_gain")]
        # A1: g <= 1 conditions need re-timed onsets
        + [Col("retime_failed_share", _F, required=False,
               note="non-censored bids whose re-timing failed or never ran"),
           Col("g1_approximate", _NB, required=False,
               note="A1: retime_failed_share >= G1_RETIME_FAILED_MAX (g = 1 approximate)")]
    ),
    "ev": tuple(  # EV (S3): event recall vs references at the model's realised FAR; aggregate
        _cols("condition reference", _S)
        + _cols("budget far recall recall_lo recall_hi tiw lead_med_s delta delta_lo delta_hi "
                "delta_p_le0", _F)
        + _cols("n_classes n_onsets", _NI)
        + [Col("s3_pass", _NB, note="model rows: Recall CI lo > shift and R recall at "
                                    "matched FAR")]
    ),
    "bids": (  # committed: Brady's own bids, hashed class id
        Col("class_hash", _S), Col("onset_s", _F), Col("ep_type", _S), Col("outcome", _S),
        Col("call_latency_s", _F), Col("seg_type", _S),
        Col("left_censored", _NB, required=False,
            note="first observed UP, not a true onset: never an event-metric onset"),
    ),
}

TABLE_NAMES = tuple(TABLES)


def schema(name: str) -> tuple[Col, ...]:
    try:
        return TABLES[name]
    except KeyError:
        raise SchemaError(f"unknown table {name!r}; known: {', '.join(TABLE_NAMES)}") from None


def columns(name: str) -> list[str]:
    """All schema columns, in order."""
    return [c.name for c in schema(name)]


def required(name: str) -> list[str]:
    return [c.name for c in schema(name) if c.required]


def dtypes(name: str) -> dict[str, str]:
    return {c.name: c.dtype for c in schema(name)}


def _pd_dtype(dtype: str):
    return object if dtype in ("any", "bytes") else dtype


def empty(name: str) -> pd.DataFrame:
    """A zero-row DataFrame with every schema column at its dtype."""
    return pd.DataFrame({c.name: pd.Series([], dtype=_pd_dtype(c.dtype)) for c in schema(name)})


def coerce(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Cast the schema columns present in `df` to their dtypes (other columns untouched)."""
    out = df.copy()
    for c in schema(name):
        if c.name in out.columns and c.dtype not in ("any", "bytes"):
            out[c.name] = out[c.name].astype(c.dtype)
    return out


def _kind_ok(series: pd.Series, dtype: str) -> bool:
    """Loose dtype check: numeric vs bool vs string. All-missing columns always pass."""
    if dtype == "any" or series.isna().all():
        return True
    d = series.dtype
    if dtype == "bytes":
        return d == object and series.dropna().map(lambda v: isinstance(v, bytes | bytearray)).all()
    if dtype in ("bool", "boolean"):
        return pd.api.types.is_bool_dtype(d) or (
            d == object and series.dropna().map(lambda v: isinstance(v, bool | np.bool_)).all())
    if dtype in ("int64", "Int64"):
        return pd.api.types.is_integer_dtype(d)
    if dtype.startswith("float"):
        return pd.api.types.is_numeric_dtype(d) and not pd.api.types.is_bool_dtype(d)
    if dtype == "string":
        return pd.api.types.is_string_dtype(d) or d == object
    return True


def validate(df: pd.DataFrame, name: str, *, strict: bool = False, check_dtypes: bool = True
             ) -> pd.DataFrame:
    """Check `df` against table `name`; returns `df` unchanged so calls can chain.

    * required columns must be present;
    * with `check_dtypes`, each present schema column must have a compatible kind;
    * `features`: every non-key column must pass `config.feature_allowed` (any featureset);
    * `strict`: no columns outside the schema (not applied to `features`).
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"{name}: expected a DataFrame, got {type(df).__name__}")
    cols = schema(name)
    problems = []
    missing = [c.name for c in cols if c.required and c.name not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    if check_dtypes:
        bad = [f"{c.name} ({df[c.name].dtype}, want {c.dtype})" for c in cols
               if c.name in df.columns and not _kind_ok(df[c.name], c.dtype)]
        if bad:
            problems.append(f"bad dtypes {bad}")
    known = {c.name for c in cols}
    extra = [c for c in df.columns if c not in known]
    if name == "features":
        blocked = [c for c in extra
                   if not config.feature_allowed(c, featureset="F-punct")]
        if blocked:
            problems.append(f"columns not in FEATURE_ALLOWLIST {blocked}")
    elif strict and extra:
        problems.append(f"unexpected columns {extra}")
    if problems:
        raise SchemaError(f"{name}: " + "; ".join(problems))
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-input columns of a features table (everything but the key columns)."""
    return [c for c in df.columns if c not in config.FEATURE_META_COLUMNS]
