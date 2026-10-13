"""Comparison conditions (spec §7 + amendments A1-A5, A9) and the cross-validation runner.

`CONDITIONS` is the registry: one `Condition` per computed row of §7 (model x feature set x
split x target x variant), plus `derived` rows that only re-score existing OOF predictions
(TG, EV), `control` rows run by controls.py, and the A5 kNN `explainer`. Each carries its
tier (mvp / must / should / could).

`run(cond, data)` fits the condition fold by fold. Outer folds are assigned ONCE over all dev
classes (`points.fold`, else computed over the whole dev set) and every population subset
and variant inherits that assignment, so a class tested in fold k is never in the data
fold k's hyperparameters were tuned on (asserted). Hyperparameters are tuned once per outer
fold on F-all (`models.tune_once`, cached) and calibrated by Platt on grouped inner OOF
scores; the inner OOF rows kept for the event thresholds cover the training fold's whole
event grid. It writes `oof/{cond}.parquet` (class_id, t_s, fold, y, p_raw, p_cal, inner,
in_G = the target's event grid) plus a JSON sidecar holding the config hash, and returns
`results` rows. A second call with an unchanged hash reads the cache. `run_ev` scores the
derived EV row (S3) from cached OOF files.

Everything stays on dev classes (`in_vault == 0`, asserted) except the V conditions, which
need `allow_vault=True` (opened once, after the G4 freeze).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import __version__, config, controls, metrics, models, schemas, splits

TIERS = ("mvp", "must", "should", "could")
KINDS = ("model", "derived", "control", "explainer")
SPLITS = ("S-group", "S-rand", "S-time", "S-loco", "vault")
SPEECH_CAUSAL_RULES = ("opening", "end", "rec_gap", "breakout", "break", "me_speaking")
LC_SIZES = (20, 40, 80)


class MissingProvider(RuntimeError):
    """A condition needs data another stage must supply (a feature rebuild, a label
    variant, GRU step inputs, ...)."""


# --------------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class Condition:
    id: str
    group: str                    # §7 row: K0, F0, TG, S, EV, P-plant, ...
    tier: str
    model: str | None = None
    featureset: str | None = None
    split: str = "S-group"
    target: str = config.HEADLINE_LABEL
    H: float = config.H_S
    g: float = config.G_S
    kind: str = "model"
    population: str = "dev"
    variant: tuple = ()           # sorted (key, value) pairs
    uses: tuple = ()              # derived rows: conditions whose OOF they read
    answers: str = ""

    def v(self, key, default=None):
        return dict(self.variant).get(key, default)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["variant"] = dict(self.variant)
        d["uses"] = list(self.uses)
        return d


def _c(id, group, tier, model=None, featureset=None, **kw) -> Condition:
    variant = tuple(sorted(kw.pop("variant", {}).items()))
    return Condition(id=id, group=group, tier=tier, model=model, featureset=featureset,
                     variant=variant, **kw)


def _build() -> dict[str, Condition]:
    C = []
    # ---- MVP (§7)
    k0 = "does anything beat the rule? (H4)"
    C += [_c("K0-AO", "K0", "mvp", "AO", "F-none", answers="FROC reference point"),
          _c("K0-Bmin", "K0", "mvp", "B-min", "F-min", answers="FROC reference curve"),
          _c("K0-B0", "K0", "mvp", "B0", "F-none", answers=k0),
          _c("K0-R", "K0", "mvp", "R", "F-q", answers=k0),
          _c("K0-B1", "K0", "mvp", "B1", "F-struct", answers=k0),
          _c("K0-M1", "K0", "mvp", "M1", "F-all", answers=k0),
          _c("K0-M2", "K0", "mvp", "M2", "F-all", answers=k0),
          _c("K0-M3", "K0", "mvp", "M3", "F-all", answers="headline; F-all side of H1")]
    h1 = "H1 and its co-requirement (F-all = K0-M3)"
    C += [_c("F0-Fstruct", "F0", "mvp", "M3", "F-struct", answers=h1),
          _c("F0-Fall-me_hand", "F0", "mvp", "M3", "F-all-me_hand", answers=h1),
          _c("F0-Fstruct-me_hand", "F0", "mvp", "M3", "F-struct-me_hand", answers=h1)]
    C.append(_c("TG", "TG", "mvp", kind="derived",
                uses=("K0-M3", "F0-Fstruct", "F0-Fall-me_hand", "F0-Fstruct-me_hand"),
                answers="H2: the class vs the phase vs the moment (global/local shifts)"))
    C += [_c("S-rand-M2", "S", "mvp", "M2", "F-all", split="S-rand",
             answers="leakage tax = AP_rand - AP_group"),
          _c("S-rand-M3", "S", "mvp", "M3", "F-all", split="S-rand",
             answers="leakage tax = AP_rand - AP_group")]
    C.append(_c("EV", "EV", "mvp", kind="derived",
                uses=("K0-M3", "K0-AO", "K0-Bmin", "K0-R"),
                answers="S3: event recall at FAR budgets vs AO, B-min, shifted, R and the "
                        "matched-rate random alarm (A9)"))
    C.append(_c("P-clock", "P-clock", "mvp", kind="control",
                answers="words, roles and video aligned (stop rule)"))
    C += [_c(f"P-plant-{m}", "P-plant", "mvp", m, "F-all", variant={"plant": True},
             answers="features and model see a planted cue (stop rule)") for m in ("M1", "M2")]
    C.append(_c("CALLON", "CALLON", "mvp", kind="control",
                answers="are labels late against the audio?"))
    # ---- must
    C.append(_c("K1-M4", "K1", "must", "M4", "F-all", answers="H5 (vs K0-M2)"))
    C += [_c("F1-Ftext", "F1", "must", "M3", "F-text", answers="H3 / text alone"),
          _c("F1-Fall-pcw", "F1", "must", "M3", "F-all-pcw", population="pcw_subset",
             answers="H3 (without PCW)"),
          _c("F1-Fall+pcwA", "F1", "must", "M3", "F-all+pcwA", population="pcw_subset",
             answers="H3 (with PCW)")]
    C += [_c(f"A-mismatch-{k}", "A-mismatch", "must", "M3", "F-all+pcwA",
             population="pcw_subset", variant={"mismatch": k},
             answers="does *matching* preparation matter?") for k in range(config.MISMATCH_N)]
    # A1: g in {1, 2, 3, 5} is must-tier; g = 2 is the headline (K0-M3).
    for g in config.G_SWEEP:
        if g != config.G_S:
            C.append(_c(f"G-g{g}", "G", "must", "M3", "F-all", target=config.y_col(15, g), g=g,
                        answers="own-precursor and clock test"))
    C += [_c(f"IH+{d}", "IH+", "must", "M2", "F-text",
             variant={"cutoff_delta": float(d), "include_me": True},
             answers="the pipeline sees the signal a called bid leaves")
          for d in config.IH_PLUS_DELTAS_S]
    C.append(_c("N-name", "N-name", "must", kind="control",
                answers="name mentions around bids (descriptive)"))
    C.append(_c("N-name-Fname", "N-name", "must", "M3", "F-name",
                answers="ΔAP(F-name − F-all) for the N-name flag"))
    for m, fs in (("M3", "F-all"), ("M3", "F-all-me_hand"), ("M3", "F-struct"), ("R", "F-q"),
                  ("B1", "F-struct")):
        C.append(_c(f"V-{m}-{fs}", "V", "must", m, fs, split="vault", population="vault",
                    answers="untouched estimate"))
    # ---- should
    C.append(_c("M5-gru", "M5", "should", "M5", "F-all", answers="does sequence modelling "
                "help?"))
    C += [_c("F2-Fpcw", "F2", "should", "M3", "F-pcw", population="pcw_subset",
             answers="preparation alone"),
          _c("F2-Fall-me", "F2", "should", "M3", "F-all-me", answers="my own history"),
          _c("F2-Fpunct", "F2", "should", "M3", "F-punct", answers="punctuation leak size")]
    C += [_c(f"F2-M2-{fs}", "F2", "should", "M2", fs, answers="M2 mirror of F")
          for fs in ("F-struct", "F-all-me_hand", "F-struct-me_hand", "F-text", "F-name",
                     "F-punct")]
    C += [_c("S2-time-M3", "S2", "should", "M3", "F-all", split="S-time",
             answers="stability over years"),
          _c("S2-loco-M3", "S2", "should", "M3", "F-all", split="S-loco",
             answers="stability over courses")]
    for h in config.H_SWEEP:
        if h != config.H_S:
            C.append(_c(f"Hs-H{h}", "Hs", "should", "M3", "F-all", target=config.y_col(h, 2),
                        H=h, answers="how far ahead the cue reaches"))
    for d in config.IH_DELTAS_S:
        name = f"IH{'+' if d > 0 else ''}{d}" if d else "IH0"
        C.append(_c(name, "IH", "should", "M2", "F-text", variant={"cutoff_delta": float(d)},
                    answers="where the signal arrives (illegal side shaded)"))
    C.append(_c("T-speech", "T", "should", "M3", "F-all-me_hand",
                target=config.y_col(15, config.G_SPEECH_S, speech=True), g=config.G_SPEECH_S,
                answers="is volunteering more or less anticipatable than speaking?"))
    C.append(_c("A-stream", "A-stream", "should", kind="control",
                answers="how much batch look-ahead inflates AP"))
    for s in config.LABEL_SHIFTS_S:
        for m in ("M2", "M3"):
            C.append(_c(f"LS-{s}-{m}", "LS", "should", m, "F-all", variant={"label_shift": s},
                        answers="does signal pile up right at the onset?"))
    C.append(_c("Live", "Live", "should", kind="control", answers="key-press → icon lag"))
    sens = "headline robustness"
    C += [_c("Sens-cal3", "Sens", "should", "M3", "F-all", population="cal_sensitivity",
             answers=sens),
          _c("Sens-sym", "Sens", "should", "M3", "F-all",
             target=config.y_col(15, 2, sym=True), answers=sens),
          _c("Sens-no-pcwT", "Sens", "should", "M3", "F-all-pcwT", answers=sens),
          _c("Sens-template", "Sens", "should", "M3", "F-all",
             population="template_validated", answers=sens),
          _c("Sens-multisession", "Sens", "should", "M3", "F-all",
             population="multi_session_included", answers=sens),
          _c("Sens-prec09", "Sens", "should", "M3", "F-all", population="precision_ge_09",
             answers=sens),
          _c("Sens-hand-preceded", "Sens", "should", "M3", "F-all",
             population="hand_preceded_ge_03", answers=sens)]
    for lv in ("votepos", "dnonly", "contested", "vote05", "blips", "nopolltick"):
        C.append(_c(f"Sens-{lv}", "Sens", "should", "M3", "F-all",
                    variant={"label_variant": lv}, answers=sens))
    for m in ("M2", "M3", "M4"):
        for n in LC_SIZES:
            C.append(_c(f"LC-{m}-{n}", "LC", "should", m, "F-all",
                        variant={"train_classes": n}, answers="data limit or ceiling?"))
    C += [_c("F-all+hist", "hist", "should", "M3", "F-all+hist",
             answers="A2: gain from Brady's cross-class history (fold-aware)"),
          _c("F-all+edu", "edu", "should", "M3", "F-all+edu",
             answers="A3: gain from EduBehaviors discourse scores"),
          _c("M3+stack", "stack", "should", "M3+stack", "F-all+stack",
             answers="A4: grouped inner-OOF M2/M4 scores stacked into M3")]
    # ---- could
    C += [_c("M6-xenc", "M6", "could", "M6", "F-all", answers="end-to-end text"),
          _c("N-stale", "Could", "could", "M3", "F-all", variant={"stale_s": config.N_STALE_S},
             answers="topic vs moment"),
          _c("A-jitter", "Could", "could", "M3", "F-all",
             variant={"jitter_sd": config.A_JITTER_SD_S}, answers="timing robustness"),
          _c("A-attrib", "Could", "could", "M3", "F-all", variant={"attrib": "fused"},
             answers="Scribe fused/cluster roles"),
          _c("turn-end", "Could", "could", "M3", "F-all", variant={"grid": "turn_end"},
             answers="turn-end decision grid"),
          _c("M4-MLP", "Could", "could", "M4-MLP", "F-all", answers="non-linear head"),
          _c("WER", "Could", "could", kind="control", answers="ASR word error rate"),
          _c("kNN", "kNN", "could", kind="explainer", uses=("K0-M3",),
             answers="A5: share of neighbours that preceded a bid, TPs vs false alarms"),
          _c("G-g0", "G01", "could", "M3", "F-all", target=config.y_col(15, 0), g=0,
             answers="kept from §7 G01; A1 moved g = 1 to must")]
    return {c.id: c for c in C}


CONDITIONS: dict[str, Condition] = _build()
SECTION7_GROUPS = ("K0", "F0", "TG", "S", "EV", "P-clock", "P-plant", "CALLON", "K1", "F1",
                   "A-mismatch", "G", "IH+", "N-name", "V", "M5", "F2", "S2", "Hs", "IH",
                   "G01", "T", "A-stream", "LS", "Live", "Sens", "LC", "Could", "M6")
AMENDMENT_CONDITIONS = ("G-g1", "G-g3", "G-g5", "F-all+hist", "F-all+edu", "M3+stack", "kNN")


def get(cond: str | Condition) -> Condition:
    if isinstance(cond, Condition):
        return cond
    try:
        return CONDITIONS[cond]
    except KeyError:
        raise KeyError(f"unknown condition {cond!r}") from None


def select(ids=None, tier: str | None = None, kind: str | None = "model",
           up_to: bool = False) -> list[Condition]:
    """Conditions by id list, tier (or every tier up to it with up_to) and kind."""
    out = []
    for c in CONDITIONS.values():
        if ids and c.id not in ids:
            continue
        if tier and not (TIERS.index(c.tier) <= TIERS.index(tier) if up_to else c.tier == tier):
            continue
        if kind and c.kind != kind:
            continue
        out.append(c)
    return out


# --------------------------------------------------------------------------- data


@dataclass
class Data:
    """Inputs of a run. `X` is the design frame aligned row-for-row with `points` (see
    models.py); `classes` is the classes table (QA, course, date, section, pcw_status).
    `providers` maps a variant key to `fn(data, condition) -> Data` for variants another
    stage must compute (cutoff_delta, mismatch, label_shift, label_variant, stale_s,
    jitter_sd, attrib, grid) and "steps" to `fn(data, featureset) -> {class_id: StepRaw}`
    for M5."""

    points: pd.DataFrame
    X: pd.DataFrame
    classes: pd.DataFrame | None = None
    providers: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.points) != len(self.X):
            raise ValueError("points and X must align row-for-row")
        a = self.points[["class_id", "t_s"]].to_numpy(float)
        b = self.X[["class_id", "t_s"]].to_numpy(float)
        if not np.array_equal(a, b):
            raise ValueError("points and X rows must share (class_id, t_s) in the same order")

    def take(self, mask) -> Data:
        m = np.asarray(mask, bool)
        return replace(self, points=self.points[m].reset_index(drop=True),
                       X=self.X[m].reset_index(drop=True))

    def fingerprint(self) -> str:
        return _fingerprint(self.points, self.X)


def _hash_frame(h, df: pd.DataFrame) -> None:
    for c in df.columns:
        h.update(str(c).encode())
        h.update(str(df[c].dtype).encode())
    emb = [c for c in df.columns if str(c).startswith("emb_")]
    rest = [c for c in df.columns if c not in emb]
    if rest:
        h.update(pd.util.hash_pandas_object(df[rest], index=False).to_numpy().tobytes())
    if emb:
        h.update(np.ascontiguousarray(df[emb].to_numpy(np.float32)).tobytes())


def _fingerprint(*frames) -> str:
    h = hashlib.sha1()
    for df in frames:
        _hash_frame(h, df)
    return h.hexdigest()[:16]


def assert_no_vault(df: pd.DataFrame) -> pd.DataFrame:
    """Dev tools must never see vault rows (§8.1)."""
    if "in_vault" in df and df["in_vault"].fillna(False).astype(bool).any():
        raise AssertionError("vault rows reached a dev-only step")
    return df


def assert_no_vault_data(data: Data) -> Data:
    assert_no_vault(data.points)
    return data


def with_cv_groups(data: Data) -> Data:
    """Add `cv_group` to the design when `classes.cv_group` groups any classes (L18: same
    section, same date), so inner GroupKFold splits (models.groups_of) keep twins together
    like the outer folds do. Never a model input (not allowlisted)."""
    if data.classes is None or "cv_group" not in data.classes or "cv_group" in data.X:
        return data
    groups = splits.cv_groups(data.points["class_id"], data.classes)
    if (groups.index.to_numpy() == groups.to_numpy()).all():
        return data
    X = data.X.assign(cv_group=data.points["class_id"].map(groups).to_numpy())
    return replace(data, X=X)


def needs_retimed_onsets(cond: Condition) -> bool:
    """g < 2 conditions (G-g1, G-g0) rest on re-timed onsets (A1)."""
    return not is_speech_target(cond.target) and float(cond.g) < config.G_S


def retime_info(points: pd.DataFrame, tables: dict, cond: Condition) -> dict | None:
    """A1 for g < 2: pooled retime_failed share over the condition's non-censored bids and
    the `g1_approximate` flag (share >= G1_RETIME_FAILED_MAX). None for other conditions;
    without hand_episodes the share is unknown and the flag stays <NA>."""
    if not needs_retimed_onsets(cond):
        return None
    from . import labels

    ep = (tables or {}).get("hand_episodes")
    if ep is None:
        return {"retime_failed_share": float("nan"), "g1_approximate": pd.NA}
    ep = ep[ep["class_id"].isin(set(points["class_id"]))]
    r = labels.retime_summary(ep)
    return {"retime_failed_share": r["retime_failed_share"],
            "g1_approximate": r["g1_approximate"]}


def config_snapshot(n_inner: int, hgb_max_iter: int) -> dict:
    """Settings that change a run's predictions (hashed with the condition and data)."""
    return {"version": __version__, "seed": config.SEED, "hgb": config.HGB_PARAMS,
            "hgb_max_iter": hgb_max_iter, "n_inner": n_inner, "grids": {
                k: list(v) for k, v in models.GRIDS.items()},
            "tfidf": {**config.TFIDF_PARAMS, "ngram_range": list(
                config.TFIDF_PARAMS["ngram_range"])}, "tfidf_min_classes":
            config.TFIDF_MIN_CLASSES, "pca": config.PCA_DIMS,
            "calibration": config.CALIBRATION_METHOD, "branch": models.PCW_BRANCH,
            "folds": config.N_OUTER_FOLDS}


def condition_hash(cond: Condition, data_fp: str, settings: dict) -> str:
    blob = json.dumps({"cond": {k: v for k, v in cond.to_dict().items()
                                if k not in ("id", "tier", "answers", "group")},
                       "data": data_fp, "settings": settings}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- populations


def _classes_or_raise(data: Data, col: str) -> pd.DataFrame:
    if data.classes is None or col not in data.classes:
        raise MissingProvider(f"population needs classes.{col}")
    return data.classes


def population_classes(data: Data, population: str) -> set | None:
    """Class ids admitted by a population (None = no class filter beyond dev/QA)."""
    if population in ("dev", "vault"):
        return None
    if population == "pcw_subset":
        cl = _classes_or_raise(data, "pcw_status")
        ok = cl["pcw_status"].isin(config.PCW_USABLE_STATUSES)
        if "pcw_has" in data.X:
            has = data.X.groupby("class_id")["pcw_has"].max()
            ok &= cl["class_id"].map(has).fillna(0).astype(float).gt(0)
        return set(cl.loc[ok, "class_id"])
    if population == "template_validated":
        cl = _classes_or_raise(data, "template_validated")
        return set(cl.loc[cl["template_validated"].fillna(False).astype(bool), "class_id"])
    if population == "precision_ge_09":
        cl = _classes_or_raise(data, "audited_precision")
        return set(cl.loc[cl["audited_precision"] >= 0.9, "class_id"])
    if population == "hand_preceded_ge_03":
        cl = _classes_or_raise(data, "hand_preceded_share")
        return set(cl.loc[cl["hand_preceded_share"] >= 0.3, "class_id"])
    if population in ("cal_sensitivity", "multi_session_included"):
        return None
    raise KeyError(f"unknown population {population!r}")


def _qa_override(data: Data, population: str) -> set:
    """Classes a sensitivity population re-admits although they failed headline QA."""
    if data.classes is None or "qa_reason" not in data.classes:
        return set()
    cl = data.classes
    if population == "multi_session_included":
        return set(cl.loc[cl["qa_reason"] == "multi_session", "class_id"])
    if population == "cal_sensitivity":
        out = set()
        for r in cl.itertuples():
            reason = str(getattr(r, "qa_reason", ""))
            if not reason.startswith(("cal_", "resid_")):
                continue
            cal = {"applied": r.cal_applied, "status": r.cal_status, "method": r.cal_method,
                   "windows": [0] * int(r.cal_n_windows or 0),
                   "max_residual_s": r.cal_max_residual_s}
            if config.calibration_qa(cal, r.resid_med_s, r.resid_p95_s, sensitivity=True)[0]:
                out.add(r.class_id)
        return out
    return set()


def is_speech_target(target: str) -> bool:
    return target.startswith("y_speech")


def train_mask(points: pd.DataFrame, target: str, readmit: set = frozenset()) -> np.ndarray:
    """Rows with a defined label inside the at-risk set (§3). Hand targets: in the causal grid
    G; speech targets: only the T-speech causal rules (no hand-derived rules)."""
    y = points[target]
    ok = y.notna().to_numpy().copy()
    excl = points["excl_reason"].astype(object) if "excl_reason" in points else None
    if is_speech_target(target):
        if excl is not None:
            ok &= ~excl.isin(SPEECH_CAUSAL_RULES).to_numpy()
    elif "in_G" in points:
        ok &= points["in_G"].fillna(False).astype(bool).to_numpy()
    if excl is not None:
        qa = (excl == "class_qa").to_numpy()
        ok &= ~qa | points["class_id"].isin(readmit).to_numpy()
    return ok


def grid_mask(points: pd.DataFrame, target: str, readmit: set = frozenset()) -> np.ndarray:
    """The target's causal event grid (§8.3), class-QA failures out unless re-admitted.

    Hand targets: `points.in_G`. T-speech: every point not removed by its own causal rules
    (opening, end, rec_gap, breakout, break, me_speaking; §2.6: no hand-derived rule), so
    hand_up / post_lowering points are on its grid. `excl_reason` holds the first rule that
    fires and the speech rules come first in that order, so a point outside them passed all.
    """
    n = len(points)
    excl = points["excl_reason"].astype(object) if "excl_reason" in points else None
    if is_speech_target(target):
        g = ~excl.isin(SPEECH_CAUSAL_RULES).to_numpy() if excl is not None else np.ones(n, bool)
    elif "in_G" in points:
        g = points["in_G"].fillna(False).astype(bool).to_numpy().copy()
    else:
        g = np.zeros(n, bool)
    if excl is not None:
        g &= (excl != "class_qa").to_numpy() | points["class_id"].isin(readmit).to_numpy()
    return g


def predict_mask(points: pd.DataFrame, target: str, readmit: set = frozenset()) -> np.ndarray:
    """Rows that get a test prediction: the target's event grid (alarms may fire at
    undefined points, §8.3) plus every trainable row."""
    return train_mask(points, target, readmit) | grid_mask(points, target, readmit)


# --------------------------------------------------------------------------- splits


def _course_of(data: Data) -> pd.Series:
    if data.classes is not None and "course_code" in data.classes:
        return data.classes.set_index("class_id")["course_code"]
    if "ctx_course" in data.X:
        return data.X.groupby("class_id")["ctx_course"].first()
    return pd.Series(dtype=object)


def _per_class(data: Data, col: str) -> pd.Series:
    if data.classes is None or col not in data.classes:
        raise MissingProvider(f"this split needs classes.{col}")
    return data.classes.drop_duplicates("class_id").set_index("class_id")[col]


def fold_labels(data: Data, split: str, *, seed: int = config.SEED) -> np.ndarray:
    """One outer-fold label per row (splits.TRAIN_ONLY = -1: always training; <NA>-free)."""
    p = data.points
    cid = p["class_id"].to_numpy()
    if split == "S-group":
        if "fold" in p and p["fold"].notna().all():
            return p["fold"].astype(int).to_numpy()
        course = p["class_id"].map(_course_of(data)).fillna("other").astype(str).to_numpy()
        groups = splits.cv_groups(cid, data.classes)
        k = min(config.N_OUTER_FOLDS, len(pd.unique(groups)))
        try:
            folds = splits.group_folds(cid, course, groups=groups, n_splits=k, seed=seed)
        except ValueError:        # every course has fewer than k classes: drop the strata
            folds = splits.group_folds(cid, None, groups=groups, n_splits=k, seed=seed)
        return splits.point_folds(cid, folds)
    if split == "S-rand":
        return splits.random_folds(len(p), seed=seed)
    if split == "S-time":
        return np.asarray(splits.time_split(p["class_id"].map(_per_class(data,
                                                                         "semester_idx"))))
    if split == "S-loco":
        courses = _course_of(data)
        if "n_bids" in getattr(data.classes, "columns", ()):
            n_bids = _per_class(data, "n_bids").astype(float)
        else:
            n_bids = controls.bid_onsets(points=p).groupby("class_id").size().astype(float)
        f = splits.loco(courses.reindex(np.unique(cid)), n_bids)
        return splits.point_folds(cid, f)
    if split == "vault":
        v = p["in_vault"].fillna(False).astype(bool).to_numpy()
        return np.where(v, 0, splits.TRAIN_ONLY)
    raise KeyError(f"unknown split {split!r}")


def outer_folds(data: Data, split: str, *, seed: int = config.SEED,
                labels: np.ndarray | None = None) -> list[tuple]:
    """[(fold_id, train_rows, test_rows)] boolean masks over data rows (splits.py folds; the
    S-group assignment is `points.fold` when present; `labels` overrides)."""
    n = len(data.points)
    out = []
    lab = fold_labels(data, split, seed=seed) if labels is None else labels
    for k, tr, te in splits.iter_folds(lab):
        a, b = np.zeros(n, bool), np.zeros(n, bool)
        a[tr], b[te] = True, True
        out.append((k, a, b))
    return out


def inherit_folds(dev: Data, dev_labels: np.ndarray, target: Data, split: str) -> np.ndarray:
    """Fold labels for `target` (a population subset or variant of `dev`) taken from the
    dev-wide assignment `dev_labels`, never recomputed on the subset: by class for class
    splits, by (class_id, t_s) for S-rand. Rows unknown to dev are TRAIN_ONLY."""
    if split == "S-rand":
        key = pd.MultiIndex.from_arrays([dev.points["class_id"], dev.points["t_s"]])
        m = pd.Series(np.asarray(dev_labels), index=key)
        m = m[~m.index.duplicated()]
        tk = pd.MultiIndex.from_arrays([target.points["class_id"], target.points["t_s"]])
        return m.reindex(tk).fillna(splits.TRAIN_ONLY).astype(int).to_numpy()
    per_class = pd.Series(np.asarray(dev_labels), index=dev.points["class_id"].to_numpy())
    per_class = per_class.groupby(level=0).first()
    return target.points["class_id"].map(per_class).fillna(splits.TRAIN_ONLY) \
        .astype(int).to_numpy()


def assert_tuning_disjoint(tune_rows: dict, dev: Data, folds: list[tuple], data: Data,
                           split: str) -> None:
    """No class (S-rand: no point) tested in outer fold k may be in the rows fold k's
    hyperparameters were tuned on."""
    for k, _, te in folds:
        if k not in tune_rows or not te.any():
            continue
        tr = tune_rows[k]
        if split == "S-rand":
            a = set(zip(dev.points["class_id"].to_numpy()[tr],
                        dev.points["t_s"].to_numpy(float)[tr], strict=True))
            b = set(zip(data.points["class_id"].to_numpy()[te],
                        data.points["t_s"].to_numpy(float)[te], strict=True))
        else:
            a = set(dev.points["class_id"].to_numpy()[tr])
            b = set(data.points["class_id"].to_numpy()[te])
        shared = a & b
        if shared:
            raise AssertionError(f"fold {k}: {len(shared)} test classes/points were in the "
                                 "hyperparameter-tuning data (outer folds misaligned)")


# --------------------------------------------------------------------------- in-fold features


def _class_start(classes: pd.DataFrame) -> pd.Series:
    key = classes["sched_start_utc"] if "sched_start_utc" in classes else classes["date"]
    return pd.Series(pd.to_datetime(key.astype("string"), utc=True, errors="coerce").to_numpy(),
                     index=classes["class_id"].to_numpy())


def hist_columns(data: Data, train_classes) -> pd.DataFrame:
    """A2 `hist_*` for every row of `data`: Brady's bids per at-risk minute over training-fold
    classes that started strictly before the row's class (same course, and overall). Test-fold
    labels never enter; NaN without an earlier training class. Aligned with data rows."""
    if data.classes is None or "course_code" not in data.classes:
        raise MissingProvider("hist_* needs classes.course_code and a start date")
    cls = data.classes.drop_duplicates("class_id")
    start, course = _class_start(cls), cls.set_index("class_id")["course_code"].astype(str)
    train = set(train_classes)
    pts = data.points[data.points["class_id"].isin(train)]
    risk = pts["at_risk"] if "at_risk" in pts else pts.get("in_G", pd.Series(True, pts.index))
    minutes = risk.fillna(False).astype(bool).groupby(pts["class_id"]).sum() * \
        config.DELTA_S / 60
    ep = data.tables.get("hand_episodes")
    if ep is not None:
        bids = controls.bid_onsets(episodes=ep[ep["class_id"].isin(train)]).groupby(
            "class_id").size()
    elif "n_bids" in cls:
        bids = cls.set_index("class_id")["n_bids"]
    else:
        bids = controls.bid_onsets(points=pts).groupby("class_id").size()
    st = pd.DataFrame({"minutes": minutes})
    st["bids"] = bids.reindex(st.index).fillna(0).astype(float)
    st["start"], st["course"] = start.reindex(st.index), course.reindex(st.index)
    rows = {}
    for c in data.points["class_id"].unique():
        prior = st[(st["start"] < start.get(c)) & (st.index != c)]
        same = prior[prior["course"] == course.get(c)]
        rows[c] = [d["bids"].sum() / d["minutes"].sum() if d["minutes"].sum() > 0 else np.nan
                   for d in (same, prior)]
    vals = np.array([rows[c] for c in data.points["class_id"]], float).reshape(-1, 2)
    return pd.DataFrame(vals, columns=list(config.HIST_COLUMNS))


# --------------------------------------------------------------------------- variants


def _plant_provider(data: Data, cond: Condition) -> Data:
    planted = controls.choose_planted(controls.bid_onsets(points=data.points))
    X, vis = controls.plant_design(data.X, planted)
    out = replace(data, X=X)
    out.tables = {**data.tables, "plant_visible": vis, "planted": planted}
    return out


def _label_provider(data: Data, cond: Condition) -> Data:
    col = f"{cond.target}_{cond.v('label_variant')}"
    if col not in data.points:
        if "label_variant" in data.providers:
            return data.providers["label_variant"](data, cond)
        raise MissingProvider(f"{cond.id} needs label column {col} (labels.py)")
    return replace(data, points=data.points.assign(**{cond.target: data.points[col]}))


BUILTIN_PROVIDERS: dict[str, Callable] = {"plant": _plant_provider,
                                          "label_variant": _label_provider}
EXTERNAL_VARIANTS = ("cutoff_delta", "include_me", "mismatch", "label_shift", "stale_s",
                     "jitter_sd", "attrib", "grid")
IN_RUN_VARIANTS = ("train_classes",)


def apply_variants(data: Data, cond: Condition) -> Data:
    """Data for a condition's variant (each provider receives the condition)."""
    done = set()
    for key, _ in cond.variant:
        if key in IN_RUN_VARIANTS or key in done:
            continue
        group_key = "cutoff_delta" if key == "include_me" else key
        if group_key in done:
            continue
        fn = data.providers.get(group_key) or BUILTIN_PROVIDERS.get(group_key)
        if fn is None:
            raise MissingProvider(f"{cond.id}: variant {group_key!r} needs a provider "
                                  f"(Data.providers[{group_key!r}])")
        data = fn(data, cond)
        done.add(group_key)
    return data


# --------------------------------------------------------------------------- run


@dataclass
class RunResult:
    condition: Condition
    oof: pd.DataFrame
    results: pd.DataFrame
    hash: str
    tuned: dict
    cached: bool = False
    meta: dict = field(default_factory=dict)


def _tune_key(split, fold, model, fp, settings) -> str:
    blob = json.dumps([split, fold, model, fp, settings], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def get_tuned(model_id: str, split: str, fold: int, X: pd.DataFrame, y, *, fp: str,
              settings: dict, tune_dir: Path | None, n_inner: int, hgb_max_iter: int) -> dict:
    """tune_once for the models `model_id` needs, on F-all, cached per (split, fold, data)."""
    need = {"M3+stack": ("M3", "M2", "M4")}.get(model_id, (model_id,))
    need = tuple(m for m in need if m in models.TUNED_MODELS)
    out = {}
    for m in need:
        key = _tune_key(split, fold, m, fp, settings)
        path = tune_dir / f"{key}.json" if tune_dir else None
        if path is not None and path.exists():
            out[m] = json.loads(path.read_text())
            continue
        res = models.tune_once(X, y, models=(m,), n_inner=n_inner,
                               hgb_max_iter=hgb_max_iter)[m]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res, default=float))
            with open(tune_dir / TUNE_INDEX, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "split": split, "fold": int(fold),
                                     "model": m}) + "\n")
        out[m] = res
    return out


TUNE_INDEX = "index.jsonl"        # in the tune dir: which split/fold/model each file holds


def tuning_curves(tune_dir: Path | str | None = None, *, split: str = "S-group"
                  ) -> dict[str, dict[int, dict]]:
    """{model: {outer fold: tune_once result (with 'curves')}} for one split, from the tune
    cache (the §6 learning-curve figures: plots.fig_hgb_learning / fig_lr_c_curve)."""
    d = Path(tune_dir) if tune_dir is not None else config.OOF_DIR / "_tune"
    index = d / TUNE_INDEX
    out: dict[str, dict[int, dict]] = {}
    if not index.exists():
        return out
    for line in index.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        path = d / f"{rec['key']}.json"
        if rec["split"] == split and path.exists():
            out.setdefault(rec["model"], {})[int(rec["fold"])] = json.loads(path.read_text())
    return out


def _subsample_classes(train: np.ndarray, cid: np.ndarray, n: int, seed: int) -> np.ndarray:
    classes = np.unique(cid[train])
    if n >= len(classes):
        return train
    keep = np.random.default_rng(seed).choice(classes, n, replace=False)
    return train & np.isin(cid, keep)


def _oof_rows(points, rows, fold, y, p_raw, p_cal, inner, grid) -> pd.DataFrame:
    """OOF rows; `in_G` is the target's event grid (`grid_mask`), so event metrics computed
    from results/ use the right grid (T-speech has no hand rules)."""
    return pd.DataFrame({"class_id": points["class_id"].to_numpy()[rows],
                         "t_s": points["t_s"].to_numpy(float)[rows], "fold": int(fold),
                         "y": np.asarray(y, np.float32), "p_raw": np.asarray(p_raw, float),
                         "p_cal": np.asarray(p_cal, float), "inner": bool(inner),
                         "in_G": np.asarray(grid, bool)[rows]})


def _rows(X: pd.DataFrame, mask) -> pd.DataFrame:
    return X.iloc[np.flatnonzero(mask)].reset_index(drop=True)


def _fit_fold_sklearn(cond, model_id, X, y_all, tr, te, tg, tuned, n_inner, estimator=None):
    """Fit on `tr`; predict `te` (test) and `tg` (the training fold's grid, for the
    rate-matched event thresholds, §8.3 step 2). The inner OOF rows returned cover all of
    `tg` (undefined labels included, §8.3 step 1), as (row mask, p_raw, p_cal)."""
    est = estimator if estimator is not None else models.make(
        model_id, cond.featureset, hyper=models.hyper_for(model_id, tuned))
    calibrate = models.MODELS[model_id].calibrate if model_id in models.MODELS else True
    cm = models.CalibratedModel(est, calibrate=calibrate, n_inner=n_inner)
    cm.fit(_rows(X, tr), y_all[tr], grid=_rows(X, tg))
    gp = cm.grid_predictions()
    inner = None if gp is None else (tg, gp[0], gp[1])
    return cm.predict(_rows(X, te)), inner, cm.predict(_rows(X, tg))


def _fit_fold_gru(cond, data, P, tr, te, tg, gru_kwargs):
    from . import gru

    fn = data.providers.get("steps")
    if fn is None:
        raise MissingProvider("M5 needs Data.providers['steps'] -> {class_id: StepRaw}")
    steps = fn(data, cond.featureset)
    cid, t = P["class_id"].to_numpy(), P["t_s"].to_numpy(float)
    full = data.points

    def seq(c, rows_mask):
        raw = steps[c]
        pc = full[full["class_id"] == c].sort_values("t_s")
        y, m = gru.loss_targets(pc, cond.target)
        allowed = set(t[(cid == c) & rows_mask])
        m &= np.isin(pc["t_s"].to_numpy(float), list(allowed))
        return raw, y, m
    train_c = list(pd.unique(cid[tr]))
    seqs = [seq(c, tr) for c in train_c]
    model = gru.GRUHazard(featureset=cond.featureset, **(gru_kwargs or {}))
    model.fit([s[0] for s in seqs], [s[1] for s in seqs], [s[2] for s in seqs])
    out_raw, out_cal = np.full(len(P), np.nan), np.full(len(P), np.nan)
    want = te | tg
    pred_c = list(pd.unique(cid[want]))
    for c, (pr, pcal) in zip(pred_c, model.predict([steps[c] for c in pred_c])):
        pos = {v: i for i, v in enumerate(steps[c].t_s)}
        rows = np.flatnonzero((cid == c) & want)
        idx = [pos[v] for v in t[rows]]
        out_raw[rows], out_cal[rows] = pr[idx], pcal[idx]
    inner = model.inner_predictions()
    inner_rows = None
    if inner is not None:                  # every grid step of the training classes
        ir, ic = np.full(len(P), np.nan), np.full(len(P), np.nan)
        for c, (raw, y, m), (pr, pcal) in zip(train_c, seqs, inner):
            pos = {v: i for i, v in enumerate(raw.t_s)}
            rows = np.flatnonzero((cid == c) & tg)
            idx = [pos[v] for v in t[rows]]
            ir[rows], ic[rows] = pr[idx], pcal[idx]
        rows = tg & np.isfinite(ir)
        inner_rows = (rows, ir[rows], ic[rows])
    return (out_raw[te], out_cal[te]), inner_rows, (out_raw[tg], out_cal[tg])


def run(cond: str | Condition, data: Data, *, out_dir: Path | str | None = None,
        cache: bool = True, force: bool = False, scorer: Callable | None = None,
        tune_dir: Path | str | None = None, allow_vault: bool = False,
        n_inner: int = config.N_INNER_FOLDS, hgb_max_iter: int = config.HGB_MAX_ITER,
        gru_kwargs: dict | None = None, seed: int = config.SEED) -> RunResult:
    """Cross-validated OOF predictions and results rows for one model condition."""
    cond = get(cond)
    if cond.kind != "model":
        raise ValueError(f"{cond.id} is a {cond.kind} row: nothing to fit")
    if cond.split == "vault" and not allow_vault:
        raise PermissionError("the vault opens once, after G4 (pass allow_vault=True)")
    out_dir = Path(out_dir) if out_dir is not None else config.OOF_DIR
    tune_dir = Path(tune_dir) if tune_dir is not None else out_dir / "_tune"
    settings = config_snapshot(n_inner, hgb_max_iter)

    # Population: dev classes (plus vault test classes for V), QA re-admissions.
    data = with_cv_groups(data)
    vault = data.points["in_vault"].fillna(False).astype(bool).to_numpy() \
        if "in_vault" in data.points else np.zeros(len(data.points), bool)
    dev = assert_no_vault_data(data.take(~vault))
    base = dev if cond.split != "vault" else data
    allowed = population_classes(base, cond.population)
    if allowed is not None:
        base = base.take(base.points["class_id"].isin(allowed).to_numpy())
    readmit = _qa_override(base, cond.population)

    vdata = apply_variants(base, cond)
    if cond.target not in vdata.points:
        raise MissingProvider(f"{cond.id}: points has no {cond.target}")
    P, X = vdata.points, vdata.X
    vfp = vdata.fingerprint()
    h = condition_hash(cond, vfp, settings)
    oof_path, meta_path = out_dir / f"{cond.id}.parquet", out_dir / f"{cond.id}.json"
    train_path = out_dir / f"{cond.id}_train.parquet"
    retime = retime_info(vdata.points, vdata.tables, cond)
    if cache and not force and oof_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("hash") == h:
            oof = pd.read_parquet(oof_path)
            ts = pd.read_parquet(train_path) if train_path.exists() else None
            res = evaluate(oof, P, cond, scorer=scorer,
                           context=scorer_context(vdata, cond, ts), retime=retime)
            return RunResult(cond, oof, res, h, meta.get("tuned", {}), cached=True, meta=meta)

    t0 = time.time()
    y_all = P[cond.target].to_numpy(float)
    trainable = train_mask(P, cond.target, readmit)
    grid = grid_mask(P, cond.target, readmit)
    predictable = trainable | grid
    cid = P["class_id"].to_numpy()
    # Outer folds: assigned once over ALL dev classes and inherited by this condition's
    # population / variant (§8.1), so tuning folds and condition folds are the same split.
    if cond.split == "vault":
        folds = outer_folds(vdata, cond.split, seed=seed)
        dev_folds = None
    else:
        dev_labels = fold_labels(dev, cond.split, seed=seed)
        folds = outer_folds(vdata, cond.split,
                            labels=inherit_folds(dev, dev_labels, vdata, cond.split))
        dev_folds = outer_folds(dev, cond.split, labels=dev_labels)
    # Hyperparameters: once per outer fold of this split, on F-all and the headline label,
    # over all dev classes (no population filter, no variant), then reused (§6, §1.3).
    needs_tuning = cond.model in models.TUNED_MODELS or cond.model == "M3+stack"
    if needs_tuning:
        head = config.HEADLINE_LABEL if config.HEADLINE_LABEL in dev.points else cond.target
        tune_ok = train_mask(dev.points, head)
        tune_y = dev.points[head].to_numpy(float)
        tune_folds = {0: np.ones(len(dev.points), bool)} if dev_folds is None else \
            {k: tr for k, tr, _ in dev_folds}
        tune_fp = dev.fingerprint()
        assert_tuning_disjoint({k: m & tune_ok for k, m in tune_folds.items()}, dev, folds,
                               vdata, cond.split)
    parts, train_parts, tuned_all = [], [], {}
    for k, tr_f, te_f in folds:
        tr, te, tg = tr_f & trainable, te_f & predictable, tr_f & predictable
        if cond.v("train_classes"):
            tr = _subsample_classes(tr, cid, int(cond.v("train_classes")), seed + k)
        if not tr.any() or not te.any():
            continue
        Xf = X
        if "hist" in models.featureset_groups(cond.featureset or "F-none"):
            hist = hist_columns(vdata, sorted(set(cid[tr])))
            Xf = X.assign(**{c: hist[c].to_numpy(np.float32) for c in config.HIST_COLUMNS})
        tuned = {}
        if needs_tuning and k in tune_folds:
            btr = tune_folds[k] & tune_ok
            tuned = get_tuned(cond.model, cond.split, k,
                              dev.X.iloc[np.flatnonzero(btr)].reset_index(drop=True),
                              tune_y[btr], fp=tune_fp, settings=settings,
                              tune_dir=tune_dir if cache else None, n_inner=n_inner,
                              hgb_max_iter=hgb_max_iter)
            tuned_all[k] = {m: {kk: vv for kk, vv in r.items() if kk != "curves"}
                            for m, r in tuned.items()}
        if cond.model == "M5":
            (p_raw, p_cal), inner, train_p = _fit_fold_gru(cond, vdata, P, tr, te, tg,
                                                           gru_kwargs)
        else:
            est = None
            if cond.model == "M6":
                from . import xenc
                est = xenc.CrossEncoderModel(**(gru_kwargs or {}))
            (p_raw, p_cal), inner, train_p = _fit_fold_sklearn(
                cond, cond.model, Xf, y_all, tr, te, tg, tuned, n_inner, estimator=est)
        parts.append(_oof_rows(P, te, k, y_all[te], p_raw, p_cal, False, grid))
        if inner is not None:                    # the training fold's whole event grid
            rows, ir, ic = inner
            parts.append(_oof_rows(P, rows, k, y_all[rows], ir, ic, True, grid))
        train_parts.append(_oof_rows(P, tg, k, y_all[tg], train_p[0], train_p[1], False, grid)
                           .drop(columns=["y", "inner"]))
    oof = pd.concat(parts, ignore_index=True) if parts else schemas.empty("oof")
    oof = schemas.validate(schemas.coerce(oof, "oof"), "oof", strict=True)
    train_scores = pd.concat(train_parts, ignore_index=True) if train_parts else None
    meta = {"hash": h, "condition": cond.to_dict(), "tuned": tuned_all,
            "seconds": round(time.time() - t0, 2), "n_rows": len(oof),
            "data_fingerprint": vfp}
    if "plant_visible" in vdata.tables:
        meta["plant"] = plant_summary(oof, P, vdata.tables["plant_visible"], cond.target)
    res = evaluate(oof, P, cond, scorer=scorer,
                   context=scorer_context(vdata, cond, train_scores), retime=retime)
    meta["scorer_extra"] = res.attrs.get("scorer_extra", {})
    if cache:
        out_dir.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(oof_path, index=False)
        if train_scores is not None:
            train_scores.to_parquet(train_path, index=False)
        meta_path.write_text(json.dumps(meta, indent=1, default=float))
    return RunResult(cond, oof, res, h, tuned_all, meta=meta)


def plant_summary(oof: pd.DataFrame, points: pd.DataFrame, visible: np.ndarray, target: str
                  ) -> dict:
    """P-plant stop rule: AP of test predictions on planted-visible positives vs all
    negatives must be >= 0.9."""
    key = pd.Series(np.asarray(visible, bool),
                    index=pd.MultiIndex.from_arrays([points["class_id"], points["t_s"]]))
    test = oof[~oof["inner"]]
    vis = key.reindex(pd.MultiIndex.from_arrays([test["class_id"], test["t_s"]])).fillna(
        False).to_numpy(bool)
    ok = test["y"].notna().to_numpy()
    ap = controls.planted_ap(test["y"].to_numpy()[ok], test["p_raw"].to_numpy()[ok], vis[ok])
    return {"planted_ap": ap, "pass": bool(np.isfinite(ap) and ap >= config.PLANT_AP_MIN),
            "n_planted_pos": int(((test["y"] == 1).to_numpy() & vis).sum())}


# --------------------------------------------------------------------------- evaluation


def _ap(y, p) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def onsets_per_class(df: pd.DataFrame) -> pd.Series:
    """K_c: bids per class among the scored rows (unique next_bid_s of positive rows, else
    runs of consecutive positive points)."""
    pos = df[df["y"] == 1]
    if "next_bid_s" in df and df["next_bid_s"].notna().any():
        return pos.groupby("class_id")["next_bid_s"].nunique()
    out = {}
    for c, g in df.sort_values(["class_id", "t_s"]).groupby("class_id"):
        y = (g["y"].to_numpy() == 1).astype(int)
        out[c] = int(((np.diff(np.concatenate([[0], y])) == 1)).sum())
    return pd.Series(out)


def ap_within(df: pd.DataFrame, col: str = "p_cal") -> tuple[float, float]:
    """(AP_w, macro AP): per-class AP over classes with >= 1 bid, weighted by K_c (§8.2)."""
    k = onsets_per_class(df)
    aps, w = [], []
    for c, g in df.groupby("class_id"):
        a = _ap(g["y"].to_numpy(), g[col].to_numpy())
        if np.isfinite(a) and k.get(c, 0) > 0:
            aps.append(a)
            w.append(k.get(c, 0))
    if not aps:
        return float("nan"), float("nan")
    return float(np.average(aps, weights=w)), float(np.mean(aps))


RESULT_FLOAT_COLUMNS = [c.name for c in schemas.schema("results") if c.dtype == "float64"]
RESULT_COLUMNS = set(RESULT_FLOAT_COLUMNS) | {"n_classes_shift_skipped"}


def _metric_row(df: pd.DataFrame, fold_aps: list | None = None) -> dict:
    """Point metrics of one fold or of the pooled test rows, all on Platt-calibrated `p_cal`
    (§8.2: pooled metrics never mix fold score scales; fold rows use the same column and
    the same K_c as the pooled row). ECE: equal-mass bins, ties never split."""
    y, pc = df["y"].to_numpy(float), df["p_cal"].to_numpy(float)
    pi = float(y.mean()) if len(y) else float("nan")
    ap = _ap(y, pc)
    brier = float(np.mean((pc - y) ** 2)) if len(y) else float("nan")
    ref = float(np.mean((pi - y) ** 2)) if len(y) else float("nan")
    apw, macro = ap_within(df, "p_cal")
    row = {"n_classes": int(df["class_id"].nunique()),
           "n_onsets": int(onsets_per_class(df).sum()), "n_points": len(df),
           "base_rate": pi, "ap": ap, "ap_lift": ap / pi if pi > 0 else float("nan"),
           "ap_within": apw, "macro_ap": macro, "roc_auc": metrics.roc_auc(y, pc),
           "brier": brier, "brier_skill": 1 - brier / ref if ref > 0 else float("nan"),
           "ece": metrics.ece(y, pc) if len(y) else float("nan")}
    if fold_aps is not None:
        fa = np.asarray([a for a in fold_aps if np.isfinite(a)], float)
        row["ap_fold_mean"] = float(fa.mean()) if len(fa) else float("nan")
        row["ap_fold_sd"] = float(fa.std(ddof=1)) if len(fa) > 1 else float("nan")
    return row


def scorer_context(data: Data, cond: Condition, train_scores: pd.DataFrame | None) -> dict:
    """Keyword context for the scorer: segments (local shift), the training-fold grid scores
    (rate-matched thresholds) and, for T-speech, the fresh speech onsets."""
    ctx = {"segments": data.tables.get("segments"), "train_scores": train_scores}
    if is_speech_target(cond.target):
        so = data.tables.get("speech_onsets")
        if so is not None and len(so):
            so = so[so["fresh"].fillna(False).astype(bool)] if "fresh" in so else so
            ctx["onsets"] = so[["class_id", "onset_s"]].reset_index(drop=True)
        else:
            ctx["onsets"] = pd.DataFrame({"class_id": [], "onset_s": []})
    return ctx


def metrics_scorer(oof, points, cond, **ctx) -> dict:
    """The default scorer: `metrics.condition_metrics` (within-class AP with its bootstrap
    CI, timing gains, event recall at the FAR budgets with rate-matched thresholds)."""
    from . import metrics

    return metrics.condition_metrics(oof, points, H=cond.H, g=cond.g, **{
        k: v for k, v in ctx.items() if v is not None})


def evaluate(oof: pd.DataFrame, points: pd.DataFrame, cond: str | Condition, *,
             scorer: Callable | None = None, context: dict | None = None,
             retime: dict | None = None) -> pd.DataFrame:
    """`results` rows (one per fold + one pooled, fold = <NA>) from OOF test rows. Point
    metrics are computed here; within-class CIs, shift, and event metrics come from
    `scorer(oof, points, cond, **context) -> dict` (default `metrics_scorer`), which receives
    the inner OOF rows as well. Scorer keys outside the results schema are kept in
    `.attrs["scorer_extra"]`. `retime` (`retime_info`, g < 2 conditions) fills
    retime_failed_share and g1_approximate on every row (A1)."""
    cond = get(cond)
    test = oof[~oof["inner"].astype(bool)]
    keys = points[["class_id", "t_s"] + (["next_bid_s"] if "next_bid_s" in points else [])]
    test = test.merge(keys, on=["class_id", "t_s"], how="left") if "next_bid_s" in keys \
        else test
    scored = test[test["y"].notna()]
    base = {"condition": cond.id, "tier": cond.tier, "model": cond.model,
            "featureset": cond.featureset, "split": cond.split, "target": cond.target,
            "H": float(cond.H), "g": float(cond.g)}
    rows, fold_aps = [], []
    for k, g in scored.groupby("fold"):
        r = {**base, "fold": int(k), **_metric_row(g)}
        fold_aps.append(_ap(g["y"].to_numpy(), g["p_cal"].to_numpy()))
        rows.append(r)
    pooled = {**base, "fold": pd.NA, **_metric_row(scored, fold_aps)}
    scorer = scorer if scorer is not None else metrics_scorer
    extra = {}
    if len(scored):
        extra = scorer(oof, points, cond, **(context or {})) or {}
        pooled.update({k: v for k, v in extra.items() if k in RESULT_COLUMNS})
    rows.append(pooled)
    out = pd.DataFrame(rows)
    for c in schemas.columns("results"):
        if c not in out:
            out[c] = np.nan
    out = out[schemas.columns("results")]
    for c in ("fold", "n_classes", "n_onsets", "n_points", "n_classes_shift_skipped"):
        out[c] = out[c].astype("Int64")
    for c in RESULT_FLOAT_COLUMNS:
        out[c] = out[c].astype(float)
    out["retime_failed_share"] = float((retime or {}).get("retime_failed_share", np.nan))
    flag = (retime or {}).get("g1_approximate", pd.NA)
    out["g1_approximate"] = pd.array([flag] * len(out), dtype="boolean")
    for c in ("condition", "tier", "model", "featureset", "split", "target"):
        out[c] = out[c].astype("string")
    out = schemas.validate(out, "results")
    out.attrs["scorer_extra"] = {k: v for k, v in extra.items() if k not in RESULT_COLUMNS}
    return out


def write_results(rows: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Merge rows into results.parquet, replacing earlier rows of the same conditions."""
    path = Path(path) if path is not None else config.table_path("results")
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["condition"].isin(set(rows["condition"]))]
        rows = pd.concat([old, rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    schemas.validate(rows, "results").to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- EV (S3)

EV_REFERENCES = {"R": "K0-R", "B-min": "K0-Bmin"}


def run_ev(points: pd.DataFrame, *, out_dir: Path | str | None = None,
           model_cond: str = "K0-M3", n_boot: int = config.N_BOOT,
           n_rep: int = config.SHIFT_REPEATS, seed: int = config.SEED,
           n_thresholds: int = 200, write: bool = True) -> pd.DataFrame:
    """The derived EV row (§7, S3): event recall of `model_cond` at each FAR budget with its
    class-bootstrap CI, against R's and B-min's recall read at the model's realised FAR on
    their FROC envelopes, the globally shifted predictions at matched FAR, the matched-rate
    random alarm and the AO point, with paired bootstrap deltas (model minus reference).
    Reads the cached OOF files (`run` must have produced them, with {model}_train.parquet
    for the rate-matched thresholds) and writes `{out_dir}/EV.parquet` (aggregate rows,
    schema `ev`)."""
    out_dir = Path(out_dir) if out_dir is not None else config.OOF_DIR
    path = out_dir / f"{model_cond}.parquet"
    if not path.exists():
        raise MissingProvider(f"EV needs {path.name}: run {model_cond} first")
    model = pd.read_parquet(path)
    train = out_dir / f"{model_cond}_train.parquet"
    ts = pd.read_parquet(train) if train.exists() else None
    refs = {name: pd.read_parquet(out_dir / f"{c}.parquet") for name, c in EV_REFERENCES.items()
            if (out_dir / f"{c}.parquet").exists()}
    cond = get(model_cond)
    ev = metrics.event_comparison(model, refs, points, condition=model_cond, H=cond.H,
                                  g=cond.g, train_scores=ts, n_boot=n_boot, n_rep=n_rep,
                                  seed=seed, n_thresholds=n_thresholds)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        ev.to_parquet(out_dir / "EV.parquet", index=False)
    return ev


# --------------------------------------------------------------------------- kNN (A5)


def knn_summary(data: Data, oof: pd.DataFrame, *, quantile: float = 0.95,
                block: str = "v60") -> pd.DataFrame:
    """A5 aggregate per outer fold: share of nearest training-class neighbours that preceded a
    bid, for true positives vs false alarms (alarms = p_cal above the fold's `quantile`; a
    descriptive cut, not the event threshold). Metadata only."""
    emb_cols = [c for c in data.X.columns if c.startswith(f"emb_{block}_")]
    if not emb_cols:
        raise MissingProvider("kNN needs emb_v60_* columns in the design")
    P, X = data.points, data.X
    test = oof[~oof["inner"]]
    rows = []
    seg = X["ctx_seg_type"].astype(object) if "ctx_seg_type" in X else pd.Series("other",
                                                                                 X.index)
    for k, g in test.groupby("fold"):
        te_cls = set(g["class_id"])
        trm = (~P["class_id"].isin(te_cls).to_numpy()
               & P[config.HEADLINE_LABEL].notna().to_numpy())
        if "in_vault" in P:
            trm &= ~P["in_vault"].fillna(False).astype(bool).to_numpy()
        meta = pd.DataFrame({"seg_type": seg[trm].to_numpy(),
                             "minute": (P["t_s"][trm] / 60).round().to_numpy(),
                             "bid_followed": (P[config.HEADLINE_LABEL][trm] == 1).to_numpy()})
        ex = models.KNNExplainer().fit(X.loc[trm, emb_cols].to_numpy(np.float32), meta)
        tau = g["p_cal"].quantile(quantile)
        key = pd.MultiIndex.from_arrays([P["class_id"], P["t_s"]])
        pos = pd.Series(np.arange(len(P)), index=key)
        idx = pos.reindex(pd.MultiIndex.from_arrays([g["class_id"], g["t_s"]])).to_numpy()
        alarm = (g["p_cal"] >= tau).to_numpy()
        tp, fa = idx[alarm & (g["y"] == 1).to_numpy()], idx[alarm & (g["y"] == 0).to_numpy()]
        res = models.knn_bid_share(ex, X.iloc[tp][emb_cols].to_numpy(np.float32),
                                   X.iloc[fa][emb_cols].to_numpy(np.float32))
        rows.append({"fold": int(k), **res})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- data loading


WINDOW_COLUMNS = {"T_all60": "txt_all60", "T_all15": "txt_all15", "T_chat60": "txt_chat60"}
WINDOW_VARIANTS = ("norm", "name", "punct")
LOCAL_TABLES = ("segments", "speech_onsets", "hand_episodes")


def build_windows(points: pd.DataFrame, words: pd.DataFrame, chat: pd.DataFrame | None = None,
                  classes: pd.DataFrame | None = None, *, variants=("norm",)) -> pd.DataFrame:
    """The design's txt_* columns (role-prefixed tokens of X_all60, X_all15, X_chat60) at every
    point, from `text.WindowBuilder` (words with end <= t - ε_w; roles once available; chat
    once t_s + ε_F <= t). Holds classmates' words: local tables only, never committed."""
    from . import text

    eps = {}
    if classes is not None and "eps_F_s" in classes:
        eps = classes.drop_duplicates("class_id").set_index("class_id")["eps_F_s"].to_dict()
    parts = []
    for cid, pts in points.groupby("class_id", sort=False):
        df = pts[["class_id", "t_s"]].reset_index(drop=True)
        w = words[words["class_id"] == cid]
        ch = chat[chat["class_id"] == cid] if chat is not None else None
        e = eps.get(cid)
        e = config.EPS_F_FALLBACK_S if e is None or not np.isfinite(e) else float(e)
        for v in variants:
            many = text.WindowBuilder(w, ch, eps_f=e, variant=v).many(df["t_s"])
            suffix = "" if v == "norm" else f"_{v}"
            for k, col in WINDOW_COLUMNS.items():
                df[col + suffix] = many[k].fillna("").astype(str).to_numpy()
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["class_id", "t_s"])


def steps_provider(tables_dir: Path | str) -> Callable:
    """Data.providers["steps"] for M5: raw causal step inputs per class from the frames
    table + frame embeddings (embed.py), words and the design's scalar features."""
    td = Path(tables_dir)
    cache: dict = {}

    def fn(data: Data, featureset: str) -> dict:
        from . import gru

        if featureset in cache:
            return cache[featureset]
        frames = pd.read_parquet(td / "frames.parquet")
        femb = np.load(td / "frames_emb.npy", mmap_mode="r")
        words = pd.read_parquet(td / "words.parquet", columns=[
            "class_id", "start_s", "end_s", "role", "role_avail_s"])
        out = {}
        for cid, idx in data.points.groupby("class_id").indices.items():
            fr = frames[(frames["class_id"] == cid) & frames["emb_row"].notna()]
            fe = np.asarray(femb[fr["emb_row"].astype(int).to_numpy()], np.float32)
            out[cid] = gru.raw_steps(cid, data.points["t_s"].to_numpy(float)[idx],
                                     features=data.X.iloc[idx], frames=fr, frame_emb=fe,
                                     words=words[words["class_id"] == cid],
                                     featureset=featureset)
        cache[featureset] = out
        return out
    return fn


def load_data(*, with_emb: bool = False, windows=("norm",), with_steps: bool = False,
              emb_matrix_path: Path | str | None = None,
              tables_dir: Path | str | None = None) -> Data:
    """Data from data/tables: points + features, the txt_* windows (read from
    windows.parquet, or built from words/chat and cached there), the pooled point
    embeddings (features_emb.npy, by emb_row) when `with_emb`, and the M5 steps provider."""
    td = Path(tables_dir) if tables_dir else config.TABLES_DIR
    need = {"points": td / "points.parquet", "features": td / "features.parquet"}
    for path in need.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `oncue labels` / `oncue features`")
    points = pd.read_parquet(need["points"]).sort_values(["class_id", "t_s"], kind="stable")
    points = points.reset_index(drop=True)
    classes = pd.read_parquet(td / "classes.parquet") if (td / "classes.parquet").exists() \
        else None
    X = points[["class_id", "t_s"]].merge(pd.read_parquet(need["features"]),
                                          on=["class_id", "t_s"], how="left")
    if windows:
        win_path = td / "windows.parquet"
        win = pd.read_parquet(win_path) if win_path.exists() else None
        want = [c + ("" if v == "norm" else f"_{v}") for v in windows
                for c in WINDOW_COLUMNS.values()]
        if win is None or not set(want) <= set(win.columns):
            if not (td / "words.parquet").exists():
                raise FileNotFoundError("text windows need words.parquet (oncue words)")
            chat = pd.read_parquet(td / "chat.parquet") if (td / "chat.parquet").exists() \
                else None
            win = build_windows(points, pd.read_parquet(td / "words.parquet"), chat, classes,
                                variants=tuple(windows))
            win.to_parquet(win_path, index=False)
        X = X.merge(win[["class_id", "t_s"] + want], on=["class_id", "t_s"], how="left")
    if with_emb:
        path = Path(emb_matrix_path) if emb_matrix_path else td / "features_emb.npy"
        mat = np.load(path, mmap_mode="r")
        X = pd.concat([X.reset_index(drop=True),
                       models.emb_frame(mat, X["emb_row"], dim=mat.shape[1] // 4)], axis=1)
    tables = {n: pd.read_parquet(td / f"{n}.parquet") for n in LOCAL_TABLES
              if (td / f"{n}.parquet").exists()}
    providers = {"steps": steps_provider(td)} if with_steps else {}
    return Data(points=points, X=X.reset_index(drop=True), classes=classes,
                providers=providers, tables=tables)


def toy_data(n_classes: int = 12, minutes: float = 20.0, *, seed: int = config.SEED,
             emb_dim: int = 8, signal: float = 1.0, vault_classes: int = 0) -> Data:
    """A small synthetic design (invented values, no words) for tests and `--demo`: bids
    follow open professor questions, and the text windows carry an invitation phrase before
    most bids. Every label column of `points` is filled from the same planted onsets."""
    rng = np.random.default_rng(seed)
    rows_p, rows_x, cls = [], [], []
    vocab = np.array(["p_the", "p_model", "p_data", "s_i", "s_think", "s_maybe", "p_so",
                      "c_ok", "p_next", "s_because"])
    for c in range(n_classes):
        cid = 900_000 + c
        T_end = minutes * 60
        grid = np.arange(config.T_OPEN_S, T_end - config.H_S + 1e-9, config.DELTA_S)
        n_bids = rng.poisson(max(1.0, minutes / 4))
        onsets = np.sort(rng.uniform(config.T_OPEN_S + 20, T_end - 5, n_bids))
        q_times = np.sort(np.concatenate([onsets - rng.uniform(3, 9, n_bids) * signal
                                          if signal else np.array([]),
                                          rng.uniform(0, T_end, max(1, n_bids // 2))]))
        course = f"CS{c % 3}"
        cls.append({"class_id": cid, "section_id": c % 4, "course_code": course,
                    "date": f"2025-{1 + c % 12:02d}-{1 + c:02d}", "semester_idx": 1 + c % 7,
                    "n_bids": n_bids, "pcw_status": "verified" if c % 3 else "none",
                    "template_validated": True, "qa_reason": "ok", "qa_pass": True})
        emb_bias = rng.normal(size=emb_dim)
        for t in grid:
            nxt = onsets[onsets > t]
            labels = {}
            for col in config.LABEL_COLUMNS:
                H, g = (15, 3) if col.startswith("y_speech") else (
                    int(col.split("_H")[1].split("_")[0]), int(col.split("_g")[1].split("_")[0]))
                guard = ((onsets > t) & (onsets <= t + g)).any()
                pos = ((onsets > t + g) & (onsets <= t + H)).any()
                labels[col] = np.nan if guard else float(pos)
            hand_up = ((onsets <= t) & (t < onsets + 20)).any()   # hand up, then lowered
            in_g = bool(rng.random() > 0.1 and not hand_up)
            past_q = q_times[q_times <= t - config.EPS_W_S]
            q_age = t - past_q[-1] if len(past_q) else np.inf
            q_open = float(q_age < 12)
            toks = list(rng.choice(vocab, 8))
            if q_age < 15:
                toks += ["p_anyone", "p_thoughts"]
            past_b = onsets[onsets <= t - 10]
            rows_p.append({"class_id": cid, "t_s": float(t), "in_G": in_g,
                           "at_risk": in_g and np.isfinite(labels[config.HEADLINE_LABEL]),
                           "excl_reason": None if in_g else (
                               "hand_up" if hand_up else "me_speaking"),
                           "label_state": "pos" if labels[config.HEADLINE_LABEL] == 1 else
                           "neg", **labels,
                           "next_bid_s": float(nxt[0]) if len(nxt) else np.nan,
                           "fold": pd.NA, "in_vault": c >= n_classes - vault_classes})
            x = {"class_id": cid, "t_s": float(t), "ctx_min": t / 60, "ctx_course": course,
                 "ctx_seg_type": rng.choice(["debrief", "intro", "activity"]),
                 "ctx_area": "CS", "spk_silence_s": float(rng.exponential(4)),
                 "spk_cur_role": rng.choice(["prof", "student"]),
                 "me_since_turn_s": float(t - past_b[-1]) if len(past_b) else np.nan,
                 "me_turns_so_far": float(len(past_b)),
                 "me_hand_n_so_far": float(len(past_b)),
                 "me_hand_since_down_s": float(t - past_b[-1]) if len(past_b) else np.nan,
                 "poll_n_so_far": 0.0, "q_open": q_open,
                 "q_age_s": float(min(q_age, config.Q_AGE_CAP_S)),
                 "q_inv_age_s": float(min(q_age, config.Q_AGE_CAP_S)),
                 "q_kind": "invite" if q_open else "plain",
                 "n_words_60": float(rng.poisson(120)), "wpm_60": float(rng.normal(140, 10)),
                 "pcw_tmpl_has": 1.0, "pcw_res_Q": float(rng.normal()),
                 "pcw_has": float(c % 3 != 0), "pcw_res_A": float(rng.normal()),
                 "txt_all60": " ".join(toks), "txt_all15": " ".join(toks[-5:]),
                 "txt_chat60": "c_ok" if rng.random() < 0.3 else ""}
            for w in models.TEXT_WINDOWS:
                for var in ("name", "punct"):
                    x[models.text_column(w, var)] = x[w]
            vec = emb_bias + rng.normal(size=(4, emb_dim)) + q_open
            for b, block in enumerate(models.EMB_BLOCKS):
                for i in range(emb_dim):
                    x[f"emb_{block}_{i:03d}"] = np.float32(vec[b, i])
            rows_x.append(x)
    points = pd.DataFrame(rows_p)
    for col in config.LABEL_COLUMNS:
        points[col] = points[col].astype(np.float32)
    points["fold"] = points["fold"].astype("Int64")
    classes = pd.DataFrame(cls)
    data = Data(points=points, X=pd.DataFrame(rows_x), classes=classes)
    folds = outer_folds(data.take(~points["in_vault"].to_numpy()), "S-group", seed=seed)
    fold_of = {}
    dev = data.take(~points["in_vault"].to_numpy())
    for k, _, te in folds:
        fold_of.update({c: k for c in dev.points.loc[te, "class_id"]})
    data.points["fold"] = data.points["class_id"].map(fold_of).astype("Int64")
    return data


# --------------------------------------------------------------------------- CLI


def _parse(extra) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="oncue train/evaluate")
    p.add_argument("--cond", action="append", default=None, help="condition id (repeatable)")
    p.add_argument("--tier", choices=TIERS, default=None)
    p.add_argument("--up-to", action="store_true", help="include every tier up to --tier")
    p.add_argument("--list", action="store_true", help="list conditions and exit")
    p.add_argument("--dry-run", action="store_true", help="print what would run")
    p.add_argument("--force", action="store_true", help="ignore cached OOF")
    p.add_argument("--open-vault", action="store_true", help="V conditions (once, after G4)")
    p.add_argument("--demo", action="store_true", help="toy synthetic data, temp output")
    p.add_argument("--out", default=None, help="OOF directory (default data/oof)")
    p.add_argument("--hgb-max-iter", type=int, default=None,
                   help=f"HGB boosting rounds cap (default {config.HGB_MAX_ITER}; quick runs)")
    return p.parse_args(list(extra or []))


def _list(conds) -> None:
    for c in conds:
        print(f"{c.id:24s} {c.tier:6s} {c.kind:9s} {c.group:10s} {c.model or '-':9s} "
              f"{c.featureset or '-':18s} {c.split:8s} {c.target}")


def qa_undecided(data: Data) -> int:
    """Number of classes in `data.points` whose classes.qa_pass is missing or NA."""
    ids = set(data.points["class_id"])
    if data.classes is None or "qa_pass" not in data.classes:
        return len(ids)
    known = data.classes.loc[data.classes["qa_pass"].notna(), "class_id"]
    return len(ids - set(known))


def cli_train(args) -> int:
    """`python -m oncue train [--cond ID ... | --tier T [--up-to]] [--list] [--dry-run]`."""
    o = _parse(getattr(args, "extra", []))
    conds = select(o.cond, o.tier, kind=None if o.list else "model", up_to=o.up_to)
    if o.list or o.dry_run:
        _list(conds)
        return 0
    if o.demo:
        import tempfile
        data = toy_data()
        out = Path(o.out or tempfile.mkdtemp(prefix="oncue_demo_"))
        conds = [c for c in conds if c.split != "vault" and not c.variant and
                 c.model not in ("M5", "M6") and c.population == "dev"]
    else:
        needs_emb = any(c.model in ("M3", "M4", "M4-MLP", "M3+stack") and
                        "emb" in models.featureset_groups(c.featureset) for c in conds)
        variants = ["norm"] + [v for v in ("name", "punct") if any(
            models.featureset_text_variant(c.featureset) == v for c in conds)]
        data = load_data(with_emb=needs_emb, windows=tuple(variants),
                         with_steps=any(c.model == "M5" for c in conds))
        out = Path(o.out) if o.out else config.OOF_DIR
        qa_open = qa_undecided(data)
        if qa_open:
            print(f"WARNING: {qa_open} classes in points have no class-QA verdict "
                  "(qa_pass is NA): run `oncue qa` before `oncue labels`, or they are "
                  "modelled unchecked (§3 class QA)")
    all_rows = []
    want_ev = (not o.demo and o.cond is not None and "EV" in o.cond)
    for c in conds:
        if c.split == "vault" and not o.open_vault:
            print(f"skip {c.id}: vault (needs --open-vault)")
            continue
        try:
            res = run(c, data, out_dir=out, force=o.force, allow_vault=o.open_vault,
                      hgb_max_iter=o.hgb_max_iter or (60 if o.demo else config.HGB_MAX_ITER))
        except MissingProvider as exc:
            print(f"skip {c.id}: {exc}")
            continue
        pooled = res.results[res.results["fold"].isna()].iloc[0]
        print(f"{c.id:24s} AP={pooled['ap']:.3f} AP_w={pooled['ap_within']:.3f} "
              f"{'(cached)' if res.cached else ''}")
        all_rows.append(res.results)
    if all_rows and not o.demo:
        write_results(pd.concat(all_rows, ignore_index=True))
    if want_ev:
        _ev_cli(data.points, out)
    return 0


def _ev_cli(points: pd.DataFrame, out: Path) -> None:
    try:
        ev = run_ev(points, out_dir=out)
    except MissingProvider as exc:
        print(f"skip EV: {exc}")
        return
    head = ev[(ev["reference"] == "model")]
    for r in head.itertuples():
        print(f"EV budget {r.budget}: recall {r.recall:.3f} [{r.recall_lo:.3f}, "
              f"{r.recall_hi:.3f}] at FAR {r.far:.3f}; S3 {'pass' if r.s3_pass else 'fail'}")


def cli_evaluate(args) -> int:
    """`python -m oncue evaluate`: re-score cached OOF files into results.parquet."""
    o = _parse(getattr(args, "extra", []))
    out = Path(o.out) if o.out else config.OOF_DIR
    conds = select(o.cond, o.tier, kind="model", up_to=o.up_to)
    if o.list or o.dry_run:
        _list(conds)
        return 0
    points = pd.read_parquet(config.table_path("points"))
    tables = {n: pd.read_parquet(config.table_path(n)) for n in LOCAL_TABLES
              if config.table_path(n).exists()}
    data_like = Data(points=points.iloc[:0], X=points.iloc[:0][["class_id", "t_s"]],
                     tables=tables)
    rows = []
    for c in conds:
        path, train = out / f"{c.id}.parquet", out / f"{c.id}_train.parquet"
        if not path.exists():
            continue
        ts = pd.read_parquet(train) if train.exists() else None
        if ts is None:
            print(f"{c.id}: no {train.name}; event thresholds stay NaN (never set on test)")
        rows.append(evaluate(pd.read_parquet(path), points, c,
                             context=scorer_context(data_like, c, ts),
                             retime=retime_info(points, tables, c)))
    if rows:
        write_results(pd.concat(rows, ignore_index=True))
    print(f"evaluated {len(rows)} conditions")
    if o.cond is None or "EV" in o.cond:
        _ev_cli(points, out)
    return 0
