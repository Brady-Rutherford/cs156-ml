"""Aggregates for the run notebook (`pipeline1.ipynb`), computable from the committed bundle.

Every function returns counts, curves or statistics only (no words, no raw ids), so the
notebook's metric, figure and table cells re-run for a grader who has `results/` but no
`data/`:

* `provenance()`: git and `prereg` hashes, the installed Scribe SHA, package versions;
* `load_bundle()`: results.parquet, oof_*.parquet, bids.parquet and funnel.csv;
* `hypothesis_table()`: H1-H5 (§7) as paired cluster-bootstrap deltas of within-class AP
  (H2 as local timing gains from the segments table, or from the committed
  hypotheses.parquet when re-run from the bundle), with fixed-sequence Holm gatekeeping at
  one-sided alpha / 2 (reject exactly when the 95% CI shown excludes 0), the §5.6 PCW
  coverage rule for H3, and n_classes / n_onsets / base_rate on every row, in the shape
  `plots.fig_forest` draws; `write_hypotheses()` commits it;
* `froc_curves()`: FROC curves from OOF rows that carry `in_G` plus the bid onsets;
* `peri_onset_curves()`: mean causal features around bid onsets vs a class-shifted control
  (local tables only);
* `summary()`: the headline numbers for `figures/summary.json`.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics

# hypothesis -> (condition a, condition b): delta = AP_w(a) - AP_w(b) on shared test points
CONTRASTS: dict[str, tuple[str, str]] = {
    "H1a": ("K0-M3", "F0-Fstruct"),
    "H1b": ("F0-Fall-me_hand", "F0-Fstruct-me_hand"),
    "H3": ("F1-Fall+pcwA", "F1-Fall-pcw"),
    "H4": ("K0-M3", "K0-R"),
    "H5": ("K1-M4", "K0-M2"),
}
H2_CONDITIONS = ("K0-M3", "F0-Fstruct")
H3_CONDITIONS = ("F1-Fall+pcwA", "F1-Fall-pcw")
HYPOTHESES_FILE = "hypotheses.parquet"
H2_NOT_COMPUTABLE = "not computable here (needs the local segments table)"
PACKAGES = ("numpy", "pandas", "scikit-learn", "torch", "sentence-transformers",
            "transformers", "setfit", "opencv-python-headless", "pyarrow", "matplotlib",
            "minervas-scribe")


# --------------------------------------------------------------------------- provenance


def _git(*args: str, cwd: Path = config.REPO_DIR) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                             timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def scribe_sha() -> str | None:
    """The installed Scribe commit (direct_url.json of `minervas-scribe`), or a note when it
    is an unrecorded editable install (A7: final results need a pushed SHA)."""
    try:
        dist = metadata.distribution("minervas-scribe")
    except metadata.PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    if not raw:
        return f"{dist.version} (no direct_url.json)"
    info = json.loads(raw)
    vcs = info.get("vcs_info", {})
    if vcs.get("commit_id"):
        return vcs["commit_id"]
    if info.get("dir_info", {}).get("editable"):
        return "editable install (unrecorded; not a pushed SHA)"
    return info.get("url")


def provenance() -> dict:
    """Read-only facts about the code that produced the results (no network)."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"git_head": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--", "pipeline1")),
            "prereg": _git("rev-parse", "--verify", "--quiet", "prereg^{commit}"),
            "scribe": scribe_sha(), "oncue": _oncue_version(), "packages": versions}


def _oncue_version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------------- the bundle


@dataclass
class Bundle:
    results: pd.DataFrame | None = None
    oofs: dict[str, pd.DataFrame] = field(default_factory=dict)
    bids: pd.DataFrame | None = None
    funnel: pd.DataFrame | None = None
    hypotheses: pd.DataFrame | None = None       # hypotheses.parquet (H2 needs local data)
    ev: pd.DataFrame | None = None               # EV.parquet (S3)

    @property
    def empty(self) -> bool:
        return self.results is None and not self.oofs


def load_bundle(results_dir: Path | str | None = None) -> Bundle:
    """The committed results/ bundle (HMAC class ids). Missing files are None/empty."""
    d = Path(results_dir) if results_dir is not None else config.RESULTS_DIR
    if not d.exists():
        return Bundle()
    oofs = {p.stem[len("oof_"):]: pd.read_parquet(p) for p in sorted(d.glob("oof_*.parquet"))}
    read = (lambda name: pd.read_parquet(d / name) if (d / name).exists() else None)
    return Bundle(results=read("results.parquet"), oofs=oofs, bids=read("bids.parquet"),
                  funnel=pd.read_csv(d / "funnel.csv") if (d / "funnel.csv").exists()
                  else None, hypotheses=read(HYPOTHESES_FILE), ev=read("EV.parquet"))


def onsets_from_bids(bids: pd.DataFrame | None) -> pd.DataFrame:
    """(class_id, onset_s) of bids from bids.parquet (class_hash) or hand_episodes."""
    if bids is None or not len(bids):
        return pd.DataFrame(columns=["class_id", "onset_s"])
    b = bids[bids["ep_type"] == config.EP_BID]
    if "left_censored" in b:
        b = b[~b["left_censored"].fillna(False).astype(bool)]
    col = "class_hash" if "class_hash" in b else "class_id"
    return b.rename(columns={col: "class_id"})[["class_id", "onset_s"]].reset_index(drop=True)


def held_out(oof: pd.DataFrame) -> pd.DataFrame:
    """Outer-fold test predictions (inner rows dropped)."""
    return oof[~oof["inner"].astype(bool)].reset_index(drop=True)


# --------------------------------------------------------------------------- hypotheses


def _merged(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Defined held-out test points that both conditions predicted (y from a)."""
    m = held_out(a).merge(held_out(b), on=["class_id", "t_s"], suffixes=("_a", "_b"))
    return m[m["y_a"].notna()].reset_index(drop=True)


def bid_weights(rows: pd.DataFrame, onsets: pd.DataFrame | None, y_col: str = "y"
                ) -> pd.Series:
    """K_c over the scored rows, defined as in `metrics.bid_counts` / the results table:
    the distinct bids with at least one positive scored point (each positive point counts
    toward the first onset after it). Without onsets, runs of consecutive positive points."""
    pos = rows[rows[y_col] == 1]
    out = {}
    if onsets is not None and len(onsets):
        on = onsets.assign(class_id=onsets["class_id"].astype(str))
        by = {c: np.sort(g["onset_s"].to_numpy(float)) for c, g in on.groupby("class_id")}
        for c, g in pos.groupby(pos["class_id"].astype(str)):
            o = by.get(c, np.zeros(0))
            k = np.searchsorted(o, g["t_s"].to_numpy(float), side="right")
            out[c] = len(np.unique(k[k < len(o)]))
    else:
        for c, g in rows.sort_values(["class_id", "t_s"]).groupby(
                rows["class_id"].astype(str)):
            y = (g[y_col].to_numpy() == 1).astype(int)
            out[c] = int((np.diff(np.concatenate([[0], y])) == 1).sum())
    cls = pd.unique(rows["class_id"].astype(str))
    return pd.Series(out, dtype=float).reindex(cls, fill_value=0.0)


def _paired_ap(m: pd.DataFrame, weights: pd.Series, n_boot: int, seed: int):
    """Paired bootstrap of AP_w(a) - AP_w(b) on the merged rows (same K_c both sides)."""
    if not len(m) or not (m["y_a"] == 1).any():
        return None
    cls = np.unique(m["class_id"].astype(str))
    g = m["class_id"].astype(str).to_numpy()
    sa = metrics.WithinAP(m["y_a"], m["p_cal_a"], g, weights=weights, classes=cls)
    sb = metrics.WithinAP(m["y_a"], m["p_cal_b"], g, weights=weights, classes=cls)
    return metrics.paired_delta(sa, sb, n_boot=n_boot, seed=seed)


def _tg_local(oof: pd.DataFrame, segments: pd.DataFrame, n_rep: int, seed: int):
    t = held_out(oof)
    t = t[t["y"].notna()].sort_values(["class_id", "t_s"])
    seg = metrics.segment_index(t["t_s"], t["class_id"], segments)
    return metrics.TimingGain(t["y"], t["p_cal"], t["class_id"], t=t["t_s"], segments=seg,
                              kind="local", n_rep=n_rep, seed=seed)


def _sizes(rows: pd.DataFrame, weights: pd.Series | None, y_col: str = "y") -> dict:
    """n_classes, n_onsets (sum of K_c over classes with positives) and base rate (§3, §8.5:
    every reported number carries them)."""
    if rows is None or not len(rows):
        return {"n_classes": 0, "n_onsets": 0, "base_rate": np.nan}
    k = weights if weights is not None else bid_weights(rows, None, y_col)
    return {"n_classes": int(rows["class_id"].nunique()), "n_onsets": int(k.sum()),
            "base_rate": float(rows[y_col].mean())}


def hypothesis_table(oofs: dict[str, pd.DataFrame], bids: pd.DataFrame | None = None, *,
                     segments: pd.DataFrame | None = None, h2: pd.DataFrame | None = None,
                     n_boot: int = config.N_BOOT, n_rep: int = config.SHIFT_REPEATS,
                     seed: int = config.SEED) -> pd.DataFrame:
    """H1-H5 rows: hypothesis, contrast, estimate, lo, hi (95% percentile CI), p (one-sided
    bootstrap P(delta* <= 0)), mde, n_classes, n_onsets, base_rate, note, then in_family /
    p_adj / reject from fixed-sequence gatekeeping (H1, then Holm over H2-H5).

    * Decision rule (§7, §8.5): one-sided p < ALPHA_ONE_SIDED (= 0.025), i.e. a hypothesis
      rejects exactly when the 95% CI shown lies above 0 (Holm p_adj >= p keeps this).
    * K_c for AP_w is the results-table definition (bids with a positive scored point,
      `bid_weights`), so each estimate is the difference of AP_w values as reported.
    * H1 is the intersection-union of H1a and H1b; H2 of H2a and H2b (local timing gains,
      from `segments`, or the precomputed rows `h2` of the committed hypotheses.parquet).
      When neither is available, H2 is marked "not computable here" and the family's
      p_adj / reject stay empty for H2-H5: the pre-registered family never shrinks.
    * H3 leaves the family (exploratory) under the §5.6 coverage rule: fewer than 40 PCW
      classes or fewer than 150 bids in its contrast.
    A hypothesis whose conditions are missing gets NaN and leaves the family."""
    onsets = onsets_from_bids(bids)
    on = onsets if len(onsets) else None
    rows = []
    for h, (a, b) in CONTRASTS.items():
        note = ""
        if a in oofs and b in oofs:
            m = _merged(oofs[a], oofs[b])
            w = bid_weights(m, on, "y_a") if len(m) else None
            r = _paired_ap(m, w, n_boot, seed) if len(m) else None
            size = _sizes(m.rename(columns={"y_a": "y"}), w)
        else:
            r, size = None, _sizes(None, None)
        if h == "H3" and r is not None and (size["n_classes"] < config.H3_MIN_CLASSES
                                            or size["n_onsets"] < config.H3_MIN_BIDS):
            note = (f"exploratory: coverage rule (§5.6) needs >= {config.H3_MIN_CLASSES} "
                    f"classes and >= {config.H3_MIN_BIDS} bids")
        rows.append({"hypothesis": h, "contrast": f"AP_w({a}) - AP_w({b})", **_boot_cols(r),
                     **size, "note": note})
    h2_rows = _h2_rows(oofs, segments, h2, on, n_boot, n_rep, seed)
    rows += h2_rows
    df = pd.DataFrame(rows)
    p = dict(zip(df["hypothesis"], df["p"], strict=True))
    notes = dict(zip(df["hypothesis"], df["note"], strict=True))
    h2_unknown = any(r.get("note") == H2_NOT_COMPUTABLE for r in h2_rows)
    family = {"H1": _iut(p, "H1a", "H1b"), "H2": _iut(p, "H2a", "H2b"),
              "H3": None if notes.get("H3") else p.get("H3"), "H4": p.get("H4"),
              "H5": p.get("H5")}
    df["alpha"] = config.ALPHA_ONE_SIDED
    if family["H1"] is None or not np.isfinite(family["H1"]):
        df["in_family"], df["p_adj"] = False, np.nan
        df["reject"] = pd.array([pd.NA] * len(df), dtype="boolean")
        return df
    gate = metrics.fixed_sequence_holm(family, alpha=config.ALPHA_ONE_SIDED, strict=True)
    top = df["hypothesis"].str[:2]
    df["in_family"] = top.map(gate["in_family"]).fillna(False).astype(bool)
    df["p_adj"] = top.map(gate["p_adj"]).astype(float)
    df["reject"] = pd.array(top.map(gate["reject"]).fillna(False).astype(bool),
                            dtype="boolean")
    if h2_unknown:                                    # refuse a shrunken family
        later = top.isin(["H2", "H3", "H4", "H5"]).to_numpy()
        df.loc[later, "p_adj"] = np.nan
        df.loc[later, "reject"] = pd.NA
        df.loc[later & (df["note"] == ""), "note"] = "family incomplete: H2 " \
            + H2_NOT_COMPUTABLE
    return df


def _h2_rows(oofs, segments, h2, onsets, n_boot, n_rep, seed) -> list[dict]:
    """H2a / H2b rows: computed from the segments table, taken from precomputed rows, or
    marked not computable."""
    if not all(c in oofs for c in H2_CONDITIONS):
        return []
    a = held_out(oofs[H2_CONDITIONS[0]])
    a = a[a["y"].notna()]
    size = _sizes(a, bid_weights(a, onsets) if len(a) else None)
    labels = {"H2a": f"TG_local({H2_CONDITIONS[0]})",
              "H2b": "TG_local({}) - TG_local({})".format(*H2_CONDITIONS)}
    if segments is not None:
        ta, tb = (_tg_local(oofs[c], segments, n_rep, seed) for c in H2_CONDITIONS)
        if not np.array_equal(ta.classes, tb.classes):      # cannot pair: never drop H2
            return [{"hypothesis": h, "contrast": labels[h], **_boot_cols(None), **size,
                     "note": H2_NOT_COMPUTABLE} for h in ("H2a", "H2b")]
        return [{"hypothesis": "H2a", "contrast": labels["H2a"], "note": "", **size,
                 **_boot_cols(metrics.cluster_bootstrap(ta, n_boot=n_boot, seed=seed))},
                {"hypothesis": "H2b", "contrast": labels["H2b"], "note": "", **size,
                 **_boot_cols(metrics.paired_delta(ta, tb, n_boot=n_boot, seed=seed))}]
    if h2 is not None and len(h2) and {"H2a", "H2b"} <= set(h2["hypothesis"]):
        keep = ["hypothesis", "contrast", "estimate", "lo", "hi", "p", "mde", "n_classes",
                "n_onsets", "base_rate"]
        pre = h2[h2["hypothesis"].isin(["H2a", "H2b"])]
        return [{**{k: r.get(k, np.nan) for k in keep}, "note": "precomputed locally"}
                for r in pre.to_dict("records")]
    return [{"hypothesis": h, "contrast": labels[h], **_boot_cols(None), **size,
             "note": H2_NOT_COMPUTABLE} for h in ("H2a", "H2b")]


def write_hypotheses(table: pd.DataFrame, results_dir: Path | str | None = None) -> Path:
    """Commit the hypothesis table (aggregate numbers only) as results/hypotheses.parquet,
    so a bundle-only re-run keeps H2 (which needs the local segments table) in the family."""
    d = Path(results_dir) if results_dir is not None else config.RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / HYPOTHESES_FILE
    out = table.copy()
    out["reject"] = out["reject"].astype("boolean") if "reject" in out else pd.NA
    out.to_parquet(path, index=False)
    return path


def _iut(p: dict, *keys: str) -> float | None:
    vals = [p.get(k) for k in keys]
    if any(v is None or not np.isfinite(v) for v in vals):
        return None
    return metrics.iut_p(*vals)


def _boot_cols(r) -> dict:
    if r is None:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "mde": np.nan}
    return {"estimate": r.point, "lo": r.lo, "hi": r.hi, "p": r.p_le0, "mde": r.mde}


# --------------------------------------------------------------------------- event curves


# FROC display names: plots.fig_froc colours by model and greys out the references
FROC_NAMES = {"K0-M3": "M3 / F-all", "K0-Bmin": "B-min", "K0-R": "R", "K1-M4": "M4 / F-all",
              "K0-M2": "M2 / F-all", "K0-B1": "B1 / F-struct"}


def froc_curves(oofs: dict[str, pd.DataFrame], bids: pd.DataFrame | None,
                conditions=("K0-M3", "K0-Bmin", "K0-R"), *, H: float = config.H_S,
                g: float = config.G_S, far_max: float = config.FROC_FAR_MAX
                ) -> dict[str, pd.DataFrame]:
    """display name (FROC_NAMES) -> DataFrame[far, recall] for each condition whose OOF
    rows carry `in_G`."""
    onsets = onsets_from_bids(bids)
    out = {}
    for cond in conditions:
        if cond not in oofs or "in_G" not in oofs[cond]:
            continue
        t = held_out(oofs[cond])
        data = metrics.EventData.from_frames(t, onsets, y_col="y", H=H, g=g)
        cur = metrics.froc(data, t["p_cal"].to_numpy(float), far_stop=2 * far_max)
        out[FROC_NAMES.get(cond, cond)] = cur[["far", "recall"]]
    return out


def always_on_point(oof: pd.DataFrame, bids: pd.DataFrame | None, *, H: float = config.H_S,
                    g: float = config.G_S) -> tuple[float, float] | None:
    """(FAR, recall) of the always-on predictor on the same grid (the AO reference point)."""
    if "in_G" not in oof:
        return None
    t = held_out(oof)
    data = metrics.EventData.from_frames(t, onsets_from_bids(bids), y_col="y", H=H, g=g)
    s = data.stats(np.ones(len(t)), 0.5).summary()
    return float(s["far"]), float(s["recall"])


# --------------------------------------------------------------------------- descriptives


def peri_onset_curves(features: pd.DataFrame, onsets: pd.DataFrame, columns, *,
                      span=(-120.0, 30.0), step: float = config.DELTA_S,
                      seed: int = config.SEED) -> pd.DataFrame:
    """Mean of causal feature columns at rel_s = t - onset in `span`, around bid onsets vs
    the same class with each onset moved to a random time (shifted control).
    DataFrame[feature, rel_s, mean, shifted_mean] (aggregate; `plots.fig_peri_onset`)."""
    rng = np.random.default_rng(seed)
    rel = np.arange(span[0], span[1] + 1e-9, step)
    rows = []
    for cid, f in features.groupby("class_id"):
        ons = onsets.loc[onsets["class_id"] == cid, "onset_s"].to_numpy(float)
        if not len(ons) or not len(f):
            continue
        t = f["t_s"].to_numpy(float)
        order = np.argsort(t)
        t = t[order]
        fake = rng.uniform(t.min() - span[0], max(t.max() - span[1], t.min() - span[0] + 1),
                           len(ons))
        for kind, centers in (("mean", ons), ("shifted_mean", fake)):
            for o in centers:
                k = np.searchsorted(t, o + rel)
                ok = (k < len(t)) & (np.abs(t[np.minimum(k, len(t) - 1)] - (o + rel)) <= step)
                for col in columns:
                    v = pd.to_numeric(f[col], errors="coerce").to_numpy(float)[order]
                    vals = np.where(ok, v[np.minimum(k, len(t) - 1)], np.nan)
                    rows.append(pd.DataFrame({"feature": col, "rel_s": rel, "kind": kind,
                                              "value": vals}))
    if not rows:
        return pd.DataFrame(columns=["feature", "rel_s", "mean", "shifted_mean"])
    d = pd.concat(rows, ignore_index=True)
    out = d.pivot_table(index=["feature", "rel_s"], columns="kind", values="value",
                        aggfunc="mean").reset_index()
    for c in ("mean", "shifted_mean"):
        if c not in out:
            out[c] = np.nan
    return out[["feature", "rel_s", "mean", "shifted_mean"]]


# --------------------------------------------------------------------------- summary


def summary(results: pd.DataFrame | None, hypotheses: pd.DataFrame | None = None,
            funnel: pd.DataFrame | None = None, ev: pd.DataFrame | None = None) -> dict:
    """Headline numbers for figures/summary.json (pooled rows, aggregate only)."""
    out: dict = {}
    if results is not None and len(results):
        pooled = results[results["fold"].isna()].set_index("condition")
        keep = ["ap_within", "ap_within_lo", "ap_within_hi", "ap", "ap_lo", "ap_hi",
                "base_rate", "timing_gain", "timing_gain_lo", "timing_gain_hi",
                "timing_gain_local", "timing_gain_local_lo", "timing_gain_local_hi",
                "recall_far01", "recall_far01_lo", "recall_far01_hi", "realised_far01",
                "eventlift_shift_far01", "lead_med_s", "n_classes", "n_onsets", "n_points",
                "retime_failed_share", "g1_approximate"]
        out["conditions"] = {c: {k: _num(r.get(k)) for k in keep if k in r}
                             for c, r in pooled.iterrows()}
    if hypotheses is not None and len(hypotheses):
        out["hypotheses"] = {r["hypothesis"]: {k: _num(r.get(k)) for k in
                                               ("estimate", "lo", "hi", "p", "p_adj", "alpha",
                                                "n_classes", "n_onsets", "base_rate")}
                             | {"reject": None if pd.isna(r.get("reject", pd.NA))
                                else bool(r.get("reject")), "note": r.get("note") or ""}
                             for _, r in hypotheses.iterrows()}
    if ev is not None and len(ev):
        out["ev"] = [{k: _num(r.get(k)) for k in ("budget", "far", "recall", "recall_lo",
                                                   "recall_hi", "delta", "delta_lo",
                                                   "delta_hi")}
                     | {"reference": r.get("reference"),
                        "s3_pass": None if pd.isna(r.get("s3_pass", pd.NA))
                        else bool(r.get("s3_pass"))}
                     for _, r in ev.iterrows()]
    if funnel is not None and len(funnel):
        out["funnel"] = funnel.to_dict(orient="records")
    return out


def _num(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return None
    if isinstance(v, bool | np.bool_):
        return bool(v)
    return float(v) if isinstance(v, int | float | np.number) else v


def write_summary(path: Path | str, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=_num))
    return path
