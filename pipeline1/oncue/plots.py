"""Figures for the write-up (spec §14; notebook §§2-17). One function per figure.

Every function takes **aggregate** inputs (results rows, OOF predictions, curves, control
summaries) and returns a matplotlib Figure; `save(fig, name)` writes SVG + PNG to
`figures/`. Nothing here draws words, names or per-person data: frames that carry text
columns are refused (`assert_aggregate`), and the hero timeline is a labelled schematic
drawn from the synthetic world.

Style: the dataviz reference palette (light surface), validated with the skill's
`validate_palette.js` (all checks pass; three slots sit below 3:1 on the surface, so every
multi-series figure carries a legend and direct labels). Thin 1.5 pt lines, recessive solid
hairline grid, one y-axis per panel, text in ink colours, never in series colours.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SHADE = "#f0efec"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
          "#e34948")          # categorical slots in fixed order (never cycled)
# Entities keep their colour across figures (colour follows the entity, not its rank).
MODEL_COLOR = {"M3": SERIES[0], "M2": SERIES[1], "M4": SERIES[2], "M1": SERIES[3],
               "B1": SERIES[4], "R": SERIES[5], "B-min": SERIES[6], "M5": SERIES[7]}
REF_GREY = "#8a8985"
TEXT_COLUMN_RE = re.compile(r"^(?:txt_|text|word|prompt|answer_text|title|section_masked)")
FIGURE_FUNCTIONS = ("fig_hgb_learning", "fig_lr_c_curve", "fig_learning_curve",
                    "fig_gru_training", "fig_pr_curves", "fig_ap_within", "fig_froc",
                    "fig_reliability", "fig_vault_vs_cv", "fig_forest", "fig_split_tax",
                    "fig_sweeps", "fig_info_horizon", "fig_mismatch", "fig_name_rates",
                    "fig_callon", "fig_p_clock", "fig_course_gap", "fig_timing_decomposition",
                    "fig_knn", "fig_peri_onset", "fig_funnel", "fig_schematic_timeline")


# --------------------------------------------------------------------------- helpers


def assert_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Refuse frames with text-bearing columns (figures are aggregate-only, §10)."""
    bad = [c for c in df.columns if TEXT_COLUMN_RE.match(str(c))]
    if bad:
        raise ValueError(f"figures take aggregate inputs only; drop text columns {bad}")
    return df


def _style():
    return plt.rc_context({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "text.color": INK, "svg.fonttype": "none"})


def _styled(fn):
    """Run a figure function entirely under the house style (legends, lines, text)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _style():
            return fn(*args, **kwargs)
    return wrapper


def _fig(nrows=1, ncols=1, w=6.4, h=3.6, **kw):
    with _style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False, **kw)
        for ax in axes.ravel():
            ax.set_axisbelow(True)
    return fig, axes


def _finish(fig, title: str | None = None, note: str | None = None):
    with _style():
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold",
                         color=INK)
        if note:
            fig.text(0.01, 0.005, note, fontsize=7, color=INK_2, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 0.95 if title else 1))
    return fig


def _legend(ax, **kw):
    """A legend only when there is something to name (>= 1 labelled series)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kw)


def _end_label(ax, x, y, text):
    """Direct label at a line's end, in ink (never the series colour)."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                color=INK_2, va="center")


def save(fig, name: str, out_dir: Path | str | None = None, formats=("svg", "png")
         ) -> list[Path]:
    out = Path(out_dir) if out_dir is not None else config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _pooled(results: pd.DataFrame) -> pd.DataFrame:
    return assert_aggregate(results[results["fold"].isna()]) if "fold" in results else results


# --------------------------------------------------------------------------- learning figures


@_styled
def fig_hgb_learning(curves_by_fold: dict, n_star_by_fold: dict | None = None):
    """(must) HGB staged log-loss, training and inner validation, per outer fold."""
    fig, ax = _fig(1, 1)
    ax = ax[0, 0]
    for fold, cur in sorted(curves_by_fold.items()):
        tr = np.mean(cur["train"], axis=0) if cur.get("train") else None
        va = np.asarray(cur.get("val_mean") or np.mean(cur["val"], axis=0))
        it = np.arange(1, len(va) + 1)
        if tr is not None:
            ax.plot(it, tr, color=SERIES[0], alpha=0.55, lw=1.2,
                    label="training" if fold == min(curves_by_fold) else None)
        ax.plot(it, va, color=SERIES[1], alpha=0.85,
                label="inner validation" if fold == min(curves_by_fold) else None)
        if n_star_by_fold and fold in n_star_by_fold:
            n = int(n_star_by_fold[fold])
            ax.plot([n], [va[n - 1]], "o", ms=5, color=SERIES[1], mec=SURFACE, mew=1.5)
    ax.set_xlabel("boosting iterations")
    ax.set_ylabel("log-loss")
    _legend(ax, loc="upper right")
    return _finish(fig, "M3 learning curves by outer fold",
                   "Dots: n* chosen on inner GroupKFold(3) log-loss.")


@_styled
def fig_lr_c_curve(curves_by_fold: dict, model: str = "M2", param: str = "C"):
    """(must) LR training vs validation AP across C (mean over outer folds, ± range)."""
    rows = []
    for fold, cur in curves_by_fold.items():
        for r in cur:
            rows.append({"fold": fold, param: r[param], "train": r["train_ap"],
                         "val": r["val_ap"]})
    df = pd.DataFrame(rows)
    fig, ax = _fig()
    ax = ax[0, 0]
    for col, color, name in (("train", SERIES[0], "training"), ("val", SERIES[1],
                                                                 "inner validation")):
        g = df.groupby(param)[col]
        x = np.asarray(g.mean().index, float)
        ax.fill_between(x, g.min().to_numpy(), g.max().to_numpy(), color=color, alpha=0.1,
                        lw=0)
        ax.plot(x, g.mean().to_numpy(), "-o", color=color, ms=4, mec=SURFACE, label=name)
    ax.set_xscale("log")
    ax.set_xlabel(param)
    ax.set_ylabel("average precision")
    _legend(ax)
    return _finish(fig, f"{model}: AP across {param}", "Band: range over outer folds.")


@_styled
def fig_learning_curve(results: pd.DataFrame, full_ids: dict | None = None):
    """(should) AP_w vs number of training classes (LC conditions) for M2, M3, M4."""
    r = _pooled(results)
    lc = r[r["condition"].str.startswith("LC-")].copy()
    lc["n_train"] = lc["condition"].str.extract(r"-(\d+)$")[0].astype(float)
    fig, ax = _fig()
    ax = ax[0, 0]
    for m in ("M2", "M3", "M4"):
        d = lc[lc["model"] == m].sort_values("n_train")
        if full_ids and m in full_ids:
            full = r[r["condition"] == full_ids[m]]
            if len(full):
                d = pd.concat([d, full.assign(n_train=np.nan)])
        if not len(d):
            continue
        x = d["n_train"].fillna(d["n_train"].max() * 1.6 if d["n_train"].notna().any()
                                else 1).to_numpy()
        ax.plot(x, d["ap_within"], "-o", color=MODEL_COLOR[m], ms=4, mec=SURFACE, label=m)
        _end_label(ax, x[-1], d["ap_within"].iloc[-1], m)
    ax.set_xlabel("training classes")
    ax.set_ylabel("AP$_w$ (within class)")
    _legend(ax)
    return _finish(fig, "Learning curve: more classes vs a ceiling")


@_styled
def fig_gru_training(history: list[dict]):
    """(should) M5 training loss and inner-validation AP per epoch (two panels, one axis
    each)."""
    fig, axes = _fig(1, 2, w=7.6)
    for h in history:
        df = pd.DataFrame(h["history"])
        lab = f"inner fold {h.get('inner_fold', 0)}"
        axes[0, 0].plot(df["epoch"], df["train_loss"], color=SERIES[0], alpha=0.7)
        if "val_ap" in df:
            axes[0, 1].plot(df["epoch"], df["val_ap"], color=SERIES[1], alpha=0.7, label=lab)
    axes[0, 0].set(xlabel="epoch", ylabel="masked BCE (training)")
    axes[0, 1].set(xlabel="epoch", ylabel="inner-validation AP")
    return _finish(fig, "M5 (GRU hazard) training")


# --------------------------------------------------------------------------- performance


@_styled
def fig_pr_curves(oofs: dict[str, pd.DataFrame]):
    """Pooled OOF precision-recall curves (p_cal) with the base rate π."""
    fig, ax = _fig(w=5.6, h=4.2)
    ax = ax[0, 0]
    pi = None
    for name, oof in oofs.items():
        t = oof[~oof["inner"].astype(bool) & oof["y"].notna()]
        if not len(t) or t["y"].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(t["y"], t["p_cal"])
        model = name.split("-")[-1]
        ax.step(rec, prec, where="post", color=MODEL_COLOR.get(model, REF_GREY), label=name)
        pi = float(t["y"].mean())
    if pi is not None:
        ax.axhline(pi, color=REF_GREY, lw=1, ls="-")
        ax.annotate(f"π = {pi:.3f}", (1, pi), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=7.5, color=INK_2)
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
    _legend(ax, loc="upper right")
    return _finish(fig, "Pooled out-of-fold precision-recall")


@_styled
def fig_ap_within(oof: pd.DataFrame, name: str = "M3 / F-all"):
    """Distribution of per-class AP (classes with >= 1 positive), with AP_w marked."""
    t = oof[~oof["inner"].astype(bool) & oof["y"].notna()]
    aps, w = [], []
    for _, g in t.groupby("class_id"):
        if 0 < g["y"].sum() < len(g):
            from sklearn.metrics import average_precision_score
            aps.append(average_precision_score(g["y"], g["p_raw"]))
            w.append(g["y"].sum())
    fig, ax = _fig(w=5.6)
    ax = ax[0, 0]
    if aps:
        ax.hist(aps, bins=np.linspace(0, 1, 21), color=SERIES[0], rwidth=0.9)
        apw = float(np.average(aps, weights=w))
        ax.axvline(apw, color=INK_2, lw=1)
        ax.annotate(f"AP$_w$ = {apw:.3f}", (apw, ax.get_ylim()[1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set(xlabel="AP within class", ylabel="classes")
    return _finish(fig, f"Per-class AP: {name}", f"n = {len(aps)} classes with a bid.")


@_styled
def fig_froc(curves: dict[str, pd.DataFrame], ao_point: tuple | None = None,
             annotate: dict | None = None, far_max: float = config.FROC_FAR_MAX):
    """FROC: recall vs false firings per at-risk minute, lower envelope over τ. `curves` maps a
    name to DataFrame[far, recall]; references (B-min, shifted, R) are drawn in grey; AO is a
    single point; `annotate` = {"far": 0.1, "recall": .., "lead_s": .., "tiw": ..}."""
    fig, ax = _fig(w=6.0, h=4.2)
    ax = ax[0, 0]
    refs = ("B-min", "shift", "local", "R", "random")
    i = 0
    for name, cur in curves.items():
        c = cur.sort_values("far")
        env = c.groupby("far")["recall"].min().reset_index()
        is_ref = any(name.startswith(r) for r in refs)
        color = REF_GREY if is_ref else MODEL_COLOR.get(name.split("/")[0].strip(), SERIES[i])
        ax.step(env["far"], env["recall"], where="post", color=color,
                lw=1.0 if is_ref else 1.6, ls="-", alpha=0.8 if is_ref else 1, label=name)
        i += 0 if is_ref else 1
    if ao_point is not None and ao_point[0] <= far_max:
        ax.plot([ao_point[0]], [ao_point[1]], "s", ms=6, color=INK_2, mec=SURFACE, mew=1.5,
                label="always-on")
    elif ao_point is not None:                   # AO fires ~4.3/min: off this axis
        ax.annotate(f"always-on → {ao_point[0]:.1f}/min, recall {ao_point[1]:.2f}",
                    (1, ao_point[1]), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7.5,
                    color=INK_2)
    for lam in config.FAR_BUDGETS:
        ax.axvline(lam, color=GRID, lw=1)
    if annotate:
        ax.annotate(f"recall {annotate['recall']:.2f}\nlead {annotate.get('lead_s', np.nan):.0f}"
                    f" s · TiW {annotate.get('tiw', np.nan):.2f}",
                    (annotate["far"], annotate["recall"]), xytext=(8, 8),
                    textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set(xlabel="false firings per at-risk minute", ylabel="event recall",
           xlim=(0, far_max * 1.05), ylim=(0, 1))
    _legend(ax, loc="center right")
    return _finish(fig, "Event-level FROC",
                   "Firing rule with refractory H − g; grey: references.")


@_styled
def fig_reliability(oof: pd.DataFrame, bins: int = config.ECE_BINS):
    """(should) Reliability diagram on p_cal, equal-mass bins."""
    t = oof[~oof["inner"].astype(bool) & oof["y"].notna()].sort_values("p_cal")
    fig, ax = _fig(w=4.6, h=4.2)
    ax = ax[0, 0]
    if len(t):
        chunks = np.array_split(np.arange(len(t)), min(bins, len(t)))
        mp = [t["p_cal"].iloc[c].mean() for c in chunks]
        my = [t["y"].iloc[c].mean() for c in chunks]
        lim = max(max(mp), max(my)) * 1.1 or 1
        ax.plot([0, lim], [0, lim], color=GRID, lw=1)
        ax.plot(mp, my, "-o", color=SERIES[0], ms=4, mec=SURFACE)
        ax.set(xlim=(0, lim), ylim=(0, lim))
    ax.set(xlabel="mean predicted (Platt)", ylabel="observed rate")
    return _finish(fig, "Calibration", "Equal-mass bins.")


@_styled
def fig_vault_vs_cv(results: pd.DataFrame, pairs: dict[str, str]):
    """Vault vs CV AP_w per model (`pairs` = {vault condition: CV condition})."""
    r = _pooled(results).set_index("condition")
    rows = [(v, r.at[cv, "ap_within"] if cv in r.index else np.nan,
             r.at[v, "ap_within"] if v in r.index else np.nan) for v, cv in pairs.items()]
    fig, ax = _fig(w=6.0, h=0.5 + 0.45 * len(rows))
    ax = ax[0, 0]
    y = np.arange(len(rows))
    ax.plot([r_[1] for r_ in rows], y, "o", color=SERIES[0], ms=6, mec=SURFACE, label="CV")
    ax.plot([r_[2] for r_ in rows], y, "D", color=SERIES[1], ms=6, mec=SURFACE, label="vault")
    ax.set_yticks(y, [r_[0].removeprefix("V-") for r_ in rows])
    ax.set_xlabel("AP$_w$")
    _legend(ax)
    return _finish(fig, "Vault vs cross-validation")


# --------------------------------------------------------------------------- comparisons


def comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """The §14 comparison table (pooled rows), ready for the README."""
    cols = ["condition", "tier", "model", "featureset", "split", "target", "ap_within",
            "ap_within_lo", "ap_within_hi", "ap", "ap_fold_mean", "ap_fold_sd", "ap_lift",
            "timing_gain", "timing_gain_local", "recall_far01", "realised_far01", "tiw_far01",
            "eventlift_shift_far01", "lead_med_s", "brier", "brier_skill", "n_classes",
            "n_onsets"]
    r = _pooled(results)
    return r[[c for c in cols if c in r]].reset_index(drop=True)


@_styled
def fig_forest(deltas: pd.DataFrame):
    """Forest plot of H1-H5: DataFrame[hypothesis, estimate, lo, hi, p_adj]."""
    d = deltas.reset_index(drop=True)
    fig, ax = _fig(w=6.0, h=0.6 + 0.5 * len(d))
    ax = ax[0, 0]
    y = np.arange(len(d))[::-1]
    ax.axvline(0, color=REF_GREY, lw=1)
    ax.hlines(y, d["lo"], d["hi"], color=SERIES[0], lw=2)
    ax.plot(d["estimate"], y, "o", color=SERIES[0], ms=6, mec=SURFACE, mew=1.5)
    for yi, r in zip(y, d.itertuples()):
        p = getattr(r, "p_adj", np.nan)
        ax.annotate(f"p = {p:.3f}" if np.isfinite(p) else "", (r.hi, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, d["hypothesis"])
    ax.set_xlabel("ΔAP$_w$ (paired cluster bootstrap, 95% CI)")
    return _finish(fig, "Pre-registered contrasts",
                   "H1 at α = 0.05; H2–H5 Holm-adjusted, tested only if H1 rejects.")


@_styled
def fig_split_tax(results: pd.DataFrame, pairs=(("K0-M2", "S-rand-M2"),
                                                ("K0-M3", "S-rand-M3"))):
    """Leakage tax: pooled AP under S-group vs S-rand."""
    r = _pooled(results).set_index("condition")
    fig, ax = _fig(w=5.0)
    ax = ax[0, 0]
    x = np.arange(len(pairs))
    for j, (col, name) in enumerate(((0, "S-group"), (1, "S-rand"))):
        vals = [r.at[p[col], "ap"] if p[col] in r.index else np.nan for p in pairs]
        ax.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=SERIES[j], label=name)
    ax.set_xticks(x, [p[0].split("-")[-1] for p in pairs])
    ax.set_ylabel("pooled AP")
    _legend(ax)
    return _finish(fig, "Leakage tax (random vs grouped splits)")


@_styled
def fig_sweeps(results: pd.DataFrame, model: str = "M3", featureset: str = "F-all"):
    """g sweep (H = 15) and H sweep (g = 2) of AP_w, two panels with their own axes."""
    r = _pooled(results)
    r = r[(r["model"] == model) & (r["featureset"] == featureset) & (r["split"] == "S-group")
          & r["target"].str.match(r"^y_H\d+_g\d+$")
          & r["condition"].str.match(r"^(?:K0-M3|G-g\d+|Hs-H\d+)$")]
    fig, axes = _fig(1, 2, w=7.6)
    g = r[r["H"] == 15].sort_values("g")
    h = r[r["g"] == 2].sort_values("H")
    axes[0, 0].plot(g["g"], g["ap_within"], "-o", color=SERIES[0], ms=5, mec=SURFACE)
    axes[0, 0].set(xlabel="guard g (s), H = 15", ylabel="AP$_w$")
    axes[0, 1].plot(h["H"], h["ap_within"], "-o", color=SERIES[0], ms=5, mec=SURFACE)
    axes[0, 1].plot(h["H"], h["base_rate"], "-", color=REF_GREY, lw=1)
    if len(h):
        _end_label(axes[0, 1], h["H"].iloc[-1], h["base_rate"].iloc[-1], "π")
    axes[0, 1].set(xlabel="horizon H (s), g = 2", ylabel="AP$_w$")
    return _finish(fig, f"{model} / {featureset}: guard and horizon sweeps")


@_styled
def fig_info_horizon(results: pd.DataFrame):
    """IH curve: AP_w vs feature cut-off shift δ; δ > 0 (future information) is shaded."""
    r = _pooled(results)
    ih = r[r["condition"].str.match(r"^IH[+-]?\d*$")].copy()
    ih["delta"] = ih["condition"].str.replace("IH", "", regex=False).replace("", "0")
    ih["delta"] = pd.to_numeric(ih["delta"], errors="coerce")
    ih = ih.sort_values("delta")
    fig, ax = _fig()
    ax = ax[0, 0]
    if len(ih):
        ax.axvspan(0, max(ih["delta"].max(), 1), color=SHADE, lw=0)
        ax.annotate("future information (illegal)", (0, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points", fontsize=7.5, color=INK_2)
        ax.plot(ih["delta"], ih["ap_within"], "-o", color=SERIES[1], ms=5, mec=SURFACE)
    ax.set(xlabel="feature cut-off t + δ (s)", ylabel="AP$_w$ (M2 / F-text)")
    return _finish(fig, "Information horizon")


@_styled
def fig_mismatch(results: pd.DataFrame, matched: str = "F1-Fall+pcwA",
                 without: str = "F1-Fall-pcw"):
    """A-mismatch: matched PCW vs deranged PCW (mean ± sd over 5 derangements) vs no PCW."""
    r = _pooled(results).set_index("condition")
    mm = r[r.index.str.startswith("A-mismatch-")]["ap_within"]
    vals = [r.at[matched, "ap_within"] if matched in r.index else np.nan,
            mm.mean() if len(mm) else np.nan,
            r.at[without, "ap_within"] if without in r.index else np.nan]
    err = [0, mm.std(ddof=1) if len(mm) > 1 else 0, 0]
    fig, ax = _fig(w=5.0)
    ax = ax[0, 0]
    ax.bar(range(3), vals, yerr=err, width=0.5, color=SERIES[0], ecolor=INK_2, capsize=3)
    ax.set_xticks(range(3), ["own PCW", "deranged ×5", "no PCW"])
    ax.set_ylabel("AP$_w$ (PCW subset)")
    return _finish(fig, "Does matching preparation matter?")


@_styled
def fig_name_rates(rates: pd.DataFrame):
    """N-name: [NAME] rate per professor minute before, after and away from bids."""
    d = assert_aggregate(rates).set_index("window").reindex(["pre_bid", "post_bid", "matched"])
    fig, ax = _fig(w=5.0)
    ax = ax[0, 0]
    x = np.arange(len(d))
    ax.bar(x, d["rate_per_min"], width=0.5, color=SERIES[0])
    ax.errorbar(x, d["rate_per_min"], yerr=[d["rate_per_min"] - d["ci_lo"],
                                            d["ci_hi"] - d["rate_per_min"]],
                fmt="none", ecolor=INK_2, capsize=3)
    ax.set_xticks(x, ["[o − 15, o − 2]", "[o, o + 10]", "matched"])
    ax.set_ylabel("[NAME] per professor minute")
    return _finish(fig, "Name mentions around bids (descriptive)")


@_styled
def fig_callon(res: dict):
    """CALLON check: phrase rate before bids vs matched points, with the rate-ratio CI."""
    fig, ax = _fig(w=4.6)
    ax = ax[0, 0]
    ax.bar([0, 1], [res["rate_bid"], res["rate_matched"]], width=0.5, color=SERIES[0])
    ax.set_xticks([0, 1], ["[o − 5, o)", "matched"])
    ax.set_ylabel("CALLON phrases per window")
    note = (f"rate ratio {res['rate_ratio']:.2f} [{res['ci_lo']:.2f}, {res['ci_hi']:.2f}]; "
            f"{'pass' if res['pass'] else 'FAIL'}")
    return _finish(fig, "Are labels late? (CALLON check)", note)


@_styled
def fig_p_clock(res: dict):
    """P-clock: AP(δ) of speech in (t + δ − 1, t + δ] for the border; expected peak band."""
    cur = res["curve"] if isinstance(res["curve"], pd.DataFrame) else pd.DataFrame(res["curve"])
    fig, ax = _fig()
    ax = ax[0, 0]
    lo, hi = config.P_CLOCK_PEAK_RANGE_S
    ax.axvspan(lo + 0.5, hi + 0.5, color=SHADE, lw=0)        # trailing 1-s window: +0.5 s
    ax.plot(cur["delta_s"], cur["ap"], "-o", color=SERIES[0], ms=4, mec=SURFACE)
    ax.axvline(res["delta_star"], color=INK_2, lw=1)
    ax.set(xlabel="δ (s)", ylabel="AP of border vs speech")
    return _finish(fig, "Clock check (P-clock)",
                   f"δ* = {res['delta_star']:+.2f} s (lag {res.get('lag_s', np.nan):+.2f} s), "
                   f"sharpness {res['sharpness']:.2f}; {'pass' if res['pass'] else 'FAIL'}. "
                   "Shaded: expected UI-lag band.")


@_styled
def fig_course_gap(per_course: pd.DataFrame):
    """Per-course AP_w gap F-text − F-struct: DataFrame[course, gap, lo, hi, n_classes]."""
    d = assert_aggregate(per_course).sort_values("gap")
    fig, ax = _fig(w=5.6, h=0.6 + 0.4 * len(d))
    ax = ax[0, 0]
    y = np.arange(len(d))
    ax.axvline(0, color=REF_GREY, lw=1)
    if {"lo", "hi"} <= set(d):
        ax.hlines(y, d["lo"], d["hi"], color=SERIES[0], lw=2)
    ax.plot(d["gap"], y, "o", color=SERIES[0], ms=6, mec=SURFACE)
    ax.set_yticks(y, d["course"])
    ax.set_xlabel("ΔAP$_w$ (F-text − F-struct)")
    return _finish(fig, "Text vs structure by course")


@_styled
def fig_timing_decomposition(results: pd.DataFrame, conditions=("K0-M3", "F0-Fstruct")):
    """AP vs globally and locally shifted AP (the class vs the phase vs the moment)."""
    r = _pooled(results).set_index("condition")
    conds = [c for c in conditions if c in r.index]
    fig, ax = _fig(w=5.6)
    ax = ax[0, 0]
    x = np.arange(len(conds))
    for j, (col, name) in enumerate((("ap", "AP"), ("ap_shift", "global shift"),
                                     ("ap_local", "local shift"))):
        ax.bar(x + (j - 1) * 0.26, [r.at[c, col] for c in conds], width=0.24,
               color=SERIES[j], label=name)
    ax.set_xticks(x, [r.at[c, "featureset"] for c in conds])
    ax.set_ylabel("pooled AP")
    _legend(ax)
    return _finish(fig, "Timing decomposition", "TG = AP − global; TG_local = AP − local.")


@_styled
def fig_knn(summary: pd.DataFrame):
    """A5: share of nearest training neighbours that preceded a bid, TPs vs false alarms."""
    d = assert_aggregate(summary)
    fig, ax = _fig(w=4.6)
    ax = ax[0, 0]
    vals = [d["share_tp"].mean(), d["share_fa"].mean()]
    ax.bar([0, 1], vals, width=0.5, color=SERIES[0])
    ax.set_xticks([0, 1], ["true positives", "false alarms"])
    ax.set_ylabel("neighbours followed by a bid")
    return _finish(fig, "What do alarms resemble? (kNN, metadata only)")


@_styled
def fig_peri_onset(curves: pd.DataFrame):
    """Peri-onset means of causal features (-120 to +30 s) vs shifted controls:
    DataFrame[feature, rel_s, mean, shifted_mean]. One small multiple per feature."""
    d = assert_aggregate(curves)
    feats = list(dict.fromkeys(d["feature"]))
    fig, axes = _fig(1, len(feats), w=3.2 * max(len(feats), 1), h=3.0)
    for ax, f in zip(axes.ravel(), feats):
        g = d[d["feature"] == f].sort_values("rel_s")
        ax.plot(g["rel_s"], g["shifted_mean"], color=REF_GREY, lw=1, label="shifted")
        ax.plot(g["rel_s"], g["mean"], color=SERIES[0], label="around bids")
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set(title=f, xlabel="seconds from onset")
    _legend(axes[0, 0])
    return _finish(fig, "Peri-onset feature curves")


@_styled
def fig_funnel(funnel: pd.DataFrame):
    """Data funnel: DataFrame[stage, classes] (bids optional), in order."""
    d = assert_aggregate(funnel)
    fig, ax = _fig(w=6.0, h=0.6 + 0.35 * len(d))
    ax = ax[0, 0]
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["classes"], height=0.6, color=SERIES[0])
    for yi, v in zip(y, d["classes"]):
        ax.annotate(f"{int(v)}", (v, yi), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, d["stage"])
    ax.set_xlabel("classes")
    ax.grid(axis="y", visible=False)
    return _finish(fig, "Data funnel")


@_styled
def fig_schematic_timeline(sc, p: np.ndarray | None = None, t: np.ndarray | None = None,
                           tau: float | None = None, span=(600.0, 1200.0)):
    """Hero image: a SCHEMATIC one-class timeline from the synthetic world (never a real
    class): role lanes, professor question ticks, p(t), firings, bids, segment shading. No
    text. `p` defaults to the rule R computed from the synthetic question times."""
    a, b = span
    fig, axes = _fig(2, 1, w=8.0, h=3.6, sharex=True,
                     gridspec_kw={"height_ratios": [1.2, 1]})
    lanes, prob = axes[0, 0], axes[1, 0]
    seg = sc.segments
    for i, s in enumerate(seg.itertuples()):
        if s.end_s > a and s.start_s < b and i % 2:
            for ax in (lanes, prob):
                ax.axvspan(max(s.start_s, a), min(s.end_s, b), color=SHADE, lw=0)
    roles = [("prof", SERIES[0]), ("student", SERIES[2]), ("me", SERIES[1])]
    w = sc.words
    for k, (role, color) in enumerate(roles):
        ww = w[(w["role"] == role) & (w["end_s"] > a) & (w["start_s"] < b)]
        lanes.hlines(np.full(len(ww), k), ww["start_s"], ww["end_s"], color=color, lw=6)
    q = sc.turns[(sc.turns["kind"] == "question") & (sc.turns["end_s"].between(a, b))] \
        if "kind" in sc.turns else sc.turns.iloc[:0]
    lanes.plot(q["end_s"], np.full(len(q), -0.8), "v", color=INK_2, ms=5)
    bids = sc.episodes[(sc.episodes["ep_type"] == config.EP_BID) &
                       sc.episodes["onset_s"].between(a, b)]
    for o in bids["onset_s"]:
        for ax in (lanes, prob):
            ax.axvline(o, color=SERIES[7], lw=1.2)
    lanes.set_yticks([-0.8, 0, 1, 2], ["question", "professor", "classmates", "Brady"])
    lanes.set_ylim(-1.4, 2.6)
    lanes.grid(False)
    if p is None:
        t = np.arange(a, b, config.DELTA_S)
        qe = np.sort(q["end_s"].to_numpy(float)) if len(q) else np.array([])
        age = np.array([x - qe[qe <= x][-1] if (qe <= x).any() else np.inf for x in t])
        p = 0.5 * np.exp(-np.minimum(age, config.Q_AGE_CAP_S) / config.R_DECAY_S) + 0.02
    prob.plot(t, p, color=SERIES[0])
    tau = tau if tau is not None else 0.5 * float(np.nanmax(p))
    prob.axhline(tau, color=REF_GREY, lw=1)
    fire, last = [], -np.inf
    for ti, pi in zip(t, p):
        if pi >= tau and ti - last >= config.alarm_refractory():
            fire.append(ti)
            last = ti
    prob.plot(fire, np.full(len(fire), tau), "o", color=SERIES[0], ms=5, mec=SURFACE, mew=1.5)
    prob.set(xlabel="class time (s)", ylabel="p(t)")
    return _finish(fig, "On Cue, schematic",
                   "Schematic from synthetic data (no real class). Red lines: hand raises; "
                   "dots: alarms; shading: activity segments.")


# --------------------------------------------------------------------------- CLI


def _load_curves(oof_dir: Path, cond: str) -> dict:
    meta = oof_dir / f"{cond}.json"
    return json.loads(meta.read_text()) if meta.exists() else {}


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _table(name: str) -> pd.DataFrame | None:
    path = config.table_path(name)
    return pd.read_parquet(path) if path.exists() else None


def _local_figures(out: Path, oofs: dict, res: pd.DataFrame | None, n_boot: int) -> list:
    """Figures from local aggregate inputs: controls JSON, the funnel, tuning curves,
    FROC, the H1-H5 forest, peri-onset curves, plus figures/summary.json. Each one is
    skipped (with a message) when its inputs are missing."""
    from . import experiments, qa, report

    made, hyp, funnel = [], None, None
    controls = config.DATA_DIR / "controls"
    episodes, classes, points = _table("hand_episodes"), _table("classes"), _table("points")

    def attempt(name, draw):
        try:
            fig = draw()
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            print(f"skip {name}: {exc}")
            return
        if fig is None:
            print(f"skip {name}: no inputs")
            return
        made.extend(save(fig, name, out))

    pclock, callon, nname = (_read_json(controls / f"{n}.json")
                             for n in ("p-clock", "callon", "n-name"))
    attempt("p_clock", lambda: fig_p_clock(pclock) if pclock else None)
    attempt("callon", lambda: fig_callon(callon) if callon else None)
    attempt("name_rates", lambda: fig_name_rates(pd.DataFrame(nname["rates"]))
            if nname else None)
    if classes is not None:
        funnel = qa.funnel_table(classes, points)
        attempt("funnel", lambda: fig_funnel(funnel))
    curves = experiments.tuning_curves()
    if "M3" in curves:
        attempt("hgb_learning", lambda: fig_hgb_learning(
            {k: v["curves"] for k, v in curves["M3"].items()},
            {k: v["n_star"] for k, v in curves["M3"].items()}))
    for m in ("M2", "M4", "B1"):
        if m in curves:
            attempt(f"lr_c_{m}", lambda m=m: fig_lr_c_curve(
                {k: v["curves"] for k, v in curves[m].items()}, model=m))
    if oofs and episodes is not None:
        oofs_g = {k: v.merge(points[["class_id", "t_s", "in_G"]], on=["class_id", "t_s"],
                             how="left") if "in_G" not in v and points is not None else v
                  for k, v in oofs.items()}
        froc = report.froc_curves(oofs_g, episodes, ("K0-M3", "K0-Bmin", "K0-R"))
        ao = report.always_on_point(oofs_g["K0-AO"], episodes) if "K0-AO" in oofs_g \
            else None
        attempt("froc", lambda: fig_froc(froc, ao_point=ao) if froc else None)
        hyp = report.hypothesis_table(oofs, episodes, segments=_table("segments"),
                                      n_boot=n_boot)
        attempt("forest", lambda: fig_forest(hyp.dropna(subset=["estimate"]))
                if hyp["estimate"].notna().any() else None)
    features = _table("features")
    if features is not None and episodes is not None and points is not None:
        dev = set(points.loc[~points["in_vault"].astype(bool), "class_id"])
        cur = report.peri_onset_curves(features[features["class_id"].isin(dev)],
                                       report.onsets_from_bids(episodes),
                                       [c for c in ("q_open", "spk_silence_s", "pcw_unspent")
                                        if c in features])
        attempt("peri_onset", lambda: fig_peri_onset(cur.dropna(subset=["mean"])))
    summary = report.summary(res, hyp, funnel)
    if summary:
        made.append(report.write_summary(out / "summary.json", summary))
    return made


def cli(args) -> int:
    """`python -m oncue figures [--out DIR] [--n-boot B]`: draw every figure whose aggregate
    inputs exist (results, OOF, controls, tuning curves, local tables) and summary.json."""
    import argparse

    p = argparse.ArgumentParser(prog="oncue figures")
    p.add_argument("--out", default=None)
    p.add_argument("--n-boot", type=int, default=config.N_BOOT,
                   help="bootstrap draws for the H1-H5 forest plot")
    o = p.parse_args(getattr(args, "extra", []) or [])
    out = Path(o.out) if o.out else config.FIGURES_DIR
    res_path = config.table_path("results")
    made = []
    if res_path.exists():
        res = pd.read_parquet(res_path)
        for name, fn in (("comparison_split_tax", fig_split_tax), ("sweeps", fig_sweeps),
                         ("info_horizon", fig_info_horizon), ("mismatch", fig_mismatch),
                         ("timing_decomposition", fig_timing_decomposition),
                         ("learning_curve", fig_learning_curve)):
            try:
                made += save(fn(res), name, out)
            except (KeyError, ValueError, IndexError) as exc:
                print(f"skip {name}: {exc}")
        comparison_table(res).to_csv(out / "comparison_table.csv", index=False)
    oofs = {}
    for cond in ("K0-B1", "K0-R", "K0-M1", "K0-M2", "K0-M3", "K1-M4"):
        path = config.OOF_DIR / f"{cond}.parquet"
        if path.exists():
            oofs[cond] = pd.read_parquet(path)
    if oofs:
        made += save(fig_pr_curves(oofs), "pr_curves", out)
        if "K0-M3" in oofs:
            made += save(fig_ap_within(oofs["K0-M3"]), "ap_within", out)
            made += save(fig_reliability(oofs["K0-M3"]), "reliability", out)
    all_oofs = {q.stem: pd.read_parquet(q) for q in sorted(config.OOF_DIR.glob("*.parquet"))
                if not q.stem.endswith("_train")} if config.OOF_DIR.exists() else {}
    made += _local_figures(out, all_oofs, pd.read_parquet(res_path) if res_path.exists()
                           else None, o.n_boot)
    print(f"wrote {len(made)} files to {out}")
    return 0
