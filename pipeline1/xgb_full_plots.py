"""Charts for xgb_full.py: how the XGBoost stacks compare, and where the models warned on test.

Reads the saved scores (data/results/flex/, data/results/xgb/) and retrains nothing. Writes three
PNGs to data/results/xgb/figures/ (local, gitignored):

    1_scores.png        AP_w and AP with 95% session-bootstrap intervals, validation and test
    2_five_classes.png  five test sessions: score, threshold, raises and warnings, plus one zoom
    3_all_test.png      every raise and warning in all 86 test sessions

The five sessions come from a fixed rule, so they are not hand-picked: among test sessions with at
least 5 raises, rank by the mean of the two models' within-session AP and take the 10th, 30th, 50th,
70th and 90th percentiles. Warnings use the rule of `baselines.warning_runs`; the counts are checked
against `baselines.guess_scores`. The chart shows XGBoost + MLP + TCN's fold models because a
threshold needs out-of-fold train scores, which the single all-train fit does not have.

`python xgb_full_plots.py` from pipeline1/.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oncue import baselines, config, flex  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2, validated in both modes
GOOD, BAD = "#0ca30c", "#d03b3b"             # status: right and wrong; always with a shape and a legend
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "font.family": "sans-serif", "font.size": 9, "legend.frameon": False, "figure.dpi": 150,
})

OUT = config.DATA_DIR / "results" / "xgb" / "figures"
NAME_F, NAME_X = "MLP + TCN, logistic (final)", "XGBoost + MLP + TCN"
ORDER = ["mlp_tcn_lr", "xgb_mlp_tcn_all", "xgb_mlp_tcn", "xgb_tcn_all", "xgb_tcn", "xgb_all", "xgb",
         "gbt_mlp_tcn", "gbt", "tcn", "mlp"]
LABEL = {"mlp_tcn_lr": NAME_F, "xgb_mlp_tcn_all": "XGBoost + MLP + TCN, all train", "xgb_mlp_tcn": NAME_X,
         "xgb_tcn_all": "XGBoost + TCN, all train", "xgb_tcn": "XGBoost + TCN", "xgb_all": "XGBoost, all train",
         "xgb": "XGBoost", "gbt_mlp_tcn": "HGB + MLP + TCN", "gbt": "HGB", "tcn": "TCN", "mlp": "MLP"}


def colour(i: str) -> str:
    return BLUE if i == "mlp_tcn_lr" else ORANGE if i.startswith("xgb") else MUTED


def with_gaps(t, q):
    cut = np.flatnonzero(np.diff(t) > 2.5) + 1
    return np.insert(t.astype(float), cut, np.nan), np.insert(q.astype(float), cut, np.nan)


def warning_list(t, y, said) -> list[tuple[float, float, bool]]:
    """(start, end, right) per warning: an unbroken run of yes moments, as `baselines.warning_runs`."""
    if not said.any():
        return []
    cont = np.r_[False, said[:-1] & (np.diff(t) <= 2.0 + 1e-6)]
    run = np.cumsum(said & ~cont)
    out = []
    for r in np.unique(run[said]):
        m = said & (run == r)
        out.append((float(t[m][0]), float(t[m][-1]), bool(y[m].any())))
    return out


def session(frame, ids, p, tau, onsets) -> dict:
    """One model on one session: its warnings and, per raise, how long before it the first guess came."""
    t, y = frame["t_s"].to_numpy(float)[ids], frame["y"].to_numpy().astype(bool)[ids]
    said = p[ids] >= tau
    lead = {}
    for o in onsets:
        w = t[said & (t >= o - 15) & (t < o - 2)]
        if len(w):
            lead[float(o)] = float(o - w.min())
    return {"t": t, "p": p[ids], "tau": tau, "warnings": warning_list(t, y, said), "lead": lead, "onsets": onsets}


def summary(s: dict) -> str:
    n, r = len(s["warnings"]), sum(w[2] for w in s["warnings"])
    return f"{n} warning{'' if n == 1 else 's'}, {r} right; {len(s['lead'])} of {len(s['onsets'])} raises warned"


def scores_figure(report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.885, bottom=0.06, hspace=0.25, wspace=0.05)
    y = np.arange(len(ORDER))[::-1]
    for r, (metric, title) in enumerate((("AP_w", "Within-session AP (the headline)"), ("AP", "Pooled AP"))):
        for c, (split, head) in enumerate((("validation", "Validation"), ("test", "Test (second look)"))):
            ax, rep = axes[r, c], report[split]
            name = {v: k for k, v in {**LABEL, "xgb_mlp_tcn": NAME_X}.items()}
            tab = rep["table"].rename(index=name)
            ax.axvline(float(tab.loc["mlp_tcn_lr", metric]), color=BLUE, lw=1, ls=(0, (4, 3)), zorder=1)
            lo_all, hi_all = [], []
            for yy, i in zip(y, ORDER):
                if i not in tab.index:
                    continue
                v, lo, hi = float(tab.loc[i, metric]), rep["ci"].loc[i, f"{metric} low"], rep["ci"].loc[i, f"{metric} high"]
                lo_all.append(lo); hi_all.append(hi)
                ax.plot([lo, hi], [yy, yy], color=colour(i), lw=2, solid_capstyle="round", zorder=2)
                ax.plot(v, yy, "o", ms=7, color=colour(i), mec=SURFACE, mew=1.5, zorder=3)
                ax.annotate(f"{v:.4f}", (hi, yy), xytext=(6, 0), textcoords="offset points", va="center",
                            color=INK if i == "mlp_tcn_lr" else INK2, fontsize=8.5)
            ax.set_xlim(min(lo_all) * 0.93, max(hi_all) * 1.1)
            for yy, i in zip(y, ORDER):
                if i not in tab.index:
                    ax.text(min(lo_all) * 0.95, yy, "never scored on test", color=MUTED, va="center", fontsize=8.5)
            ax.set_title(f"{head}: {title}", fontweight="bold")
            ax.set_yticks(y, [LABEL[i] for i in ORDER])
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(f"{metric} (dot) with 95% session-bootstrap interval")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=BLUE, lw=2, label="Final pipeline (dashed line: its score)"),
                        Line2D([0], [0], marker="o", color=ORANGE, lw=2, label="XGBoost"),
                        Line2D([0], [0], marker="o", color=MUTED, lw=2, label="HGB and single specialists")],
               loc="upper left", bbox_to_anchor=(0.17, 0.955), ncols=3, fontsize=9)
    fig.suptitle("XGBoost beats HGB on validation, and only ties MLP + TCN", x=0.012, y=0.985, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "1_scores.png", dpi=150)
    plt.close(fig)


def draw_timeline(ax, s: dict, colr: str) -> None:
    x, q = with_gaps(s["t"], s["p"])
    ax.plot(x / 60, q, color=colr, lw=0.8, solid_joinstyle="round")
    ax.axhline(s["tau"], color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_ylim(0, max(np.nanmax(q), s["tau"]) * 1.4)
    tr = ax.get_xaxis_transform()
    for o in s["onsets"]:
        ax.axvline(o / 60, color=INK, lw=0.9, zorder=1)
    wr = [w[0] / 60 for w in s["warnings"] if not w[2]]
    ok = [w[0] / 60 for w in s["warnings"] if w[2]]
    ax.plot(wr, [0.86] * len(wr), "|", color=BAD, ms=8, mew=1.6, transform=tr, zorder=3)
    ax.plot(ok, [0.86] * len(ok), "^", color=GOOD, ms=6.5, mec=SURFACE, mew=0.8, transform=tr, zorder=4)
    warned = [o / 60 for o in s["lead"]]
    ax.plot(warned, [0.97] * len(warned), "o", color=INK, ms=4.5, mec=SURFACE, mew=0.8, transform=tr, zorder=4, clip_on=False)


def draw_zoom(ax, s: dict, colr: str, lo: float, hi: float) -> None:
    keep = (s["t"] >= lo) & (s["t"] <= hi)
    x, q = with_gaps(s["t"][keep], s["p"][keep])
    ax.plot(x / 60, q, color=colr, lw=1.5, marker="o", ms=2)
    ax.axhline(s["tau"], color=INK2, lw=1, ls=(0, (4, 3)))
    top = max(np.nanmax(q), s["tau"]) * 1.45
    ax.set_ylim(0, top)
    for a, b, right in s["warnings"]:
        if b >= lo and a <= hi:
            ax.axvspan(a / 60, (b + 2) / 60, color=GOOD if right else BAD, alpha=0.25, lw=0)
    for o in s["onsets"]:
        if lo <= o <= hi:
            ax.axvline(o / 60, color=INK, lw=1.2)
            txt = f"warned {s['lead'][float(o)]:.0f} s ahead" if float(o) in s["lead"] else "not warned"
            late = o > lo + 0.8 * (hi - lo)
            ax.text(o / 60, top * 0.97, f"{txt} " if late else f" {txt}", color=INK, va="top", fontsize=8.5,
                    ha="right" if late else "left")
    ax.set_xlim(lo / 60, hi / 60)


def five_classes_figure(frame, p_f, tau_f, p_x, tau_x, raises, course) -> None:
    y = frame["y"].to_numpy()
    idx = frame.groupby("class_id", sort=False).indices
    n_r = raises[raises["class_id"].isin(idx)].groupby("class_id").size()
    rank = {c: 0.5 * (average_precision_score(y[ix], p_f[ix]) + average_precision_score(y[ix], p_x[ix]))
            for c, ix in idx.items() if n_r.get(c, 0) >= 5 and y[ix].any()}
    order = sorted(rank, key=rank.get)
    picks = [(q, int(order[round(q / 100 * (len(order) - 1))])) for q in (10, 30, 50, 70, 90)]
    print(f"{len(order)} eligible sessions; chosen: {picks}")

    fig = plt.figure(figsize=(14, 21.5))
    outer = GridSpec(6, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1.15], left=0.05, right=0.985,
                     top=0.925, bottom=0.03, hspace=0.34)
    median = None
    for row, (q, c) in enumerate(picks):
        ids = idx[c]
        on = np.sort(raises.loc[raises["class_id"] == c, "onset_s"].to_numpy(float))
        sf, sx = session(frame, ids, p_f, tau_f, on), session(frame, ids, p_x, tau_x, on)
        if q == 50:
            median = (c, sf, sx)
        inner = outer[row].subgridspec(2, 1, hspace=0.2)
        a = fig.add_subplot(inner[0])
        b = fig.add_subplot(inner[1], sharex=a)
        draw_timeline(a, sf, BLUE)
        draw_timeline(b, sx, ORANGE)
        a.set_title(f"{q}th percentile: session {c} ({course.get(c, '?')}), {len(on)} raises", fontweight="bold", loc="left", pad=4)
        a.text(0.995, 1.03, f"MLP + TCN: {summary(sf)}", transform=a.transAxes, ha="right", color=INK2, fontsize=8.5)
        b.text(0.995, 1.03, f"XGBoost + MLP + TCN: {summary(sx)}", transform=b.transAxes, ha="right", color=INK2, fontsize=8.5)
        a.tick_params(labelbottom=False)
        a.set_ylabel("score"); b.set_ylabel("score")
        b.set_xlabel("minute of the session")

    c, sf, sx = median
    both = [o for o in sf["lead"] if o in sx["lead"]]
    pool = both or list(sf["lead"]) or list(sx["lead"]) or list(sf["onsets"])
    lead = lambda o: sf["lead"].get(o, sx["lead"].get(o, 99))
    focus = min(pool, key=lambda o: abs(lead(o) - 3.5))
    lo, hi = focus - 150, focus + 150
    inner = outer[5].subgridspec(2, 1, hspace=0.12)
    a = fig.add_subplot(inner[0])
    b = fig.add_subplot(inner[1], sharex=a)
    draw_zoom(a, sf, BLUE, lo, hi)
    draw_zoom(b, sx, ORANGE, lo, hi)
    a.tick_params(labelbottom=False)
    a.set_title(f"Zoom: 5 minutes of session {c}, one point per counting moment (2 s)", fontweight="bold", pad=4)
    a.set_ylabel("score"); b.set_ylabel("score")
    b.set_xlabel("minute of the session")

    handles = [Line2D([0], [0], color=BLUE, lw=2, label="MLP + TCN score"),
               Line2D([0], [0], color=ORANGE, lw=2, label="XGBoost + MLP + TCN score"),
               Line2D([0], [0], color=INK2, lw=1, ls=(0, (4, 3)), label="Threshold, fixed on train"),
               Line2D([0], [0], color=INK, lw=1.2, label="My raise"),
               Line2D([0], [0], marker="o", color=INK, ms=4.5, lw=0, label="Raise warned 2-15 s ahead"),
               Line2D([0], [0], marker="^", color=GOOD, ms=6.5, lw=0, label="Warning, right (shaded green in zoom)"),
               Line2D([0], [0], marker="|", color=BAD, ms=8, mew=1.6, lw=0, label="Warning, wrong (shaded red in zoom)")]
    fig.suptitle("Five test sessions, weakest to strongest", x=0.05, y=0.992, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.05, 0.977, "Each model has its own scale, so compare a score with that model's threshold. "
             "Gaps in a line are moments that do not count (my hand up, me speaking, breakouts).",
             color=INK2, fontsize=9.5)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.974), ncols=4, fontsize=9)
    fig.savefig(OUT / "2_five_classes.png", dpi=130)
    plt.close(fig)


def all_test_figure(frame, models: list, raises, checks: dict) -> None:
    idx = frame.groupby("class_id", sort=False).indices
    n_r = raises[raises["class_id"].isin(idx)].groupby("class_id").size()
    order = sorted(idx, key=lambda c: (-n_r.get(c, 0), c))
    fig, axes = plt.subplots(1, 2, figsize=(14, 13.8), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.925, bottom=0.04, wspace=0.03)
    for ax, (name, p, tau) in zip(axes, models):
        tot = {"warnings": 0, "right": 0, "warned": 0, "raises": 0}
        for row, c in enumerate(order):
            on = np.sort(raises.loc[raises["class_id"] == c, "onset_s"].to_numpy(float))
            s = session(frame, idx[c], p, tau, on)
            yy = len(order) - row
            miss = [o / 60 for o in on if float(o) not in s["lead"]]
            hit = [o / 60 for o in s["lead"]]
            wr = [w[0] / 60 for w in s["warnings"] if not w[2]]
            ok = [w[0] / 60 for w in s["warnings"] if w[2]]
            ax.plot(miss, [yy] * len(miss), "o", ms=3.5, mfc="none", mec=MUTED, mew=0.8, zorder=2)
            ax.plot(hit, [yy] * len(hit), "o", ms=4.5, color=INK, mec=SURFACE, mew=0.5, zorder=4)
            ax.plot(wr, [yy + 0.32] * len(wr), "|", ms=4.5, color=BAD, mew=1.1, zorder=3)
            ax.plot(ok, [yy + 0.32] * len(ok), "^", ms=4.5, color=GOOD, mec=SURFACE, mew=0.4, zorder=5)
            tot["warnings"] += len(s["warnings"]); tot["right"] += len(ok)
            tot["warned"] += len(hit); tot["raises"] += len(on)
        chk = checks[name]
        assert (tot["warnings"], tot["right"], tot["warned"], tot["raises"]) == (
            chk["warnings"], chk["warnings_right"], chk["raises_warned"], chk["raises"]), (name, tot, chk)
        ax.set_title(f"{name}: {tot['warnings']} warnings, {tot['right']} right ({100 * tot['right'] / tot['warnings']:.1f}%); "
                     f"{tot['warned']} of {tot['raises']} raises warned", fontweight="bold")
        ax.set_xlabel("minute of the session")
        ax.grid(axis="y", visible=False)
        ax.set_yticks([])
        ax.set_ylim(0.2, len(order) + 1)
    axes[0].set_ylabel("86 test sessions, most raises at the top")
    handles = [Line2D([0], [0], marker="o", color=INK, ms=4.5, lw=0, label="Raise, warned 2-15 s ahead"),
               Line2D([0], [0], marker="o", mfc="none", mec=MUTED, ms=3.5, lw=0, label="Raise, not warned"),
               Line2D([0], [0], marker="^", color=GOOD, ms=4.5, lw=0, label="Warning, right"),
               Line2D([0], [0], marker="|", color=BAD, ms=6, mew=1.2, lw=0, label="Warning, wrong")]
    fig.suptitle("Every raise and warning on test: MLP + TCN warns of about twice as many raises, about as accurately",
                 x=0.03, y=0.992, ha="left", fontsize=13, fontweight="bold")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.972), ncols=4, fontsize=9)
    fig.savefig(OUT / "3_all_test.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT.parent / "scores.pkl", "rb") as fh:
        res = pickle.load(fh)
    with open(OUT.parent / "report.pkl", "rb") as fh:
        report = pickle.load(fh)
    comb, T = flex.load_result(flex.FINAL), flex.load_result("test_final")
    split_csv = config.DATA_DIR / "03_split" / "split.csv"
    data = baselines.load(config.TABLES_DIR, split_csv)
    raises = baselines.onsets(config.TABLES_DIR, data["class_id"].unique())
    course = pd.read_csv(split_csv, usecols=["class_id", "course_code"]).set_index("class_id")["course_code"].to_dict()
    ytr = data.loc[data["split"] == "train", "y"].to_numpy()
    frame = data[data["split"] == "test"].reset_index(drop=True)
    p_f, p_x = np.asarray(T["test"]), np.asarray(res["xgb_mlp_tcn"]["test"])
    tau_f, tau_x = baselines.tau_like_truth(ytr, comb["oof"]), baselines.tau_like_truth(ytr, res["xgb_mlp_tcn"]["oof"])
    print(f"thresholds: final {tau_f:.4f}, XGBoost + MLP + TCN {tau_x:.4f}")

    checks = {"MLP + TCN": baselines.guess_scores(frame, p_f, tau_f, raises),
              "XGBoost + MLP + TCN": baselines.guess_scores(frame, p_x, tau_x, raises)}
    scores_figure(report)
    five_classes_figure(frame, p_f, tau_f, p_x, tau_x, raises, course)
    all_test_figure(frame, [("MLP + TCN", p_f, tau_f), ("XGBoost + MLP + TCN", p_x, tau_x)], raises, checks)
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
