"""Stress test: clock calibration when Forum never logs the professor's mic.

Simulates classes with a known Forum->audio clock (offset + drift), students-only voice events
and diarized ASR words, then scores calibration methods by their mean timing error over the
class, and the downstream speaker attribution (Scribe fusion) and role rule (pipeline).

Run from pipeline1/:  ../.venv/bin/python 04_words_events/stress_test/stress_sync.py 40
Writes stress_sync_rows.csv and stress_sync_summary.csv next to this file (synthetic only).
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scribe_core.calibrate import calibrate, make_shift, shift
from scribe_core.fuse import (
    EDGE,
    calibration_words,
    cluster_to_user,
    fuse,
    smooth,
    speech_without_mic,
)

HERE = Path(__file__).resolve().parent
T = 5400.0  # 90-minute class


def words_for(rng, s, e):
    out, t = [], s
    while t < e - 0.1:
        d = rng.uniform(0.18, 0.45)
        we = min(e, t + d)
        out.append((t, we))
        t = we + (rng.uniform(0.3, 1.2) if rng.random() < 0.08 else rng.uniform(0.03, 0.12))
    return out


def simulate(rng, *, stud_share=0.33, n_students=12, precision="ms", offset_max=3.0, drift=0.0,
             prof_split=0.0, flip=0.0, merge=0, bleed=0.0, noise_per_min=0.0, miss=0.0):
    offset = rng.uniform(-offset_max, offset_max)
    drift = rng.uniform(-drift, drift)
    b0 = rng.uniform(1500, 3000)
    breakout = (b0, b0 + rng.uniform(300, 600))
    mean_p = 40.0
    mean_turn = stud_share / (1 - stud_share) * mean_p / 1.7
    turns, t = [], 5.0
    while t < T - 20:
        if breakout[0] <= t < breakout[1]:
            t = breakout[1] + rng.uniform(1, 3)
            continue
        e = min(t + rng.uniform(5, 2 * mean_p), T - 5)
        if t < breakout[0] < e:
            e = breakout[0] - 1
        if e > t + 1:
            turns.append(("P", t, e))
        t = e + rng.uniform(0.6, 3.0)                       # unmute delay
        for _ in range(1 + rng.poisson(0.7)):
            if t >= T - 20 or breakout[0] <= t < breakout[1]:
                break
            e = min(t + rng.uniform(3, 2 * mean_turn), T - 5)
            if t < breakout[0] < e:
                e = breakout[0] - 1
            if e > t + 1:
                turns.append((f"S{rng.integers(n_students)}", t, e))
            t = e + rng.uniform(0.3, 2.0)

    # diarization clusters
    cl_of = {f"S{i}": i + 1 for i in range(n_students)}
    for _ in range(merge):
        a, b = rng.choice(n_students, 2, replace=False)
        cl_of[f"S{b}"] = cl_of[f"S{a}"]
    all_cl = [0, 99] + sorted(set(cl_of.values()))
    words = []
    for who, s, e in turns:
        c = (99 if rng.random() < prof_split else 0) if who == "P" else cl_of[who]
        for ws, we in words_for(rng, s, e):
            cc = int(rng.choice(all_cl)) if rng.random() < flip else c
            words.append({"start": ws, "end": we, "cluster": cc, "speaker": who,
                          "cluster_conf": 0.9})

    # Forum voice events (students only), in audio time first
    ev = []
    for who, s, e in turns:
        if who == "P":
            if rng.random() < bleed:                         # professor's voice on a student mic
                a = rng.uniform(s, max(s, e - 4))
                ev.append((f"S{rng.integers(n_students)}", a, min(e, a + rng.uniform(1, 4))))
            continue
        if rng.random() < miss:
            continue
        ws = [w for w in words if w["speaker"] == who and s <= w["start"] < e]
        if not ws:
            continue
        piece = [ws[0]]
        pieces = []
        for w in ws[1:]:
            if w["start"] - piece[-1]["end"] > 1.0:
                pieces.append(piece)
                piece = []
            piece.append(w)
        pieces.append(piece)
        for p in pieces:
            a = p[0]["start"] - rng.uniform(0.05, 0.4)
            b = p[-1]["end"] + rng.uniform(0.1, 0.6)
            while a < b:                                     # Forum fragments (~3 s)
                f = min(b, a + rng.uniform(2, 4))
                ev.append((who, a, f))
                a = f + 0.15
    for _ in range(rng.poisson(noise_per_min * T / 60)):
        a = rng.uniform(0, T - 2)
        ev.append((f"S{rng.integers(n_students)}", a, a + rng.uniform(0.3, 1.5)))

    # to the Forum clock: audio = f + offset + drift * f
    voice = []
    for who, a, b in ev:
        fa, fb = (a - offset) / (1 + drift), (b - offset) / (1 + drift)
        if precision == "s":                                 # END timestamp in whole seconds
            r = fb - np.floor(fb)
            fa, fb = fa - r, fb - r
        voice.append({"start": fa, "end": fb, "user_id": who})
    voice.sort(key=lambda v: v["start"])
    words.sort(key=lambda w: (w["start"], w["end"]))
    talk = defaultdict(float)
    for who, s, e in turns:
        talk[who == "P"] += e - s
    return {"words": words, "voice": voice, "offset": offset, "drift": drift, "turns": turns,
            "stud_share": talk[False] / max(talk[False] + talk[True], 1)}


def oracle_voice(sim):
    """The old (wrong) world: professor events present, as if Forum logged them."""
    off, dr = sim["offset"], sim["drift"]
    prof = [{"start": (s - off) / (1 + dr), "end": (e - off) / (1 + dr), "user_id": "P"}
            for who, s, e in sim["turns"] if who == "P"]
    return sorted(sim["voice"] + prof, key=lambda v: v["start"])


def onset_intervals(iv, gap=0.5, width=0.3):
    out, last = [], -1e9
    for s, e in sorted(iv):
        if s - last >= gap:
            out.append({"start": s, "end": s + width})
        last = max(last, e)
    return out


def quiet_clusters(words, voice, min_share, min_s=30.0):
    total, quiet = speech_without_mic(words, voice)
    return {c for c, s in total.items() if s >= min_s and quiet[c] / s >= min_share}


def methods(sim):
    w, v = sim["words"], sim["voice"]
    nocl = [{**x, "cluster": None} for x in w]
    out = {}
    out["A all words"] = calibrate(w, v, T)
    out["B minus prof cluster"] = calibrate(calibration_words(w, v), v, T)
    cal_b = out["B minus prof cluster"]
    vb = shift(v, cal_b) if cal_b["applied"] else v
    drop = quiet_clusters(w, vb, 0.6)
    rest = [x for x in w if x["cluster"] not in drop] or w
    out["B2 iterate, drop all quiet"] = calibrate(rest, v, T)
    drop0 = quiet_clusters(w, v, 0.5)
    out["C drop all quiet (raw clock)"] = calibrate(
        [x for x in w if x["cluster"] not in drop0] or w, v, T)
    out["E onsets only (no clusters)"] = calibrate(
        onset_intervals([(x["start"], x["end"]) for x in w]),
        onset_intervals([(x["start"], x["end"]) for x in v]), T)
    out["A' all words, no clusters"] = calibrate(nocl, v, T)
    out["O oracle: prof had events"] = calibrate(w, oracle_voice(sim), T)
    return out


def timing_error(cal, sim):
    tf = np.arange(0, T, 10.0)
    true = tf + sim["offset"] + sim["drift"] * tf
    if not cal["applied"]:
        return float(np.mean(np.abs(true - tf))), False
    f = make_shift(cal)
    pred = np.array([f(x) for x in tf])
    return float(np.mean(np.abs(pred - true))), True


def attribution(sim, cal):
    """Duration-weighted share of words credited to the right person (Scribe fusion with the
    professor fix), and the pipeline's prof/student role rule accuracy."""
    w = sim["words"]
    v = shift(sim["voice"], cal) if cal["applied"] else sim["voice"]
    mapping = cluster_to_user(w, v, teacher="P")
    fused = smooth(fuse(w, v, mapping, teacher="P"))
    dur = np.array([x["end"] - x["start"] for x in w])
    right = np.array([f["user_id"] == x["speaker"] for f, x in zip(fused, w)])
    prof_true = np.array([x["speaker"] == "P" for x in w])
    prof_pred = np.array([f["user_id"] == "P" for f in fused])
    # pipeline rule: no student event within EDGE -> professor
    vs = np.array([x["start"] for x in v]); ve = np.array([x["end"] for x in v])
    rule_prof = np.array([not np.any((vs - EDGE < x["end"]) & (ve + EDGE > x["start"]))
                          for x in w])
    return {"speaker_acc": float((dur * right).sum() / dur.sum()),
            "prof_recall": float((dur * (prof_true & prof_pred)).sum() / (dur * prof_true).sum()),
            "prof_precision": float((dur * (prof_true & prof_pred)).sum()
                                    / max((dur * prof_pred).sum(), 1e-9)),
            "rule_role_acc": float((dur * (rule_prof == prof_true)).sum() / dur.sum())}


FAMILIES = {
    "1 baseline (ms, small offset)": {},
    "2 whole-second timestamps (2023)": {"precision": "s"},
    "3 large offset (up to 25 s)": {"offset_max": 25.0},
    "4 clock drift (up to 2.7 s/90 min)": {"drift": 0.0005},
    "5 lecture-heavy (15% student talk)": {"stud_share": 0.15},
    "6 discussion-heavy (55% student talk)": {"stud_share": 0.55},
    "7 messy diarization": {"prof_split": 0.3, "flip": 0.05, "merge": 2},
    "8 bleed + mic noise + missing events": {"bleed": 0.3, "noise_per_min": 1.0, "miss": 0.1},
    "9 worst case (all of the above, 2023)": {"precision": "s", "offset_max": 25.0,
                                              "drift": 0.0005, "stud_share": 0.2,
                                              "prof_split": 0.3, "flip": 0.05, "merge": 2,
                                              "bleed": 0.3, "noise_per_min": 1.0, "miss": 0.1},
}


def main(n: int = 20, seed: int = 156):
    rng = np.random.default_rng(seed)
    rows = []
    for fam, kw in FAMILIES.items():
        for i in range(n):
            sim = simulate(rng, **kw)
            cals = methods(sim)
            for m, cal in cals.items():
                err, applied = timing_error(cal, sim)
                row = {"family": fam, "i": i, "method": m, "err": err, "applied": applied}
                if m in ("A all words", "B minus prof cluster", "B2 iterate, drop all quiet",
                         "O oracle: prof had events"):
                    row.update(attribution(sim, cal))
                rows.append(row)
        print(f"done {fam}", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "stress_sync_rows.csv", index=False)
    g = df.groupby(["family", "method"])
    summary = pd.DataFrame({
        "median_err_s": g["err"].median().round(3),
        "p90_err_s": g["err"].quantile(0.9).round(3),
        "within_0.25s": g["err"].apply(lambda e: (e <= 0.25).mean()).round(2),
        "within_0.5s": g["err"].apply(lambda e: (e <= 0.5).mean()).round(2),
        "bad_>1s": g["err"].apply(lambda e: (e > 1.0).mean()).round(2),
        "unverified": g["applied"].apply(lambda a: (~a).mean()).round(2),
        "speaker_acc": g["speaker_acc"].mean().round(3),
        "prof_recall": g["prof_recall"].mean().round(3),
        "prof_precision": g["prof_precision"].mean().round(3),
        "rule_role_acc": g["rule_role_acc"].mean().round(3),
    })
    summary.to_csv(HERE / "stress_sync_summary.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(summary)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
