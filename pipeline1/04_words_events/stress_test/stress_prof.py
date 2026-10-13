"""Who gets labelled professor: the biggest mic-silent cluster only, every large mic-silent
cluster, or (shipped) every large mic-silent cluster spread over >= 3 five-minute stretches.
Measured on the calibrated clock, with a video/guest scenario that should NOT be called the
professor. Run from pipeline1/:  ../.venv/bin/python 04_words_events/stress_test/stress_prof.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scribe_core.calibrate import calibrate, shift
from scribe_core.fuse import (
    calibration_words,
    cluster_to_user,
    fuse,
    smooth,
    teacher_clusters,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stress_sync as S


def add_offmic(rng, sim, minutes):
    """A video clip / guest: `minutes` of speech in its own cluster, no mic events, replacing
    professor speech in a random stretch."""
    w = sim["words"]
    a = rng.uniform(600, 4000); b = a + minutes * 60
    for x in w:
        if a <= x["start"] < b and x["speaker"] == "P":
            x["speaker"], x["cluster"] = "V", 77
    return sim

def score(sim, cal, mode):
    w = sim["words"]
    v = shift(sim["voice"], cal) if cal["applied"] else sim["voice"]
    mapping = cluster_to_user(w, v, teacher="P")   # shipped: biggest + spread-out extras
    if mode == "biggest only":
        for c in teacher_clusters(w, v)[1:]:
            mapping.pop(c, None)
    elif mode == "all quiet":
        for c in S.quiet_clusters(w, v, 0.6):
            mapping[c] = "P"
    f = smooth(fuse(w, v, mapping, teacher="P"))
    d = np.array([x["end"] - x["start"] for x in w])
    t = np.array([x["speaker"] for x in w]); p = np.array([y["user_id"] for y in f], dtype=object)
    pt, pp = t == "P", p == "P"
    return {"speaker_acc": (d * (p == t)).sum() / d.sum(),
            "prof_recall": (d * (pt & pp)).sum() / (d * pt).sum(),
            "prof_precision": (d * (pt & pp)).sum() / max((d * pp).sum(), 1e-9),
            "offmic_called_prof": (d * ((t == "V") & pp)).sum() / max((d * (t == "V")).sum(), 1e-9)}

FAM = {
    "prof split in 2 clusters (30%)": ({"prof_split": 0.3}, 0),
    "prof split in 2 clusters (50%)": ({"prof_split": 0.5}, 0),
    "3-min video/guest, no mic": ({}, 3),
    "split 30% + 3-min video": ({"prof_split": 0.3}, 3),
    "worst case (2023, all messy)": (dict(S.FAMILIES["9 worst case (all of the above, 2023)"]), 0),
}
rng = np.random.default_rng(7)
rows = []
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for fam, (kw, vid) in FAM.items():
    for i in range(n):
        sim = S.simulate(rng, **kw)
        if vid:
            add_offmic(rng, sim, vid)
        cal = calibrate(calibration_words(sim["words"], sim["voice"]), sim["voice"], S.T)
        for mode in ("biggest only", "all quiet", "shipped: all quiet + spread"):
            rows.append({"family": fam, "mode": mode, **score(sim, cal, mode)})
df = pd.DataFrame(rows)
with pd.option_context("display.width", 200):
    print(df.groupby(["family", "mode"], sort=False).mean().round(3).to_string())
