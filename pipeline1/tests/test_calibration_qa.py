"""Class calibration QA (spec §3 class QA, §4.3, §4.4 `test_calibration_qa`).

A class is headline-eligible on timing only when Scribe's calibration was applied with status
ok/piecewise, `method == "windowed"` with >= 6 usable windows, max_residual_s <= 2.5 s, and the
measured residuals (calibrated voice start vs nearest ASR word start) have p95 <= 2.5 s and
median <= 0.5 s. A global-fallback fit reports max_residual_s = 0 and must NEVER pass, not even
in the 3 s sensitivity population. Synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from oncue import config, synthetic, transcribe

SALT = "synthetic-test-salt"


def _build(sc, *, asr=None, raw_events=None):
    return transcribe.build_words(sc.class_id, asr or sc.asr, sc.raw["class"],
                                  sc.raw["events"] if raw_events is None else raw_events,
                                  me_id=sc.meta["me_id"], salt=SALT)


@pytest.fixture(scope="module")
def builds(world):
    return {sc.class_id: _build(sc) for sc in world}


def _cal(**kw):
    cal = {"status": "ok", "applied": True, "method": "windowed", "windows": [[0, 0, 1]] * 8,
           "max_residual_s": 0.4, "offset": 0.2, "drift": 0.0,
           "pieces": [{"t_from": None, "offset": 0.2, "drift": 0.0}]}
    cal.update(kw)
    return cal


# --------------------------------------------------------------------------- real calibrate()


def test_windowed_class_passes_and_recovers_the_offset(builds, world):
    sc = world.classes[0]
    t = builds[sc.class_id].timing
    assert t["cal_applied"] and t["cal_method"] == "windowed"
    assert t["cal_n_windows"] >= config.CAL_MIN_WINDOWS
    assert t["cal_qa_pass"] and t["cal_qa_reason"] == "ok"
    assert abs(t["cal_offset_s"] - sc.meta["cal_offset_s"]) < 0.15
    assert t["resid_n"] > 0
    assert t["resid_med_s"] <= config.CAL_RESID_MED_MAX_S
    assert t["resid_p95_s"] <= config.CAL_RESID_P95_MAX_S
    assert t["eps_F_s"] == pytest.approx(
        max(config.EPS_F_MIN_MS_S, t["resid_p95_s"], t["cal_max_residual_s"]))
    assert t["eps_F_s"] <= config.EPS_F_HEADLINE_MAX_S


def test_global_fallback_is_never_headline(world):
    """Fewer than 3 usable windows: Scribe fits the whole class and reports
    max_residual_s = 0 and status "ok". QA must still fail, in both populations."""
    sc = world.classes[0]
    cut = 280.0                                    # windows at 0 s and 150 s only
    cut_ms = sc.meta["sessions"][0][0] + (cut - sc.meta["cal_offset_s"]) * 1000
    evs = [e for e in sc.raw["events"]
           if e.get("event-type", e.get("event_type")) != "voice" or e["timestamp"] < cut_ms]
    asr = {**sc.asr, "words": [w for w in sc.asr["words"] if w["end"] < cut]}
    b = _build(sc, asr=asr, raw_events=evs)
    t = b.timing
    assert b.calibration.cal["method"] == "global" and t["cal_status"] == "ok"
    assert t["cal_max_residual_s"] == 0.0 and t["cal_applied"]
    assert t["resid_med_s"] <= config.CAL_RESID_MED_MAX_S       # residuals look fine ...
    assert not t["cal_qa_pass"] and t["cal_qa_reason"] == "cal_method_global"
    assert not t["cal_qa_sens_pass"]                               # ... and it still fails
    # the bare rule, with perfect residuals
    assert config.calibration_qa(b.calibration.cal, 0.0, 0.0) == (False, "cal_method_global")


def test_planted_global_calibration_fails(world, truth):
    sc = world.classes[2]
    cal = sc.meta["calibration"]
    assert cal["method"] == "global" and cal["max_residual_s"] == 0.0
    row = transcribe.calibration_qa_row(cal, np.full(40, 0.05), "ms")
    assert not row["cal_qa_pass"] and row["cal_qa_reason"] == "cal_method_global"
    assert not row["cal_qa_sens_pass"]
    assert not bool(truth[sc.class_id]["classes"]["qa_pass"].iloc[0])


def test_fewer_than_six_windows_fails(small_world):
    sc = small_world.classes[0]                    # 12 minutes -> 4 usable windows
    t = _build(sc).timing
    assert t["cal_method"] == "windowed"
    assert t["cal_n_windows"] < config.CAL_MIN_WINDOWS
    assert not t["cal_qa_pass"] and t["cal_qa_reason"] == "cal_few_windows"
    assert not t["cal_qa_sens_pass"]


def test_unverified_calibration_is_identity_with_fallback_eps(world):
    sc = world.classes[0]
    no_voice = [e for e in sc.raw["events"]
                if e.get("event-type", e.get("event_type")) != "voice"]
    b = _build(sc, raw_events=no_voice)
    t = b.timing
    assert t["cal_status"] == "unverified" and not t["cal_applied"]
    assert not t["cal_qa_pass"] and t["cal_qa_reason"] == "cal_not_applied"
    assert t["eps_F_s"] == config.EPS_F_FALLBACK_S
    w = b.words                                    # no Forum voice: every role reads unk
    assert (w["role"] == config.ROLE_UNK).all() and w["spk_hash"].isna().all()
    assert np.allclose(w["role_avail_s"], w["end_s"] + config.EPS_W_S)


def test_whole_second_class_has_eps_floor_of_one_second(builds, world):
    sc = world.classes[1]
    assert not sc.meta["precise"]
    t = builds[sc.class_id].timing
    assert t["timing_precision"] == "s"
    assert t["eps_F_s"] >= config.EPS_F_MIN_WHOLE_S


def test_every_class_row_is_consistent(builds):
    for b in builds.values():
        t = b.timing
        ok, reason = config.calibration_qa(b.calibration.cal, t["resid_med_s"], t["resid_p95_s"])
        assert (t["cal_qa_pass"], t["cal_qa_reason"]) == (ok, reason)
        assert t["eps_F_s"] >= config.eps_f(t["timing_precision"])


# --------------------------------------------------------------------------- residuals, rules


def test_residuals_use_quiet_events_and_a_2s_search():
    vs = [10.0, 11.0, 20.0, 30.5, 50.0]
    ve = [10.5, 15.0, 21.0, 31.0, 51.0]
    words = [10.2, 11.1, 20.7, 33.0, 49.0]
    # 11.0 follows speech that ended 0.5 s earlier; 30.5 has no word start within 2 s
    got = transcribe.residuals(vs, ve, words)
    assert np.allclose(np.sort(got), [0.2, 0.7, 1.0])
    assert len(transcribe.residuals([], [], words)) == 0
    assert len(transcribe.residuals(vs, ve, [])) == 0


@pytest.mark.parametrize("cal, resid, ok, reason, sens_ok", [
    (_cal(), [0.1] * 19 + [0.9], True, "ok", True),
    (_cal(status="piecewise"), [0.1] * 20, True, "ok", True),
    (_cal(), [0.1] * 10 + [2.8] * 10, False, "resid_p95", False),       # p95 2.8 > 2.5
    (_cal(), [0.1] * 18 + [2.9] * 2, False, "resid_p95", True),          # p95 in (2.5, 3]
    (_cal(max_residual_s=2.6), [0.1] * 20, False, "cal_max_residual", True),
    (_cal(max_residual_s=3.2), [0.1] * 20, False, "cal_max_residual", False),
    (_cal(windows=[[0, 0, 1]] * 5), [0.1] * 20, False, "cal_few_windows", False),
    (_cal(method="global", max_residual_s=0.0, windows=[[0, 0, 1]] * 2), [0.0] * 20,
     False, "cal_method_global", False),
    (_cal(applied=False), [0.1] * 20, False, "cal_not_applied", False),
    (_cal(), [], False, "resid_p95", False),                             # nothing measured
])
def test_qa_row_rules(cal, resid, ok, reason, sens_ok):
    row = transcribe.calibration_qa_row(cal, np.asarray(resid, float), "ms")
    assert (row["cal_qa_pass"], row["cal_qa_reason"]) == (ok, reason)
    assert row["cal_qa_sens_pass"] == sens_ok


def test_max_residual_is_dropped_not_clamped():
    row = transcribe.calibration_qa_row(_cal(max_residual_s=2.6), np.full(20, 0.1), "ms")
    assert row["eps_F_s"] == pytest.approx(2.6)            # logged as measured ...
    assert not row["cal_qa_pass"]                          # ... and the class leaves headline


def test_timing_table_round_trip(builds, tmp_path):
    rows = [b.timing for b in builds.values()]
    df = transcribe.timing_frame(rows)
    df.to_parquet(tmp_path / f"{transcribe.TIMING_TABLE}.parquet", index=False)
    for b in builds.values():
        cid = b.timing["class_id"]
        got = transcribe.read_timing(cid, tables_dir=tmp_path)
        assert got["eps_F_s"] == pytest.approx(b.timing["eps_F_s"])
        assert got["cal"]["pieces"] == pytest.approx(b.calibration.cal["pieces"])
        assert bool(got["cal_qa_pass"]) == b.timing["cal_qa_pass"]
    assert transcribe.read_timing(123, tables_dir=tmp_path) is None
    assert set(pd.read_parquet(tmp_path / "class_timing.parquet")["cal_qa_reason"]) <= \
        {"ok", "cal_method_global", "cal_few_windows", "resid_p95", "resid_median",
         "cal_max_residual", "cal_not_applied"}


def test_truth_qa_column_agrees_with_rule(world, truth):
    """The synthetic truth's classes.qa_pass uses the same rule (multi-session aside)."""
    for sc in world:
        row = truth[sc.class_id]["classes"].iloc[0]
        ok, _ = config.calibration_qa(sc.meta["calibration"], row["resid_med_s"],
                                      row["resid_p95_s"])
        if not sc.meta["multi_session"]:
            assert bool(row["qa_pass"]) == ok
    assert synthetic.world(1, config.SEED).classes[0].meta["calibration"]["method"] == "windowed"
