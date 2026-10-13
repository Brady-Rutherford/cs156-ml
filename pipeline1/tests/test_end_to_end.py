"""End-to-end: every stage on the synthetic world, through the real CLI, in a sandbox.

`oncue.driver.run_synthetic` writes six invented 20-minute classes in the real on-disk
layout and runs select -> words -> events -> pcw -> frames -> embed -> scan -> track ->
episodes -> retime -> qa -> labels -> features -> train -> evaluate -> controls -> figures ->
results-bundle -> privacy. The saved synthetic ASR stands in for `transcribe`, synthetic
frames for the recording, and the offline hashing model for bge-small. The run happens
once per module (about a minute: 15 model conditions); the tests below inspect its outputs.

Also here: regression tests for the interface fixes made while wiring the stages together
(M3 on all-NaN columns, left-censored runs that end in NA, class QA, the offline encoder,
[ME] in the N-name count).
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest
from oncue import (
    causal,
    config,
    driver,
    embed,
    labels,
    models,
    privacy,
    qa,
    report,
    schemas,
    synthetic,
)
from oncue import features as F

N_CLASSES, MINUTES = 6, 20.0          # >= 6 calibration windows and LOCO anchors need ~20 min
TEST_CONDITIONS = driver.DEFAULT_CONDITIONS   # every MVP model row (§7) plus K1-M4
STAGES = ["select", "words", "events", "pcw", "frames", "embed", "scan", "track", "episodes",
          "retime", "qa", "labels", "features", "train", "evaluate", "controls", "figures",
          "results-bundle", "privacy"]
REAL_OUTPUTS = [config.PIPELINE_DIR / d for d in ("data", "results", "figures", "audit")]
HEX12 = re.compile(r"^[0-9a-f]{12}$")


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    before = {p: p.exists() for p in REAL_OUTPUTS}
    paths = dict(driver.sandbox_paths("/unused"))
    saved = {k: getattr(config, k) for k in paths}
    salt_env = {k: os.environ.get(k) for k in config.ID_SALT_ENV}
    res = driver.run_synthetic(tmp_path_factory.mktemp("e2e"), n_classes=N_CLASSES,
                               minutes=MINUTES, conditions=TEST_CONDITIONS, hgb_max_iter=20,
                               n_boot=20, workers=2, echo=None)
    res.before = before
    res.config_restored = all(getattr(config, k) == v for k, v in saved.items())
    res.env_restored = all(os.environ.get(k) == v for k, v in salt_env.items())
    return res


@pytest.fixture(scope="module")
def world():
    return synthetic.world(N_CLASSES, config.SEED, minutes=MINUTES)


@pytest.fixture(scope="module")
def tables(e2e):
    return causal.load_tables(causal.CLASS_TABLES + ("points",),
                              tables_dir=e2e.root / "tables")


def _dev_ok(classes: pd.DataFrame) -> list[int]:
    ok = classes["qa_pass"].astype(bool) & ~classes["in_vault"].astype(bool)
    return classes.loc[ok, "class_id"].astype(int).tolist()


# --------------------------------------------------------------------------- the run


def test_every_stage_ran_in_order(e2e):
    assert [s.name for s in e2e.stages] == STAGES
    failed = [(s.name, s.rc, s.error) for s in e2e.stages if not s.ok]
    assert not failed and e2e.ok


def test_driver_refuses_a_sandbox_inside_pipeline1(tmp_path):
    for bad in (config.PIPELINE_DIR, config.PIPELINE_DIR / "data", config.PIPELINE_DIR / "x"):
        with pytest.raises(ValueError, match="outside"):
            driver.run_synthetic(bad, n_classes=1, echo=None)
    driver.check_root(tmp_path)                                # fine


def test_sandbox_left_no_trace(e2e):
    """No real data dir, results, figures or audit output appeared; config and env restored."""
    assert all(p.exists() == e2e.before[p] for p in REAL_OUTPUTS)
    assert e2e.config_restored and e2e.env_restored
    assert config.load_env is not None and config.TABLES_DIR == config.DATA_DIR / "tables"


def test_every_table_matches_its_schema(e2e):
    td = e2e.root / "tables"
    for name in ("classes", "words", "frames", "voice", "segments", "chat", "polls",
                 "presence", "pcw_blocks", "hand_frames", "hand_episodes", "speech_onsets",
                 "features", "results"):
        schemas.validate(pd.read_parquet(td / f"{name}.parquet"), name)
    schemas.validate(pd.read_parquet(td / "points.parquet"), "points", strict=True)
    for cid in e2e.class_ids:
        schemas.validate(pd.read_parquet(td / "hand_tiles" / f"{cid}.parquet"), "hand_tiles")
    emb = np.load(td / "features_emb.npy")
    feats = pd.read_parquet(td / "features.parquet")
    assert emb.shape[0] == len(feats) and emb.shape[1] == 4 * config.EMB_DIM


def test_class_qa_merges_every_stage(e2e, world):
    c = e2e.table("classes").set_index("class_id")
    assert c["phase"].notna().all()                         # --always: the whole world
    assert c["cal_method"].eq("windowed").all() and np.isfinite(c["eps_F_s"]).all()
    assert c["tile_status"].eq("located").all() and c["n_bids"].notna().all()
    assert set(c["pcw_status"]) <= set(config.PCW_STATUSES)
    assert c["src_res"].eq(config.SRC_RES).all() and c["n_sessions"].notna().all()
    multi = [sc.class_id for sc in world if sc.meta["multi_session"]]
    assert multi and not c.loc[multi, "qa_pass"].any()
    assert all("multi_session" in r for r in c.loc[multi, "qa_reason"])
    assert c.loc[~c.index.isin(multi), "qa_pass"].all()
    assert len(_dev_ok(c.reset_index())) >= 4                # a non-vacuous dev set


def test_detected_bids_match_the_truth(e2e, world):
    eps = e2e.table("hand_episodes")
    for sc in world:
        truth = sc.episodes[sc.episodes["ep_type"] == config.EP_BID]["onset_s"].to_numpy()
        got = eps[(eps["class_id"] == sc.class_id) & (eps["ep_type"] == config.EP_BID)]
        assert len(got) == len(truth)
        assert np.abs(np.sort(got["onset_s"].to_numpy()) - np.sort(truth)).max() <= 0.5
        assert not got["retime_failed"].astype(bool).any()


def test_points_follow_the_label_definition(e2e):
    p = e2e.table("points")
    c = e2e.table("classes").set_index("class_id")
    assert set(p["class_id"]) == set(c.index)
    failed = c.index[~c["qa_pass"].astype(bool)]
    assert not p.loc[p["class_id"].isin(failed), "at_risk"].any()
    pos = p[p[config.HEADLINE_LABEL] == 1.0]
    assert len(pos) >= 50
    lead = pos["next_bid_s"] - pos["t_s"]
    assert ((lead > config.G_S) & (lead <= config.H_S + 1e-9)).all()
    assert p.loc[p["at_risk"], config.HEADLINE_LABEL].notna().all()


def test_fast_features_equal_the_causal_reference_on_integrated_tables(e2e, tables):
    """The §4.4 truncation contract, on tables the stages produced (not the truth tables)."""
    enc = F.Memo(F.HashEncoder(48))
    rng = np.random.default_rng(3)
    classes = tables["classes"]
    for cid in _dev_ok(classes)[:2]:
        tb = causal.class_tables(tables, cid)
        info = causal.class_info(tb["classes"], cid)
        eps = tb["hand_episodes"]
        near = np.concatenate([eps["onset_s"].to_numpy(float) + d for d in (-3.1, 0.6, 2.4)]
                              + [eps["offset_live_s"].dropna().to_numpy(float) + 0.3])
        ts = np.sort(np.concatenate([rng.uniform(0, info.T_end, 25), near]))
        fast = F.build(cid, ts, tb, encoder=enc)
        ref = causal.reference_frame([causal.CausalView(cid, t, tb, encoder=enc) for t in ts])
        pd.testing.assert_frame_equal(fast.reset_index(drop=True), ref.reset_index(drop=True),
                                      check_exact=False, rtol=1e-5, atol=1e-6)
        rules = labels.causal_rules(ts, tb, info)
        for i, t in enumerate(ts):
            view = causal.CausalView(cid, t, tb, encoder=enc)
            assert view.causal_rules() == {r: bool(rules[r].iloc[i])
                                           for r in config.EXCL_CAUSAL}, (cid, t)


def test_train_warns_when_class_qa_was_skipped():
    from oncue import experiments

    data = experiments.toy_data(n_classes=3, minutes=4.0, emb_dim=2)
    assert experiments.qa_undecided(data) == 0
    data.classes["qa_pass"] = data.classes["qa_pass"].astype("boolean")
    data.classes.loc[0, "qa_pass"] = pd.NA
    assert experiments.qa_undecided(data) == 1


def test_models_score_every_condition(e2e):
    res = e2e.table("results")
    pooled = res[res["fold"].isna()].set_index("condition")
    assert set(TEST_CONDITIONS) <= set(pooled.index)
    dev = len(_dev_ok(e2e.table("classes")))
    for cond in TEST_CONDITIONS:
        row = pooled.loc[cond]
        assert row["n_classes"] == dev and row["n_points"] > 0 and row["n_onsets"] > 0
        assert np.isfinite(row["ap_within"]) and np.isfinite(row["ap"])
    # the synthetic world plants professor questions before bids: rule R and the
    # structural LR beat the always-on and prior baselines on within-class AP
    assert pooled.loc["K0-R", "ap_within"] > pooled.loc["K0-AO", "ap_within"]
    assert pooled.loc["K0-B1", "ap_within"] > pooled.loc["K0-B0", "ap_within"]
    for cond in TEST_CONDITIONS:
        oof = pd.read_parquet(e2e.root / "oof" / f"{cond}.parquet")
        schemas.validate(oof, "oof", strict=True)
        vault = [c for c in e2e.class_ids if config.in_vault(c)]
        assert not oof["class_id"].isin(vault).any()


def test_controls_and_figures_were_written(e2e):
    for name in ("p-clock", "callon", "n-name"):
        assert (e2e.root / "controls" / f"{name}.json").exists()
    figs = {p.name for p in (e2e.root / "figures").iterdir()}
    for name in ("pr_curves", "ap_within", "reliability", "comparison_split_tax"):
        assert {f"{name}.svg", f"{name}.png"} <= figs
    assert "comparison_table.csv" in figs


def test_results_bundle_is_privacy_safe(e2e):
    out = e2e.root / "results"
    files = {p.name for p in out.iterdir()}
    assert {"results.parquet", "bids.parquet", "funnel.csv"} <= files
    assert {f"oof_{c}.parquet" for c in TEST_CONDITIONS} <= files
    assert not any(n.endswith("_train.parquet") for n in files)
    assert "left_censored" in pd.read_parquet(out / "bids.parquet")      # never an onset
    raw = {str(c) for c in e2e.class_ids}
    for name in files - {"results.parquet", "funnel.csv"}:
        ids = pd.read_parquet(out / name).filter(regex="^class_(id|hash)$").iloc[:, 0]
        assert ids.astype(str).str.match(HEX12).all() and not ids.astype(str).isin(raw).any()
    roster = privacy.load_roster(e2e.root / "scribe" / "raw")
    assert roster and not privacy.scan([out, e2e.root / "figures"], roster,
                                       privacy.load_hashed_ids(e2e.root / "tables"))
    funnel = pd.read_csv(out / "funnel.csv").set_index("stage")
    assert funnel.loc["synced", "classes"] == N_CLASSES
    assert funnel.loc["dev", "classes"] == len(_dev_ok(e2e.table("classes")))
    assert (funnel["classes"].diff().fillna(0).iloc[:-2] <= 0).all()   # cumulative
    assert funnel.loc["dev", "points_at_risk"] > 0


# --------------------------------------------------------------------------- integration fixes


def test_m3_fits_with_an_all_nan_feature_column():
    """sklearn 1.9 HGB raises on an all-NaN column; M3 must drop it in-fold."""
    from oncue import experiments

    data = experiments.toy_data(n_classes=4, emb_dim=4)
    X = data.X.assign(ctx_session_frac=np.nan)
    y = data.points[config.HEADLINE_LABEL].fillna(0).to_numpy()
    est = models.make("M3", "F-struct", hyper={"n_star": 5}).fit(X, y)
    assert np.isfinite(est.predict_proba(X)[:, 1]).all()
    tf = models.ToFloat(drop_empty=True).fit(pd.DataFrame({"a": [1.0, np.nan],
                                                           "b": [np.nan, np.nan]}))
    assert list(tf.get_feature_names_out()) == ["a"] and list(tf.transform(
        pd.DataFrame({"a": [2.0], "b": [3.0]})).columns) == ["a"]


def test_left_censored_run_ending_in_na_stops_counting_as_hand_up(small_world):
    """A left-censored UP run never lowered live (it ended in NA) is hand_up only until its
    observation ends (the live state holds through 1 s of NA, then goes NA), in the fast
    rules and the truncated reference alike. Both read the hand samples, not episodes."""
    sc = small_world.classes[0]
    tb = synthetic.truth_tables(sc)
    busy = sc.episodes[["onset_s", "offset_s"]].to_numpy(float)
    a = next(x for x in np.arange(150.0, sc.T_end_s - 120, 10.0)
             if not ((busy[:, 0] < x + 90) & (busy[:, 1] > x - 30)).any())
    hf = tb["hand_frames"].copy()
    t = hf["t_s"].to_numpy(float)
    na = ((t >= a - 5) & (t < a)) | ((t >= a + 20) & (t < a + 40))
    hf.loc[na, ["raw_state", "live_state"]] = config.STATE_NA
    hf.loc[na, "tile_found"] = False
    up = (t >= a) & (t < a + 20)
    hf.loc[up, ["raw_state", "live_state"]] = config.STATE_UP
    hf.loc[up, "shape_ok"] = True
    tb["hand_frames"] = hf
    row = tb["hand_episodes"].iloc[[0]].copy()
    row[["ep_id", "onset_s", "onset_retimed_s", "offset_s", "dur_s"]] = [99, a, a, a + 20, 20]
    row["offset_live_s"] = np.nan
    row["left_censored"] = True
    row["ep_type"], row["outcome"] = config.EP_BID, "open"
    tb["hand_episodes"] = pd.concat([tb["hand_episodes"], row], ignore_index=True)
    info = causal.class_info(tb["classes"], sc.class_id)
    ts = np.array([a + 5.0, a + 19.0, a + 21.0, a + 60.0])
    fast = labels.causal_rules(ts, tb, info)["hand_up"].tolist()
    assert fast == [True, True, False, False]
    enc = F.Memo(F.HashEncoder(16))
    assert [causal.CausalView(sc.class_id, t, tb, encoder=enc).causal_rules()["hand_up"]
            for t in ts] == fast


def _qa_classes(**cols) -> pd.DataFrame:
    base = {"class_id": [1, 2, 3, 4, 5], "n_rec_sessions": [1, 1, 2, 1, 2],
            "phase": ["A"] * 5, "in_vault": [False] * 5, "multi_session_ok": [pd.NA] * 5}
    base.update(cols)
    return pd.DataFrame(base)


def test_qa_rules_and_reason_order():
    timing = pd.DataFrame({
        "class_id": [1, 2, 3, 4, 5], "n_words": [10, 10, 10, 10, 10],
        "src_res": [config.SRC_RES] * 4 + [None],
        "cal_qa_pass": [True, False, False, True, True],
        "cal_qa_reason": ["ok", "cal_method_global", "resid_p95", "ok", "ok"],
        "cal_method": ["windowed", "global", "windowed", "windowed", "windowed"],
        "cal_status": ["ok"] * 5,
        "eps_F_s": [0.5] * 5})
    hand = pd.DataFrame({"class_id": [1, 2, 3, 4, 5], "tile_status": ["located"] * 3
                         + ["unlocated", "located"], "av_lag_s": [0.1, 0.12, 0.08, 0.1, 2.0],
                         "n_bids": [3, 2, 1, 0, 4]})
    out = qa.apply(_qa_classes(), timing, hand).set_index("class_id")
    assert out.loc[1, "qa_pass"] and out.loc[1, "qa_reason"] == "ok"
    assert out.loc[2, "qa_reason"] == "cal_method_global"           # calibration only
    assert out.loc[3, "qa_reason"] == "multi_session+resid_p95"     # neither relaxable alone
    assert out.loc[4, "qa_reason"] == "tile_unlocated"
    assert out.loc[5, "qa_reason"] == "rendition_unknown+av_lag+multi_session"
    assert not out["qa_pass"].iloc[1:].any()
    assert out.loc[1, "multi_session_ok"] and not out.loc[3, "multi_session_ok"]
    assert not bool(out.loc[5, "av_ok"]) and bool(out.loc[1, "av_ok"])
    # a multi-session class cleared by hand (D14) is not failed for it
    cleared = qa.apply(_qa_classes(multi_session_ok=[pd.NA, pd.NA, True, pd.NA, pd.NA]),
                       timing, hand).set_index("class_id")
    assert cleared.loc[3, "qa_reason"] == "resid_p95"
    # no words stage: no_asr; no detector QA: not_scanned
    lone = qa.apply(_qa_classes()).set_index("class_id")
    assert lone.loc[1, "qa_reason"] == "no_asr+not_scanned"
    funnel = qa.funnel_table(out.reset_index())
    assert funnel.set_index("stage").loc["qa_pass", "classes"] == 1


def test_offline_hash_model_is_deterministic_and_uncached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMB_CACHE_DIR", tmp_path / "emb")
    e1, e2 = embed.Embedder(embed.HASH_MODEL), embed.Embedder(embed.HASH_MODEL)
    assert e1.cache is None
    v1 = e1.encode(["the model overfits", "", "the model overfits"])
    v2 = e2.encode(["the model overfits"])
    assert v1.shape == (3, config.EMB_DIM) and np.array_equal(v1[0], v2[0])
    assert np.allclose(np.linalg.norm(v1[[0, 2]], axis=1), 1) and not v1[1].any()
    assert not (tmp_path / "emb").exists()
    # a real model name keeps the call-time cache default
    assert embed.Embedder("some/model").cache.root == tmp_path / "emb" / "some__model"


def test_name_count_includes_brady_me_token():
    from oncue import controls

    assert controls._name_count("thanks [ME] and [NAME], then [NAME]") == 3


# --------------------------------------------------------------------------- notebook helpers


def test_report_rebuilds_aggregates_from_the_committed_bundle(e2e, tmp_path):
    """What the notebook's metric, figure and table cells do with results/ alone."""
    import matplotlib.pyplot as plt
    from oncue import plots

    b = report.load_bundle(e2e.root / "results")
    assert not b.empty and set(TEST_CONDITIONS) <= set(b.oofs)
    assert all("in_G" in df for df in b.oofs.values())        # event metrics re-runnable
    hyp = report.hypothesis_table(b.oofs, b.bids, n_boot=50)
    h = hyp.set_index("hypothesis")
    assert {"H1a", "H1b", "H3", "H4", "H5"} <= set(h.index)
    assert np.isfinite(h.loc[["H1a", "H1b", "H4", "H5"], "estimate"]).all()
    assert np.isnan(h.loc["H3", "estimate"])                   # F1 conditions not run here
    assert (h.loc[["H1a", "H4"], "lo"] <= h.loc[["H1a", "H4"], "hi"]).all()
    assert h["p_adj"].notna().any()
    plots.fig_forest(hyp.dropna(subset=["estimate"]))
    # H2 needs the segments table: local only
    segs = e2e.table("segments")
    local = {c: pd.read_parquet(e2e.root / "oof" / f"{c}.parquet")
             for c in ("K0-M3", "F0-Fstruct")}
    h2 = report.hypothesis_table(local, e2e.table("hand_episodes"), segments=segs,
                                 n_boot=30, n_rep=3).set_index("hypothesis")
    assert {"H2a", "H2b"} <= set(h2.index)
    curves = report.froc_curves(b.oofs, b.bids, ("K0-M3", "K0-R"))
    assert set(curves) == {"M3 / F-all", "R"}                  # fig_froc display names
    assert all(c["recall"].between(0, 1).all() and (c["far"] >= 0).all()
               for c in curves.values())
    ao = report.always_on_point(b.oofs["K0-AO"], b.bids)
    assert ao is not None and ao[0] > 1.0                      # always-on fires a lot
    plots.fig_froc(curves, ao_point=ao)
    plots.fig_funnel(b.funnel)
    plt.close("all")
    out = report.write_summary(tmp_path / "summary.json",
                               report.summary(b.results, hyp, b.funnel))
    data = pd.read_json(out, typ="series")
    assert "K0-M3" in data["conditions"] and "H1a" in data["hypotheses"]
    prov = report.provenance()
    assert {"git_head", "prereg", "scribe", "packages"} <= set(prov)


def test_peri_onset_curves_from_local_tables(e2e):
    feats = e2e.table("features")
    onsets = report.onsets_from_bids(e2e.table("hand_episodes"))
    cur = report.peri_onset_curves(feats, onsets, ["q_open", "spk_silence_s"])
    assert set(cur["feature"]) == {"q_open", "spk_silence_s"}
    assert cur["rel_s"].min() == -120 and cur["rel_s"].max() == 30
    assert cur["mean"].notna().any() and cur["shifted_mean"].notna().any()
