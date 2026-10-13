"""Models (§6), the experiments registry and runner (§7), the cross-encoder helpers (M6) and
the aggregate-only figures (§14). Tiny synthetic design matrices only (experiments.toy_data:
invented numbers and role-prefixed filler tokens, no words of anyone)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from oncue import config, experiments, models, plots, schemas, xenc
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore", category=UserWarning)
FAST = {"n_inner": 2, "hgb_max_iter": 15}


def _null_scorer(*_, **__):
    return {}


@pytest.fixture(scope="module")
def toy():
    return experiments.toy_data(n_classes=12, minutes=8.0, seed=7)


@pytest.fixture(scope="module")
def split(toy):
    """One outer fold of the toy: (X_train, y_train, X_test, y_test)."""
    P, X = toy.points, toy.X
    ok = experiments.train_mask(P, config.HEADLINE_LABEL)
    fold = P["fold"].to_numpy()
    tr, te = ok & (fold != 0), ok & (fold == 0)
    y = P[config.HEADLINE_LABEL].to_numpy(float)
    return (X[tr].reset_index(drop=True), y[tr], X[te].reset_index(drop=True), y[te])


# --------------------------------------------------------------------------- registry, sets


def test_model_registry_covers_spec_section6():
    spec = {"AO", "B-min", "B0", "R", "B1", "M1", "M2", "M3", "M4", "M5", "M6"}
    assert spec <= set(models.MODELS)
    assert {"M3+stack", "M4-MLP"} <= set(models.MODELS)              # A4, §6 could
    assert models.MODELS["M3"].tuned == "n_star" and models.MODELS["M2"].tuned == "C"
    assert not models.MODELS["AO"].calibrate and not models.MODELS["B0"].calibrate
    assert config.CALIBRATION_METHOD == "sigmoid"                      # A9: never isotonic


def test_featuresets_compose_groups():
    g = models.featureset_groups
    struct = {"ctx", "spk", "me", "me_hand", "poll"}
    assert g("F-struct") == struct
    assert g("F-all") == struct | {"q", "text", "emb", "pcwT"}
    assert g("F-all", branch="H") == g("F-all") | {"pcwA"}
    assert g("F-all−me_hand") == g("F-all") - {"me_hand"}           # spec's Unicode minus
    assert g("F-struct-me_hand") == struct - {"me_hand"}
    assert g("F-all+hist") - g("F-all") == {"hist"}
    assert g("F-all+edu") - g("F-all") == {"edu"}
    assert g("F-all+stack") - g("F-all") == {"stack"}
    assert models.featureset_text_variant("F-name") == "name"
    assert models.featureset_text_variant("F-punct") == "punct"
    with pytest.raises(KeyError):
        g("F-nonsense")


def test_featuresets_agree_with_features_module():
    features = pytest.importorskip("oncue.features")
    for name, spec in features.FEATURESETS.items():
        if name not in models.FEATURESETS or name == "F-min":   # F-min: splines vs ctx_min
            continue
        assert set(spec["groups"]) == models.featureset_groups(name), name


def test_select_columns_respects_allowlist(toy):
    X = toy.X.assign(has_qmark=False, y_H15_g2=1.0, in_G=True, emb_row=0, ctx_min_spl_0=0.1)
    for fs in models.FEATURESETS:
        for kind in ("num", "cat", "text", "emb", "minute", "ctx_tokens"):
            cols = models.select_columns(X, fs, kind)
            for c in cols:
                if c == "class_id":
                    assert kind in ("cat", "text")              # only for class counts
                    continue
                assert config.feature_allowed(c, fs), (fs, kind, c)
                assert not config.BLOCKED_RE.match(c), c
    assert "has_qmark" in models.select_columns(X, "F-punct", "num")
    assert "has_qmark" not in models.select_columns(X, "F-all", "num")
    assert "ctx_min_spl_0" not in models.select_columns(X, "F-all", "num")  # in-fold splines
    assert models.select_columns(X, "F-struct", "text") == []
    assert models.select_columns(X, "F-struct", "emb") == []
    assert "txt_all60_name" in models.select_columns(X, "F-name", "text")
    assert "txt_all60" not in models.select_columns(X, "F-name", "text")
    assert len(models.select_columns(X, "F-all", "emb_v60")) == 8


# --------------------------------------------------------------------------- transformers


def test_text_vectorizer_min_df_counts_training_classes():
    docs = pd.DataFrame({"txt_all60": ["p_common p_rare"] * 4 + ["p_common"] * 6,
                         "class_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    v = models.MultiTextVectorizer("count", min_classes=5, ngram_range=(1, 1)).fit(docs)
    assert "p_common" in v.vocab_["txt_all60"]
    assert "p_rare" not in v.vocab_["txt_all60"]                  # 4 rows but 3 classes
    test = pd.DataFrame({"txt_all60": ["p_unseen p_common"], "class_id": [99]})
    m = v.transform(test)
    assert m.shape == (1, len(v.vocab_["txt_all60"])) and m.sum() == 1   # unseen ignored


def test_course_filter_counts_classes_not_rows():
    X = pd.DataFrame({"ctx_course": ["CS1"] * 50 + ["CS2"] * 3 + ["CS3"] * 3,
                      "class_id": [1] * 50 + [2, 3, 4] + [5, 6, 7]})
    prep = models.CategoryPrep(as_category=True).fit(X)
    out = prep.transform(pd.DataFrame({"ctx_course": ["CS1", "CS2", "NEW", None],
                                       "class_id": [0] * 4}))
    assert list(out["ctx_course"].astype(object)[:3]) == ["other", "CS2", "other"]
    assert pd.isna(out["ctx_course"].iloc[3])
    assert isinstance(out["ctx_course"].dtype, pd.CategoricalDtype)


def test_context_pseudo_tokens():
    ct = models.ContextTokens().fit(None)
    X = pd.DataFrame({"ctx_seg_type": ["debrief"], "ctx_min": [33.0],
                      "me_since_turn_s": [400.0], "q_open": [1.0]})
    names = ct.get_feature_names_out()[ct.transform(X).toarray()[0] > 0]
    assert set(names) == {"ctx_seg_debrief", "ctx_min_30_45", "ctx_sinceme_5_10m",
                          "ctx_qopen"}
    never = ct.transform(X.assign(me_since_turn_s=np.nan, q_open=0.0))
    assert "ctx_sinceme_never" in set(ct.get_feature_names_out()[never.toarray()[0] > 0])


def test_emb_frame_expands_rows_with_missing():
    mat = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
    df = models.emb_frame(mat, [1, None, 0], dim=2)
    assert list(df.columns[:2]) == ["emb_v15_000", "emb_v15_001"]
    assert df.shape == (3, 8)
    assert np.array_equal(df.iloc[0].to_numpy(), mat[1]) and df.iloc[1].isna().all()


# --------------------------------------------------------------------------- models fit/predict


@pytest.mark.parametrize("model_id", ["AO", "B0", "R", "B-min", "B1", "M1", "M2", "M3", "M4",
                                      "M3+stack"])
def test_models_fit_predict_shapes(model_id, split):
    Xtr, ytr, Xte, _ = split
    hyper = {"n_star": 15} if model_id in ("M3", "M3+stack") else None
    est = models.make(model_id, hyper=hyper)
    cm = models.CalibratedModel(est, calibrate=models.MODELS[model_id].calibrate, n_inner=2)
    cm.fit(Xtr, ytr)
    p_raw, p_cal = cm.predict(Xte)
    assert p_raw.shape == p_cal.shape == (len(Xte),)
    assert np.all((p_cal >= 0) & (p_cal <= 1)) and np.isfinite(p_raw).all()
    if model_id == "AO":
        assert np.all(p_raw == 1.0)
    if model_id == "B0":
        assert np.allclose(p_raw, ytr.mean())
    inner = cm.inner_predictions()
    assert (inner is None) == (not models.MODELS[model_id].calibrate)
    if inner is not None:
        assert inner[0].shape == (len(Xtr),)


def test_m3_uses_native_categoricals_and_pca_blocks(split):
    Xtr, ytr, _, _ = split
    pipe = models.make("M3", hyper={"n_star": 10}).fit(Xtr, ytr)
    Xt = pipe[:-1].transform(Xtr)
    cats = [c for c in Xt.columns if isinstance(Xt[c].dtype, pd.CategoricalDtype)]
    assert "cat__ctx_course" in cats
    assert pipe[-1].is_categorical_[list(Xt.columns).index("cat__ctx_course")]
    assert any(c.startswith("v60__pca_v60_") for c in Xt.columns)
    assert any(c.startswith("svd__svd_") for c in Xt.columns)
    assert not any("class_id" in c or c.endswith("t_s") for c in Xt.columns)
    assert pipe[-1].max_iter == 10 and not pipe[-1].early_stopping


def test_rule_r_formula():
    p = models.rule_r([1, 0, 1], [300, 300, 5], [4, 4, np.inf])
    assert np.allclose(p, [np.exp(-0.8), 0.5 * np.exp(-0.8), np.exp(-0.5)])


class _ClassMemo(ClassifierMixin, BaseEstimator):
    """p = 0.9 for classes seen in training, 0.1 otherwise (detects in-class leakage)."""

    def fit(self, X, y):
        self.seen_ = set(X["class_id"])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.where(X["class_id"].isin(self.seen_), 0.9, 0.1)
        return np.column_stack([1 - p, p])


def test_inner_oof_is_grouped_by_class(split):
    Xtr, ytr, _, _ = split
    s = models.inner_oof(_ClassMemo(), Xtr, ytr, n_splits=3, scores=False)
    assert np.allclose(s, 0.1)                       # no row scored by a model that saw it


def test_platt_recovers_parameters_and_handles_degenerate():
    rng = np.random.default_rng(0)
    s = rng.normal(size=20_000)
    y = (rng.random(len(s)) < 1 / (1 + np.exp(-(1.7 * s - 2.0)))).astype(float)
    pl = models.Platt.fit(s, y)
    assert abs(pl.a - 1.7) < 0.1 and abs(pl.b + 2.0) < 0.1
    const = models.Platt.fit(np.zeros(10), np.r_[np.ones(2), np.zeros(8)])
    assert const.a == 0 and 0 < const(0.0)[()] < 0.5


def test_tune_once_returns_grids_and_curves(split):
    Xtr, ytr, _, _ = split
    tuned = models.tune_once(Xtr, ytr, models=("M3", "M2", "B1"), n_inner=2,
                             hgb_max_iter=12)
    assert 1 <= tuned["M3"]["n_star"] <= 12
    assert len(tuned["M3"]["curves"]["val_mean"]) == 12
    assert tuned["M2"]["C"] in config.M2_C_GRID and tuned["B1"]["C"] in config.B1_C_GRID
    assert [r["C"] for r in tuned["M2"]["curves"]] == list(config.M2_C_GRID)
    assert models.hyper_for("M3+stack", tuned)["n_star"] == tuned["M3"]["n_star"]


def test_stacked_m3_adds_grouped_oof_columns(split):
    Xtr, ytr, Xte, _ = split
    st = models.make("M3+stack", hyper={"n_star": 8}).fit(Xtr, ytr)
    cols = st.stack_columns(Xte)
    assert set(cols) == set(config.STACK_COLUMNS)
    Xt = st.m3_[:-1].transform(Xte.assign(**cols))
    assert {"num__stack_m2", "num__stack_m4"} <= set(Xt.columns)


def test_knn_explainer_outputs_metadata_only():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(40, 6))
    meta = pd.DataFrame({"seg_type": ["debrief"] * 40, "minute": np.arange(40.0),
                         "bid_followed": np.arange(40) % 4 == 0})
    ex = models.KNNExplainer(k=3).fit(emb, meta)
    out = ex.explain(emb[:2])
    assert list(out.columns) == ["query", "rank", *config.KNN_OUTPUT_COLUMNS]
    assert len(out) == 6 and (out.loc[out["rank"] == 0, "distance"] < 1e-9).all()
    with pytest.raises(ValueError):
        models.KNNExplainer().fit(emb, meta.assign(txt_all60="p_words"))
    share = models.knn_bid_share(ex, emb[:4], emb[4:8])
    assert 0 <= share["share_tp"] <= 1 and share["n_fa"] == 4


# --------------------------------------------------------------------------- experiments


def test_registry_complete_and_tiered():
    conds = experiments.CONDITIONS
    groups = {c.group for c in conds.values()}
    assert set(experiments.SECTION7_GROUPS) <= groups
    assert set(experiments.AMENDMENT_CONDITIONS) <= set(conds)
    assert all(c.tier in experiments.TIERS and c.kind in experiments.KINDS
               for c in conds.values())
    k0 = {c.model for c in conds.values() if c.group == "K0"}
    assert k0 == {"AO", "B-min", "B0", "R", "B1", "M1", "M2", "M3"}
    assert all(c.tier == "mvp" for c in conds.values() if c.group in ("K0", "F0", "P-plant"))
    g = {c.g for c in conds.values() if c.group == "G"} | {conds["K0-M3"].g}
    assert g == set(config.G_SWEEP) and all(conds[f"G-g{x}"].tier == "must" for x in (1, 3, 5))
    for c in conds.values():
        assert c.target in config.LABEL_COLUMNS
        if c.kind == "model":
            assert c.model in models.MODELS and c.featureset in models.FEATURESETS
            assert c.split in experiments.SPLITS
        else:
            assert c.model is None
    assert conds["F-all+hist"].tier == conds["F-all+edu"].tier == "should"   # A2, A3
    assert conds["M3+stack"].model == "M3+stack" and conds["kNN"].kind == "explainer"
    assert {c.id for c in experiments.select(tier="mvp", kind="model")} >= {"K0-M3", "S-rand-M2"}
    assert len({c.id for c in conds.values()}) == len(conds)


def test_run_writes_schema_valid_oof_and_caches(toy, tmp_path):
    res = experiments.run("K0-M2", toy, out_dir=tmp_path, scorer=_null_scorer, **FAST)
    oof = schemas.validate(res.oof, "oof", strict=True)
    test, inner = oof[~oof["inner"]], oof[oof["inner"]]
    P = toy.points
    assert len(test) == experiments.predict_mask(P, config.HEADLINE_LABEL).sum()
    assert set(inner["fold"]) == set(test["fold"]) == set(P["fold"].dropna().astype(int))
    for k, g in test.groupby("fold"):                    # tested classes never trained on
        assert not set(g["class_id"]) & set(inner.loc[inner["fold"] == k, "class_id"])
    assert (tmp_path / "K0-M2.parquet").exists() and (tmp_path / "K0-M2_train.parquet").exists()
    r = schemas.validate(res.results, "results")
    pooled = r[r["fold"].isna()].iloc[0]
    assert pooled["n_classes"] == P["class_id"].nunique() and 0 < pooled["ap"] <= 1
    assert res.tuned and all("M2" in v for v in res.tuned.values())
    again = experiments.run("K0-M2", toy, out_dir=tmp_path, scorer=_null_scorer, **FAST)
    assert again.cached and again.hash == res.hash
    pd.testing.assert_frame_equal(again.oof, res.oof)


def test_scorer_receives_inner_rows_and_context(toy, tmp_path):
    seen = {}

    def scorer(oof, points, cond, **ctx):
        seen.update(inner=bool(oof["inner"].any()), ctx=set(ctx), H=cond.H)
        return {"recall_far01": 0.5, "tau_method": "test"}
    res = experiments.run("K0-R", toy, out_dir=tmp_path, scorer=scorer, cache=False, **FAST)
    assert seen["inner"] and {"segments", "train_scores"} <= seen["ctx"] and seen["H"] == 15
    pooled = res.results[res.results["fold"].isna()].iloc[0]
    assert pooled["recall_far01"] == 0.5 and res.meta["scorer_extra"] == {"tau_method": "test"}


def test_vault_never_reaches_dev_runs(tmp_path):
    data = experiments.toy_data(n_classes=12, minutes=8.0, seed=3, vault_classes=2)
    res = experiments.run("K0-R", data, out_dir=tmp_path, scorer=_null_scorer, **FAST)
    vault_ids = set(data.points.loc[data.points["in_vault"], "class_id"])
    assert not set(res.oof["class_id"]) & vault_ids
    with pytest.raises(PermissionError):
        experiments.run("V-R-F-q", data, out_dir=tmp_path, scorer=_null_scorer)
    v = experiments.run("V-R-F-q", data, out_dir=tmp_path, scorer=_null_scorer,
                        allow_vault=True, **FAST)
    assert set(v.oof.loc[~v.oof["inner"], "class_id"]) == vault_ids


def test_hist_columns_use_earlier_training_classes_only(toy):
    cls = toy.classes.sort_values("date")
    ids = list(cls["class_id"])
    train = ids[:6]
    h = experiments.hist_columns(toy, train)
    first = toy.points["class_id"].to_numpy() == ids[0]
    assert np.isnan(h.loc[first, "hist_bid_rate_all"]).all()     # nothing earlier
    later = toy.points["class_id"].to_numpy() == ids[-1]
    assert (h.loc[later, "hist_bid_rate_all"] > 0).all()
    # changing a test-fold class's bids leaves every hist value unchanged
    tweaked = experiments.Data(toy.points, toy.X, toy.classes.assign(
        n_bids=np.where(toy.classes["class_id"].isin(ids[6:]), 999, toy.classes["n_bids"])))
    pd.testing.assert_frame_equal(h, experiments.hist_columns(tweaked, train))


def test_plant_variant_and_missing_providers(toy, tmp_path):
    res = experiments.run("P-plant-M1", toy, out_dir=tmp_path, scorer=_null_scorer, **FAST)
    assert res.meta["plant"]["n_planted_pos"] > 0 and 0 <= res.meta["plant"]["planted_ap"] <= 1
    for cid in ("IH+30", "A-mismatch-0", "LS-1-M2", "M5-gru", "Sens-votepos"):
        with pytest.raises(experiments.MissingProvider):
            experiments.run(cid, toy, out_dir=tmp_path, scorer=_null_scorer, **FAST)


def test_knn_summary_is_metadata_only(toy, tmp_path):
    res = experiments.run("K0-R", toy, out_dir=tmp_path, scorer=_null_scorer, **FAST)
    summ = experiments.knn_summary(toy, res.oof, quantile=0.8)
    assert set(summ["fold"]) == set(res.oof["fold"])
    assert list(summ.columns) == ["fold", "share_tp", "share_fa", "n_tp", "n_fa"]
    assert summ[["share_tp", "share_fa"]].stack().between(0, 1).all()


def test_cli_list_and_dry_run(capsys):
    from argparse import Namespace

    assert experiments.cli_train(Namespace(extra=["--list"])) == 0
    out = capsys.readouterr().out
    assert "K0-M3" in out and "P-clock" in out
    assert experiments.cli_train(Namespace(extra=["--tier", "mvp", "--dry-run"])) == 0
    assert "CALLON" not in capsys.readouterr().out          # dry-run lists model rows only


# --------------------------------------------------------------------------- M6 helpers


def _tiny_xenc():
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
    )

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[", "]", "|", "<", ">", "p", "s",
             "cs", "debrief", "min", "since", "me", "never", "m", "anyone", "thoughts", "the",
             "model", "pcw", ":", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "other"]
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(words)})
    cfg = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32, num_labels=1,
                     max_position_embeddings=512)
    return tok, BertForSequenceClassification(cfg)


def test_xenc_text_and_truncation():
    assert xenc.tagged_from_prefixed("p_what p_now s_i s_think c_ok") == \
        "<P> what now <S> i think <C> ok"
    assert xenc.header("CS", "debrief", 33.7, 400) == "[CS] [debrief] [min 33] [since me 6m]"
    assert "[since me never]" in xenc.header("CS", "intro", 5, np.nan)
    tok, _ = _tiny_xenc()
    window = " ".join(["the"] * 300 + ["anyone"])
    ids = xenc.format_input(tok, "[CS]", window, "model model", window_tokens=224)
    assert len(ids) <= xenc.MAX_LEN
    body = tok.convert_ids_to_tokens(ids)
    assert "anyone" in body                               # left truncation keeps the latest
    assert body.count("the") <= 224


def test_xenc_subsample_and_prior_correction():
    y = np.r_[np.ones(20), np.zeros(1000)]
    keep = xenc.subsample_negatives(y, 0.2, seed=1)
    assert (y[keep] == 1).sum() == 20 and 150 < (y[keep] == 0).sum() < 250
    assert np.allclose(xenc.prior_correct([0.0], 0.2), [np.log(0.2)])
    X = pd.DataFrame({"ctx_area": "CS", "ctx_seg_type": "debrief", "ctx_min": 10.0,
                      "me_since_turn_s": np.nan,
                      "txt_all60": ["p_anyone p_thoughts"] * 6 + ["p_the p_model"] * 30})
    yy = np.r_[np.ones(6), np.zeros(30)]
    m = xenc.CrossEncoderModel(model_factory=_tiny_xenc, epochs=1,
                               batch_size=8, device="cpu").fit(X, yy)
    p = m.predict_proba(X)
    assert p.shape == (36, 2) and np.isfinite(p).all()
    raw = m.model_
    assert raw.config.num_labels == 1 and len(m.history_) == 1


# --------------------------------------------------------------------------- figures


def _results_rows():
    rows = []
    for cid, model, fs, split, H, g, ap in (
            ("K0-M3", "M3", "F-all", "S-group", 15, 2, 0.3),
            ("K0-M2", "M2", "F-all", "S-group", 15, 2, 0.25),
            ("S-rand-M2", "M2", "F-all", "S-rand", 15, 2, 0.4),
            ("S-rand-M3", "M3", "F-all", "S-rand", 15, 2, 0.45),
            ("F0-Fstruct", "M3", "F-struct", "S-group", 15, 2, 0.2),
            ("G-g1", "M3", "F-all", "S-group", 15, 1, 0.33),
            ("Hs-H30", "M3", "F-all", "S-group", 30, 2, 0.4),
            ("IH-10", "M2", "F-text", "S-group", 15, 2, 0.1),
            ("IH+10", "M2", "F-text", "S-group", 15, 2, 0.5),
            ("A-mismatch-0", "M3", "F-all+pcwA", "S-group", 15, 2, 0.2),
            ("A-mismatch-1", "M3", "F-all+pcwA", "S-group", 15, 2, 0.22),
            ("F1-Fall+pcwA", "M3", "F-all+pcwA", "S-group", 15, 2, 0.3),
            ("LC-M3-20", "M3", "F-all", "S-group", 15, 2, 0.2)):
        rows.append({"condition": cid, "tier": "mvp", "model": model, "featureset": fs,
                     "split": split, "target": config.y_col(H, g), "H": float(H),
                     "g": float(g), "fold": pd.NA, "ap": ap, "ap_within": ap,
                     "ap_shift": ap / 2, "ap_local": ap * 0.8, "base_rate": 0.02})
    return pd.DataFrame(rows)


def test_figures_render_from_aggregates(tmp_path, small_world):
    res = _results_rows()
    rng = np.random.default_rng(0)
    oof = pd.DataFrame({"class_id": np.repeat([1, 2, 3], 50), "t_s": np.tile(np.arange(50.), 3),
                        "fold": 0, "y": (rng.random(150) < 0.2).astype(float),
                        "p_raw": rng.random(150), "inner": False})
    oof["p_cal"] = oof["p_raw"]
    figs = {
        "split_tax": plots.fig_split_tax(res), "sweeps": plots.fig_sweeps(res),
        "ih": plots.fig_info_horizon(res), "mismatch": plots.fig_mismatch(res),
        "timing": plots.fig_timing_decomposition(res), "lc": plots.fig_learning_curve(res),
        "pr": plots.fig_pr_curves({"K0-M3": oof, "K0-M2": oof}),
        "apw": plots.fig_ap_within(oof), "rel": plots.fig_reliability(oof),
        "hgb": plots.fig_hgb_learning({0: {"train": [[0.3, 0.2, 0.1]], "val": [[0.3, 0.25,
                                                                                 0.27]]}},
                                      {0: 2}),
        "lr": plots.fig_lr_c_curve({0: [{"C": 0.1, "train_ap": 0.3, "val_ap": 0.2},
                                        {"C": 1.0, "train_ap": 0.4, "val_ap": 0.21}]}),
        "forest": plots.fig_forest(pd.DataFrame({"hypothesis": ["H1", "H2"],
                                                 "estimate": [0.02, 0.01], "lo": [0.0, -0.01],
                                                 "hi": [0.04, 0.03], "p_adj": [0.04, 0.3]})),
        "froc": plots.fig_froc({"M3": pd.DataFrame({"far": [0, 0.1, 0.5],
                                                    "recall": [0, 0.3, 0.6]}),
                                "shift": pd.DataFrame({"far": [0, 0.5], "recall": [0, 0.1]})},
                               ao_point=(4.3, 0.93), annotate={"far": 0.1, "recall": 0.3,
                                                                "lead_s": 8, "tiw": 0.05}),
        "callon": plots.fig_callon({"rate_bid": 0.1, "rate_matched": 0.12, "rate_ratio": 0.8,
                                    "ci_lo": 0.4, "ci_hi": 1.5, "pass": True}),
        "pclock": plots.fig_p_clock({"curve": pd.DataFrame({"delta_s": [-1.0, 0.0, 1.0],
                                                            "ap": [0.3, 0.6, 0.2]}),
                                     "delta_star": 0.0, "sharpness": 0.3, "pass": True}),
        "names": plots.fig_name_rates(pd.DataFrame({"window": ["pre_bid", "post_bid",
                                                               "matched"],
                                                    "rate_per_min": [1.0, 2.0, 0.9],
                                                    "ci_lo": [0.5, 1.5, 0.6],
                                                    "ci_hi": [1.5, 2.5, 1.2]})),
        "knn": plots.fig_knn(pd.DataFrame({"share_tp": [0.6], "share_fa": [0.3]})),
        "funnel": plots.fig_funnel(pd.DataFrame({"stage": ["synced", "QA"],
                                                 "classes": [694, 140]})),
        "hero": plots.fig_schematic_timeline(small_world.classes[0], span=(120.0, 600.0)),
    }
    for name, fig in figs.items():
        paths = plots.save(fig, name, tmp_path, formats=("png",))
        assert paths[0].stat().st_size > 1000, name
    assert set(plots.FIGURE_FUNCTIONS) <= set(dir(plots))
    table = plots.comparison_table(res)
    assert "ap_within" in table and len(table) == len(res)


def test_figures_refuse_text_inputs():
    with pytest.raises(ValueError):
        plots.assert_aggregate(pd.DataFrame({"txt_all60": ["p_words"], "ap": [0.1]}))
    with pytest.raises(ValueError):
        plots.fig_funnel(pd.DataFrame({"stage": ["a"], "classes": [1], "text_masked": ["x"]}))


def test_cli_evaluate_with_metrics_scorer_and_figures(toy, tmp_path, monkeypatch, capsys):
    """End to end on local paths: train one condition, re-score it with `oncue evaluate`
    (default scorer = metrics.condition_metrics), then draw figures from results.parquet."""
    from argparse import Namespace

    pytest.importorskip("oncue.metrics")
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(config, "OOF_DIR", tmp_path / "oof")
    (tmp_path / "tables").mkdir()
    toy.points.to_parquet(config.table_path("points"), index=False)
    experiments.run("K0-R", toy, out_dir=config.OOF_DIR, scorer=_null_scorer, **FAST)
    assert experiments.cli_evaluate(Namespace(extra=["--cond", "K0-R"])) == 0
    res = pd.read_parquet(config.table_path("results"))
    pooled = res[res["fold"].isna()].iloc[0]
    assert pooled["condition"] == "K0-R"
    assert np.isfinite(pooled["ap_within_lo"])           # TG: NaN, toy classes < 400 points
    assert np.isfinite(pooled["recall_far01"])          # rate-matched via inner + train rows
    assert plots.cli(Namespace(extra=["--out", str(tmp_path / "fig")])) == 0
    assert (tmp_path / "fig" / "comparison_table.csv").exists()
    assert "wrote" in capsys.readouterr().out


def test_build_windows_role_prefixed_and_causal(world, truth):
    pytest.importorskip("oncue.text")
    sc = world.classes[0]
    T = truth[sc.class_id]
    t = np.array([300.0, 600.0, 900.0])
    pts = pd.DataFrame({"class_id": sc.class_id, "t_s": t})
    win = experiments.build_windows(pts, T["words"], T["chat"], T["classes"],
                                    variants=("norm", "name"))
    assert {"txt_all60", "txt_all15", "txt_chat60", "txt_all60_name"} <= set(win.columns)
    toks = " ".join(win["txt_all60"]).split()
    assert toks and all(tok[:2] in ("p_", "s_", "m_", "u_") for tok in toks)
    assert "[name]" not in " ".join(win["txt_all60"]).lower()
    for i, ti in enumerate(t):                 # words after t change nothing at t
        w = T["words"][T["words"]["start_s"] <= ti]
        c = T["chat"][T["chat"]["t_s"] <= ti]
        one = experiments.build_windows(pts.iloc[[i]], w, c, T["classes"])
        assert one["txt_all60"].iloc[0] == win["txt_all60"].iloc[i]
        assert one["txt_chat60"].iloc[0] == win["txt_chat60"].iloc[i]


# --------------------------------------------------------------------------- review fixes


def test_population_conditions_never_tune_on_their_test_classes(tmp_path, monkeypatch):
    """§8.1: one fold assignment over all dev classes. With points.fold unset (the state
    labels.py used to leave), a PCW-subset condition (H3) used to draw its folds on the
    subset while n* was tuned on folds drawn over all dev classes: most test classes were
    in their own fold's tuning data."""
    data = experiments.toy_data(n_classes=15, minutes=8.0, seed=11)
    data.points["fold"] = pd.array([pd.NA] * len(data.points), dtype="Int64")
    seen_tune, seen_test = {}, []
    real_tuned, real_fit = experiments.get_tuned, experiments._fit_fold_sklearn

    def spy_tuned(model_id, split, fold, X, y, **kw):
        seen_tune[fold] = set(X["class_id"])
        return real_tuned(model_id, split, fold, X, y, **kw)

    def spy_fit(cond, model_id, X, y_all, tr, te, tg, tuned, n_inner, estimator=None):
        seen_test.append(set(X["class_id"].to_numpy()[te]))
        return real_fit(cond, model_id, X, y_all, tr, te, tg, tuned, n_inner, estimator)
    monkeypatch.setattr(experiments, "get_tuned", spy_tuned)
    monkeypatch.setattr(experiments, "_fit_fold_sklearn", spy_fit)
    experiments.run("F1-Fall+pcwA", data, out_dir=tmp_path, scorer=_null_scorer, cache=False,
                    **FAST)
    assert seen_test and len(seen_tune) == len(seen_test)
    for k, test in enumerate(seen_test):
        assert not test & seen_tune[k], k
    # the guard itself
    dev = data
    folds = experiments.outer_folds(dev, "S-group")
    other = {k: tr for k, tr, _ in experiments.outer_folds(
        dev, "S-group", seed=config.SEED + 1)}
    with pytest.raises(AssertionError, match="tuning"):
        experiments.assert_tuning_disjoint(other, dev, folds, dev, "S-group")


def test_subsets_inherit_dev_folds_and_twins_share_groups():
    data = experiments.toy_data(n_classes=12, minutes=8.0, seed=5)
    data.points["fold"] = pd.array([pd.NA] * len(data.points), dtype="Int64")
    dev_lab = experiments.fold_labels(data, "S-group")
    sub = data.take(data.points["class_id"].isin(sorted(set(data.points["class_id"]))[:5]))
    got = experiments.inherit_folds(data, dev_lab, sub, "S-group")
    want = sub.points["class_id"].map(
        pd.Series(dev_lab, index=data.points["class_id"]).groupby(level=0).first())
    assert (got == want.to_numpy()).all()
    # L18: same-section same-date twins share a CV group in outer and inner folds
    cls = data.classes.assign(cv_group=data.classes["class_id"])
    a, b = cls["class_id"].iloc[0], cls["class_id"].iloc[1]
    cls.loc[cls["class_id"] == b, "cv_group"] = a
    twin = experiments.Data(points=data.points, X=data.X, classes=cls)
    lab = experiments.fold_labels(twin, "S-group")
    per = pd.Series(lab, index=twin.points["class_id"]).groupby(level=0).first()
    assert per.loc[a] == per.loc[b]
    X = experiments.with_cv_groups(twin).X
    assert "cv_group" in X and (X.loc[twin.points["class_id"] == b, "cv_group"] == a).all()
    assert (models.groups_of(X) == X["cv_group"].to_numpy()).all()
    assert "cv_group" not in models.select_columns(X, "F-all", "num")


def test_inner_rows_cover_the_whole_training_grid(toy, tmp_path):
    """§8.3 step 1 reads FAR on the inner OOF rows: they must cover the training fold's
    whole causal grid G (guard / vote / pending points included), not only defined rows."""
    res = experiments.run("K0-M2", toy, out_dir=tmp_path, scorer=_null_scorer, cache=False,
                          **FAST)
    P = toy.points
    grid = experiments.grid_mask(P, config.HEADLINE_LABEL)
    for k in sorted(res.oof["fold"].unique()):
        inner = res.oof[(res.oof["fold"] == k) & res.oof["inner"]]
        train_grid = P[grid & (P["fold"] != k).to_numpy()]
        got = set(zip(inner["class_id"], inner["t_s"]))
        assert set(zip(train_grid["class_id"], train_grid["t_s"])) <= got
        assert inner["y"].isna().any() and inner["p_cal"].notna().all()


def test_saturating_naive_bayes_keeps_its_ranking_after_platt():
    """Platt reads unclipped log-odds: MultinomialNB on ~150-token windows puts almost
    every predict_proba below 1e-7, which used to collapse p_cal onto a handful of values."""
    rng = np.random.default_rng(1)
    vocab = np.array([f"p_w{i}" for i in range(300)] + [f"s_w{i}" for i in range(300)])
    n_cls, per = 16, 250
    y = (rng.random(n_cls * per) < 0.05).astype(float)
    probs = np.ones((2, vocab.size))
    probs[1, :20] += 2.0
    toks = np.empty((len(y), 150), dtype=object)
    for v in (0, 1):
        m = y == v
        toks[m] = rng.choice(vocab, size=(int(m.sum()), 150), p=probs[v] / probs[v].sum())
    X = pd.DataFrame({"class_id": 1000 + np.repeat(np.arange(n_cls), per),
                      "t_s": np.tile(120.0 + 2 * np.arange(per), n_cls),
                      "txt_all60": [" ".join(r) for r in toks],
                      "txt_all15": [" ".join(r[-40:]) for r in toks], "txt_chat60": "",
                      "ctx_min": np.tile(2 + np.arange(per) / 30, n_cls), "q_open": 0.0,
                      "me_since_turn_s": np.nan, "ctx_seg_type": "debrief"})
    tr = (X["class_id"] < 1010).to_numpy()
    cm = models.CalibratedModel(models.make("M1", "F-all", hyper={"alpha": 0.3}))
    cm.fit(X[tr].reset_index(drop=True), y[tr])
    p_raw, p_cal = cm.predict(X[~tr].reset_index(drop=True))
    clipped = np.clip(p_raw, 1e-7, 1 - 1e-7)                          # the old Platt input
    assert np.mean(clipped == 1e-7) > 0.25                            # it does saturate
    assert len(np.unique(p_cal)) > 0.9 * (~tr).sum()
    order_raw = np.argsort(np.argsort(models.decision_scores(cm.model_, X[~tr])))
    order_cal = np.argsort(np.argsort(p_cal))
    assert (order_raw == order_cal).all() and cm.platt_.a > 0
    from oncue import metrics as M
    assert M.ap(y[~tr], p_cal) == pytest.approx(M.ap(y[~tr], models.decision_scores(
        cm.model_, X[~tr])))
    lr = models.make("M2", "F-all")
    lr.fit(X[tr], y[tr])
    np.testing.assert_allclose(models._sigmoid(models.decision_scores(lr, X[~tr])),
                               models.predict_p(lr, X[~tr]), rtol=1e-9, atol=1e-12)


def test_results_use_p_cal_and_tie_safe_ece():
    """Pooled ROC-AUC and every fold row's AP_w on p_cal; ECE keeps tied scores in one bin,
    so a constant, well-calibrated predictor scores ~0 even when class rates differ."""
    rng = np.random.default_rng(0)
    rows = []
    for c in range(30):
        pi = rng.choice([0.0, 0.01, 0.05])
        y = (rng.random(400) < pi).astype(float)
        rows.append(pd.DataFrame({"class_id": c, "t_s": 120 + 2.0 * np.arange(400), "y": y}))
    df = pd.concat(rows, ignore_index=True)
    const = df.assign(p_raw=df["y"].mean(), p_cal=df["y"].mean(), fold=0)
    row = experiments._metric_row(const)
    assert row["ece"] == pytest.approx(0.0, abs=1e-9)
    noisy = df.assign(p_raw=rng.random(len(df)), fold=0)
    noisy["p_cal"] = 1 - noisy["p_raw"] + 0.3 * df["y"]               # a different ranking
    from oncue import metrics as M
    row = experiments._metric_row(noisy)
    assert row["roc_auc"] == pytest.approx(M.roc_auc(noisy["y"], noisy["p_cal"]))
    assert row["ap_within"] == pytest.approx(experiments.ap_within(noisy, "p_cal")[0])
    assert not hasattr(experiments, "ece")


def test_tspeech_grid_has_no_hand_rules(tmp_path):
    """T-speech is scored on its own causal grid: hand_up / post_lowering points can fire."""
    data = experiments.toy_data(n_classes=10, minutes=8.0, seed=2)
    P = data.points
    target = experiments.get("T-speech").target
    grid = experiments.grid_mask(P, target)
    hand_up = (P["excl_reason"] == "hand_up").to_numpy()
    assert hand_up.any() and grid[hand_up].all() and not P["in_G"][hand_up].any()
    assert experiments.predict_mask(P, target)[grid].all()
    res = experiments.run("T-speech", data, out_dir=tmp_path, scorer=_null_scorer,
                          cache=False, **FAST)
    test = res.oof[~res.oof["inner"]]
    merged = test.merge(P[["class_id", "t_s", "excl_reason"]], on=["class_id", "t_s"])
    assert merged.loc[merged["excl_reason"] == "hand_up", "in_G"].all()


def test_g1_rows_are_flagged_approximate_when_retiming_failed(tmp_path):
    """A1: g = 1 needs re-timed onsets; with >= 10% of bid onsets not re-timed the G-g1
    rows say so (`g1_approximate`, `retime_failed_share`); other conditions carry <NA>."""
    data = experiments.toy_data(n_classes=10, minutes=8.0, seed=4)
    onsets = data.points[["class_id", "next_bid_s"]].dropna().drop_duplicates()
    ep = pd.DataFrame({"class_id": onsets["class_id"].to_numpy(),
                       "onset_s": onsets["next_bid_s"].to_numpy(), "ep_type": config.EP_BID,
                       "left_censored": False})
    ep["retime_failed"] = pd.array(np.arange(len(ep)) % 4 == 0, dtype="boolean")
    data.tables["hand_episodes"] = ep
    g1 = experiments.run("G-g1", data, out_dir=tmp_path, scorer=_null_scorer, cache=False,
                         **FAST).results
    share = float(ep["retime_failed"].astype(bool).mean())
    assert np.allclose(g1["retime_failed_share"].to_numpy(float), share)
    assert g1["g1_approximate"].astype(bool).all() and share >= config.G1_RETIME_FAILED_MAX
    head = experiments.run("K0-R", data, out_dir=tmp_path, scorer=_null_scorer, cache=False,
                           **FAST).results
    assert head["g1_approximate"].isna().all() and head["retime_failed_share"].isna().all()
    schemas.validate(g1, "results", strict=True)


def test_ev_compares_the_model_with_references_at_its_realised_far(tmp_path):
    """EV (S3): M3's recall at each budget with a class-bootstrap CI, R / B-min / shifted
    predictions read at M3's realised FAR on their envelopes, paired deltas, the AO point
    and the S3 verdict, from the cached OOF files."""
    data = experiments.toy_data(n_classes=12, minutes=12.0, seed=9)
    for c in ("K0-M3", "K0-R", "K0-Bmin"):
        experiments.run(c, data, out_dir=tmp_path, scorer=_null_scorer, **FAST)
    ev = experiments.run_ev(data.points, out_dir=tmp_path, n_boot=200, n_rep=3)
    schemas.validate(ev, "ev")
    assert (tmp_path / "EV.parquet").exists()
    refs = set(ev["reference"])
    assert {"model", "R", "B-min", "shift", "random", "AO"} <= refs
    m = ev[ev["reference"] == "model"].set_index("budget")
    assert (m["recall_lo"] <= m["recall"]).all() and (m["recall"] <= m["recall_hi"]).all()
    assert m["s3_pass"].notna().all()
    r = ev[ev["reference"] == "R"].set_index("budget")
    assert (r["far"] == m["far"]).all()                           # matched FAR
    assert np.allclose(r["delta"], m["recall"] - r["recall"])
    assert (r["delta_lo"] <= r["delta"]).all() and (r["delta"] <= r["delta_hi"]).all()
    ao = ev[ev["reference"] == "AO"].iloc[0]
    assert ao["far"] > 1.0 and 0.5 < ao["recall"] <= 1.0
    with pytest.raises(experiments.MissingProvider):
        experiments.run_ev(data.points, out_dir=tmp_path / "empty")
