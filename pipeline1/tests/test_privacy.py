"""Tests for oncue.privacy: the pre-push scan and the committed results bundle.

Invented people only (the synthetic roster: Ada Lovelace, Grace Hopper, ...). The ID salt is
the fixed synthetic test salt; `.env` is never read (load_env is patched to fail).
"""

from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import pandas as pd
import pytest
from oncue import __main__ as cli
from oncue import config, privacy, schemas, synthetic

ROSTER = pd.DataFrame(
    [{"first": f, "last": la} for f, la in synthetic.STUDENT_POOL]
    + [{"first": synthetic.ME_FIRST, "last": synthetic.ME_LAST}])


@pytest.fixture
def no_env(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("tests must never load .env")
    monkeypatch.setattr(config, "load_env", boom)


def _kinds(hits):
    return sorted(h.kind for h in hits)


# --------------------------------------------------------------------------- scan: text


def test_scan_finds_names_case_insensitively(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Results\nThanks to Ada for the idea.\n\nlovelace said so\n"
                      "Brady Rutherford wrote this. Brady did the audits.\n", encoding="utf-8")
    hits = privacy.scan([readme], ROSTER)
    assert {(h.line, h.term) for h in hits} == {(2, "Ada"), (4, "lovelace")}
    assert all(h.kind == "name" for h in hits)


def test_common_word_names_match_only_capitalised(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("we will see; a grace period; mark the spot.\n", encoding="utf-8")
    assert privacy.scan([f], ROSTER) == []
    f.write_text("Will raised a point. grace hopper agreed.\n", encoding="utf-8")
    terms = sorted(h.term for h in privacy.scan([f], ROSTER))
    assert terms == ["Will", "grace hopper"]        # full names: any case


def test_names_inside_words_and_initials_do_not_match(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("adaptive Eulerian Teslas A. B.\n", encoding="utf-8")
    assert privacy.scan([f], ROSTER) == []


def test_hashed_ids_and_salt(tmp_path, id_salt):
    ids = [synthetic.spk_hash(u) for u in (1, 2, 3)]
    f = tmp_path / "table.csv"
    f.write_text(f"spk,n\n{ids[0]},3\nx{ids[1]}y,4\n", encoding="utf-8")
    hits = privacy.scan([f], None, ids)
    assert [(h.kind, h.line) for h in hits] == [("hashed_id", 2)]
    g = tmp_path / "leak.json"
    g.write_text(json.dumps({"key": id_salt}), encoding="utf-8")
    assert _kinds(privacy.scan([g])) == ["salt"]
    assert privacy.scan([g], salt="another-long-salt") == []


def test_hits_are_redacted_when_printed(tmp_path):
    f = tmp_path / "r.md"
    f.write_text("Florence Nightingale\n", encoding="utf-8")
    (hit,) = privacy.scan([f], ROSTER)
    assert hit.term == "Florence Nightingale"
    assert "Nightingale" not in str(hit) and "Florence" not in hit.redacted
    assert str(hit).startswith(f"{f}:1: name F")


# --------------------------------------------------------------------------- scan: formats


def test_notebook_outputs_scanned_images_skipped(tmp_path):
    fake_png = base64.b64encode(b"\x00\x01").decode() + "/Ada+Turing/"
    nb = {"metadata": {}, "nbformat": 4, "nbformat_minor": 5, "cells": [
        {"cell_type": "code", "source": ["print(1)\n"], "metadata": {},
         "outputs": [{"output_type": "stream", "name": "stdout", "text": ["ok\n"]}]},
        {"cell_type": "code", "source": "x", "metadata": {}, "outputs": [
            {"output_type": "display_data", "metadata": {},
             "data": {"image/png": fake_png, "text/plain": ["<Figure Euler>"]}}]},
    ]}
    path = tmp_path / "pipeline1.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    hits = privacy.scan([path], ROSTER)
    assert [(h.line, h.term) for h in hits] == [(2, "Euler")]


def test_data_uris_are_not_grepped(tmp_path):
    f = tmp_path / "README.md"
    f.write_text("![x](data:image/png;base64,QUJD/Ada+Turing/==)\n", encoding="utf-8")
    assert privacy.scan([f], ROSTER) == []


def test_svg_png_metadata_and_binaries(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = tmp_path / "figures"
    figs.mkdir()
    (figs / "a.svg").write_text("<svg><text>Noether</text></svg>", encoding="utf-8")
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    fig.savefig(figs / "clean.png")
    fig.savefig(figs / "tagged.png", metadata={"Title": "Mary Somerville"})
    plt.close(fig)
    (figs / "b.pdf").write_bytes(b"%PDF-1.4 binary")
    rep = privacy.scan_report([figs, tmp_path / "missing_dir"], ROSTER)
    assert sorted((h.path.split("/")[-1], h.term) for h in rep.hits) == [
        ("a.svg", "Noether"), ("tagged.png", "Mary Somerville")]
    assert [p.split("/")[-1] for p in rep.skipped] == ["b.pdf"]
    assert rep.missing == [str(tmp_path / "missing_dir")]


def test_parquet_values_columns_and_class_ids(tmp_path, id_salt):
    good = pd.DataFrame({"class_id": [config.hash_id(5), config.hash_id(6)],
                         "t_s": [1.0, 2.0], "model": ["M3", "M2"]})
    good.to_parquet(tmp_path / "oof_ok.parquet", index=False)
    assert privacy.scan([tmp_path / "oof_ok.parquet"], ROSTER) == []
    bad = pd.DataFrame({"class_id": [5, 6], "spk_hash": ["u_1", "u_2"],
                        "note": ["fine", "asked Alan"]})
    bad.to_parquet(tmp_path / "oof_bad.parquet", index=False)
    hits = privacy.scan([tmp_path / "oof_bad.parquet"], ROSTER)
    assert _kinds(hits) == ["column", "name", "raw_class_id", "raw_class_id"]
    assert [h.line for h in hits if h.kind == "name"] == [2]


# --------------------------------------------------------------------------- loaders


def test_load_roster_and_hashed_ids(data_dir, world, tmp_path):
    names = privacy.load_roster(data_dir["raw"])
    students = pd.concat([sc.roster for sc in world])
    others = students[students["role"] != "me"]
    assert set(others["first"]) <= set(names) and set(others["last"]) <= set(names)
    assert synthetic.ME_FIRST not in names and synthetic.ME_LAST not in names
    sc = world.classes[0]
    full = sc.roster[sc.roster["role"] != "me"].iloc[0]
    assert f"{full['first']} {full['last']}" in names
    words = synthetic.truth_tables(sc)["words"]
    words.to_parquet(tmp_path / "words.parquet", index=False)
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "other.parquet", index=False)
    assert privacy.load_hashed_ids(tmp_path) == sorted(words["spk_hash"].unique())


# --------------------------------------------------------------------------- results bundle


def _oof(world, cond_seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(cond_seed)
    rows = []
    for k, cid in enumerate(world.class_ids):
        n = 50
        rows.append(pd.DataFrame({
            "class_id": cid, "t_s": 120 + 2.0 * np.arange(n), "fold": k % 5,
            "y": rng.integers(0, 2, n).astype("float32"), "p_raw": rng.random(n),
            "p_cal": rng.random(n), "inner": np.arange(n) >= 40, "in_G": True,
            "text_norm": "words that must never be committed"}))
    return pd.concat(rows, ignore_index=True)


def _results_row() -> pd.DataFrame:
    row = {c.name: np.nan for c in schemas.schema("results")}
    row.update({"condition": "K0", "tier": "MVP", "model": "M3", "featureset": "F-all",
                "split": "S-group", "target": "hand", "H": 15.0, "g": 2.0})
    df = pd.DataFrame([row])
    for c in ("fold", "n_classes", "n_onsets", "n_points"):
        df[c] = pd.array([pd.NA], dtype="Int64")
    return df


def test_results_bundle_writes_only_hashed_ids(world, truth, tmp_path, id_salt):
    eps = pd.concat([t["hand_episodes"] for t in truth.values()], ignore_index=True)
    out = tmp_path / "results"
    written = privacy.results_bundle(
        out, oof={"K0/M3 F-all": _oof(world)}, results=_results_row(), bids=eps,
        funnel=pd.DataFrame({"stage": ["selected", "qa"], "n_classes": [6, 5]}))
    assert sorted(written) == ["bids.parquet", "funnel.csv", "oof_K0_M3_F-all.parquet",
                               "results.parquet"]
    oof = pd.read_parquet(written["oof_K0_M3_F-all.parquet"])
    assert list(oof.columns) == schemas.required("oof") + ["in_G"]  # text never written
    assert not oof["inner"].any() and len(oof) == 40 * len(world)
    want = {config.hash_id(c, id_salt) for c in world.class_ids}
    assert set(oof["class_id"]) == want
    assert oof["class_id"].str.fullmatch(r"[0-9a-f]{12}").all()
    bids = pd.read_parquet(written["bids.parquet"])
    assert set(bids["class_hash"]) <= want and "class_id" not in bids
    assert set(bids["ep_type"]) <= {"bid", "vote"}
    raw_ids = {str(c) for c in world.class_ids}
    for path in written.values():
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            vals = set(df.astype(str).to_numpy().ravel())
            assert not (vals & raw_ids)
    names = privacy.roster_terms(pd.concat([sc.roster for sc in world]))
    ids = {synthetic.spk_hash(u) for sc in world for u in sc.roster["user_id"]}
    assert privacy.scan([out], names, ids) == []


def test_results_bundle_refuses_unsafe_input(world, tmp_path, id_salt):
    with pytest.raises(schemas.SchemaError):
        privacy.results_bundle(tmp_path / "a", results=_results_row().assign(spk_hash="u_1"))
    with pytest.raises(privacy.PrivacyError):
        privacy.results_bundle(tmp_path / "b",
                               funnel=pd.DataFrame({"class_id": [1], "n_bids": [3]}))
    with pytest.raises(privacy.PrivacyError):
        privacy.results_bundle(tmp_path / "c", oof={"K0": _oof(world)}, max_bytes=100)


def test_results_bundle_needs_the_salt(world, tmp_path):
    with pytest.raises(RuntimeError):
        privacy.results_bundle(tmp_path, oof={"K0": _oof(world)})
    assert not list(tmp_path.glob("*.parquet"))


def test_hash_class_ids():
    s = privacy.hash_class_ids([5, 5.0, 7], "k" * 10)
    assert s.iloc[0] == s.iloc[1] == config.hash_id(5, "k" * 10)
    assert privacy.hash_class_ids(s, "other-salt").equals(s)       # already hashed
    with pytest.raises(ValueError):
        privacy.hash_class_ids([1.5], "k" * 10)


# --------------------------------------------------------------------------- CLI


def test_cli_scan(data_dir, world, tmp_path, no_env, capsys):
    sc = world.classes[0]
    person = sc.roster[sc.roster["role"] == "student"].iloc[0]
    readme = tmp_path / "README.md"
    readme.write_text(f"Discussion with {person['first']} {person['last']}.\n",
                      encoding="utf-8")
    extra = ["--path", str(readme), "--raw-dir", str(data_dir["raw"]),
             "--tables-dir", str(tmp_path / "no_tables")]
    assert cli.main(["privacy", *extra]) == 1
    out = capsys.readouterr().out
    assert person["last"] not in out and "1 hits" in out
    readme.write_text("Aggregate results only.\n", encoding="utf-8")
    assert privacy.cli_scan(argparse.Namespace(extra=extra)) == 0


def test_cli_results_bundle(world, truth, tmp_path, id_salt, no_env):
    oof_dir, tables, out = tmp_path / "oof", tmp_path / "tables", tmp_path / "results"
    oof_dir.mkdir()
    tables.mkdir()
    _oof(world).drop(columns=["text_norm"]).to_parquet(oof_dir / "K0.parquet", index=False)
    eps = pd.concat([t["hand_episodes"] for t in truth.values()], ignore_index=True)
    eps.to_parquet(tables / "hand_episodes.parquet", index=False)
    rc = cli.main(["results-bundle", "--out", str(out), "--oof-dir", str(oof_dir),
                   "--tables-dir", str(tables)])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["bids.parquet", "oof_K0.parquet"]
