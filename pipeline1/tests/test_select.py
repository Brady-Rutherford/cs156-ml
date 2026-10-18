"""select.py: classes.parquet from manifest + raw, full-schedule session numbers, section
choice (§10), phases, the vault rule and dedupe. Synthetic data only."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest
from oncue import config, schemas, select, synthetic
from scribe_core.privacy import pseudonym

COMPARE = ["section_id", "course_code", "course_area", "term", "semester_idx", "session_no",
           "date", "weekday", "hour_utc", "rec_offset_s", "rec_segments_json", "n_rec_sessions",
           "T_end_s", "enrolled", "n_my_voice_main", "in_vault", "src_res", "timing_precision",
           "asr_provider"]


def vault_ids(n, start=1):
    return [c for c in range(start, start + 400) if config.in_vault(c)][:n]


def dev_ids(n, start=1):
    return [c for c in range(start, start + 400) if not config.in_vault(c)][:n]


@pytest.fixture(scope="module")
def built(data_dir):
    with pytest.MonkeyPatch.context() as mp:            # the synthetic truth has no local timezone
        mp.setattr(config, "TERM_TZ", {})
        return select.build_classes(data_dir["manifest"], data_dir["raw"],
                                    schedule_dir=data_dir["scribe"] / "schedule",
                                    me_path=data_dir["scribe"] / "me.json",
                                    hasher=select.spk_hasher("synthetic-test-salt"), log_path=False)


# --------------------------------------------------------------------------- build_classes


def test_build_classes_matches_the_synthetic_truth(built, truth, world):
    schemas.validate(built, "classes")
    assert list(built["class_id"]) == world.class_ids
    want = pd.concat([truth[c]["classes"] for c in world.class_ids], ignore_index=True)
    got = built.set_index("class_id")
    want = want.set_index("class_id")
    for col in COMPARE:
        for cid in world.class_ids:
            g, w = got.loc[cid, col], want.loc[cid, col]
            if col == "rec_segments_json":
                g, w = json.loads(g), json.loads(w)
            if isinstance(w, float):
                assert g == pytest.approx(w, abs=1e-3), (cid, col)
            else:
                assert g == w, (cid, col, g, w)
    for cid in world.class_ids:
        assert pd.Timestamp(got.loc[cid, "sched_start_utc"]) == \
            pd.Timestamp(want.loc[cid, "sched_start_utc"])
    assert got["phase"].isna().all() and got["qa_pass"].isna().all()


def test_prof_hash_is_the_salted_pseudonym(built, world):
    for sc in world:
        row = built.set_index("class_id").loc[sc.class_id]
        assert row["prof_hash"] == pseudonym(sc.meta["prof_id"], "synthetic-test-salt")
        assert sc.roster.iloc[0]["last"] not in row["prof_hash"]


def test_without_a_salt_prof_hash_is_empty_not_an_error(data_dir):
    df = select.build_classes(data_dir["manifest"], data_dir["raw"],
                              schedule_dir=data_dir["scribe"] / "schedule",
                              me_path=data_dir["scribe"] / "me.json", log_path=False)
    assert df["prof_hash"].isna().all()


def test_salt_from_environment(data_dir, id_salt):
    df = select.build_classes(data_dir["manifest"], data_dir["raw"],
                              schedule_dir=data_dir["scribe"] / "schedule",
                              me_path=data_dir["scribe"] / "me.json", log_path=False)
    assert df["prof_hash"].str.startswith("u_").all()


def test_session_numbers_count_unrecorded_classes():
    sched = {"results": [
        {"id": 30, "date": "2025-01-27", "start-time": "2025-01-27T16:00:00Z"},
        {"id": 10, "date": "2025-01-13", "start-time": "2025-01-13T16:00:00Z"},
        {"id": 99, "date": "2025-01-20"},                       # unrecorded, no start-time
        {"id": 20, "date": "2025-01-27", "start-time": "2025-01-27T09:00:00Z"},
        {"title": "no id"},
    ]}
    assert select.session_no_full_schedule(sched) == {10: 1, 99: 2, 20: 3, 30: 4}
    assert select.session_no_full_schedule(sched["results"]) == {10: 1, 99: 2, 20: 3, 30: 4}
    assert select.session_no_full_schedule(None) == {}


def test_session_no_falls_back_to_manifest_dates(data_dir, tmp_path, world):
    df = select.build_classes(data_dir["manifest"], data_dir["raw"],
                              schedule_dir=tmp_path / "missing",
                              me_path=data_dir["scribe"] / "me.json", log_path=False)
    for sc in world:
        assert df.set_index("class_id").loc[sc.class_id, "session_no"] == sc.meta["session_no"]


def test_hour_local_uses_the_term_timezone(data_dir, monkeypatch, world):
    monkeypatch.setattr(config, "TERM_TZ", {"F26": "Asia/Tokyo"})     # only F26 has one here
    df = select.build_classes(data_dir["manifest"], data_dir["raw"],
                              schedule_dir=data_dir["scribe"] / "schedule",
                              me_path=data_dir["scribe"] / "me.json", log_path=False)
    f26 = df[df["term"] == "F26"]
    assert ((f26["hour_utc"] + 9) % 24 == f26["hour_local"]).all()
    assert df.loc[df["term"] != "F26", "hour_local"].isna().all()


def test_missing_me_and_empty_manifest(data_dir, tmp_path):
    df = select.build_classes(data_dir["manifest"], data_dir["raw"],
                              schedule_dir=data_dir["scribe"] / "schedule",
                              me_path=tmp_path / "none.json", log_path=False)
    assert df["n_my_voice_main"].isna().all()
    empty = select.build_classes(pd.DataFrame(columns=["class_id", "section_id"]),
                                 data_dir["raw"], log_path=False)
    assert empty.empty and list(empty.columns) == schemas.columns("classes")


# --------------------------------------------------------------------------- dedupe


def test_dedupe_drops_repeats_and_flags_same_day(built):
    dup = pd.concat([built, built.iloc[[0]]], ignore_index=True)
    twin = built.iloc[[0]].copy()
    twin["class_id"] = 123
    out = select.dedupe(pd.concat([dup, twin], ignore_index=True))
    assert out["class_id"].is_unique and len(out) == len(built) + 1
    assert out.attrs["same_section_date"] == [tuple(sorted((123, int(built.iloc[0].class_id))))]
    assert select.dedupe(built).attrs["same_section_date"] == []
    # L18: the flag is persisted as a shared CV group, so the twins never split across folds
    first = int(built.iloc[0].class_id)
    group = out.set_index("class_id")["cv_group"]
    assert group.loc[123] == group.loc[first] == min(123, first)
    others = group.drop([123, first])
    assert (others.index == others.to_numpy()).all()
    schemas.validate(out, "classes")
    from oncue import splits
    many = pd.DataFrame({"class_id": range(20), "course_code": "CS1", "in_vault": False,
                         "cv_group": [0, 0] + list(range(2, 20))})
    pts = pd.DataFrame({"class_id": range(20), "t_s": 0.0})
    for seed in range(5):
        fold = splits.fold_column(pts, many, seed=seed)
        assert fold.iloc[0] == fold.iloc[1]


# --------------------------------------------------------------------------- vault


def test_vault_flag_is_the_config_rule():
    ids = list(range(900_000, 901_000))
    assert [select.vault_flag(c) for c in ids] == [config.in_vault(c) for c in ids]
    assert 0.15 < np.mean([select.vault_flag(c) for c in ids]) < 0.25


def test_assert_no_vault_on_ids_frames_and_columns():
    v, d = vault_ids(1, start=81_000)[0], dev_ids(3, start=81_000)
    select.assert_no_vault(d)
    select.assert_no_vault(pd.DataFrame({"class_id": d, "in_vault": False}))
    select.assert_no_vault(pd.Series(d))
    select.assert_no_vault([])
    for bad in ([*d, v], v, pd.Series([v]), pd.DataFrame({"class_id": [v]}),
                pd.DataFrame({"class_id": d, "in_vault": [False, True, False]})):
        with pytest.raises(select.VaultError) as err:
            select.assert_no_vault(bad)
        assert str(v) not in str(err.value)                # ids are counted, never printed


# --------------------------------------------------------------------------- sections, phases


def fake_classes() -> pd.DataFrame:
    """Five sections in F23 (means 9, 7, 6, 4 and an excluded 8), one F26 CS156 section with
    a low mean, one F26 other course, and unrecorded classes that must not count."""
    spec = [(1, "SS110", "F23", 9.0, 6), (2, "NS111", "F23", 7.0, 6), (3, "AH112", "F23", 6.0, 6),
            (4, "B113", "F23", 4.0, 6), (5, "CS114", "F23", 8.0, 6),
            (6, "CS156", "F26", 1.0, 5), (7, "CS999", "F26", 20.0, 5), (8, "SS120", "Sp24", 5.0, 4)]
    rows, cid = [], 1
    for sid, code, term, mean, n in spec:
        for k in range(n):
            cid += 1
            rows.append({"class_id": cid, "section_id": sid, "course_code": code, "term": term,
                         "semester_idx": config.SEMESTER_IDX[term], "n_rec_sessions": 1,
                         "n_my_voice_main": mean + (1 if k % 2 else -1), "in_vault":
                         config.in_vault(cid)})
        cid += 1
        rows.append({"class_id": cid, "section_id": sid, "course_code": code, "term": term,
                     "semester_idx": config.SEMESTER_IDX[term], "n_rec_sessions": 0,
                     "n_my_voice_main": 0, "in_vault": config.in_vault(cid)})
    df = pd.DataFrame(rows)
    df["n_my_voice_main"] = df["n_my_voice_main"].astype("Int64")
    return df


def test_choose_sections_follows_section_10():
    secs = select.choose_sections(fake_classes(), exclude_sections=[5]).set_index("section_id")
    assert secs.loc[1, "tier"] == "headline" and secs.loc[1, "reason"] == "top1"
    assert secs.loc[2, "tier"] == "headline" and secs.loc[2, "reason"] == "top2"
    assert secs.loc[3, "tier"] == "phase_c" and secs.loc[3, "reason"] == "rank3"
    assert pd.isna(secs.loc[4, "tier"]) and secs.loc[4, "reason"] == "below_min"
    assert pd.isna(secs.loc[5, "tier"]) and secs.loc[5, "reason"] == "excluded"
    assert secs.loc[6, "tier"] == "headline" and secs.loc[6, "reason"] == "always"
    assert pd.isna(secs.loc[7, "tier"]) and secs.loc[7, "reason"] == "not_past_term"
    assert secs.loc[8, "tier"] == "headline" and secs.loc[8, "mean_my_voice_main"] == 5.0
    assert secs.loc[1, "n_classes"] == 6                     # unrecorded classes not counted
    no_c = select.choose_sections(fake_classes(), phase_c=False).set_index("section_id")
    assert no_c.loc[5, "tier"] == "headline" and no_c.loc[5, "reason"] == "top2"
    assert pd.isna(no_c.loc[2, "tier"]) and no_c.loc[2, "reason"] == "rank3"


def test_assign_phases_vault_free_phase_a_and_round_robin():
    classes = fake_classes()
    secs = select.choose_sections(classes)
    pilot = [c for c in classes.loc[classes["section_id"] == 4, "class_id"]
             if not config.in_vault(c) and c in set(classes.loc[classes["n_rec_sessions"] > 0,
                                                                "class_id"])][:1]
    out = select.assign_phases(classes, secs, pilot_ids=pilot)
    ph = out.set_index("class_id")["phase"]
    assert list(ph[ph == "pilot"].index) == pilot
    a = out[out["phase"] == "A"]
    assert not a["in_vault"].any()
    assert a.groupby("section_id").size().max() <= config.PHASE_A_PER_SECTION
    headline = set(secs.loc[secs["tier"] == "headline", "section_id"])
    rec = out[out["n_rec_sessions"] > 0]
    assert set(rec.loc[rec["section_id"].isin(headline), "phase"]) <= {"A", "B"}
    assert rec.loc[rec["section_id"].isin(headline), "phase"].notna().all()
    assert set(out.loc[out["phase"] == "C", "section_id"]) == \
        set(secs.loc[secs["tier"] == "phase_c", "section_id"])
    assert out.loc[out["n_rec_sessions"] == 0, "phase"].isna().all()
    b_vault = out[(out["phase"] == "B") & out["in_vault"]]
    assert len(b_vault) == out.loc[out["section_id"].isin(headline) & out["in_vault"]
                                   & (out["n_rec_sessions"] > 0)].shape[0]
    # run order: pilot < A < B < C, contiguous, and round-robin over sections within A
    sel = out[out["phase"].notna()].sort_values("run_order")
    assert list(sel["run_order"]) == list(range(len(sel)))
    rank = {"pilot": 0, "A": 1, "B": 2, "C": 3}
    assert sel["phase"].map(rank).is_monotonic_increasing
    a_order = sel[sel["phase"] == "A"]["section_id"].tolist()
    n_sec = a["section_id"].nunique()
    assert len(set(a_order[:n_sec])) == n_sec
    # deterministic, and the within-section order depends only on (seed, section)
    again = select.assign_phases(classes, secs, pilot_ids=pilot)
    pd.testing.assert_frame_equal(out, again)
    other_seed = select.assign_phases(classes, secs, pilot_ids=pilot, seed=7)
    assert not other_seed["run_order"].equals(out["run_order"])


def test_pilots_must_be_dev_classes():
    classes = fake_classes()
    secs = select.choose_sections(classes)
    v = next(c for c in classes["class_id"] if config.in_vault(c))
    with pytest.raises(select.VaultError):
        select.assign_phases(classes, secs, pilot_ids=[v])


def test_funnel_counts_only():
    classes = select.assign_phases(fake_classes(), select.choose_sections(fake_classes()))
    f = select.funnel(classes)
    assert f["synced"] == len(classes) and f["recorded"] == (classes["n_rec_sessions"] > 0).sum()
    assert f["selected"] == f["selected_dev"] + f["selected_vault"]
    assert sum(f[f"phase_{p}"] for p in config.PHASES) == f["selected"]


def test_end_to_end_on_the_synthetic_world(built, world):
    secs = select.choose_sections(built, always=[("CS901", "F26")], min_mean=1.0)
    assert set(secs["tier"].dropna()) == {"headline"}
    out = select.assign_phases(built, secs)
    assert out["phase"].notna().all() and not out.loc[out["phase"] == "A", "in_vault"].any()
    schemas.validate(out, "classes")


# --------------------------------------------------------------------------- CLI


def test_cli_writes_classes_parquet(world, data_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "load_env", lambda *a, **k: False)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "DROPPED_KEYS_LOG", tmp_path / "logs" / "dropped_keys.csv")
    out = tmp_path / "classes.parquet"
    pilot = next(c for c in world.class_ids if not config.in_vault(c))
    args = argparse.Namespace(command="select", phase=None, workers=1, class_ids=None,
                              extra=["--scribe-dir", str(data_dir["scribe"]), "--out", str(out),
                                     "--always", "CS901:F26", "--pilot", str(pilot)])
    assert select.cli(args) == 0
    df = pd.read_parquet(out)
    schemas.validate(df, "classes")
    assert df.set_index("class_id").loc[pilot, "phase"] == "pilot"
    assert (tmp_path / "logs" / "sections_selected.csv").exists()
    text = capsys.readouterr().out
    assert "funnel:" in text and "CS901" in text
    names = {n for sc in world for n in sc.roster["last"]}
    assert not any(n in text for n in names)
    assert synthetic.ME_LAST not in text
