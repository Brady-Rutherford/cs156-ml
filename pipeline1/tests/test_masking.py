"""oncue.text: name masking (§5.1), normalisation (§5.3), causal frames (§5.2), windows at t,
the professor-question detector and rule R (§5.5). Synthetic data only (invented names)."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from oncue import config, schemas, synthetic, text

EPS = config.EPS_W_S
ROSTER = [
    {"user_id": "1", "first": "Ada", "last": "Lovelace"},
    {"user_id": "2", "first": "Grace", "last": "Hopper"},
    {"user_id": "3", "first": "Will", "last": "Herschel"},
    {"user_id": "4", "first": "Bo", "last": "Li"},
    {"user_id": "5", "first": "Emmy", "last": "Noether"},
    {"user_id": "7777", "first": "Brady", "last": "Rutherford"},
]


@pytest.fixture(scope="module")
def pats():
    return text.roster_patterns(ROSTER, me_id=7777)


def m(s: str, pats, **kw) -> str:
    return " ".join(text.mask(s.split(), pats, **kw))


def mkwords(rows, class_id: int = 1) -> pd.DataFrame:
    """rows: (start, end, spk, role, text[, role_avail_s])."""
    out = []
    for i, r in enumerate(rows):
        start, end, spk, role, txt = r[:5]
        out.append({"class_id": class_id, "w_idx": i, "start_s": float(start),
                    "end_s": float(end), "spk_hash": spk, "role": role, "text_norm": txt,
                    "text_masked": txt, "has_qmark": txt.endswith("?"),
                    "role_avail_s": float(r[5]) if len(r) > 5 else float(end) + EPS})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- masking


def test_exact_names_possessives_and_punctuation(pats):
    assert m("I agree with Ada's point, Lovelace.", pats) == "I agree with [NAME] point, [NAME]."
    assert m("thanks (Noether) and Herschel's idea", pats) == "thanks ([NAME]) and [NAME] idea"
    # a non-common name matches case-insensitively, even lowercase mid-sentence
    assert m("as lovelace said", pats) == "as [NAME] said"
    # already-masked tokens are left alone
    assert m("so [NAME] said", pats) == "so [NAME] said"


def test_common_word_names_need_capital_mid_sentence(pats):
    assert text.common_words() >= {"grace", "will", "mark"}
    assert m("Grace is a virtue.", pats) == "Grace is a virtue."            # sentence-initial
    assert m("I will say it with grace.", pats) == "I will say it with grace."
    assert m("Thanks Will. I see Grace now.", pats) == "Thanks [NAME]. I see [NAME] now."
    # vocative extension: sentence-initial, capitalised, followed by a comma
    assert m("Grace, go ahead.", pats) == "[NAME], go ahead."
    assert m("Grace, go ahead.", pats, vocative=False) == "Grace, go ahead."


def test_brady_variants_always_masked(pats):
    # "brady" and "rutherford" are wordfreq-common, but Brady's name is always masked
    assert m("Brady, go ahead.", pats) == "[NAME], go ahead."
    assert m("i think brady and Rutherford agree", pats) == "i think [NAME] and [NAME] agree"
    for v in ("Bradley", "Braden", "Braydon"):
        assert m(f"Okay {v}.", pats) == "Okay [NAME]."
    assert m("Brady, go ahead.", pats, me_token=config.ME_TOKEN) == "[ME], go ahead."
    assert m("thanks Brady", pats, me_token=config.ME_TOKEN) == "thanks [ME]"


def test_levenshtein_only_capitalised_mid_sentence(pats):
    assert m("I agree with Lovelase here", pats) == "I agree with [NAME] here"
    assert m("I agree with Hoper here", pats) == "I agree with [NAME] here"   # -> Hopper
    assert m("I agree with lovelase here", pats) == "I agree with lovelase here"
    assert m("Lovelase said so", pats) == "Lovelase said so"                  # initial
    # names under 4 characters match exactly only
    assert m("ask Li and Lu", pats) == "ask [NAME] and Lu"
    # Brady typo -> me_token
    assert m("thanks Bradey", pats, me_token=config.ME_TOKEN) == "thanks [ME]"


def test_initial_override_and_masker_adapter(pats):
    toks = ["thanks", "Grace", "yes", "Will", "agrees"]
    # a speaker change makes "Will" sentence-initial -> the common-word name stays
    assert text.mask(toks, pats, initial=[True, False, True, True, False]) \
        == ["thanks", "[NAME]", "yes", "Will", "agrees"]
    masker = text.TextMasker(pats, me_token=config.ME_TOKEN)
    masked, kinds = masker.mask_tokens(["Brady,", "and", "Ada", "too"],
                                       [True, False, False, False])
    assert masked == ["[ME],", "and", "[NAME]", "too"]
    assert kinds == ["me", None, "name", None]
    assert masker.mask_text("ask Noether") == "ask [NAME]"
    assert "noether" not in repr(masker).lower()


def test_roster_patterns_from_class_json_and_repr(synth_class):
    pats = text.roster_patterns(synth_class.raw["class"], me_id=synthetic.ME_ID)
    first = synth_class.roster.loc[synth_class.roster["role"] == "student", "first"].iloc[0]
    assert first.lower() in pats.names
    assert "brady" in pats.me_names and "rutherford" in pats.me_names
    assert first not in repr(pats) and "Brady" not in repr(pats)
    # snake_case class JSON works too
    snake = synthetic._snake(synth_class.raw["class"])
    assert text.roster_patterns(snake, me_id=synthetic.ME_ID).names == pats.names


def test_mask_text_handles_missing(pats):
    assert text.mask_text(None, pats) == ""
    assert text.mask_text(float("nan"), pats) == ""
    assert text.mask_text("Case activity: Ada Lovelace's proposal", pats) \
        == "Case activity: [NAME] [NAME] proposal"


def test_mask_audit_and_build_failure(pats):
    orig = ["I", "agree", "with", "Ada", "and", "Lovelase", "here"]
    masked = text.mask(orig, pats)
    audit = text.assert_mask_safe(orig, masked, pats)
    assert audit["n_masked"] == 2 and audit["bad_forms"] == []
    assert audit["masked_rate"] == pytest.approx(2 / 7)
    # a top-1,000 word masked although it is not a roster name -> the build fails
    with pytest.raises(text.MaskingError):
        text.assert_mask_safe(["the", "Ada"], ["[NAME]", "[NAME]"], pats)


def test_synthetic_masking_recall_and_precision(world):
    for sc in world:
        pats = text.roster_patterns(sc.raw["class"], me_id=synthetic.ME_ID)
        toks = [w["punctuated_word"] for w in sc.asr["words"]]
        masked = text.mask(toks, pats)
        is_name = sc.words["is_name"].to_numpy(bool)
        got = np.array(["[NAME]" in x for x in masked])
        assert got[is_name].all(), "a planted name escaped masking"
        assert not got[~is_name].any(), "a non-name token was masked"
        norm = " ".join(text.normalise_words(masked))
        assert "brady" not in norm.split()
        noncommon = {k for k in pats.names if k not in pats.common}
        assert not noncommon & set(norm.split())


# --------------------------------------------------------------------------- normalisation


def test_normalise_headline_name_and_punct_variants():
    s = "Okay, [NAME], go ahead. What's the X-factor here?"
    assert text.normalise(s) == "okay whats the x factor here"
    assert text.normalise(s, keep_names=True) == "okay _name_ whats the x factor here"
    assert text.normalise("thanks [ME] and [NAME]", keep_names=True) == "thanks _me_ and _name_"
    assert text.normalise(s, punct=True) == "Okay, What's the X-factor here?"
    assert text.normalise(s, callon=False) == "okay go ahead whats the x factor here"
    assert text.normalise(None) == "" and text.normalise(float("nan")) == ""


def test_delete_callon_variants():
    for phrase in ("go ahead", "you're up", "youre up", "I see a hand", "i see your hand",
                   "unmute", "take it away"):
        assert text.delete_callon(f"alright {phrase} now") == "alright now", phrase
    assert text.delete_callon("going ahead with it") == "going ahead with it"


def test_normalise_words_keeps_multiword_callon_for_causal_joins():
    toks = ["[NAME],", "go", "ahead.", "Please", "unmute."]
    assert text.normalise_words(toks) == ["", "go", "ahead", "please", ""]
    cols = text.text_columns(["Is", "that", "right?"], text.roster_patterns([]))
    assert cols["has_qmark"] == [False, False, True]
    assert cols["text_norm"] == ["is", "that", "right"]


# --------------------------------------------------------------------------- frames (§5.2)


def test_pause_cut_boundary_fixture():
    # a pause cut needs no onset in (end, end + 0.4]: an onset at exactly +0.4 s blocks it
    assert 0.9 - 0.5 == config.FRAME_PAUSE_S          # exact 0.4 s gap in floats
    w = mkwords([(0.0, 0.5, "a", "prof", "one"), (0.9, 1.2, "a", "prof", "two")])
    assert len(text.frames(w)) == 1
    w = mkwords([(0.0, 0.5, "a", "prof", "one"), (0.91, 1.2, "a", "prof", "two")])
    f = text.frames(w)
    assert len(f) == 2
    c0 = 0.5 + config.FRAME_PAUSE_S + EPS
    assert f["close_s"].iloc[0] == pytest.approx(c0)
    assert len(text.frames_at(w, c0)) == 1 and len(text.frames_at(w, c0 - 1e-9)) == 0
    # the last frame closes by the pause rule once 0.4 s of silence is observable
    last = 1.2 + config.FRAME_PAUSE_S + EPS
    assert f["close_s"].iloc[1] == pytest.approx(last)
    assert len(text.frames_at(w, last)) == 2


def test_speaker_change_and_role_availability():
    w = mkwords([(0.0, 0.5, "a", "prof", "one"), (0.6, 1.0, "b", "student", "two")])
    f = text.frames(w)
    assert f["close_s"].iloc[0] == pytest.approx(1.0 + EPS)
    # the next speaker's role becomes known late -> the cut is observable only then
    w = mkwords([(0.0, 0.5, "a", "prof", "one"), (0.6, 1.0, "b", "student", "two", 2.0)])
    assert text.frames(w)["close_s"].iloc[0] == pytest.approx(2.0)
    assert len(text.frames_at(w, 1.99)) == 0
    assert len(text.frames_at(w, 2.0)) == 2        # "two" also pause-closed by then
    # a frame waits for its own words' roles
    w = mkwords([(0.0, 0.5, "a", "prof", "one", 3.0), (2.0, 2.4, "a", "prof", "two")])
    assert text.frames(w)["close_s"].iloc[0] == pytest.approx(3.0)


def test_length_cut_includes_crossing_word():
    rows = [(0.5 * k, 0.5 * k + 0.45, "a", "prof", f"w{k}") for k in range(16)]
    f = text.frames(mkwords(rows))
    assert f["n_words"].iloc[0] == 11                 # word 10 ends at 5.45 >= 5 s
    assert f["close_s"].iloc[0] == pytest.approx(5.45 + EPS)
    assert f["start_s"].iloc[1] == pytest.approx(5.5)


def test_close_is_monotone_and_text_has_no_callon(truth):
    for tables in truth.values():
        f = text.frames(tables["words"])
        schemas.validate(f, "frames", strict=True)
        assert (np.diff(f["close_s"]) >= 0).all()
        assert (f["close_s"] >= f["end_s"] + EPS - 1e-9).all()
        assert not f["text_norm"].str.contains(r"\bgo ahead\b|\[name\]", regex=True).any()
        assert f["f_idx"].tolist() == list(range(len(f)))


def test_fast_frames_equal_reference_path(world, truth):
    cols = ["start_s", "end_s", "close_s", "role", "spk_hash", "text_norm", "n_words"]
    rng = np.random.default_rng(156)
    for sc in world.classes[:3]:
        w = truth[sc.class_id]["words"]
        full = text.frames(w)
        ts = list(rng.uniform(0, sc.T_end_s, 25))
        ts += list(full["close_s"].iloc[::17]) + list(full["close_s"].iloc[::17] - 1e-9)
        for t in ts:
            fast = full[full["close_s"] <= t].reset_index(drop=True)
            ref = text.frames_at(w, t)
            pd.testing.assert_frame_equal(fast[cols], ref[cols], check_dtype=False)


def test_sentinel_words_after_cutoff_change_no_closed_frame(truth, synth_class):
    w = truth[synth_class.class_id]["words"]
    full = text.frames(w)
    for t in (400.0, 777.7, 1300.3):
        sent = mkwords([(t - EPS + 0.05, t - 0.01, "zz", "prof", "zqxsentinel"),
                        (t + 0.1, t + 0.4, "zz", "prof", "zqxsentinel")],
                       class_id=synth_class.class_id)
        sent["w_idx"] += 10**6
        w2 = pd.concat([w, sent.reindex(columns=w.columns)], ignore_index=True)
        a = full[full["close_s"] <= t].reset_index(drop=True)
        b = text.frames_at(w2, t)
        pd.testing.assert_frame_equal(a[["start_s", "close_s", "text_norm"]],
                                      b[["start_s", "close_s", "text_norm"]])


def test_assign_frames_matches_frames(truth, synth_class):
    w = truth[synth_class.class_id]["words"]
    f_of, close = text.assign_frames(w)
    f = text.frames(w)
    assert np.allclose(close, f["close_s"])
    assert f_of.max() == len(f) - 1
    counts = pd.Series(f_of).value_counts().sort_index()
    assert counts.sum() == len(w)


# --------------------------------------------------------------------------- windows (§5.3)


def _win_words():
    return mkwords([
        (0.0, 0.4, "p", "prof", "so"), (0.45, 0.8, "p", "prof", "what"),
        (0.85, 1.1, "p", "prof", "do"), (1.15, 1.4, "p", "prof", "you"),
        (1.45, 1.8, "p", "prof", "think"),
        (3.0, 3.3, "s", "student", "i"), (3.35, 3.7, "s", "student", "agree"),
        (5.0, 5.3, "p", "prof", ""), (5.35, 5.6, "p", "prof", "go"),
        (5.65, 6.0, "p", "prof", "ahead"),
        (8.0, 8.4, "me", "me", "my"), (8.45, 8.9, "me", "me", "point"),
        (9.0, 9.5, "p", "prof", "good", 11.0),
    ])


def test_windows_use_only_available_words_and_roles():
    wb = text.WindowBuilder(_win_words())
    r = wb.at(9.5 + EPS)                      # "good" text is available, its role is not
    assert r["X_all60"] == "<P> so what do you think <S> i agree <M> my point <U> good"
    assert r["T_all15"].split()[:2] == ["p_so", "p_what"]
    assert r["T_all15"].endswith("m_my m_point u_good")
    assert r["X_prof60"] == "so what do you think"
    assert r["X_stud60"] == "i agree"
    assert "go" not in r["X_all60"].split()             # CALLON deleted from joined text
    r2 = wb.at(9.5 + EPS - 1e-6)
    assert not r2["X_all60"].endswith("good")
    assert wb.at(11.0)["X_all60"].endswith("<P> good")


def test_window_spans_and_lastprof():
    wb = text.WindowBuilder(_win_words())
    r = wb.at(16.9)
    assert "so" not in r["X_all15"].split() and "think" not in r["X_all15"].split()
    assert r["X_all15"].startswith("<S> i agree")
    assert r["X_lastprof"] == "good" and r["lastprof_end_s"] == pytest.approx(9.5)
    r = wb.at(4.0)
    assert r["X_lastprof"] == "so what do you think"
    assert r["lastprof_end_s"] == pytest.approx(1.8)


def test_windows_chat_availability():
    chat = pd.DataFrame({"class_id": 1, "t_s": [10.0, 38.5, 39.2],
                         "spk_hash": ["s", "s", "s"], "role": "student",
                         "text_masked": ["early note", "Thanks [NAME]!", "too new"]})
    with pytest.raises(ValueError):
        text.WindowBuilder(_win_words(), chat)
    wb = text.WindowBuilder(_win_words(), chat, eps_f=1.0)
    r = wb.at(40.0)
    assert r["X_chat60"] == "<C> early note <C> thanks"
    assert r["T_chat60"] == "c_early c_note c_thanks"
    assert wb.at(71.0)["X_chat60"] == "<C> thanks <C> too new"


def test_tail_and_me_sofar():
    wb = text.WindowBuilder(_win_words())
    r = wb.at(2.1)                       # "think" available (2.05), frame open until 2.45
    assert r["X_tail"] == "so what do you think"
    assert r["tail_s"] == pytest.approx(1.8 + EPS)
    assert wb.at(2.0)["X_tail"] == "so what do you"
    r = wb.at(9.2)                       # Brady's words are never in the tail
    assert "my" not in r["X_tail"].split()
    assert len(r["me_sofar"]) == 0
    r = wb.at(12.0)
    assert len(r["me_sofar"]) == 1


def test_window_sentinels_change_nothing(truth, synth_class):
    tables = truth[synth_class.class_id]
    w, chat = tables["words"], tables["chat"]
    base = text.WindowBuilder(w, chat, eps_f=1.0)
    for t in (300.0, 640.0, 1111.0):
        sent = mkwords([(t - EPS + 0.02, t - 0.01, "zz", "prof", "zqxsentinel"),
                        (t + 0.1, t + 0.5, "zz", "prof", "zqxsentinel")],
                       class_id=synth_class.class_id)
        sent["w_idx"] += 10**6
        wb = text.WindowBuilder(pd.concat([w, sent.reindex(columns=w.columns)],
                                          ignore_index=True), chat,
                                eps_f=1.0)
        a, b = base.at(t), wb.at(t)
        for k in text.WINDOW_KEYS:
            if isinstance(a[k], np.ndarray):
                assert np.array_equal(a[k], b[k]), k
            elif isinstance(a[k], float) and np.isnan(a[k]):
                assert np.isnan(b[k]), k
            else:
                assert a[k] == b[k], k


def test_callon_and_name_inside_a_run_never_reach_window_text():
    rows = [(0.0, 0.4, "p", "prof", "so"), (0.45, 0.8, "p", "prof", "what"),
            (0.85, 1.1, "p", "prof", "next")]
    with_cue = rows[:2] + [(0.81, 0.83, "p", "prof", ""), (0.835, 0.84, "p", "prof", "go"),
                           (0.842, 0.845, "p", "prof", "ahead")] + rows[2:]
    a = text.WindowBuilder(mkwords(rows)).at(5.0)
    b = text.WindowBuilder(mkwords(with_cue)).at(5.0)
    for k in ("X_all15", "X_all60", "T_all15", "T_all60", "X_prof60", "X_lastprof"):
        assert a[k] == b[k], k


def test_name_and_punct_variants():
    w = mkwords([(0.0, 0.4, "p", "prof", "Thanks"), (0.45, 0.8, "p", "prof", "[NAME],"),
                 (0.85, 1.1, "p", "prof", "right?")])
    w["text_norm"] = text.normalise_words(w["text_masked"])
    assert text.WindowBuilder(w).at(5)["X_all60"] == "<P> thanks right"
    assert text.WindowBuilder(w, variant="name").at(5)["X_all60"] == "<P> thanks _name_ right"
    assert text.WindowBuilder(w, variant="punct").at(5)["X_all60"] == "<P> Thanks right?"
    many = text.WindowBuilder(w).many([1.0, 5.0])
    assert list(many.columns) == ["t_s", *text.WINDOW_KEYS] and len(many) == 2


# --------------------------------------------------------------------------- questions (§5.5)


def test_question_rules():
    assert text.is_question("what do you think about this")
    assert not text.is_question("what now")                   # < 4 words
    assert text.is_question("so that is the idea right")
    assert text.is_question("that is the idea isnt it")
    assert not text.is_question("the answer is clear")
    assert not text.is_question("the answer is clear?")       # no '?' rule in the headline
    assert text.is_question("the answer is clear?", punct=True)
    assert text.is_question("the answer is clear", punct=True, qmark=True)
    assert text.question_kind("does that make sense") == "check"
    assert text.question_kind("are there any questions") == "check"
    assert text.question_kind("does anyone want to share") == "invite"
    assert text.question_kind("why does x matter") == "plain"


def _q_words(resume_at: float | None = None, student_at: float | None = 6.0):
    rows = [(0.0, 0.3, "p", "prof", "so"), (0.35, 0.6, "p", "prof", "here"),
            (0.65, 0.9, "p", "prof", "it"), (0.95, 1.2, "p", "prof", "is"),
            (1.8, 2.0, "p", "prof", "what"), (2.05, 2.3, "p", "prof", "is"),
            (2.35, 2.6, "p", "prof", "the"), (2.65, 3.0, "p", "prof", "point")]
    if resume_at is not None:
        rows.append((resume_at, resume_at + 0.3, "p", "prof", "anyway"))
    if student_at is not None:
        rows += [(student_at, student_at + 0.3, "s", "student", "i"),
                 (student_at + 0.35, student_at + 0.6, "s", "student", "would"),
                 (student_at + 0.65, student_at + 0.9, "s", "student", "say")]
    return mkwords(sorted(rows))


def test_question_detector_events():
    q = text.question_detector(_q_words())
    ev = q[q["end_s"] == 3.0].iloc[0]
    assert ev["is_q"] and ev["new_q"] and ev["kind"] == "plain" and not ev["rhetorical"]
    assert ev["avail_s"] == pytest.approx(3.0 + config.Q_SILENCE_S + EPS)
    assert ev["start_s"] == 0.0                     # the 0.6 s pause does not split it
    # the spec-literal test (whole utterance only) misses the trailing question
    q0 = text.question_detector(_q_words(), split_pause_s=None)
    assert not q0.loc[q0["end_s"] == 3.0, "is_q"].iloc[0]
    # a pause of 0.6 s after "is" is < 0.8 s silence -> no evaluation there
    assert not (q["end_s"] == 1.2).any()


def test_rhetorical_and_qmark_rule():
    q = text.question_detector(_q_words(resume_at=4.5))
    ev = q[q["end_s"] == 3.0].iloc[0]
    assert ev["rhetorical"]
    assert ev["rhetorical_avail_s"] == pytest.approx(3.0 + config.Q_RHETORICAL_RESUME_S + EPS)
    w = mkwords([(0.0, 0.3, "p", "prof", "the"), (0.35, 0.6, "p", "prof", "answer"),
                 (0.65, 1.0, "p", "prof", "is?")])
    w["text_norm"] = text.normalise_words(w["text_masked"])
    assert not text.question_detector(w)["is_q"].any()
    assert text.question_detector(w, punct=True)["is_q"].all()


def test_q_features_and_rule_r():
    w = _q_words(student_at=8.0)
    q = text.question_detector(w)
    f = text.frames(w)
    av = 3.0 + config.Q_SILENCE_S + EPS
    ts = [av - 0.01, av, 6.0, 9.6]                  # student frame closes at 9.55
    qf = text.q_features(q, w, f, ts).set_index("t_s")
    assert qf.loc[ts[0], "q_open"] == 0 and qf.loc[ts[0], "q_kind"] == "none"
    assert qf.loc[ts[0], "q_age_s"] == config.Q_AGE_CAP_S
    assert qf.loc[av, "q_open"] == 1 and qf.loc[av, "q_age_s"] == pytest.approx(av - 3.0)
    assert qf.loc[6.0, "q_count_300"] == 1 and qf.loc[6.0, "q_kind"] == "plain"
    assert qf.loc[9.6, "q_open"] == 0                # the student 3-word frame closed
    r = text.rule_R(qf.reset_index())
    assert r[1] == pytest.approx(1.0 * np.exp(-min(300, 2 * (av - 3.0)) / 10))
    assert text.rule_R([1.0], [0.0], [300.0])[0] == pytest.approx(1.0)
    assert text.rule_R([0.0], [10.0], [5.0])[0] == pytest.approx(0.5 * np.exp(-0.5))
    # rhetorical once known: kind flips and the question stops being open
    wr = _q_words(resume_at=4.5, student_at=None)
    qr = text.q_features(text.question_detector(wr), wr, text.frames(wr), [3.5, 6.5])
    assert qr["q_kind"].tolist() == ["none", "rhetorical"]
    assert qr["q_open"].tolist() == [0.0, 0.0]


def test_q_features_are_causal(world, truth):
    rng = np.random.default_rng(7)
    for sc in world.classes[:2]:
        w = truth[sc.class_id]["words"]
        full = text.q_features(text.question_detector(w), w, text.frames(w),
                               rng.uniform(150, sc.T_end_s, 30))
        for _, row in full.iterrows():
            t = row["t_s"]
            tw = text.truncate_words(w, t)
            qt = text.q_features(text.question_detector(tw), tw, text.frames_at(w, t), [t])
            for col in text.Q_FEATURE_COLUMNS:
                assert qt[col].iloc[0] == row[col] or (
                    isinstance(row[col], float) and qt[col].iloc[0] == pytest.approx(row[col])
                ), (t, col)


def test_synthetic_questions_before_bids_are_detected(world, truth):
    hits = total = 0
    for sc in world:
        w = truth[sc.class_id]["words"]
        q = text.question_detector(w)
        qq = q[q["is_q"]]
        bids = sc.episodes[(sc.episodes["ep_type"] == "bid") & sc.episodes["has_question"]]
        for qe in bids["q_end_s"]:
            total += 1
            hits += bool((np.abs(qq["end_s"] - qe) < 0.05).any())
    assert total >= 10 and hits / total >= 0.8


def test_callon_events_for_the_callon_check(truth, synth_class):
    w = truth[synth_class.class_id]["words"].copy()
    # truth text_masked holds the cased words; CALLON tokens are planted by the world
    ev = text.callon_events(w)
    n_planted = int((synth_class.turns["kind"] == "callon").sum())
    assert len(ev) >= n_planted and (ev["avail_s"] >= ev["end_s"] + EPS - 1e-9).all()


# --------------------------------------------------------------------------- CLI


def test_cli_frames_writes_valid_table(tmp_path, truth):
    words = pd.concat([t["words"] for t in truth.values()], ignore_index=True)
    words.to_parquet(tmp_path / "words.parquet", index=False)
    args = types.SimpleNamespace(extra=["--tables", str(tmp_path)], class_ids=None)
    assert text.cli_frames(args) == 0
    out = pd.read_parquet(tmp_path / "frames.parquet")
    schemas.validate(out, "frames")
    assert out["class_id"].nunique() == len(truth)
    one = str(next(iter(truth)))
    args = types.SimpleNamespace(extra=["--tables", str(tmp_path)], class_ids=[one])
    assert text.cli_frames(args) == 0
    again = pd.read_parquet(tmp_path / "frames.parquet")
    assert len(again) == len(out)
