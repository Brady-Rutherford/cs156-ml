"""The real-strip scanner (oncue.strip), the split selector and the work-dir switch."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
import pytest

from oncue import config, hands, ingest, splits, strip

HAND_BGR = tuple(int(x) for x in cv2.cvtColor(np.uint8([[[32, 220, 220]]]),
                                              cv2.COLOR_HSV2BGR)[0, 0])
RING_BGR = tuple(int(x) for x in cv2.cvtColor(np.uint8([[[76, 200, 200]]]),
                                              cv2.COLOR_HSV2BGR)[0, 0])


def frame(n_tiles=6, w=90, h=70, y0=50, hand_on=None, ring_on=None, words=None):
    """A dark frame with a strip of grey tiles at y0, an icon / ring on the given tiles and a
    white word drawn at the bottom of every tile."""
    f = np.zeros((360, 1920, 3), np.uint8)
    for i in range(n_tiles):
        x0 = 4 + i * (w + 4)
        f[y0:y0 + h, x0:x0 + w] = (120, 110, 100)
        if ring_on == i:
            cv2.rectangle(f, (x0, y0), (x0 + w - 1, y0 + h - 1), RING_BGR, 3)
        if hand_on == i:
            cv2.circle(f, (x0 + 13, y0 + 13), 8, HAND_BGR, -1)
            cv2.rectangle(f, (x0 + 10, y0 + 9), (x0 + 15, y0 + 17), (255, 255, 255), -1)
        text = (words or {}).get(i, f"P{i}x")
        cv2.putText(f, text, (x0 + 12, y0 + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return f


def v_of(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 2] / 255.0


def test_find_band_locates_strip_under_header_and_below_a_banner():
    b = strip.find_band(v_of(frame()))
    assert (b.y0, b.y1) == (50, 120) and len(b.tiles) == 6
    f = frame(y0=102)
    f[40:53, :1500] = 200                                  # a thin banner above the strip
    b = strip.find_band(v_of(f))
    assert (b.y0, b.y1) == (102, 172)


def test_find_band_none_without_a_tile_row():
    assert strip.find_band(v_of(np.zeros((360, 1920, 3), np.uint8))) is None


def test_tile_stats_measure_icon_and_ring_with_forum_colours():
    f = frame(hand_on=2, ring_on=4)
    b = strip.find_band(v_of(f))
    st = strip.tile_stats(np.ascontiguousarray(f[b.y0:b.y1]), b)
    up = (st["frac_hand_s1"] >= strip.FORUM_DETECTOR.theta_on) & hands.shape_ok(
        st["shape_s1"], st["white_s1"], strip.FORUM_DETECTOR)
    assert np.flatnonzero(up).tolist() == [2]
    assert np.flatnonzero(st["border"] >= 0.3).tolist() == [4]
    assert all(p is not None for p in st["word"])


def test_match_name_fixes_ocr_confusions_and_needs_a_margin():
    names = {"1": ["brady"], "2": ["grace"], "3": ["bradley"]}
    assert strip.match_name("Bracly", names)[0] in ("1", None)
    assert strip.match_name(["Bracy", "Brady"], names)[0] == "1"
    assert strip.match_name("xq", names)[0] is None
    # two equally good people: no margin, no match
    assert strip.match_name("brad", {"1": ["brady"], "2": ["bradi"]})[0] is None


def test_label_book_reads_once_per_label_and_width():
    calls = []

    def fake_ocr(white):
        calls.append(1)
        return ["Brady"]

    book = strip.LabelBook({"1": ["brady"], "2": ["grace"]}, "1", ocr=fake_ocr)
    patch = np.zeros((strip.WORD_H, strip.WORD_W), np.float32)
    patch[4:12, 2:40] = 1
    white = np.ones((10, 40), bool)
    assert book.identify(90, patch, white) == "1"
    assert book.identify(91, patch, white) == "1"            # same width within 2 px
    assert len(calls) == 1
    assert book.identify(150, patch, white) == "1" and len(calls) == 2
    w, p = book.me_patches()
    assert sorted(w.tolist()) == [90, 150] and p.shape[1:] == (strip.WORD_H, strip.WORD_W)


def test_label_book_retries_unread_labels_then_gives_up():
    book = strip.LabelBook({"1": ["brady"]}, "1", ocr=lambda w: [""])
    patch = np.ones((strip.WORD_H, strip.WORD_W), np.float32)
    patch[:, :10] = 0
    for _ in range(strip.OCR_RETRIES + 3):
        assert book.identify(90, patch, np.ones((4, 4), bool)) is None
    assert book.n_ocr == strip.OCR_RETRIES


def test_scan_class_marks_my_tile_and_checks_anchors():
    me_word = "Brady"
    frames = [(float(t), frame(words={3: me_word}, hand_on=3 if t >= 2 else None,
                               ring_on=3 if t < 2 else None)) for t in np.arange(0, 4, 0.5)]

    def ocr_by_tile(white):                               # tile 3's label reads "Brady"
        return [me_word] if int(white.sum()) == me_key else ["Grace"]

    f0 = frames[0][1]
    b = strip.find_band(v_of(f0))
    me_key = int(strip.label_mask(f0[b.y0:b.y1, b.tiles[3, 0]:b.tiles[3, 1]]).sum())
    book = strip.LabelBook({"1": ["brady"], "2": ["grace"]}, "1", ocr=ocr_by_tile)
    voice = pd.DataFrame({"room_main": [True], "role": [config.ROLE_ME], "word_bearing": [True],
                          "start_s": [-0.5], "end_s": [2.5]})
    ht, info = strip.scan_class(7, frames, book, voice=voice)
    me_rows = ht[ht["ncc_me"] > 0]
    assert set(me_rows["tile_idx"]) == {3} and me_rows["pts_s"].nunique() == len(frames)
    assert info["valid_share"] == 1.0 and info["n_anchor_frames"] >= 1
    hf = hands.track(ht, None, det=strip.FORUM_DETECTOR)
    assert hf.loc[hf["t_s"] >= 2.5, "raw_state"].eq("UP").all()
    assert hf.loc[hf["t_s"] < 1.5, "raw_state"].eq("DOWN").all()


# --------------------------------------------------------------------------- split, work dir


def test_split_ids_and_sample_by_term(tmp_path):
    p = tmp_path / "split.csv"
    rows = [{"class_id": i, "course_code": "C", "term": t, "date": f"2024-01-{i % 28 + 1:02d}",
             "split": s}
            for i, (t, s) in enumerate([(t, s) for t in ("Fall 2023", "Spring 2024")
                                        for s in ("train",) * 5 + ("test", "excluded")])]
    pd.DataFrame(rows).to_csv(p, index=False)
    assert len(splits.split_ids("train", path=p)) == 10
    assert len(splits.split_ids(path=p)) == 12               # excluded never returned
    with pytest.raises(ValueError):
        splits.split_ids("vault", path=p)
    s = splits.split_table(p)
    picked = ingest.sample_by_term(s[s["split"] == "train"], 4)
    assert picked["term"].value_counts().to_dict() == {"Fall 2023": 2, "Spring 2024": 2}
    assert ingest.sample_by_term(s[s["split"] == "train"], 4)["class_id"].tolist() == \
        picked["class_id"].tolist()                           # seeded


def test_stage_argvs_pass_classes_and_workers():
    argvs = ingest.stage_argvs([1, 2], ("transcribe", "select", "scan"), workers=3)
    assert argvs[0] == ["transcribe", "--class", "1", "--class", "2", "--workers", "3"]
    assert argvs[1] == ["select"]
    assert argvs[2][-2:] == ["--workers", "3"]


def test_use_work_dir_moves_outputs_not_inputs(tmp_path, monkeypatch):
    saved = {k: getattr(config, k) for k in ("TRANSCRIPTS_DIR", "TABLES_DIR", "CACHE_DIR",
                                             "EMB_CACHE_DIR", "EDU_CACHE_DIR", "LOGS_DIR",
                                             "DROPPED_KEYS_LOG", "OOF_DIR", "RESULTS_DIR",
                                             "FIGURES_DIR", "AUDIT_DIR")}
    monkeypatch.delenv(config.WORK_DIR_ENV, raising=False)
    try:
        config.use_work_dir(tmp_path / "pilot")
        assert config.TABLES_DIR == (tmp_path / "pilot" / "tables").resolve()
        assert config.AUDIT_DIR.name == "audit" and config.RAW_DIR == config.SCRIBE_DIR / "raw"
        assert config.table_path("words").parent == config.TABLES_DIR
        with pytest.raises(ValueError):
            config.use_work_dir(config.DATA_DIR)
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def test_cli_split_selects_classes(monkeypatch):
    from oncue import events

    monkeypatch.setattr(splits, "split_ids", lambda s=None, **k: [5, 6] if s == "test" else [])
    assert events._class_ids(argparse.Namespace(class_ids=None, split="test", phase=None)) == [5, 6]
    assert events._class_ids(argparse.Namespace(class_ids=["9"], split="test", phase=None)) == [9]


def test_pdf_has_summary_then_transcript(tmp_path):
    from oncue import transcript_pdf

    t = {"class": {"id": 1, "title": "CS156 Session 2", "course_code": "CS156",
                   "date": "2026-09-01"},
         "speakers": {"s1": {"name": "Speaker One"}},
         "utterances": [{"start": 61.5, "speaker_id": "s1", "text": "hello"},
                        {"start": 70.0, "speaker_id": None, "text": "unclear"}]}
    s = {"raises": [{"ep_id": 0, "onset_s": 100.0, "onset_retimed_s": 99.8, "dur_s": 12.0,
                     "outcome": "called", "call_latency_s": 4.0}],
         "n_votes": 1, "my_seconds": 42.0, "my_turns": 3, "my_fresh_turns": 2,
         "my_longest_turn_s": 20.0, "my_first_speech_s": 300.0, "n_students_spoke": 9,
         "talk_by_role_s": {"prof": 1800.0, "me": 42.0, "student": 600.0},
         "segments": [{"start_s": 0.0, "title_masked": "Intro", "seg_type": "lecture"}],
         "n_chat": 5, "n_my_chat": 1, "n_polls": 2, "n_my_poll_answers": 2,
         "timing": {"cal_status": "ok", "cal_method": "windowed", "cal_n_windows": 18,
                    "cal_offset_s": 1.23, "resid_med_s": 0.1, "resid_p95_s": 0.4,
                    "cal_qa_pass": True},
         "hand_qc": {"av_lag_s": 0.2, "av_lag_brady_s": 0.25, "coverage": 0.93}, "clips": []}
    out = transcript_pdf.write_pdf(t, s, tmp_path / "x.pdf",
                                   video_url="https://forum.minerva.edu/app/courses/1/sections/2/classes/1")
    assert out.exists() and out.read_bytes()[:4] == b"%PDF"
    assert transcript_pdf.clock(99.8) == "1:39.8" and transcript_pdf.clock(3725) == "1:02:05.0"
    t2 = {"speakers": {"s1": {"name": "Speaker One"}},
          "utterances": [{"start": 90.0, "end": 95.0, "speaker_id": "s1", "text": "a"},
                         {"start": 104.0, "end": 110.0, "speaker_id": "s1", "text": "b"}]}
    assert transcript_pdf.raise_context(92.0, t2).startswith("while Speaker One was speaking")
    assert transcript_pdf.raise_context(99.8, t2) == \
        "in a pause: quiet for 4.8 s, speech resumed 4.2 s later"
    assert transcript_pdf.forum_url({"id": 5, "section": {"id": 7, "course": {"id": 3}}}) == \
        "https://forum.minerva.edu/app/courses/3/sections/7/classes/5"


def test_carry_me_only_through_an_overlay_not_read_as_someone_else():
    f = frame(words={3: "Brady"})
    b = strip.find_band(v_of(f))
    a, z = b.tiles[3]
    f[b.y0:b.y1, a:z] = HAND_BGR                       # the raise overlay fills tile 3
    s = np.ascontiguousarray(f[b.y0:b.y1])
    st = strip.tile_stats(s, b)
    last = (10.0, int(a), int(z))
    hidden = strip.LabelBook({"1": ["brady"], "2": ["grace"]}, "1", ocr=lambda w: [""])
    assert strip.carry_me(st, b, s, hidden, last, 11.0) == 3            # label hidden: mine
    assert strip.carry_me(st, b, s, hidden, last, 14.5) == -1           # too long since read
    other = strip.LabelBook({"1": ["brady"], "2": ["grace"]}, "1", ocr=lambda w: ["Grace"])
    st["word"][3] = np.ones((strip.WORD_H, strip.WORD_W), np.float32)   # a readable label
    st["word"][3][:, 40:] = 0
    assert strip.carry_me(st, b, s, other, last, 11.0) == -1            # a classmate's raise
    st2 = strip.tile_stats(np.ascontiguousarray(frame()[b.y0:b.y1]), b)  # no overlay at all
    assert strip.carry_me(st2, b, s, hidden, last, 11.0) == -1


def test_strict_book_needs_a_second_read_before_a_label_is_mine():
    reads = iter([["Brady"], ["Brady"]])
    book = strip.LabelBook({"1": ["brady"], "2": ["grace"]}, "1", strict=True,
                           ocr=lambda w: next(reads))
    patch = np.zeros((strip.WORD_H, strip.WORD_W), np.float32)
    patch[3:13, 4:60] = 1
    white = np.ones((4, 4), bool)
    assert book.identify(84, patch, white) is None            # first read: pending
    assert len(book.me_patches()[0]) == 0
    assert book.identify(84, patch, white) == "1"              # read as mine again: confirmed
    assert len(book.me_patches()[0]) == 1
    # a one-off misread that looks nothing like his confirmed label stays pending
    stray = np.zeros_like(patch)
    stray[:, 70:] = 1
    once = strip.LabelBook({"1": ["brady"], "2": ["grace"]}, "1", strict=True,
                           ocr=lambda w: ["Brady"] if once.n_ocr <= 1 else [""])
    once.widths, once.patches, once.uids, once.tries, once.confirmed = [84], [patch], ["1"], [1], [True]
    assert once.identify(84, stray, white) is None
    assert len(once.me_patches()[0]) == 1
