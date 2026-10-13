"""oncue.pcw: Quill parsing, class-level status, blocks, chunks and git as-of (§5.6).
Synthetic data only; git runs through a fake runner, or read-only (rev-list) on this repo."""

from __future__ import annotations

import copy
import itertools
import json
import subprocess
import types
from datetime import UTC, datetime

import pandas as pd
import pytest
from oncue import config, pcw, schemas, synthetic

START = "2026-09-09T14:00:00.000Z"


def _status(sc, **kw):
    r = sc.raw
    return pcw.pcw_status(r["pcw_meta"], r["class"]["start-time"], doc=r["pcw_doc"],
                          template=r["pcw_template"], **kw)


# --------------------------------------------------------------------------- text extraction


def test_quill_to_text_shapes():
    delta = {"ops": [{"insert": "Explain "}, {"insert": "bias", "attributes": {"bold": True}},
                     {"insert": {"image": "https://example.invalid/x.png"}},
                     {"insert": " and "}, {"insert": {"formula": "x^2"}}, {"insert": "\n"}]}
    assert pcw.quill_to_text(delta) == "Explain bias and x^2"
    assert pcw.quill_to_text(json.dumps(delta)) == "Explain bias and x^2"
    assert pcw.quill_to_text(delta["ops"]) == "Explain bias and x^2"
    assert pcw.quill_to_text("plain   text\n\nhere ") == "plain text\nhere"
    assert pcw.quill_to_text("{not json") == "{not json"
    assert pcw.quill_to_text(None) == "" and pcw.quill_to_text({"ops": []}) == ""


def test_html_code_filename_and_notebook_text():
    assert pcw.html_to_text("<p>Read the <b>chapter</b> &amp; answer.</p><p>Then</p>") \
        == "Read the chapter & answer.\nThen"
    src = ('import numpy as np\ndef fitModel_v2(trainData):\n    # estimate the bias first\n'
           '    label = "zqx secret string"\n    return trainData\n')
    out = pcw.code_text(src)
    assert out.startswith("estimate the bias first")
    assert out.split()[4:] == ["numpy", "np", "fit", "model", "train", "data", "label"]
    assert "zqx" not in out and "def" not in out.split() and "return" not in out.split()
    assert pcw.identifier_words("fitModelV2") == ["fit", "model", "v", "2"]
    assert pcw.filename_words("gradient_notesV2.ipynb") == "gradient notes v 2"
    assert pcw.filename_words("https://example.invalid/a_b.py?sig=1") == "a b"
    nb = {"cells": [{"cell_type": "markdown", "source": ["# My answer\n", "Bias matters."]},
                    {"cell_type": "code", "source": "x_train = 1  # split data"}]}
    assert pcw.notebook_text(json.dumps(nb)) == "# My answer\nBias matters.\nsplit data train"


# --------------------------------------------------------------------------- status


def test_status_on_synthetic_world(world):
    for sc in world:
        st = _status(sc)
        if sc.meta["pcw_status"] == "verified":
            assert st == "verified"
        elif sc.meta["pcw_status"] == "none":
            assert st == "none"
        else:   # edited after start; its doc was also created after start -> none (§5.6)
            doc = sc.raw["pcw_doc"]
            created_late = pcw.to_utc(doc["created_at"]) > pcw.to_utc(sc.raw["class"]["start-time"])
            assert st == ("none" if created_late else "unknown")
            assert _status(sc, trust_created_at=False) == "unknown"


def test_status_branches(synth_class):
    r = synth_class.raw
    meta, doc, tmpl = r["pcw_meta"], r["pcw_doc"], r["pcw_template"]
    start = r["class"]["start-time"]
    assert pcw.pcw_status(meta, start, doc=doc, template=tmpl) == "verified"
    assert pcw.pcw_status(meta, start, doc=doc, template=tmpl, reads_verified=False) == "unknown"
    empty = {**doc, "answers": {}}
    assert pcw.pcw_status(meta, start, doc=empty, template=tmpl) == "empty"
    assert pcw.pcw_status(meta, start, doc=doc, template=tmpl, asof_doc=doc) == "asof_start"
    assert pcw.pcw_status(meta, start, doc=doc, template=tmpl, asof_doc=empty) == "empty"
    assert pcw.pcw_status({**meta, "created": True}, start, doc=empty, template=tmpl) == "empty"
    bad = {**meta, "doc_updated_at": "not a date"}
    assert pcw.pcw_status(bad, start, doc={**doc, "updated_at": None}, template=tmpl) == "unknown"
    late = {**meta, "doc_updated_at": "2026-09-09T15:00:00Z"}
    assert pcw.pcw_status(late, start, doc=doc, template=tmpl) == "unknown"
    assert pcw.pcw_status(None, start) == "none"
    assert pcw.pcw_status({"template_id": "t"}, start) == "none"          # no doc
    assert pcw.pcw_status(None, start, git_found=True) == "git"
    assert pcw.pcw_status(None, start, git_found=False) == "none"
    with pytest.raises(ValueError):
        pcw.pcw_status(meta, None, doc=doc, template=tmpl)


def test_status_is_never_a_feature():
    assert not config.feature_allowed("pcw_status")
    assert not config.feature_allowed("pcw_edited_after_start")
    assert "pcw_status" not in schemas.columns("pcw_blocks")


def test_to_utc_formats():
    a = pcw.to_utc("2026-09-09T16:00:00+02:00")
    ms = int(datetime(2026, 9, 9, 14, tzinfo=UTC).timestamp() * 1000)
    assert a == pcw.to_utc("2026-09-09T14:00:00Z") == pcw.to_utc(ms) == pcw.to_utc(ms / 1000)
    assert pcw.to_utc(pd.Timestamp("2026-09-09 14:00")) == a
    assert pcw.to_utc("garbage") is None and pcw.to_utc(None) is None


# --------------------------------------------------------------------------- blocks


def test_blocks_match_synthetic_truth(world):
    for sc in world:
        b = pcw.blocks(sc.raw["pcw_template"], sc.raw["pcw_doc"], sc.class_id)
        schemas.validate(b, "pcw_blocks", strict=True)
        if sc.pcw["template"] is None:
            assert b.empty
            continue
        assert b["block_order"].tolist() == list(range(len(b)))
        for row, prompt, ans, kind in zip(b.itertuples(), sc.pcw["prompts"],
                                          sc.pcw["answers_text"], sc.pcw["answer_kinds"],
                                          strict=True):
            assert prompt in row.prompt_text and row.prompt_text.startswith("Question ")
            if kind == "text":
                assert row.answer_text == ans and not row.has_code and row.n_files == 0
            elif kind == "code":
                assert row.has_code and "estimate the" in row.answer_text
                assert "def" not in row.answer_text.split()
            else:
                assert row.n_files == 1 and row.answer_text.endswith("notes")
            assert row.answer_words == len(row.answer_text.split())
        blob = " ".join(b["prompt_text"]) + " ".join(b["answer_text"])
        assert "example.invalid" not in blob and "Good start" not in blob


def test_blocks_read_only_whitelisted_fields(synth_class):
    tmpl = copy.deepcopy(synth_class.raw["pcw_template"])
    doc = copy.deepcopy(synth_class.raw["pcw_doc"])
    for blk in tmpl["blocks"]:
        blk["feedback"] = "zqxtrap"
        blk["updated_at"] = "zqxtrap"
    for a in doc["answers"].values():
        a["updated_at"] = "zqxtrap"
        a["comments"] = [{"text": "zqxtrap"}]
        for f in a.get("files", []):
            f["url"] = "https://zqxtrap.invalid/x?sig=1"
    doc["comments"] = [{"author": "Ada", "text": "zqxtrap"}]
    b = pcw.blocks(tmpl, doc, synth_class.class_id)
    assert not b.astype(str).apply(lambda c: c.str.contains("zqxtrap")).any().any()
    # kebab-case input gives the same table
    from scribe_core.forum_events import kebab

    pd.testing.assert_frame_equal(b, pcw.blocks(kebab(tmpl), kebab(doc), synth_class.class_id))
    # no doc: prompts only
    nodoc = pcw.blocks(tmpl, None, synth_class.class_id)
    assert (nodoc["answer_text"] == "").all() and (nodoc["answer_words"] == 0).all()
    assert pcw.blocks(None, doc).empty


def test_blocks_masking_hook(synth_class):
    b = pcw.blocks(synth_class.raw["pcw_template"], synth_class.raw["pcw_doc"],
                   mask_fn=lambda s: s.replace("Question", "Q"))
    assert b["prompt_text"].str.startswith("Q ").all()


# --------------------------------------------------------------------------- chunks


def test_chunks_sizes_and_overlap():
    assert pcw.chunks("") == [] and pcw.chunks(None) == []
    assert pcw.chunks("a b c") == ["a b c"]
    words = [f"w{i}" for i in range(300)]
    cs = pcw.chunks(" ".join(words))
    assert all(len(c.split()) <= config.PCW_CHUNK_WORDS for c in cs)
    for a, b in itertools.pairwise(cs):
        assert a.split()[-config.PCW_CHUNK_OVERLAP:] == b.split()[:config.PCW_CHUNK_OVERLAP]
    covered = {w for c in cs for w in c.split()}
    assert covered == set(words)
    assert len(pcw.chunks(" ".join(words[:120]))) == 1
    assert len(pcw.chunks(" ".join(words[:121]))) == 2
    with pytest.raises(ValueError):
        pcw.chunks(" ".join(words), size=10, overlap=10)


def test_answer_chunks_table(synth_class):
    b = pcw.blocks(synth_class.raw["pcw_template"], synth_class.raw["pcw_doc"],
                   synth_class.class_id)
    ch = pcw.answer_chunks(b)
    assert list(ch.columns) == ["class_id", "block_uuid", "block_order", "chunk_idx", "text",
                                "n_words"]
    assert set(ch["block_uuid"]) <= set(b["block_uuid"])
    assert (ch["n_words"] <= config.PCW_CHUNK_WORDS).all()


def test_class_materials_whitelisted(synth_class):
    mats = pcw.class_materials(synth_class.raw["class_edit_page"])
    assert mats["learning_outcomes"] and all("Analyze how" in x for x in mats["learning_outcomes"])
    assert mats["study_guide"][0].startswith("Key terms")
    assert mats["prep"].startswith("Read the chapter")
    blob = json.dumps(mats)
    assert "graded" not in blob and "score" not in blob


# --------------------------------------------------------------------------- git as-of


class FakeGit:
    def __init__(self, sha="abc123", content="{}", show_rc=0):
        self.calls, self.sha, self.content, self.show_rc = [], sha, content, show_rc

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        assert kw.get("check") is False and kw.get("capture_output")
        if "rev-list" in cmd:
            return types.SimpleNamespace(returncode=0, stdout=self.sha + "\n")
        return types.SimpleNamespace(returncode=self.show_rc, stdout=self.content)


def test_git_asof_uses_utc_cutoff_and_show():
    fake = FakeGit(content='{"cells": []}')
    got = pcw.git_asof("PCW/Session 2/pcw.ipynb", "2026-09-09T16:00:00+02:00", repo="/repo",
                       runner=fake)
    assert got == ("abc123", '{"cells": []}')
    rev, show = fake.calls
    assert rev == ["git", "-C", "/repo", "rev-list", "-1",
                   "--before=2026-09-09T14:00:00+00:00", "HEAD"]
    assert show == ["git", "-C", "/repo", "show", "abc123:PCW/Session 2/pcw.ipynb"]
    assert pcw.git_asof("x.ipynb", START, runner=FakeGit(sha="")) is None
    assert pcw.git_asof("x.ipynb", START, runner=FakeGit(show_rc=128)) is None
    for cmd in fake.calls:   # read-only git only
        assert cmd[3] in ("rev-list", "show")


def test_git_asof_real_repo_is_read_only():
    """Against this repo (read-only): a cut-off before any commit gives None."""
    if subprocess.run(["git", "-C", str(config.REPO_DIR), "rev-parse", "HEAD"],
                      capture_output=True, check=False).returncode != 0:
        pytest.skip("no git repository")
    assert pcw.git_asof("README.md", "1990-01-01T00:00:00Z") is None


# --------------------------------------------------------------------------- per class


def test_pcw_for_class_from_disk(world, data_dir, tmp_path):
    log = tmp_path / "dropped_keys.csv"
    statuses, tables = {}, []
    for sc in world:
        st, b = pcw.pcw_for_class(sc.class_id, raw_dir=data_dir["raw"], log_path=log)
        assert st == _status(sc)
        statuses[sc.class_id] = st
        tables.append(b)
        if st == "none":
            assert (b["answer_text"] == "").all()
    table, st2 = pcw.blocks_table(world.class_ids, raw_dir=data_dir["raw"], log_path=False)
    assert st2 == statuses
    schemas.validate(table, "pcw_blocks", strict=True)
    # read through the whitelist: the planted doc traps were dropped (and logged locally)
    dropped = pd.read_csv(log)
    assert dropped["path"].str.contains("comments|files\\[\\]\\.url|updated-at").any()
    assert "zqx" not in " ".join(table["answer_text"])
    assert pcw.usable("verified") and not pcw.usable("unknown") and not pcw.usable(None)


def test_reference_pool_uses_only_usable_answers(world, truth):
    classes = pd.concat([t["classes"] for t in truth.values()], ignore_index=True)
    blocks = pd.concat([pcw.blocks(sc.raw["pcw_template"], sc.raw["pcw_doc"], sc.class_id)
                        for sc in world], ignore_index=True)
    pool = pcw.reference_pool(blocks, classes)
    status = classes.set_index("class_id")["pcw_status"]
    ans = pool[pool["kind"] == "answer"]
    assert set(ans["class_id"]).issubset({c for c, s in status.items() if pcw.usable(s)})
    assert set(pool.loc[pool["kind"] == "prompt", "class_id"]) == set(blocks["class_id"])
    assert pool["chunk_key"].is_unique
    late = [sc.class_id for sc in world if sc.meta["pcw_status"] == "unknown"]
    assert not set(ans["class_id"]) & set(late)


def test_synthetic_world_is_used(synth_class):
    # guard: the fixtures really are the invented world (no real data in these tests)
    assert synth_class.meta["me_id"] == str(synthetic.ME_ID)
