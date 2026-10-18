"""Replay of SLE workbook revisions (oncue.sharedb): Quill deltas, JSON0 components, the
padding the app relies on, and the class-start cutoff."""

from __future__ import annotations

import pytest

from oncue import sharedb


def test_quill_delta_insert_retain_delete_and_attributes():
    doc = {"ops": [{"insert": "Hello world\n"}]}
    out = sharedb.apply_delta(doc, {"ops": [{"retain": 6}, {"delete": 5}, {"insert": "there"}]})
    assert out == {"ops": [{"insert": "Hello there\n"}]}
    bold = sharedb.apply_delta(out, [{"retain": 5, "attributes": {"bold": True}}])
    assert bold["ops"][0] == {"insert": "Hello", "attributes": {"bold": True}}
    assert sharedb.apply_delta(None, [{"insert": "x"}]) == {"ops": [{"insert": "x"}]}


def test_json0_components_and_string_list_positions():
    doc = {"a": [[{"v": 1}]], "b": "abc"}
    doc = sharedb.apply_component(doc, {"p": ["c"], "oi": {"ops": []}})
    doc = sharedb.apply_component(doc, {"p": ["a", "0", "1"], "li": {"v": 2}})   # "0" is a list index
    assert doc["a"][0] == [{"v": 1}, {"v": 2}]
    doc = sharedb.apply_component(doc, {"p": ["a", 0, 0], "ld": {"v": 1}})
    assert doc["a"][0] == [{"v": 2}]
    doc = sharedb.apply_component(doc, {"p": ["c"], "t": "rich-text", "o": [{"insert": "hi"}]})
    assert doc["c"] == {"ops": [{"insert": "hi"}]}
    doc = sharedb.apply_component(doc, {"p": ["c"], "od": {"ops": []}})
    assert "c" not in doc
    doc = sharedb.apply_component(doc, {"p": ["b", 1], "si": "X"})
    assert doc["b"] == "aXbc"


def test_unfilled_list_positions_are_padded_like_the_app():
    doc = {"choices": []}
    doc = sharedb.apply_component(doc, {"p": ["choices", 2], "ld": None, "li": 2})
    assert doc["choices"] == [None, None, 2]
    doc = sharedb.apply_component(doc, {"p": ["choices", 9], "ld": 5})     # nothing to delete
    assert doc["choices"] == [None, None, 2]


def test_text_subtype_for_code_cells():
    doc = {"cell": {"code": "print(1)"}}
    doc = sharedb.apply_component(doc, {"p": ["cell", "code"], "t": "text",
                                        "o": [{"p": 6, "d": "1"}, {"p": 6, "i": "42"}]})
    assert doc["cell"]["code"] == "print(42)"


def test_doc_at_stops_at_the_cutoff_and_unknown_components_fail():
    revs = [{"create": {"type": "json0", "data": {"x": {"ops": []}}}, "m": {"ts": 1000}},
            {"op": [{"p": ["x"], "t": "rich-text", "o": [{"insert": "before"}]}], "m": {"ts": 2000}},
            {"op": [{"p": ["x"], "t": "rich-text", "o": [{"retain": 6}, {"insert": " after"}]}],
             "m": {"ts": 3000}}]
    assert sharedb.doc_at(revs, 500) is None                      # did not exist yet
    assert sharedb.doc_at(revs, 2500) == {"x": {"ops": [{"insert": "before"}]}}
    assert sharedb.doc_at(revs) == {"x": {"ops": [{"insert": "before after"}]}}
    with pytest.raises(sharedb.ReplayError):
        sharedb.apply_component({"x": 1}, {"p": ["x"], "zz": 1})
