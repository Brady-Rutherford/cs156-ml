"""Rebuild an SLE pre-class workbook as it stood at a given moment from its revision history.

The SLE app stores each workbook as a ShareDB document of type JSON0
(http://sharejs.org/types/JSONv0) and `revs/{doc}` returns every revision: the first one
`{"create": {"type": ..., "data": <initial doc>}, "m": {"ts": ms}}`, then edits
`{"op": [component, ...], "m": {"ts": ms}}`. Components seen in the 339 harvested workbooks
(18 Oct 2026): object insert/delete (`oi`/`od`), list insert/delete/replace (`li`/`ld`),
and the rich-text subtype (`t` = "rich-text", `o` = a Quill delta) for answers typed in text
fields. Also handled: string insert/delete (`si`/`sd`), number add (`na`), list move (`lm`).

`doc_at(revisions, t)` applies every revision with `m.ts <= t`. Replaying all revisions must
give the harvested doc back (`tests/test_sharedb.py`; checked on every workbook by
`pcw.replay_check`), so a replay stopped at class start is the workbook before class.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence


class ReplayError(ValueError):
    """A revision could not be applied (unknown component or a path that does not exist)."""


# --------------------------------------------------------------------------- Quill deltas


def _units(ops) -> list[tuple[object, dict | None]]:
    """A Quill document delta as units: one per character of a string insert, one per embed."""
    out = []
    for op in ops or []:
        if not isinstance(op, dict) or "insert" not in op:
            continue
        ins, attrs = op["insert"], op.get("attributes")
        if isinstance(ins, str):
            out.extend((ch, attrs) for ch in ins)
        else:
            out.append((ins, attrs))
    return out


def _merge_attrs(base: dict | None, change: dict | None) -> dict | None:
    if not change:
        return base
    out = dict(base or {})
    for k, v in change.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    return out or None


def _regroup(units) -> list[dict]:
    """Units back to a compact delta (neighbouring characters with equal attributes join)."""
    ops: list[dict] = []
    for ins, attrs in units:
        if isinstance(ins, str) and ops and isinstance(ops[-1]["insert"], str) \
                and ops[-1].get("attributes") == attrs:
            ops[-1]["insert"] += ins
            continue
        op = {"insert": ins}
        if attrs:
            op["attributes"] = attrs
        ops.append(op)
    return ops


def apply_delta(document, change) -> dict:
    """Apply a Quill change delta (retain / insert / delete) to a document delta."""
    doc_ops = document.get("ops", []) if isinstance(document, dict) else (document or [])
    chg_ops = change.get("ops", change) if isinstance(change, dict) else (change or [])
    src, out, i = _units(doc_ops), [], 0
    for op in chg_ops:
        if not isinstance(op, dict):
            continue
        if "insert" in op:
            ins, attrs = op["insert"], op.get("attributes")
            out.extend(((ch, attrs) for ch in ins) if isinstance(ins, str) else [(ins, attrs)])
        elif "retain" in op:
            n = int(op["retain"])
            for ins, attrs in src[i:i + n]:
                out.append((ins, _merge_attrs(attrs, op.get("attributes"))))
            i += n
        elif "delete" in op:
            i += int(op["delete"])
    out.extend(src[i:])
    return {"ops": _regroup(out)}


def apply_text(s: str, ops) -> str:
    """text0 subtype ops ({"p": offset, "i": str} / {"p": offset, "d": str}) on a string."""
    for op in ops or []:
        if not isinstance(op, dict):
            continue
        pos = int(op.get("p", 0))
        if "d" in op:
            s = s[:pos] + s[pos + len(op["d"]):]
        if "i" in op:
            s = s[:pos] + op["i"] + s[pos:]
    return s


# --------------------------------------------------------------------------- JSON0


def _key(node, key):
    """List positions may arrive as strings ("0", "1") in table answers: use them as ints."""
    if isinstance(node, list) and isinstance(key, str) and key.lstrip("-").isdigit():
        return int(key)
    return key


def _walk(doc, path: Sequence):
    node = doc
    for key in path:
        try:
            node = node[_key(node, key)]
        except (KeyError, IndexError, TypeError) as exc:
            raise ReplayError(f"path {list(path)} does not exist") from exc
    return node


def apply_component(doc, comp: dict):
    """Apply one JSON0 component to `doc` (in place where possible); returns the new doc."""
    p = list(comp.get("p") or [])
    if comp.get("t") in ("text", "text0", "http://sharejs.org/types/textv1"):   # code cells
        parent = _walk(doc, p[:-1])
        key = _key(parent, p[-1])
        s = parent[key] if (isinstance(parent, dict) and key in parent) or (
            isinstance(parent, list) and isinstance(key, int) and key < len(parent)) else ""
        parent[key] = apply_text(s if isinstance(s, str) else "", comp.get("o"))
        return doc
    if "t" in comp:                                     # subtype: rich-text Quill delta
        if comp["t"] not in ("rich-text", "http://sharejs.org/types/rich-text/v1"):
            raise ReplayError(f"unknown subtype {comp['t']!r}")
        if not p:
            return apply_delta(doc, comp.get("o"))
        parent = _walk(doc, p[:-1])
        key = _key(parent, p[-1])
        cur = parent[key] if (isinstance(parent, dict) and key in parent) or (
            isinstance(parent, list) and isinstance(key, int) and key < len(parent)) else None
        if isinstance(parent, list) and isinstance(key, int) and key >= len(parent):
            parent.extend([None] * (key + 1 - len(parent)))
        parent[key] = apply_delta(cur or {"ops": []}, comp.get("o"))
        return doc
    if not p:                                           # whole-document replace
        return copy.deepcopy(comp["oi"]) if "oi" in comp else doc
    parent = _walk(doc, p[:-1])
    key = _key(parent, p[-1])
    if "na" in comp:
        parent[key] = parent[key] + comp["na"]
    elif "si" in comp or "sd" in comp:                  # string ops: key is an offset
        s_parent = _walk(doc, p[:-2])
        s_key = _key(s_parent, p[-2])
        s = s_parent[s_key]
        if "sd" in comp:
            s = s[:key] + s[key + len(comp["sd"]):]
        if "si" in comp:
            s = s[:key] + comp["si"] + s[key:]
        s_parent[s_key] = s
    elif "lm" in comp:
        item = parent.pop(key)
        parent.insert(int(comp["lm"]), item)
    elif "li" in comp or "ld" in comp:
        if not isinstance(parent, list):
            raise ReplayError(f"list op on a non-list at {p}")
        # The app replaces positions of a list it never filled (multiple-choice answers start
        # as []): pad with None, as the stored docs show, instead of failing.
        if "ld" in comp and "li" in comp:
            if key >= len(parent):
                parent.extend([None] * (key + 1 - len(parent)))
            parent[key] = copy.deepcopy(comp["li"])
        elif "ld" in comp:
            if key < len(parent):
                del parent[key]
        else:
            parent.insert(key, copy.deepcopy(comp["li"]))
    elif "oi" in comp or "od" in comp:
        if not isinstance(parent, dict):
            raise ReplayError(f"object op on a non-object at {p}")
        if "oi" in comp:
            parent[key] = copy.deepcopy(comp["oi"])
        else:
            parent.pop(key, None)
    else:
        raise ReplayError(f"unknown component {sorted(comp)}")
    return doc


def _ts(rev) -> float | None:
    m = rev.get("m") if isinstance(rev, dict) else None
    ts = m.get("ts") if isinstance(m, dict) else None
    return float(ts) if isinstance(ts, (int, float)) else None


def doc_at(revisions: Sequence[dict], t_ms: float | None = None):
    """The doc after every revision with `m.ts <= t_ms` (all of them when `t_ms` is None).
    None when the doc did not exist yet at `t_ms`."""
    doc, created = None, False
    for rev in revisions or []:
        ts = _ts(rev)
        if t_ms is not None and ts is not None and ts > t_ms:
            break
        if "create" in rev:
            doc, created = copy.deepcopy((rev["create"] or {}).get("data")), True
            continue
        if rev.get("del"):
            doc, created = None, False
            continue
        if not created:
            continue
        for comp in rev.get("op") or []:
            doc = apply_component(doc, comp)
    return doc
