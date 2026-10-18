"""Pre-class work (PCW): Quill parsing, class-level status, blocks, chunks, git as-of (§5.6).

* Prompts Q_b are the concatenated Quill `insert` strings of each template block.
* Answers A_b come from the doc, keyed by block uuid, from whitelisted paths only
  (`answers.*.type`, `answers.*.value`, `answers.*.files[].name|filename`): text, code
  comments and identifiers (split on `_` and camelCase), and upload FILENAMES.
* `pcw_status` is class-level QA decided before modelling. It is never a feature value and
  never stored outside `classes.pcw_status`; an edit after class start is future information.

Nothing here touches the network: harvesting is `harvest.py`'s job (GET-only, cached once).
Raw dicts may use snake_case or kebab-case keys; everything is read after `kebab()`.
"""

from __future__ import annotations

import html
import json
import keyword
import math
import re
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import config, schemas

STATUS_ASOF, STATUS_VERIFIED, STATUS_GIT = "asof_start", "verified", "git"
STATUS_EMPTY, STATUS_NONE, STATUS_UNKNOWN = "empty", "none", "unknown"


def _kebab(obj):
    from scribe_core.forum_events import kebab

    return kebab(obj)


# --------------------------------------------------------------------------- text extraction


def _clean(text: str) -> str:
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in str(text).splitlines()]
    return "\n".join(ln for ln in lines if ln)


def quill_to_text(ops) -> str:
    """Plain text of a Quill delta: a JSON string, {"ops": [...]}, or a list of ops.

    String inserts are concatenated; embeds are skipped except formulas. A non-JSON string
    is returned as text (whitespace tidied).
    """
    if ops is None:
        return ""
    if isinstance(ops, str):
        s = ops.strip()
        if not s.startswith(("{", "[")):
            return _clean(s)
        try:
            ops = json.loads(s)
        except ValueError:
            return _clean(s)
    if isinstance(ops, dict):
        ops = ops.get("ops", [])
    parts: list[str] = []
    for op in ops if isinstance(ops, list) else []:
        if isinstance(op, str):
            parts.append(op)
        elif isinstance(op, dict):
            ins = op.get("insert")
            if isinstance(ins, str):
                parts.append(ins)
            elif isinstance(ins, dict) and isinstance(ins.get("formula"), str):
                parts.append(ins["formula"])
    return _clean("".join(parts))


def html_to_text(s) -> str:
    """Tags stripped and entities unescaped (class_edit_page descriptions are HTML)."""
    if not s:
        return ""
    s = re.sub(r"<\s*(br|/p|/li|/h\d|/div)\s*/?>", "\n", str(s), flags=re.IGNORECASE)
    return _clean(html.unescape(re.sub(r"<[^>]+>", " ", s)))


_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STRING_LIT = re.compile(r"(\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"(?:\\.|[^\"\\])*\""
                         r"|'(?:\\.|[^'\\])*')")
_SKIP_IDENTS = frozenset(keyword.kwlist) | {"self", "cls", "print", "return", "none", "true",
                                            "false"}


def identifier_words(name: str) -> list[str]:
    """`fit_modelV2` -> ["fit", "model", "v", "2"]: split on `_` and camelCase, lowercase."""
    out = []
    for part in str(name).split("_"):
        out += [w.lower() for w in _CAMEL.findall(part)]
    return out


def code_text(src: str | None) -> str:
    """Comments and identifier words of a code answer (no string literals, no keywords)."""
    if not src:
        return ""
    src = str(src)
    comments = [c.strip() for c in re.findall(r"#[ \t]*(.*)", src) if c.strip()]
    bare = re.sub(r"#.*", " ", _STRING_LIT.sub(" ", src))
    words, seen = [], set()
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", bare):
        if ident in _SKIP_IDENTS or ident.lower() in _SKIP_IDENTS:
            continue
        for w in identifier_words(ident):
            if len(w) >= 2 and w not in seen:
                seen.add(w)
                words.append(w)
    return " ".join(comments + words).strip()


def filename_words(name: str | None) -> str:
    """Upload FILENAME (never its content or URL) -> words of its stem."""
    if not name:
        return ""
    stem = Path(str(name).split("?")[0]).name
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return " ".join(w for part in re.split(r"[\s\-.]+", stem) for w in identifier_words(part))


def notebook_text(nb) -> str:
    """Markdown text plus code comments/identifiers of an .ipynb (dict or JSON string)."""
    if isinstance(nb, str | bytes):
        nb = json.loads(nb)
    parts = []
    for cell in (nb or {}).get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if cell.get("cell_type") == "markdown":
            parts.append(_clean(src))
        elif cell.get("cell_type") == "code":
            parts.append(code_text(src))
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- answers and status


def _answers(doc) -> dict:
    doc = _kebab(doc or {})
    ans = doc.get("answers") if isinstance(doc, dict) else None
    return ans if isinstance(ans, dict) else {}


def answer_fields(answer) -> dict:
    """Whitelisted fields of one doc answer: {"type", "text", "has_code", "n_files"}.

    Reads only `type`, `value` and `files[].name|filename` (per-block timestamps, file URLs
    and comments are never touched).
    """
    a = answer if isinstance(answer, dict) else {}
    kind = str(a.get("type") or "")
    value = a.get("value")
    files = a.get("files") if isinstance(a.get("files"), list) else []
    names = [f.get("name") or f.get("filename") for f in files if isinstance(f, dict)]
    names = [n for n in names if n]
    if kind == "code":
        text = code_text(value if isinstance(value, str) else quill_to_text(value))
    elif value is not None:
        text = quill_to_text(value)
    else:
        text = ""
    if names:
        text = "\n".join(x for x in [text] + [filename_words(n) for n in names] if x)
    return {"type": kind, "text": text, "has_code": kind == "code" and bool(value),
            "n_files": len(names)}


def has_answered_block(doc) -> bool:
    """True when at least one answer has text, code or a file."""
    return any(answer_fields(a)["text"] or answer_fields(a)["n_files"]
               for a in _answers(doc).values())


def to_utc(ts) -> datetime | None:
    """ISO-8601 string (with Z or offset), epoch ms/s number, datetime or Timestamp -> UTC."""
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    if isinstance(ts, int | float):
        v = float(ts)
        return datetime.fromtimestamp(v / 1000.0 if v > 1e11 else v, tz=UTC)
    try:
        dt = datetime.fromisoformat(str(ts).strip())
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def pcw_status(meta, class_start, *, doc=None, template=None, asof_doc=None,
               reads_verified: bool = True, git_found: bool | None = None,
               trust_created_at: bool = True) -> str:
    """Class-level PCW status (§5.6 table). QA only: NEVER a feature value.

    * `git_found` (CS156 route): "git" if a commit before class start holds the notebook,
      else "none";
    * no template, or no doc (no `doc-id` in meta and no doc) -> "none";
    * `asof_doc` (rebuilt as of class start from a history endpoint, Branch H) ->
      "asof_start", or "empty" if it has no answered block;
    * a harvest-created doc (`meta.created`) -> "empty" (no answers) else "unknown";
    * doc `created-at` after class start -> "none" (no doc existed before class start);
    * edit time unreadable, after class start, or reads not verified side-effect free by D8
      (`reads_verified=False`) -> "unknown";
    * otherwise "verified", or "empty" if no block is answered.
    """
    if git_found is not None:
        return STATUS_GIT if git_found else STATUS_NONE
    start = to_utc(class_start)
    if start is None:
        raise ValueError("pcw_status needs the class start time")
    meta = _kebab(meta or {})
    doc_k = _kebab(doc) if doc is not None else None
    tmpl_k = _kebab(template) if template is not None else None
    if tmpl_k is None and not meta.get("template-id"):
        return STATUS_NONE
    if doc_k is None and not meta.get("doc-id"):
        return STATUS_NONE
    if asof_doc is not None:
        return STATUS_ASOF if has_answered_block(asof_doc) else STATUS_EMPTY
    answered = has_answered_block(doc_k) if doc_k is not None else False
    if meta.get("created"):
        return STATUS_EMPTY if not answered else STATUS_UNKNOWN
    if trust_created_at and doc_k is not None:
        created = to_utc(doc_k.get("created-at"))
        if created is not None and created > start:
            return STATUS_NONE
    updated = to_utc(meta.get("doc-updated-at")
                     or (doc_k or {}).get("updated-at") or (doc_k or {}).get("modified-at"))
    if updated is None or updated > start or not reads_verified:
        return STATUS_UNKNOWN
    return STATUS_VERIFIED if answered else STATUS_EMPTY


# --------------------------------------------------------------------------- blocks and chunks


def blocks(template, doc, class_id: int = 0, *, mask_fn: Callable[[str], str] | None = None
           ) -> pd.DataFrame:
    """`pcw_blocks` rows for one class: prompt (template) and answer (doc) per block.

    Blocks are ordered by `order` (else template position). `doc=None` gives empty answers.
    `mask_fn` (e.g. a roster mask) is applied to prompt and answer text when given.
    """
    if template is None:
        return schemas.empty("pcw_blocks")
    tmpl = _kebab(template)
    raw_blocks = [b for b in (tmpl.get("blocks") or []) if isinstance(b, dict)]
    ordered = sorted(enumerate(raw_blocks),
                     key=lambda ib: (ib[1].get("order") if isinstance(ib[1].get("order"), int)
                                     else ib[0], ib[0]))
    answers = _answers(doc)
    rows = []
    for pos, (_, b) in enumerate(ordered):
        uuid = str(b.get("id") or b.get("uuid") or "")
        comps = [c for c in (b.get("block-components") or []) if isinstance(c, dict)]
        prompt = "\n".join(t for t in (quill_to_text(c.get("value")) for c in comps) if t)
        a = answer_fields(answers.get(uuid))
        answer = a["text"]
        if mask_fn is not None:
            prompt, answer = mask_fn(prompt), mask_fn(answer)
        rows.append({"class_id": class_id, "block_uuid": uuid, "block_order": pos,
                     "prompt_text": prompt, "answer_text": answer,
                     "answer_words": len(answer.split()), "has_code": a["has_code"],
                     "n_files": a["n_files"]})
    if not rows:
        return schemas.empty("pcw_blocks")
    return schemas.coerce(pd.DataFrame(rows), "pcw_blocks")


def chunks(text: str | None, size: int = config.PCW_CHUNK_WORDS,
           overlap: int = config.PCW_CHUNK_OVERLAP) -> list[str]:
    """Word chunks of <= `size` words with `overlap` words shared by neighbours."""
    words = str(text or "").split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than size")
    return [" ".join(words[s:s + size]) for s in range(0, len(words) - overlap, step)]


def answer_chunks(pcw_blocks: pd.DataFrame, *, size: int = config.PCW_CHUNK_WORDS,
                  overlap: int = config.PCW_CHUNK_OVERLAP) -> pd.DataFrame:
    """Answer chunks a_j: class_id, block_uuid, block_order, chunk_idx, text, n_words."""
    rows = []
    for r in pcw_blocks.itertuples(index=False):
        for j, c in enumerate(chunks(r.answer_text, size, overlap)):
            rows.append({"class_id": r.class_id, "block_uuid": r.block_uuid,
                         "block_order": r.block_order, "chunk_idx": j, "text": c,
                         "n_words": len(c.split())})
    return pd.DataFrame(rows, columns=["class_id", "block_uuid", "block_order", "chunk_idx",
                                       "text", "n_words"])


def class_materials(class_edit_page) -> dict:
    """Pre-class instructions (A12) from whitelisted class_edit_page paths only:
    {"learning_outcomes": [...], "study_guide": [paragraphs], "prep": str}."""
    page = _kebab(class_edit_page or {})
    los = []
    for fo in ((page.get("assessment") or {}).get("focused-outcomes") or []):
        lo = (fo or {}).get("learning-outcome") or {}
        text = " ".join(x for x in (html_to_text(lo.get("name")),
                                    html_to_text(lo.get("description"))) if x)
        if text:
            los.append(text)
    sg = page.get("study-guide")
    sg_text = html_to_text(sg.get("description") if isinstance(sg, dict) else sg)
    prep = html_to_text((page.get("preparatory-assignment") or {}).get("description"))
    guide = [p for p in sg_text.split("\n") if p.strip()]
    if not guide and prep:            # this Forum version keeps the study guide as the prep
        guide = [p for p in prep.split("\n") if p.strip()]
    return {"learning_outcomes": los, "study_guide": guide, "prep": prep}


# --------------------------------------------------------------------------- git as-of (CS156)


def git_asof(path: str | Path, start, *, repo: str | Path = config.REPO_DIR,
             runner: Callable | None = None) -> tuple[str, str] | None:
    """(sha, content) of `path` as of the last commit before `start` (UTC), else None.

    Runs `git rev-list -1 --before=<start UTC> HEAD` then `git show <sha>:<path>`.
    Read-only; `runner` (subprocess.run-compatible) is injectable for tests.
    """
    run = runner or subprocess.run
    when = to_utc(start)
    if when is None:
        raise ValueError("git_asof needs a start time")
    stamp = when.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    rev = run(["git", "-C", str(repo), "rev-list", "-1", f"--before={stamp}", "HEAD"],
              capture_output=True, text=True, check=False)
    sha = (rev.stdout or "").strip()
    if rev.returncode != 0 or not sha:
        return None
    rel = Path(path).as_posix()
    shown = run(["git", "-C", str(repo), "show", f"{sha}:{rel}"],
                capture_output=True, text=True, check=False)
    if shown.returncode != 0:
        return None
    return sha, shown.stdout


# --------------------------------------------------------------------------- per class


def _read_raw(class_id, source: str, raw_dir: Path | None, log_path=None):
    """A raw source for one class through `loaders.load_raw` (per-source JSON-path
    whitelist, dropped keys logged); plain JSON + kebab() only if loaders is absent."""
    try:
        from . import loaders
    except ModuleNotFoundError as exc:
        if exc.name != f"{__package__}.loaders":
            raise
        loaders = None
    if loaders is not None:
        return loaders.load_raw(class_id, source, raw_dir=raw_dir, log_path=log_path)
    path = Path(raw_dir or config.RAW_DIR) / str(class_id) / config.RAW_SOURCE_FILES[source]
    if not path.exists():
        return None
    return _kebab(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- SLE workbooks (D8)
# The real SLE format (harvested 18 Oct 2026): template blocks {type, uuid, block_components:
# [{type, uuid, value}]}; prompts are the "Text" components (a Quill delta as a JSON string);
# answers live in the doc keyed by the uuid of the block's answer component. These files hold
# only Brady's own workbook, so they are read directly (the raw whitelist was written for an
# earlier guess of the format) and only the fields below are used.

ANSWER_COMPONENTS = ("RichTextField", "CodeField", "ChoicesField", "FileField", "TableField")


def _raw_json(class_id, name: str, raw_dir: Path | None):
    path = Path(raw_dir or config.RAW_DIR) / str(class_id) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sle_answer(kind: str, value, starter=None) -> dict:
    """{"text", "has_code", "n_files", "answered"} of one answer component."""
    out = {"text": "", "has_code": False, "n_files": 0, "answered": False}
    if kind == "RichTextField" and isinstance(value, dict):
        out["text"] = quill_to_text(value)
    elif kind == "CodeField":
        code = value.get("code") if isinstance(value, dict) else value
        if isinstance(code, str) and code.strip() and code.strip() != str(starter or "").strip():
            out["text"], out["has_code"] = code_text(code), True
    elif kind == "ChoicesField" and isinstance(value, list):
        out["answered"] = any(v is not None for v in value)
    elif kind == "FileField" and isinstance(value, list):
        names = [f.get("name") or f.get("filename") for f in value if isinstance(f, dict)]
        names = [n for n in names if n]
        out["text"], out["n_files"] = " ".join(filename_words(n) for n in names), len(names)
    elif kind == "TableField" and isinstance(value, list):
        cells = [c for row in value if isinstance(row, list) for c in row if isinstance(c, dict)]
        out["text"] = "\n".join(t for t in (quill_to_text(c.get("value")) for c in cells
                                             if not c.get("readOnly")) if t)
    out["answered"] = out["answered"] or bool(out["text"]) or out["n_files"] > 0
    return out


def sle_blocks(template, doc, class_id: int = 0) -> pd.DataFrame:
    """`pcw_blocks` rows for one class from the real SLE format (doc may be None)."""
    rows = []
    for pos, b in enumerate((template or {}).get("blocks") or []):
        comps = [c for c in (b.get("block_components") or []) if isinstance(c, dict)]
        prompt = "\n".join(t for t in (quill_to_text(c.get("value")) for c in comps
                                        if c.get("type") == "Text") if t)
        ans = {"text": "", "has_code": False, "n_files": 0}
        for c in comps:
            if c.get("type") in ANSWER_COMPONENTS and isinstance(doc, dict):
                a = sle_answer(c["type"], doc.get(c.get("uuid")), c.get("value"))
                ans = {"text": "\n".join(x for x in (ans["text"], a["text"]) if x),
                       "has_code": ans["has_code"] or a["has_code"],
                       "n_files": ans["n_files"] + a["n_files"]}
        rows.append({"class_id": class_id, "block_uuid": str(b.get("uuid") or ""),
                     "block_order": pos, "prompt_text": prompt, "answer_text": ans["text"],
                     "answer_words": len(ans["text"].split()), "has_code": ans["has_code"],
                     "n_files": ans["n_files"]})
    if not rows:
        return schemas.empty("pcw_blocks")
    return schemas.coerce(pd.DataFrame(rows), "pcw_blocks")


def _class_start(class_id, raw_dir: Path | None):
    """Scheduled start (UTC): the class's start-time, else the listing's full timestamp."""
    klass = _raw_json(class_id, "class", raw_dir) or {}
    listing = (_raw_json(class_id, "listing", raw_dir) or {}).get("class") or {}
    ts = klass.get("start-time") or (listing.get("date") if "T" in str(listing.get("date") or "")
                                     else None)
    return to_utc(ts)


def sle_for_class(class_id, *, raw_dir: Path | str | None = None) -> tuple[str, pd.DataFrame]:
    """(pcw_status, pcw_blocks) with the doc rebuilt AS OF CLASS START from its revisions:
    "asof_start" (answers before class), "empty" (no answer yet at class start, or a doc the
    harvest created), "none" (no workbook, or none existed before class), "unknown" (no
    start time or the replay failed: answers are then left out)."""
    from .sharedb import ReplayError, doc_at

    raw_dir = Path(raw_dir) if raw_dir is not None else None
    tmpl = _raw_json(class_id, "pcw_template", raw_dir)
    meta = _raw_json(class_id, "pcw_meta", raw_dir) or {}
    if not tmpl:
        return STATUS_NONE, schemas.empty("pcw_blocks")
    prompts_only = sle_blocks(tmpl, None, int(class_id))
    start = _class_start(class_id, raw_dir)
    if meta.get("created"):
        return STATUS_EMPTY, prompts_only
    revs = meta.get("history")
    if start is None or not isinstance(revs, list) or not revs:
        return STATUS_UNKNOWN, prompts_only
    try:
        asof = doc_at(revs, start.timestamp() * 1000)
    except (ReplayError, IndexError, KeyError, TypeError, ValueError):
        return STATUS_UNKNOWN, prompts_only
    if asof is None:
        return STATUS_NONE, prompts_only
    b = sle_blocks(tmpl, asof, int(class_id))
    return (STATUS_ASOF if (b["answer_words"].sum() > 0 or b["n_files"].sum() > 0)
            else STATUS_EMPTY), b


def pcw_for_class(class_id, *, raw_dir: Path | str | None = None, class_start=None,
                  reads_verified: bool = True, log_path=None) -> tuple[str, pd.DataFrame]:
    """(pcw_status, pcw_blocks) for one class from raw/{cid}/pcw_*.json and class.json.

    Files are read through `loaders.load_raw` (whitelisted paths only; `log_path` as there:
    None = data/logs/dropped_keys.csv, False = no log). `class_start` defaults to the
    class's scheduled `start-time`. Blocks are returned for
    every status (prompts are known before class); answers feed features only when the
    status is in `config.PCW_USABLE_STATUSES` (the features stage enforces that).
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else None
    tmpl = _read_raw(class_id, "pcw_template", raw_dir, log_path)
    doc = _read_raw(class_id, "pcw_doc", raw_dir, log_path)
    meta = _read_raw(class_id, "pcw_meta", raw_dir, log_path)
    if class_start is None:
        klass = _read_raw(class_id, "class", raw_dir, log_path) or {}
        class_start = klass.get("start-time")
    status = pcw_status(meta, class_start, doc=doc, template=tmpl,
                        reads_verified=reads_verified)
    if status == STATUS_NONE:
        doc = None
    return status, blocks(tmpl, doc, int(class_id))


def usable(status: str | None) -> bool:
    """Answers may feed `pcwA` only for these statuses (as-of-start, verified, git)."""
    return status in config.PCW_USABLE_STATUSES


def reference_pool(pcw_blocks: pd.DataFrame, classes: pd.DataFrame, *,
                   size: int = config.PCW_CHUNK_WORDS,
                   overlap: int = config.PCW_CHUNK_OVERLAP) -> pd.DataFrame:
    """Chunks for the z-null (§5.6): prompts of every class with a template, plus answer
    chunks of classes whose `pcw_status` is usable. Columns: chunk_key, class_id,
    course_code, class_start (UTC ISO), kind ("prompt" | "answer"), text."""
    meta = classes.set_index("class_id")
    rows = []
    for cid, g in pcw_blocks.groupby("class_id", sort=True):
        if cid not in meta.index:
            continue
        m = meta.loc[cid]
        start = to_utc(m.get("sched_start_utc"))
        base = {"class_id": cid, "course_code": m.get("course_code"),
                "class_start": start.isoformat() if start else None}
        ok = usable(m.get("pcw_status"))
        for r in g.itertuples(index=False):
            for j, c in enumerate(chunks(r.prompt_text, size, overlap)):
                rows.append({**base, "chunk_key": f"{cid}:{r.block_uuid}:prompt:{j}",
                             "kind": "prompt", "text": c})
            if ok:
                for j, c in enumerate(chunks(r.answer_text, size, overlap)):
                    rows.append({**base, "chunk_key": f"{cid}:{r.block_uuid}:answer:{j}",
                                 "kind": "answer", "text": c})
    cols = ["chunk_key", "class_id", "course_code", "class_start", "kind", "text"]
    return pd.DataFrame(rows, columns=cols)


def blocks_table(class_ids: Sequence, *, raw_dir: Path | str | None = None,
                 reads_verified: bool = True, log_path=None) -> tuple[pd.DataFrame, dict]:
    """`pcw_blocks` for many classes plus {class_id: pcw_status} (status goes to classes)."""
    parts, status = [], {}
    rd = Path(raw_dir) if raw_dir is not None else None
    for cid in class_ids:
        meta = _raw_json(cid, "pcw_meta", rd)
        if meta is None:                                        # never harvested
            s, b = STATUS_NONE, schemas.empty("pcw_blocks")
        elif meta.get("history") is not None or not meta.get("template_id"):
            s, b = sle_for_class(cid, raw_dir=raw_dir)         # real SLE format (D8)
        else:
            s, b = pcw_for_class(cid, raw_dir=raw_dir, reads_verified=reads_verified,
                                 log_path=log_path, class_start=_class_start(cid, rd))
        status[cid] = s
        if len(b):
            parts.append(b)
    table = pd.concat(parts, ignore_index=True) if parts else schemas.empty("pcw_blocks")
    return schemas.coerce(table, "pcw_blocks"), status


# --------------------------------------------------------------------------- CLI

STATUS_TABLE = "pcw_status"     # local: class_id, pcw_status, template_asof (merged by `qa`)


def status_frame(status: dict, blocks_df: pd.DataFrame) -> pd.DataFrame:
    """One row per class: pcw_status plus template_asof ("harvest": the template version
    current at harvest time, §5.6 caveat; <NA> without a template)."""
    with_tmpl = set(blocks_df["class_id"].astype(int)) if len(blocks_df) else set()
    rows = [{"class_id": int(c), "pcw_status": s,
             "template_asof": "harvest" if int(c) in with_tmpl else None}
            for c, s in status.items()]
    df = pd.DataFrame(rows, columns=["class_id", "pcw_status", "template_asof"])
    return df.astype({"class_id": "int64", "pcw_status": "string", "template_asof": "string"})


def _cli_class_ids(args, raw_dir: Path) -> list[int]:
    if getattr(args, "class_ids", None):
        return [int(c) for c in args.class_ids]
    cpath = config.table_path("classes")
    if cpath.exists():
        classes = pd.read_parquet(cpath, columns=["class_id", "phase"])
        if getattr(args, "phase", None):
            classes = classes[classes["phase"] == args.phase]
        return sorted(int(c) for c in classes["class_id"])
    return sorted(int(p.name) for p in raw_dir.iterdir()
                  if p.is_dir() and p.name.isdigit()) if raw_dir.exists() else []


def cli(args) -> int:
    """`python -m oncue pcw [--class CID ...] [--reads-unverified]`: pcw_blocks.parquet and
    the class-level pcw_status.parquet (QA only; `oncue qa` merges it into classes)."""
    import argparse

    from .events import write_table

    p = argparse.ArgumentParser(prog="oncue pcw")
    p.add_argument("--reads-unverified", action="store_true",
                   help="D8 found that reads bump last-modified: every doc is 'unknown'")
    opts = p.parse_args(getattr(args, "extra", None) or [])
    raw_dir = Path(config.RAW_DIR)
    ids = _cli_class_ids(args, raw_dir)
    if not ids:
        print("oncue pcw: no classes")
        return 1
    table, status = blocks_table(ids, raw_dir=raw_dir,
                                 reads_verified=not opts.reads_unverified)
    write_table("pcw_blocks", table, ids)
    write_table(STATUS_TABLE, status_frame(status, table), ids)
    counts = pd.Series(list(status.values())).value_counts().to_dict()
    print(f"oncue pcw: {len(table)} blocks over {table['class_id'].nunique()} classes; "
          f"status {counts}")
    return 0
