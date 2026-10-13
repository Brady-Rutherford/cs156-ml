"""Privacy scan and the committed results bundle (spec §10 "Privacy", §13).

`scan(paths, roster, hashed_ids)` greps the committed paths (README, the run notebook's cells
and outputs, figures/, results/) for every roster name, every hashed user id and the ID salt,
and returns the hits; anything but zero hits blocks a push. Names are matched as whole words,
case-insensitively; a name that is also a common English word (Will, Grace, Mark ...) only in
its capitalised form; full names with any whitespace between the parts. Base64 payloads (data
URIs, notebook images) are not grepped, because random base64 can spell short names. Parquet
files are also checked structurally: `class_id` / `class_hash` must be 12-hex HMACs, and no
column may look like text, names or user ids.

`results_bundle(...)` writes `results/` (<= 50 MB): `oof_{cond}.parquet` with
HMAC(ID_SALT, class_id)[:12], `results.parquet`, Brady's own `bids.parquet` (hashed class id)
and `funnel.csv`. Only schema columns are written, raw class ids never are, and the written
files are re-checked before returning.

Hits keep the matched term in memory only; anything printed uses `Hit.redacted`.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, schemas

HASH_RE = re.compile(r"^[0-9a-f]{12}$")
ID_COLUMNS = frozenset({"class_id", "class_hash"})        # must hold 12-hex HMACs
# Columns a committed table must never have (text, names, people, user-level ids).
BLOCKED_COLUMN_RE = re.compile(
    r"(text|word|name|spk|user|actor|prompt|answer|title|chat|message|roster|transcript"
    r"|(?:^|_)hash$|(?:^|_)id$)", re.IGNORECASE)
TEXT_SUFFIXES = frozenset({
    ".md", ".txt", ".csv", ".tsv", ".json", ".py", ".html", ".htm", ".svg", ".tex", ".yaml",
    ".yml", ".toml", ".mmd", ".log", ".js", ".css", ".rst", ".xml", ".cfg", ".ini"})
SKIP_DIRS = frozenset({".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"})
DATA_URI_RE = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=\s]+")
DEFAULT_SCAN_PATHS = ("README.md", "PLAN.md", "pipeline1.ipynb", "figures", "results")
BUNDLE_MAX_BYTES = 50 * 1024 * 1024
BUNDLE_EXTRA_OOF_COLUMNS = ("in_G",)        # needed to re-run event metrics from results/
MIN_TERM_LEN = 2
MIN_SALT_LEN = 8


class PrivacyError(RuntimeError):
    """A committed file would leak a name, an id or the salt."""


@dataclass(frozen=True)
class Hit:
    path: str
    kind: str          # name | hashed_id | salt | raw_class_id | column | term
    line: int          # 1-based line / notebook cell / parquet row; 0 when not applicable
    term: str = field(repr=False)

    @property
    def redacted(self) -> str:
        t = self.term
        return t[:1] + "*" * max(len(t) - 1, 0) if self.kind != "column" else t

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind} {self.redacted}"


@dataclass
class ScanReport:
    hits: list[Hit]
    scanned: list[str]
    skipped: list[str]      # binary files that cannot be grepped (listed, not silently passed)
    missing: list[str]


# --------------------------------------------------------------------------- needles


@lru_cache(maxsize=1)
def _common_words() -> frozenset[str]:
    try:
        from wordfreq import top_n_list
    except ImportError:          # strict fallback: every name is matched case-insensitively
        return frozenset()
    return frozenset(w.lower() for w in top_n_list("en", config.COMMON_WORDS_TOPN))


_NAME_COLUMNS = ("first", "last", "first_name", "last_name", "preferred_name", "name",
                 "full_name", "first-name", "last-name", "preferred-name")


def roster_terms(roster) -> set[str]:
    """Names (single tokens and "First Last" full names) from a DataFrame with name columns,
    an iterable of strings, or an iterable of dicts."""
    terms: set[str] = set()

    def add_person(parts: Iterable) -> None:
        clean = [str(x).strip() for x in parts if not pd.isna(x) and str(x).strip()]
        for x in clean:
            terms.update(x.split())
            if len(x.split()) > 1:
                terms.add(" ".join(x.split()))
        if len(clean) >= 2:
            terms.add(f"{clean[0]} {clean[1]}")

    if roster is None:
        return terms
    if isinstance(roster, pd.DataFrame):
        cols = [c for c in _NAME_COLUMNS if c in roster.columns]
        firsts = [c for c in cols if "first" in c or c in ("name", "full_name")]
        lasts = [c for c in cols if "last" in c]
        others = [c for c in cols if c not in firsts and c not in lasts]
        for _, row in roster[cols].iterrows():
            add_person([row[c] for c in firsts + lasts + others])
        return terms
    for item in roster:
        if isinstance(item, Mapping):
            add_person([item.get(c) for c in _NAME_COLUMNS if item.get(c) is not None])
        else:
            add_person([item])
    return terms


def _boundary(alts: list[str]) -> str:
    alts = sorted(set(alts), key=len, reverse=True)
    return r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)"


@dataclass
class _Needles:
    names_ci: re.Pattern | None
    names_cs: re.Pattern | None
    ids: re.Pattern | None
    salt: str | None
    terms: dict[str, str]     # lower-cased term -> kind (for extra terms)

    def find(self, text: str) -> list[tuple[str, str, int]]:
        """[(kind, matched text, offset)] in `text`; a match inside a longer one (e.g. a
        first name inside a full name) is reported once, as the longer match."""
        found = []
        for kind, rx in (("name", self.names_ci), ("name", self.names_cs),
                         ("hashed_id", self.ids)):
            if rx is not None:
                for m in rx.finditer(text):
                    k = self.terms.get(m.group(0).lower(), kind)
                    found.append((m.start(), m.end(), k, m.group(0)))
        if self.salt:
            start = text.find(self.salt)
            while start >= 0:
                found.append((start, start + len(self.salt), "salt", self.salt))
                start = text.find(self.salt, start + 1)
        out, reach = [], -1
        for start, end, kind, term in sorted(found, key=lambda f: (f[0], -f[1])):
            if end <= reach:
                continue
            out.append((kind, term, start))
            reach = max(reach, end)
        return out


def _needles(roster, hashed_ids, *, salt, extra_terms, exclude, allow) -> _Needles:
    excl = {x.lower() for x in exclude}
    allowed = {x.lower() for x in allow}
    common = _common_words()
    ci, cs, kinds = [], [], {}
    for term in roster_terms(roster):
        low = term.lower()
        tokens = low.split()
        if len(term) < MIN_TERM_LEN or low in allowed or all(t in excl for t in tokens):
            continue
        if " " in term:
            ci.append(r"\s+".join(re.escape(t) for t in term.split()))
        elif low in common:
            cs.append(re.escape(term[:1].upper() + term[1:]))
        else:
            ci.append(re.escape(term))
    for term in extra_terms:
        if len(term) >= MIN_TERM_LEN and term.lower() not in allowed:
            ci.append(re.escape(term))
            kinds[term.lower()] = "term"
    ids = sorted({str(h) for h in hashed_ids if h is not None and len(str(h)) >= 6})
    return _Needles(
        names_ci=re.compile(_boundary(ci), re.IGNORECASE) if ci else None,
        names_cs=re.compile(_boundary(cs)) if cs else None,
        ids=re.compile(_boundary([re.escape(h) for h in ids])) if ids else None,
        salt=salt if salt and len(salt) >= MIN_SALT_LEN else None,
        terms=kinds)


# --------------------------------------------------------------------------- file readers


def _text_hits(path: str, text: str, nd: _Needles, line_base: int | None = None) -> list[Hit]:
    text = DATA_URI_RE.sub("data:,", text)
    hits = []
    for kind, term, pos in nd.find(text):
        line = line_base if line_base is not None else text.count("\n", 0, pos) + 1
        hits.append(Hit(path, kind, line, term))
    return hits


def _notebook_texts(nb: dict) -> list[tuple[int, str]]:
    """(cell number, text) for sources, text outputs and metadata; image payloads skipped."""
    out = [(0, json.dumps(nb.get("metadata", {}), ensure_ascii=False))]
    for i, cell in enumerate(nb.get("cells", []), start=1):
        src = cell.get("source", "")
        out.append((i, "".join(src) if isinstance(src, list) else str(src)))
        out.append((i, json.dumps(cell.get("metadata", {}), ensure_ascii=False)))
        for o in cell.get("outputs", []) or []:
            for key in ("text", "traceback", "ename", "evalue"):
                if key in o:
                    v = o[key]
                    out.append((i, "\n".join(v) if isinstance(v, list) else str(v)))
            for mime, v in (o.get("data") or {}).items():
                if mime.startswith("image/") and mime != "image/svg+xml":
                    continue
                if not isinstance(v, str | list):
                    v = json.dumps(v, ensure_ascii=False)
                out.append((i, "".join(v) if isinstance(v, list) else v))
    return out


def _png_texts(data: bytes) -> list[str]:
    """tEXt / iTXt / zTXt chunks of a PNG (the only greppable text in it)."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return []
    out, pos = [], 8
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        try:
            if ctype == b"tEXt":
                out.append(body.decode("latin-1"))
            elif ctype == b"zTXt":
                key, _, rest = body.partition(b"\x00")
                out.append(key.decode("latin-1") + " " + zlib.decompress(rest[1:])
                           .decode("latin-1"))
            elif ctype == b"iTXt":
                key, _, rest = body.partition(b"\x00")
                comp, rest = rest[0], rest[2:]
                _, _, rest = rest.partition(b"\x00")      # language tag
                _, _, text = rest.partition(b"\x00")      # translated keyword
                text = zlib.decompress(text) if comp else text
                out.append(key.decode("latin-1") + " " + text.decode("utf-8", "replace"))
        except (zlib.error, IndexError, UnicodeDecodeError):
            out.append(body.decode("latin-1", "replace"))
        if ctype == b"IEND":
            break
    return out


def _parquet_hits(path: str, nd: _Needles) -> list[Hit]:
    df = pd.read_parquet(path)
    hits = []
    for col in df.columns:
        name = str(col)
        hits += [Hit(path, k, 0, t) for k, t, _ in nd.find(name)]
        if name not in ID_COLUMNS and BLOCKED_COLUMN_RE.search(name):
            hits.append(Hit(path, "column", 0, name))
        s = df[col].reset_index(drop=True)
        is_text = pd.api.types.is_string_dtype(s) or s.dtype == object
        if name in ID_COLUMNS:
            bad = (~s.astype(str).str.fullmatch(HASH_RE.pattern)).to_numpy(dtype=bool) \
                if is_text else np.ones(len(s), bool)
            for row in np.flatnonzero(bad)[:5]:
                hits.append(Hit(path, "raw_class_id", int(row) + 1, str(s.iloc[row])))
        if is_text:
            for row, v in s.dropna().astype(str).drop_duplicates().items():
                hits += [Hit(path, k, int(row) + 1, t) for k, t, _ in nd.find(v)]
    return hits


def _iter_files(paths: Iterable) -> tuple[list[Path], list[str]]:
    files, missing = [], []
    for p in paths:
        p = Path(p)
        if not p.exists():
            missing.append(str(p))
        elif p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file()
                            and not (set(f.relative_to(p).parts) & SKIP_DIRS)
                            and f.name != ".DS_Store")
        else:
            files.append(p)
    return files, missing


# --------------------------------------------------------------------------- scan


def scan_report(paths: Iterable, roster=None, hashed_ids: Iterable = (), *,
                salt: str | None = None, extra_terms: Iterable[str] = (),
                exclude: Iterable[str] = config.BRADY_VARIANTS,
                allow: Iterable[str] = ()) -> ScanReport:
    """Scan files and directories; see `scan`. Also lists skipped binaries and missing paths."""
    if salt is None:
        try:
            salt = config.id_salt()
        except RuntimeError:
            salt = None
    nd = _needles(roster, list(hashed_ids), salt=salt, extra_terms=list(extra_terms),
                  exclude=list(exclude), allow=list(allow))
    files, missing = _iter_files(paths)
    hits, scanned, skipped = [], [], []
    for f in files:
        sp, suffix = str(f), f.suffix.lower()
        if suffix == ".ipynb":
            nb = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            for cell, text in _notebook_texts(nb):
                hits += _text_hits(sp, text, nd, line_base=cell)
        elif suffix == ".parquet":
            hits += _parquet_hits(sp, nd)
        elif suffix in TEXT_SUFFIXES:
            hits += _text_hits(sp, f.read_text(encoding="utf-8", errors="replace"), nd)
        elif suffix == ".png":
            for text in _png_texts(f.read_bytes()):
                hits += _text_hits(sp, text, nd, line_base=0)
        else:
            skipped.append(sp)
            continue
        scanned.append(sp)
    return ScanReport(hits=hits, scanned=scanned, skipped=skipped, missing=missing)


def scan(paths: Iterable, roster=None, hashed_ids: Iterable = (), *, salt: str | None = None,
         extra_terms: Iterable[str] = (), exclude: Iterable[str] = config.BRADY_VARIANTS,
         allow: Iterable[str] = ()) -> list[Hit]:
    """Every occurrence of a roster name, a hashed user id or the ID salt in `paths` (files or
    directories), plus structural parquet problems. Zero hits are required before a push.

    `roster`: DataFrame with name columns, or names; Brady's own name variants are excluded.
    `hashed_ids`: pseudonyms such as spk_hash values. `salt`: defaults to the environment's
    ID salt when set. `allow`: terms Brady has verified as false positives (logged by caller).
    """
    return scan_report(paths, roster, hashed_ids, salt=salt, extra_terms=extra_terms,
                       exclude=exclude, allow=allow).hits


def load_roster(raw_dir: Path | str = config.RAW_DIR, me_id=None) -> list[str]:
    """Every class-users name (first, last, preferred; full names) from raw/*/class.json,
    in memory only. Brady (me.json id, or `me_id`) is left out."""
    from scribe_core.forum_events import kebab

    raw_dir = Path(raw_dir)
    if me_id is None:
        me_path = raw_dir.parent / "me.json"
        if me_path.exists():
            me_id = json.loads(me_path.read_text(encoding="utf-8")).get("id")
    names: set[str] = set()
    for path in sorted(raw_dir.glob("*/class.json")):
        cls = kebab(json.loads(path.read_text(encoding="utf-8")))
        for u in cls.get("class-users") or []:
            user = u.get("user") if isinstance(u.get("user"), dict) else {}
            uid = user.get("id", u.get("user-id", u.get("id")))
            if me_id is not None and str(uid) == str(me_id):
                continue
            parts = [user.get(k, u.get(k)) for k in ("first-name", "last-name")]
            pref = user.get("preferred-name", u.get("preferred-name"))
            names |= roster_terms([{"first": parts[0], "last": parts[1],
                                    "preferred_name": pref}])
    return sorted(names)


def load_hashed_ids(tables_dir: Path | str = config.TABLES_DIR) -> list[str]:
    """Pseudonymous user ids (spk_hash, prof_hash) from the local tables."""
    import pyarrow.parquet as pq

    out: set[str] = set()
    for path in sorted(Path(tables_dir).glob("*.parquet")):
        cols = [c for c in pq.read_schema(path).names if c in ("spk_hash", "prof_hash")]
        if cols:
            df = pd.read_parquet(path, columns=cols)
            for c in cols:
                out |= set(df[c].dropna().astype(str))
    return sorted(out)


# --------------------------------------------------------------------------- results bundle


def hash_class_ids(values, salt: str) -> pd.Series:
    """HMAC(salt, class_id)[:12] per value; values that are already 12-hex pass through."""
    s = pd.Series(values).reset_index(drop=True)
    as_str = s.astype(str)
    if len(s) and as_str.str.fullmatch(HASH_RE.pattern).all() and not \
            pd.api.types.is_numeric_dtype(s):
        return as_str
    if pd.api.types.is_float_dtype(s):
        if not np.all(np.mod(s.dropna(), 1) == 0):
            raise ValueError("class ids must be integers")
        s = s.astype("Int64")
    mapping = {v: config.hash_id(v, salt) for v in pd.unique(s.dropna())}
    if s.isna().any():
        raise ValueError("missing class ids")
    return s.map(mapping).astype(str)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "condition"


def _bids_frame(df: pd.DataFrame, salt: str, ep_types) -> pd.DataFrame:
    if "class_hash" not in df.columns:
        df = df[df["ep_type"].isin(ep_types)].copy()
        df["class_hash"] = hash_class_ids(df["class_id"], salt).to_numpy()
    out = df[[c for c in schemas.columns("bids") if c in df.columns]].reset_index(drop=True)
    return schemas.validate(out, "bids", strict=True)


def results_bundle(out_dir: Path | str | None = None, *,
                   oof: Mapping[str, pd.DataFrame] | None = None,
                   results: pd.DataFrame | None = None, bids: pd.DataFrame | None = None,
                   funnel: pd.DataFrame | None = None, salt: str | None = None,
                   include_inner: bool = False, ep_types=(config.EP_BID, config.EP_VOTE),
                   max_bytes: int = BUNDLE_MAX_BYTES) -> dict[str, Path]:
    """Write the committed, privacy-safe bundle and return {file name: path}.

    * `oof`: {condition: oof frame (raw class ids)} -> oof_{condition}.parquet with schema
      columns (+ in_G if present), class_id replaced by HMAC(salt, class_id)[:12]; inner-OOF
      rows are dropped unless `include_inner`;
    * `results`: aggregate rows, validated strictly against the results schema;
    * `bids`: Brady's hand_episodes (class_id) or a ready bids frame (class_hash);
    * `funnel`: aggregate counts (no id columns) -> funnel.csv.
    Raises PrivacyError if a written file fails the structural check or the size cap.
    """
    salt = salt if salt is not None else config.id_salt()
    out = Path(out_dir) if out_dir is not None else config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for cond, df in (oof or {}).items():
        schemas.validate(df, "oof")
        if not include_inner and "inner" in df:
            df = df[~df["inner"].astype(bool)]
        cols = schemas.required("oof") + [c for c in BUNDLE_EXTRA_OOF_COLUMNS if c in df]
        frame = df[cols].reset_index(drop=True).copy()
        frame["class_id"] = hash_class_ids(frame["class_id"], salt).to_numpy()
        path = out / f"oof_{_slug(cond)}.parquet"
        frame.to_parquet(path, index=False)
        written[path.name] = path
    if results is not None:
        schemas.validate(results, "results", strict=True)
        path = out / "results.parquet"
        results.reset_index(drop=True).to_parquet(path, index=False)
        written[path.name] = path
    if bids is not None:
        path = out / "bids.parquet"
        _bids_frame(bids, salt, ep_types).to_parquet(path, index=False)
        written[path.name] = path
    if funnel is not None:
        bad = [c for c in funnel.columns if BLOCKED_COLUMN_RE.search(str(c))
               or str(c) in ID_COLUMNS]
        if bad:
            raise PrivacyError(f"funnel has id or text columns: {bad}")
        path = out / "funnel.csv"
        funnel.to_csv(path, index=False)
        written[path.name] = path
    hits = scan(list(written.values()), roster=None, salt=salt)
    if hits:
        raise PrivacyError("results bundle failed the privacy check: "
                           + "; ".join(str(h) for h in hits[:10]))
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if total > max_bytes:
        raise PrivacyError(f"results bundle is {total / 2**20:.1f} MB > "
                           f"{max_bytes / 2**20:.0f} MB")
    return written


# --------------------------------------------------------------------------- CLI


def _scan_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m oncue privacy", add_help=False)
    ap.add_argument("--path", action="append", default=None)
    ap.add_argument("--raw-dir", default=str(config.RAW_DIR))
    ap.add_argument("--tables-dir", default=str(config.TABLES_DIR))
    ap.add_argument("--allow", action="append", default=[])
    ap.add_argument("--load-env", action="store_true",
                    help="load pipeline1/.env first so the ID salt is checked too")
    return ap


def cli_scan(args) -> int:
    """`python -m oncue privacy [--path P ...]`: exit 1 on any hit (printed redacted)."""
    opts = _scan_parser().parse_args(list(getattr(args, "extra", []) or []))
    if opts.load_env:
        config.load_env()
    paths = opts.path or [config.PIPELINE_DIR / p for p in DEFAULT_SCAN_PATHS]
    raw_dir = Path(opts.raw_dir)
    roster = load_roster(raw_dir) if raw_dir.exists() else []
    tables = Path(opts.tables_dir)
    ids = load_hashed_ids(tables) if tables.exists() else []
    rep = scan_report(paths, roster, ids, allow=opts.allow)
    for h in rep.hits:
        print(h)
    for s in rep.skipped:
        print(f"{s}: binary, not scanned (check by hand)")
    print(f"privacy scan: {len(rep.scanned)} files, {len(roster)} names, {len(ids)} ids, "
          f"{len(rep.hits)} hits")
    return 1 if rep.hits else 0


TRAIN_SCORES_SUFFIX = "_train"


def _funnel(tables: Path) -> pd.DataFrame | None:
    """funnel.csv in the tables dir if Brady wrote one, else counts from classes (+ points)."""
    path = tables / "funnel.csv"
    if path.exists():
        return pd.read_csv(path)
    cpath, ppath = tables / "classes.parquet", tables / "points.parquet"
    if not cpath.exists():
        return None
    from .qa import funnel_table

    points = pd.read_parquet(ppath, columns=["class_id", "at_risk"]) if ppath.exists() \
        else None
    return funnel_table(pd.read_parquet(cpath), points)


def _bundle_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m oncue results-bundle", add_help=False)
    ap.add_argument("--out", default=str(config.RESULTS_DIR))
    ap.add_argument("--oof-dir", default=str(config.OOF_DIR))
    ap.add_argument("--tables-dir", default=str(config.TABLES_DIR))
    ap.add_argument("--condition", action="append", default=None)
    ap.add_argument("--include-inner", action="store_true")
    return ap


def cli_results_bundle(args) -> int:
    """`python -m oncue results-bundle`: local OOF/results/episodes -> results/ (HMAC ids)."""
    opts = _bundle_parser().parse_args(list(getattr(args, "extra", []) or []))
    try:
        salt = config.id_salt()
    except RuntimeError:
        config.load_env()
        salt = config.id_salt()
    oof_dir, tables = Path(opts.oof_dir), Path(opts.tables_dir)
    # experiments also writes {cond}_train.parquet (training-grid scores for rate-matched
    # thresholds): local only, not an OOF table.
    oof = {p.stem: pd.read_parquet(p) for p in sorted(oof_dir.glob("*.parquet"))
           if not p.stem.endswith(TRAIN_SCORES_SUFFIX)
           and (opts.condition is None or p.stem in opts.condition)}
    res_path, ep_path = tables / "results.parquet", tables / "hand_episodes.parquet"
    written = results_bundle(
        opts.out, oof=oof, salt=salt, include_inner=opts.include_inner,
        results=pd.read_parquet(res_path) if res_path.exists() else None,
        bids=pd.read_parquet(ep_path) if ep_path.exists() else None,
        funnel=_funnel(tables))
    for name, path in written.items():
        print(f"{name}: {path.stat().st_size / 1024:.1f} KB")
    return 0
