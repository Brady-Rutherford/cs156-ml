"""Class selection (spec §8.1, §10, L18, L19, L32): `classes.parquet` from the sync manifest
plus whitelisted raw files.

* `build_classes()` makes one row per synced class with the columns knowable before any
  transcription or scan (course, term, session number from the FULL section schedule, clock
  fields, roster size, Brady's main-room voice count, vault flag, ASR provider). Later stages
  fill the rest (calibration, A/V lag, tracking, PCW status, QA).
* `choose_sections()` applies §10: population = sections where Brady's mean main-room voice
  events per recorded class >= 5; headline = the top 2 eligible sections per past semester,
  whole, plus every recorded CS156 F26 session; the third-ranked section is Phase C.
* `assign_phases()` sets `phase` (pilot / A / B / C) and `run_order` (round-robin over
  sections, random within a section with seed 156). Phase A is non-vault only.
* `vault_flag` / `assert_no_vault` wrap the pre-registered vault rule (`config.in_vault`).
* `dedupe()` drops duplicate class ids and flags same-section same-date pairs (L18).

The population selects on participation; that is declared in the write-up (§10).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, loaders, schemas

log = logging.getLogger(__name__)

# (course code, term) whose recorded classes are always in the headline set (§10).
ALWAYS_INCLUDE = (("CS156", "F26"),)
PAST_TERMS = tuple(t for t in config.TERMS if t != "F26")
TIER_HEADLINE, TIER_PHASE_C = "headline", "phase_c"
EXTRA_COLUMNS = ("run_order",)     # beyond schemas.TABLES["classes"]


class VaultError(AssertionError):
    """A dev tool was handed vault classes (spec §8.1, L32)."""


# --------------------------------------------------------------------------- vault


def vault_flag(class_id) -> bool:
    """The pre-registered vault rule: sha1(f"{class_id}:oncue-vault") % 5 == 0."""
    return config.in_vault(class_id)


def _class_ids(obj) -> list:
    if isinstance(obj, pd.DataFrame):
        return list(obj["class_id"]) if "class_id" in obj else []
    if isinstance(obj, pd.Series):
        return list(obj)
    if isinstance(obj, str | int | np.integer):
        return [obj]
    return list(obj)


def assert_no_vault(obj, *, what: str = "dev tool") -> None:
    """Raise VaultError if `obj` (DataFrame with class_id / in_vault, or class ids) contains a
    vault class. Every dev tool calls this; vault ids are never printed, only counted."""
    ids = [c for c in _class_ids(obj) if c is not None and not pd.isna(c)]
    n_bad = sum(1 for c in ids if config.in_vault(int(c)))
    if isinstance(obj, pd.DataFrame) and "in_vault" in obj:
        n_bad = max(n_bad, int(obj["in_vault"].fillna(False).astype(bool).sum()))
    if n_bad:
        raise VaultError(f"{what}: {n_bad} vault class(es) present; dev tools must filter "
                         "in_vault == 0 (spec §8.1)")


# --------------------------------------------------------------------------- schedule


def _when_ms(row: dict) -> float:
    from scribe_core.timeutil import iso_to_ms

    for key in ("start-time", "date"):
        ms = iso_to_ms(row.get(key))
        if ms is not None:
            return float(ms)
    return math.inf


def session_no_full_schedule(schedule) -> dict[int, int]:
    """{class_id: 1-based session number} from a section's full `paginated-classes` schedule
    (unrecorded classes included, so missing recordings never shift the numbering)."""
    rows = schedule.get("results") if isinstance(schedule, dict) else schedule
    keyed = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get("id") is not None:
            keyed[int(row["id"])] = _when_ms(row)
    order = sorted(keyed, key=lambda cid: (keyed[cid], cid))
    return {cid: i + 1 for i, cid in enumerate(order)}


# --------------------------------------------------------------------------- one class


def spk_hasher(salt: str | None = None):
    """User id -> Scribe-style pseudonym `u_` + 10 hex (HMAC with the ID salt)."""
    from scribe_core.privacy import pseudonym

    key = salt if salt is not None else config.id_salt()
    return lambda uid: pseudonym(uid, key)


def _val(v):
    """None for missing values (None, NaN, pd.NA, "")."""
    if v is None or (not isinstance(v, str | dict | list) and pd.isna(v)) or v == "":
        return None
    return v


def _term_name(course: dict, listing: dict, manifest_row: dict):
    term = course.get("term")
    if isinstance(term, dict):
        term = term.get("name") or term.get("title")
    return _val(term) or _val(listing.get("term")) or _val(manifest_row.get("term"))


def _local(dt, term: str | None):
    tz_name = config.TERM_TZ.get(term) if term else None
    if dt is None or not tz_name:
        return None
    from zoneinfo import ZoneInfo

    return dt.astimezone(ZoneInfo(tz_name))


def _enrolled(klass: dict):
    users = klass.get("class-users")
    if not users:
        return None
    return sum(1 for u in users if str(u.get("role") or "").lower() not in config.PROF_ROLES)


def class_row(class_id: int, *, raw_dir=None, manifest_row: dict | None = None,
              me: str | None = None, hasher=None, session_nos: dict | None = None,
              log_path=None) -> dict:
    """The select-time columns of one `classes` row, from whitelisted raw files only."""
    from scribe_core.forum_events import (
        event_user_id,
        main_room_info,
        parse_voice_events,
        recording_timeline,
        room_key,
        timeline_duration,
        timing_precision,
    )
    from scribe_core.timeutil import parse_iso

    man = manifest_row or {}
    kw = {"raw_dir": raw_dir, "log_path": log_path}
    klass = loaders.load_raw(class_id, "class", **kw) or {}
    listing = loaders.load_raw(class_id, "listing", **kw) or {}
    events = loaders.load_raw(class_id, "events", **kw) or []
    section = klass.get("section") or {}
    course = section.get("course") or {}
    item = listing.get("class") or {}

    section_id = (_val(section.get("id")) or _val(listing.get("section-id"))
                  or _val(man.get("section_id")))
    course_code = (_val(course.get("course-code")) or _val(listing.get("course-code"))
                   or _val(man.get("course_code")))
    term = config.term_code(_term_name(course, listing, man))
    date = _val(klass.get("date")) or _val(item.get("date")) or _val(man.get("date"))
    sched = parse_iso(klass.get("start-time") or item.get("start-time"))
    local = _local(sched, term)
    segs = recording_timeline(klass)
    rec_start = segs[0].wall_start_ms if segs else None
    t_end = timeline_duration(segs)
    prof = klass.get("teacher-user-id", section.get("teacher-user-id"))

    room, _ = main_room_info(events)
    n_main = None
    if me is not None:
        n_main = sum(1 for ev in events if ev.get("event-type") == "voice"
                     and room is not None and room_key(ev) == room and event_user_id(ev) == me)
    precision = timing_precision(parse_voice_events(events, segs)) if segs else None
    weekday_dt = local or sched
    return {
        "class_id": int(class_id),
        "section_id": None if section_id is None else int(section_id),
        "course_code": course_code,
        "course_area": config.course_area(course_code),
        "term": term,
        "semester_idx": config.SEMESTER_IDX.get(term),
        "session_no": (session_nos or {}).get(int(class_id)),
        "date": str(date)[:10] if date else (sched.date().isoformat() if sched else None),
        "weekday": weekday_dt.weekday() if weekday_dt else None,
        "hour_local": local.hour + local.minute / 60 if local else np.nan,
        "hour_utc": sched.hour + sched.minute / 60 if sched else np.nan,
        "sched_start_utc": (sched.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                            if sched else None),
        "rec_offset_s": ((rec_start - sched.timestamp() * 1000) / 1000.0
                         if sched and rec_start is not None else np.nan),
        "rec_segments_json": json.dumps([[s.wall_start_ms, s.wall_end_ms,
                                          round(s.video_start, 3)] for s in segs]),
        "n_rec_sessions": len(segs),
        "T_end_s": np.nan if t_end is None else float(t_end),
        "prof_hash": hasher(prof) if (hasher and prof is not None) else None,
        "enrolled": _enrolled(klass),
        "n_my_voice_main": n_main,
        "in_vault": vault_flag(class_id),
        "src_res": config.SRC_RES,
        "timing_precision": precision,
        "era": term,
        "asr_provider": config.asr_provider(course_code),
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    cols = schemas.columns("classes")
    if not rows:
        return schemas.empty("classes")
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df:
            df[c] = None
    df = df[cols + [c for c in df.columns if c not in cols]]
    return schemas.coerce(df, "classes")


def build_classes(manifest=None, raw_dir=None, *, schedule_dir=None, me_path=None,
                  hasher=None, log_path=None) -> pd.DataFrame:
    """One `classes` row per class in the sync manifest (DataFrame or path; default
    `config.MANIFEST_PATH`). Classes without a section id are skipped (logged)."""
    man = manifest if isinstance(manifest, pd.DataFrame) else loaders.load_manifest(manifest)
    me = loaders.me_id(me_path)
    if hasher is None:
        try:
            hasher = spk_hasher()
        except RuntimeError:
            log.warning("No ID salt in the environment: prof_hash left empty")
    sessions: dict[int, dict] = {}
    rows, skipped = [], 0
    for rec in man.to_dict("records"):
        cid = int(rec["class_id"])
        sid = rec.get("section_id")
        if sid is not None and not pd.isna(sid) and int(sid) not in sessions:
            sched = loaders.load_schedule(int(sid), schedule_dir=schedule_dir,
                                          log_path=log_path)
            sessions[int(sid)] = session_no_full_schedule(sched) if sched else {}
        row = class_row(cid, raw_dir=raw_dir, manifest_row=rec, me=me, hasher=hasher,
                        session_nos=sessions.get(int(sid)) if sid is not None and
                        not pd.isna(sid) else None, log_path=log_path)
        if row["section_id"] is None:
            skipped += 1
            continue
        rows.append(row)
    if skipped:
        log.warning("%d classes have no section id and were skipped", skipped)
    df = _frame(rows)
    df = _fill_session_no(df)
    df = _fill_n_sessions(df, sessions)
    return dedupe(df)


def _fill_n_sessions(df: pd.DataFrame, sessions: dict[int, dict]) -> pd.DataFrame:
    """n_sessions: the size of the section's full schedule (public before class; feeds
    ctx_session_frac), else the number of synced classes of the section."""
    out = df.copy()
    sizes = {sid: len(m) for sid, m in sessions.items() if m}
    n = out["section_id"].map(sizes)
    fallback = out.groupby("section_id")["class_id"].transform("size")
    out["n_sessions"] = n.fillna(fallback).astype("Int64")
    return out


def _fill_session_no(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback when a schedule file is missing: rank by date within the section among ALL
    synced classes (the sync lists unrecorded classes too)."""
    miss = df["session_no"].isna()
    if not miss.any():
        return df
    out = df.copy()
    key = pd.to_datetime(out["sched_start_utc"].fillna(out["date"]), utc=True, errors="coerce",
                         format="ISO8601")
    rank = (out.assign(_k=key).sort_values(["_k", "class_id"])
            .groupby("section_id").cumcount() + 1)
    out.loc[miss, "session_no"] = rank[miss].astype("Int64")
    log.info("session_no from manifest dates for %d classes (no schedule file)", int(miss.sum()))
    return out


# --------------------------------------------------------------------------- dedupe (L18)


def same_section_date(df: pd.DataFrame) -> list[tuple[int, ...]]:
    """Groups of recorded classes sharing section and date (possible re-recordings)."""
    rec = df[df["n_rec_sessions"] > 0].dropna(subset=["date"])
    groups = rec.groupby(["section_id", "date"])["class_id"].apply(lambda s: tuple(sorted(s)))
    return [g for g in groups if len(g) > 1]


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated class ids (first kept) and flag same-section same-date classes (L18).

    Flagged twins may be one class recorded twice under two ids (identical audio and
    labels), so they share `cv_group` (the smallest class id of the group; every other class
    is its own group): `splits.fold_column` and the inner folds keep a group in one fold.
    The groups are also in `df.attrs["same_section_date"]`. Asserts class ids are unique."""
    out = df.loc[~df["class_id"].duplicated(keep="first")].reset_index(drop=True)
    if len(out) < len(df):
        log.warning("dedupe: dropped %d duplicate class rows", len(df) - len(out))
    groups = same_section_date(out) if len(out) else []
    if groups:
        log.warning("dedupe: %d same-section same-date groups (check for re-recordings); "
                    "each shares one CV group", len(groups))
    assert out["class_id"].is_unique, "class ids must be unique after dedupe (L18)"
    group_of = {int(c): int(min(g)) for g in groups for c in g}
    out["cv_group"] = pd.array([group_of.get(int(c), int(c)) for c in out["class_id"]],
                               dtype="Int64")
    out.attrs["same_section_date"] = groups
    return out


# --------------------------------------------------------------------------- sections, phases


def _norm_code(code) -> str:
    return "".join(ch for ch in str(code or "").upper() if ch.isalnum())


def choose_sections(classes: pd.DataFrame, *, per_semester: int = config.SECTIONS_PER_SEMESTER,
                    min_mean: float = config.POP_MIN_MEAN_MAIN_VOICE,
                    past_terms: Iterable[str] = PAST_TERMS,
                    always: Iterable[tuple[str, str]] = ALWAYS_INCLUDE,
                    exclude_sections: Iterable[int] = (), phase_c: bool = True
                    ) -> pd.DataFrame:
    """One row per section with recorded classes: mean main-room voice events, eligibility,
    rank within its semester and tier ("headline", "phase_c" or <NA>), plus a reason."""
    past = set(past_terms)
    always_set = {(_norm_code(c), t) for c, t in always}
    excluded = {int(s) for s in exclude_sections}
    rec = classes[classes["n_rec_sessions"] > 0]
    cols = ["section_id", "course_code", "term", "semester_idx", "n_classes",
            "mean_my_voice_main", "eligible", "rank", "tier", "reason"]
    if rec.empty:
        return pd.DataFrame(columns=cols)
    g = rec.groupby("section_id").agg(
        course_code=("course_code", "first"), term=("term", "first"),
        semester_idx=("semester_idx", "first"), n_classes=("class_id", "size"),
        mean_my_voice_main=("n_my_voice_main", lambda s: s.astype(float).mean())).reset_index()
    g["excluded"] = g["section_id"].isin(excluded)
    g["eligible"] = (g["mean_my_voice_main"] >= min_mean) & ~g["excluded"]
    rankable = g["eligible"] & g["term"].isin(past)
    g["rank"] = pd.Series(pd.NA, index=g.index, dtype="Int64")
    ranked = g[rankable].sort_values(["term", "mean_my_voice_main", "section_id"],
                                     ascending=[True, False, True])
    g.loc[ranked.index, "rank"] = ranked.groupby("term").cumcount().add(1).astype("Int64")
    is_always = [(_norm_code(c), t) in always_set for c, t in zip(g["course_code"], g["term"],
                                                                   strict=True)]
    g["always"] = pd.Series(is_always, index=g.index) & ~g["excluded"]

    def tier_of(r) -> tuple[str | None, str]:
        rank = None if pd.isna(r.rank) else int(r.rank)
        if r.excluded:
            return None, "excluded"
        if r.always:
            return TIER_HEADLINE, "always"
        if rank is not None and rank <= per_semester:
            return TIER_HEADLINE, f"top{rank}"
        if rank is not None and rank == per_semester + 1 and phase_c:
            return TIER_PHASE_C, f"rank{rank}"
        if rank is not None:
            return None, f"rank{rank}"
        return None, "not_past_term" if r.term not in past else "below_min"

    tiers = [tier_of(r) for r in g.itertuples()]
    g["tier"] = pd.Series([t for t, _ in tiers], index=g.index, dtype="string")
    g["reason"] = [why for _, why in tiers]
    return g.sort_values(["semester_idx", "section_id"]).reset_index(drop=True)[cols]


def _round_robin(queues: list[list]) -> list:
    out, i = [], 0
    while any(i < len(q) for q in queues):
        out += [q[i] for q in queues if i < len(q)]
        i += 1
    return out


def assign_phases(classes: pd.DataFrame, sections: pd.DataFrame, *, pilot_ids=(),
                  per_section_a: int = config.PHASE_A_PER_SECTION, seed: int = config.SEED
                  ) -> pd.DataFrame:
    """`phase` (pilot | A | B | C | <NA>) and `run_order` for recorded classes (§10).

    Within each section the order is a seeded permutation (seed, section_id). Phase A takes
    the first `per_section_a` non-vault classes of each headline section, B the rest of the
    headline set (vault included), C the phase_c sections. Pilots (non-vault, any section)
    come first. `run_order` is round-robin over sections inside each phase.
    """
    pilots = {int(c) for c in pilot_ids}
    assert_no_vault(sorted(pilots), what="pilot classes")
    out = classes.copy()
    tiers = dict(zip(sections["section_id"], sections["tier"], strict=True)) \
        if len(sections) else {}
    queues: dict[str, dict] = {p: {} for p in config.PHASES}
    rec = out[out["n_rec_sessions"] > 0]
    missing = pilots - {int(c) for c in rec["class_id"]}
    if missing:
        log.warning("%d pilot class(es) are not recorded classes in the manifest", len(missing))
    sec_order = (rec.groupby("section_id")["semester_idx"].first().fillna(99)
                 .reset_index().sort_values(["semester_idx", "section_id"])["section_id"])
    for sid in sec_order:
        ids = sorted(int(c) for c in rec.loc[rec["section_id"] == sid, "class_id"])
        rng = np.random.default_rng([int(seed), int(sid)])
        order = [ids[j] for j in rng.permutation(len(ids))]
        if any(c in pilots for c in order):
            queues["pilot"][sid] = [c for c in order if c in pilots]
        rest = [c for c in order if c not in pilots]
        tier = tiers.get(sid)
        if tier is None or pd.isna(tier):
            continue
        if tier == TIER_HEADLINE:
            a = [c for c in rest if not config.in_vault(c)][:per_section_a]
            queues["A"][sid] = a
            queues["B"][sid] = [c for c in rest if c not in set(a)]
        elif tier == TIER_PHASE_C:
            queues["C"][sid] = rest
    phase_of, order_of, n = {}, {}, 0
    for phase in config.PHASES:
        for cid in _round_robin([queues[phase][s] for s in sec_order if s in queues[phase]]):
            phase_of[cid], order_of[cid] = phase, n
            n += 1
    out["phase"] = pd.Series(out["class_id"].map(phase_of), index=out.index, dtype="string")
    out["run_order"] = pd.Series(out["class_id"].map(order_of), index=out.index, dtype="Int64")
    assert_no_vault(out[out["phase"].isin(["pilot", "A"])], what="pilot / Phase A")
    return out


def funnel(classes: pd.DataFrame) -> dict:
    """Counts only (no ids): synced -> recorded -> selected -> phases, dev vs vault."""
    rec = classes[classes["n_rec_sessions"] > 0]
    sel = classes[classes["phase"].notna()]
    return {
        "synced": len(classes), "recorded": len(rec), "selected": len(sel),
        "selected_dev": int((~sel["in_vault"]).sum()), "selected_vault": int(sel["in_vault"].sum()),
        **{f"phase_{p}": int((sel["phase"] == p).sum()) for p in config.PHASES},
    }


# --------------------------------------------------------------------------- CLI


def cli(args) -> int:
    """`python -m oncue select [--pilot CID,...] [--exclude-section SID] [--scribe-dir DIR]`."""
    p = argparse.ArgumentParser(prog="python -m oncue select")
    p.add_argument("--scribe-dir", type=Path, default=None,
                   help="scribe sync root (default data/scribe)")
    p.add_argument("--out", type=Path, default=None, help="default data/tables/classes.parquet")
    p.add_argument("--pilot", default="", help="comma-separated non-vault pilot class ids")
    p.add_argument("--exclude-section", type=int, action="append", default=[])
    p.add_argument("--always", action="append", default=None, metavar="COURSE:TERM",
                   help="always-included course and term (default CS156:F26)")
    p.add_argument("--per-semester", type=int, default=config.SECTIONS_PER_SEMESTER)
    ns = p.parse_args(getattr(args, "extra", None) or [])
    config.load_env()                                    # ID salt for prof_hash
    root = ns.scribe_dir or config.SCRIBE_DIR
    classes = build_classes(root / "manifest.csv", root / "raw",
                            schedule_dir=root / "schedule", me_path=root / "me.json")
    always = ALWAYS_INCLUDE if ns.always is None else \
        tuple(tuple(a.split(":", 1)) for a in ns.always)
    sections = choose_sections(classes, per_semester=ns.per_semester, always=always,
                               exclude_sections=ns.exclude_section)
    pilots = [int(c) for c in ns.pilot.split(",") if c.strip()]
    classes = assign_phases(classes, sections, pilot_ids=pilots)
    if config.SPLIT_PATH.exists():
        from .splits import split_of

        classes["split"] = classes["class_id"].map(split_of()).astype("string")
    out = ns.out or config.table_path("classes")
    out.parent.mkdir(parents=True, exist_ok=True)
    schemas.validate(classes, "classes")
    classes.to_parquet(out, index=False)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    sections.to_csv(config.LOGS_DIR / "sections_selected.csv", index=False)
    print(sections[sections["tier"].notna()][["course_code", "term", "n_classes",
                                               "mean_my_voice_main", "tier"]]
          .to_string(index=False))
    print("funnel: " + ", ".join(f"{k} {v}" for k, v in funnel(classes).items()))
    print(f"wrote {out}")
    return 0
