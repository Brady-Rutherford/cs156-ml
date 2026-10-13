"""Harvest (spec §10, §11 D0/D8, §13): Forum sync, class edit pages and pre-class work (PCW).

Everything is fetched once and cached under `data/scribe/raw/{cid}/` (gitignored):

* `run_sync(out)` runs Scribe's own `scribe sync` (class, class-events, polls, poll-responses,
  breakouts, ...) with an injectable `ForumClient`, and also saves each section's full
  schedule (`paginated-classes`, incl. unrecorded classes) to `schedule/{sid}.json`.
* `class_edit_page(client, cid)` GETs the class page (learning outcomes, study guide).
* `pcw_fetch(cid)` POSTs for the PCW signed URL, then GETs the SLE template and doc and
  writes `pcw_template.json`, `pcw_doc.json`, `pcw_meta.json` (meta last = complete).
* `pcw_side_effect_probe(cid)` is D8: POST + GETs, wait 2 min, repeat, diff metadata and
  content hashes, and probe history/version endpoints with GETs only.
* `import_export(dir)` is Route C: per-class console exports -> the same raw layout.

Network discipline (every client is injectable; tests run offline with fakes):

* Forum GETs go through `scribe_core.forum.ForumClient` (cookie only to forum.minerva.edu,
  no redirects, `/api/v1/class-users` refused).
* `post_json` is the ONLY Forum POST. It is allowlisted to the single
  `pre_class_resource_signed_url` path and sends `X-CSRFToken` (from the `csrftoken`
  cookie) and `Referer: https://forum.minerva.edu/`.
* `SleClient` is GET-only (it has no other method), talks only to https
  `sle-collaboration*` hosts, follows redirects only within them, and never sees the Forum
  cookie. Never PUT/PATCH/POST to SLE, never open websockets.
* Signed URLs and tokens are never logged, printed or written to disk.

The SLE URL shapes, auth reproduction and Forum's fallback POST body are PROVISIONAL until
D8 (look for "PROVISIONAL" below); each lives in one constant or one function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from urllib.parse import parse_qsl, urljoin, urlsplit

import requests
from scribe_core.forum import (
    COOKIE_ENV,
    FORUM_BASE,
    ForumAuthError,
    ForumClient,
    ForumError,
    ForumHTTPError,
    normalize_cookie,
)
from scribe_core.forum_events import kebab
from scribe_core.redact import strip_query
from scribe_core.timeutil import parse_iso

from . import config

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- endpoints

CLASS_EDIT_PAGE_PATH = "/api/v1/classes/{cid}/class_edit_page"
PCW_SIGNED_URL_PATH = "/api/v1/pre_class_work/classes/{cid}/pre_class_resource_signed_url"
POST_ALLOWED_RE = re.compile(
    r"^/api/v1/pre_class_work/classes/\d+/pre_class_resource_signed_url/?$")
# D8: try {"iframed": false} first (no saving indicators), then Forum's own body.
PCW_POST_BODIES = ({"iframed": False}, {"iframed": True})   # 2nd: PROVISIONAL (D8)
CSRF_COOKIE = "csrftoken"
FORUM_REFERER = FORUM_BASE + "/"
USER_AGENT = "oncue-harvest (CS156 pipeline1; GET-only on SLE)"

SLE_HOST_LABEL = "sle-collaboration"      # a hostname label must start with this
SLE_API_PREFIX = "/api/v1/"
SLE_TEMPLATE_PATH = "templates/{uuid}/"
SLE_DOC_PATH = "doc/{uuid}/"
# GET-only probes for a history / versions endpoint (D8 decides Branch H vs N).
SLE_HISTORY_CANDIDATES = ("doc/{doc}/history/", "doc/{doc}/versions/", "doc/{doc}/revisions/",
                          "templates/{tmpl}/versions/", "templates/{tmpl}/history/")
# Set once D8 finds a history endpoint (Branch H), e.g. "doc/{doc}/history/".
PCW_HISTORY_PATH: str | None = None

HARVEST_RATE_PER_S = 1.0                  # §10: about 165 x 4 requests at 1 req/s, once
SYNC_RATE_PER_S = 3.0                     # `scribe sync`'s own default (694 classes)
PROBE_WAIT_S = 120.0                      # D8: wait 2 min between the two passes
CREATED_TOL_S = 5.0                       # doc created this close to the POST => created by it
NO_PCW_STATUSES = (400, 404, 405, 410)    # POST answers meaning "this class has no PCW"

PCW_FILES = ("pcw_template", "pcw_doc", "pcw_meta")
# Route C export keys (top level, "_" -> "-") -> raw source. Unlisted keys are never written.
ROUTE_C_KEYS = {
    "class": ("class", "class-json", "klass"),
    "events": ("events", "class-events"),
    "polls": ("polls",),
    "poll_responses": ("poll-responses",),
    "breakouts": ("breakouts",),
    "class_edit_page": ("class-edit-page",),
    "pcw_template": ("pcw-template", "template"),
    "pcw_doc": ("pcw-doc", "doc"),
    "pcw_meta": ("pcw-meta",),
    "listing": ("listing",),
}
ROUTE_C_META_KEYS = ("class-id", "id", "exported-at", "me", "schema", "version")
_SIGNED_RE = re.compile(r"^https?://\S+\?\S*(?:signature|expires|token|key-pair-id|x-amz-|sig=)",
                        re.IGNORECASE)
_URL_KEY_RE = re.compile(r"(?:^|[-_])(?:url|src|href)$", re.IGNORECASE)
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                      r"-[0-9a-fA-F]{12}$")
_TOKEN_PARAM_RE = re.compile(r"^(?:token|access[-_]?token|jwt|auth|key|signature|sig)$",
                             re.IGNORECASE)


class HarvestError(RuntimeError):
    """A harvest step refused a request or could not complete."""


# --------------------------------------------------------------------------- small helpers


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _class_dir(class_id, raw_dir=None) -> Path:
    return (Path(raw_dir) if raw_dir is not None else config.RAW_DIR) / str(int(class_id))


def _digest(obj) -> str | None:
    if obj is None:
        return None
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha1(blob).hexdigest()


class _Throttle:
    def __init__(self, rate_per_sec: float, sleep=time.sleep, clock=time.monotonic):
        self._min = 1.0 / rate_per_sec if rate_per_sec and rate_per_sec > 0 else 0.0
        self._sleep, self._clock, self._last = sleep, clock, None

    def __call__(self) -> None:
        if self._last is not None and self._min:
            wait = self._min - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()


# --------------------------------------------------------------------------- Forum POST


def csrf_token(cookie: str | None) -> str | None:
    """The `csrftoken` value inside a full `Cookie` header string."""
    for part in (cookie or "").split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name.strip() == CSRF_COOKIE:
            return value.strip().strip('"') or None
    return None


def post_json(path: str, body: dict, *, cookie: str | None = None, session=None,
              timeout: float = 60.0):
    """POST JSON to Forum. Allowlisted to ONE path: the PCW `pre_class_resource_signed_url`.

    Sends the Forum cookie, `X-CSRFToken` (from its `csrftoken`), `Referer` and JSON. Never
    follows redirects. Returns the decoded JSON (None for an empty body). Raises HarvestError
    for any other path, ForumAuthError for 401/403/redirects, ForumHTTPError for other
    errors. `cookie` defaults to FORUM_COOKIE from the environment.
    """
    if not isinstance(path, str) or not POST_ALLOWED_RE.match(path):
        raise HarvestError(f"Refusing POST to {path!r}: post_json is allowlisted to "
                           f"{PCW_SIGNED_URL_PATH} only.")
    if not isinstance(body, dict):
        raise HarvestError("post_json body must be a JSON object")
    cookie = normalize_cookie(cookie if cookie is not None else os.environ.get(COOKIE_ENV))
    if not cookie:
        raise ForumAuthError(None, None, "FORUM_COOKIE is not set.")
    token = csrf_token(cookie)
    if not token:
        raise HarvestError("No csrftoken in FORUM_COOKIE: re-copy the full cookie header (D0).")
    sess = session if session is not None else requests.Session()
    headers = {"Cookie": cookie, "X-CSRFToken": token, "Referer": FORUM_REFERER,
               "Origin": FORUM_BASE, "Accept": "application/json",
               "Content-Type": "application/json", "User-Agent": USER_AGENT}
    try:
        resp = sess.post(FORUM_BASE + path, json=body, headers=headers, timeout=timeout,
                         allow_redirects=False)
    except requests.RequestException as exc:
        raise ForumError(f"Network error calling {path}: {type(exc).__name__}") from None
    status = resp.status_code
    if status in (401, 403) or 300 <= status < 400:
        raise ForumAuthError(status, path)
    if status >= 400:
        raise ForumHTTPError(status, path)
    if not (getattr(resp, "text", "") or "").strip():
        return None
    try:
        return resp.json()
    except ValueError:
        raise ForumAuthError(status, path, f"Forum returned non-JSON for {path}.") from None


class ForumPoster:
    """`post_json` with one cookie, one session and a 1 req/s throttle (one per harvest)."""

    def __init__(self, cookie: str | None = None, *, session=None,
                 rate_per_sec: float = HARVEST_RATE_PER_S, timeout: float = 60.0,
                 sleep=time.sleep, clock=time.monotonic):
        self._cookie = cookie
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout
        self._throttle = _Throttle(rate_per_sec, sleep, clock)
        self.requests_made = 0

    def __call__(self, path: str, body: dict):
        self._throttle()
        self.requests_made += 1
        return post_json(path, body, cookie=self._cookie, session=self._session,
                         timeout=self._timeout)


# --------------------------------------------------------------------------- SLE (GET only)


def check_sle_url(url: str) -> str:
    """Return `url` if it is https on an `sle-collaboration*` host, else raise HarvestError."""
    parts = urlsplit(str(url))
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not any(lbl.startswith(SLE_HOST_LABEL)
                                          for lbl in host.split(".")):
        raise HarvestError(f"Refusing to contact {host or strip_query(url)!r}: SLE requests go "
                           f"only to https://{SLE_HOST_LABEL}* hosts.")
    return str(url)


@dataclass
class SleResponse:
    status: int
    data: object = None
    last_modified: str | None = None
    etag: str | None = None


class SleClient:
    """GET-only client for the SLE collaboration API.

    There is deliberately no post/put/patch/delete method; every request goes through
    `session.get` with redirects followed by hand, and only to SLE hosts. The Forum cookie is
    never attached. `params`/`headers` carry the iframe's auth (see `sle_auth_params`).
    """

    def __init__(self, base_url: str, *, session=None, params: dict | None = None,
                 headers: dict | None = None, rate_per_sec: float = HARVEST_RATE_PER_S,
                 timeout: float = 60.0, max_redirects: int = 5, sleep=time.sleep,
                 clock=time.monotonic):
        self.base_url = check_sle_url(str(base_url).rstrip("/") + "/")
        self._session = session if session is not None else requests.Session()
        self._params = dict(params or {})
        self._headers = {"Accept": "application/json", "User-Agent": USER_AGENT,
                         **(headers or {})}
        self._timeout = timeout
        self._max_redirects = max_redirects
        self._throttle = _Throttle(rate_per_sec, sleep, clock)
        self.requests_made = 0

    def __repr__(self) -> str:
        return f"SleClient({strip_query(self.base_url)!r})"

    def _abs(self, path_or_url: str) -> str:
        if "://" in str(path_or_url):
            return check_sle_url(path_or_url)
        return check_sle_url(urljoin(self.base_url, str(path_or_url).lstrip("/")))

    def _get(self, url: str, params: dict | None):
        for _ in range(self._max_redirects + 1):
            self._throttle()
            self.requests_made += 1
            try:
                resp = self._session.get(url, params=params or None, headers=self._headers,
                                         timeout=self._timeout, allow_redirects=False)
            except requests.RequestException as exc:
                raise HarvestError(f"Network error calling {strip_query(url)}: "
                                   f"{type(exc).__name__}") from None
            location = (getattr(resp, "headers", None) or {}).get("Location")
            if 300 <= resp.status_code < 400 and location:
                url, params = check_sle_url(urljoin(url, location)), None
                continue
            return resp
        raise HarvestError(f"Too many redirects from {strip_query(url)}")

    def get(self, path_or_url: str, params: dict | None = None) -> SleResponse:
        """GET a path under `base_url` (or an absolute SLE URL); JSON decoded when 2xx."""
        resp = self._get(self._abs(path_or_url), {**self._params, **(params or {})})
        data = None
        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
            except ValueError:
                data = None
        headers = getattr(resp, "headers", None) or {}
        return SleResponse(resp.status_code, data, headers.get("Last-Modified"),
                           headers.get("ETag"))

    def open(self, url: str) -> int:
        """GET the signed link once, as the iframe does (session cookies are kept)."""
        return self._get(check_sle_url(url), None).status_code


@dataclass
class PcwTarget:
    """Where a class's PCW lives. `link` is the signed SLE URL: never persisted or logged."""

    link: str | None
    template_uuid: str | None = None
    doc_uuid: str | None = None
    created: bool | None = None

    def __repr__(self) -> str:
        return (f"PcwTarget(link={strip_query(self.link) or None!r}, "
                f"template={self.template_uuid!r}, doc={self.doc_uuid!r})")

    @property
    def api_base(self) -> str | None:
        if not self.link:
            return None
        host = urlsplit(self.link).hostname
        return f"https://{host}{SLE_API_PREFIX}" if host else None


def sle_auth_params(link: str | None) -> dict:
    """Auth to reproduce from the signed link: token-like query parameters, sent along on
    every SLE GET. PROVISIONAL until D8 shows the iframe's real auth (header or query)."""
    if not link:
        return {}
    return {k: v for k, v in parse_qsl(urlsplit(link).query) if _TOKEN_PARAM_RE.match(k)}


def sle_client_for(target: PcwTarget, *, session=None, **kwargs) -> SleClient:
    """Default SLE factory: a fresh session (no Forum cookie) plus `sle_auth_params`."""
    if not target.api_base:
        raise HarvestError("PCW target has no SLE link")
    return SleClient(target.api_base, session=session, params=sle_auth_params(target.link),
                     **kwargs)


_LINK_KEYS = ("url", "signed-url", "link", "resource-url", "iframe-url", "src", "href")
_TMPL_KEYS = ("template-uuid", "template-id", "template")
_DOC_KEYS = ("doc-uuid", "doc-id", "document-id", "doc", "document")


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _uuid_of(data: dict, keys) -> str | None:
    for k in keys:
        v = data.get(k)
        if isinstance(v, dict):
            v = v.get("uuid", v.get("id"))
        if isinstance(v, str) and _UUID_RE.match(v):
            return v
    return None


def _uuid_in_link(link: str | None, names) -> str | None:
    if not link:
        return None
    parts = urlsplit(link)
    for route in (parts.path, parts.fragment.split("?", 1)[0]):
        segs = [s for s in route.split("/") if s]
        for a, b in pairwise(segs):
            if a.lower() in names and _UUID_RE.match(b):
                return b
    for query in (parts.query, parts.fragment.partition("?")[2]):
        for k, v in parse_qsl(query):
            key = k.lower().replace("_", "-")
            if any(key in (n, f"{n}-id", f"{n}-uuid") for n in names) and _UUID_RE.match(v):
                return v
    return None


def locate_pcw(response) -> PcwTarget:
    """Find the SLE link and template/doc uuids in the signed-URL POST response.

    PROVISIONAL (D8): looks for a link under common keys (or any SLE URL in the response),
    uuids under template/doc keys, else after `/template(s)/` and `/doc(s)/` in the link's
    path, fragment or query.
    """
    data = kebab(response) if isinstance(response, dict) else {}
    if isinstance(response, str):
        data = {"url": response}
    link = None
    for key in _LINK_KEYS:
        v = data.get(key)
        if isinstance(v, str) and v.startswith(("https://", "http://")):
            link = v
            break
    if link is None:
        for s in _strings(data):
            if s.startswith("https://") and SLE_HOST_LABEL in (urlsplit(s).hostname or ""):
                link = s
                break
    tmpl = _uuid_of(data, _TMPL_KEYS) or _uuid_in_link(link, ("template", "templates"))
    doc = _uuid_of(data, _DOC_KEYS) or _uuid_in_link(link, ("doc", "docs", "document",
                                                            "documents"))
    created = data.get("created") if isinstance(data.get("created"), bool) else None
    return PcwTarget(link, tmpl, doc, created)


# --------------------------------------------------------------------------- PCW harvest


def _first(d: dict, keys) -> str | None:
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def pcw_meta(template, doc, target: PcwTarget, fetched_at: datetime,
             template_resp: SleResponse | None = None, doc_resp: SleResponse | None = None,
             history=None) -> dict:
    """The class-level PCW metadata the pcw module turns into `pcw_status` (QA only).

    Keys (snake_case, whitelisted after kebab): template_id, doc_id, template_updated_at,
    doc_updated_at, created, fetched_at, [history].
    """
    t = kebab(template) if isinstance(template, dict) else {}
    d = kebab(doc) if isinstance(doc, dict) else {}
    created = target.created
    created_at = parse_iso(_first(d, ("created-at", "created")))
    if created is None and created_at is not None:
        created = created_at >= fetched_at - timedelta(seconds=CREATED_TOL_S)
    meta = {
        "template_id": target.template_uuid or t.get("id"),
        "doc_id": target.doc_uuid or d.get("id"),
        "template_updated_at": (_first(t, ("updated-at", "modified-at"))
                                or (template_resp.last_modified if template_resp else None)),
        "doc_updated_at": (_first(d, ("updated-at", "modified-at", "last-modified"))
                           or (doc_resp.last_modified if doc_resp else None)),
        "created": created,
        "fetched_at": _iso(fetched_at),
    }
    if history is not None:
        meta["history"] = history
    return meta


@dataclass
class _Snapshot:
    template: object
    doc: object
    meta: dict
    target: PcwTarget
    sle: SleClient | None = None
    template_resp: SleResponse | None = None
    doc_resp: SleResponse | None = None


def _sle_json(sle: SleClient, path: str, what: str, *, missing_ok: bool):
    r = sle.get(path)
    if r.status == 200:
        return r
    if missing_ok and r.status in (404, 410):
        return r
    raise HarvestError(f"SLE {what} GET returned {r.status} (auth? see D8)")


def _pcw_snapshot(class_id, poster, sle_factory, locate, now) -> _Snapshot:
    t0 = now()
    path = PCW_SIGNED_URL_PATH.format(cid=int(class_id))
    target = PcwTarget(None)
    for body in PCW_POST_BODIES:
        try:
            resp = poster(path, dict(body))
        except ForumHTTPError as exc:
            if exc.status in NO_PCW_STATUSES:
                continue
            raise
        target = locate(resp)
        if target.link:
            break
    if not target.link:
        return _Snapshot(None, None, pcw_meta(None, None, target, t0), target)
    if not (target.template_uuid or target.doc_uuid):
        raise HarvestError(f"class {class_id}: PCW link found but no template/doc uuid in it; "
                           "fix locate_pcw (D8)")
    sle = sle_factory(target)
    sle.open(target.link)
    tr = dr = None
    if target.template_uuid:
        tr = _sle_json(sle, SLE_TEMPLATE_PATH.format(uuid=target.template_uuid), "template",
                       missing_ok=False)
    if target.doc_uuid:
        dr = _sle_json(sle, SLE_DOC_PATH.format(uuid=target.doc_uuid), "doc", missing_ok=True)
    template = tr.data if tr is not None and tr.status == 200 else None
    doc = dr.data if dr is not None and dr.status == 200 else None
    history = None
    if PCW_HISTORY_PATH and target.doc_uuid:
        h = sle.get(PCW_HISTORY_PATH.format(doc=target.doc_uuid, tmpl=target.template_uuid))
        history = h.data if h.status == 200 else None
    meta = pcw_meta(template, doc, target, t0, tr, dr, history)
    return _Snapshot(template, doc, meta, target, sle, tr, dr)


def make_forum_client(**kwargs) -> ForumClient:
    """The real Forum client (reads FORUM_COOKIE). Tests replace this factory."""
    return ForumClient(rate_per_sec=kwargs.pop("rate_per_sec", HARVEST_RATE_PER_S), **kwargs)


def make_poster(**kwargs) -> ForumPoster:
    return ForumPoster(**kwargs)


def pcw_fetch(class_id, *, poster=None, sle_factory=None, raw_dir=None, refresh: bool = False,
              locate=locate_pcw, now=_now) -> tuple[object, object, dict]:
    """(template, doc, meta) for one class, fetched ONCE and cached in raw/{cid}/.

    A cached `pcw_meta.json` means done: nothing is requested again unless `refresh`. A class
    with no PCW gets a meta with null ids (cached too). Errors raise and cache nothing.
    """
    cdir = _class_dir(class_id, raw_dir)
    files = {k: cdir / config.RAW_SOURCE_FILES[k] for k in PCW_FILES}
    if files["pcw_meta"].exists() and not refresh:
        return (_read_json(files["pcw_template"]), _read_json(files["pcw_doc"]),
                _read_json(files["pcw_meta"]))
    poster = poster if poster is not None else make_poster()
    sle_factory = sle_factory if sle_factory is not None else sle_client_for
    snap = _pcw_snapshot(class_id, poster, sle_factory, locate, now)
    for key, obj in (("pcw_template", snap.template), ("pcw_doc", snap.doc)):
        if obj is not None:
            _write_json(files[key], obj)
        elif refresh:
            files[key].unlink(missing_ok=True)
    _write_json(files["pcw_meta"], snap.meta)            # last: its presence = complete
    return snap.template, snap.doc, snap.meta


def probe_history(sle: SleClient | None, target: PcwTarget,
                  candidates=SLE_HISTORY_CANDIDATES) -> dict[str, int | None]:
    """D8: GET each candidate history/version path; returns {path template: HTTP status}."""
    out: dict[str, int | None] = {}
    for cand in candidates:
        needs_doc, needs_tmpl = "{doc}" in cand, "{tmpl}" in cand
        if sle is None or (needs_doc and not target.doc_uuid) or (
                needs_tmpl and not target.template_uuid):
            out[cand] = None
            continue
        out[cand] = sle.get(cand.format(doc=target.doc_uuid, tmpl=target.template_uuid)).status
    return out


_META_COMPARE = ("template_id", "doc_id", "template_updated_at", "doc_updated_at", "created")


def pcw_side_effect_probe(class_id, *, poster=None, sle_factory=None, locate=locate_pcw,
                          wait_s: float = PROBE_WAIT_S, sleep=time.sleep, now=_now,
                          history_candidates=SLE_HISTORY_CANDIDATES) -> dict:
    """D8: do reads change anything? POST + GETs, wait, repeat, diff (no content returned).

    The report holds metadata (timestamps, uuids, flags), HTTP statuses and content hashes
    only, so it can be logged. Nothing is written to raw/ (harvest stays once-and-cached).
    """
    poster = poster if poster is not None else make_poster()
    sle_factory = sle_factory if sle_factory is not None else sle_client_for
    a = _pcw_snapshot(class_id, poster, sle_factory, locate, now)
    sleep(wait_s)
    b = _pcw_snapshot(class_id, poster, sle_factory, locate, now)
    before = {k: a.meta.get(k) for k in _META_COMPARE}
    after = {k: b.meta.get(k) for k in _META_COMPARE}
    lm = {"template_before": a.template_resp.last_modified if a.template_resp else None,
          "template_after": b.template_resp.last_modified if b.template_resp else None,
          "doc_before": a.doc_resp.last_modified if a.doc_resp else None,
          "doc_after": b.doc_resp.last_modified if b.doc_resp else None}
    return {
        "class_id": int(class_id),
        "wait_s": float(wait_s),
        "has_pcw": bool(a.target.link),
        "has_doc": a.doc is not None,
        "doc_appeared": a.doc is None and b.doc is not None,
        "meta_before": before,
        "meta_after": after,
        "meta_changed": sorted(k for k in _META_COMPARE if before[k] != after[k]),
        "last_modified": lm,
        "last_modified_changed": (lm["template_before"] != lm["template_after"]
                                  or lm["doc_before"] != lm["doc_after"]),
        "template_changed": _digest(a.template) != _digest(b.template),
        "doc_changed": _digest(a.doc) != _digest(b.doc),
        "history": probe_history(b.sle, b.target, history_candidates),
    }


# --------------------------------------------------------------------------- class edit page


def class_edit_page(client, class_id, *, raw_dir=None, refresh: bool = False):
    """GET `classes/{cid}/class_edit_page` once and cache it (None if Forum refuses)."""
    path = _class_dir(class_id, raw_dir) / config.RAW_SOURCE_FILES["class_edit_page"]
    if path.exists() and not refresh:
        return _read_json(path)
    data, status = client.try_get(CLASS_EDIT_PAGE_PATH.format(cid=int(class_id)))
    if data is None:
        log.info("class %s: class_edit_page unavailable (HTTP %s)", class_id, status)
        return None
    _write_json(path, data)
    return data


def harvest_classes(class_ids, *, client, poster=None, sle_factory=None, raw_dir=None,
                    pcw: bool = True, refresh: bool = False, now=_now) -> dict:
    """class_edit_page + PCW for each class. Per-class failures are recorded (and retried
    next run); a dead cookie (401 or redirect) stops the run."""
    summary = {"classes": 0, "edit_page": 0, "edit_page_missing": 0, "pcw_found": 0,
               "pcw_none": 0, "failed": 0}
    if pcw and poster is None:
        poster = make_poster()                 # one session and throttle for the whole run
    for cid in class_ids:
        cdir = _class_dir(cid, raw_dir)
        summary["classes"] += 1
        try:
            page = class_edit_page(client, cid, raw_dir=raw_dir, refresh=refresh)
            summary["edit_page" if page is not None else "edit_page_missing"] += 1
            if pcw:
                _, _, meta = pcw_fetch(cid, poster=poster, sle_factory=sle_factory,
                                       raw_dir=raw_dir, refresh=refresh, now=now)
                summary["pcw_found" if meta.get("template_id") or meta.get("doc_id")
                        else "pcw_none"] += 1
            (cdir / "_harvest_error.json").unlink(missing_ok=True)
        except ForumAuthError as exc:
            if exc.status != 403:
                raise
            summary["failed"] += 1
            _write_json(cdir / "_harvest_error.json", {"at": _iso(now()), "error": str(exc)})
        except (ForumError, HarvestError) as exc:
            summary["failed"] += 1
            _write_json(cdir / "_harvest_error.json", {"at": _iso(now()), "error": str(exc)})
            print(f"class {cid}: FAILED ({exc})", file=sys.stderr)
    return summary


# --------------------------------------------------------------------------- scribe sync


class _ScheduleRecorder:
    """Wraps a ForumClient; remembers every section's `classes()` listing (the full
    `paginated-classes` schedule) so `run_sync` can save it without extra requests."""

    def __init__(self, client):
        self._client = client
        self.schedules: dict = {}

    def classes(self, section_id, *args, **kwargs):
        items = self._client.classes(section_id, *args, **kwargs)
        self.schedules[section_id] = list(items or [])
        return items

    def __getattr__(self, name):
        return getattr(self._client, name)


def write_schedules(schedules: dict, schedule_dir=None) -> list[Path]:
    """Write `schedule/{sid}.json` in the paginated shape {"count", "results", ...}."""
    base = Path(schedule_dir) if schedule_dir is not None else config.SCHEDULE_DIR
    out = []
    for sid, items in schedules.items():
        out.append(_write_json(base / f"{sid}.json", {"count": len(items), "next": None,
                                                      "previous": None, "results": items}))
    return out


def run_sync(out=None, *, client=None, sections=("past", "current"), limit: int = 0,
             extras: bool = True, schedules: bool = True) -> int:
    """`scribe sync --out {out}` with an injectable ForumClient, plus `schedule/{sid}.json`.

    Resumable: classes with `_complete.json` are skipped. Returns Scribe's exit code (1 when
    some classes failed; run again to retry them).
    """
    from hosts import cli as scribe_cli

    out = Path(out) if out is not None else config.SCRIBE_DIR
    rec = _ScheduleRecorder(client if client is not None else make_forum_client())
    argv = ["sync", "--out", str(out), "--sections", ",".join(sections)]
    if limit:
        argv += ["--limit", str(int(limit))]
    if not extras:
        argv.append("--no-extras")
    rc = scribe_cli.cmd_sync(scribe_cli.build_parser().parse_args(argv), rec)
    if schedules:
        write_schedules(rec.schedules, out / "schedule")
    return rc


# --------------------------------------------------------------------------- Route C import


def _scrub_signed(obj):
    """Drop the query string of signed-looking URLs (CloudFront, SLE tokens) and of every
    value under a URL-ish key (`url`, `signed_url`, `download-url`, `src`, `href`)."""
    if isinstance(obj, str):
        return strip_query(obj) if _SIGNED_RE.match(obj) else obj
    if isinstance(obj, dict):
        return {k: (strip_query(v) if isinstance(v, str) and _URL_KEY_RE.search(str(k))
                    and "://" in v else _scrub_signed(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_signed(v) for v in obj]
    return obj


def _export_class_id(data: dict, path: Path) -> int | None:
    top = {str(k).replace("_", "-"): v for k, v in data.items()}
    for key in ("class-id", "id"):
        if top.get(key) is not None:
            return int(top[key])
    klass = top.get("class") or top.get("class-json") or top.get("klass")
    if isinstance(klass, dict) and klass.get("id") is not None:
        return int(klass["id"])
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else None


def _listing_from_class(klass: dict, cid: int) -> dict:
    k = kebab(klass)
    section = k.get("section") if isinstance(k.get("section"), dict) else {}
    course = section.get("course") if isinstance(section.get("course"), dict) else {}
    term = course.get("term")
    if isinstance(term, dict):
        term = term.get("name") or term.get("title")
    return {"section_id": section.get("id"), "course_code": course.get("course-code"),
            "term": term, "class": {"id": cid, "title": k.get("title"), "date": k.get("date"),
                                    "start-time": k.get("start-time"),
                                    "recording-sessions": k.get("recording-sessions") or []}}


def import_export(src_dir, *, raw_dir=None, overwrite: bool = False,
                  write_manifest: bool = True) -> dict:
    """Route C: turn console exports `{src_dir}/*.json` (one per class) into raw/{cid}/.

    Export format (one JSON object per class, snake_case or kebab-case top-level keys):
    `class_id`, `exported_at`, optional `me` {id, role}, and any of `class`, `events` (or
    `class_events`), `polls`, `poll_responses`, `breakouts`, `class_edit_page`,
    `pcw_template` (or `template`), `pcw_doc` (or `doc`), `pcw_meta`, `listing`. Any other
    key (analytics, outcome assessments, recording or signed URLs, ...) is never written.
    Signed-looking URLs inside kept documents lose their query string. Existing files are
    kept unless `overwrite`. Returns {"classes": {cid: [sources]}, "skipped_keys": {...}}.
    """
    src_dir = Path(src_dir)
    raw = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    scribe_dir = raw.parent
    alias = {a: src for src, names in ROUTE_C_KEYS.items() for a in names}
    report: dict = {"classes": {}, "skipped_keys": {}, "files": 0}
    me = None
    for path in sorted(src_dir.glob("*.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        cid = _export_class_id(data, path)
        if cid is None:
            continue
        docs: dict = {}
        for key, value in data.items():
            k = str(key).replace("_", "-")
            if k in alias:
                if value is not None:
                    docs[alias[k]] = _scrub_signed(value)
            elif k == "me" and isinstance(value, dict):
                me = {"id": value.get("id"), "role": value.get("role")}
            elif k not in ROUTE_C_META_KEYS:
                report["skipped_keys"][k] = report["skipped_keys"].get(k, 0) + 1
        if "listing" not in docs and isinstance(docs.get("class"), dict):
            docs["listing"] = _listing_from_class(docs["class"], cid)
        if "pcw_meta" not in docs and ("pcw_template" in docs or "pcw_doc" in docs):
            t = kebab(docs.get("pcw_template") or {})
            d = kebab(docs.get("pcw_doc") or {})
            exported = parse_iso(data.get("exported_at", data.get("exported-at"))) or _now()
            target = PcwTarget(None, t.get("id"), d.get("id"))
            docs["pcw_meta"] = pcw_meta(docs.get("pcw_template"), docs.get("pcw_doc"), target,
                                        exported)
        cdir = raw / str(cid)
        written = []
        for source, doc in docs.items():
            dest = cdir / config.RAW_SOURCE_FILES[source]
            if dest.exists() and not overwrite:
                continue
            _write_json(dest, doc)
            written.append(source)
        done = cdir / "_complete.json"
        if "class" in docs and (overwrite or not done.exists()):
            stamp = data.get("exported_at", data.get("exported-at")) or _iso(_now())
            _write_json(done, {"fetched_at": stamp, "source": "route_c",
                               "status": {s: 200 for s in docs}})
        report["classes"][cid] = sorted(written)
        report["files"] += len(written)
    if me and not (scribe_dir / "me.json").exists():
        _write_json(scribe_dir / "me.json", me)
    if write_manifest and report["classes"]:
        from hosts import cli as scribe_cli

        me_doc = _read_json(scribe_dir / "me.json", {}) or {}
        scribe_cli.write_manifest(scribe_dir, me_doc.get("id"))
    return report


# --------------------------------------------------------------------------- CLI


def _selected_class_ids(phase: str | None) -> list[int]:
    import pandas as pd

    path = config.table_path("classes")
    if not path.exists():
        raise HarvestError(f"{path} not found: run `python -m oncue select` or pass --class")
    df = pd.read_parquet(path, columns=["class_id", "phase", "run_order"])
    df = df[df["phase"].notna()]
    if phase:
        df = df[df["phase"] == phase]
    return [int(c) for c in df.sort_values("run_order")["class_id"]]


def cli_harvest(args) -> int:
    """`python -m oncue harvest [--sync] [--no-pcw] [--probe CID] [--class CID ...]`."""
    p = argparse.ArgumentParser(prog="python -m oncue harvest")
    p.add_argument("--sync", action="store_true", help="run `scribe sync` into data/scribe first")
    p.add_argument("--sync-only", action="store_true", help="only `scribe sync` + schedules")
    p.add_argument("--sections", default="past,current")
    p.add_argument("--limit", type=int, default=0, help="max new classes per sync run")
    p.add_argument("--no-pcw", action="store_true", help="class_edit_page only")
    p.add_argument("--refresh", action="store_true", help="refetch cached pages (avoid: D8)")
    p.add_argument("--probe", type=int, action="append", metavar="CID",
                   help="D8 side-effect probe on this non-vault class (no harvest)")
    p.add_argument("--wait", type=float, default=PROBE_WAIT_S, help="probe wait (s)")
    ns = p.parse_args(getattr(args, "extra", None) or [])
    config.load_env()
    if ns.probe:
        from .select import assert_no_vault

        assert_no_vault(ns.probe, what="D8 probe")
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        poster = make_poster()
        for cid in ns.probe:
            report = pcw_side_effect_probe(cid, poster=poster, wait_s=ns.wait)
            _write_json(config.LOGS_DIR / f"pcw_probe_{cid}.json", report)
            print(json.dumps(report, indent=1))
        return 0
    if ns.sync or ns.sync_only:
        rc = run_sync(config.SCRIBE_DIR, client=make_forum_client(rate_per_sec=SYNC_RATE_PER_S),
                      limit=ns.limit, sections=tuple(s for s in ns.sections.split(",") if s))
        if ns.sync_only or rc:
            return rc
    client = make_forum_client()
    ids = [int(c) for c in args.class_ids] if getattr(args, "class_ids", None) else \
        _selected_class_ids(getattr(args, "phase", None))
    summary = harvest_classes(ids, client=client, poster=None if ns.no_pcw else make_poster(),
                              pcw=not ns.no_pcw, refresh=ns.refresh)
    print("harvest: " + ", ".join(f"{k} {v}" for k, v in summary.items()))
    return 1 if summary["failed"] else 0


def cli_import(args) -> int:
    """`python -m oncue harvest-import DIR [--overwrite]` (Route C exports)."""
    p = argparse.ArgumentParser(prog="python -m oncue harvest-import")
    p.add_argument("src", type=Path, help="folder of forum_export/{cid}.json files")
    p.add_argument("--overwrite", action="store_true")
    ns = p.parse_args(getattr(args, "extra", None) or [])
    if not ns.src.is_dir():
        print(f"harvest-import: {ns.src} is not a directory", file=sys.stderr)
        return 2
    report = import_export(ns.src, overwrite=ns.overwrite)
    print(f"harvest-import: {len(report['classes'])} classes, {report['files']} files written"
          + (f"; never written: {sorted(report['skipped_keys'])}" if report["skipped_keys"]
             else ""))
    return 0
