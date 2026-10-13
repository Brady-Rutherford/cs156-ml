"""harvest.py offline: POST allowlist and headers, GET-only SLE client, once-and-cached PCW
harvest, the D8 side-effect probe, class edit pages, the Scribe sync wrapper and Route C
import. Every HTTP call goes to fakes; the conftest also blocks real sockets."""

from __future__ import annotations

import argparse
import csv
import json

import pytest
from oncue import config, harvest, loaders, select, synthetic
from scribe_core.forum import FORUM_BASE, ForumAuthError, ForumHTTPError

COOKIE = "sessionid=SESSIONSECRET; csrftoken=CSRF123; other=1"
SLE_HOST = "sle-collaboration.example.invalid"
TOKEN = "TOKENSECRET"


class FakeResp:
    def __init__(self, status=200, payload=None, headers=None, text=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else ("" if payload is None else json.dumps(payload))

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Records every request. Only .get and .post exist, as on requests.Session."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    def get(self, url, **kw):
        self.calls.append({"method": "GET", "url": url, **kw})
        return self.responder("GET", url, kw)

    def post(self, url, **kw):
        self.calls.append({"method": "POST", "url": url, **kw})
        return self.responder("POST", url, kw)


def pcw_class(world):
    return next(c for c in world if c.pcw["status"] == "verified")


def signed_link(sc) -> str:
    t, d = sc.raw["pcw_template"]["id"], sc.raw["pcw_doc"]["id"]
    return f"https://{SLE_HOST}/app/templates/{t}/doc/{d}?token={TOKEN}&lang=en"


class FakeSle:
    """A fake SLE server (template, doc, optional history) behind a FakeSession."""

    def __init__(self, sc, *, bump_doc=False, history_status=404):
        self.sc, self.bump, self.history_status = sc, bump_doc, history_status
        self.doc_gets = 0
        self.session = FakeSession(self.respond)

    def respond(self, method, url, kw):
        assert method == "GET"
        path = url.split(SLE_HOST, 1)[1].split("?", 1)[0]
        t, d = self.sc.raw["pcw_template"]["id"], self.sc.raw["pcw_doc"]["id"]
        if path.startswith("/app/"):
            return FakeResp(200, None, text="<html>iframe</html>")
        if path == f"/api/v1/templates/{t}/":
            return FakeResp(200, self.sc.raw["pcw_template"],
                            headers={"Last-Modified": "Mon, 05 Oct 2026 10:00:00 GMT"})
        if path == f"/api/v1/doc/{d}/":
            self.doc_gets += 1
            doc = dict(self.sc.raw["pcw_doc"])
            if self.bump and self.doc_gets > 1:
                doc["updated_at"] = "2026-10-13T08:02:00.000Z"
            return FakeResp(200, doc)
        return FakeResp(self.history_status, None)

    def factory(self, target):
        return harvest.sle_client_for(target, session=self.session, rate_per_sec=0)


class FakePoster:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, path, body):
        self.calls.append((path, body))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def all_text(root) -> str:
    return "\n".join(p.read_text(errors="ignore") for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- post_json


def ok_session(payload=None, status=200):
    return FakeSession(lambda m, u, kw: FakeResp(status, payload))


@pytest.mark.parametrize("path", [
    "/api/v1/class-users/7777",
    "/api/v1/pre_class_work/classes/5/pre_class_resource_signed_url?x=1",
    "/api/v1/pre_class_work/classes/5/pre_class_resource_signed_url/../../../class-users/1",
    "/api/v1/pre_class_work/classes/abc/pre_class_resource_signed_url",
    "/api/v1/pre_class_work/classes/5/other",
    "https://forum.minerva.edu/api/v1/pre_class_work/classes/5/pre_class_resource_signed_url",
    "//evil.example/api/v1/pre_class_work/classes/5/pre_class_resource_signed_url",
    "/api/v1/class_grader/classes/5",
])
def test_post_json_refuses_every_other_path(path):
    sess = ok_session({"url": "x"})
    with pytest.raises(harvest.HarvestError, match="allowlisted"):
        harvest.post_json(path, {"iframed": False}, cookie=COOKIE, session=sess)
    assert sess.calls == []


def test_post_json_sends_csrf_referer_cookie_and_json():
    sess = ok_session({"url": "https://x"})
    path = harvest.PCW_SIGNED_URL_PATH.format(cid=900101)
    assert harvest.post_json(path, {"iframed": False}, cookie=COOKIE, session=sess) == \
        {"url": "https://x"}
    (call,) = sess.calls
    assert call["method"] == "POST" and call["url"] == FORUM_BASE + path
    h = call["headers"]
    assert h["X-CSRFToken"] == "CSRF123" and h["Referer"] == "https://forum.minerva.edu/"
    assert h["Cookie"] == COOKIE and h["Content-Type"] == "application/json"
    assert call["json"] == {"iframed": False} and call["allow_redirects"] is False


def test_post_json_auth_and_error_handling(monkeypatch):
    path = harvest.PCW_SIGNED_URL_PATH.format(cid=1)
    monkeypatch.delenv("FORUM_COOKIE", raising=False)
    with pytest.raises(ForumAuthError):
        harvest.post_json(path, {}, session=ok_session())
    with pytest.raises(harvest.HarvestError, match="csrftoken"):
        harvest.post_json(path, {}, cookie="sessionid=abc", session=ok_session())
    monkeypatch.setenv("FORUM_COOKIE", COOKIE)            # read from the env at call time
    assert harvest.post_json(path, {}, session=ok_session({"a": 1})) == {"a": 1}
    for status in (401, 403, 302):
        with pytest.raises(ForumAuthError):
            harvest.post_json(path, {}, session=ok_session({}, status))
    with pytest.raises(ForumHTTPError):
        harvest.post_json(path, {}, session=ok_session({}, 404))
    assert harvest.post_json(path, {}, session=ok_session(None, 204)) is None
    with pytest.raises(harvest.HarvestError):
        harvest.post_json(path, ["not", "a", "dict"], cookie=COOKIE, session=ok_session())


def test_csrf_token_parsing():
    assert harvest.csrf_token(COOKIE) == "CSRF123"
    assert harvest.csrf_token('a=1;csrftoken="Q"') == "Q"
    assert harvest.csrf_token("a=1") is None and harvest.csrf_token(None) is None


def test_forum_poster_throttles():
    sleeps, clock = [], iter([0.0, 0.2, 1.0])
    poster = harvest.ForumPoster(COOKIE, session=ok_session({}), rate_per_sec=1.0,
                                 sleep=sleeps.append, clock=lambda: next(clock))
    path = harvest.PCW_SIGNED_URL_PATH.format(cid=1)
    poster(path, {})
    poster(path, {})
    assert poster.requests_made == 2 and sleeps == [pytest.approx(0.8)]


# --------------------------------------------------------------------------- SLE client


def test_sle_client_is_get_only_and_host_locked():
    for name in ("post", "put", "patch", "delete", "request"):
        assert not hasattr(harvest.SleClient, name)
    for bad in ("http://sle-collaboration.example.invalid/api/v1/",
                "https://forum.minerva.edu/api/v1/", "https://evil-sle-collaboration.x/",
                "https://example.invalid/sle-collaboration/"):
        with pytest.raises(harvest.HarvestError):
            harvest.SleClient(bad, session=FakeSession(None))
    sle = harvest.SleClient(f"https://{SLE_HOST}/api/v1", session=FakeSession(None))
    with pytest.raises(harvest.HarvestError):
        sle.get("https://forum.minerva.edu/api/v1/class-users/1")


def test_sle_redirects_stay_on_sle_hosts():
    def responder(method, url, kw):
        if url.endswith("/a/"):
            return FakeResp(302, None, headers={"Location": "/api/v1/b/"})
        if url.endswith("/b/"):
            return FakeResp(200, {"ok": True}, headers={"Last-Modified": "X"})
        return FakeResp(302, None, headers={"Location": "https://forum.minerva.edu/login"})

    sess = FakeSession(responder)
    sle = harvest.SleClient(f"https://{SLE_HOST}/api/v1/", session=sess, rate_per_sec=0,
                            params={"token": TOKEN})
    r = sle.get("a/")
    assert (r.status, r.data, r.last_modified) == (200, {"ok": True}, "X")
    assert all(c["method"] == "GET" and c["allow_redirects"] is False for c in sess.calls)
    assert sess.calls[0]["params"] == {"token": TOKEN} and "Cookie" not in sess.calls[0]["headers"]
    with pytest.raises(harvest.HarvestError, match="Refusing"):
        sle.get("c/")


@pytest.mark.parametrize("resp,tmpl,doc", [
    ({"url": "https://sle-collaboration.x.invalid/templates/{t}/docs/{d}?token=1"}, True, True),
    ({"signed_url": "https://sle-collaboration.x.invalid/view?template={t}&doc_id={d}"},
     True, True),
    ({"data": {"src": "https://sle-collaboration.x.invalid/#/doc/{d}"}}, False, True),
    ({"url": "https://sle-collaboration.x.invalid/i", "template_id": "{t}", "doc": {"uuid": "{d}"}},
     True, True),
])
def test_locate_pcw_shapes(resp, tmpl, doc):
    t, d = "6f1c2d3e-0000-4000-8000-000000000001", "6f1c2d3e-0000-4000-8000-000000000002"
    resp = json.loads(json.dumps(resp).replace("{t}", t).replace("{d}", d))
    target = harvest.locate_pcw(resp)
    assert target.link.startswith("https://sle-collaboration.x.invalid/")
    assert target.template_uuid == (t if tmpl else None) and target.doc_uuid == (d if doc else None)
    assert "token" not in repr(target)
    assert target.api_base == "https://sle-collaboration.x.invalid/api/v1/"


def test_locate_pcw_without_link():
    assert harvest.locate_pcw({"detail": "no pre-class work"}).link is None
    assert harvest.locate_pcw(None).link is None
    assert harvest.sle_auth_params("https://h/x?token=a&lang=en&jwt=b") == \
        {"token": "a", "jwt": "b"}


# --------------------------------------------------------------------------- PCW harvest


def test_pcw_fetch_writes_cache_once_and_never_the_token(world, tmp_path):
    sc = pcw_class(world)
    sle = FakeSle(sc)
    poster = FakePoster([{"url": signed_link(sc)}])
    when = synthetic._iso(sc.meta["sched_start_ms"] + 86_400_000)
    now = lambda: harvest.parse_iso(when)
    tmpl, doc, meta = harvest.pcw_fetch(sc.class_id, poster=poster, sle_factory=sle.factory,
                                        raw_dir=tmp_path, now=now)
    assert tmpl == sc.raw["pcw_template"] and doc["answers"] == sc.raw["pcw_doc"]["answers"]
    assert poster.calls == [(harvest.PCW_SIGNED_URL_PATH.format(cid=sc.class_id),
                             {"iframed": False})]
    assert meta["template_id"] == sc.raw["pcw_template"]["id"]
    assert meta["doc_id"] == sc.raw["pcw_doc"]["id"]
    assert meta["doc_updated_at"] == sc.raw["pcw_doc"]["updated_at"]
    assert meta["created"] is False and meta["fetched_at"].startswith(when[:16])
    assert all(c["method"] == "GET" for c in sle.session.calls)
    api = [c for c in sle.session.calls if "/api/v1/" in c["url"]]
    assert api and all(c["params"] == {"token": TOKEN} for c in api)
    cdir = tmp_path / str(sc.class_id)
    assert {p.name for p in cdir.iterdir()} == {"pcw_template.json", "pcw_doc.json",
                                                "pcw_meta.json"}
    assert TOKEN not in all_text(tmp_path)
    # cached: a second call makes no request at all
    n_sle, n_post = len(sle.session.calls), len(poster.calls)
    again = harvest.pcw_fetch(sc.class_id, poster=poster, sle_factory=sle.factory,
                              raw_dir=tmp_path)
    assert again[2] == meta and len(sle.session.calls) == n_sle and len(poster.calls) == n_post
    # and the loader whitelist accepts what was written
    loaded = loaders.load_raw(sc.class_id, "pcw_meta", raw_dir=tmp_path, log_path=False)
    assert loaded["doc-id"] == meta["doc_id"]


def test_pcw_fetch_falls_back_to_forums_body_then_caches_no_pcw(world, tmp_path):
    sc = pcw_class(world)
    sle = FakeSle(sc)
    poster = FakePoster([{"detail": "nothing"}, {"url": signed_link(sc)}])
    _, doc, _ = harvest.pcw_fetch(sc.class_id, poster=poster, sle_factory=sle.factory,
                                  raw_dir=tmp_path)
    assert [b for _, b in poster.calls] == list(harvest.PCW_POST_BODIES) and doc is not None
    none = next(c for c in world if c.pcw["status"] == "none")
    poster = FakePoster([ForumHTTPError(404, "/x")])
    tmpl, doc, meta = harvest.pcw_fetch(none.class_id, poster=poster, sle_factory=sle.factory,
                                        raw_dir=tmp_path)
    assert tmpl is None and doc is None and meta["doc_id"] is None and meta["created"] is None
    assert len(poster.calls) == len(harvest.PCW_POST_BODIES)
    harvest.pcw_fetch(none.class_id, poster=poster, sle_factory=sle.factory, raw_dir=tmp_path)
    assert len(poster.calls) == len(harvest.PCW_POST_BODIES)                  # cached


def test_pcw_fetch_errors_cache_nothing(world, tmp_path):
    sc = pcw_class(world)
    poster = FakePoster([{"url": f"https://{SLE_HOST}/app/nothing-here"}])
    with pytest.raises(harvest.HarvestError, match="locate_pcw"):
        harvest.pcw_fetch(sc.class_id, poster=poster, raw_dir=tmp_path,
                          sle_factory=FakeSle(sc).factory)
    deny = FakeSle(sc)
    deny.respond = lambda m, u, kw: FakeResp(401, None)
    deny.session = FakeSession(deny.respond)
    with pytest.raises(harvest.HarvestError, match="401"):
        harvest.pcw_fetch(sc.class_id, poster=FakePoster([{"url": signed_link(sc)}]),
                          raw_dir=tmp_path, sle_factory=deny.factory)
    assert not (tmp_path / str(sc.class_id) / "pcw_meta.json").exists()
    with pytest.raises(ForumAuthError):
        harvest.pcw_fetch(sc.class_id, poster=FakePoster([ForumAuthError(401, "/x")]),
                          raw_dir=tmp_path)


def test_created_flag_when_the_post_creates_the_doc(world, tmp_path):
    sc = pcw_class(world)
    fetched = harvest.parse_iso(sc.raw["pcw_doc"]["created_at"])
    target = harvest.PcwTarget("https://sle-collaboration.x/a", "t", "d")
    meta = harvest.pcw_meta(sc.raw["pcw_template"], sc.raw["pcw_doc"], target, fetched)
    assert meta["created"] is True
    target.created = False
    assert harvest.pcw_meta(None, sc.raw["pcw_doc"], target, fetched)["created"] is False


def test_side_effect_probe_reports_bumps_without_content(world):
    sc = pcw_class(world)
    sle = FakeSle(sc, bump_doc=True, history_status=404)
    slept = []
    report = harvest.pcw_side_effect_probe(
        sc.class_id, poster=FakePoster([{"url": signed_link(sc)}]), sle_factory=sle.factory,
        sleep=slept.append)
    assert slept == [harvest.PROBE_WAIT_S]
    assert report["has_pcw"] and report["has_doc"] and not report["doc_appeared"]
    assert report["meta_changed"] == ["doc_updated_at"]
    assert report["doc_changed"] and not report["template_changed"]
    assert set(report["history"].values()) == {404}
    text = json.dumps(report)
    assert TOKEN not in text and "I argue that" not in text and "Explain how" not in text
    assert all(c["method"] == "GET" for c in sle.session.calls)


def test_side_effect_probe_on_a_class_without_pcw(world):
    report = harvest.pcw_side_effect_probe(
        1, poster=FakePoster([ForumHTTPError(404, "/x")]), sleep=lambda s: None)
    assert report["has_pcw"] is False and report["meta_changed"] == []
    assert set(report["history"].values()) == {None}


# --------------------------------------------------------------------------- class edit page


class FakeForumClient:
    def __init__(self, pages: dict, status: int = 403):
        self.pages, self.status, self.paths = pages, status, []

    def try_get(self, path, params=None):
        self.paths.append(path)
        cid = int(path.split("/")[4])
        return (self.pages[cid], 200) if cid in self.pages else (None, self.status)


def test_class_edit_page_is_fetched_once(world, tmp_path):
    sc = world.classes[0]
    client = FakeForumClient({sc.class_id: sc.raw["class_edit_page"]})
    page = harvest.class_edit_page(client, sc.class_id, raw_dir=tmp_path)
    assert page == sc.raw["class_edit_page"]
    assert client.paths == [f"/api/v1/classes/{sc.class_id}/class_edit_page"]
    assert harvest.class_edit_page(client, sc.class_id, raw_dir=tmp_path) == page
    assert len(client.paths) == 1
    assert harvest.class_edit_page(client, 42, raw_dir=tmp_path) is None
    assert not (tmp_path / "42").exists()


def test_harvest_classes_records_failures_and_stops_on_dead_cookie(world, tmp_path):
    sc = pcw_class(world)
    other = world.classes[1]
    client = FakeForumClient({sc.class_id: sc.raw["class_edit_page"]})
    sle = FakeSle(sc)

    def poster(path, body):
        if str(other.class_id) in path:
            raise ForumHTTPError(500, path)
        return {"url": signed_link(sc)}

    summary = harvest.harvest_classes([sc.class_id, other.class_id], client=client,
                                      poster=poster, sle_factory=sle.factory, raw_dir=tmp_path)
    assert summary["pcw_found"] == 1 and summary["failed"] == 1
    assert (tmp_path / str(other.class_id) / "_harvest_error.json").exists()

    def dead(path, body):
        raise ForumAuthError(401, path)

    with pytest.raises(ForumAuthError):
        harvest.harvest_classes([other.class_id], client=client, poster=dead,
                                sle_factory=sle.factory, raw_dir=tmp_path)


# --------------------------------------------------------------------------- scribe sync


class FakeForum:
    """Stands in for ForumClient during `scribe sync` (synthetic world, no network)."""

    def __init__(self, world):
        self.world = world
        self.bundle_calls = []

    def me(self):
        return dict(self.world.me)

    def sections(self, past=False):
        return [s for s in self.world.sections] if past else []

    def classes(self, sid, **kw):
        return self.world.schedules[sid]["results"]

    def class_bundle(self, cid, extras=True):
        self.bundle_calls.append(cid)
        try:
            sc = self.world.by_id(cid)
        except KeyError:                       # an unrecorded class: no events, no sessions
            return {"class": {"id": cid, "title": "Unrecorded session"}, "events": [],
                    "fetched_at": "2026-10-12T21:00:00Z", "status": {"class": 200}}
        bundle = {k: sc.raw[k] for k in ("class", "events", "polls", "poll_responses",
                                         "breakouts", "outcome_assessments",
                                         "focused_outcomes", "analytics")}
        return {**bundle, "fetched_at": "2026-10-12T21:00:00Z",
                "status": {k: 200 for k in bundle}}


def test_run_sync_writes_raw_manifest_and_full_schedules(world, tmp_path, capsys):
    forum = FakeForum(world)
    out = tmp_path / "scribe"
    assert harvest.run_sync(out, client=forum) == 0
    for sc in world:
        cdir = out / "raw" / str(sc.class_id)
        assert (cdir / "class.json").exists() and (cdir / "_complete.json").exists()
        assert (cdir / "listing.json").exists()
    for sid, sched in world.schedules.items():
        saved = json.loads((out / "schedule" / f"{sid}.json").read_text())
        assert [r["id"] for r in saved["results"]] == [r["id"] for r in sched["results"]]
    rows = list(csv.DictReader((out / "manifest.csv").open()))
    assert {int(r["class_id"]) for r in rows} >= set(world.class_ids)
    n = len(forum.bundle_calls)
    assert harvest.run_sync(out, client=forum) == 0 and len(forum.bundle_calls) == n  # resumable
    # select runs straight off the synced layout, numbering sessions from the full schedule
    classes = select.build_classes(out / "manifest.csv", out / "raw",
                                   schedule_dir=out / "schedule", me_path=out / "me.json",
                                   hasher=lambda u: "h", log_path=False)
    recorded = classes[classes["n_rec_sessions"] > 0].set_index("class_id")
    for sc in world:
        assert recorded.loc[sc.class_id, "session_no"] == sc.meta["session_no"]
        assert recorded.loc[sc.class_id, "course_code"] == sc.meta["course_code"]
    assert (classes["n_rec_sessions"] == 0).sum() == len(world.schedules)   # unrecorded rows


# --------------------------------------------------------------------------- Route C


def export_file(sc, folder):
    data = {"class_id": sc.class_id, "exported_at": "2026-10-13T07:00:00.000Z",
            "me": {"id": synthetic.ME_ID, "role": "student", "first_name": "Brady"},
            "class": sc.raw["class"], "class_events": sc.raw["events"], "polls": sc.raw["polls"],
            "poll-responses": sc.raw["poll_responses"], "breakouts": sc.raw["breakouts"],
            "class_edit_page": sc.raw["class_edit_page"],
            "analytics": sc.raw["analytics"],
            "outcome_assessments": sc.raw["outcome_assessments"],
            "recording_urls": {"lowres": "https://d1.cloudfront.net/r.mp4?Expires=1&Signature=SIG"}}
    if sc.raw.get("pcw_template"):
        data["template"] = sc.raw["pcw_template"]
        data["doc"] = sc.raw["pcw_doc"]
        data["pcw_signed_url"] = {"url": signed_link(sc)}
    path = folder / f"{sc.class_id}.json"
    path.write_text(json.dumps(data))
    return path


def test_import_export_builds_the_raw_layout(world, data_dir, tmp_path):
    src = tmp_path / "forum_export"
    src.mkdir()
    for sc in world:
        export_file(sc, src)
    raw = tmp_path / "scribe" / "raw"
    report = harvest.import_export(src, raw_dir=raw)
    assert set(report["classes"]) == set(world.class_ids)
    assert {"analytics", "outcome-assessments", "recording-urls", "pcw-signed-url"} <= \
        set(report["skipped_keys"])
    text = all_text(tmp_path / "scribe")
    assert "Signature=" not in text and TOKEN not in text and "Expires=" not in text
    assert "example.invalid/signed?x=1" not in text           # upload file url scrubbed
    for sc in world:
        cdir = raw / str(sc.class_id)
        names = {p.name for p in cdir.iterdir()}
        assert not names & {"analytics.json", "outcome_assessments.json"}
        assert {"class.json", "events.json", "listing.json", "_complete.json",
                "class_edit_page.json"} <= names
        assert ("pcw_meta.json" in names) == (sc.raw.get("pcw_template") is not None)
        for source in ("class", "events", "class_edit_page"):
            assert loaders.load_raw(sc.class_id, source, raw_dir=raw, log_path=False) == \
                loaders.load_raw(sc.class_id, source, raw_dir=data_dir["raw"], log_path=False)
    me = json.loads((tmp_path / "scribe" / "me.json").read_text())
    assert me == {"id": synthetic.ME_ID, "role": "student"}
    rows = list(csv.DictReader((tmp_path / "scribe" / "manifest.csv").open()))
    assert {int(r["class_id"]) for r in rows} == set(world.class_ids)
    meta = json.loads((raw / str(pcw_class(world).class_id) / "pcw_meta.json").read_text())
    assert meta["doc_id"] == pcw_class(world).raw["pcw_doc"]["id"]


def test_import_export_keeps_existing_files_unless_overwrite(world, tmp_path):
    sc = world.classes[0]
    src = tmp_path / "exp"
    src.mkdir()
    export_file(sc, src)
    raw = tmp_path / "raw"
    cdir = raw / str(sc.class_id)
    cdir.mkdir(parents=True)
    (cdir / "class.json").write_text('{"id": 1}')
    report = harvest.import_export(src, raw_dir=raw, write_manifest=False)
    assert "class" not in report["classes"][sc.class_id]
    assert json.loads((cdir / "class.json").read_text()) == {"id": 1}
    harvest.import_export(src, raw_dir=raw, overwrite=True, write_manifest=False)
    assert json.loads((cdir / "class.json").read_text())["id"] == sc.class_id


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Point every data path at tmp and make sure no .env is ever read."""
    calls = []
    monkeypatch.setattr(config, "load_env", lambda *a, **k: calls.append(1) or False)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "scribe" / "raw")
    monkeypatch.setattr(config, "SCRIBE_DIR", tmp_path / "scribe")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "DROPPED_KEYS_LOG", tmp_path / "logs" / "dropped_keys.csv")
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path / "tables")
    return calls


def ns(**kw):
    base = {"command": "harvest", "phase": None, "workers": 1, "class_ids": None, "extra": []}
    return argparse.Namespace(**{**base, **kw})


def test_cli_harvest_with_injected_clients(world, monkeypatch, cli_env, tmp_path):
    sc = pcw_class(world)
    sle = FakeSle(sc)
    monkeypatch.setattr(harvest, "make_forum_client",
                        lambda **k: FakeForumClient({sc.class_id: sc.raw["class_edit_page"]}))
    monkeypatch.setattr(harvest, "make_poster", lambda **k: FakePoster([{"url": signed_link(sc)}]))
    real = harvest.sle_client_for
    monkeypatch.setattr(harvest, "sle_client_for",
                        lambda t, **k: real(t, session=sle.session, rate_per_sec=0))
    assert harvest.cli_harvest(ns(class_ids=[str(sc.class_id)])) == 0
    cdir = tmp_path / "scribe" / "raw" / str(sc.class_id)
    assert (cdir / "class_edit_page.json").exists() and (cdir / "pcw_meta.json").exists()
    assert cli_env, "the CLI loads .env (stubbed here) before building real clients"


def test_cli_sync_only_uses_scribes_rate(world, monkeypatch, cli_env, tmp_path, capsys):
    made = []

    def factory(**kw):
        made.append(kw)
        return FakeForum(world)

    monkeypatch.setattr(harvest, "make_forum_client", factory)
    assert harvest.cli_harvest(ns(extra=["--sync-only", "--sections", "past"])) == 0
    assert made == [{"rate_per_sec": harvest.SYNC_RATE_PER_S}]
    assert (tmp_path / "scribe" / "manifest.csv").exists()
    assert len(list((tmp_path / "scribe" / "schedule").glob("*.json"))) == len(world.schedules)


def test_cli_harvest_without_classes_table_fails_cleanly(cli_env, monkeypatch):
    monkeypatch.setattr(harvest, "make_forum_client", lambda **k: FakeForumClient({}))
    with pytest.raises(harvest.HarvestError, match="select"):
        harvest.cli_harvest(ns())


def test_cli_probe_refuses_vault_classes(cli_env):
    vault = next(c for c in range(1, 200) if config.in_vault(c))
    with pytest.raises(select.VaultError):
        harvest.cli_harvest(ns(extra=["--probe", str(vault)]))


def test_cli_import(world, cli_env, tmp_path, capsys):
    src = tmp_path / "exp"
    src.mkdir()
    export_file(world.classes[0], src)
    assert harvest.cli_import(ns(command="harvest-import", extra=[str(src)])) == 0
    assert (tmp_path / "scribe" / "raw" / str(world.classes[0].class_id) / "class.json").exists()
    assert "never written" in capsys.readouterr().out
    assert harvest.cli_import(ns(command="harvest-import", extra=[str(tmp_path / "no")])) == 2


def test_cli_dispatch_resolves_harvest():
    from oncue import __main__ as entry

    assert entry.resolve("harvest") is harvest.cli_harvest
    assert entry.resolve("harvest-import") is harvest.cli_import
    assert entry.resolve("select") is select.cli
