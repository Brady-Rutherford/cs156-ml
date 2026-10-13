"""Shared fixtures for On Cue tests. Synthetic data only: invented people, no real
transcripts, no credentials, no network.

* `oncue` is importable because pipeline1/ is put on sys.path here.
* Every outbound socket connection is refused for the whole session (loopback and Unix
  sockets still work), and Hugging Face libraries are forced offline.
* `ONCUE_ID_SALT` / `ID_SALT` are removed from the environment so no test can pick up a
  real salt; tests that need one set it with `id_salt` (a fixed synthetic value).
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]      # /Users/brady/cs156-ml/pipeline1
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from oncue import config, synthetic

WORLD_CLASSES = 6
WORLD_MINUTES = 30.0
SYNTH_SALT = "synthetic-test-salt"

_real_connect = socket.socket.connect
_LOCAL = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def _guarded_connect(self, address):
    if isinstance(address, str | bytes):          # AF_UNIX
        return _real_connect(self, address)
    host = address[0] if isinstance(address, tuple) else address
    if host in _LOCAL:
        return _real_connect(self, address)
    raise RuntimeError(f"network access is blocked in tests (tried {host!r})")


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    socket.socket.connect = _guarded_connect
    try:
        yield
    finally:
        socket.socket.connect = _real_connect


@pytest.fixture(autouse=True)
def _no_real_salt(monkeypatch):
    for name in config.ID_SALT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _dropped_keys_log_in_tmp(monkeypatch, tmp_path):
    """A loaders.load_raw call without log_path must never write pipeline1/data/logs."""
    monkeypatch.setattr(config, "DROPPED_KEYS_LOG", tmp_path / "logs" / "dropped_keys.csv")


@pytest.fixture
def id_salt(monkeypatch) -> str:
    """Set a synthetic ID salt for the duration of one test."""
    monkeypatch.setenv(config.ID_SALT_ENV[0], SYNTH_SALT)
    return SYNTH_SALT


@pytest.fixture(scope="session")
def world() -> synthetic.World:
    """Six invented 30-minute classes (3 sections; class index 2 has a global-fallback
    calibration, index 3 no PCW, index 4 a PCW edited after start, index 5 two recording
    sessions). Session-scoped: treat it as read-only."""
    return synthetic.world(WORLD_CLASSES, config.SEED, minutes=WORLD_MINUTES)


@pytest.fixture(scope="session")
def small_world() -> synthetic.World:
    """Two short (12-minute) classes for fast tests."""
    return synthetic.world(2, config.SEED + 1, minutes=12.0)


@pytest.fixture(scope="session")
def synth_class(world) -> synthetic.SynthClass:
    """The first class of `world` (single recording session, verified PCW, CS course)."""
    return world.classes[0]


@pytest.fixture(scope="session")
def truth(world) -> dict:
    """{class_id: {table name: DataFrame}} oncue-schema tables built from the truth."""
    return {sc.class_id: synthetic.truth_tables(sc) for sc in world}


@pytest.fixture(scope="session")
def data_dir(world, tmp_path_factory) -> dict:
    """The world written in the real layout (scribe sync raw/ + transcripts/) under a tmp
    dir; returns synthetic.write_world's dict of paths."""
    return synthetic.write_world(world, tmp_path_factory.mktemp("oncue_data"))
