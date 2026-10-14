"""Label detector: Brady's raised-hand icon on his own Forum tile (spec §2, amendment A1).

No face recognition anywhere: Brady's tile is found by NCC on binarised *name-label* glyphs,
and the hand state is a hue + shape test of the tile's top-left corner square.

Stages (every stage after pass 2 runs offline from `hand_tiles`, with no re-streaming):

* **pass 1** (`pass1`): strip auto-detection on 6 frames (`detect_strip`), tile spans from dark
  gutters (`strip_tiles`), and *anchors*: glyph patches (`glyph_patch`) of the bordered tile
  in frames inside Brady's isolated, word-bearing main-room voice events (`anchor_times`).
* **templates** (`build_era_templates`): leave-one-class-out pixelwise median of the *other*
  classes' anchors in the same UI era. A class's own anchors only validate
  (`validate_tracking`), so tracking never depends on the class's own speech.
* **pass 2** (`scan`): per frame and tile at 2 fps -> `hand_tiles` (corner-ROI hue fractions
  at s1/s2, shape scores, border score, NCC, packed 12 x 48 glyph bits). Frames come from an
  injectable `(pts_s, frame)` iterator; the real one (`FfmpegStripReader`) pipes ffmpeg
  (`-copyts` + `showinfo`) over a fresh signed URL that is never logged.
* `track`: NCC tracking (with a LOCO NCC floor against lookalike name labels), the per-class
  DOWN baseline, hysteresis raw states and the causal live state, `n_others` ->
  `hand_frames`. Two raw states are stored: `raw_state` (whole-class two-pass baseline,
  label side: episodes, observability, QA) and `raw_state_live` (running baseline over
  samples <= t only), which alone feeds `live_state` and every causal rule / `me_hand_*`.
* `episodes`: the retrospective state machine on *confirmed* runs (NA bridging, D_pre,
  merges, glitches, left-censoring), vote typing (Δn_others or the VOTE lexicon), outcomes
  -> `hand_episodes` (label side only).
  `retime` refines onsets from a full-frame-rate re-read; `av_lag` measures the per-class
  A/V lag (speaker border vs ASR speech).
* audit helpers (`contact_sheet`, `audit_clip`, `onset_clips`) write only under `audit/`,
  pixelate every tile interior except Brady's tracked tile, and refuse vault classes.

Times are master-clock seconds: frame time = pts_time - audio start_time (§2.1).
Constants marked PROVISIONAL (in `UiSpec` and `Detector`) are placeholders until D2/D3.
"""

from __future__ import annotations

import base64
import bisect
import csv
import json
import logging
import math
import queue
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import causal, config, schemas
from .labels import retime_summary

log = logging.getLogger("oncue.hands")

UP, DOWN, NA = config.STATE_UP, config.STATE_DOWN, config.STATE_NA
_DOWN_C, _UP_C, _NA_C = causal.DOWN_C, causal.UP_C, causal.NA_C   # integer state codes
_STATE_NAMES = causal.STATE_NAMES
GLYPH_BITS = config.GLYPH_H * config.GLYPH_W         # 576 bits = 72 packed bytes
GLYPH_BYTES = GLYPH_BITS // 8
GLYPH_MIN_CONTRAST = 40                              # PROVISIONAL: below this a label is blank
SHAPE_MIN_AREA_PX = 6                                # smaller hue blobs never pass the shape test
STRIP_ACF_MIN = 0.2                                  # PROVISIONAL: gutter periodicity at the pitch
ANCHOR_EDGE_S = 1.0                                  # anchor frames sit >= 1 s inside the event
ANCHOR_PAD_S = 0.25                                  # isolation margin from other speakers' events
AUDIT_HEADER_PX = 18


# --------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class UiSpec:
    """Forum strip geometry and colours that D2 measures. Every default is PROVISIONAL and is
    set from the synthetic world, which mirrors the layout Brady described."""

    label_h_frac: float = 13 / 64           # label bar height / tile height
    label_inset_px: int = config.LABEL_INSET_PX
    label_icon_right_px: int = 8            # mic / mute / signal icon columns, right of label
    roi_s1: float = config.ROI_S1
    roi_s2: float = config.ROI_S2
    roi_inset_px: int = config.ROI_INSET_PX
    ring_px: int = 3                        # speaker-border ring width (= ROI inset)
    border_h_range: tuple[int, int] = (45, 70)   # OpenCV hue of the speaker border
    border_sv_min: int = 115
    min_tile_w: int = 30                    # D2 pass rule: tiles >= 30 px wide
    shape_gate: float = 0.02                # shape test runs only where frac_hand >= this
    strip_row_dark_min: float = 0.9         # share of gutter pixels dark in a strip row


@dataclass(frozen=True)
class Detector:
    """Detector thresholds (§1.2). Everything here can be retuned offline from `hand_tiles`."""

    fps: float = config.F_SCAN
    theta_on: float = config.THETA_ON
    theta_off: float = config.THETA_OFF
    roi: str = "s1"                         # which ROI side ("s1" | "s2") drives the state
    circ_min: float = config.SHAPE_CIRCULARITY_MIN
    glyph_frac: tuple[float, float] = config.SHAPE_GLYPH_FRAC
    tick_na: bool = True                    # PROVISIONAL (D2): a visible poll tick => NA
    k: int = config.K_CONFIRM
    ncc_min: float = config.NCC_MIN
    ncc_gap: float = config.NCC_GAP
    border_min: float = 0.30                # PROVISIONAL: ring share that counts as bordered
    d_pre_s: float = config.D_PRE_S
    na_bridge_s: float = config.NA_BRIDGE_S
    merge_gap_s: float = config.DOWN_GAP_MERGE_S
    min_episode_s: float = config.MIN_EPISODE_S
    me_hand_min_s: float = config.ME_HAND_MIN_S
    me_hand_min_shape: int = config.ME_HAND_MIN_SHAPE_SAMPLES
    na_hold: int = causal.NA_HOLD           # live state holds through NA runs this short
    # Real UI (`oncue.strip`): the raise animation can hide the name label at the onset, so
    # an onset reached through an NA run of at most this length, itself preceded by D_pre of
    # confirmed DOWN, is not left-censored; its onset is the start of that NA run (the
    # earliest the raise can have begun) and re-timing finds the first frame. 0 = off.
    pre_na_onset_s: float = 0.0
    # Votes (§2.5): Δn_others >= max(vote_min_dn, ceil(vote_frac * (n_tiles - 1))) within
    # VOTE_DN_WINDOW_S, or the professor's VOTE lexicon.
    vote_min_dn: int = config.VOTE_MIN_DN
    vote_frac: float = config.VOTE_FRAC
    # An UP run shorter than k samples ("blip") never starts, extends or ends an episode.
    # The spec's glitch ("confirmed episode < 1.0 s") cannot happen at k = 2 / 2 fps, so a
    # blip that is isolated from every confirmed UP run (DOWN gaps >= merge_gap_s on both
    # sides) is typed `glitch` (its window is undefined: it may be a brief real raise);
    # blips near a real raise are dropped as detector noise. Every DOWN-ended blip starts
    # the causal R_low (`post_glitch`) at its end, decided from samples <= t.
    blips_as_glitches: bool = True
    # PROVISIONAL (D3): LOCO anchor-NCC percentile and margin of the impostor floor; D3
    # checks the coverage it costs on Brady's own frames (NA rate by tile state and width)
    ncc_floor_q: float = 5.0
    ncc_floor_margin: float = 0.02


UI = UiSpec()
DETECTOR = Detector()


@dataclass(frozen=True)
class StripGeometry:
    """Where the participant strip sits in a frame (pass 1). Tile rows are relative to the
    strip; tile x-spans are found per frame because the strip reflows."""

    found: bool
    strip_y: int = 0
    strip_h: int = 0
    tile_y0: int = 0
    tile_h: int = 0
    n_tiles: int = 0
    pitch: float = 0.0
    n_frames_found: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StripGeometry:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class ScanError(RuntimeError):
    """ffmpeg / ffprobe failed. Messages never contain the signed URL."""


# --------------------------------------------------------------------------- pixel helpers


def _value(frame: np.ndarray) -> np.ndarray:
    """HSV V channel in [0, 1] (= max of B, G, R)."""
    b, g, r = cv2.split(np.ascontiguousarray(frame))
    return cv2.max(cv2.max(b, g), r).astype(np.float32) / 255.0


def _runs(mask: np.ndarray) -> np.ndarray:
    """(start, end) pairs of the True runs in a 1-D bool array (end exclusive)."""
    d = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1)]


def _rle(codes: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length encoding: [(value, start, end)] with end exclusive."""
    n = len(codes)
    if n == 0:
        return []
    cut = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, n]
    return [(int(codes[a]), int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def _gt(x: float) -> int:
    """Smallest uint8 value strictly greater than x."""
    return math.floor(x) + 1


def _lt(x: float) -> int:
    """Largest uint8 value strictly less than x."""
    return math.ceil(x) - 1


def hsv_masks(img_bgr: np.ndarray, ui: UiSpec = UI) -> dict[str, np.ndarray]:
    """Boolean masks (same shape as the image) for the hand hue, the poll-tick hue, white
    glyph pixels and the speaker-border colour (§2.3, OpenCV HSV with H in 0..179), plus
    "v", the V channel in [0, 1]. Strict inequalities of §2.3 map to integer bounds."""
    hsv = cv2.cvtColor(np.ascontiguousarray(img_bgr), cv2.COLOR_BGR2HSV)
    sv = _gt(config.HAND_SV_MIN)
    (h0, h1), (t0, t1), (b0, b1) = config.HAND_H_RANGE, config.TICK_H_RANGE, ui.border_h_range
    bsv = _gt(ui.border_sv_min)

    def rng(lo, hi):
        return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0

    return {
        "hand": rng((h0, sv, sv), (h1, 255, 255)),
        "tick": rng((t0, sv, sv), (t1, 255, 255)),
        "white": rng((0, 0, _gt(config.WHITE_V_MIN)), (179, _lt(config.WHITE_S_MAX), 255)),
        "border": rng((b0, bsv, bsv), (b1, 255, 255)),
        "v": hsv[..., 2].astype(np.float32) / 255.0,
    }


# --------------------------------------------------------------------------- strip and tiles


def _tiles_from_profile(profile: np.ndarray, min_w: int = UI.min_tile_w) -> np.ndarray:
    """Tile spans (n, 2) from a column-mean V profile: gutters are runs of >= 2 px below
    GUTTER_V_MAX; tiles are the spans between gutters within ±20% of the median width."""
    gut = _runs(profile < config.GUTTER_V_MAX)
    gut = gut[(gut[:, 1] - gut[:, 0]) >= config.GUTTER_MIN_PX]
    if len(gut) < 2:
        return np.zeros((0, 2), dtype=int)
    spans = np.c_[gut[:-1, 1], gut[1:, 0]]
    width = spans[:, 1] - spans[:, 0]
    wide = width >= min_w
    if not wide.any():
        return np.zeros((0, 2), dtype=int)
    med = float(np.median(width[wide]))
    keep = wide & (np.abs(width - med) <= config.TILE_WIDTH_TOL * med)
    return spans[keep].astype(int)


def strip_tiles(strip: np.ndarray, *, ui: UiSpec = UI) -> np.ndarray:
    """Tile x-spans `[[x0, x1), ...]` in one strip image (BGR, strip rows only), §2.2 step 2."""
    return _tiles_from_profile(_value(strip).mean(axis=0), ui.min_tile_w)


def _gutter_periodicity(profile: np.ndarray, tiles: np.ndarray) -> tuple[float, float]:
    """(pitch, autocorrelation of the dark-gutter indicator at the pitch)."""
    if len(tiles) < 2:
        return 0.0, 0.0
    pitch = float(np.median(np.diff(tiles[:, 0])))
    a, b = max(0, int(tiles[0, 0]) - 8), min(len(profile), int(tiles[-1, 1]) + 8)
    d = (profile[a:b] < config.GUTTER_V_MAX).astype(np.float64)
    d -= d.mean()
    denom = float((d * d).sum())
    if denom <= 0:
        return pitch, 0.0
    lags = range(max(1, int(0.8 * pitch)), min(len(d) - 1, math.ceil(1.2 * pitch)) + 1)
    acf = [float((d[:-lag] * d[lag:]).sum()) / denom for lag in lags if lag < len(d)]
    return pitch, max(acf, default=0.0)


def _detect_one(frame: np.ndarray, heights: Sequence[int], ui: UiSpec) -> dict | None:
    """Strip candidate in one full frame, or None."""
    V = _value(frame)
    H, _W = V.shape
    C = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, axis=0, dtype=np.float64)])
    best = None
    for h in heights:
        if h < 4 or h >= H:
            continue
        M = (C[h:] - C[:-h]) / h                       # band column means, one row per y
        dark = M < config.GUTTER_V_MAX
        pair = dark[:, 1:] & dark[:, :-1]
        n_runs = pair[:, 0].astype(int) + (pair[:, 1:] & ~pair[:, :-1]).sum(axis=1)
        for y in np.flatnonzero(n_runs >= 3)[:: max(1, h // 12)]:
            tiles = _tiles_from_profile(M[y], ui.min_tile_w)
            if len(tiles) < 2:
                continue
            pitch, acf = _gutter_periodicity(M[y], tiles)
            if acf < STRIP_ACF_MIN:
                continue
            key = (len(tiles), acf, h)
            if best is None or key > best[0]:
                best = (key, int(y), int(h), tiles, pitch)
    if best is None:
        return None
    _, y, h, tiles, pitch = best
    dark2d = V < config.GUTTER_V_MAX
    gut_cols = np.zeros(V.shape[1], bool)
    for (_, a), (b, _) in pairwise(tiles):
        gut_cols[a:b] = True                               # interior gutters only
    tile_cols = np.zeros(V.shape[1], bool)
    for a, b in tiles:
        tile_cols[a:b] = True
    if not gut_cols.any():
        return None
    row_ok = dark2d[:, gut_cols].mean(axis=1) >= ui.strip_row_dark_min
    inside = np.flatnonzero(row_ok[y:y + h])
    if len(inside) == 0:
        return None
    yc = y + int(inside[len(inside) // 2])
    y0 = yc
    while y0 > 0 and row_ok[y0 - 1]:
        y0 -= 1
    y1 = yc + 1
    while y1 < H and row_ok[y1]:
        y1 += 1
    strip_h = y1 - y0
    row_dark = dark2d[y0:y1][:, tile_cols].mean(axis=1) >= ui.strip_row_dark_min
    margin = 0
    while margin < strip_h // 3 and row_dark[margin]:
        margin += 1
    tile_h = strip_h - 2 * margin            # PROVISIONAL: tiles centred vertically in the strip
    if tile_h < 8:
        return None
    return {"strip_y": int(y0), "strip_h": int(strip_h), "tile_y0": int(margin),
            "tile_h": int(tile_h), "n_tiles": len(tiles), "pitch": float(pitch)}


def detect_strip(frames: Sequence[np.ndarray], *, expected_h: float | None = None,
                 n_present: int | Sequence[int] | None = None, ui: UiSpec = UI
                 ) -> StripGeometry:
    """Auto-detect the participant strip in a few full frames of one class (§2.2 step 1).

    Searches horizontal bands whose column-mean V shows periodic dark gutters (an
    autocorrelation peak at the tile pitch), refines the band to the rows where the gutter
    columns stay dark, and checks `n_tiles` within ±2 of `n_present` (per frame or one
    value) and the height within ±30% of `expected_h` (from D2) when given. A fixed y is
    never used: if no frame qualifies, `found` is False (tile_status "unlocated").
    """
    if len(frames) == 0:
        return StripGeometry(found=False, reason="no_frames")
    H = int(frames[0].shape[0])
    if expected_h:
        lo, hi = (1 - config.STRIP_H_TOL) * expected_h, (1 + config.STRIP_H_TOL) * expected_h
        heights = sorted({round(x) for x in np.linspace(0.6 * lo, hi, 6)})
    else:
        lo, hi = 0.0, float(H)
        heights = sorted({round(x) for x in np.geomspace(24, max(25, H // 3), 7)})
    per_present = (list(n_present) if n_present is not None and np.ndim(n_present) > 0
                   else [n_present] * len(frames))
    if len(per_present) != len(frames):
        raise ValueError("n_present needs one value per frame (or a single value)")
    found, reasons = [], []
    for frame, present in zip(frames, per_present, strict=True):
        r = _detect_one(frame, heights, ui)
        if r is None:
            reasons.append("no_band")
            continue
        if expected_h and not lo <= r["strip_h"] <= hi:
            reasons.append("height")
            continue
        if present is not None and abs(r["n_tiles"] - int(present)) > config.N_TILES_TOL:
            reasons.append("n_present")
            continue
        found.append(r)
    if not found:
        reason = max(set(reasons), key=reasons.count) if reasons else "no_band"
        return StripGeometry(found=False, reason=reason)
    keys = [(r["strip_y"], r["strip_h"]) for r in found]
    sy, sh = max(set(keys), key=keys.count)
    same = [r for r in found if (r["strip_y"], r["strip_h"]) == (sy, sh)]
    med = {k: float(np.median([r[k] for r in same])) for k in ("tile_y0", "tile_h", "n_tiles",
                                                                   "pitch")}
    return StripGeometry(found=True, strip_y=sy, strip_h=sh, tile_y0=int(med["tile_y0"]),
                         tile_h=int(med["tile_h"]), n_tiles=round(med["n_tiles"]),
                         pitch=med["pitch"], n_frames_found=len(found), reason="ok")


def as_strip(frame: np.ndarray, geometry: StripGeometry) -> np.ndarray:
    """The strip rows of a full frame (frames already cropped to the strip pass through)."""
    if frame.shape[0] == geometry.strip_h:
        return frame
    return frame[geometry.strip_y:geometry.strip_y + geometry.strip_h]


# --------------------------------------------------------------------------- glyphs and NCC


def _glyph_from_gray(gray_tile: np.ndarray, ui: UiSpec = UI) -> np.ndarray:
    """Binarised 12 x 48 name-glyph patch from a grey tile image (see `glyph_patch`)."""
    th, tw = gray_tile.shape[:2]
    out = np.zeros((config.GLYPH_H, config.GLYPH_W), dtype=bool)
    lh = max(3, round(ui.label_h_frac * th))
    ins = ui.label_inset_px
    y0, y1 = th - lh, th - ins
    x0, x1 = ins, tw - max(ins, ui.label_icon_right_px)
    if y1 - y0 < 2 or x1 - x0 < 4:
        return out
    label = gray_tile[y0:y1, x0:x1]
    if int(label.max()) - int(label.min()) < GLYPH_MIN_CONTRAST:
        return out
    _, bw = cv2.threshold(np.ascontiguousarray(label), 0, 1,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape
    new_w = max(1, round(w * config.GLYPH_H / h))
    interp = cv2.INTER_AREA if h > config.GLYPH_H else cv2.INTER_LINEAR
    r = cv2.resize(bw.astype(np.float32), (new_w, config.GLYPH_H), interpolation=interp) >= 0.5
    out[:, :min(config.GLYPH_W, new_w)] = r[:, :config.GLYPH_W]
    return out


def glyph_patch(tile: np.ndarray, *, ui: UiSpec = UI) -> np.ndarray:
    """Name-label glyph patch of one tile (BGR): the label bar inset by 4 px (no border ring),
    icon columns removed, Otsu-binarised, resized to 12 px high, left-aligned 48 px crop.
    Depends on the name text only (not mute state, border, hand, video or tile width)."""
    gray = tile if tile.ndim == 2 else cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return _glyph_from_gray(gray, ui)


def pack_glyph(patch: np.ndarray) -> bytes:
    """12 x 48 bool patch -> 72 bytes."""
    return np.packbits(np.asarray(patch, dtype=bool).ravel()).tobytes()


def unpack_glyphs(bits: Iterable[bytes]) -> np.ndarray:
    """Packed glyph bytes -> (n, 12, 48) bool. Anything that is not 72 bytes -> all zeros."""
    rows = [b if isinstance(b, bytes | bytearray) and len(b) == GLYPH_BYTES
            else bytes(GLYPH_BYTES) for b in bits]
    if not rows:
        return np.zeros((0, config.GLYPH_H, config.GLYPH_W), dtype=bool)
    buf = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), GLYPH_BYTES)
    return np.unpackbits(buf, axis=1)[:, :GLYPH_BITS].reshape(
        len(rows), config.GLYPH_H, config.GLYPH_W).astype(bool)


def ncc(patches: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalised cross-correlation of each patch with the template (0 when either
    is constant). `patches` is (n, 12, 48) or (12, 48)."""
    P = np.asarray(patches, dtype=np.float32).reshape(-1, GLYPH_BITS)
    T = np.asarray(template, dtype=np.float32).ravel()
    P = P - P.mean(axis=1, keepdims=True)
    T = T - T.mean()
    den = np.sqrt((P * P).sum(axis=1) * float((T * T).sum()))
    num = P @ T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def pick_tile(scores: np.ndarray, det: Detector = DETECTOR, ncc_floor: float | None = None
              ) -> tuple[int, float, float]:
    """(tile index or -1, best NCC, gap to the runner-up) under NCC_min / NCC_gap (§2.2.7)
    and the class's impostor floor (`ncc_floors`)."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return -1, float("nan"), float("nan")
    order = np.argsort(-scores)
    best = float(scores[order[0]])
    gap = best - (float(scores[order[1]]) if len(scores) > 1 else -1.0)
    floor = det.ncc_min if ncc_floor is None else max(det.ncc_min, float(ncc_floor))
    ok = best >= floor and gap >= det.ncc_gap
    return (int(order[0]) if ok else -1), best, gap


# --------------------------------------------------------------------------- icon tests


def border_score(tile: np.ndarray, *, ui: UiSpec = UI, mask: np.ndarray | None = None) -> float:
    """Share of the tile's outer ring (ring_px wide) in the speaker-border colour."""
    m = hsv_masks(tile, ui)["border"] if mask is None else mask
    h, w = m.shape
    r = ui.ring_px
    if h <= 2 * r or w <= 2 * r:
        return 0.0
    ring_n = h * w - (h - 2 * r) * (w - 2 * r)
    return float(m.sum() - m[r:h - r, r:w - r].sum()) / ring_n


def shape_score(hue_mask: np.ndarray, white_mask: np.ndarray) -> tuple[float, float]:
    """(circularity 4πA/P² of the largest hue-passing component, white-glyph share of its
    enclosing circle); (0, 0) when there is no usable component."""
    m = np.ascontiguousarray(hue_mask, dtype=np.uint8)
    if int(m.sum()) < SHAPE_MIN_AREA_PX:
        return 0.0, 0.0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0.0, 0.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < SHAPE_MIN_AREA_PX:
        return 0.0, 0.0
    contours, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    area, perim = float(cv2.contourArea(c)), float(cv2.arcLength(c, True))
    if area <= 0 or perim <= 0:
        return 0.0, 0.0
    (cx, cy), rad = cv2.minEnclosingCircle(c)
    yy, xx = np.mgrid[: m.shape[0], : m.shape[1]]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
    white = float((np.asarray(white_mask, bool) & inside).sum()) / max(int(inside.sum()), 1)
    return 4.0 * math.pi * area / perim ** 2, white


def shape_ok(circ, white, det: Detector = DETECTOR) -> np.ndarray:
    """Vectorised shape test: circularity >= circ_min and white share within glyph_frac.
    A NaN white share (tables without white_* columns) skips the glyph criterion."""
    circ = np.asarray(circ, dtype=float)
    white = np.asarray(white, dtype=float)
    lo, hi = det.glyph_frac
    glyph = np.isnan(white) | ((white >= lo) & (white <= hi))
    return (np.nan_to_num(circ, nan=0.0) >= det.circ_min) & glyph


def shape_test(hue_mask: np.ndarray, white_mask: np.ndarray, det: Detector = DETECTOR) -> bool:
    """§2.3 shape test on one ROI: roughly circular blob containing a white glyph."""
    circ, white = shape_score(hue_mask, white_mask)
    return bool(shape_ok(circ, white, det))


def _roi_slices(tile_y0: int, tile_h: int, x0: int, x1: int, s: float, ui: UiSpec
                ) -> tuple[slice, slice]:
    side = round(s * tile_h)
    r0 = tile_y0 + ui.roi_inset_px
    c0 = x0 + ui.roi_inset_px
    return (slice(r0, min(r0 + side, tile_y0 + tile_h)), slice(c0, min(c0 + side, x1)))


# --------------------------------------------------------------------------- per-frame stats

_TILE_FLOAT_COLS = ("frac_hand_s1", "frac_hand_s2", "frac_tick_s1", "frac_tick_s2", "shape_s1",
                    "shape_s2", "white_s1", "white_s2", "border", "ncc_me")


def _rect_sums(integral: np.ndarray, r0, r1, c0, c1) -> np.ndarray:
    """Pixel counts of rectangles [r0, r1) x [c0, c1) from a cv2.integral image."""
    r0, r1, c0, c1 = (np.asarray(v, dtype=int) for v in (r0, r1, c0, c1))
    return (integral[r1, c1] - integral[r0, c1] - integral[r1, c0]
            + integral[r0, c0]).astype(float)


def frame_stats(strip: np.ndarray, geometry: StripGeometry, *, ui: UiSpec = UI,
                template: np.ndarray | None = None, tiles: np.ndarray | None = None
                ) -> dict[str, np.ndarray]:
    """Per-tile statistics of one strip frame: x0, x1, frac_hand/frac_tick at s1 and s2,
    shape (circularity) and white share at s1 and s2, border score, NCC against `template`
    and the 12 x 48 glyph patch."""
    masks = hsv_masks(strip, ui)
    tiles = (_tiles_from_profile(masks["v"].mean(axis=0), ui.min_tile_w) if tiles is None
             else np.asarray(tiles, dtype=int).reshape(-1, 2))
    n = len(tiles)
    out = {c: np.full(n, np.nan, dtype=np.float32) for c in _TILE_FLOAT_COLS}
    out["x0"] = tiles[:, 0].astype(int)
    out["x1"] = tiles[:, 1].astype(int)
    out["glyph"] = np.zeros((n, config.GLYPH_H, config.GLYPH_W), dtype=bool)
    if n == 0:
        return out
    gray = cv2.cvtColor(np.ascontiguousarray(strip), cv2.COLOR_BGR2GRAY)
    ty0, th = geometry.tile_y0, geometry.tile_h
    x0s, x1s = out["x0"], out["x1"]
    # ROI and ring pixel counts for all tiles at once from integral images
    integ = {m: cv2.integral(masks[m].view(np.uint8)) for m in ("hand", "tick", "border")}
    for tag, s in (("s1", ui.roi_s1), ("s2", ui.roi_s2)):
        r0 = ty0 + ui.roi_inset_px
        r1 = min(r0 + round(s * th), ty0 + th)
        c0 = x0s + ui.roi_inset_px
        c1 = np.minimum(c0 + round(s * th), x1s)
        area = np.maximum((r1 - r0) * (c1 - c0), 0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            fh = np.where(area > 0, _rect_sums(integ["hand"], r0, r1, c0, c1) / area, np.nan)
            ft = np.where(area > 0, _rect_sums(integ["tick"], r0, r1, c0, c1) / area, np.nan)
        out[f"frac_hand_{tag}"][:], out[f"frac_tick_{tag}"][:] = fh, ft
        out[f"shape_{tag}"][area > 0] = 0.0
        out[f"white_{tag}"][area > 0] = 0.0
        for k in np.flatnonzero(np.nan_to_num(fh) >= ui.shape_gate):
            rs, cs = slice(r0, r1), slice(int(c0[k]), int(c1[k]))
            out[f"shape_{tag}"][k], out[f"white_{tag}"][k] = shape_score(
                masks["hand"][rs, cs], masks["white"][rs, cs])
    r, y1 = ui.ring_px, ty0 + th
    w = x1s - x0s
    ring_n = th * w - np.maximum(th - 2 * r, 0) * np.maximum(w - 2 * r, 0)
    ring = (_rect_sums(integ["border"], ty0, y1, x0s, x1s)
            - _rect_sums(integ["border"], ty0 + r, y1 - r, x0s + r, x1s - r))
    out["border"][:] = np.where((w > 2 * r) & (th > 2 * r), ring / np.maximum(ring_n, 1), 0.0)
    for k, (x0, x1) in enumerate(tiles):
        out["glyph"][k] = _glyph_from_gray(gray[ty0:ty0 + th, x0:x1], ui)
    if template is not None:
        out["ncc_me"] = ncc(out["glyph"], template).astype(np.float32)
    return out


def scan(cid, fps: float = config.F_SCAN, frame_source: Iterable | None = None, *,
         geometry: StripGeometry, template: np.ndarray | None = None, ui: UiSpec = UI,
         url_fn: Callable[[], str] | None = None, media: dict | None = None,
         popen=None) -> pd.DataFrame:
    """Pass 2: per-tile statistics of every frame -> `hand_tiles` (§2.2 step 6).

    `frame_source` yields `(pts_s, frame)` with pts on the master clock; frames may be full
    frames or strip crops. When it is None, the real source is built: `FfmpegStripReader`
    over `url_fn()` (a fresh signed URL per (re)start, never logged) with `media` from
    `probe()` (width, a_start, fmt_start). Frames are processed as they stream and never
    written to disk. A frame with no tiles keeps one placeholder row (tile_idx = -1), so
    `track` still sees it (as NA).
    """
    if not geometry.found:
        raise ValueError(f"class {cid}: strip not located; nothing to scan")
    if frame_source is None:
        if url_fn is None or media is None:
            raise ValueError("scan needs frame_source, or url_fn + media for ffmpeg")
        frame_source = FfmpegStripReader(
            url_fn, width=int(media["width"]), strip_y=geometry.strip_y,
            strip_h=geometry.strip_h, fps=fps, a_start=float(media.get("a_start") or 0.0),
            fmt_start=media.get("fmt_start"), popen=popen or subprocess.Popen)
    cols: dict[str, list] = {c: [] for c in ("pts_s", "tile_idx", "x0", "x1", "glyph_bits",
                                              *_TILE_FLOAT_COLS)}
    blank = pack_glyph(np.zeros((config.GLYPH_H, config.GLYPH_W), bool))
    for pts, frame in frame_source:
        st = frame_stats(as_strip(np.asarray(frame), geometry), geometry, ui=ui,
                         template=template)
        n = len(st["x0"])
        if n == 0:
            cols["pts_s"].append(float(pts))
            cols["tile_idx"].append(-1)
            cols["x0"].append(-1)
            cols["x1"].append(-1)
            cols["glyph_bits"].append(blank)
            for c in _TILE_FLOAT_COLS:
                cols[c].append(np.nan)
            continue
        cols["pts_s"].extend([float(pts)] * n)
        cols["tile_idx"].extend(range(n))
        cols["x0"].extend(st["x0"].tolist())
        cols["x1"].extend(st["x1"].tolist())
        cols["glyph_bits"].extend(pack_glyph(g) for g in st["glyph"])
        for c in _TILE_FLOAT_COLS:
            cols[c].extend(st[c].tolist())
    df = pd.DataFrame({"class_id": np.full(len(cols["pts_s"]), int(cid), dtype=np.int64),
                       "pts_s": np.asarray(cols["pts_s"], dtype=float),
                       "tile_idx": np.asarray(cols["tile_idx"], dtype=np.int64),
                       "x0": np.asarray(cols["x0"], dtype=np.int64),
                       "x1": np.asarray(cols["x1"], dtype=np.int64)})
    for c in _TILE_FLOAT_COLS:
        df[c] = np.asarray(cols[c], dtype=np.float32)
    df["glyph_bits"] = pd.Series(cols["glyph_bits"], dtype=object)
    return schemas.validate(df, "hand_tiles")


# --------------------------------------------------------------------------- pass 1, templates


@dataclass
class AnchorFrame:
    """One anchor frame: the uniquely bordered tile during Brady's isolated speech."""

    t_s: float
    tile_idx: int
    patches: np.ndarray                    # (n_tiles, 12, 48) bool: every tile in the frame

    @property
    def patch(self) -> np.ndarray:
        return self.patches[self.tile_idx]


@dataclass
class Pass1:
    class_id: int
    geometry: StripGeometry
    anchors: list[AnchorFrame] = field(default_factory=list)
    n_anchor_events: int = 0

    def to_json(self) -> dict:
        return {"class_id": int(self.class_id), "geometry": self.geometry.to_dict(),
                "n_anchor_events": int(self.n_anchor_events),
                "anchors": [{"t_s": a.t_s, "tile_idx": a.tile_idx,
                             "patches": [base64.b64encode(pack_glyph(p)).decode()
                                         for p in a.patches]} for a in self.anchors]}

    @classmethod
    def from_json(cls, d: dict) -> Pass1:
        anchors = [AnchorFrame(float(a["t_s"]), int(a["tile_idx"]),
                               unpack_glyphs(base64.b64decode(p) for p in a["patches"]))
                   for a in d.get("anchors", [])]
        return cls(int(d["class_id"]), StripGeometry.from_dict(d["geometry"]), anchors,
                   int(d.get("n_anchor_events", 0)))


def anchor_times(voice: pd.DataFrame, *, me_role: str = config.ROLE_ME,
                 min_dur_s: float = config.ANCHOR_MIN_DUR_S,
                 max_events: int = config.ANCHOR_MAX_EVENTS, edge_s: float = ANCHOR_EDGE_S,
                 pad_s: float = ANCHOR_PAD_S) -> list[list[float]]:
    """Frame times inside Brady's anchor events (§2.2 step 4): main-room, word-bearing voice
    events lasting >= 2 s that overlap no other speaker's main-room event (± pad). Three
    frames per event at start + 1 s, the middle and end - 1 s, so a speaker border lagging
    by up to 1 s still covers them. Up to 12 events, spread evenly over the class."""
    v = voice[voice["room_main"].astype(bool)]
    me = v[(v["role"] == me_role) & v["word_bearing"].astype(bool)]
    me = me[(me["end_s"] - me["start_s"]) >= min_dur_s].sort_values("start_s")
    other = v[v["role"] != me_role]
    oa, ob = other["start_s"].to_numpy(float), other["end_s"].to_numpy(float)
    keep = []
    for r in me.itertuples():
        if not ((oa < r.end_s + pad_s) & (ob > r.start_s - pad_s)).any():
            keep.append((float(r.start_s), float(r.end_s)))
    if len(keep) > max_events:
        idx = np.unique(np.linspace(0, len(keep) - 1, max_events).round().astype(int))
        keep = [keep[i] for i in idx]
    out = []
    for a, b in keep:
        times = sorted({round(a + edge_s, 3), round((a + b) / 2, 3), round(b - edge_s, 3)})
        out.append([t for t in times if a < t < b] or [round((a + b) / 2, 3)])
    return out


def _anchor_from_strip(t: float, strip: np.ndarray, geometry: StripGeometry, ui: UiSpec,
                       det: Detector) -> AnchorFrame | None:
    st = frame_stats(strip, geometry, ui=ui)
    bordered = np.flatnonzero(np.nan_to_num(st["border"]) >= det.border_min)
    if len(bordered) != 1:
        return None                               # zero or several bordered tiles: skip
    return AnchorFrame(float(t), int(bordered[0]), st["glyph"])


def pass1(cid, frame_reader: Callable[[Sequence[float]], Iterable[tuple[float, np.ndarray]]],
          *, T_end_s: float, voice: pd.DataFrame | None = None,
          n_present: int | Sequence[int] | None = None, expected_h: float | None = None,
          ui: UiSpec = UI, det: Detector = DETECTOR) -> Pass1:
    """Pass 1 (§2.2 steps 1-4): strip detection on 6 frames spread over the class, then up to
    3 anchor frames in each of up to 12 anchor events. `frame_reader(times)` returns
    `(t, full frame)` pairs (injectable; the real one is `FfmpegClipReader.frames_at`)."""
    times = np.linspace(0.0, float(T_end_s), config.PASS1_N_FRAMES + 2)[1:-1]
    got = list(frame_reader([float(t) for t in times]))
    geom = detect_strip([f for _, f in got], expected_h=expected_h, n_present=n_present, ui=ui)
    res = Pass1(int(cid), geom)
    if not geom.found or voice is None or len(voice) == 0:
        return res
    for ev in anchor_times(voice):
        hit = False
        for t, frame in frame_reader(ev):
            a = _anchor_from_strip(t, as_strip(np.asarray(frame), geom), geom, ui, det)
            if a is not None:
                res.anchors.append(a)
                hit = True
        res.n_anchor_events += int(hit)
    return res


def build_era_templates(anchors: dict, eras: dict, *, loco: bool = True
                        ) -> dict[int, np.ndarray | None]:
    """Brady's glyph template per class (§2.2 step 5): the pixelwise median of anchor patches
    from the *other* classes of the same UI era (`loco=True`), or of all classes of the era
    (`loco=False`, diagnostics only). `anchors` maps class_id -> list of 12 x 48 patches (or
    a Pass1). Returns class_id -> float32 template, or None when no other class has anchors."""
    pools = {}
    for cid, a in anchors.items():
        pats = [x.patch for x in a.anchors] if isinstance(a, Pass1) else list(a)
        pools[cid] = [np.asarray(p, dtype=np.float32) for p in pats]
    out: dict = {}
    for cid, era in eras.items():
        pool = [p for other, ps in pools.items()
                if eras.get(other) == era and (other != cid or not loco) for p in ps]
        out[cid] = np.median(np.stack(pool), axis=0).astype(np.float32) if pool else None
    return out


def validate_tracking(anchors: Sequence[AnchorFrame], template: np.ndarray | None,
                      det: Detector = DETECTOR, ncc_floor: float | None = None
                      ) -> tuple[float, int]:
    """(share of anchor frames where the tracked tile is the bordered tile, n frames)."""
    if template is None or not anchors:
        return float("nan"), len(anchors)
    hits = [pick_tile(ncc(a.patches, template), det, ncc_floor)[0] == a.tile_idx
            for a in anchors]
    return float(np.mean(hits)), len(hits)


def ncc_floors(anchors: dict, templates: dict, eras: dict, *, det: Detector = DETECTOR
               ) -> dict[int, float | None]:
    """Per-class impostor floor for tracking (L19-safe: leave-one-class-out).

    A classmate whose name label shares Brady's first glyphs ("Brady S.", "Brad M.") can pass
    NCC_min and win by a wide gap whenever Brady's own tile is off the strip (pagination, on
    stage). Brady's real label scores higher against his template than such a lookalike, so
    a pick must also reach the floor: the `ncc_floor_q` percentile, minus a margin, of the
    NCC of the bordered tile in the anchor frames of the OTHER classes of the same era, each
    scored against its own LOCO template. It never depends on the class's own speech.
    None when the era has fewer than MIN_ANCHORS such frames (NCC_min alone applies)."""
    per_class: dict = {}
    for cid, a in anchors.items():
        tmpl = templates.get(cid)
        frames = a.anchors if isinstance(a, Pass1) else list(a)
        if tmpl is None or not frames:
            continue
        per_class[cid] = [float(ncc(f.patch, tmpl)[0]) for f in frames]
    out: dict = {}
    for cid, era in eras.items():
        pool = [x for other, xs in per_class.items() if other != cid and eras.get(other) == era
                for x in xs]
        out[cid] = (float(np.percentile(pool, det.ncc_floor_q)) - det.ncc_floor_margin
                    if len(pool) >= config.MIN_ANCHORS else None)
    return out


def tile_status(geometry: StripGeometry, coverage_share: float, n_anchor_events: int,
                valid_share: float) -> tuple[str, bool]:
    """(tile_status, template_validated) per §2.2 steps 5 and 8."""
    validated = (n_anchor_events >= config.MIN_ANCHORS
                 and not np.isnan(valid_share) and valid_share >= config.TRACK_VALID_MIN)
    if not geometry.found or not coverage_share >= config.COVERAGE_MIN:
        return "unlocated", validated
    if (n_anchor_events >= config.MIN_ANCHORS
            and not (valid_share >= config.TRACK_VALID_MIN)):
        return "tracking_failed", False
    return "located", validated


# --------------------------------------------------------------------------- tracking, states


def baseline(frac: np.ndarray, up_mask: np.ndarray) -> tuple[float, bool]:
    """Per-class DOWN baseline (§2.3): median frac_hand over tracked samples that are not UP
    on the first pass; flagged when it exceeds θ_off / 2. Whole class: label side only."""
    frac = np.asarray(frac, dtype=float)
    vals = frac[~np.asarray(up_mask, bool) & ~np.isnan(frac)]
    b = float(np.median(vals)) if len(vals) else 0.0
    return b, b > config.DOWN_BASELINE_FLAG


def running_baseline(frac: np.ndarray, up_mask: np.ndarray) -> np.ndarray:
    """Causal DOWN baseline per sample: the median of frac_hand over tracked, not-UP
    (first-pass) samples up to and including that sample (0 before any). Its value at the
    last sample equals `baseline`; it never reads a later frame, so the live state built on
    it uses samples <= t only (§2.4)."""
    frac = np.asarray(frac, dtype=float)
    ok = ~np.asarray(up_mask, bool) & ~np.isnan(frac)
    out = np.zeros(len(frac))
    vals: list[float] = []
    for i in range(len(frac)):
        if ok[i]:
            bisect.insort(vals, float(frac[i]))
        m = len(vals)
        if m:
            out[i] = 0.5 * (vals[(m - 1) // 2] + vals[m // 2])
    return out


def raw_states(frac_adj: np.ndarray, shape_pass: np.ndarray, found: np.ndarray,
               frac_tick: np.ndarray | None = None, det: Detector = DETECTOR) -> np.ndarray:
    """Hysteresis raw state codes (§2.4): UP when frac' >= θ_on and the shape test passes;
    DOWN when frac' < θ_off; NA when untracked (or a visible tick under the tick rule);
    otherwise the previous state (NA before any state)."""
    frac_adj = np.asarray(frac_adj, dtype=float)
    code = np.full(len(frac_adj), np.nan)
    code[frac_adj < det.theta_off] = _DOWN_C
    code[(frac_adj >= det.theta_on) & np.asarray(shape_pass, bool)] = _UP_C
    if det.tick_na and frac_tick is not None:
        code[np.nan_to_num(np.asarray(frac_tick, float)) >= det.theta_on] = _NA_C
    code[~np.asarray(found, bool)] = _NA_C
    return pd.Series(code).ffill().fillna(_NA_C).to_numpy(dtype=np.int64)


def live_states(raw: np.ndarray, k: int = config.K_CONFIRM, hold: int = causal.NA_HOLD
                ) -> np.ndarray:
    """Causal confirmed state codes (§2.4; `causal.live_codes`): a change becomes live at
    the k-th consecutive agreeing sample, NA runs of <= `hold` samples keep the last
    confirmed state; uses samples <= t only. NA until the first confirmation."""
    return causal.live_codes(raw, k, hold)


def track(hand_tiles: pd.DataFrame, template: np.ndarray | None = None, *,
          det: Detector = DETECTOR, ncc_floor: float | None = None) -> pd.DataFrame:
    """Offline tracking and per-frame state -> `hand_frames` (§2.2 step 7, §2.3, §2.4).

    Brady's tile per frame is the argmax NCC against `template` (recomputed from the stored
    glyph bits; `ncc_me` is used when no template is given) if NCC >= max(NCC_min,
    `ncc_floor`) and it beats the runner-up by NCC_gap, else NA (`ncc_floors`: the LOCO
    impostor floor). Per frame and tile everything is causal; the DOWN baseline is not:

    * `raw_state` subtracts the whole-class two-pass baseline (label side: episodes,
      observability, the flagged-class audit);
    * `raw_state_live` subtracts the running baseline of samples <= t, and `live_state` is
      its confirmed state (`live_states`), so no frame after t changes the live state at t.

    `n_others` counts tiles other than Brady's that pass frac' >= θ_on and the shape test;
    when Brady is NA, the best-NCC tile (his, if he is on the strip at all) is left out too
    (§2.5: his own tile is never in Δn). Never a feature. `attrs` carries down_baseline,
    flagged, others_baseline, ncc_floor and the lookalike diagnostics: `n_lookalike_frames`
    (a second tile reaches NCC_min) and `n_floor_rejected` (a lone pick above NCC_min but
    under the floor), so sections with name lookalikes can be audited.
    """
    ht = hand_tiles.sort_values(["pts_s", "tile_idx"], kind="stable").reset_index(drop=True)
    cid = int(ht["class_id"].iloc[0]) if len(ht) else -1
    times = np.unique(ht["pts_s"].to_numpy(float))
    n = len(times)
    real = ht[ht["tile_idx"] >= 0].copy()
    if template is not None:
        real["ncc"] = ncc(unpack_glyphs(real["glyph_bits"]), template)
    else:
        real["ncc"] = real["ncc_me"].astype(float)
    real["fi"] = np.searchsorted(times, real["pts_s"].to_numpy(float))
    roi = det.roi
    real["frac"] = real[f"frac_hand_{roi}"].astype(float)
    white = (real[f"white_{roi}"] if f"white_{roi}" in real else
             pd.Series(np.nan, index=real.index))
    real["shape_ok"] = shape_ok(real[f"shape_{roi}"].astype(float), white.astype(float), det)
    if "overlay" in real:              # the real UI's full-tile raise animation (oncue.strip)
        real["shape_ok"] |= real["overlay"].fillna(False).astype(bool)

    srt = real.sort_values(["fi", "ncc"], ascending=[True, False], kind="stable")
    first = srt.groupby("fi", sort=True).head(1).set_index("fi")
    second = srt.groupby("fi", sort=True).nth(1).set_index("fi")["ncc"]
    n_tiles = np.bincount(real["fi"], minlength=n)

    best_ncc = first["ncc"].reindex(range(n)).to_numpy(float)
    runner = second.reindex(range(n)).fillna(-1.0).to_numpy(float)
    gap = best_ncc - runner
    floor = det.ncc_min if ncc_floor is None or not np.isfinite(ncc_floor) else \
        max(det.ncc_min, float(ncc_floor))
    found = (best_ncc >= floor) & (gap >= det.ncc_gap)
    rejected = (best_ncc >= det.ncc_min) & (gap >= det.ncc_gap) & ~found
    best_tile = first["tile_idx"].reindex(range(n)).fillna(-1).to_numpy()
    tile_idx = np.where(found, best_tile, -1)
    frac = np.where(found, first["frac"].reindex(range(n)).to_numpy(float), np.nan)
    shp = found & first["shape_ok"].reindex(range(n)).fillna(False).to_numpy(bool)
    tick = np.where(found, first[f"frac_tick_{roi}"].reindex(range(n)).to_numpy(float), np.nan)

    up0 = (np.nan_to_num(frac) >= det.theta_on) & shp
    b, flagged = baseline(np.where(found, frac, np.nan), up0)
    raw = raw_states(frac - b, shp, found, tick, det)                  # label side
    b_run = running_baseline(np.where(found, frac, np.nan), up0)
    raw_live = raw_states(frac - b_run, shp, found, tick, det)        # causal
    live = live_states(raw_live, det.k, det.na_hold)
    frac_adj = frac - b

    # Brady's tile never counts as an "other": when he is NA, drop the best-NCC tile too
    fi = real["fi"].to_numpy()
    chosen = real["tile_idx"].to_numpy() == best_tile[fi]
    oth = real[~chosen]
    ob = float(np.nanmedian(oth["frac"])) if len(oth) and oth["frac"].notna().any() else 0.0
    passing = (oth["frac"] - ob >= det.theta_on) & oth["shape_ok"]
    n_others = np.bincount(oth.loc[passing, "fi"], minlength=n)
    border_any = np.zeros(n, bool)
    hit = real.loc[np.nan_to_num(real["border"].astype(float)) >= det.border_min, "fi"]
    border_any[hit.to_numpy()] = True

    hf = pd.DataFrame({
        "class_id": np.full(n, cid, dtype=np.int64), "t_s": times,
        "n_tiles": n_tiles.astype(np.int64), "tile_found": found.astype(bool),
        "tile_idx": pd.array(np.where(found, tile_idx, 0), dtype="Int64"),
        "ncc": best_ncc.astype(np.float32), "ncc_gap": gap.astype(np.float32),
        "frac_hand_adj": frac_adj.astype(np.float32), "shape_ok": shp.astype(bool),
        "frac_tick": tick.astype(np.float32), "n_others": n_others.astype(np.int64),
        "border_any": border_any, "raw_state": _STATE_NAMES[raw],
        "live_state": _STATE_NAMES[live], "raw_state_live": _STATE_NAMES[raw_live]})
    hf.loc[~found, "tile_idx"] = pd.NA
    for c in ("raw_state", "live_state", "raw_state_live"):
        hf[c] = hf[c].astype("string")
    hf.attrs.update(down_baseline=b, flagged=bool(flagged), others_baseline=ob, roi=roi,
                    ncc_floor=floor, n_floor_rejected=int(rejected.sum()),
                    n_lookalike_frames=int((runner >= det.ncc_min).sum() + rejected.sum()))
    return schemas.validate(hf, "hand_frames")


def coverage(hand_frames: pd.DataFrame, exclude: Iterable[tuple[float, float]] = ()) -> float:
    """Share of samples outside `exclude` spans (breakouts, gaps) with tile_found (§2.2.8)."""
    t = hand_frames["t_s"].to_numpy(float)
    keep = np.ones(len(t), bool)
    for a, b in exclude:
        keep &= ~((t >= a) & (t < b))
    if not keep.any():
        return float("nan")
    return float(hand_frames["tile_found"].to_numpy(bool)[keep].mean())


def live_transitions(hand_frames: pd.DataFrame) -> pd.DataFrame:
    """Every change of the causal live state: (t_s, from_state, to_state). Labels can use
    the UP -> DOWN rows for `post_lowering` inside toggle-merged episodes."""
    live = hand_frames["live_state"].astype(str).to_numpy()
    t = hand_frames["t_s"].to_numpy(float)
    idx = np.flatnonzero(live[1:] != live[:-1]) + 1
    return pd.DataFrame({"t_s": t[idx], "from_state": live[idx - 1], "to_state": live[idx]})


# --------------------------------------------------------------------------- episodes


def bridge_na(codes: np.ndarray, dt: float, max_gap_s: float = config.NA_BRIDGE_S
              ) -> np.ndarray:
    """Label-side NA bridging: NA gaps <= max_gap_s with the same state on both sides are
    filled with that state; longer gaps (or different sides) stay NA."""
    out = np.asarray(codes, dtype=np.int64).copy()
    n = len(out)
    for v, a, b in _rle(out):
        if v != _NA_C or a == 0 or b == n:
            continue
        if (b - a) * dt <= max_gap_s + 1e-9 and out[a - 1] == out[b] and out[b] != _NA_C:
            out[a:b] = out[b]
    return out


def confirmed_codes(codes: np.ndarray, k: int = config.K_CONFIRM) -> np.ndarray:
    """Label-side confirmed state per sample: a DOWN or UP run shorter than k samples never
    happened (it takes the previous confirmed state, NA at the start or after NA); NA runs
    are kept. Episodes are built on this sequence, so a blip neither starts, extends nor
    ends one, and a one-sample DOWN inside a raise is not a toggle (§2.4)."""
    c = np.asarray(codes, dtype=np.int64)
    out = c.copy()
    prev = _NA_C
    for v, a, b in _rle(c):
        if v != _NA_C and b - a < k:
            out[a:b] = prev
        else:
            prev = v
    return out


def _down_gap(conf: np.ndarray, i: int, step: int) -> int | None:
    """Confirmed DOWN samples from i (inclusive) in direction `step` up to the first confirmed
    UP sample; None if NA or the recording edge comes first."""
    n, m = len(conf), 0
    while 0 <= i < n:
        if conf[i] == _UP_C:
            return m
        if conf[i] != _DOWN_C:
            return None
        m += 1
        i += step
    return None


def _censored(t: np.ndarray, conf: np.ndarray, first: int, d_pre_s: float, dt: float) -> bool:
    """Left-censored: the 2 s before the first UP sample are not all confirmed DOWN (NA or the
    recording start)."""
    n_pre = round(d_pre_s / dt)
    t_up = float(t[first])
    prev = np.flatnonzero((t >= t_up - d_pre_s - 1e-6) & (np.arange(len(t)) < first))
    return not (first >= n_pre and len(prev) >= n_pre and t[0] <= t_up - d_pre_s + 1e-6
                and (conf[prev] == _DOWN_C).all())


def _segment_episodes(t: np.ndarray, codes: np.ndarray, shape_pass: np.ndarray,
                      det: Detector, live: np.ndarray | None = None) -> list[dict]:
    """Retrospective episodes (§2.4) from raw state codes: NA bridging, then confirmed runs
    (`confirmed_codes`), merges of DOWN gaps < 3 s, D_pre / left-censoring, onset = midpoint
    between the last confirmed DOWN and the first UP sample of the first confirmed UP run.
    `offset_live_s` = the live exit from UP after the episode's last UP sample (`live`:
    the causal live codes; into DOWN or into a long NA run), NaN if it never left UP.
    Isolated blips (`Detector.blips_as_glitches`) become glitch episodes."""
    n = len(t)
    if n == 0:
        return []
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0 / det.fps
    bridged = bridge_na(codes, dt, det.na_bridge_s)
    conf = confirmed_codes(bridged, det.k)
    live = live_states(codes, det.k, det.na_hold) if live is None else np.asarray(live)
    shape = np.asarray(shape_pass, bool)
    runs = _rle(conf)
    out, i = [], 0
    while i < len(runs):
        v, a, b = runs[i]
        if v != _UP_C:
            i += 1
            continue
        ups, toggles, j = [(a, b)], 0, i + 1
        while (j + 1 < len(runs) and runs[j][0] == _DOWN_C and runs[j + 1][0] == _UP_C
               and (runs[j][2] - runs[j][1]) * dt < det.merge_gap_s - 1e-9):
            ups.append((runs[j + 1][1], runs[j + 1][2]))
            toggles += 1
            j += 2
        i = j
        first, last = ups[0][0], ups[-1][1]            # [first, last) spans the episode
        onset = (float(t[first - 1] + t[first]) / 2 if first > 0 and conf[first - 1] == _DOWN_C
                 else float(t[first]))
        censored = _censored(t, conf, first, det.d_pre_s, dt)
        if censored and det.pre_na_onset_s > 0:
            g = first
            while g > 0 and conf[g - 1] == _NA_C:
                g -= 1
            if (g < first and (first - g) * dt <= det.pre_na_onset_s + 1e-9 and g > 0
                    and conf[g - 1] == _DOWN_C and not _censored(t, conf, g, det.d_pre_s, dt)):
                censored = False
                onset = float(t[g - 1] + t[g]) / 2
        ended = "down" if last < n and conf[last] == _DOWN_C else ("na" if last < n else "end")
        offset = float(t[last]) if ended == "down" else float(t[last - 1] + dt)
        after = np.flatnonzero(live[last:] != _UP_C) + last if last < n else np.zeros(0, int)
        offset_live = float(t[after[0]]) if len(after) else float("nan")
        up_idx = np.concatenate([np.arange(a2, b2) for a2, b2 in ups])
        up_idx = up_idx[bridged[up_idx] == _UP_C]
        dur = offset - onset
        out.append({"onset_s": onset, "t_up_s": float(t[first]), "offset_s": offset,
                    "offset_live_s": offset_live, "dur_s": dur, "n_toggles": toggles,
                    "left_censored": censored,
                    "glitch": bool(dur < det.min_episode_s), "ended": ended,
                    "n_shape": int(shape[up_idx].sum()), "i_first": first, "i_last": last})
    if det.blips_as_glitches:
        gap_n = det.merge_gap_s / dt - 1e-9
        for v, a, b in _rle(bridged):
            if v != _UP_C or b - a >= det.k or b >= n or bridged[b] != _DOWN_C:
                continue
            before, after_n = _down_gap(conf, a - 1, -1), _down_gap(conf, b, +1)
            if (before is not None and before < gap_n) or (after_n is not None
                                                           and after_n < gap_n):
                continue                                # noise next to a real raise
            onset = (float(t[a - 1] + t[a]) / 2 if a > 0 and conf[a - 1] == _DOWN_C
                     else float(t[a]))
            out.append({"onset_s": onset, "t_up_s": float(t[a]), "offset_s": float(t[b]),
                        "offset_live_s": float(t[b]), "dur_s": float(t[b]) - onset,
                        "n_toggles": 0,
                        "left_censored": _censored(t, conf, a, det.d_pre_s, dt),
                        "glitch": True, "ended": "down",
                        "n_shape": int(shape[a:b].sum()), "i_first": a, "i_last": b})
    return sorted(out, key=lambda e: e["onset_s"])


def me_turn_starts(voice: pd.DataFrame | None, *, me_role: str = config.ROLE_ME) -> np.ndarray:
    """Starts of Brady's word-bearing main-room turns (events merged across gaps < 2 s)."""
    if voice is None or len(voice) == 0:
        return np.zeros(0)
    v = voice[(voice["role"] == me_role) & voice["room_main"].astype(bool)
              & voice["word_bearing"].astype(bool)].sort_values("start_s")
    starts, last_end = [], -np.inf
    for a, b in zip(v["start_s"].to_numpy(float), v["end_s"].to_numpy(float), strict=True):
        if a - last_end >= config.TURN_MERGE_GAP_S:
            starts.append(a)
        last_end = max(last_end, b)
    return np.asarray(starts, dtype=float)


def _prof_text(words: pd.DataFrame | None, a: float, b: float) -> str:
    if words is None or len(words) == 0:
        return ""
    w = words[(words["role"] == config.ROLE_PROF) & (words["start_s"] >= a)
              & (words["start_s"] <= b)]
    col = "text_norm" if "text_norm" in w else "text_masked"
    return " ".join(str(x) for x in w[col].fillna("") if str(x))


def _seg_type_at(segments: pd.DataFrame | None, t: float) -> str | None:
    if segments is None or len(segments) == 0:
        return None
    s = segments.sort_values("start_s")
    i = int(np.searchsorted(s["start_s"].to_numpy(float), t, side="right")) - 1
    return str(s["seg_type"].iloc[i]) if i >= 0 else None


def delta_n_others(t: np.ndarray, n_others: np.ndarray, onset: float) -> int:
    """Δn_others(o) = max over [o - 5, o + 10] of n_others minus n_others(o - 5) (§2.5)."""
    lo, hi = config.VOTE_DN_WINDOW_S
    i0 = max(0, int(np.searchsorted(t, onset + lo, side="right")) - 1)
    win = (t >= onset + lo - 1e-9) & (t <= onset + hi + 1e-9)
    if not win.any():
        return 0
    return int(n_others[win].max() - n_others[i0])


def q_open_window(words: pd.DataFrame | None, *,
                  lookback_s: float = config.CONTESTED_LOOKBACK_S) -> Callable[[float], bool]:
    """`q_open(o)` for contested typing (§2.5): True when the question detector
    (`text.question_detector`, headline rules) evaluated a professor question whose
    utterance closed in [o - lookback_s, o]. Label side only."""
    from . import text

    if words is None or not len(words) or "role" not in words:
        return lambda o: False
    q = text.question_detector(words)
    ends = np.sort(q.loc[q["is_q"].astype(bool), "end_s"].to_numpy(float)) if len(q) \
        else np.zeros(0)

    def q_open(o: float) -> bool:
        lo = np.searchsorted(ends, o - lookback_s, side="left")
        hi = np.searchsorted(ends, o, side="right")
        return bool(hi > lo)
    return q_open


def episodes(hand_frames: pd.DataFrame, *, words: pd.DataFrame | None = None,
             voice: pd.DataFrame | None = None, segments: pd.DataFrame | None = None,
             q_open: Callable[[float], bool] | None = None, det: Detector = DETECTOR,
             vote_frac: float | None = None) -> pd.DataFrame:
    """Episode state machine, vote typing and outcomes -> `hand_episodes` (§2.4, §2.5).

    Label side only: merged episodes, their offsets and their types look at samples after
    an onset, so no causal rule or feature reads this table (they use `causal.hand_runs`).
    `words` (table `words`) feeds the VOTE / INVITE lexicons over the professor's words in
    [o - 60, o] (label-side typing); `q_open(o)` (see `q_open_window`) adds detected open
    questions in [o - 60, o] to the contested rule; `voice` (table `voice`) gives Brady's
    word-bearing main-room turns for outcomes; `segments` gives `seg_type` at the onset.
    Left-censored episodes keep `onset_s` = their first observed UP time and must not be
    used as onsets.
    """
    hf = hand_frames.sort_values("t_s").reset_index(drop=True)
    cid = int(hf["class_id"].iloc[0]) if len(hf) else -1
    t = hf["t_s"].to_numpy(float)
    codes = (pd.Series(hf["raw_state"].astype(str))
             .map({DOWN: _DOWN_C, UP: _UP_C, NA: _NA_C}).fillna(_NA_C).to_numpy(np.int64))
    eps = _segment_episodes(t, codes, hf["shape_ok"].to_numpy(bool), det)
    n_oth = hf["n_others"].to_numpy(np.int64)
    n_tiles = hf["n_tiles"].to_numpy(np.int64)
    turns = me_turn_starts(voice)
    rows = []
    for k, e in enumerate(eps):
        o = e["onset_s"]
        dn = delta_n_others(t, n_oth, o)
        thr = config.vote_threshold(int(n_tiles[min(e["i_first"], len(t) - 1)]),
                                    det.vote_frac if vote_frac is None else vote_frac,
                                    det.vote_min_dn)
        prof = _prof_text(words, o - config.VOTE_LEX_LOOKBACK_S, o)
        vote_lex = bool(config.VOTE_RE.search(prof))
        dn_vote = dn >= thr
        invite = bool(config.INVITE_RE.search(prof)) or bool(q_open and q_open(o))
        ep_type = (config.EP_GLITCH if e["glitch"] else
                   config.EP_VOTE if (dn_vote or vote_lex) else config.EP_BID)
        outcome, latency = None, np.nan
        if ep_type == config.EP_BID:
            later = turns[(turns >= o) & (turns <= e["offset_s"] + config.CALLED_AFTER_OFFSET_S)]
            if len(later):
                outcome, latency = "called", float(later[0] - o)
            elif e["ended"] in ("na", "end"):
                outcome = "open"
            else:
                outcome = "withdrawn"
        rows.append({
            "class_id": cid, "ep_id": k, "onset_s": o, "onset_retimed_s": np.nan,
            "retime_failed": pd.NA, "offset_s": e["offset_s"],
            "offset_live_s": e["offset_live_s"], "dur_s": e["dur_s"],
            "n_toggles": e["n_toggles"], "ep_type": ep_type, "delta_n_others": dn,
            "vote_lex": vote_lex, "contested": bool(dn_vote and not vote_lex and invite
                                                   and ep_type == config.EP_VOTE),
            "left_censored": e["left_censored"],
            "me_hand_eligible": bool(ep_type != config.EP_GLITCH
                                     and e["n_shape"] >= det.me_hand_min_shape
                                     and e["dur_s"] >= det.me_hand_min_s),
            "outcome": outcome, "call_latency_s": latency, "seg_type": _seg_type_at(segments, o)})
    if not rows:
        return schemas.empty("hand_episodes")
    df = pd.DataFrame(rows)
    df["retime_failed"] = df["retime_failed"].astype("boolean")
    df["delta_n_others"] = df["delta_n_others"].astype("Int64")
    for c in ("ep_type", "outcome", "seg_type"):
        df[c] = df[c].astype("string")
    return schemas.validate(schemas.coerce(df, "hand_episodes"), "hand_episodes")


# --------------------------------------------------------------------------- re-timing


def locate_me(strip: np.ndarray, geometry: StripGeometry, template: np.ndarray, *,
              ui: UiSpec = UI, det: Detector = DETECTOR) -> tuple[int, int] | None:
    """Brady's tile x-span in one strip frame by glyph NCC, or None (NA)."""
    tiles = strip_tiles(strip, ui=ui)
    if len(tiles) == 0:
        return None
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    ty0, th = geometry.tile_y0, geometry.tile_h
    pats = np.stack([_glyph_from_gray(gray[ty0:ty0 + th, a:b], ui) for a, b in tiles])
    k, _, _ = pick_tile(ncc(pats, template), det)
    return None if k < 0 else (int(tiles[k, 0]), int(tiles[k, 1]))


def _positive(strip: np.ndarray, span: tuple[int, int], geometry: StripGeometry,
              base: float, ui: UiSpec, det: Detector) -> bool:
    s = ui.roi_s1 if det.roi == "s1" else ui.roi_s2
    rs, cs = _roi_slices(geometry.tile_y0, geometry.tile_h, span[0], span[1], s, ui)
    m = hsv_masks(np.ascontiguousarray(strip[rs, cs]), ui)
    if m["hand"].size == 0 or m["hand"].mean() - base < det.theta_on:
        return False
    return shape_test(m["hand"], m["white"], det)


def retime_onset(frames: Sequence[tuple[float, np.ndarray]], t0: float, *,
                 geometry: StripGeometry, template: np.ndarray | None = None,
                 span: tuple[int, int] | None = None, base: float = 0.0,
                 ui: UiSpec = UI, det: Detector = DETECTOR,
                 seek_tol_s: float = 0.05) -> float | None:
    """Re-timed onset from a full-frame-rate read starting at t0: the first frame with
    frac' >= θ_on and a shape pass followed by >= 3 positive frames. None when the seek
    missed (first pts more than seek_tol_s from t0) or no such frame exists."""
    if not frames or abs(float(frames[0][0]) - t0) > seek_tol_s:
        return None
    pos = []
    for _, frame in frames:
        strip = as_strip(np.asarray(frame), geometry)
        where = locate_me(strip, geometry, template, ui=ui, det=det) if template is not None \
            else span
        pos.append(where is not None and _positive(strip, where, geometry, base, ui, det))
    need = config.RETIME_MIN_POS_FRAMES
    for i in range(len(pos) - need):
        if pos[i] and all(pos[i + 1:i + 1 + need]):
            return float(frames[i][0])
    return None


def retime(hand_episodes: pd.DataFrame, reader: Callable[[float, float], Sequence],
           *, geometry: StripGeometry, template: np.ndarray | None = None,
           spans: dict | None = None, base: float = 0.0, ui: UiSpec = UI,
           det: Detector = DETECTOR) -> pd.DataFrame:
    """Sub-second onset re-timing (§2.4, should-tier). For each episode with an onset,
    `reader(t0, dur)` returns full-frame-rate `(t, frame)` pairs over [t_up - 1.0,
    t_up - 1.0 + RETIME_SPAN_S] (= t_up + 0.5: the onset lies in (t_up - 0.5, t_up] and
    needs 3 more positive frames after it; the real reader is `FfmpegClipReader`, -copyts
    + showinfo pts). On a failed seek or no qualifying frame: onset_retimed_s = the coarse
    midpoint, retime_failed = True. `spans` (ep_id -> tile x-span) is used when no template
    is given."""
    out = hand_episodes.copy()
    for i, ep in out.iterrows():
        if bool(ep["left_censored"]):
            continue
        t_up = float(ep["onset_s"]) + 0.5 / det.fps
        t0 = t_up - config.RETIME_PRE_S
        try:
            frames = list(reader(t0, config.RETIME_SPAN_S))
        except Exception as exc:                       # noqa: BLE001 - any seek failure
            log.warning("retime: read failed for episode %s (%s)", ep["ep_id"],
                        type(exc).__name__)
            frames = []
        span = (spans or {}).get(int(ep["ep_id"]))
        res = retime_onset(frames, t0, geometry=geometry, template=template, span=span,
                           base=base, ui=ui, det=det)
        out.at[i, "onset_retimed_s"] = float(ep["onset_s"]) if res is None else res
        out.at[i, "retime_failed"] = res is None
    out["retime_failed"] = out["retime_failed"].astype("boolean")
    return out


# --------------------------------------------------------------------------- A/V lag


def _activity(starts: np.ndarray, ends: np.ndarray, grid0: float, n: int, res: float
              ) -> np.ndarray:
    acc = np.zeros(n + 1)
    a = np.clip(np.floor((np.asarray(starts) - grid0) / res).astype(int), 0, n)
    b = np.clip(np.ceil((np.asarray(ends) - grid0) / res).astype(int), 0, n)
    np.add.at(acc, a, 1)
    np.add.at(acc, b, -1)
    return (np.cumsum(acc)[:n] > 0).astype(np.float64)


def av_lag(hand_frames: pd.DataFrame, words: pd.DataFrame, *, border: np.ndarray | None = None,
           role: str | None = None, search_s: float = config.AV_LAG_SEARCH_S,
           res_s: float = config.AV_LAG_RES_S, min_speech_s: float = 0.0) -> float:
    """Per-class A/V lag (§2.1.3): cross-correlate the speaker border (`border_any`, or the
    given per-sample `border`) with ASR speech activity over ±search_s at res_s resolution.
    Positive = the border lags the words. `role` restricts the words (e.g. "me" for the
    Brady-only check, which needs min_speech_s of speech). NaN when not measurable."""
    t = hand_frames["t_s"].to_numpy(float)
    bord = (hand_frames["border_any"].to_numpy(bool) if border is None
            else np.asarray(border, bool)).astype(np.float64)
    w = words if role is None else words[words["role"] == role]
    if len(t) < 4 or len(w) == 0:
        return float("nan")
    if float((w["end_s"] - w["start_s"]).sum()) < min_speech_s:
        return float("nan")
    grid = np.arange(t[0], t[-1], res_s)
    n = len(grid)
    nearest = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest -= (grid - t[nearest - 1]) < (t[nearest] - grid)       # centred hold
    x = bord[nearest]
    y = _activity(w["start_s"].to_numpy(float), w["end_s"].to_numpy(float), grid[0], n, res_s)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    lags = np.arange(-round(search_s / res_s), round(search_s / res_s) + 1)
    corr = np.empty(len(lags))
    for j, L in enumerate(lags):                       # x(t) vs y(t - L)
        a, b = (x[L:], y[:n - L]) if L >= 0 else (x[:n + L], y[-L:])
        sa, sb = a.std(), b.std()
        corr[j] = ((a - a.mean()) * (b - b.mean())).mean() / (sa * sb) if sa > 0 and sb > 0 \
            else -np.inf
    j = int(np.argmax(corr))
    shift = 0.0
    if 0 < j < len(lags) - 1 and np.isfinite(corr[j - 1]) and np.isfinite(corr[j + 1]):
        den = corr[j - 1] - 2 * corr[j] + corr[j + 1]
        shift = 0.5 * (corr[j - 1] - corr[j + 1]) / den if den != 0 else 0.0
    return float((lags[j] + shift) * res_s)


def me_border(hand_frames: pd.DataFrame, hand_tiles: pd.DataFrame,
              det: Detector = DETECTOR) -> pd.Series:
    """Per hand_frames row (same index): Brady's tracked tile is bordered (False when NA).
    This is the `me_border` input of `controls.p_clock_inputs` (Brady version)."""
    key = hand_tiles[["pts_s", "tile_idx", "border"]].astype({"tile_idx": "Int64"})
    j = hand_frames[["t_s", "tile_idx"]].astype({"tile_idx": "Int64"}).merge(
        key, left_on=["t_s", "tile_idx"], right_on=["pts_s", "tile_idx"], how="left")
    found = hand_frames["tile_found"].fillna(False).to_numpy(bool)
    return pd.Series(found & (np.nan_to_num(j["border"].to_numpy(float)) >= det.border_min),
                     index=hand_frames.index, name="me_border")


def av_lag_brady(hand_frames: pd.DataFrame, hand_tiles: pd.DataFrame, words: pd.DataFrame,
                 det: Detector = DETECTOR) -> float:
    """Brady-only lag (his bordered tile vs his ASR words), when he has >= 20 s of speech."""
    return av_lag(hand_frames, words, border=me_border(hand_frames, hand_tiles, det),
                  role=config.ROLE_ME, min_speech_s=config.AV_BRADY_MIN_SPEECH_S)


# --------------------------------------------------------------------------- ffmpeg plumbing

_SHOWINFO_RE = re.compile(r"Parsed_showinfo.*?\bpts_time:\s*(-?\d+(?:\.\d*)?(?:e-?\d+)?)")
_URL_RE = re.compile(r"((?:https?|s3)://[^\s'\"<>?#]+)\?[^\s'\"<>#]*")


def redact(text: str, *secrets: str | None) -> str:
    """Remove every given secret (signed URLs) and any URL query string from `text`."""
    out = str(text)
    for s in secrets:
        if s:
            out = out.replace(s, "<signed-url>")
    return _URL_RE.sub(r"\1?<redacted>", out)


def _is_remote(url: str) -> bool:
    return str(url).startswith(("http://", "https://"))


def ffmpeg_cmd(url: str, *, crop: tuple[int, int] | None = None, fps: float | None = None,
               seek: float | None = None, duration: float | None = None,
               frames: int | None = None, hwaccel: str | None = None,
               ffmpeg: str = "ffmpeg") -> list[str]:
    """ffmpeg argv for rawvideo BGR frames with `showinfo` pts on stderr (§2.2 pass 2).
    `crop` = (strip_y, strip_h). Times stay absolute (`-copyts`); input seeking keeps pts
    absolute. Reconnect flags are added for http(s) inputs only.

    The `fps` filter runs with `round=up`: with ffmpeg's default (`round=near`) the output
    slot with pts t carries the last input frame with pts < t + 0.5 / fps, i.e. a sample
    labelled t shows content up to 0.25 s later than t at 2 fps (coarse onsets early by
    ~0.24 s, re-timing windows missing the onset, a biased A/V lag). With `round=up` a
    sample never shows content from after its label (at most one source frame earlier)."""
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info"]
    if _is_remote(url):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "30",
                # a stream that stays open but sends nothing (the Mac slept) errors out after
                # 30 s, and the reader restarts with a fresh signed URL
                "-rw_timeout", "30000000"]
    cmd += ["-copyts"]
    if config.FFMPEG_THREADS > 0:
        cmd += ["-threads", str(config.FFMPEG_THREADS),
                "-filter_threads", str(config.FFMPEG_THREADS)]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    if seek is not None:
        cmd += ["-ss", f"{max(0.0, seek):.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", url]
    filters = []
    if crop is not None:
        filters.append(f"crop=iw:{int(crop[1])}:0:{int(crop[0])}")
    if fps:
        filters.append(f"fps={fps:g}:round=up")
    filters.append("showinfo")
    cmd += ["-vf", ",".join(filters), "-an", "-sn"]
    if frames:
        cmd += ["-frames:v", str(int(frames))]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    return cmd


def _read_exact(stream, n: int) -> bytes | None:
    chunks, got = [], 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


class _FfmpegRun:
    """One ffmpeg process: frames from stdout, pts from showinfo lines on stderr."""

    def __init__(self, cmd: list[str], shape: tuple[int, int], popen, timeout_s: float,
                 secret: str):
        self.shape, self.timeout_s, self.secret = shape, timeout_s, secret
        self.returncode: int | None = None
        self.tail: deque[str] = deque(maxlen=20)
        self._q: queue.Queue = queue.Queue()
        self.proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            for raw in iter(self.proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
                m = _SHOWINFO_RE.search(line)
                if m:
                    self._q.put(float(m.group(1)))
                elif line.strip():
                    self.tail.append(redact(line.rstrip(), self.secret))
        finally:
            self._q.put(None)

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        h, w = self.shape
        nbytes = h * w * 3
        while True:
            buf = _read_exact(self.proc.stdout, nbytes)
            if buf is None:
                break
            try:
                pts = self._q.get(timeout=self.timeout_s)
            except queue.Empty:
                raise ScanError("ffmpeg frame without a showinfo pts (timeout)") from None
            if pts is None:
                raise ScanError("ffmpeg frame without a showinfo pts")
            yield pts, np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        self.returncode = self.proc.wait(timeout=self.timeout_s)
        self._thread.join(timeout=5)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        if self.returncode is None:
            self.returncode = self.proc.returncode

    def error(self, what: str) -> ScanError:
        tail = " | ".join(list(self.tail)[-5:])
        return ScanError(redact(f"{what} (exit {self.returncode}): {tail}", self.secret))


class FfmpegStripReader:
    """Iterator of `(t_s, strip frame)` for one class at `fps` from a signed URL (§2.2 pass 2).

    `url_fn()` is called for a fresh signed URL at every (re)start; the URL is passed to
    ffmpeg only, never logged or put in an exception. Frame times come from `showinfo`
    pts under `-copyts`: t = pts_time - a_start (the audio start_time). When ffmpeg fails
    (e.g. a 403 on an expired URL), the reader fetches a fresh URL, resumes with input `-ss`
    at (last pts - 2 s) and drops frames whose pts were already delivered.
    """

    def __init__(self, url_fn: Callable[[], str], *, width: int, strip_y: int, strip_h: int,
                 fps: float = config.F_SCAN, a_start: float = 0.0,
                 fmt_start: float | None = None, start_s: float = 0.0,
                 end_s: float | None = None, hwaccel: str | None = "videotoolbox",
                 popen=subprocess.Popen, max_restarts: int = 5, resume_back_s: float = 2.0,
                 timeout_s: float = 120.0, ffmpeg: str = "ffmpeg"):
        self.url_fn, self.width, self.strip_y, self.strip_h = url_fn, width, strip_y, strip_h
        self.fps, self.a_start = fps, float(a_start)
        self.fmt_start = self.a_start if fmt_start is None else float(fmt_start)
        self.start_s, self.end_s, self.hwaccel, self.popen = start_s, end_s, hwaccel, popen
        self.max_restarts, self.resume_back_s = max_restarts, resume_back_s
        self.timeout_s, self.ffmpeg = timeout_s, ffmpeg
        self.restarts = 0

    def command(self, url: str, t_from: float) -> list[str]:
        seek = (t_from + self.a_start - self.fmt_start) if t_from > 0 else None
        dur = None if self.end_s is None else max(0.0, self.end_s - t_from)
        return ffmpeg_cmd(url, crop=(self.strip_y, self.strip_h), fps=self.fps, seek=seek,
                          duration=dur, hwaccel=self.hwaccel, ffmpeg=self.ffmpeg)

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        last, t_from = -math.inf, float(self.start_s)
        while True:
            url = self.url_fn()
            run = _FfmpegRun(self.command(url, t_from), (self.strip_h, self.width),
                             self.popen, self.timeout_s, url)
            try:
                for pts, frame in run:
                    if pts <= last + 1e-6:
                        continue                      # already delivered before a restart
                    last = pts
                    yield pts - self.a_start, frame
            finally:
                run.close()
            if run.returncode == 0:
                return
            self.restarts += 1
            if self.restarts > self.max_restarts:
                raise run.error(f"ffmpeg failed {self.restarts} times")
            t_last = last - self.a_start if math.isfinite(last) else self.start_s
            t_from = max(float(self.start_s), t_last - self.resume_back_s)
            log.warning("ffmpeg stopped (exit %s) near t=%.1f s; restart %d with a fresh URL",
                        run.returncode, t_last, self.restarts)


class FfmpegClipReader:
    """Short reads at the native frame rate (re-timing, anchors, audits).

    `reader(t0, dur)` -> list of `(t, frame)` over [t0, t0 + dur]; `frames_at(times)` -> one
    frame per time. `crop` = (strip_y, strip_h) or None for full frames.
    """

    def __init__(self, url_fn: Callable[[], str], *, width: int, height: int,
                 crop: tuple[int, int] | None = None, fps: float | None = None,
                 a_start: float = 0.0, fmt_start: float | None = None,
                 hwaccel: str | None = None, popen=subprocess.Popen, timeout_s: float = 60.0,
                 ffmpeg: str = "ffmpeg"):
        self.url_fn, self.width, self.height, self.crop, self.fps = url_fn, width, height, crop, fps
        self.a_start = float(a_start)
        self.fmt_start = self.a_start if fmt_start is None else float(fmt_start)
        self.hwaccel, self.popen, self.timeout_s, self.ffmpeg = hwaccel, popen, timeout_s, ffmpeg

    def _run(self, t0: float, dur: float | None, frames: int | None):
        url = self.url_fn()
        cmd = ffmpeg_cmd(url, crop=self.crop, fps=self.fps,
                         seek=t0 + self.a_start - self.fmt_start, duration=dur, frames=frames,
                         hwaccel=self.hwaccel, ffmpeg=self.ffmpeg)
        h = self.crop[1] if self.crop else self.height
        run = _FfmpegRun(cmd, (h, self.width), self.popen, self.timeout_s, url)
        try:
            out = [(pts - self.a_start, f.copy()) for pts, f in run]
        finally:
            run.close()
        if run.returncode != 0:
            raise run.error("ffmpeg read failed")
        return out

    def __call__(self, t0: float, dur: float) -> list[tuple[float, np.ndarray]]:
        return self._run(t0, dur, None)

    def frames_at(self, times: Sequence[float]) -> list[tuple[float, np.ndarray]]:
        out = []
        for t in times:
            got = self._run(float(t), None, 1)
            if got:
                out.append(got[0])
        return out


def probe(url: str, *, runner=subprocess.run, ffprobe: str = "ffprobe",
          timeout_s: float = 120.0) -> dict:
    """Stream start times and size from ffprobe: {a_start, v_start, fmt_start, width, height,
    duration}. Frame time on the master clock = pts_time - a_start (§2.1.2)."""
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_entries",
           "format=start_time,duration:stream=codec_type,start_time,width,height", url]
    proc = runner(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        # redact the whole message first: a cut taken first can split the URL so that
        # neither the URL nor the query-string pattern matches any more
        err = redact(proc.stderr or "", url)
        raise ScanError(f"ffprobe failed (exit {proc.returncode}): {err[-300:]}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    vid = next((s for s in streams if s.get("codec_type") == "video"), {})
    aud = next((s for s in streams if s.get("codec_type") == "audio"), {})

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    fmt = data.get("format") or {}
    a_start = _f(aud.get("start_time"))
    fmt_start = _f(fmt.get("start_time"))
    return {"a_start": a_start if a_start is not None else (fmt_start or 0.0),
            "v_start": _f(vid.get("start_time")), "fmt_start": fmt_start,
            "width": vid.get("width"), "height": vid.get("height"),
            "duration": _f(fmt.get("duration"))}


# --------------------------------------------------------------------------- audit helpers


def _audit_dir(out_dir: Path | str | None) -> Path:
    """Resolve an audit output dir; anything not under a directory named `audit` is refused."""
    p = Path(out_dir if out_dir is not None else config.AUDIT_DIR).resolve()
    if config.AUDIT_DIR.name not in p.parts:
        raise ValueError(f"audit output must live under an '{config.AUDIT_DIR.name}/' "
                         f"directory (got {p})")
    p.mkdir(parents=True, exist_ok=True)
    return p


def assert_dev(class_ids: Iterable, what: str = "hands audit") -> None:
    """Refuse vault classes in any audit material (§8.1, L32): `select.assert_no_vault`
    (raises select.VaultError, an AssertionError), with the config rule as a fallback."""
    ids = [int(c) for c in class_ids]
    try:
        from .select import assert_no_vault
    except ImportError:                                # select stage not installed
        n_bad = sum(config.in_vault(c) for c in ids)
        if n_bad:
            raise AssertionError(f"{what}: {n_bad} vault class(es) present; dev only") from None
        return
    assert_no_vault(ids, what=what)


def _pixelate(region: np.ndarray, blocks: int) -> np.ndarray:
    h, w = region.shape[:2]
    if h < 2 or w < 2:
        return region
    small = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def render_strip(strip: np.ndarray, tiles: np.ndarray, *, geometry: StripGeometry,
                 me_tile: int | None = None, state: str | None = None,
                 t_s: float | None = None, ui: UiSpec = UI, pixelate: bool = True,
                 sharp_me: bool = True, blocks: int = config.G2_SAMPLING["pixelate_blocks"],
                 scale: int = 2) -> np.ndarray:
    """One audit frame (§2.7): tile interiors pixelated to `blocks` x `blocks` cells except
    the top-left corner square and the name label (and Brady's tracked tile when
    `sharp_me`); the tracked tile outlined or an "NA" banner; burned-in timecode and a
    detector-state dot (green UP, grey DOWN, red NA)."""
    img = strip.copy()
    ty0, th = geometry.tile_y0, geometry.tile_h
    corner = ui.roi_inset_px + round(ui.roi_s2 * th)
    label = round(ui.label_h_frac * th)
    if pixelate:
        for k, (x0, x1) in enumerate(tiles):
            if sharp_me and me_tile is not None and k == me_tile:
                continue
            body = img[ty0:ty0 + th - label, x0:x1]
            keep = body[:corner, :corner].copy()
            body[:] = _pixelate(body, blocks)
            body[:corner, :corner] = keep
    if me_tile is not None and 0 <= me_tile < len(tiles):
        x0, x1 = tiles[me_tile]
        cv2.rectangle(img, (int(x0) - 2, max(0, ty0 - 2)),
                      (int(x1) + 1, min(img.shape[0] - 1, ty0 + th + 1)), (0, 0, 255), 1)
    head = np.zeros((AUDIT_HEADER_PX, img.shape[1], 3), np.uint8)
    if t_s is not None:
        hh, rem = divmod(float(t_s), 3600.0)
        mm, ss = divmod(rem, 60.0)
        cv2.putText(head, f"{int(hh):02d}:{int(mm):02d}:{ss:05.2f}", (4, 13),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    colour = {UP: (0, 200, 0), DOWN: (150, 150, 150), NA: (0, 0, 230)}.get(state or NA)
    cv2.circle(head, (128, 9), 6, colour, -1)
    if me_tile is None:
        cv2.putText(head, "NA", (150, 14), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255), 2)
    out = np.vstack([head, img])
    return cv2.resize(out, (out.shape[1] * scale, out.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST) if scale != 1 else out


def _frame_lookup(hand_frames: pd.DataFrame):
    t = hand_frames["t_s"].to_numpy(float)
    found = hand_frames["tile_found"].to_numpy(bool)
    idx = hand_frames["tile_idx"].fillna(-1).to_numpy(int)
    state = hand_frames["raw_state"].astype(str).to_numpy()

    def at(ts: float) -> tuple[int | None, str]:
        if len(t) == 0:
            return None, NA
        i = int(np.clip(np.searchsorted(t, ts), 1, len(t) - 1)) if len(t) > 1 else 0
        if len(t) > 1 and abs(t[i - 1] - ts) <= abs(t[i] - ts):
            i -= 1
        return (int(idx[i]) if found[i] else None), str(state[i])
    return at


def _write_clip(path: Path, frames: Iterable[np.ndarray], fps: float) -> Path:
    writer = None
    try:
        for img in frames:
            if writer is None:
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (img.shape[1], img.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open a video writer for {path.name}")
            writer.write(img)
    finally:
        if writer is not None:
            writer.release()
    return path


def audit_clip(class_id, strip_source: Callable[[float, float, float], Iterable],
               hand_frames: pd.DataFrame, *, geometry: StripGeometry, t0: float = 0.0,
               t1: float | None = None, out_dir: Path | str | None = None,
               pixelate: bool = True, sharp_me: bool = True, name: str | None = None,
               ui: UiSpec = UI) -> Path:
    """Recall-audit clip (§2.7 Recall a): the full strip at 2 fps rendered at 16 fps (8x),
    tile interiors pixelated, the tracker's tile outlined or an NA banner, timecode and
    state dot. `strip_source(t0, t1, fps)` yields `(t, strip)`. Written under audit/ only."""
    assert_dev([class_id])
    out = _audit_dir(out_dir) / (name or f"recall_{int(class_id)}.mp4")
    fps = config.F_SCAN
    t1 = float(hand_frames["t_s"].max()) if t1 is None else t1
    at = _frame_lookup(hand_frames)

    def frames():
        for t, frame in strip_source(t0, t1, fps):
            strip = as_strip(np.asarray(frame), geometry)
            me, state = at(float(t))
            yield render_strip(strip, strip_tiles(strip, ui=ui), geometry=geometry, me_tile=me,
                               state=state, t_s=float(t), ui=ui, pixelate=pixelate,
                               sharp_me=sharp_me)
    return _write_clip(out, frames(), fps * config.G2_SAMPLING["audit_speedup"])


def onset_clips(sample: pd.DataFrame, strip_sources: dict, hand_frames: dict,
                geometries: dict, *, span_s: float = config.G2_SAMPLING["onset_clip_span_s"],
                out_dir: Path | str | None = None, pixelate: bool = True,
                ui: UiSpec = UI) -> list[Path]:
    """Recall-audit clips (§2.7 Recall b): one full-strip clip over [o_s - 120, o_s] per
    sampled fresh speech onset (`sample`: class_id, onset_s), rendered like `audit_clip`
    (240 samples at 8x = 15 s). Inputs are dicts keyed by class_id."""
    assert_dev(sample["class_id"])
    paths = []
    for j, r in enumerate(sample.itertuples()):
        cid = int(r.class_id)
        paths.append(audit_clip(cid, strip_sources[cid], hand_frames[cid],
                                geometry=geometries[cid], t0=max(0.0, r.onset_s - span_s),
                                t1=float(r.onset_s), out_dir=out_dir, pixelate=pixelate,
                                name=f"onset_{j:03d}_{cid}_{int(r.onset_s)}.mp4", ui=ui))
    return paths


def contact_sheet(sample: pd.DataFrame, strip_at: Callable[[int, float], np.ndarray | None],
                  *, templates: dict, geometries: dict, out_dir: Path | str | None = None,
                  tag: str = "precision", offsets: Sequence[float] = (-1.0, 0.0, 1.0),
                  per_page: int = 25, zoom: int = 3, ui: UiSpec = UI,
                  det: Detector = DETECTOR) -> list[Path]:
    """Precision audit sheet (§2.7): one row per sampled episode with crops of Brady's tile
    only (found by glyph NCC) at o - 1, o and o + 1 s; an "NA" cell when untracked. Writes
    PNG pages plus a CSV manifest with an empty `verdict` column, under audit/ only."""
    assert_dev(sample["class_id"])
    out = _audit_dir(out_dir)
    rows_img, manifest = [], []
    cell_h, cell_w = 0, 0
    items = list(sample.reset_index(drop=True).itertuples())
    for i, r in enumerate(items):
        cid = int(r.class_id)
        geom, tmpl = geometries[cid], templates.get(cid)
        crops = []
        for off in offsets:
            frame = strip_at(cid, float(r.onset_s) + off)
            crop = None
            if frame is not None and tmpl is not None:
                strip = as_strip(np.asarray(frame), geom)
                span = locate_me(strip, geom, tmpl, ui=ui, det=det)
                if span is not None:
                    crop = strip[geom.tile_y0:geom.tile_y0 + geom.tile_h, span[0]:span[1]]
            crops.append(crop)
        cell_h = max(cell_h, *(c.shape[0] for c in crops if c is not None), 1)
        cell_w = max(cell_w, *(c.shape[1] for c in crops if c is not None), 1)
        rows_img.append((i, r, crops))
        manifest.append({"item": i, "class_id": cid, "ep_id": getattr(r, "ep_id", ""),
                         "onset_s": round(float(r.onset_s), 3),
                         "ep_type": getattr(r, "ep_type", ""),
                         "seg_type": getattr(r, "seg_type", ""),
                         "stratum": getattr(r, "stratum", ""), "verdict": ""})
    paths = []
    label_w = 150
    for p0 in range(0, len(rows_img), per_page):
        page = []
        for i, r, crops in rows_img[p0:p0 + per_page]:
            cells = []
            for c in crops:
                cell = np.zeros((cell_h, cell_w, 3), np.uint8)
                if c is None:
                    cv2.putText(cell, "NA", (2, min(cell_h - 2, 14)), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 255), 1)
                else:
                    cell[:c.shape[0], :c.shape[1]] = c
                cells.append(cv2.resize(cell, (cell_w * zoom, cell_h * zoom),
                                        interpolation=cv2.INTER_NEAREST))
            text = np.zeros((cell_h * zoom, label_w, 3), np.uint8)
            cv2.putText(text, f"#{i} {getattr(r, 'ep_type', '')}", (4, 16),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
            sep = np.full((cell_h * zoom, 4, 3), 60, np.uint8)
            page.append(np.hstack([text] + [x for c in cells for x in (c, sep)]))
            page.append(np.full((3, page[-1].shape[1], 3), 90, np.uint8))
        path = out / f"contact_{tag}_{p0 // per_page:02d}.png"
        cv2.imwrite(str(path), np.vstack(page))
        paths.append(path)
    man = out / f"contact_{tag}.csv"
    with open(man, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0]) if manifest else ["item"])
        writer.writeheader()
        writer.writerows(manifest)
    paths.append(man)
    return paths


def precision_sample(hand_episodes: pd.DataFrame, *, segments: pd.DataFrame | None = None,
                     flagged: Iterable = (), seed: int = config.SEED) -> pd.DataFrame:
    """Stratified precision sample (§2.7): up to 3 random bids per class, every bid in a poll
    segment (cap 40), every typed vote (cap 40), every bid of a flagged class. Dev only."""
    s = config.G2_SAMPLING
    e = hand_episodes[~hand_episodes["left_censored"].astype(bool)]
    assert_dev(e["class_id"].unique())
    rng = np.random.default_rng(seed)
    bids = e[e["ep_type"] == config.EP_BID]
    parts = []
    for _cid, g in bids.groupby("class_id"):
        take = rng.permutation(len(g))[: s["precision_per_class"]]
        parts.append(g.iloc[np.sort(take)].assign(stratum="per_class"))
    poll = bids[bids["seg_type"] == "poll"]
    parts.append(poll.iloc[rng.permutation(len(poll))[: s["precision_poll_cap"]]]
                 .assign(stratum="poll"))
    votes = e[e["ep_type"] == config.EP_VOTE]
    parts.append(votes.iloc[rng.permutation(len(votes))[: s["precision_vote_cap"]]]
                 .assign(stratum="vote"))
    flag = {int(c) for c in flagged}
    parts.append(bids[bids["class_id"].astype(int).isin(flag)].assign(stratum="flagged"))
    out = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(
        len(p) for p in parts) else e.head(0).assign(stratum=pd.Series(dtype=str))
    return out.drop_duplicates(["class_id", "ep_id"]).reset_index(drop=True)


def flagged_audit_runs(hand_frames: pd.DataFrame, det: Detector = DETECTOR) -> pd.DataFrame:
    """Audit material for a flagged class (§2.3: DOWN baseline > θ_off / 2): runs of >= k
    tracked samples that pass θ_on and the shape test BEFORE the whole-class baseline is
    subtracted but are not UP in `raw_state`. A late lime object can raise the baseline and
    erase an earlier real raise; such raises are no longer bids, so "audit every bid of a
    flagged class" cannot find them. Rows: class_id, start_s, end_s (aggregate times only)."""
    cols = ["class_id", "start_s", "end_s"]
    if hand_frames is None or not len(hand_frames):
        return pd.DataFrame(columns=cols)
    hf = hand_frames.sort_values("t_s")
    b = float(hand_frames.attrs.get("down_baseline", 0.0) or 0.0)
    frac = hf["frac_hand_adj"].to_numpy(float) + b
    cand = (np.nan_to_num(frac) >= det.theta_on) & hf["shape_ok"].to_numpy(bool) & \
        (hf["raw_state"].astype(str).to_numpy() != UP)
    t = hf["t_s"].to_numpy(float)
    cid = int(hf["class_id"].iloc[0])
    rows = [{"class_id": cid, "start_s": float(t[a]), "end_s": float(t[e - 1])}
            for a, e in _runs(cand) if e - a >= det.k]
    return pd.DataFrame(rows, columns=cols)


def onset_sample(speech_onsets: pd.DataFrame, n: int = config.G2_SAMPLING["recall_onset_clips"],
                 *, seed: int = config.SEED) -> pd.DataFrame:
    """Random fresh speech onsets for the recall (b) clips, dev classes only."""
    fresh = speech_onsets[speech_onsets["fresh"].astype(bool)]
    fresh = fresh[~fresh["class_id"].map(lambda c: config.in_vault(int(c)))]
    rng = np.random.default_rng(seed)
    take = rng.permutation(len(fresh))[:n]
    return fresh.iloc[np.sort(take)].reset_index(drop=True)


# --------------------------------------------------------------------------- diagnostics


def left_censored_share(hand_episodes: pd.DataFrame) -> float:
    """Share of UP runs (non-glitch episodes) that are left-censored; > 10% means tracking is
    fixed before G2 (§2.2.9, §2.7)."""
    e = hand_episodes[hand_episodes["ep_type"] != config.EP_GLITCH]
    return float(e["left_censored"].astype(bool).mean()) if len(e) else float("nan")


def na_rate_before(hand_frames: pd.DataFrame, times: Iterable[float],
                   span_s: float = 60.0) -> float:
    """Share of samples in [t - span_s, t] (pooled over `times`) where the tile is not found
    or the raw state is NA. Compare fresh speech onsets with random baseline times (§2.2.9:
    observability must not depend on future speech, L20)."""
    t = hand_frames["t_s"].to_numpy(float)
    bad = (~hand_frames["tile_found"].fillna(False).to_numpy(bool)
           | (hand_frames["raw_state"].astype(str).to_numpy() == NA))
    tot = n_bad = 0
    for x in times:
        a, b = np.searchsorted(t, x - span_s, "left"), np.searchsorted(t, x, "right")
        tot += b - a
        n_bad += int(bad[a:b].sum())
    return n_bad / tot if tot else float("nan")


def vote_diagnostics(hand_episodes: pd.DataFrame, hand_frames: pd.DataFrame | None = None,
                     vote_frac: float = config.VOTE_FRAC) -> dict:
    """§2.5 vote validation: share of Δn-typed votes with a VOTE-lexicon hit (expect >= 0.6),
    share of lexicon-typed votes with Δn_others >= 2, and the counts behind them. The Δn
    threshold uses n_tiles at the onset from `hand_frames` (else the minimum, 4)."""
    v = hand_episodes[hand_episodes["ep_type"] == config.EP_VOTE]
    dn = v["delta_n_others"].fillna(0).astype(int).to_numpy()
    lex = v["vote_lex"].astype(bool).to_numpy()
    thr = np.full(len(v), config.VOTE_MIN_DN)
    if hand_frames is not None and len(v):
        for j, r in enumerate(v.itertuples()):
            hf = hand_frames[hand_frames["class_id"] == r.class_id].sort_values("t_s")
            if len(hf):
                i = max(0, int(np.searchsorted(hf["t_s"].to_numpy(float), r.onset_s)) - 1)
                thr[j] = config.vote_threshold(int(hf["n_tiles"].iloc[i]), vote_frac)
    dn_typed = dn >= thr
    return {"n_votes": len(v), "n_dn_typed": int(dn_typed.sum()),
            "n_lex_typed": int(lex.sum()),
            "share_dn_with_lex": float(lex[dn_typed].mean()) if dn_typed.any() else float("nan"),
            "share_lex_with_dn2": float((dn[lex] >= 2).mean()) if lex.any() else float("nan"),
            "n_contested": int(v["contested"].astype(bool).sum())}


# --------------------------------------------------------------------------- class summary


def class_summary(cid, *, geometry: StripGeometry, hand_frames: pd.DataFrame,
                  hand_episodes: pd.DataFrame, n_anchor_events: int = 0,
                  valid_share: float = float("nan"), era: str | None = None,
                  exclude: Iterable[tuple[float, float]] = (), av_lag_s: float = float("nan"),
                  av_lag_brady_s: float = float("nan")) -> dict:
    """The `classes` columns this module owns, for the select / labels stage to merge."""
    cov = coverage(hand_frames, exclude) if len(hand_frames) else float("nan")
    status, validated = tile_status(geometry, cov, n_anchor_events, valid_share)
    e = hand_episodes
    return {"class_id": int(cid), "era": era, "strip_y": float(geometry.strip_y),
            "strip_h": float(geometry.strip_h), "strip_found": bool(geometry.found),
            "template_validated": bool(validated), "coverage": cov, "tile_status": status,
            "down_baseline": float(hand_frames.attrs.get("down_baseline", np.nan)),
            "flagged": bool(hand_frames.attrs.get("flagged", False)),
            "av_lag_s": av_lag_s, "av_lag_brady_s": av_lag_brady_s,
            "n_bids": int(((e["ep_type"] == config.EP_BID) & ~e["left_censored"]).sum()),
            "n_votes": int((e["ep_type"] == config.EP_VOTE).sum()),
            "n_contested": int(e["contested"].astype(bool).sum()),
            "n_anchor_events": int(n_anchor_events), "track_valid_share": valid_share,
            "n_prebaseline_runs": (len(flagged_audit_runs(hand_frames))
                                   if hand_frames.attrs.get("flagged") else 0),
            "ncc_floor": float(hand_frames.attrs.get("ncc_floor", np.nan)),
            "n_lookalike_frames": int(hand_frames.attrs.get("n_lookalike_frames", 0)),
            "n_floor_rejected": int(hand_frames.attrs.get("n_floor_rejected", 0))}


# --------------------------------------------------------------------------- CLI


def _pass1_dir() -> Path:
    return config.CACHE_DIR / "hands"


def hand_tiles_path(cid) -> Path:
    """Per-class `hand_tiles` file (large, local): data/tables/hand_tiles/{cid}.parquet."""
    return config.TABLES_DIR / "hand_tiles" / f"{int(cid)}.parquet"


def _read_table(name: str) -> pd.DataFrame | None:
    p = config.table_path(name)
    return pd.read_parquet(p) if p.exists() else None


def _for_class(df: pd.DataFrame | None, cid) -> pd.DataFrame | None:
    return None if df is None else df[df["class_id"] == int(cid)].reset_index(drop=True)


def _selected_classes(args) -> pd.DataFrame:
    classes = _read_table("classes")
    if classes is None:
        raise SystemExit("oncue hands: data/tables/classes.parquet is missing (run select)")
    if getattr(args, "class_ids", None):
        classes = classes[classes["class_id"].isin([int(c) for c in args.class_ids])]
    elif getattr(args, "split", None):             # the classes of one step-3 split
        from .splits import split_ids

        classes = classes[classes["class_id"].isin(split_ids(args.split))]
    elif "phase" in classes:                       # the selected population only
        classes = classes[classes["phase"].notna()]
    if getattr(args, "phase", None):
        classes = classes[classes["phase"] == args.phase]
    return classes.reset_index(drop=True)


def _era(row) -> str:
    era = getattr(row, "era", None)
    return str(era) if isinstance(era, str) and era else str(getattr(row, "term", "all"))


def load_pass1(cid) -> Pass1 | None:
    p = _pass1_dir() / f"{int(cid)}.json"
    return Pass1.from_json(json.loads(p.read_text())) if p.exists() else None


def _pass1_json(cid) -> dict:
    p = _pass1_dir() / f"{int(cid)}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_pass1(res: Pass1) -> Path:
    d = _pass1_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{res.class_id}.json"
    p.write_text(json.dumps(res.to_json()))
    return p


def _anchors_eras(classes: pd.DataFrame) -> tuple[dict, dict]:
    eras, anchors = {}, {}
    for r in classes.itertuples():
        eras[int(r.class_id)] = _era(r)
        res = load_pass1(r.class_id)
        if res is not None:
            anchors[int(r.class_id)] = res
    return anchors, eras


def templates_for(classes: pd.DataFrame) -> dict:
    """LOCO era templates for every class with a cached pass 1 (all eras' anchors)."""
    anchors, eras = _anchors_eras(classes)
    return build_era_templates(anchors, eras, loco=True)


def floors_for(classes: pd.DataFrame, templates: dict) -> dict:
    """LOCO impostor NCC floors (`ncc_floors`) for every class with a cached pass 1."""
    anchors, eras = _anchors_eras(classes)
    return ncc_floors(anchors, templates, eras)



# "strip": the real Forum UI (`oncue.strip`, labels read by OCR); "glyph": the synthetic
# world's UI (name-glyph NCC, below). The synthetic end-to-end run switches to "glyph".
SCANNER = "strip"


def _detector() -> Detector:
    """The thresholds of the active scanner (`strip.FORUM_DETECTOR` for real recordings)."""
    if SCANNER == "strip":
        from .strip import FORUM_DETECTOR

        return FORUM_DETECTOR
    return DETECTOR

_URL_LOCK = threading.Lock()          # one Forum request at a time across scan workers
URL_REUSE_S = 120.0                   # PROVISIONAL: well under the URL lifetime D2 measures


def _forum_client():
    """ForumClient from FORUM_COOKIE (loaded from pipeline1/.env). Never used by tests."""
    from scribe_core.forum import ForumClient

    config.load_env()
    return ForumClient()


def _sources(cid, client, res: str = config.SRC_RES) -> dict:
    """Real frame sources for one class: fresh-URL function, probe info, full-frame grabber.
    Signed URLs live in memory only (never logged or stored). The pass-2 strip reader takes
    a fresh URL at every (re)start; the many short pass-1 reads reuse one URL for up to
    URL_REUSE_S. Tests replace this function; nothing here is reached offline."""
    def url_fn() -> str:
        with _URL_LOCK:
            urls = client.recording_urls(cid)
        url = urls.get(res)                 # strict: same rendition as the ASR (§2.1.1)
        if not url:
            raise ScanError(f"class {cid}: no {res} recording URL")
        return url

    held = {"url": None, "at": -math.inf}

    def reused_url() -> str:
        if held["url"] is None or time.monotonic() - held["at"] > URL_REUSE_S:
            held["url"], held["at"] = url_fn(), time.monotonic()
        return held["url"]

    media = probe(reused_url())
    grab = FfmpegClipReader(reused_url, width=int(media["width"]),
                            height=int(media["height"]), a_start=media["a_start"],
                            fmt_start=media["fmt_start"])
    return {"url_fn": url_fn, "media": media, "frame_reader": grab.frames_at,
            "strip_source": None, "clip_reader": None}


def _n_present(presence: pd.DataFrame | None, T_end: float) -> list[int] | None:
    if presence is None or len(presence) == 0:
        return None
    times = np.linspace(0.0, T_end, config.PASS1_N_FRAMES + 2)[1:-1]
    p = presence[presence["role"] != config.ROLE_PROF]
    return [int((p["enter_s"] <= t).sum()) for t in times]


def _upsert(name: str, df: pd.DataFrame) -> Path:
    """Write `df` into data/tables/{name}.parquet, replacing only its classes' rows."""
    path = config.table_path(name)
    old = _read_table(name)
    if old is not None and len(old):
        old = old[~old["class_id"].isin(df["class_id"].unique())]
        df = pd.concat([old, df], ignore_index=True) if len(old) else df
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("class_id", kind="stable").to_parquet(path, index=False)
    return path


def _scan_one(cid: int, src: dict, res: Pass1, template) -> Path:
    ht = scan(cid, frame_source=src.get("strip_source"), geometry=res.geometry,
              template=template, url_fn=src.get("url_fn"), media=src.get("media"))
    out = hand_tiles_path(cid)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    ht.to_parquet(tmp, index=False)
    tmp.replace(out)                       # a crashed scan never leaves a partial file
    return out


def cli_scan(args) -> int:
    """`oncue scan`: pass 1 for every selected class (cached), LOCO era templates from all
    cached anchors, then pass 2 for classes without hand_tiles (`--workers` in parallel).
    Real recordings go to `strip.cli_scan` (SCANNER = "strip")."""
    from concurrent.futures import ThreadPoolExecutor

    if SCANNER == "strip":
        from . import strip

        return strip.cli_scan(args)

    classes = _selected_classes(args)
    voice_all, presence_all = _read_table("voice"), _read_table("presence")
    client = _forum_client()
    srcs: dict[int, dict] = {}

    def src(cid: int) -> dict:
        if cid not in srcs:
            srcs[cid] = _sources(cid, client)
        return srcs[cid]

    for r in classes.itertuples():
        cid = int(r.class_id)
        if load_pass1(cid) is None:
            res = pass1(cid, src(cid)["frame_reader"], T_end_s=float(r.T_end_s),
                        voice=_for_class(voice_all, cid),
                        n_present=_n_present(_for_class(presence_all, cid), float(r.T_end_s)))
            save_pass1(res)
            log.info("class %s: strip %s, %d anchor events", cid, res.geometry.reason,
                     res.n_anchor_events)
    templates = templates_for(_read_table("classes"))
    todo = []
    for r in classes.itertuples():
        cid, res = int(r.class_id), load_pass1(r.class_id)
        if not hand_tiles_path(cid).exists() and res is not None and res.geometry.found:
            todo.append((cid, src(cid), res, templates.get(cid)))
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path in pool.map(lambda job: _scan_one(*job), todo):
            log.info("wrote %s", path.name)
    return 0


def _breakout_spans(segments: pd.DataFrame | None, T_end: float) -> list[tuple[float, float]]:
    if segments is None or len(segments) == 0:
        return []
    s = segments.sort_values("start_s")
    starts = s["start_s"].to_numpy(float)
    ends = np.r_[starts[1:], T_end]
    return [(a, b) for a, b, k in zip(starts, ends, s["seg_type"], strict=True)
            if k == "breakout"]


def cli_track(args) -> int:
    """`oncue track`: hand_tiles -> hand_frames, plus class-level detector QA (strip,
    template validation, coverage, baseline, A/V lag) in data/tables/hand_qc.parquet."""
    classes = _selected_classes(args)
    all_classes = _read_table("classes")
    templates = templates_for(all_classes)
    floors = floors_for(all_classes, templates)
    words_all, segs_all = _read_table("words"), _read_table("segments")
    frames, qc = [], []
    for r in classes.itertuples():
        cid = int(r.class_id)
        path, res = hand_tiles_path(cid), load_pass1(cid)
        if not path.exists() or res is None:
            continue
        ht = pd.read_parquet(path)
        hf = track(ht, templates.get(cid), ncc_floor=floors.get(cid), det=_detector())
        if hf.attrs.get("n_lookalike_frames", 0):
            log.warning("class %s: %d frames with a lookalike name label (audit the section)",
                        cid, hf.attrs["n_lookalike_frames"])
        share, _ = validate_tracking(res.anchors, templates.get(cid), ncc_floor=floors.get(cid))
        info = _pass1_json(cid)
        if info.get("scanner") == "strip":    # anchor check made during the strip scan
            share = float("nan") if info.get("valid_share") is None else info["valid_share"]
        words = _for_class(words_all, cid)
        lag = av_lag(hf, words) if words is not None else float("nan")
        lag_me = av_lag_brady(hf, ht, words) if words is not None else float("nan")
        qc.append(class_summary(cid, geometry=res.geometry, hand_frames=hf,
                                hand_episodes=schemas.empty("hand_episodes"),
                                n_anchor_events=res.n_anchor_events, valid_share=share,
                                era=_era(r), av_lag_s=lag, av_lag_brady_s=lag_me,
                                exclude=_breakout_spans(_for_class(segs_all, cid),
                                                        float(r.T_end_s))))
        frames.append(hf)
    if frames:
        _upsert("hand_frames", pd.concat(frames, ignore_index=True))
        _upsert("hand_qc", pd.DataFrame(qc))
    return 0


def cli_episodes(args) -> int:
    """`oncue episodes`: hand_frames -> hand_episodes (votes, outcomes, seg_type); updates the
    bid / vote / contested counts in hand_qc."""
    classes = _selected_classes(args)
    hf_all = _read_table("hand_frames")
    if hf_all is None:
        raise SystemExit("oncue episodes: run `oncue track` first")
    words, voice, segs = _read_table("words"), _read_table("voice"), _read_table("segments")
    out = []
    for r in classes.itertuples():
        cid = int(r.class_id)
        hf = _for_class(hf_all, cid)
        if hf is None or len(hf) == 0:
            continue
        w = _for_class(words, cid)
        out.append(episodes(hf, words=w, voice=_for_class(voice, cid),
                            segments=_for_class(segs, cid), q_open=q_open_window(w),
                            det=_detector()).assign(class_id=cid))
    if not out:
        return 0
    eps = pd.concat(out, ignore_index=True)
    old_eps = _read_table("hand_episodes")
    if old_eps is not None and len(old_eps):
        mine = old_eps[old_eps["class_id"].isin(eps["class_id"].unique())]
        n_rt = int(mine["onset_retimed_s"].notna().sum()) if "onset_retimed_s" in mine else 0
        if n_rt:
            log.warning("oncue episodes: %d re-timed onsets in hand_episodes are being replaced "
                        "by coarse ones; run `oncue retime` again (A1, g <= 1)", n_rt)
    _upsert("hand_episodes", eps)
    qc = _read_table("hand_qc")
    if qc is not None and len(qc):
        e = eps.assign(bid=(eps["ep_type"] == config.EP_BID) & ~eps["left_censored"].astype(bool),
                       vote=eps["ep_type"] == config.EP_VOTE,
                       cont=eps["contested"].astype(bool))
        counts = e.groupby("class_id")[["bid", "vote", "cont"]].sum()
        done = qc["class_id"].isin([int(c) for c in classes["class_id"]])
        for col, src_col in (("n_bids", "bid"), ("n_votes", "vote"), ("n_contested", "cont")):
            qc.loc[done, col] = (qc.loc[done, "class_id"].map(counts[src_col]).fillna(0)
                                 .astype(int))
        qc.to_parquet(config.table_path("hand_qc"), index=False)
    return 0


def cli_retime(args) -> int:
    """`oncue retime`: full-frame-rate re-read around every onset (fresh signed URLs)."""
    if SCANNER == "strip":
        from . import strip

        return strip.cli_retime(args)
    classes = _selected_classes(args)
    eps_all = _read_table("hand_episodes")
    if eps_all is None:
        raise SystemExit("oncue retime: run `oncue episodes` first")
    templates = templates_for(_read_table("classes"))
    qc = _read_table("hand_qc")
    client = _forum_client()
    out = []
    for r in classes.itertuples():
        cid = int(r.class_id)
        eps, res = _for_class(eps_all, cid), load_pass1(cid)
        if eps is None or len(eps) == 0 or res is None:
            continue
        src = _sources(cid, client)
        reader = src.get("clip_reader") or FfmpegClipReader(
            src["url_fn"], width=int(src["media"]["width"]),
            height=int(src["media"]["height"]),
            crop=(res.geometry.strip_y, res.geometry.strip_h),
            a_start=src["media"]["a_start"], fmt_start=src["media"]["fmt_start"])
        base = 0.0
        if qc is not None and (qc["class_id"] == cid).any():
            base = float(qc.loc[qc["class_id"] == cid, "down_baseline"].iloc[0])
        out.append(retime(eps, reader, geometry=res.geometry, template=templates.get(cid),
                          base=base))
    if out:
        eps = pd.concat(out, ignore_index=True)
        _upsert("hand_episodes", eps)
        update_retime_qc(eps)
    return 0


def update_retime_qc(eps: pd.DataFrame) -> pd.DataFrame | None:
    """A1: per-class and pooled retime_failed share (non-censored bids) into hand_qc
    (columns n_retimed, n_retime_failed, retime_failed_share, g1_approximate); logs the
    pooled share and whether g = 1 must be reported as approximate."""
    rows = [{"class_id": int(c), **retime_summary(g)} for c, g in eps.groupby("class_id")]
    pooled = retime_summary(eps)
    log.info("retime: %d/%d bid onsets failed or not re-timed (share %.2f)%s",
             pooled["n_retime_failed"], pooled["n_bids"], pooled["retime_failed_share"],
             "; g = 1 is APPROXIMATE (A1)" if pooled["g1_approximate"] is True else "")
    qc = _read_table("hand_qc")
    if qc is None or not rows:
        return None
    add = pd.DataFrame(rows).set_index("class_id")
    for col in ("n_retimed", "n_retime_failed", "retime_failed_share", "g1_approximate"):
        vals = qc["class_id"].map(add[col])
        qc[col] = vals.where(qc["class_id"].isin(add.index), qc[col]) if col in qc else vals
    qc.to_parquet(config.table_path("hand_qc"), index=False)
    return qc
