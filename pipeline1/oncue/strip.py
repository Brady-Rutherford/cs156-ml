"""The real Forum participant strip: find Brady's tile by its name label, measure the raised-hand
icon in its corner (D2, measured on 13 Oct 2026 on the 20 pilot classes, 7 terms).

What the recordings show (1920 x 1080; `lowres` has the same pixels at a lower bitrate):

* the strip is one row of equal tiles near the top, usually right under a 50 px dark header
  (y = 50/51; lower when a banner sits above it). Tiles are about 4:3 and 69-200 px wide,
  depending on how many people are present, and the strip reflows within a class, so the
  band and the tiles are measured on every frame. Brady is always on this strip (also while
  he is on the main stage), so the stage tiles are never read;
* each tile shows the person's name in white text over the video, near the bottom;
* the speaker border is a teal ring (OpenCV hue 75-77). It only means "speaking": it is
  used to check the tracking during Brady's own voice events (and for the A/V lag), never
  as a label;
* the label is the raised-hand icon: a yellow-green disc (hue 26-40) with a white hand,
  in the tile's top-left corner.

Brady's tile: every name label is read with Tesseract OCR and matched to the class roster
(`LabelBook`). Names stay in memory; what is stored is only whether a tile is Brady's
(`ncc_me` = 1) and anonymous label patches in the local cache. A label already read at the
same tile width is recognised from its pixels (NCC of the blurred word patch), so OCR runs
about once per (person, tile width) instead of on every frame.

`scan_class` writes `hand_tiles` in the layout `hands.track` reads, plus a pass-1 record
(median geometry; anchor check: in frames inside Brady's isolated voice events, the one
bordered tile must be the tile his label marks). `retime_class` re-reads every onset at the
full frame rate and finds his tile on each frame by its label, so a strip that reorders
when a hand goes up cannot move the read to someone else's tile.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config, hands, schemas

log = logging.getLogger("oncue.strip")

# Measured on real frames (13 Oct): the speaker ring sits at hue 75-77 in every pilot class;
# the icon starts 4-5 px in from the tile's top-left corner and is 16 px wide on tiles up to
# 74 px high, 22-24 px up to 96 px and 31 px at 156 px, so a corner square of 34% of the tile
# height holds it at every size. Its hue pixels fill 0.2-0.45 of that square; tiles without
# an icon read 0 at the 99th percentile, so UP needs 0.10 (and the disc-with-a-white-hand
# shape test) and DOWN is below 0.04.
FORUM_UI = hands.UiSpec(border_h_range=(68, 86), roi_s1=0.34, roi_s2=0.40)
# Reactions and the raise overlay hide the label for ~1.5-2 s, so NA runs up to 3 s are
# bridged, and an onset reached through such a run after clean DOWN is kept (re-timed).
# Votes (Brady, 13 Oct): only when the professor calls for one (the VOTE lexicon) or 6+ others
# raise within [onset - 5, onset + 10] s; no share-of-class rule.
FORUM_DETECTOR = hands.Detector(theta_on=0.10, theta_off=0.04, na_bridge_s=3.0,
                                na_hold=6, pre_na_onset_s=3.0, vote_min_dn=6, vote_frac=0.0)

SCAN_ROWS = 360                  # rows read from the top of the frame (the strip is above this)
BAND_FROM_Y = 30
BAND_NONDARK_MIN = 0.20          # a strip row has >= 20% non-dark pixels ...
BAND_END_NONDARK = 0.10          # ... and ends at a row with < 10% (the gap below)
BAND_X_MAX = 1500                # the right-hand panel starts further right
TILE_H_RANGE = (30, 260)
TILE_ASPECT = (0.55, 1.0)        # tile height / width
LABEL_FROM = 0.58                # the name sits in the bottom 42% of the tile
LABEL_S_MAX, LABEL_V_MIN = 40, 225   # name text: near-pure white
WORD_H, WORD_W = 16, 96
WORD_BLUR = 1.0
SAME_LABEL_NCC = 0.85            # same person at the same width: median 0.93; others <= 0.69
# ... but a label is taken as Brady's without OCR only when it is this close to one of his. Full
# run, 14 Oct (80193): a classmate's label scored 0.91 against his stored labels, so it was
# taken as his, two tiles were his, and his tile went unread in 57% of his solo speech. Below
# this, the label is read once, and the classmate's label gets its own entry.
ME_SAME_LABEL_NCC = 0.95
SAME_WIDTH_PX = 2
ME_LABEL_NCC = 0.80              # a tile matching one of Brady's read labels at this width,
ME_LABEL_NCC_ANY = 0.88          # or at any width (a reaction tints the label; tiles resize)
ME_LABEL_GAP = 0.15              # ... and beating every other tile by this much
OCR_RETRIES = 6                  # an unread label is OCR'd again on later frames
# Courses where one OCR misread of another tile's label as "Brady" stored that tile's pixels
# as one of his labels (B111: 2 stray patches made his tile ambiguous in 41% of frames). There a
# new label read as his is used only once confirmed: it resembles one of his confirmed labels
# (NCC >= ME_CONFIRM_NCC at any width), or OCR reads it as his again on a later frame.
STRICT_ME_LABEL_COURSES = frozenset({"B111"})
ME_CONFIRM_NCC = 0.75
NAME_MATCH_MIN = 0.60            # difflib ratio between the OCR text and a roster name ...
NAME_MATCH_MARGIN = 0.12         # ... that beats every other person's best ratio by this
# A label counts as Brady's only when its OCR text is this close to his name. Full run, 14 Oct
# (83098): a classmate whose display name is not on the roster read as him at 0.67, so her
# label joined his, both tiles then matched him, and his tile went unread for minutes at a
# time. His own label reads at 0.80-1.00 ("Bradl y", "° Brady"); names at 0.67 ("Andy").
ME_NAME_MIN = 0.75
# Tesseract's usual confusions on small white-on-video text ("Bracly" for "Brady").
OCR_FIXES = (("cl", "d"), ("ci", "d"), ("rn", "m"), ("0", "o"), ("1", "l"), ("|", "l"))


# --------------------------------------------------------------------------- one frame


@dataclass
class Band:
    y0: int
    y1: int
    tiles: np.ndarray            # (n, 2) x-spans

    @property
    def h(self) -> int:
        return self.y1 - self.y0


GUTTER_VALLEY_MAX = 0.30         # a gutter lifted by the encoder still stays this dark ...
GUTTER_VALLEY_RATIO = 0.5        # ... and at most half as bright as the tiles on both sides
GUTTER_SIDE_PX = 8


def gutter_mask(profile: np.ndarray) -> np.ndarray:
    """Gutter columns of a column-mean V profile: dark (< GUTTER_V_MAX), or a valley that is
    at most half as bright as the brightest column within 8 px on each side. On keyframes the
    encoder bleeds the tiles into their 4 px gutters (V 0.02 -> 0.18 in the pilot, every 2 s),
    which a fixed threshold misses."""
    p = np.asarray(profile, dtype=float)
    n, k = len(p), GUTTER_SIDE_PX
    pad = np.r_[np.zeros(k), p, np.zeros(k)]
    win = np.lib.stride_tricks.sliding_window_view(pad, k)
    left = win[:n].max(axis=1)                      # columns x-k .. x-1
    right = win[k + 1:k + 1 + n].max(axis=1)        # columns x+1 .. x+k
    valley = (p <= GUTTER_VALLEY_MAX) & (p <= GUTTER_VALLEY_RATIO * np.minimum(left, right))
    return (p < config.GUTTER_V_MAX) | valley


def tiles_from_profile(profile: np.ndarray, min_w: int = FORUM_UI.min_tile_w) -> np.ndarray:
    """Tile x-spans from a column-mean V profile: spans between dark gutters (>= 2 px below
    GUTTER_V_MAX), the median width taken as the tile width. A span k tile-widths wide (two
    tiles whose 4 px gutter the encoder smeared: in the pilot this happened every 2 s, on the
    keyframes) is split into k tiles at the tile pitch; other widths are dropped."""
    gut = hands._runs(gutter_mask(profile))
    gut = gut[(gut[:, 1] - gut[:, 0]) >= config.GUTTER_MIN_PX]
    if len(gut) < 2:
        return np.zeros((0, 2), dtype=int)
    spans = np.c_[gut[:-1, 1], gut[1:, 0]]
    width = spans[:, 1] - spans[:, 0]
    wide = width >= min_w
    if not wide.any():
        return np.zeros((0, 2), dtype=int)
    med = float(np.median(width[wide]))
    gap = float(np.median(gut[1:-1, 1] - gut[1:-1, 0])) if len(gut) > 2 else 4.0
    out = []
    for (a, b), w in zip(spans, width, strict=True):
        k = max(1, round((w + gap) / (med + gap)))
        if abs(w - (k * med + (k - 1) * gap)) > config.TILE_WIDTH_TOL * med:
            continue
        step = (w + gap) / k
        out += [(round(a + i * step), round(a + i * step + (w - (k - 1) * gap) / k))
                for i in range(k)]
    return np.array(out, dtype=int).reshape(-1, 2)


def find_band(v: np.ndarray, *, x_max: int = BAND_X_MAX) -> Band | None:
    """The participant strip in one frame (V channel in [0, 1], top rows): the topmost run
    of bright rows whose dark gutters split it into >= 2 equal tiles of a tile's shape."""
    nondark = (v[:, :x_max] >= config.GUTTER_V_MAX).mean(axis=1)
    start = BAND_FROM_Y
    while start < len(nondark):
        rows = np.flatnonzero(nondark[start:] >= BAND_NONDARK_MIN)
        if not len(rows):
            return None
        y0 = start + int(rows[0])
        y1 = y0 + 1
        while y1 < len(nondark) and nondark[y1] >= BAND_END_NONDARK:
            y1 += 1
        h = y1 - y0
        if TILE_H_RANGE[0] <= h <= TILE_H_RANGE[1]:
            tiles = tiles_from_profile(v[y0:y1].mean(axis=0), FORUM_UI.min_tile_w)
            if len(tiles) >= 2:
                w = float(np.median(tiles[:, 1] - tiles[:, 0]))
                if TILE_ASPECT[0] <= h / w <= TILE_ASPECT[1]:
                    return Band(int(y0), int(y1), tiles)
        start = y1 + 1
    return None


def steady_band(band: Band | None, prev: Band | None, tol: int = 3) -> Band | None:
    """The previous frame's tiles when this frame only lost some of them: same band rows and
    every tile found here matches one of the previous tiles. A reflow (someone joins or
    leaves) moves or resizes tiles; a smeared gutter only drops tiles. Uses the previous
    frame only (causal)."""
    if band is None or prev is None or len(band.tiles) >= len(prev.tiles):
        return band
    if abs(band.y0 - prev.y0) > tol or abs(band.y1 - prev.y1) > tol:
        return band
    d0 = np.abs(band.tiles[:, None, 0] - prev.tiles[None, :, 0])
    d1 = np.abs(band.tiles[:, None, 1] - prev.tiles[None, :, 1])
    if ((d0 <= tol) & (d1 <= tol)).any(axis=1).all():
        return Band(band.y0, band.y1, prev.tiles)
    return band


def word_patch(white: np.ndarray) -> np.ndarray | None:
    """The name in one tile's label area (bool mask of white-text pixels), as a blurred
    WORD_H x WORD_W patch: the largest run of joined letters, scaled to WORD_H rows."""
    lab = white.astype(np.uint8)
    if int(lab.sum()) < 8:
        return None
    joined = cv2.dilate(lab, np.ones((1, 5), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(st[k, i]) for i in range(4))
    if h < 4 or w < 8:
        return None
    word = (lab[y:y + h, x:x + w] > 0) & (cc[y:y + h, x:x + w] == k)
    nw = max(1, round(w * WORD_H / h))
    r = cv2.resize(word.astype(np.float32), (nw, WORD_H), interpolation=cv2.INTER_AREA)
    out = np.zeros((WORD_H, WORD_W), np.float32)
    out[:, :min(WORD_W, nw)] = r[:, :WORD_W]
    return cv2.GaussianBlur(out, (0, 0), WORD_BLUR)


def label_mask(tile_bgr: np.ndarray) -> np.ndarray:
    """White-text mask of a tile's label area (bottom 42%, 3 px in from the sides)."""
    th = tile_bgr.shape[0]
    lab = tile_bgr[int(th * LABEL_FROM):th - 2, 3:tile_bgr.shape[1] - 3]
    hsv = cv2.cvtColor(np.ascontiguousarray(lab), cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < LABEL_S_MAX) & (hsv[..., 2] > LABEL_V_MIN)


def _ncc_rows(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = P.reshape(len(P), -1) - P.reshape(len(P), -1).mean(axis=1, keepdims=True)
    t = t.ravel() - t.mean()
    den = np.sqrt((P * P).sum(axis=1) * float((t * t).sum()))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (P @ t) / np.where(den > 0, den, 1.0), 0.0)


def tile_stats(strip: np.ndarray, band: Band, *, ui: hands.UiSpec = FORUM_UI) -> dict:
    """Per-tile statistics in the `hand_tiles` layout (frac_hand/tick at s1/s2, shape,
    white, border) plus each tile's word patch. `strip` = the band rows of the frame."""
    geom = hands.StripGeometry(found=True, strip_y=0, strip_h=band.h, tile_y0=0,
                               tile_h=band.h)
    st = hands.frame_stats(strip, geom, ui=ui, tiles=band.tiles)
    st["word"] = [word_patch(label_mask(strip[:, a:b])) for a, b in band.tiles]
    return st


# --------------------------------------------------------------------------- whose label


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def roster_names(klass: dict) -> dict[str, list[str]]:
    """user id -> name forms a label may show (preferred / first name, with the surname)."""
    from scribe_core.asr_deepgram import roster

    out: dict[str, list[str]] = {}
    for r in roster(klass):
        if not r["user_id"]:
            continue
        forms = {r["first"], r["preferred"], f"{r['preferred'] or r['first']} {r['last']}",
                 f"{r['first']} {r['last']}"}
        out[r["user_id"]] = [f for f in (_norm(x) for x in forms) if len(f) >= 2]
    return out


def _ocr_norm(text: str) -> list[str]:
    """Candidate spellings of an OCR line: each word and the words joined, lowercased, with
    the OCR_FIXES applied (both with and without them)."""
    out = []
    for w in re.split(r"\s+", str(text or "").lower()):
        fixed = w
        for a, b in OCR_FIXES:
            fixed = fixed.replace(a, b)
        out += [x for x in {_norm(w), _norm(fixed)} if len(x) >= 2]
    return out + ["".join(out)] if len(out) > 1 else out


def match_name(texts: str | Sequence[str], names: dict[str, list[str]]
               ) -> tuple[str | None, float]:
    """(user id, score) of the roster person whose name is closest to the OCR text(s), or
    None unless the score reaches NAME_MATCH_MIN and beats every other person's best score
    by NAME_MATCH_MARGIN. Each word is compared with each name form ("Brady K" -> brady)."""
    cands = [c for t in ([texts] if isinstance(texts, str) else texts) for c in _ocr_norm(t)]
    per_user = {u: max((difflib.SequenceMatcher(None, c, f).ratio() for f in forms
                        for c in cands), default=0.0) for u, forms in names.items()}
    if not per_user:
        return None, 0.0
    ranked = sorted(per_user.items(), key=lambda kv: -kv[1])
    uid, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    ok = best >= NAME_MATCH_MIN and best - second >= NAME_MATCH_MARGIN
    return (uid if ok else None), best


def _tesseract(img: np.ndarray, tesseract: str) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    p = subprocess.run([tesseract, "stdin", "stdout", "--psm", "7"], input=buf.tobytes(),
                       capture_output=True, timeout=30, check=False)
    return p.stdout.decode("utf-8", "replace").strip()


def ocr_label(white: np.ndarray, *, tesseract: str = "tesseract") -> list[str]:
    """Tesseract (one text line) on the white-text mask, black text on white, scaled so the
    label area is ~60 px high: once with a 2 px opening (drops specks), once without.
    Returns the raw texts, kept in memory only."""
    m = white.astype(np.uint8) * 255
    fx = max(2, round(60 / max(1, m.shape[0])))
    m = cv2.resize(m, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return [_tesseract(cv2.copyMakeBorder(255 - x, 12, 12, 12, 12, cv2.BORDER_CONSTANT,
                                          value=255), tesseract) for x in (opened, m)]


@dataclass
class LabelBook:
    """Name labels seen in one class: word patch + tile width -> user id (or None, unread).

    `identify` returns the user id of a label, reading it with OCR the first time a label is
    seen at a width and recognising it by NCC afterwards. An unread label is OCR'd again on
    up to OCR_RETRIES later frames (a different video frame behind the text reads
    differently). Names never leave this object; `save` keeps only patches and is-me flags.
    """

    names: dict[str, list[str]]
    me_id: str
    ocr: Callable[[np.ndarray], str | list[str]] = ocr_label
    strict: bool = False             # STRICT_ME_LABEL_COURSES: his labels need confirming
    widths: list[int] = field(default_factory=list)
    patches: list[np.ndarray] = field(default_factory=list)
    uids: list[str | None] = field(default_factory=list)
    tries: list[int] = field(default_factory=list)
    confirmed: list[bool] = field(default_factory=list)
    n_ocr: int = 0
    version: int = 0                 # bumped whenever a label is added or re-read

    def identify(self, width: int, patch: np.ndarray | None, white: np.ndarray) -> str | None:
        if patch is None:
            return None
        k = self._lookup(width, patch)
        if k is not None and self.confirmed[k] and (self.uids[k] is not None
                                                    or self.tries[k] >= OCR_RETRIES):
            return self.uids[k]
        if k is not None and self.tries[k] >= OCR_RETRIES:
            return None                                   # never confirmed as anyone's
        self.n_ocr += 1
        self.version += 1
        uid, score = match_name(self.ocr(white), self.names)
        if uid == self.me_id and score < ME_NAME_MIN:
            uid = None                                    # too far from his name to be his
        if k is None:
            ok = not (self.strict and uid == self.me_id and not self._like_me(patch))
            self.widths.append(int(width))
            self.patches.append(patch)
            self.uids.append(uid)
            self.tries.append(1)
            self.confirmed.append(ok)
            return uid if ok else None
        self.tries[k] += 1
        if not self.confirmed[k]:                         # a pending label of his
            if uid == self.me_id:
                self.confirmed[k] = True                  # read as his a second time
                return uid
            self.uids[k] = uid
            self.confirmed[k] = uid != self.me_id
            return uid if self.confirmed[k] else None
        self.uids[k] = uid
        return uid

    def _like_me(self, patch: np.ndarray) -> bool:
        k = [i for i, u in enumerate(self.uids) if u == self.me_id and self.confirmed[i]]
        return bool(k) and float(_ncc_rows(np.stack([self.patches[i] for i in k]),
                                           patch).max()) >= ME_CONFIRM_NCC

    def _lookup(self, width: int, patch: np.ndarray) -> int | None:
        idx = [i for i, w in enumerate(self.widths) if abs(w - width) <= SAME_WIDTH_PX]
        if not idx:
            return None
        s = _ncc_rows(np.stack([self.patches[i] for i in idx]), patch)
        j = int(np.argmax(s))
        need = ME_SAME_LABEL_NCC if self.uids[idx[j]] == self.me_id else SAME_LABEL_NCC
        return idx[j] if s[j] >= need else None

    def me_patches(self) -> tuple[np.ndarray, np.ndarray]:
        """(widths, patches) of the labels read as Brady's."""
        k = [i for i, u in enumerate(self.uids) if u == self.me_id and self.confirmed[i]]
        return (np.array([self.widths[i] for i in k], dtype=int),
                np.stack([self.patches[i] for i in k]) if k else
                np.zeros((0, WORD_H, WORD_W), np.float32))

    def summary(self) -> dict:
        n_me = sum(u == self.me_id for u in self.uids)
        return {"n_labels": len(self.uids), "n_read": sum(u is not None for u in self.uids),
                "n_me_labels": int(n_me), "n_ocr": int(self.n_ocr)}


def find_me(st: dict, widths: np.ndarray, me_widths: np.ndarray, me_patches: np.ndarray
            ) -> int:
    """Index of Brady's tile in one frame from his known label patches, or -1.

    A tile qualifies with NCC >= ME_LABEL_NCC against his labels read at the same width
    (±SAME_WIDTH_PX), or >= ME_LABEL_NCC_ANY at any width (a reaction animation tints the
    label; the strip resizes as people leave). The best qualifying tile must beat every other
    tile's best score by ME_LABEL_GAP, else the frame stays unread. Pilot (20 classes): with
    the margin, 0 wrong tiles in all classes; without it, up to 42 in one class (89838), where
    another label scores close to his."""
    if not len(me_widths):
        return -1
    any_w = np.full(len(widths), -1.0)
    ok = np.zeros(len(widths), bool)
    for i, (p, w) in enumerate(zip(st["word"], widths, strict=True)):
        if p is None:
            continue
        s = _ncc_rows(me_patches, p)
        any_w[i] = float(s.max())
        same = np.abs(me_widths - w) <= SAME_WIDTH_PX
        ok[i] = any_w[i] >= ME_LABEL_NCC_ANY or (same.any() and float(s[same].max()) >= ME_LABEL_NCC)
    if not ok.any():
        return -1
    k = int(np.argmax(np.where(ok, any_w, -2.0)))
    others = np.delete(any_w, k)
    return k if not len(others) or any_w[k] - others.max() >= ME_LABEL_GAP else -1


def carry_me(st: dict, band: Band, strip: np.ndarray, book: LabelBook,
             last: tuple[float, int, int] | None, t: float) -> int:
    """Brady's tile through the raise overlay, which can hide his name: the tile at his last
    identified x-span (±CARRY_PX, identified <= CARRY_S s earlier) if it shows the overlay and
    its label is not read as someone else (Forum reorders the strip, and a classmate's own
    overlay can land in his old slot). Earlier frames only (causal); -1 otherwise."""
    if last is None or t - last[0] > CARRY_S:
        return -1
    near = np.flatnonzero((np.abs(band.tiles[:, 0] - last[1]) <= CARRY_PX)
                          & (np.abs(band.tiles[:, 1] - last[2]) <= CARRY_PX))
    if len(near) != 1:
        return -1
    j = int(near[0])
    overlay = (st["frac_hand_s1"][j] >= OVERLAY_MIN) and (st["frac_hand_s2"][j] >= OVERLAY_MIN)
    if not overlay:
        return -1
    a, b = band.tiles[j]
    who = book.identify(int(b - a), st["word"][j], label_mask(strip[:, a:b]))
    return j if who in (None, book.me_id) else -1


def me_tile(st: dict, band: Band, strip: np.ndarray, book: LabelBook, known: dict) -> int:
    """Brady's tile in one frame, or -1. His labels read so far are tried first (no OCR);
    only when none matches are the tiles' labels identified (OCR for new ones). Two tiles
    read as Brady make the frame ambiguous (-1). `known` caches his patches per book
    version."""
    if known.get("version") != book.version:
        known["widths"], known["patches"] = book.me_patches()
        known["version"] = book.version
    widths = band.tiles[:, 1] - band.tiles[:, 0]
    if len(known["widths"]):
        k = find_me(st, widths, known["widths"], known["patches"])
        if k >= 0:
            return k
    ids = [book.identify(int(w), p, label_mask(strip[:, a:b]))
           for (a, b), w, p in zip(band.tiles, widths, st["word"], strict=True)]
    mine = [i for i, u in enumerate(ids) if u == book.me_id]
    return mine[0] if len(mine) == 1 else -1


# Forum's raise animation (seen 13 Oct, class 79900 at 553.54 s, 24 fps): on the key press the
# whole tile turns yellow-green with a big white hand for ~1.7 s, covering the name label, and
# only then shrinks to the corner icon. Its first frame is the raise's true start.
OVERLAY_MIN = 0.9                # both corner squares this full of hand hue: the overlay
OVERLAY_START_TILE = 0.03        # retime: a growing overlay already covers this much of the tile
CARRY_S = 3.0                    # the overlay tile is Brady's if his label was read there this recently
CARRY_PX = 3
RETIME_WINDOW_S = 4.5            # re-timing read: [onset - 1 s, onset + 3.5 s]


def overlay_mask(ht: pd.DataFrame) -> np.ndarray:
    """Rows of `hand_tiles` showing the raise overlay (any tile)."""
    return ((ht["frac_hand_s1"].fillna(0) >= OVERLAY_MIN)
            & (ht["frac_hand_s2"].fillna(0) >= OVERLAY_MIN)).to_numpy()


def mark_overlays(ht: pd.DataFrame) -> pd.DataFrame:
    """Add `overlay` (the raise animation on any tile); `hands.track` counts it as a hand
    that passes the shape test. Brady's identity through it is carried during the scan
    (`carry_me`)."""
    ht = ht.copy()
    ht["overlay"] = overlay_mask(ht)
    return ht


# A live view of the scan (Brady, 14 Oct: "see the video of the tile as it gets scraped"):
# every minute of video, the grey view of the strip goes to {logs}/live/scan_{cid}.png.
LIVE_EVERY_FRAMES = 120          # 60 s at 2 fps
LIVE = os.environ.get("ONCUE_LIVE") == "1"      # pod/run.sh turns it on


def live_snapshot(cid: int, t: float, strip_img: np.ndarray, band: Band, k: int, st: dict,
                  det: hands.Detector | None = None) -> None:
    """Write the strip as the scan sees it (grey view: only teal and green in colour, every
    tile but Brady's pixelated) with a caption: class, video time, tile, hand state."""
    det = det or FORUM_DETECTOR
    try:
        view = grey_view(strip_img, band, k)
        hand = "-"
        if k >= 0:
            a, b = band.tiles[k]
            cv2.rectangle(view, (int(a) - 2, 0), (int(b) + 1, band.h - 1), (40, 40, 40), 2)
            up = (st["frac_hand_s1"][k] >= det.theta_on and hands.shape_ok(
                st["shape_s1"][k:k + 1], st["white_s1"][k:k + 1], det)[0]) or (
                st["frac_hand_s1"][k] >= OVERLAY_MIN and st["frac_hand_s2"][k] >= OVERLAY_MIN)
            hand = "HAND UP" if up else "hand down"
        view = view[:, :int(band.tiles[-1, 1]) + 4]
        w = 1400
        view = cv2.resize(view, (w, max(1, round(view.shape[0] * w / view.shape[1]))),
                          interpolation=cv2.INTER_NEAREST)
        head = np.full((34, w, 3), 255, np.uint8)
        cap = (f"class {cid}   video {int(t // 60):3d}:{int(t % 60):02d}   "
               f"my tile {'found' if k >= 0 else 'NOT found'}   {hand}")
        cv2.putText(head, cap, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 120, 0) if hand == "HAND UP" else (40, 40, 40), 2, cv2.LINE_AA)
        out = config.LOGS_DIR / "live" / f"scan_{cid}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.png")
        cv2.imwrite(str(tmp), np.vstack([head, view]))
        tmp.replace(out)
    except Exception as exc:  # noqa: BLE001 - a preview must never stop the scan
        log.debug("live snapshot failed: %s", exc)


# --------------------------------------------------------------------------- one class


def _anchor_windows(voice: pd.DataFrame | None) -> list[tuple[float, float]]:
    """[start + 1 s, end - 1 s] of Brady's isolated, word-bearing main-room voice events
    (`hands.anchor_times` rules, every event)."""
    if voice is None or not len(voice):
        return []
    ev = hands.anchor_times(voice, max_events=10 ** 6)
    return [(min(ts), max(ts)) for ts in ev if ts]


def ring_check_read(ht: pd.DataFrame, voice: pd.DataFrame | None) -> tuple[float, int]:
    """The ring check over the frames where Brady's tile is read (QA; Brady, 14 Oct: breakouts
    and unread stretches must not drop a class, a wrong tile must). Inside his isolated speech,
    of the frames with exactly one bordered tile and his tile identified, the share where the
    bordered tile is his. Returns (share, n frames); share is NaN when there are none."""
    windows = _anchor_windows(voice)
    if not windows or ht is None or not len(ht):
        return float("nan"), 0
    wa = np.array([a for a, _ in windows])
    wb = np.array([b for _, b in windows])
    h = ht[ht["tile_idx"] >= 0]
    t = h["pts_s"].to_numpy(float)
    j = np.searchsorted(wa, t, side="right") - 1
    h = h[(j >= 0) & (t <= wb[np.clip(j, 0, None)])]
    if not len(h):
        return float("nan"), 0
    bordered = np.nan_to_num(h["border"].to_numpy(float)) >= FORUM_DETECTOR.border_min
    mine = h["ncc_me"].to_numpy(float) >= 1
    g = pd.DataFrame({"t": h["pts_s"].to_numpy(float), "b": bordered, "bm": bordered & mine,
                      "m": mine}).groupby("t")[["b", "bm", "m"]].sum()
    use = (g["b"] == 1) & (g["m"] >= 1)
    n = int(use.sum())
    return (float(g.loc[use, "bm"].sum()) / n if n else float("nan")), n


def cli_ringcheck(args) -> int:
    """`oncue ringcheck`: re-score the ring check of scanned classes over the frames where
    Brady's tile is read (`ring_check_read`) and update hand_qc (track_valid_share,
    tile_status, template_validated) in place. Nothing else changes: the raises and their
    re-timing stay. Then run `oncue qa`."""
    classes = hands._selected_classes(args)
    qc = hands._read_table("hand_qc")
    voice = hands._read_table("voice")
    if qc is None:
        raise SystemExit("oncue ringcheck: run `oncue track` first")
    qc = qc.copy()
    changed = 0
    for cid in (int(c) for c in classes["class_id"]):
        row = qc["class_id"] == cid
        p = hands.hand_tiles_path(cid)
        if not row.any() or not p.exists() or p.stat().st_size == 0:
            continue
        ht = pd.read_parquet(p, columns=["pts_s", "tile_idx", "border", "ncc_me"])
        share, _n = ring_check_read(ht, hands._for_class(voice, cid))
        r = qc.loc[row].iloc[0]
        n_anchor = int(r["n_anchor_events"]) if pd.notna(r["n_anchor_events"]) else 0
        geom = hands.StripGeometry(found=bool(r["strip_found"]) if pd.notna(r["strip_found"])
                                   else False)
        cov = float(r["coverage"]) if pd.notna(r["coverage"]) else 0.0
        status, validated = hands.tile_status(geom, cov, n_anchor, share)
        changed += int(status != r["tile_status"])
        qc.loc[row, "track_valid_share"] = share
        qc.loc[row, "tile_status"] = status
        qc.loc[row, "template_validated"] = validated
    hands._upsert("hand_qc", qc)
    print(f"oncue ringcheck: {int(qc['class_id'].isin(classes['class_id']).sum())} classes "
          f"re-scored over read frames, {changed} changed tile status", flush=True)
    return 0


def scan_class(cid: int, frames: Iterable[tuple[float, np.ndarray]], book: LabelBook, *,
               voice: pd.DataFrame | None = None, ui: hands.UiSpec = FORUM_UI
               ) -> tuple[pd.DataFrame, dict]:
    """Every frame -> `hand_tiles` rows (ncc_me = 1 on Brady's tile) and the pass-1 record."""
    cols: dict[str, list] = {c: [] for c in ("pts_s", "tile_idx", "x0", "x1", "y0", "y1",
                                              *hands._TILE_FLOAT_COLS)}
    blank = hands.pack_glyph(np.zeros((config.GLYPH_H, config.GLYPH_W), bool))
    windows = _anchor_windows(voice)
    wa = np.array([a for a, _ in windows]) if windows else np.zeros(0)
    wb = np.array([b for _, b in windows]) if windows else np.zeros(0)
    anchor_hits, anchor_frames, anchor_events = 0, 0, set()
    bands, n_frames = [], 0
    known: dict = {}
    prev = None
    last_me = None                         # (t, x0, x1) of Brady's last identified tile
    for t, frame in frames:
        n_frames += 1
        v = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2HSV)[..., 2] / 255.0
        band = prev = steady_band(find_band(v), prev)
        if band is None:
            _placeholder(cols, t)
            continue
        strip = np.ascontiguousarray(frame[band.y0:band.y1])
        st = tile_stats(strip, band, ui=ui)
        widths = band.tiles[:, 1] - band.tiles[:, 0]
        k = me_tile(st, band, strip, book, known)
        if k < 0:
            k = carry_me(st, band, strip, book, last_me, float(t))
        if k >= 0:
            last_me = (float(t), int(band.tiles[k, 0]), int(band.tiles[k, 1]))
        me = np.arange(len(band.tiles)) == k
        st["ncc_me"] = me.astype(np.float32)
        bands.append((band.y0, band.h, len(band.tiles), float(np.median(widths))))
        if LIVE and n_frames % LIVE_EVERY_FRAMES == 1:
            live_snapshot(cid, float(t), strip, band, int(k), st)
        n = len(band.tiles)
        cols["pts_s"].extend([float(t)] * n)
        cols["tile_idx"].extend(range(n))
        cols["x0"].extend(band.tiles[:, 0].tolist())
        cols["x1"].extend(band.tiles[:, 1].tolist())
        cols["y0"].extend([band.y0] * n)
        cols["y1"].extend([band.y1] * n)
        for c in hands._TILE_FLOAT_COLS:
            cols[c].extend(np.asarray(st[c], dtype=np.float32).tolist())
        # anchor check: one bordered tile inside Brady's isolated speech must be his
        j = np.searchsorted(wa, t, side="right") - 1
        if j >= 0 and t <= wb[j]:
            bordered = np.flatnonzero(np.nan_to_num(st["border"]) >= FORUM_DETECTOR.border_min)
            if len(bordered) == 1:
                anchor_frames += 1
                anchor_hits += bool(me[int(bordered[0])])
                anchor_events.add(int(j))
    n_rows = len(cols["pts_s"])
    df = pd.DataFrame({"class_id": np.full(n_rows, int(cid), dtype=np.int64),
                       "pts_s": np.asarray(cols["pts_s"], dtype=float),
                       "tile_idx": np.asarray(cols["tile_idx"], dtype=np.int64),
                       "x0": np.asarray(cols["x0"], dtype=np.int64),
                       "x1": np.asarray(cols["x1"], dtype=np.int64),
                       "y0": np.asarray(cols["y0"], dtype=np.int64),
                       "y1": np.asarray(cols["y1"], dtype=np.int64)})
    for c in hands._TILE_FLOAT_COLS:
        df[c] = np.asarray(cols[c], dtype=np.float32)
    df["glyph_bits"] = pd.Series([blank] * n_rows, dtype=object)
    df = mark_overlays(df)
    b = np.array(bands) if bands else np.zeros((0, 4))
    geom = hands.StripGeometry(
        found=bool(len(b)), strip_y=int(np.median(b[:, 0])) if len(b) else 0,
        strip_h=int(np.median(b[:, 1])) if len(b) else 0, tile_y0=0,
        tile_h=int(np.median(b[:, 1])) if len(b) else 0,
        n_tiles=int(np.median(b[:, 2])) if len(b) else 0,
        pitch=float(np.median(b[:, 3])) if len(b) else 0.0, n_frames_found=len(b),
        reason="ok" if len(b) else "no_band")
    info = {"class_id": int(cid), "geometry": geom.to_dict(), "anchors": [],
            "n_anchor_events": len(anchor_events), "scanner": "strip",
            "n_anchor_frames": anchor_frames,
            "valid_share": (anchor_hits / anchor_frames) if anchor_frames else None,
            "n_frames": n_frames, "band_share": (len(b) / n_frames) if n_frames else 0.0,
            "n_me_frames": int(df.loc[df["ncc_me"] > 0, "pts_s"].nunique()),
            **book.summary()}
    return schemas.validate(df, "hand_tiles"), info


def _placeholder(cols: dict, t: float) -> None:
    cols["pts_s"].append(float(t))
    for c in ("tile_idx", "x0", "x1", "y0", "y1"):
        cols[c].append(-1)
    for c in hands._TILE_FLOAT_COLS:
        cols[c].append(np.nan)


# --------------------------------------------------------------------------- re-timing


def retime_onset(frames: Sequence[tuple[float, np.ndarray]], t0: float, *, base: float,
                 me_widths: np.ndarray, me_patches: np.ndarray,
                 span: tuple[int, int] | None = None, ui: hands.UiSpec = FORUM_UI,
                 det: hands.Detector = FORUM_DETECTOR, seek_tol_s: float = 0.05
                 ) -> float | None:
    """The first full-rate frame on which Brady's tile shows a hand, followed by >=
    RETIME_MIN_POS_FRAMES positive frames; None when the seek missed or no such frame exists.
    His tile is found on each frame by its label; while the raise overlay hides the label,
    the tile at `span` (his x-span at the coarse onset) is used. A frame is positive when the
    corner reaches θ_on and either the icon passes the shape test or the overlay is growing
    over the tile (>= OVERLAY_START_TILE of it in hand hue), so the onset is the overlay's
    first frame. `frames` are top-of-frame crops."""
    if not frames or abs(float(frames[0][0]) - t0) > seek_tol_s:
        return None
    pos = []
    prev = None
    for _, frame in frames:
        v = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2HSV)[..., 2] / 255.0
        band = prev = steady_band(find_band(v), prev)
        if band is None:
            pos.append(False)
            continue
        strip = np.ascontiguousarray(frame[band.y0:band.y1])
        st = tile_stats(strip, band, ui=ui)
        k = find_me(st, band.tiles[:, 1] - band.tiles[:, 0], me_widths, me_patches)
        if k < 0 and span is not None:
            near = np.flatnonzero((np.abs(band.tiles[:, 0] - span[0]) <= CARRY_PX)
                                  & (np.abs(band.tiles[:, 1] - span[1]) <= CARRY_PX))
            k = int(near[0]) if len(near) == 1 else -1
        if k < 0:
            pos.append(False)
            continue
        frac = float(st[f"frac_hand_{det.roi}"][k]) - base
        ok = hands.shape_ok(st[f"shape_{det.roi}"][k:k + 1], st[f"white_{det.roi}"][k:k + 1],
                            det)[0]
        a, b = band.tiles[k]
        tile_hue = float(hands.hsv_masks(strip[:, a:b], ui)["hand"].mean())
        pos.append(bool(frac >= det.theta_on and (ok or tile_hue >= OVERLAY_START_TILE)))
    need = config.RETIME_MIN_POS_FRAMES
    for i in range(len(pos) - need):
        if pos[i] and all(pos[i + 1:i + 1 + need]):
            return float(frames[i][0])
    return None


def onset_span(ht: pd.DataFrame, onset_s: float, within_s: float = 4.0
               ) -> tuple[int, int] | None:
    """Brady's tile x-span at the first sample of an episode (from `hand_tiles`)."""
    r = ht[(ht["ncc_me"] > 0) & (ht["pts_s"] >= onset_s - 1e-6)
           & (ht["pts_s"] <= onset_s + within_s)].sort_values("pts_s")
    return None if not len(r) else (int(r["x0"].iloc[0]), int(r["x1"].iloc[0]))


def retime_class(eps: pd.DataFrame, reader: Callable[[float, float], Sequence], *,
                 base: float, me_widths: np.ndarray, me_patches: np.ndarray,
                 hand_tiles: pd.DataFrame | None = None,
                 det: hands.Detector = FORUM_DETECTOR) -> pd.DataFrame:
    """`hands.retime` for the real strip: onset_retimed_s / retime_failed per episode."""
    out = eps.copy()
    for i, ep in out.iterrows():
        if bool(ep["left_censored"]):
            continue
        # [onset - 1 s, onset + 3.5 s]: covers an onset reached through a hidden-label gap
        t0 = float(ep["onset_s"]) - config.RETIME_PRE_S
        try:
            frames = list(reader(t0, RETIME_WINDOW_S))
        except Exception as exc:                       # noqa: BLE001 - any seek failure
            log.warning("retime: read failed for episode %s (%s)", ep["ep_id"],
                        type(exc).__name__)
            frames = []
        span = None if hand_tiles is None else onset_span(hand_tiles, float(ep["onset_s"]))
        res = retime_onset(frames, t0, base=base, me_widths=me_widths, me_patches=me_patches,
                           span=span, det=det)
        out.at[i, "onset_retimed_s"] = float(ep["onset_s"]) if res is None else res
        out.at[i, "retime_failed"] = res is None
    out["retime_failed"] = out["retime_failed"].astype("boolean")
    return out


# --------------------------------------------------------------------------- audit clips

AUDIT_GREY = 205                 # the light grey everything except teal and green turns into
AUDIT_PIXEL_BLOCKS = 6           # other people's tiles: 6 x 6 blocks before greying
AUDIT_PRE_S, AUDIT_POST_S = 3.0, 3.0


def grey_view(band_bgr: np.ndarray, band: Band, me_k: int, *, ui: hands.UiSpec = FORUM_UI
              ) -> np.ndarray:
    """Audit rendering of the strip: every tile except Brady's pixelated, then every pixel
    turned light grey (a faint copy of its brightness) except the teal speaker ring and the
    yellow-green hand, which keep their colour (Brady, 13 Oct: "only the teal and green")."""
    # Pixelate the WHOLE band in tile-sized blocks, then paste back only Brady's tile: a tile
    # the gutter split missed must never show unpixelated (a face or a name).
    img = band_bgr.copy()
    pitch = int(np.median(band.tiles[:, 1] - band.tiles[:, 0])) if len(band.tiles) else 64
    for a in range(0, img.shape[1], max(8, pitch)):
        img[:, a:a + pitch] = hands._pixelate(img[:, a:a + pitch], AUDIT_PIXEL_BLOCKS)
    if 0 <= me_k < len(band.tiles):
        a, b = band.tiles[me_k]
        img[:, a:b] = band_bgr[:, a:b]
    m = hands.hsv_masks(img, ui)
    keep = m["hand"] | m["border"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dull = (AUDIT_GREY + (gray - 128.0) * 0.25).clip(0, 255).astype(np.uint8)
    out = cv2.cvtColor(dull, cv2.COLOR_GRAY2BGR)
    out[keep] = img[keep]
    return out


def audit_frames(frames: Sequence[tuple[float, np.ndarray]], onset_s: float, me_widths,
                 me_patches, *, span: tuple[int, int] | None = None, width_out: int = 1280
                 ) -> list[np.ndarray]:
    """Rendered audit frames (`grey_view` of the strip, cropped to the tiles around Brady's,
    a time header, a red bar from the onset on)."""
    out = []
    prev = None
    for t, frame in frames:
        v = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2HSV)[..., 2] / 255.0
        band = prev = steady_band(find_band(v), prev)
        if band is None:
            continue
        strip_img = np.ascontiguousarray(frame[band.y0:band.y1])
        st = tile_stats(strip_img, band)
        k = find_me(st, band.tiles[:, 1] - band.tiles[:, 0], me_widths, me_patches)
        if k < 0 and span is not None:                # the overlay hides his label
            near = np.flatnonzero((np.abs(band.tiles[:, 0] - span[0]) <= CARRY_PX)
                                  & (np.abs(band.tiles[:, 1] - span[1]) <= CARRY_PX))
            k = int(near[0]) if len(near) == 1 else -1
        view = grey_view(strip_img, band, k)
        if k >= 0:                                    # mark Brady's tile with a thin frame
            a, b = band.tiles[k]
            cv2.rectangle(view, (int(a) - 2, 0), (int(b) + 1, band.h - 1), (40, 40, 40), 1)
        x1 = int(band.tiles[-1, 1]) + 4
        view = view[:, :x1]
        scale = width_out / view.shape[1]
        view = cv2.resize(view, (width_out, max(1, round(view.shape[0] * scale))),
                          interpolation=cv2.INTER_NEAREST)
        head = np.full((26, width_out, 3), 255, np.uint8)
        after = t >= onset_s - 1e-6
        txt = f"t = {t:9.3f} s   onset {onset_s:9.3f} s   {'AFTER' if after else 'before'}"
        cv2.putText(head, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 200) if after else (60, 60, 60), 1, cv2.LINE_AA)
        if after:
            head[-4:] = (0, 0, 220)
        out.append(np.vstack([head, view]))
    return out


def write_clip(path: Path, frames: Sequence[np.ndarray], fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
    vw.release()
    return path


def contact_sheet(frames: Sequence[np.ndarray], path: Path, *, cols: int = 1) -> Path:
    """Frames stacked into one PNG (one per row by default)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = max(f.shape[1] for f in frames)
    rows = [np.pad(f, ((0, 2), (0, w - f.shape[1]), (0, 0)), constant_values=255) for f in frames]
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def onset_audit(cid: int, eps: pd.DataFrame, reader: Callable[[float, float], Sequence], *,
                me_widths, me_patches, hand_tiles: pd.DataFrame | None = None,
                fps: float = 24.0) -> list[dict]:
    """A clip (onset - 3 s .. onset + 3 s, grey view) and a contact sheet of the frames from
    0.5 s before to 0.5 s after every bid onset, under audit/{cid}/. Returns one row per
    episode with the paths."""
    rows = []
    for ep in eps.itertuples():
        onset = float(ep.onset_retimed_s) if pd.notna(ep.onset_retimed_s) else float(ep.onset_s)
        try:
            raw = list(reader(onset - AUDIT_PRE_S, AUDIT_PRE_S + AUDIT_POST_S))
        except Exception as exc:                       # noqa: BLE001 - audit is best effort
            log.warning("audit read failed for class %s episode %s (%s)", cid, ep.ep_id,
                        type(exc).__name__)
            continue
        span = None if hand_tiles is None else onset_span(hand_tiles, float(ep.onset_s))
        frames = audit_frames(raw, onset, me_widths, me_patches, span=span)
        if not frames:
            continue
        times = [t for t, _ in raw][:len(frames)]
        stem = f"raise{int(ep.ep_id):02d}_{onset:07.1f}s".replace(".", "p")    # no dots: not a suffix
        base = config.AUDIT_DIR / str(int(cid)) / stem
        clip = write_clip(base.with_name(stem + ".mp4"), frames, fps)
        near = [f for t, f in zip(times, frames, strict=False) if abs(t - onset) <= 0.5]
        sheet = contact_sheet(near, base.with_name(base.name + "_frames.png")) if near else None
        rows.append({"class_id": int(cid), "ep_id": int(ep.ep_id), "onset_s": onset,
                     "clip": str(clip), "sheet": None if sheet is None else str(sheet)})
    return rows


# --------------------------------------------------------------------------- files


def labels_path(cid) -> Path:
    return config.CACHE_DIR / "hands" / f"{int(cid)}_me_labels.npz"


def save_me_labels(cid, book: LabelBook) -> Path:
    """Brady's own label patches (no names, nobody else's labels) for `retime`."""
    w, p = book.me_patches()
    path = labels_path(cid)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, widths=w, patches=p)
    return path


def load_me_labels(cid) -> tuple[np.ndarray, np.ndarray] | None:
    path = labels_path(cid)
    if not path.exists():
        return None
    z = np.load(path)
    return z["widths"], z["patches"]


def save_pass1(info: dict) -> Path:
    path = config.CACHE_DIR / "hands" / f"{info['class_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info))
    return path


def pass1_info(cid) -> dict | None:
    path = config.CACHE_DIR / "hands" / f"{int(cid)}.json"
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- CLI


def _one_thread() -> None:
    """Worker processes run one thread each (the parallelism is the process pool). Without
    this, OpenCV, BLAS and Tesseract each start a thread per visible core: in a container that
    is the whole host (96), and 16 workers buried 40 CPUs under a load of 200."""
    cv2.setNumThreads(1)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_THREAD_LIMIT"):
        os.environ.setdefault(k, "1")


def _scan_job(cid: int) -> tuple[int, dict | None]:
    """Scan one class (run in its own process: the per-frame work is Python and holds the
    GIL, so threads would share one core). The work dir arrives through ONCUE_WORK_DIR."""
    from . import events

    _one_thread()
    _pick_gpu(cid)
    try:
        voice_all = hands._read_table("voice")
        from .transcribe import course_code_of

        klass = events.read_raw(cid, "class") or {}
        book = LabelBook(roster_names(klass), str(events.load_me_id()),
                         strict=course_code_of({"class": klass}) in STRICT_ME_LABEL_COURSES)
        src = hands._sources(cid, hands._forum_client())
        m = src["media"]
        reader = hands.FfmpegStripReader(
            src["url_fn"], width=int(m["width"]), strip_y=0,
            strip_h=min(SCAN_ROWS, int(m["height"])), fps=config.F_SCAN,
            a_start=float(m.get("a_start") or 0.0), fmt_start=m.get("fmt_start"),
            hwaccel=config.HWACCEL)            # software unless ONCUE_HWACCEL=cuda (the pod's GPUs);
                                               # on the Mac software beats VideoToolbox, 33x vs 13x
        ht, info = scan_class(cid, reader, book, voice=hands._for_class(voice_all, cid))
        info["media"] = {k: m.get(k) for k in ("a_start", "v_start", "fmt_start", "width",
                                               "height", "duration")}
        out = hands.hand_tiles_path(cid)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        ht.to_parquet(tmp, index=False)
        tmp.replace(out)                       # a crashed scan never leaves a partial file
        save_me_labels(cid, book)
        save_pass1(info)
        return cid, info
    except Exception as exc:  # noqa: BLE001 - one bad class must not stop the batch
        log.error("scan class %s failed: %s", cid, hands.redact(f"{type(exc).__name__}: {exc}"))
        return cid, None


def cli_scan(args) -> int:
    """`oncue scan` on real recordings: every selected class without hand_tiles is streamed
    once at F_SCAN (top SCAN_ROWS rows only, frames never stored), `--workers` processes."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    classes = hands._selected_classes(args)
    todo = [int(c) for c in classes["class_id"] if not hands.hand_tiles_path(c).exists()]
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    results = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for cid, info in pool.map(_scan_job, todo):
            results.append((cid, info))
            if info is not None:
                print(f"scan {cid}: {info['n_frames']} frames, strip {info['band_share']:.0%}, "
                      f"my tile on {info['n_me_frames']}, ring check {info['valid_share']} "
                      f"({info['n_anchor_frames']} frames), {info['n_ocr']} OCR reads",
                      flush=True)
    bad = [c for c, info in results if info is None]
    print(f"oncue scan: {len(results) - len(bad)} classes scanned, {len(bad)} failed, "
          f"{len(classes) - len(todo)} already done", flush=True)
    return 0 if not bad else 1


def _pick_gpu(cid: int) -> None:
    """With ONCUE_HWACCEL=cuda and ONCUE_GPUS=n, this job's ffmpeg decodes on the next GPU in turn
    (a counter shared by every worker on the machine, so the streams spread evenly)."""
    if config.HWACCEL == "cuda" and config.GPUS > 1:
        import fcntl

        path = config.LOGS_DIR / ".gpu_turn"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            n = int(f.read().strip() or 0)
            f.seek(0)
            f.truncate()
            f.write(str(n + 1))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(n % config.GPUS)


def _clip_reader(cid: int, client) -> hands.FfmpegClipReader:
    src = hands._sources(cid, client)
    m = src["media"]
    _pick_gpu(cid)
    return hands.FfmpegClipReader(
        src["url_fn"], width=int(m["width"]), height=int(m["height"]),
        crop=(0, min(SCAN_ROWS, int(m["height"]))), a_start=float(m.get("a_start") or 0.0),
        fmt_start=m.get("fmt_start"), hwaccel=config.HWACCEL)


def _retime_job(job: tuple[int, pd.DataFrame, float]) -> pd.DataFrame | None:
    """Re-time one class's onsets (own process, own Forum client)."""
    cid, eps, base = job
    _one_thread()
    try:
        labels = load_me_labels(cid)
        htp = hands.hand_tiles_path(cid)
        return retime_class(eps, _clip_reader(cid, hands._forum_client()), base=base,
                            me_widths=labels[0], me_patches=labels[1],
                            hand_tiles=pd.read_parquet(htp) if htp.exists() else None)
    except Exception as exc:  # noqa: BLE001 - one bad class must not stop the batch
        log.error("retime class %s failed: %s", cid, hands.redact(f"{type(exc).__name__}: {exc}"))
        return None


def _pool(workers: int):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=max(1, workers),
                               mp_context=multiprocessing.get_context("spawn"))


def cli_retime(args) -> int:
    """`oncue retime` on real recordings: full-frame-rate re-read around every onset,
    `--workers` classes at a time."""
    classes = hands._selected_classes(args)
    eps_all = hands._read_table("hand_episodes")
    if eps_all is None:
        raise SystemExit("oncue retime: run `oncue episodes` first")
    qc = hands._read_table("hand_qc")
    jobs = []
    for cid in (int(c) for c in classes["class_id"]):
        eps, labels = hands._for_class(eps_all, cid), load_me_labels(cid)
        if eps is None or not len(eps) or labels is None or not len(labels[0]):
            continue
        base = 0.0
        if qc is not None and (qc["class_id"] == cid).any():
            base = float(qc.loc[qc["class_id"] == cid, "down_baseline"].iloc[0])
        jobs.append((cid, eps, base))
    with _pool(int(getattr(args, "workers", 1) or 1)) as pool:
        out = [r for r in pool.map(_retime_job, jobs) if r is not None]
    if out:
        eps = pd.concat(out, ignore_index=True)
        hands._upsert("hand_episodes", eps)
        hands.update_retime_qc(eps)
    print(f"oncue retime: {len(out)} classes re-timed, {len(jobs) - len(out)} failed",
          flush=True)
    return 0 if len(out) == len(jobs) else 1


def _audit_job(job: tuple[int, pd.DataFrame]) -> int:
    """Clips and frame sheets for one class's raises (own process, own Forum client)."""
    cid, eps = job
    _one_thread()
    try:
        labels = load_me_labels(cid)
        htp = hands.hand_tiles_path(cid)
        rows = onset_audit(cid, eps, _clip_reader(cid, hands._forum_client()),
                           me_widths=labels[0], me_patches=labels[1],
                           hand_tiles=pd.read_parquet(htp) if htp.exists() else None)
        if rows:
            pd.DataFrame(rows).to_csv(config.AUDIT_DIR / str(cid) / "raises.csv", index=False)
        return len(rows)
    except Exception as exc:  # noqa: BLE001 - audit is best effort
        log.error("audit class %s failed: %s", cid, hands.redact(f"{type(exc).__name__}: {exc}"))
        return 0


def cli_audit(args) -> int:
    """`oncue audit`: a grey-view clip and frame sheet around every bid onset (audit/{cid}/),
    plus audit/{cid}/raises.csv, `--workers` classes at a time. Local only (audit/ is
    gitignored)."""
    classes = hands._selected_classes(args)
    eps_all = hands._read_table("hand_episodes")
    if eps_all is None:
        raise SystemExit("oncue audit: run `oncue episodes` (and `retime`) first")
    jobs = []
    for cid in (int(c) for c in classes["class_id"]):
        eps, labels = hands._for_class(eps_all, cid), load_me_labels(cid)
        if eps is None or labels is None or not len(labels[0]):
            continue
        eps = eps[(eps["ep_type"] == config.EP_BID) & ~eps["left_censored"].astype(bool)]
        if len(eps):
            jobs.append((cid, eps))
    with _pool(int(getattr(args, "workers", 1) or 1)) as pool:
        n = sum(pool.map(_audit_job, jobs))
    print(f"oncue audit: {n} raise clips under {config.AUDIT_DIR}", flush=True)
    return 0
