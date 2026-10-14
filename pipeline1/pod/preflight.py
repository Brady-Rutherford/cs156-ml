"""Check a pod (or this laptop) before the full ingestion: `python pod/preflight.py [CLASS_ID]`.

Run from `pipeline1/`. Every check prints PASS or FAIL; the exit code is 1 if any fails.
Nothing is transcribed and nothing is written: it reads one class's recording for 20 s.
Prints no secrets and no names.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> object:
    try:
        ok, detail, value = fn()
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        ok, detail, value = False, f"{type(exc).__name__}: {str(exc)[:200]}", None
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    return value


def main() -> int:
    from oncue import config

    config.load_env()

    def tools():
        missing = [t for t in ("ffmpeg", "ffprobe", "tesseract") if not shutil.which(t)]
        v = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        first = (v.stdout or v.stderr).splitlines()[0] if not missing else ""
        return not missing, f"missing {missing}" if missing else first, None
    check("ffmpeg, ffprobe, tesseract", tools)

    def secrets():
        need = ("FORUM_COOKIE", "DEEPGRAM_API_KEY", "ONCUE_ID_SALT")
        absent = [k for k in need if not os.environ.get(k)]
        return not absent, f"missing {absent}" if absent else "all set (values not shown)", None
    check("secrets in pipeline1/.env", secrets)

    def data():
        from oncue.splits import split_ids

        n = len(split_ids())
        raw = sum(1 for _ in config.RAW_DIR.glob("*/class.json"))
        return n == 477 and raw >= n, f"{n} classes in the split, {raw} synced", None
    check("Forum sync and split", data)

    def forum():
        from scribe_core.forum import ForumClient

        me = ForumClient().me()
        return bool(me.get("id")), f"cookie works (role {me.get('role')})", None
    check("Forum cookie from this machine", forum)

    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 104899

    def video():
        from oncue import hands

        src = hands._sources(cid, hands._forum_client())
        m = src["media"]
        return (m.get("width") or 0) >= 1280, f"class {cid}: {m['width']}x{m['height']}, " \
            f"{m['duration']:.0f} s, audio start {m['a_start']}", src
    src = check("signed recording URL + ffprobe", video)

    def strip_frames():
        import cv2

        from oncue import hands, strip

        m = src["media"]
        reader = hands.FfmpegStripReader(src["url_fn"], width=int(m["width"]), strip_y=0,
                                         strip_h=strip.SCAN_ROWS, fps=2.0, start_s=900,
                                         end_s=920, hwaccel=None)
        t0, n, found = time.time(), 0, 0
        for _, f in reader:
            n += 1
            v = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 2] / 255.0
            found += strip.find_band(v) is not None
        return n >= 30 and found >= n // 2, \
            f"{n} frames in {time.time() - t0:.1f} s, strip found on {found}", None
    if src is not None:
        check("stream 20 s and find the strip", strip_frames)

    def ocr():
        import numpy as np

        from oncue import strip

        white = np.zeros((14, 60), bool)
        out = strip.ocr_label(white)
        return isinstance(out, list), "tesseract runs from Python", None
    check("OCR call", ocr)

    def deepgram():
        import requests

        r = requests.get("https://api.deepgram.com/v1/projects", timeout=30,
                         headers={"Authorization": f"Token {os.environ.get('DEEPGRAM_API_KEY', '')}"})
        return r.status_code == 200, f"HTTP {r.status_code}", None
    check("Deepgram key", deepgram)

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\npreflight: {len(RESULTS) - len(bad)}/{len(RESULTS)} passed"
          + (f"; fix: {', '.join(bad)}" if bad else "; ready for pod/run.sh"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
