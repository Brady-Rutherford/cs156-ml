"""Re-scan the classes listed in a file, each new scan replacing the old one in one step:
`python pod/rescan.py LIST WORKERS` (run from pipeline1/). Used by pod/pod_mild.sh."""

from __future__ import annotations

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from oncue import strip

    ids = [int(x) for x in Path(sys.argv[1]).read_text().split()]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    bad = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for cid, info in pool.map(strip._scan_job, ids):
            if info is None:
                bad.append(cid)
                continue
            print(f"scan {cid}: {info['n_frames']} frames, strip {info['band_share']:.0%}, my tile on "
                  f"{info['n_me_frames']}, ring check {info['valid_share']} "
                  f"({info['n_anchor_frames']} frames), {info['n_ocr']} OCR reads", flush=True)
    print(f"re-scan: {len(ids) - len(bad)} classes scanned, {len(bad)} failed {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
