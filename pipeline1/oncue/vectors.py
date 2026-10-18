"""The content vector store: everything Brady prepared, said, and was taught, as bge vectors.

`python -m oncue vectors [--split all]` writes, under data/vectors/ (gitignored, local only):

* `items.parquet`: one row per item: class_id, course_code, course_area, term, date,
  sched_start_utc, kind, block_order, avail_s, n_words, text, text_sha1, row;
* `embeddings.npy`: float16, one row per item (the same encoder and cache as the features);
* `meta.json`: model, dimension, counts per kind, build time.

Kinds and when each becomes available (`avail_s`, seconds on that class's recording clock;
-inf = before class start):

| kind | source | avail_s |
|---|---|---|
| learning_outcome | the class page's focused outcomes | -inf |
| study_guide | the class page's study guide (prep) paragraphs | -inf |
| pcw_prompt | the workbook template's prompt per block | -inf |
| pcw_answer | Brady's answers AS OF CLASS START, 120-word chunks with 30 overlap | -inf |
| my_utterance | Brady's own transcript frames | the frame's close_s |

Only Brady's own words and the course's material are stored, never a classmate's. A model
that uses the store at moment t of class c may read items of earlier classes, and of class c
only with avail_s <= t (the same causal rule as every other input).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

VECTORS_DIR_NAME = "vectors"


def _materials(class_id) -> dict:
    from . import loaders, pcw

    return pcw.class_materials(loaders.load_raw(class_id, "class_edit_page", log_path=False))


def build_items(class_ids, classes: pd.DataFrame, pcw_blocks: pd.DataFrame,
                pcw_status: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """The item table (text included, local only) for `class_ids`."""
    from . import pcw

    meta = classes.set_index("class_id")
    usable = set(pcw_status.loc[pcw_status["pcw_status"].isin(config.PCW_USABLE_STATUSES),
                                "class_id"].astype(int))
    rows = []

    def add(cid, kind, text, *, block=None, avail=-np.inf):
        text = " ".join(str(text or "").split())
        if not text:
            return
        m = meta.loc[cid] if cid in meta.index else {}
        rows.append({"class_id": int(cid), "course_code": m.get("course_code"),
                     "course_area": m.get("course_area"), "term": m.get("term"),
                     "date": m.get("date"), "sched_start_utc": m.get("sched_start_utc"),
                     "kind": kind, "block_order": block, "avail_s": float(avail),
                     "n_words": len(text.split()), "text": text})

    for cid in class_ids:
        mat = _materials(cid)
        for lo in mat["learning_outcomes"]:
            add(cid, "learning_outcome", lo)
        for para in mat["study_guide"]:
            add(cid, "study_guide", para)
        b = pcw_blocks[pcw_blocks["class_id"] == cid]
        for r in b.itertuples():
            add(cid, "pcw_prompt", r.prompt_text, block=int(r.block_order))
            if cid in usable:
                for chunk in pcw.chunks(r.answer_text):
                    add(cid, "pcw_answer", chunk, block=int(r.block_order))
        mine = frames[(frames["class_id"] == cid) & (frames["role"] == config.ROLE_ME)]
        for r in mine.itertuples():
            add(cid, "my_utterance", r.text_norm, avail=float(r.close_s))
    out = pd.DataFrame(rows)
    if len(out):
        import hashlib

        out["text_sha1"] = [hashlib.sha1(t.encode()).hexdigest() for t in out["text"]]
        out["row"] = np.arange(len(out))
    return out


def cli(args) -> int:
    from .embed import Embedder
    from .splits import split_ids

    t0 = time.time()
    tables = config.TABLES_DIR
    ids = sorted(split_ids(getattr(args, "split", None) or "all"))
    classes = pd.read_parquet(tables / "classes.parquet")
    items = build_items(ids, classes, pd.read_parquet(tables / "pcw_blocks.parquet"),
                        pd.read_parquet(tables / "pcw_status.parquet"),
                        pd.read_parquet(tables / "frames.parquet",
                                        columns=["class_id", "role", "close_s", "text_norm"]))
    emb = Embedder(config.EMB)
    vecs = emb.encode(list(items["text"])) if len(items) else np.zeros((0, 384), np.float16)
    out = config.DATA_DIR / VECTORS_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    items.to_parquet(out / "items.parquet", index=False)
    np.save(out / "embeddings.npy", np.asarray(vecs, dtype=np.float16))
    counts = items["kind"].value_counts().to_dict() if len(items) else {}
    (out / "meta.json").write_text(json.dumps({
        "model": config.EMB, "dim": int(vecs.shape[1]) if len(items) else None,
        "n_items": int(len(items)), "counts": counts, "classes": len(ids),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print(f"oncue vectors: {len(items)} items {counts} -> {out} ({time.time() - t0:.0f} s)")
    return 0
