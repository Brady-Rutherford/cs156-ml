# Running steps 4–5 on a rented pod

The full ingestion (Deepgram transcripts, Forum events, the hand scan, raises, clips, QA for all
477 classes) is CPU-bound: about 14 h on the laptop, about 2–3 h on a pod with ~72 cores (e.g.
Prime Intellect, GH200: 72 Arm cores, 480 GB RAM, $1.99/h). The GPU is not used. Steps 6–9 and
the PDFs run on the laptop afterwards.

## What goes where

| On the pod | Never on the pod |
|---|---|
| the code (both repos, from the laptop), the Forum sync (metadata, ~160 MB), the three secrets in `pipeline1/.env` | the video (streamed and dropped), PDFs with names (made on the laptop) |
| while running: transcripts, tables, audit clips (classmates pixelated) | anything after `wipe.sh` |

## Steps

**0. Rent the pod.** Pick a CPU-heavy machine with ~50 GB of disk or more and an Ubuntu image,
and add your SSH public key. The dashboard shows how to connect, e.g.
`ssh root@203.0.113.7 -p 22`. Below, `HOST` is `root@203.0.113.7` and `PORT` is `22`.

**1. Laptop: send the code, the data and the secrets** (~1 min)

```
cd ~/cs156-ml
bash pipeline1/pod/push.sh HOST PORT            # add a key path as 3rd argument if needed
```

**2. Pod: set up** (~5–10 min)

```
ssh HOST -p PORT
bash ~/cs156-ml/pipeline1/pod/setup.sh
cd ~/cs156-ml/pipeline1 && ../.venv/bin/python pod/preflight.py
```

The preflight must print `8/8 passed`. If **Forum cookie** or **signed recording URL** fails,
Forum is refusing this machine's IP: stop here, and run on the laptop instead.

**3. Pod: run everything** (~2–3 h; Deepgram ~$190 for 477 classes)

```
tmux new -s ingest
bash ~/cs156-ml/pipeline1/pod/run.sh
```

Detach with `Ctrl-b` then `d` (it keeps running if SSH drops). Come back with
`tmux attach -t ingest`, or follow the log with `tail -f ~/ingest.log`. The last lines give the
number of classes that pass QA and the raises found.

If it stops (the Forum cookie expired, the network failed), fix the cause and run `run.sh`
again: finished transcripts and scans are kept. To refresh the cookie, update
`~/.minerva/.env` on the laptop and run `push.sh` again (it only copies what changed).

**4. Laptop: bring the results back** (~1.3 GB, or ~4 GB with the audit clips)

```
bash pipeline1/pod/pull.sh HOST PORT            # 4th argument --no-audit skips the clips
```

**5. Pod: wipe, then terminate it in the dashboard**

```
bash ~/cs156-ml/pipeline1/pod/wipe.sh
```

**6. Laptop: check and read**

```
cd ~/cs156-ml/pipeline1
../.venv/bin/python -m oncue ingest-report --split all   # checks A–E per class
../.venv/bin/python -m oncue pdf --split all             # one PDF per class (local only)
```

## Settings

`run.sh` reads `TRANSCRIBE_WORKERS` (default 24), `SCAN_WORKERS` (default 40% of the cores) and
`ONCUE_FFMPEG_THREADS` (default 2), e.g. `SCAN_WORKERS=24 bash run.sh`.

## Sharing the scan with the laptop (option B, 14 Oct)

The full run used both machines: the pod scanned train and test, the laptop validation.
(`data/logs/mac_test_ids.txt` can hand test classes to the laptop; it ended up empty: the 19
first given to it went back to the pod before any was started, as the laptop ran ~14/h.)

- `mac_share.sh` (laptop): pulls the pod's transcripts, then scans every validation class in
  batches of 12 (6 workers), then track, episodes, retime, audit and QA. Classes taken by the
  other machine have an empty `data/tables/hand_tiles/{class}.parquet` placeholder, so neither
  machine scans a class twice (track skips a class without a scan record).
- `pod_test.sh` (pod, tmux session `test`): waits for the validation + test transcripts, drops
  the pod's test placeholders, then words, events, select and the scan for test, next to
  `run.sh`'s train scan. `run.sh` does the rest of test after train.
- `pull_merge.sh` (laptop, replaces `pull.sh`): non-empty hand tiles only, new transcripts, caches
  and clips, then the tables merged (the pod's rows for train + test, the laptop's for
  validation; the laptop's tables are backed up in `data/tables_before_merge/`), then QA over
  all classes and the report.

```
bash pipeline1/pod/pull_merge.sh root@HOST PORT ~/.ssh/id_ed25519
```

## Cutting the pod early

`cut_to_mac.sh` (laptop) stops the pod's runs, brings back transcripts, finished scans and their
caches, removes every placeholder, then runs steps 4–5 for all classes on the laptop
(transcribe what is missing, words, events, select, one scan pool over every unscanned class,
then track, episodes, retime, audit, QA). Stop `mac_share.sh` first. Then wipe and terminate
the pod.

**How it ended (14 Oct, evening).** With GPU decoding (`ONCUE_HWACCEL=cuda`, `ONCUE_GPUS=5`, 50
workers) the pod became ~8× faster than the laptop, so everything moved to the pod: the laptop's
16 clean validation scans were pushed to it (`data/logs/mac_push_ids.txt`), and the pod scans the
rest, re-times, makes the clips and runs QA for all 477. No placeholders are left, so the plain
`pull.sh` brings everything back; `pull_merge.sh` and `cut_to_mac.sh` were not needed.
