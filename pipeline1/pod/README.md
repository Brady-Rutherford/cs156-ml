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
