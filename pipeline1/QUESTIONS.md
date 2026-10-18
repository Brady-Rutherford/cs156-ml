# Open questions and answers (pilot, 13 Oct 2026)

Brady's questions during the Deepgram ingestion and the 20-class pilot, with the answer as it
stands. Revisit each before the write-up.

## How is my hand raise found in the video? Is it a learned model, like a tensor of pixels for MNIST?

No model is trained for the label. It is two fixed rules on the pixels:

1. **Which tile is me:** the name written in white at the bottom of each tile in the top strip.
   Each label is read with OCR (Tesseract) and matched to the class roster. A label already
   read at the same tile width is recognised from its pixels after that: a 16 × 96 patch of
   the word, compared with normalised cross-correlation. So it is a pixel tensor, but matched
   against my own label, not classified by a trained network.
2. **Is my hand up:** a colour-and-shape test in the top-left corner of my tile. The frame is
   converted to HSV, and the share of yellow-green pixels (OpenCV hue 26–40, bright and
   saturated) in a square of 34% of the tile height is measured. It must pass 0.10, and the
   blob must be a round disc with a white hand inside. Two agreeing samples at 2 fps confirm a
   change (a state machine with hysteresis). Each onset is then re-read at the full frame rate
   (`retime`).

A small classifier on these pixel patches (an MNIST-style CNN, or kNN on label patches) is
possible. The rules already separate the classes cleanly on the pilot frames, though, and a
rule has no training data to leak.

## The green border means speaking; the yellow-green hand is the label

Yes. The teal speaker ring (hue 75–77) is never a label. It is used in two checks only:

- **tracking:** while I speak alone (my own Forum voice events), the one bordered tile must be
  the tile my name label marks. 28 of 28 pilot frames agreed;
- **A/V lag:** the border on my tile against my words in the transcript.

## Main stage

Not read. My tile is always on the thin top strip, also while I am on stage, so only the strip
is scanned. One pilot class (103152, B165) shows the strip down the left side instead. The
scanner finds no top strip there and marks those frames NA. Check how often that layout occurs.

## Classmates' names during the scan

Allowed (Brady, 13 Oct): the scan may read classmates' name labels to tell tiles apart, as long
as only ids are kept after processing. Implementation: names stay in memory inside `LabelBook`.
The stored tables say only whether a tile is mine (`ncc_me`), and the cache keeps only my own
label patches. Printed diagnostics are counts, never names.

## Grey background, only teal and green visible

That is what the detector does internally: each frame becomes two masks (teal-ring pixels and
yellow-green hand pixels), and every other colour is ignored. The same idea will be used for
the audit clips: everything desaturated to light grey except teal and green, so the border and
the icon stand out. Other tiles stay pixelated.

## Why the 15-minute limit on Deepgram? Can we get around it?

It is not a Deepgram limit, nor a limit on video length. Scribe's Deepgram client gives each
request a 900-second read timeout (`DeepgramASR(timeout=900.0)`). In URL mode Deepgram first
fetches the whole video from Forum's CDN. That took about 5 minutes for the 2025–26 recordings
and more than 15 for the 2023–24 ones, so those requests timed out and fell back to upload mode
(download locally, extract the audio, upload ~20 MB).

Fix for the full ingestion: use upload mode for every class (`url_mode=False`). It is the same
lowres file the scan reads, with no wait on Deepgram's fetch, and every class gets the same
audio processing. Upload mode's clean-up filter (high-pass, low-pass, loudness) does not trim
or stretch the audio, so word times stay on the video clock. The pilot compares the
calibration offsets of URL-mode and upload-mode classes to confirm this.

## How is the video read? Do we download it, or stream it?

**For the hand scan: streamed, never downloaded.**

1. Scribe's `ForumClient` asks Forum's API (with my login cookie) for the class's recording
   links. Forum returns signed CloudFront URLs, valid for a limited time, for two renditions.
   We always use `lowres` (`zencoder-recording-600k.mp4`: still 1920 × 1080, just a lower
   bitrate), the same file Deepgram hears.
2. `ffmpeg` opens that URL over HTTPS and reads it progressively. It decodes every frame in
   memory and keeps 2 per second (`fps=2:round=up`, so a sample never shows anything later than
   its time stamp). It crops the top 360 rows, where the strip is, and pipes the raw pixels to
   Python. With `-copyts` plus the `showinfo` filter, every frame arrives with its exact
   presentation time, and frame time = that time minus the audio start (0 in all 20 pilot
   classes).
3. Python (OpenCV + numpy) measures the frame and throws it away. Only numbers are kept (per
   tile: hand share, ring share, whether it is my tile).
4. When the signed URL expires mid-class, a fresh one is fetched and ffmpeg resumes 2 s before
   where it stopped; frames already seen are dropped by their time stamps.

`retime` (each onset at full frame rate) and `audit` (a 6 s clip per raise) make short reads the
same way: seek to the time, read 1.5–6 s. The only video written is the grey-view audit clip
(local, gitignored).

**For transcription there are two modes:**

- *URL mode* (used until 13 Oct): Deepgram gets the signed URL and downloads the video on its
  side. That was slow for some recordings (> 15 min, the client's timeout).
- *Upload mode* (default now): Scribe downloads the lowres mp4 into a temporary job folder
  (~450 MB, with `requests`), `ffmpeg` extracts the audio as 16 kHz mono Opus (~20 MB), the video
  is deleted at once, the audio goes to Deepgram's EU endpoint (model-improvement opt-out on),
  and the job folder is deleted afterwards.

Tools: `ffmpeg` / `ffprobe` (Homebrew), Scribe's `ForumClient` (signed URLs), `requests`
(download, Deepgram upload), OpenCV + numpy (pixels), Tesseract (name labels).

## What is the stack? How do you "see" the video?

| Step | Tool |
|---|---|
| signed recording URL | Forum API through Scribe's `ForumClient` (my cookie) |
| stream facts (start times, size, duration) | `ffprobe` |
| streaming + decoding, 2 fps, crop to the strip, exact frame times | `ffmpeg` (`-copyts`, `fps=2:round=up`, `crop`, `showinfo`), raw pixels to a pipe |
| pixels as arrays (360 × 1920 × 3 numbers per frame) | numpy |
| colour masks (HSV), circle/shape test, resize, blur | OpenCV (`cv2`) |
| strip and gutters, label patch matching (cross-correlation) | numpy |
| reading name labels | Tesseract (CLI), matched to the roster with `difflib` |
| per-tile numbers | pandas / parquet |
| audit clips and frame sheets | OpenCV (`VideoWriter`, PNG) |

Claude cannot watch video. To check something by eye it renders still images (rows of frames,
grey view, every other tile pixelated) to PNG with OpenCV and opens them with its file reader,
which shows the image. Those images go to the model, so other people are pixelated first. The
mp4 clips in `audit/` are for Brady to watch.

## Did OCR get resolved? How?

Yes. First tries read my own label in only 13 of 31 frames. The fixes:

1. a white-text mask (S < 40, V > 225) instead of a grey image, scaled so the label is ~60 px
   high;
2. two variants per label (with and without a 2 px speck filter);
3. correcting Tesseract's usual confusions ("cl" → "d", so "Bracly" → "Brady");
4. matching to the *roster*, not to a fixed word: the closest roster name wins if its score
   is ≥ 0.60 and beats every other person by ≥ 0.12.

Result on the pilot frames: my label was read as me in 32 of 33 frames, and 0 of 695 classmate
labels were read as me.

## PDF transcripts

Requested: one PDF per class with the class's video link at the top and a summary (hand-raise
times with links to their strip clips, seconds I spoke, number of turns, talk share by role,
polls, chat, lesson segments, clock-sync quality). Built after the hand scan has run on the
pilot.

## Found during the pilot

- **Other icons are reactions** (Brady, 13 Oct). In class 92517 at 2032 s my corner showed an
  orange icon with a different glyph. Only the yellow-green hand is a raise. The orange icon sits
  at hue 10–19 and the hand at 30–34, so the hand test never counts a reaction. The teal ring
  does not leak into the corner square either (poll-tick share 0 while bordered).
- **Real UI vs the synthetic one.** The real strip reflows within a class (tiles 69–200 px
  wide), the speaker ring is teal (hue 75–77, not 45–70), and names are white text over the
  video, not a label bar. The strip scanner (`oncue/strip.py`) replaces the glyph scanner for
  real recordings. The synthetic end-to-end test still uses the old one.
- **A signed URL in an error message.** An ffprobe error was cut to its last 300 characters
  before redaction, so part of a signed URL reached the log. Fixed: redact first, then cut.
- **Roster names in an error message.** A transient SSL error during a Deepgram upload quoted
  the request path, whose query string holds the roster keyterms (classmates' names). The
  names reached the local pilot log. Fixed: the log was scrubbed; Scribe's `redact_urls` and
  oncue's `redact_error` now strip query strings from bare request paths too; transient
  network errors are retried (3 attempts) instead of failing the class.
- **The raise animation is the true onset.** When the raise key is pressed, Forum fills the
  whole tile with yellow-green and a big white hand for ~1.7 s, then shrinks it to the corner
  icon (class 79900: overlay from 553.54 s, icon from 555.29 s; 24 fps). The overlay hides the
  name label, so my tile went unread and every raise looked like it started from NA (all
  discarded as left-censored). Fixed: the overlay counts as a hand; the overlay tile at my
  last read position (±3 px, ≤ 3 s earlier) is mine; re-timing finds the overlay's first frame.
- **Gutters smeared on keyframes.** Every 2 s (the keyframes), the encoder bleeds tiles into
  their 4 px dark gutters (brightness 0.02 → 0.18), two tiles merge, and my tile dropped out of
  the tile list, which split long raises into pieces. Fixed: gutters are also found as narrow
  dark valleys, and a frame that only lost tiles keeps the previous frame's tile list (a real
  reflow moves or resizes tiles; a smeared gutter only removes them).
- **The first audit view leaked.** It pixelated only the tiles the scanner had found, so a
  missed tile showed unpixelated (a few names, the professor's face) in one image Claude
  viewed. Fixed: the whole strip is pixelated, then only my tile is pasted back.
- **Polls were empty (step 4 bug).** `polls_table` and the raw-data allowlist had been written
  for a guessed Forum layout (`timing-text`, `poll-id`, `answer`). The real one is
  `poll-sessions[].started-at` plus responses with `poll-session-id` / `response`, so every poll
  had no time and no answer (class 104899: 5 answers in the raw data, 0 in the table). Fixed:
  one row per poll session, open = `started-at`, close = the next segment start ≥ 10 s later
  (sessions have no end, so close is approximate), my latest non-draft answer.
- **Forum reorders the strip, so position alone never identifies me.** In 105076 at 2443 s a
  classmate who had just raised their hand moved into the slot where my tile had been, and
  the first carry-over rule ("an overlay at my last position is mine") counted their raise
  as mine: 1–2 s false "raises". Fixed: the overlay is carried as mine only when its label is
  not read as someone else (on the 2026 UI the name stays readable on the overlay).
- **Reactions cover the tile.** A full-tile reaction animation (e.g. orange) can play over my
  tile while my hand is up. The yellow-green corner icon stays visible, but the tint changes
  my label patch: same-width match 0.97 → 0.68, so my tile dropped out for ~1.5 s and long
  raises split. Fixed: my label is also matched against my patches at *any* tile width
  (NCC ≥ 0.88, beating every other tile by ≥ 0.15). This also covers the end of class, when
  the strip resizes as people leave.
- **Breakouts have no strip.** 94–100% of breakout frames show no main-room strip, so there
  is no label there. That matches the breakout exclusion rule.
- **Short unreadable gaps split raises.** With reactions and overlays hiding the label for
  ~1.5–2 s, the synthetic design's 1 s NA bridge split long raises, and an onset reached
  through a hidden-label gap was thrown out as left-censored (49% of episodes). Fixed for the
  real UI: NA runs ≤ 3 s are bridged, and an onset through such a gap after 2 s of clean DOWN is
  kept (onset = start of the gap, then re-timed). On the pilot, discarded episodes fell from
  217/300 to 84/202 before the last re-scan.
- **The Mac slept mid-run.** On battery (42%), macOS went into sleep from ~16:55 to 17:24 and
  the three video streams hung with no data and no error. Fixed: ffmpeg gets `-rw_timeout`
  (30 s), so a silent stream errors out and the reader resumes with a fresh signed URL; runs go
  under `caffeinate -i`. **For the overnight full run the Mac must be plugged in with the lid
  open:** closing the lid on battery still forces sleep.


## The full run on a rented pod (14 Oct)

- **Would a pod with more CPU be faster than the Mac? Is the GH200 a 72-core CPU?** Yes, the
  scan is bound by the CPU (software H.264 decoding of each 1080p frame before the crop, then
  Python per frame), so it scales with cores. The GH200's Grace CPU has 72 Arm cores. The pod
  actually rented (5×A40) gives its container 40.8 CPUs by cgroup quota although `nproc` shows
  96; `run.sh` reads the quota. The GPUs are not used: NVDEC would decode differently from the
  pilot (the run must match the pilot exactly) and the Python part stays on the CPU anyway.
- **Why did 2 of the 20 pilot classes fail?** 83649 and 95228 miss the strict calibration rule
  (residual p95 1.68 s and 1.52 s against ≤ 1.5 s). Both pass the ≤ 3 s sensitivity rule. Which
  rule to use is still Brady's call.
- **Why is it so slow?** Each pod vCPU is about a quarter of a Mac core. Also, at first every
  worker started one thread per library per visible core (96), and the load hit 204. Fixed:
  one thread per library in each worker, and `-filter_threads` for ffmpeg. After that, about
  40 train classes an hour with 16 workers.
- **Are the pod and the Mac running at the same time?** Yes, on different classes ("option
  B"): the pod does train and test, the Mac validation. An empty `hand_tiles/{class}.parquet` on one machine marks a
  class the other machine does, so no class is scanned twice. `pod/pull_merge.sh` joins the
  results at the end: per-class files, plus the tables merged class by class.
- **Can more move to the Mac?** No. With 6 workers and 16 GB RAM, the Mac scans about 14
  classes an hour (batches of 12 wait for their slowest class). Every 5th test class went to
  the Mac at first; at 14/h it would have finished last, so they went back to the pod before
  any was started. Validation alone (94) on the Mac and train + test (383) on the pod finish
  their scans at about the same time.
- **The network is good. Is it the bottleneck?** No. Both machines are short on CPU, not
  bandwidth: a 600 kbps stream is tiny, but decoding it at full resolution is the expensive part.
- **What is the scan doing?** Per class: a fresh signed URL for the `lowres` recording; ffmpeg
  streams it, decodes every frame, keeps 2 per second (~12,100 frames for 100 minutes) and
  crops the top 360 rows; Python finds the strip and its tiles, finds my tile by its name label
  (stored label patches first, Tesseract OCR + roster for a new label), and measures every tile's
  top-left corner (yellow-green share, round-hand shape) and its teal ring. It writes
  `hand_tiles/{class}.parquet` (one row per tile per frame, `ncc_me` = 1 on mine; no names) and
  the ring check. A log line `12121 frames, strip 80%, my tile on 9089, ring check 1.0 (294
  frames), 1122 OCR reads` = frames read, share showing the strip, frames with my tile found,
  ring check over that many solo-speech frames, OCR calls needed.
- **What do the `pod_live` pictures show?** One PNG per class, rewritten every 60 s of video
  while it is scanned (the last one stays): the strip in grey view (every tile but mine
  pixelated, only teal and yellow-green kept in colour; my tile boxed), captioned with the
  class, the video time, "my tile found / NOT found" and "HAND UP / hand down". A small
  yellow-green square on another tile is a classmate's raised hand.

## Found during the full run (14 Oct)

- **The ring check mixes two things.** 17 of the first 92 train classes (all 2023–24) score
  < 90%, against 1 of 20 in the pilot. Almost every miss is a frame where my tile was not read
  at all (985 frames), not a wrong tile (52 frames: another tile ringed while mine was read).
  When my tile is read during my solo speech, it is the ringed one in 95–100% of frames in 15 of
  the 17 (81174: 64% of 42 frames; 79603: 81% of 21). Classmates speaking alone: 27 of 96,710
  frames had their ring on "my" tile (0.03%). Unread frames never become DOWN: moments whose
  window is not fully observed are left out of the labels (`unobservable`). So the 90% rule as
  written drops classes for low coverage, which the label mask already handles. Decision for
  Brady before the final QA: keep it, or apply 90% to the read frames only (plus the coverage
  rule).
- **"Does my tile have to be in 90% of the class? Breakouts are longer than that."** No. The
  ring check only uses moments when Brady speaks alone in the main room; breakouts, before and
  after class never count. Brady's rule (14 Oct): **drop no class** for coverage or breakouts,
  and miss no raise.
- **Why was my tile unread for minutes in some classes? (root cause, 14 Oct)** A classmate whose
  display name is not on the roster got matched to "Brady" by OCR at 0.67 (the bar for any name
  is 0.60). Their label then joined Brady's, both tiles matched him, and two tiles named Brady
  make a frame ambiguous, so his tile went unread while both were on screen (83098: 68% of the
  strip time). Brady's own label was clear and read correctly in those frames. Fix:
  `ME_NAME_MIN = 0.75` (OCR text must be this close to his name; his own misreads score
  0.80–1.00). Classes scanned before the fix with > 2 min unread in long stretches, > 10% unread,
  or a classmate's ring on "his" tile are scanned again (57 train on the pod, 13 train copies +
  10 validation on the laptop; old results in `data/rescan_backup/`).
- **Is the progress board accurate?** It counted the other machine's empty placeholder files as
  "scanned". Fixed: `scanned` counts real scans only, and a new column shows the classes left
  to the laptop. One validation class (103407) failed transcription (the download broke) and is
  being retried.
- **"That's really expensive. How do we speed it up?" (14 Oct, 18:30)** Use the pod's five
  A40 GPUs, idle until then, to decode the video: decoding was ~70% of the scan's CPU (ffmpeg ~14
  CPUs, Python ~6, Tesseract little). `ONCUE_HWACCEL=cuda` decodes on the GPU, keeps 2 frames a
  second there, downloads only those and converts them exactly as the software path does. The
  frames are bit-identical to the pilot's software decode (240/240 strip frames of 104899, and
  the scan reader's 120/120 frames with the same timestamps in 83098), at ~¼ of ffmpeg's CPU.
  The laptop keeps software decoding (VideoToolbox was slower there, 13× vs 33× real time).
- **"How did you make it faster, and where are the numbers?" (19:20)** GPU decoding (above), 50
  scans at once dealt over the 5 GPUs in turn, and all work on the pod (~130 classes/hour against
  ~45 before and ~14 on the laptop). 19:20: 132/477 scanned.
- **A second cause of unread stretches (80193, 19:30).** After the first fix, 13 of 72 classes
  still had Brady's tile unread during his solo speech, but never the wrong tile (when read, it
  was his in 100% of frames in 12 of the 13). In 80193 a classmate's label looked 0.91 like his
  stored labels (blurred 16 × 96 word patches), so it was taken as his without OCR (same-width
  match ≥ 0.85), and two tiles were his. Fix: `ME_SAME_LABEL_NCC = 0.95`, so a label is taken as
  his without OCR only when it is nearly identical to one of his; otherwise it is read once and the
  classmate's label gets its own entry. Being verified on 80193 and a healthy class (76260)
  before it goes into the run.
- **"All the transcriptions are done, correct?" (20:40)** 477/477 transcripts exist and 475 have
  words. 98590 (test) and 101578 (train) were recorded without sound: the `lowres` audio is
  digital silence (−91 dB at 10, 40 and 70 min) and `highres` has no audio track. Deepgram
  rightly returns no words; their raises come from the video, but there is no transcript and no
  clock calibration. Keep or leave out: Brady's call. Calibration so far (train + test, 383): 371
  ok, 7 piecewise, 5 unverified (the two silent ones + 3 to look at).
- **"Is it running the faster process?" (20:33)** Yes: 50 GPU-decoding scans. It looked slow
  because the restart for label fix 2 put every worker back at the start of a class (~18–20 min
  each). The fix also makes Python work harder where labels look alike, and the CPU is at its
  limit, so the steady rate is ~120–130/hour instead of ~150.
- **"796 left-censored (32%): can you fix that?" (23:10)** Mostly they are not lost raises. Of
  the 682 in train classes outside the milder re-scan, 573 (84%) have a hand-UP reading in the
  10 s before: the same raise continuing through a short unread stretch or a flicker, already
  counted from its first piece. Only ~20 have a clean DOWN reading ≤ 3 s before the hand goes up
  and no UP in the 10 s before (~45 if the start may be 5 s uncertain). A first look at 7 s
  windows had suggested ~400, which was wrong. A "single DOWN reading" rule would gain ~1%, so it
  was not adopted (code written, measured on the pod's frames, reverted).
- **Clips: "are you saving them? keep them" (23:20).** Yes: `pipeline1/audit/{class}/`, one 6 s
  grey-view clip (3 s before, 3 s after) and a frame sheet per re-timed raise, plus `raises.csv`;
  local, gitignored, kept (2,774 clips, 2.9 GB). For face work later: colour clips of his own tile
  only (8 s before, 3 s after, plus non-raise moments), cut fresh from Forum's recordings with the
  saved raise times and tile positions. Not done tonight (Brady: "only for next time").
- **QA rules (Brady, 23:40):** clock limits 2.5 s; ring check over read frames; drop no class for
  breakouts or unread stretches, but no useless data. Applied on the laptop after the pull:
  446/477 classes pass (was 401 under the old rules).

## 18 Oct: pre-class work, inputs, models, notebook

- **Pre-class work: was it saved?** Not until 18 Oct. The harvest (`oncue harvest`) fetched every
  kept class's page (learning outcomes, study guide) and workbook (template, answers, every
  revision). Side-effect check first (D8): two requests 2 min apart changed nothing. One class
  (101578, never opened) got a blank workbook created by the request, as opening it in the
  browser would; the guard stopped there and no other was created. 339 workbooks saved; all were
  still edited after class began, so each is rebuilt **as of class start** by replaying its
  revisions (`oncue/sharedb.py`; replaying all revisions reproduces the saved doc in 338 of 340).
  Result: 325 classes `asof_start`, 12 empty, 139 none, 1 unknown.
- **What are the 76 inputs?** Class context (22), who is talking (10), my participation (6), my
  raise history (4), polls (4), questions (6), words of the last 15/60 s (7), pre-class work and
  class content (16), punctuation (1). Dropped: people present (Brady: who spoke covers it), my
  breakout group (not parsed), teaching-move scores (models removed from Hugging Face).
- **What does "counts" mean / what is a positive moment?** A moment counts (`at_risk`) when a
  raise could start then and the label can be trusted (hand down, not just lowered, not
  speaking, main room, my tile readable, class passed QA): 628,079 of 1,427,872. A positive
  moment is a counting moment whose raise comes 2-15 s later: 9,206 (1.5%), about 3.3 per raise.
- **Embeddings and the vector store.** One encoder for all text: `BAAI/bge-small-en-v1.5`, 384
  numbers per text (Brady: small is enough; only the transformer reads raw text). 512,381
  transcript frames feed the per-moment inputs. The store (`data/vectors/`, local) holds only my
  words and course material: 33,354 items (my utterances, study guide, prompts, learning
  outcomes, my answers as of class start), tagged with class, course, term, date.
- **Was the faster build less precise?** No: vectors streamed to disk instead of held in memory,
  stored as float16 (the cache was already float16), and the teaching-move models were absent
  anyway. The only precision choice is bge-small vs bge-base.
- **Many models: one pipeline or exploration?** One pipeline with the trees as the final model;
  baselines first (question timing, Naive Bayes, TF-IDF), then specialists stacked under the
  trees (content-alignment MLP, TCN, fine-tuned transformer); a reinforcement-learning layer
  could later decide *when* to cue. Draft: `design/model_pipeline_proposal.md`.
- **Can the model give the exact time of the next raise?** Two views: the probability timeline
  (fire at a threshold set on train; hit rate, lead time, false alarms per hour), and later a
  time-to-event (hazard) view that predicts a time and is scored in seconds.
- **What sets the threshold? Are 30 false alarms per hour OK?** The threshold is the probability
  at which the model says "a raise is coming in 2-15 s". It is not learned: it is picked on the
  train classes so the model fires at a chosen rate, then frozen for validation and test. The
  first rule picked it for 6 or 30 false alarms per hour; 30 per hour was too loose (below).
- **"3 of 16 raises warned": did I raise 16 times? Train, validation or test?** Yes, in that one
  example validation class. Every model is trained on train only and scored on validation;
  test is scored once, with everything frozen.
- **What is AP?** Average precision: rank every moment by the model's probability, take the
  precision at each truly positive moment, and average:
  AP = (1/N+) Σ_{k: y_k = 1} P(k). Random ranking gives the positive share (0.015); perfect
  gives 1. Worked example (asked again 18 Oct): ranked moments ✓ ✗ ✓ ✗ ✗ ✓ give
  precisions 1/1, 2/3, 3/6 at the positives, so AP = 0.72. Validation AP of the naive models: base rate 0.015, professor-question rule 0.025,
  question timing 0.023, Naive Bayes 0.017, TF-IDF 0.024.
- **"We guessed way too many times": how to fix it?** The first rule (30 false alarms per hour)
  made about 27 guesses per class for about 6 raises. Now the threshold is set so the model
  guesses about as often as I raise (about 8 per hour of counting time on train), and it must
  be high 2 moments in a row (4 s). TF-IDF on validation: about 5 guesses per class, but the
  share of guesses followed by a raise stays at 3.3%, so the naive models lack signal rather
  than restraint.
- **Why a 60 s cooldown? I sometimes raise twice in 30 s.** It was a blunt way to stop one long
  high stretch from counting as many guesses, chosen without checking my raises. On train, 10.5%
  of my raises come within 60 s of the previous one (5.7% within 30 s, 3.4% within 15 s), so
  even a perfect model could warn at most 93% of them. Dropped (Brady: "it needs to be able to
  make a guess whenever it's possible"). The rule now: after a guess, the model may guess again
  once its probability dips below the threshold and comes back, or once I have raised; never
  sooner than 13 s (H - g, spec §8.3), because one guess already covers the next 13 s. A
  perfect model could then warn 100% of my raises. The number of guesses is set by the
  threshold, not by this gap.
- **Does the model look at the past x seconds, with the probability rising as a raise nears?**
  Yes: every 2 s it reads the class so far (the professor-question rule and question timing:
  the state of the last question; Naive Bayes and TF-IDF: the words of the last 60 s) and gives
  P(raise in the next 2-15 s). On validation, in the minute before my raises: the
  professor-question rule climbs to 3x its usual level in the last 10 s; Naive Bayes sits at
  about 2x the whole minute (it reacts to the topic, not the timing); question timing and
  TF-IDF barely move (1.2x). On average none reaches its guess line before a raise. Notebook
  Step 2 now has one 2-panel figure per model (the minute before my raises; one class), the 4
  squares (TP / FP / FN / TN per 2-s moment), one combined figure and an accuracy table.
- **Is making only 4-6 guesses an improvement already, even for the naive models?** Fewer
  guesses is what we asked for, but the count is a setting (the threshold), not skill: any
  model can be made to guess 5 times per class. Skill is how often a guess is right compared
  with a random guess (1.5%): 3.3-3.8% on validation, 3.0-4.7% on test, so 2-3x better than
  random, but still about 96% of guesses are wrong and only 2-5% of my raises are warned. Against
  the first rule: wrong guesses fell from about 22-27 to about 5 per class, and caught raises
  fell from 8-16% to 2-4%. Same models and the same AP, at a different point of the trade-off.
- **Are we now capping how many guesses it can make? More variance than before? Did the change
  break "raise in the next 2-15 s"?** No cap: the threshold is one number fixed on train, and
  on validation and test the model guesses whenever its probability is above it. The target,
  the models and their predictions are unchanged (AP identical to 4 decimals). The jumpiness
  comes from small counts: the models warn a few percent of the 504 raises in the 86 scored
  validation classes (3% is about 15 raises), so one or two raises move the numbers. Brady: the previous approach was better,
  so no cooldowns or firing rules; every 2 s the model says yes when its probability is above
  the threshold (open: which threshold).
- **How do we build the flexible models, step by step: MLP, TCN, fine-tuned transformer, all
  fed into the gradient-boosted trees?** Yes, a stacked pipeline (design/model_pipeline_proposal.md).
  Each specialist gives a score that becomes one more input to the trees. The trees must only
  see out-of-fold scores (each specialist trained on 3/4 of the train classes and scoring the
  held-out quarter, 4 times), or they learn to trust overfit scores. Proposed order: the trees
  on the 76 inputs first, then + MLP, + TCN, + transformer (on a pod), each measured on
  validation, so every step leaves a finished result and the comparison table shows what each
  part adds.
- **How do the naive models make a prediction, and when?** Every 2 s, at each counting moment,
  each model looks back (the question models: the state of the professor's last question; the
  word models: the words of the last 60 s, tagged by who said them) and gives one probability;
  at or above its threshold, that moment is a guess "my hand goes up in the next 2-15 s".
  Walk-through with figures on one validation class (87550, 37:00-42:30): private page
  https://claude.ai/artifact/PWjPAyPLau3baZDfVHDuxd.
- **Is the model cheating, reading the transcript and seeing me talk after the raise?** No. A
  word enters the window only once it has been fully spoken (end <= t - 0.25 s), chat once
  posted; moments while my hand is up, while I speak or just after I lower it do not count,
  and a guess only scores 2-15 s before the raise. Words and the hand icon come from the same
  recording (one clock); names and call-on phrases are removed. Checked 18 Oct: the 20 causal
  tests pass, incl. the real-data one (inputs rebuilt from data cut at t equal the stored
  ones; the test's sampler was fixed to skip silent classes with no end time), and 300 of 300
  random positive moments have a 60-s window identical to one rebuilt from data cut at t.
- **The 4 squares make no sense: I am not raising most of the time, so true negatives are
  silly.** Agreed (Brady): score only the guesses. How many times it guessed a raise in the
  next 2-15 s, how many were right (my hand went up), how many wrong.
- **What do the moment metrics mean (accuracy, precision, recall, median lead)?** Each treats a
  2-s moment as one prediction. Accuracy (moments) = (right yes + right no) ÷ all; dropped,
  it is almost all true negatives ("always no" scores 98.5%). Precision (moments) = right ÷
  yes; kept as "% right". Recall (moments) = positive moments that got a yes ÷ all positive
  moments; replaced by "raises warned", which counts raises, not moments. Median lead =
  seconds between the earliest warning and the raise; replaced by the warning windows below.
  The notebook's Step 2 now opens with a table explaining every number.
- **Is "raises warned" how many it could have got if it was perfect?** No: "24 of 504" = the
  model warned 24 of all 504 validation raises. A perfect model (guessing exactly at the
  positive moments) warns only 326 of 504: 178 raises have no counting moment 2-15 s before
  them (e.g. my tile unreadable, or I was speaking), so no model can warn them. Every table now
  has a "perfect model (ceiling)" row.
- **Why remove "warned more than 2 s"? 2-5 s is the ideal range.** It was not removed: every
  warning is at least 2 s ahead by definition, so "raises warned" was that count; now labelled
  "a guess 2-15 s ahead". Added (Brady): raises with a guess 2-5 s ahead (the ideal), 5-10 s,
  and 10-15 s (a raise can count in more than one window).
- **Part 3 started (18 Oct, Brady: trees first, then the MLP; Part 3 of the notebook).**
  `oncue/flex.py`, `python -m oncue flex --step all` (402 s on the laptop). 3.1 trees
  (HistGradientBoosting) on 68 inputs (76 minus has_qmark, spk_cur_in_my_group, ctx_n_present
  and the 5 minute splines); 3 tree sizes x up to 600 trees, chosen on validation: 15 leaves,
  80 trees. 3.2 content-alignment MLP (25,473 parameters) on 5 frozen bge vectors: the
  professor's latest words, the last 60 s, the class material, my pre-class answers as of class
  start, my earlier words (earlier classes + this class before t; tested causal), plus 7
  cosines and 5 flags; best epoch 3 of 20 by validation AP; 4 grouped out-of-fold models for
  the trees. GBT + MLP: the trees with the MLP's score and the cosines (31 leaves, 160 trees).
  Validation AP: best naive 0.0245, trees 0.0270, MLP 0.0276, GBT + MLP 0.0292 (AP_w 0.062,
  ROC AUC 0.70, the best on every ranking number); its guesses: 19.2 per class, 3.9% right,
  18 of 504 raises warned (15 with a guess 2-5 s ahead). The MLP's score is the input the
  stacked trees rely on most. Test not scored for Part 3 (held for the final pipeline).
- **Did you not test the naive models? Why is it not in the notebook?** It is: section 2.6, just
  before Part 3, with the models and thresholds frozen. Test AP: professor-question rule 0.0246,
  question timing 0.0263, Naive Bayes 0.0196, TF-IDF 0.0268 (close to validation); 533 test
  raises, a perfect model could warn 322. If the notebook is open in an editor from before the
  rerun, reopen it (do not save the old copy over the new file).
- **I do not raise 20.7 times per class, so why does the perfect model guess 20.7 times?** It
  counts yes *moments*, not raises: the model predicts every 2 s and "a raise is coming in
  2-15 s" is true at every counting moment of the 13 s before a raise, about 3.5 per raise
  (1,778 positive moments / 504 raises), so 5.9 raises x 3.5 = 20.7 per class. Counting an
  unbroken run of yes moments as one *warning* (counting only; no cooldown), the perfect model
  makes 3.9 warnings per class (one per raise it can see; a third of raises have no counting
  moment before them). Validation, warnings per class / % right: professor-question rule
  14.8 / 3.4%, question timing 5.0 / 6.3%, TF-IDF 2.4 / 5.3%, trees 2.8 / 4.1%, MLP 8.8 /
  6.6%, trees + MLP 4.8 / 4.8%. Proposed: count guesses as warnings in the notebook (asked).
- **Why do even the naive models have most warnings 2-5 s before the raise? Does probability
  build up as time passes?** Mostly a real build-up: the share of counting moments where a model
  says yes is highest 2-5 s before a raise (validation, 2-5 / 5-10 / 10-15 s / elsewhere:
  professor-question rule 4.6 / 3.2 / 2.5 / 1.3%; question timing 5.2 / 4.3 / 3.3 / 1.3%; MLP
  8.5 / 6.9 / 6.5 / 2.7%; trees + MLP 5.2 / 3.3 / 2.7 / 1.4%), because what prompts a raise (a
  question ending, the floor opening, a turn ending) usually happens only seconds before it,
  and a model sees it only once it has happened. Also: more counting moments close to a raise
  (153 vs 126 per second of window at 2-5 vs 10-15 s; further back I am more often speaking or
  in my previous raise), and the window columns overlap (a raise counts in every window with a
  guess: MLP 30 + 28 + 22 = 80 for 45 warned raises), while 2-5 s is the narrowest window.
- **We build the TCN next? Will it give the biggest rise, given what it can reflect on?** Yes,
  3.3 is next. It may help most because it reads the class as a sequence (one step per 2 s,
  ~8.5 min look-back) and so sees the order of events, which the trees (one snapshot) and the
  MLP (vectors) cannot; but with ~1,670 train raises it can overfit, so a modest gain is the
  honest expectation. Plan: the 68 inputs + a 32-number compression of the sentence vectors per
  step, causal dilated convolutions (kernel 3, dilations 1-64), ~30-50k parameters, loss on
  counting moments only, epochs chosen on validation, 4 out-of-fold models, its score added to
  the trees (trees + MLP + TCN), plus a no-future test. Waiting for Brady's go.
- **Is a GRU different from a TCN? Which should do better?** Both read the class as a sequence.
  A GRU reads one step at a time with a running memory (in principle the whole class, but it
  fades; slower, harder to train on ~2,700 steps); a TCN looks at a fixed window all at once
  through stacked dilated filters (here 8.5 min; fast, stable). No guaranteed winner; Bai,
  Kolter & Koltun (2018) found generic TCNs match or beat recurrent nets on most tasks, and the
  whole-class memory a GRU adds is already in the inputs as running counts. Built the TCN; the
  repo's GRU (oncue/gru.py) can run later as a comparison row.
- **3.3 TCN built (18 Oct).** Reuses oncue/gru.py's tested causal step builders (sentence
  vector of the frames closing in each 2-s step, PCA-64 fitted on train; seconds of speech per
  role; the trees' numeric inputs + one-hot categories): 214 numbers per step, every grid step
  kept, loss on counting moments only. 7 causal dilated convolutions (kernel 3, dilations
  1-64, 32 channels), 46,113 parameters, receptive field 255 steps = 8.5 min (tested: nothing
  after t, nothing older than the window). Best epoch 7 of 15 (overfits after). Then GBT + MLP
  + TCN. Found and fixed a stacking mismatch: validation scores came from a fifth model on a
  different scale from the 4 out-of-fold models the trees learned on; validation is now scored
  by the same 4 fold models (averaged). Validation AP: best naive 0.0245, trees 0.0262, MLP
  0.0282, trees + MLP 0.0282, TCN 0.0299, trees + MLP + TCN 0.0310 (AP_w 0.069, ROC AUC 0.71,
  best on every ranking number); the TCN's score is the second input the stack relies on.
  Open: the threshold set on single-model OOF train scores does not carry over to the averaged
  validation scores (every flexible model says yes less often than the truth: 4.0-13.3 yes
  moments per class vs 20.7), so their warning counts undercount. Guess counting switched to
  warnings (one unbroken run of yes moments = one warning).
- **What is within-class AP vs normal AP? Why is it so much better within class, and is within
  class all we care about?** AP ranks all validation moments of all classes in one list (needs
  "when" and "which classes", on one scale); AP_w ranks each class's moments only against
  that class, then averages weighted by positives (only "when within this class"). Its random
  level is higher (a constant model gets each class's share of positives; weighted by
  positives that is 0.022 on validation, vs 0.015 pooled), but not only that: trees + MLP + TCN
  is 3.1x random within class (0.069) vs 2.0x pooled (0.031); professor-question rule 2.3x vs
  1.6x. Pooled ranking suffers when a model's scores drift between classes. AP_w best matches
  "in this class, is a raise coming?" (proposed as the headline ranking number, and a novel
  metric for the rubric), but pooled AP and the warnings still matter because the guess
  threshold is one number for all classes. Proposed: keep both, AP_w first, with "x random".
- **Within class means within B111 etc.? Isn't that just overfitting?** A class is one session
  (one recorded meeting, e.g. 87550), not a course. Not overfitting: AP_w is computed on the
  86 validation sessions the models never trained on, with one model for all sessions; it only
  scores the same predictions per session instead of pooled. Real limit: train, validation and
  test share the same 26 courses and the trees lean on ctx_course, so results describe new
  sessions of known courses, not a brand-new course; a leave-one-course-out check
  (splits.loco) would measure that (offered, not run).
- **What does AP_w mean? (still confused)** Per session: sort its moments by probability, take
  that session's AP (how near the top its real pre-raise moments landed); then average over
  sessions, weighted by their raises. Toy example: session A (scores run high) 0.9 no, 0.8 yes,
  0.7 no, 0.6 no -> AP 0.5; session B (scores run low) 0.4 yes, 0.3, 0.2, 0.1 no -> AP 1.0;
  AP_w = 0.75, but pooled AP = (1/2 + 2/5) / 2 = 0.45, punishing B's perfect timing only for
  its lower scores. Trees + MLP + TCN: AP_w 0.069 vs 0.022 random, about 3x chance.
- **So we can use AP_w reliably on test later?** Yes, with three conditions: fix it as the
  headline before looking at Part 3's test (choosing after would be cherry-picking); give it
  error bars, because 86 test sessions / 533 raises are few (question timing moved 0.049 ->
  0.063 from validation to test), via a bootstrap over sessions (and a paired version for
  "is model A really better than B?"); and report it with the warnings, since it ignores
  whether one threshold works across sessions. Proposed: add the session bootstrap (asked).
- **Does the current pipeline make sense? Is there an obviously better way or combination?**
  The structure is sound (causal inputs tested, baselines first, session-grouped out-of-fold
  stacking, validation-only choices, test held back) and the gains are real but small (AP
  0.0245 -> 0.031, AP_w 0.050 -> 0.069). The bottleneck is information and labels, not
  architecture: every flexible model starts memorising train within a few epochs (~1,670
  train raises). Ranked ideas: (1) my own video tile before the raise (causal frames; likely
  the strongest 2-5 s cue; new vision pipeline); (2) classmates' raises as extra training
  events (~20x labels; events OK, identities never stored; rescanning cost); (3) cheap fixes:
  threshold option 1 + calibrating the stack, averaging 5 training runs of TCN/MLP, a
  logistic-regression combiner vs the trees, session-bootstrap error bars; (4) the fine-tuned
  transformer (smaller expected gain, biggest cost). Checked: "I start speaking" as an extra
  target adds only 4,814 train positives (fewer than raises). Alternative combination to test:
  one TCN that also gets the MLP's alignment vectors per step. Recommended: (3) now, then
  Brady chooses between the transformer and own-tile video.
- **What was the outcome of the pilot?** The 20-class pilot (13 Oct, train only; QUALITY.md
  section 2): one clock per recording (audio and video start at 0 s, lengths match Forum within
  ±1 s); Forum timestamps calibrate with no drift (18 of 20 met the strict rule then); video,
  transcript and Forum agree (ring vs voice events ~0.06 s, vs transcript ~0.1 s); my tile never
  confused (0 wrong-tile frames, 0 classmates as me, ~41,000 frames); 118 raises found, re-timed
  to 1/24 s; 18 of 20 passed QA. The full run followed: 446 of 477 pass under the 2.5 s rule.
  data/pilot/ (193 MB) is still there, marked to be discarded; not deleted (Brady's call). The
  24 + 8 class "smoke runs" of Part 3 only checked that the code runs.
- **Cheap fixes + "just TCN + MLP, no trees" (18 Oct, Brady's go).** All Part 3 models retrained
  (python -m oncue flex --step all, 1,155 s): each validation session scored by ONE fold model
  (round-robin) so validation scores are made like the out-of-fold train scores; 5 training
  runs averaged per fold for the MLP and TCN (epochs chosen on the mean curve: 4 for the TCN);
  logistic combiners without trees; Platt calibration; a 1,000-resample session bootstrap.
  Validation AP_w / AP: trees 0.053 / 0.026, MLP 0.057 / 0.027, TCN 0.057 / 0.028, trees + MLP
  + TCN 0.058 / 0.027, logistic on all inputs 0.065 / 0.030, MLP + TCN without trees
  0.073 / 0.031 (3.3x random; 95% interval 0.058-0.090). Paired bootstrap, MLP + TCN vs:
  every naive and single model 99-100% of resamples, trees + MLP + TCN 97% (gain +0.015,
  -0.001 to +0.033), logistic on all inputs 89%. A plain average of the two scores gives the
  same (0.074), and the two scores correlate only 0.53, so the gain is the specialists'
  complementarity, not the combiner. Out-of-fold train AP_w shows the same order (MLP 0.054,
  TCN 0.063, both 0.068).
- **So you are removing the trees?** No: the trees stay as comparison rows (part of the story
  and the comparison table); the proposed final pipeline is MLP + TCN without trees (best and
  simplest); Brady's call.
- **Can we train with AP_w?** Measuring it on (out-of-fold) train: yes (numbers above).
  Optimising it directly: no, because it depends only on the order of moments, so it is flat
  almost everywhere and has no gradient for the networks or the trees to follow. What works: a
  smooth stand-in, a within-session ranking loss (penalise every ordinary moment that outscores
  a pre-raise moment of the same session), and/or choosing settings by AP_w. Offered (~1 h).
- **Final test with AP and AP_w? Which is more legit? Can we set everything by AP and still look
  at AP_w?** Both are legitimate; what matters is fixing before test which is the headline and
  how the final model is picked. Settings were chosen by AP on validation; the final model is
  the same by either (MLP + TCN). Proposed: report both on test with bootstrap intervals,
  AP_w as the headline, declared before scoring.
- **Would a Temporal Fusion Transformer (attention) be better than the TCN?** Probably not with
  this data: attention models are data-hungry, the 46k-parameter TCN already overfits after
  4-7 epochs on ~1,670 train raises, and TFT's strengths (multi-horizon continuous forecasts,
  static covariates, variable selection) matter less for a rare yes/no event whose cues land
  seconds before it. Cheaper to try first: one small attention layer on top of the TCN, and a
  multi-horizon head (2-5, 5-10, 10-15 s); a small TFT as a comparison row (Lim et al., 2021).
- **Move all equations into the assignment doc; notebook = overview, code, runs.** Done 18 Oct:
  in the Google Doc, the old draft model bullets ((1)-(4) and the rules) were replaced by
  Models / Naive models (N1-N5) / Flexible models (F1-F6) / Metrics (M1-M6), each a bold
  heading, 3-4 sentences with results from the runs, and the LaTeX ($$...$$); the template
  reference placeholders were replaced by 26 APA references (alphabetical, hanging indent).
  The notebook's markdown now carries no equations, only pointers to those sections. Source:
  scratchpad doc/make_doc.py.
- **DECLARED BEFORE SCORING TEST (18 Oct 2026, 21:51), Brady: "yes thats perfect".** Final
  pipeline: MLP + TCN without trees (a logistic regression on the two specialists' scores, C =
  0.01), chosen on validation. Test headline: within-session AP (AP_w) with its 95% session-
  bootstrap interval; AP alongside; warnings at the threshold fixed on the out-of-fold train
  scores. Test is scored once, with every model, setting and threshold frozen.
- **TEST, scored once (18 Oct, after the declaration).** `python -m oncue flex --step test`
  (156 s + 84 s bootstrap): the fold models retrained with the same settings and seeds
  reproduce validation exactly (AP_w 0.0734 both, correlation 1.000). Final pipeline (MLP +
  TCN, no trees) on the 86 test sessions: **AP_w 0.076 (3.0x random 0.025; 95% interval
  0.059-0.096)**, AP 0.035 (2.3x random 0.015; 0.027-0.047); validation was 0.073 / 0.031, so
  it holds up. Paired bootstrap (AP_w / AP): beats the professor-question rule, Naive Bayes and
  the MLP alone in 100% of resamples; question timing and TF-IDF in 93% / 100%; the TCN alone
  only 80% / 85% (TCN alone on test: AP_w 0.071). Warnings: 6.4 per session, 6.7% right
  (random one-moment guess 1.5%), 35 of 533 raises warned (a perfect model: 322), 17 with a
  guess 2-5 s ahead. Notebook section 3.6.
- **An example timeline at the end of the notebook (test: where the predictions happened and
  my raises).** Section 3.7, Figure 3.15: one test session start to end (probability,
  threshold, warnings green = right / red = wrong, my raises) plus a 5-minute zoom around one
  warned raise. Session chosen by a fixed rule so it is not flattering: the median
  within-session AP among the 45 test sessions with >= 5 raises (86380: 15 raises, 9 warnings,
  2 right, 2 raises warned; in the zoom a wrong warning ~35 s early, then a right one 4 s before
  the raise). The session with the most raises (100374) was tried first and ranked 16th of 45
  (34 warnings, 1 right, 1 of 20 raises warned), so the median rule is more representative.
- **"The probability rises steeply before almost every raise, just not enough to hit the
  threshold. Is that a strong sign? What adds the most weight to the probability?"** Yes, on
  test: before 62% of raises the final pipeline's probability is higher 4 s before than 40 s
  before (46% for random moments; median odds rise x1.12 against x0.97), but only 5% of raises
  are above the threshold 4 s before. A tree surrogate of the final scores (R² 0.85) ranks what
  moves it: talk in the last 60 s (Spearman +0.39), course, time into the segment (+0.34), time
  since the professor's last question (-0.36), current silence (-0.48); CS156 / CS114, intros and
  debriefs highest. The combiner weighs the TCN about twice the MLP (0.77 vs 0.37 per SD).
  Saved: data/results/flex/explain.pkl.
- **Write-up doc: graphs, then "no, remove them; just leave the write-up good, I will place the
  graphs".** The pasted figures and the caps notes were removed again; the doc follows the
  11-item structure with text for every item (pipeline, data and ingestion, code and notebook,
  models, metrics, learning, final performance, comparisons with a table, discussion,
  references). The final review is Brady's.
- **"Figures characterizing the model's learning: can we get a visualization of the learning
  process (of the final model)?"** New step `python -m oncue flex --step learning` (validation
  only, 556 s): the MLP and TCN fold models retrained exactly as in the final run (same folds,
  settings, seeds) for 15 epochs, scoring validation after every epoch, combined as the final
  pipeline; epoch 4 reproduces the final pipeline exactly (AP_w 0.0734, correlation 1.000).
  Train loss falls every epoch (TCN 0.076 -> 0.050, MLP 0.075 -> 0.066); validation loss turns up
  after 2-3 epochs (TCN 0.0755 -> 0.094). Pipeline validation AP_w: 0.064 after 1 epoch, a
  plateau 0.073-0.074 at epochs 3-5 (0.0742 at 5, within noise of the 4 used, which were fixed
  before from each specialist's own curve and are not changed after test), 0.062 at 15. The
  pre-raise build-up (probability / normal level, 40 s -> 4 s before a raise): 1.30 -> 1.41
  after 1 epoch, 1.32 -> 1.53 at 4, 1.25 -> 1.40 at 15. Notebook section 3.6, Figures 3.15-3.16.
- **Notebook additions (18 Oct):** Figure 1.1 (one raise frame by frame, my tile only,
  classmates pixelated), Figure 2.6 (naive models compared, best by AP_w highlighted: TF-IDF +
  LR), Figure 3.14 (all models compared on validation), 3.6 learning, Figure 3.18 (test
  comparison); test is now 3.7 and the example session 3.8. Colab: the setup cell clones the
  repo and downloads the private data zip from Drive (shared only with pwatson@minerva.edu).
- **"Explain the labels paragraph more: splitting the top of the page, the grey view, Tesseract
  for the name, the green hue, lining the video up with the transcript, and how everything stays
  centred."** The doc's data section now has two paragraphs (from oncue/strip.py and
  ingest_qa.py): the scan reads only the top of each 1920 x 1080 frame, finds the strip under
  Forum's dark header and splits it into equal tiles, re-measured every frame (the strip
  reflows); my tile is found by reading each white name label with Tesseract OCR (black on
  white, matched to the roster, fixes for misreads like "Bracly"), then recognised by its
  pixels; the icon is a yellow-green disc (OpenCV hue 26-40) with a white hand in the top-left
  corner, UP at >= 10% of that corner plus the shape test, DOWN below 4%; every onset is re-read
  at full frame rate and re-timed to 1/24 s, my tile re-found by its label on each frame; audit
  sheets use the grey view (all grey except the teal ring and the green icon, other tiles
  pixelated). Sync: the transcript comes from the same recording; Forum events are calibrated
  onto it in windows (p95 <= 2.5 s); the teal speaker ring on my tile must start with my words
  (about 0.1 s in the pilot), and a lag more than 0.5 s off the median drops the session. So
  everything is centred on one clock (the recording's frame times) and the read stays on my tile.
- **Data on Drive (18 Oct): "here it is, make Colab and the notebook use it."** The zip
  (pipeline1_data_private_v2.zip, Drive file id 1KXS3xJFMr6GxeaWPDUwL-4xhBm3dw67d, in the school
  Drive folder "CS156 Pipeline 1 data") is shared only with pwatson@minerva.edu as Viewer, with
  the email notice; general access stays Restricted. The notebook's setup cell has that id and an
  "Open in Colab" link: in Colab it clones the repo, signs in, downloads and unzips the data.
  Tested by running the notebook from a fresh clone of main with only the zip as data and the
  transcriber package blocked (as in Colab): no errors, the same 28 figures. That test found
  that the uploaded zip lacks the Forum event tables (voice, chat, polls, segments; 8.8 MB,
  speaker ids hashed, text masked) that section 1.4 shows; the fixed zip is in ~/Downloads (same
  name) for Brady to upload as a new version of the same Drive file, and 1.4 now skips cleanly
  if they are missing.
- **"Revert it, keep the image recognition, don't overexplain the transcribing, a bit
  shorter."** The data section is back to two paragraphs (373 words): the label paragraph keeps
  the strip at the top of the frame, Tesseract on the name label, the yellow-green hue, the
  re-timing and the grey audit view; the sync is one sentence (one clock, the recording's frame
  times), followed by the original workbook / frames / grid / split sentences.
