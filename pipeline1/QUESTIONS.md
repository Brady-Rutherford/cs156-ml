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

