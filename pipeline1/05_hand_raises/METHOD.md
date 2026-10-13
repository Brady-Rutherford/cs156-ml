# How hand raises are found in the video, and how everything gets onto one clock

*Written during the 20-class pilot, 13 Oct 2026. Numbers in this file come from that pilot
(train split only; `data/pilot/`, discarded afterwards).*

## 1. One clock: the recording's own time

Everything is put on one clock: **seconds since the first audio sample of the class
recording**. Forum serves each recording as two renditions. We always use `lowres`
(1920 × 1080, 600 kbps) for both the transcript and the video scan, so both read the same
file.

In all 20 pilot classes, `ffprobe` gives an audio start of 0.000 s and a video start of
0.000 s. Every class is a single recording session, and the video's length agrees with
Forum's session length to within −0.6 / +1.0 s, so nothing is cut out of or spliced into the
recording.

| Source | How it gets onto the clock | Check |
|---|---|---|
| transcript words | Deepgram transcribes the audio of that same file, so word times are already on the clock | the ASR audio is 0.25–0.31 s shorter than the video in every class (the end of the container); the A/V lag check below shows whether the *start* is shifted |
| Forum events (voice, chat, polls, segments) | wall-clock ms → seconds into the recording session, then **calibrated**: Forum's student-mic activity and the transcript's speech are each turned into a 10 ms "someone is talking" signal and cross-correlated in 5-minute windows; a line `offset + drift · t` is fitted through the windows (the professor's voice cluster is left out, because Forum never logs the professor's mic, A12) | all 20 classes: windowed fit, drift ≈ 0, offset −0.4 to +2.8 s, residual median 0.14–0.44 s; 18 of 20 pass the p95 ≤ 1.5 s rule |
| video frames | `ffmpeg -copyts` with the `showinfo` filter prints each decoded frame's presentation time; frame time = that time − the audio start (0) | `fps=2:round=up` means a 2 fps sample never shows content from after its time stamp |

Whole-second Forum time stamps (2023–24) shift the calibrated offset by about +0.5 s on
average. The fit absorbs that, and their residuals stay below 1.3 s at p95.

## 2. Reading the strip

The recording is a screen capture of Forum. Brady's tile always appears in the thin strip
of participant tiles at the top, also while he is on the main stage, so only the strip is
read. The main stage is never looked at.

For every frame (2 per second; the video is 24 fps):

1. **Find the strip.** It is the topmost band of rows (from y ≈ 50, under Forum's header)
   that is mostly bright and splits into ≥ 2 equal tiles of a tile's shape (height/width
   0.55–1.0). The strip reflows when people join or leave (tiles 69–200 px wide), so this is
   done on every frame.
2. **Split it into tiles.** Gutters are the dark 4–5 px columns between tiles. On keyframes
   (every 2 s) the encoder bleeds the tiles into the gutters (brightness 0.02 → 0.18), so a
   gutter is also any narrow valley at most half as bright as the tiles on both sides. A frame
   that only *lost* tiles keeps the previous frame's tile list: a real reflow moves or resizes
   tiles, while a smeared gutter only removes them.
3. **Find Brady's tile by his name.** Each tile shows the person's name in white over the
   video. The white text (saturation < 40, value > 225) is cut out, scaled and read with
   Tesseract OCR. The text is corrected for Tesseract's usual confusions ("cl" → "d") and
   matched against the class roster: the closest name wins if its score is ≥ 0.60 and beats
   every other person by ≥ 0.12. Each label read is also stored as a 16 × 96 pixel patch, so the
   same label at the same tile width is recognised later by cross-correlation without OCR.
   Classmates' names stay in memory only; what is stored is just "this tile is Brady's".
   Two tiles both read as Brady make the frame unreadable (NA), never a guess. A label is
   matched first against his labels at the same tile width (NCC ≥ 0.80), then at any width
   (≥ 0.88, beating every other tile by ≥ 0.15). This covers reactions that tint the label and
   the strip resizing as people leave. Forum also *reorders* the strip (a classmate who
   raises can move into his old slot), so a position alone never identifies him.
4. **Check the tracking with the speaker ring.** The teal ring (hue 75–77) means "speaking",
   never a raise. While Brady speaks alone (his own Forum voice events, ≥ 2 s, no one else
   overlapping), the one bordered tile must be the tile his label marks. Pilot: wrong tile 0
   times; a classmate's bordered tile marked as Brady's 0 times in 3,983 frames; ring check
   98–100% per class.

## 3. The label: the yellow-green hand

In Brady's tile, the label test reads the **top-left corner square** (34% of the tile height,
3 px in from the edge, so the ring never enters it):

- **the icon:** at least 10% of the square in hand hue (OpenCV hue 26–40, saturation and
  value > 115), forming a round disc (circularity ≥ 0.6) with a white hand inside (8–45% white).
  Tiles without an icon read 0 at the 99th percentile. Other icons are *reactions* and are
  never counted: the orange one sits at hue 10–19;
- **the raise animation:** pressing the raise key first fills the **whole tile** with yellow-
  green and a big white hand for about 1.7 s, and only then shrinks it to the corner icon. Its
  first frame is the moment of the raise (class 79900: overlay at 553.54 s, icon at 555.29 s).
  The overlay can hide the name label (2023–24 UI), so the overlay tile at Brady's last read
  position (±3 px, read ≤ 3 s earlier, earlier frames only) is taken as his, unless its label
  is read as someone else's. The overlay counts as a hand.

The per-sample states then go through the existing state machine (`hands.track`,
`hands.episodes`):

- UP when the hand test passes, DOWN when the hand share falls below 4%, otherwise the
  previous state (hysteresis); a change is confirmed by 2 agreeing samples (1 s);
- unreadable (NA) runs of up to 3 s with the same state on both sides are bridged. Reactions
  (full-tile emoji animations) and the raise overlay hide the label for ~1.5–2 s, so the 1 s
  bridge of the synthetic design split long raises into pieces;
- an episode needs ≥ 2 s of confirmed DOWN before it. When the onset is reached through an NA
  run of at most 3 s (the overlay hiding the label), with those 2 s of DOWN before the run,
  the episode is kept and its onset put at the start of the run; re-timing then finds the
  first frame. Any other episode that starts from NA is "left-censored", and its start time is
  not trusted;
- class-wide votes (≥ 4, or 30% of the others, raising within [−5, +10] s, or the professor
  asking for a vote) are typed `vote`, not a bid;
- outcome: `called` when Brady's next main-room turn starts before the hand comes down + 5 s,
  `withdrawn` or `open` otherwise.

**Re-timing.** Each onset is re-read at the full 24 fps over [onset − 1.0, onset + 3.5] s. The onset becomes the first frame on which Brady's tile turns yellow-green
in the corner and either the icon passes the shape test or the overlay is spreading over the
tile, followed by ≥ 3 more such frames. Resolution: one frame, 42 ms.

## 4. Checks that the clocks line up

| Check | What is compared | Pilot |
|---|---|---|
| A. transcript ↔ Forum | calibration fit and residuals (above) | 20/20 windowed; 18/20 pass p95 |
| B. video ↔ transcript | the ring on the recording against speech in the transcript (`av_lag_s`), and Brady's ring against his words (`av_lag_brady_s`) | all rings: median +0.08 s (−0.13 to +0.30); Brady's ring leads his ASR words by ~0.3 s (his mic opens before Deepgram's first word) |
| B'. video ↔ Forum | Brady's ring against his own Forum voice events (`ring_vs_forum_s`) | median −0.06 s, mostly within ±0.1 s (−0.50 to +0.18) |
| C. the video's cut | video length vs Forum session vs ASR audio | within ±1 s; ASR 0.25–0.31 s shorter in every class |
| D. raise → speech | gap from each called raise's onset to Brady's next turn, and its trend over the class (drift = clock problem) | median 2.3–6.9 s in most classes; trends are noise (4–10 raises per class; calibration drift ≈ 0) |
| E. ASR mode | URL vs upload mode offsets | no systematic difference |
| audit | a 6 s grey-view clip per raise (everything light grey except teal and green, all other tiles pixelated) and a frame sheet ±0.5 s | `audit/{class}/` |

## 5. How the video is read

Nothing is downloaded for the scan. Forum's API gives a signed CloudFront URL (Scribe's
`ForumClient`, Brady's cookie). `ffmpeg` streams it over HTTPS, decodes it in software (33×
real time, faster here than Apple's hardware decoder), keeps 2 frames per second, crops the
top 360 rows and pipes raw pixels to Python. numpy and OpenCV measure each frame, which is
then dropped. A URL that expires mid-class is replaced and the stream resumes 2 s back.
Transcription downloads the same file once into a temporary folder, extracts a 16 kHz mono
Opus track (~20 MB), deletes the video and uploads the audio to Deepgram's EU endpoint.

## 6. Known limits

- During breakouts there is no main-room strip (94–100% of breakout frames): no label there,
  which matches the breakout exclusion rule.
- One pilot class (103152) shows the strip down the left side; the scanner reads no top strip
  there and those frames are NA.
- A raise that starts while Brady's tile is unread for > 3 s cannot be carried through the
  overlay. It starts at the corner icon (~1.7 s late) or is left-censored.
- OCR occasionally fails to read a label; those frames are NA, never DOWN.
- Re-timed onsets are about 0.17 s late: the raise animation starts with the big white hand
  fading in (class 104899: from 806.375 s) before the yellow-green reaches the corner square
  (806.542 s). This is a constant shift, small next to g = 2 s; detecting the white hand would
  remove it.
- One pilot class (89838) leaves my tile unread in 41% of frames, because another label there
  scores close to mine and the margin rule refuses to guess. Its ring check (61%) fails the 90%
  bar, so QA marks it `tile_tracking_failed` and it is excluded.

## 7. Pilot result (20 train classes)

- 116 raises (58 called), every one re-timed to a frame; 84 episodes left-censored (28 of them
  in 89838, which fails QA).
- Identification: 0 wrong tiles and 0 classmates marked as Brady in every class; his tile is
  read on 59–99% of the frames that show the strip.
- QA: 17 of 20 classes pass. 83649 and 95228 fail the calibration residual rule (p95 1.68 s and
  1.52 s), 89838 fails tracking.

