# Data quality: what the pilot proved, every metric, and what needs approval

*13 Oct 2026, after the 20-class pilot (train split only, `data/pilot/`, to be discarded).*

## 1. Where the pipeline stands

| Step | Status |
|---|---|
| 1 Get data | done: 739 classes synced from Forum |
| 2 Clean | done: 477 usable classes |
| 3 Split | done: 289 train / 94 validation / 94 test, per course |
| 4 Words + events | built and **piloted on 20 classes**: Deepgram transcripts, the words table, Forum events, one clock |
| 5 Hand raises | **rebuilt for the real Forum UI** (`oncue/strip.py`) and piloted on 20 classes |
| 6 Moments (labels on the 2 s grid) | built, tested on synthetic data, not run on real data |
| 7 Snapshots (features) | same |
| 8 Train | same |
| 9 Score | same |

Next: run steps 4 and 5 on all 477 classes (train → validation → test), then 6–9.

## 2. What the pilot proved

1. **Every recording is one clock.** Audio and video both start at 0.000 s in all 20 classes, every
   class is a single recording session, and the video's length equals Forum's session length
   within ±1 s. Nothing is cut out of or spliced into the video.
2. **Forum's timestamps can be put on that clock.** All 20 classes calibrate with the proper
   windowed fit and no drift; 18 of 20 meet the strict residual rule.
3. **The video, the transcript and Forum agree.** Brady's speaking ring in the video lands within
   about 0.06 s of his own Forum voice events (median), and within 0.1 s of the transcript's
   speech (all speakers).
4. **Brady's tile is found reliably and never confused.** In every class: 0 frames with the
   wrong tile marked as his, and 0 classmates marked as him (about 41,000 frames of
   classmates speaking alone checked).
5. **Raises are detected and timed to the frame.** 118 raises (59 called), every one
   re-timed to 1/24 s; spot checks by eye land on the raise animation.
6. **QA: 18 of 20 classes pass** (the 2 failures are 0.02 s and 0.18 s over the strict clock rule).

## 3. Every quality metric, in plain words

### A. Transcript ↔ Forum clock (`class_timing`)

Forum logs each student microphone switching on and off. The transcript has every word's
time. Calibration slides Forum's times until mic activity and speech line up.

| Metric | Meaning | Rule | Pilot |
|---|---|---|---|
| `cal_offset_s` | how many seconds Forum's clock must be shifted to land on the video | none (it is measured, then applied) | −0.4 to +2.8 s; +0.6 to +1.8 s for 2023–24 classes, whose Forum times are whole seconds |
| `cal_drift` | whether the two clocks run at different speeds (seconds of shift gained per second of class) | ≈ 0 | 0.000 in all classes |
| `cal_method`, `cal_n_windows` | the fit is made in 5-minute windows through the class (not one global guess) | windowed, ≥ 6 windows | windowed, 6–37 windows |
| `resid_med_s` | after the shift: the typical distance between a Forum mic-on and the nearest transcribed word | ≤ 0.5 s | 0.14–0.44 s |
| `resid_p95_s` | the same, for the worst 5% | ≤ 1.5 s (sensitivity: ≤ 3 s) | 0.79–1.68 s; 2 classes over |
| `timing_precision` | Forum timestamps in milliseconds (2025–26) or whole seconds (2023–24) | — | — |

### B. Video ↔ transcript and video ↔ Forum (`hand_qc`, `ingest_report`)

The teal speaking ring is drawn in the video when someone's mic carries sound. If the
clocks agree, the ring appears when the transcript says someone is talking.

| Metric | Meaning | Rule | Pilot |
|---|---|---|---|
| `av_lag_s` | how far the ring (any tile) lags the transcript's speech | within ±0.5 s of the median over all classes; that median must be in [−0.25, 1.0] s | median +0.08 s (−0.13 to +0.30) |
| `av_lag_brady_s` | Brady's ring vs his own transcribed words (needs ≥ 20 s of his speech) | diagnostic | about −0.3 s: his mic opens slightly before Deepgram's first word |
| `ring_vs_forum_s` | Brady's ring vs his own Forum voice events, after calibration | diagnostic | median −0.06 s, mostly ±0.1 s |

### C. The cut of the video

| Metric | Meaning | Pilot |
|---|---|---|
| `video_minus_forum_s` | video length − Forum recording-session length | −0.6 to +1.0 s: nothing trimmed |
| `video_minus_asr_s` | video length − length of the audio Deepgram transcribed | 0.25–0.31 s in every class (the end of the file; the start is checked by B) |

### D. Finding Brady's tile (`hand_tiles`, scan record)

| Metric | Meaning | Rule | Pilot |
|---|---|---|---|
| `band_share` ("strip found") | share of all frames where the top strip of tiles is on screen | — | 61–88%: no strip during breakouts (left layout), before class and after it ends |
| `coverage` | share of frames outside breakouts where Brady's tile is identified | ≥ 30% | 72–95% |
| `track_valid_share` ("ring check") | while Brady speaks alone, the share of frames where the ringed tile is the one identified as his | ≥ 90% (with ≥ 3 such speech events) | 91–100% (B111 now 100%) |
| wrong tile / classmate as Brady | pilot diagnostics: identified tile ≠ ringed tile during his solo speech; a classmate's solo-speech ring on a tile marked as his | 0 | 0 and 0 in every class |
| `n_ocr` | OCR reads needed (cost) | — | 200–2,600 per class |

### E. Hand raises (`hand_episodes`)

| Metric | Meaning | Pilot |
|---|---|---|
| episode types | `bid` = Brady's raise; `vote` = class-wide (≥ 4 others, or 30%, raise within −5…+10 s, or the professor calls a vote); `glitch` = a blip < 1 s | 118 bids, 4 votes, 5 glitches |
| `left_censored` | the raise's start was not visible (no 2 s of clean DOWN before it); excluded from the labels | 64 of 185 episodes (35%); mostly fragments of long raises and the end of class |
| `dur_s` ("up for") | how long the hand stayed up | median 14 s; 10%–90%: 4–88 s |
| `outcome` | `called` (Brady's next turn starts before the hand comes down + 5 s), `withdrawn`, `open` | 59 called (50%), 36 withdrawn, 23 open |
| `call_latency_s` ("called after") | raise → called | — |
| retime shift | coarse onset (2 fps) − re-timed onset (24 fps) | median 0.21 s, 90th percentile 1.0 s; 0 retime failures |
| raise → speech gap | when called: raise → Brady starts speaking; a gap that drifts within a class would mean a clock problem | median 2.3–6.9 s; no drift |
| known offset | the raise animation starts with a white hand fading in ~0.17 s before the green reaches the corner, so onsets are ~0.17 s late | constant |

### F. The class verdict (`qa_pass`)

A class enters the dataset only if: it has a transcript of the `lowres` rendition; its tile is
`located` (strip found, coverage ≥ 30%, ring check ≥ 90%); its A/V lag is within ±0.5 s of the
median; and its calibration meets rule A. Any failure gives a reason (`resid_p95`,
`tile_tracking_failed`, …) and the class is left out (under the sensitivity rule, classes with
residuals ≤ 3 s are kept).

## 4. Decisions to approve

1. **The QA thresholds** in §3 (pre-registered in the spec): residual p95 ≤ 1.5 s / median
   ≤ 0.5 s; ring check ≥ 90%; coverage ≥ 30%; A/V lag ±0.5 s.
2. **The detector settings measured on real frames:** hand hue 26–40, UP at ≥ 10% of the corner
   square (34% of the tile height) with the round-hand shape, DOWN below 4%; raise overlay = both
   corner squares ≥ 90% hand hue; 2 samples to confirm a change; unreadable gaps ≤ 3 s bridged;
   an onset through a ≤ 3 s hidden-label gap is kept.
3. **Identification:** label NCC ≥ 0.80 (same width) or ≥ 0.88 (any width), beating every
   other tile by 0.15; OCR name match ≥ 0.60 with a 0.12 margin; B111 needs every new label of
   Brady's confirmed.
4. **The 0.17 s onset offset:** leave it (it is constant and small next to the 2 s guard), or
   subtract it, or detect the white hand.
5. **Left-censored raises are excluded** from the labels (their start is unknown).
6. **Breakouts are excluded** (there is no main-room strip, and Brady does not raise there).

## 5. Speed: this Mac or rented machines (e.g. Prime Intellect)

| Part | Bound by | Here | On a big CPU machine |
|---|---|---|---|
| transcription | Deepgram (remote) + download/upload bandwidth | ~3–4 h for 477 classes | faster network (datacenter link); about 1 h |
| hand scan | CPU: software H.264 decoding + Python per frame (~5 ms) + OCR | ~34 classes/hour with 6 workers (~14 h) | scales with cores: a 64-core box runs ~30 classes at once (~2 h) |
| GPU | could decode video (NVDEC), but the Python part stays on the CPU | — | helps little; more CPU cores help most |

Moving it needs the Forum cookie and the Deepgram key on the rented machine, and classmates'
audio and video would be processed there (in memory, not stored). That is a privacy and
security decision to make deliberately.

## 6. Decisions taken and the full run (14-15 Oct)

- **Approved by Brady:** clock limits 2.5 s (max residual and p95, was 1.5 s; median <= 0.5 s
  stays); ring check over the frames where his tile is read (`oncue ringcheck`); coverage >= 30%;
  drop no class for breakouts or unread stretches; vote = the professor calls one or 6+ others
  raise within [-5, +10] s; left-censored raises stay out of the labels (84% of them are pieces
  of a raise already counted; a "single DOWN reading" rule would add ~1% and was not adopted).
- **Still open:** the 0.17 s onset offset (leave, subtract, or detect the white hand).
- **Result, all 477 classes:** 446 pass QA (train 274/289, validation 86/94, test 86/94).
  Failures: too few calibration windows 6, A/V lag 6, residual median 5, tile unlocated 4 (with
  the 3 silent recordings: 7), no calibration 3, max residual 2, wrong tile 2.
- **Raises:** 2,774 with a known start (train 1,674, validation 541, test 559), all re-timed
  to the frame, 1,295 called; 2,674 in classes that pass QA. Per-class checks:
  `data/logs/ingest_report.csv`.
