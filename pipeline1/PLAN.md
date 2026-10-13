# On Cue: plan

> **Can a model tell, from what is being said in class, that I am about to raise my hand?**

In theatre, the *cue* is the line that tells you it is your turn. Minerva records every class as a
screen capture of Forum, so the moment my raised-hand icon appears is in the video, and three years
of class speech, timeline segments, polls and my own pre-class work sit next to it. This pipeline
asks whether the class itself announces my turn before I take it.

| | |
|---|---|
| **Course** | CS156 First Pipeline · Brady Rutherford · Prof. Watson |
| **Due** | Fri 16 Oct 2026. Closes Sun 18 Oct; no late work after Mon 19 Oct 01:00 |
| **Status** | Design frozen Mon 12 Oct. Package skeleton, constants, table schemas and a synthetic test world are in place; the stages are being built against synthetic data. No results exist yet |
| **Full design** | [`design/on_cue_spec.md`](design/on_cue_spec.md) (authoritative; "spec §n" below points into it) |
| **Overrides** | [`design/amendments.md`](design/amendments.md) (A1–A9 win wherever they conflict with the spec) |
| **Evidence** | [`reports/Hand raise prediction model options.md`](<reports/Hand raise prediction model options.md>) and [`research_notes/`](research_notes/) |
| **Presentation** | [`site/index.html`](site/index.html): pipeline explorer, causal snapshot, label dials |
| **Archived** | [`later/the_tell_face.md`](later/the_tell_face.md), the earlier face-only plan (now P2) |

A readable summary for Brady and Prof. Watson; where it disagrees with spec + amendments, they win.

## 1. The question

Using only what I could have **heard** in class, **seen** in Forum, or **written before class**, up
to second *t*: can a model predict that I will raise my hand in the next *H* = 15 seconds, at least
*g* seconds before the icon appears? And does *what is being said*, including how closely it
matches my own pre-class work, add anything **within a class** beyond the *structure* of the class
(minute, activity, turn-taking, my own rhythm)?

| Sub-question | Answered by |
|---|---|
| (a) Content vs structure | within-class ΔAP_w(M3 on F-all − M3 on F-struct), and the same pair without my detector-derived history (H1) |
| (b) The moment vs the phase vs the class | local timing gain TG_local; the global TG is the "which class" floor (H2) |
| (c) Preparation | H3 on classes with a verified pre-class workbook; the A-mismatch control |
| (d) Beyond the obvious ("he raises when the professor asks") | M3 on F-all vs the rule baseline R (H4) |
| (e) Lead | the H-sweep, the guard dial *g*, the information-horizon curve, event lead times |

**Why this is not obvious.** No student API exposes hand raises (§2.1), so the label comes from
the recording. P1 is a text-and-context classifier whose only computer vision is that label
detector. It never sees faces, prosody, other students' hands, or the professor reacting to my
hand, and each exclusion is **measured** by a control (§10). P2 adds vision of my own tile; P3
predicts *what* I say from the same causal view.

## 2. What the data actually is (verified 12 Oct 2026)

Every endpoint was called live from my logged-in Forum session (cookie auth, no API key); the last
three rows were confirmed on 12 Oct through a browser tab running in that session (A6).

| What | Endpoint | What came back | Used in On Cue as |
|---|---|---|---|
| Me | `GET /api/v1/users/self?show_inactive=false` | my user id, name, role | picks out *my* rows; never hard-coded |
| Current sections | `GET /api/v1/sections?hide-defaults=true` | 6 sections (Fall 2026) | selection |
| Past sections | `GET /api/v1/sections?state=past&hide-defaults=true` | 47 sections, Fall 2023 → Spring 2026; 46 flagged `should-record-sessions` | selection |
| Classes in a section | `GET /api/v1/paginated-classes?category=past&page_size=50&filter_section_id={sid}` | `{count, next, results}`; each class has `recording-sessions` | `ctx_session_no` from the full schedule (D1 confirms unrecorded classes are listed) |
| Class record | `GET /api/v1/class_grader/classes/{cid}` | the class record (Scribe's `class.json`): roster, teacher, recording sessions (roster field names pinned on D1) | roles, attendance so far, clock; names only for in-memory masking |
| Class events | `GET /api/v1/class_grader/classes/{cid}/class-events` | `voice`, `message`, `timeline-segment`, `bookmark-note` (≈ 190 voice events per class) | **the only source of event tables**, type allowlist {voice, message, timeline-segment} |
| Recording timing | `recording-sessions[]` on each class | `recording-started`, `recording-ended` (UTC) | clock and `rec_gap`; `recording-ended` is label-side only |
| Video | `GET /api/v1/recording-sessions/{cid}/generate_recording_urls` | `{highres: …/recording.mp4, lowres: …/zencoder-recording-600k.mp4}`, CloudFront-signed, expiring | the hand scan, on the same rendition the ASR used |
| Breakouts | `GET /api/v1/class_grader/classes/{cid}/breakouts` | breakout groups | my own group only |
| Polls | `…/classes/{cid}/polls`, `…/poll-responses` | prompts; my written answers | poll features (shapes checked on D9) |
| Class edit page | the `class_edit_page` JSON | `preparatory-assignment.description`, `assessment.focused-outcomes[].learning-outcome.description` | learning outcomes and PCW instructions, known before class |
| PCW link | `POST /api/v1/pre_class_work/classes/{cid}/pre_class_resource_signed_url` | a signed link into sle-collaboration | the one allowlisted POST (side effects checked on D8) |
| PCW template and doc | sle-collaboration `GET /api/v1/templates/{uuid}/`, `GET /api/v1/doc/{uuid}/` | blocks with Quill `ops`; my answers | PCW prompt and answer features |

**Size of the archive:** 694 recorded class sessions across my 53 sections. **A voice event**
(fields verified; values from one of my own turns, 8 Oct 2026):

```json
{ "event-type": "voice", "actor-id": "<me>", "room-id": "<main room>",
  "start-time": "2026-10-08T15:06:23Z", "end-time": "2026-10-08T15:07:16Z",
  "timestamp": 1791472036041, "event-data": { "duration": 53000 } }
```

Details the code has to get right:

1. **For `voice`, `timestamp` is the *end* of the utterance**, in epoch ms; `start-time` and
   `end-time` are whole seconds. Precise onset = `timestamp − duration`. For `message` and
   `timeline-segment`, `timestamp` is the start. 2023 classes carry whole-second timestamps
   (`…000`), so their Forum margin ε_F is at least 1 s.
2. **Two key spellings.** A direct call returns kebab-case (`actor-id`); Forum's web app gets
   snake_case from the same URL. Loaders accept both (`scribe_core.forum_events.kebab()`).
3. **Video time ≠ event time − `recording-started`, until measured.** The arithmetic matched
   Forum's own player on one turn (0:07:18), but the audio clock is calibrated per class by Scribe
   and checked by QA (spec §4.3).
4. **Voice events are fragmented.** My durations: p10 0.33 s, median 2.9 s, p90 30 s. Turns merge
   at gaps < 2 s; a *fresh* onset needs ≥ 30 s of my silence.
5. **`room-signature == "Main Classroom"` marks the main room**, the only room recorded.
   `recording-state` is `NOREC` on all 694 sessions, including ones with video, so it is ignored.
6. **Scribe's calibration has a silent fallback** *(verified in code)*. With fewer than 3 usable
   windows, `calibrate()` fits the whole class and still returns `status "ok"`, `method "global"`
   and `max_residual_s = 0.0`. QA therefore requires `method == "windowed"` with ≥ 6 windows.
7. **Scribe clamps and fills segments** *(verified in code)*. Its `timeline_segments` moves a
   segment in a recording gap to the restart and fills `end` from the next segment. On Cue builds
   segments from raw events instead, with no end column.

Quirks: unfiltered `paginated-classes` returns **11,984 classes org-wide** and ignores `section=`;
only `filter_section_id` works. `/api/v1/analytics/class/{cid}` is 403 for a student. My speaking
rate varies by course (26 voice events per class in F23 Complex Systems, 39 in a F24 business
course, 7 in CS156 F26, and **0** in one Spring 2024 section that D1 must explain).

### 2.1 Where hand raises go (from Forum's own frontend code)

- `h` is bound to `raiseHand` → `currentClassUser.toggleHandRaise()` →
  `saveThenPublish("hand-raised", true)`: a **PATCH to `/api/v1/class-users/{id}`** that saves only
  the *current* state, plus a WebSocket broadcast. The endpoint answers GET with 405. Scribe's
  client refuses that path outright and On Cue never calls it.
- The professor dashboard reads per-student `user_engagement.hand_raises` counts from
  `/api/v1/analytics/class/{cid}`: 403 for students, and counts, not times.
- The class grader renders only `voice`, `message`, `timeline-segment` and `bookmark-note`. My
  account gets no hand events from `class-events`.

So the **video is the label source**. Live logging of my `H` presses this week (Scribe v3's
`handraise-logger.js`) is **timing QA only**: `oncue hand-log-import` puts it in a separate
`hand_log` table. It never goes through `scribe transcribe --hand-log`, which would write
`hand-raise-key` events into the transcript (pipeline.py:239–247), where they are the label itself.

### 2.2 The hand icon in the recording

The recording shows the strip of tiles (each with a name label), the slides, the chat and a
*Class Signals* panel. A raised hand puts a yellow-green circle with a white hand in the
**top-left corner of the tile**, the same spot as the green poll tick, in a different hue. The
speaker's tile gets a green border. A prototype colour test on one real frame:

| | Hue band | Found | Truth |
|---|---|---|---|
| Hand icon | 52–80° (OpenCV H 26–40), S and V > 0.45 | 2 blobs, both on tiles showing the icon | Class Signals showed **2** raised hands ✓ |
| Poll tick | 120–165° (OpenCV H 60–82) | 11 blobs | the ticked tiles, plus the speaker's green border |

**Updated for On Cue.** A quick OCR pass missed "Brady", and the face-matching route is archived.
My tile is now found with **no face recognition**: the label bar is binarised to a 12 × 48 px name
glyph and matched by NCC against a **leave-one-class-out** template built from other classes'
anchor frames, where Forum's border marks me as the only speaker. A **shape test** (round blob,
white glyph inside) rejects lime shirts, plants and border bleed, which pass the hue test alone.

## 3. Label: the hand-icon detector (spec §2)

| Step | What happens |
|---|---|
| Clock | One master clock: media seconds of the rendition Scribe's ASR actually used (`src_res`), with frame times from pts (`-copyts` + `showinfo`). Per-class A/V lag from the speaker border vs ASR speech activity; a class leaves the headline if its lag deviates > 0.5 s from the median |
| Strip | Auto-detected per class from periodic dark gutters, checked against attendance; never a fixed y |
| Tracking | Glyph NCC ≥ 0.75 with a 0.10 margin over the runner-up, else NA. Coverage < 0.30 → `unlocated` |
| Hand state | Baseline-subtracted icon fraction in the corner ROI, hysteresis θ_on 0.20 / θ_off 0.08, shape test, poll-tick NA rule; scanned at 2 fps, frames piped and never stored |
| Live state | A change is confirmed by k = 2 agreeing samples and uses samples ≤ *t* only. This is what causal rules may read |
| Episodes | Onset after ≥ 2 s confirmed DOWN; DOWN gaps < 3 s merge; episodes < 1 s are `glitch`; an UP run without 2 s of observed DOWN before it is `left_censored` (no onset) |
| Re-timing | One 1.25 s read around each onset at full frame rate: onset to ±40 ms, else `retime_failed` |
| Type | **vote** if other tiles rise together (Δn_others ≥ max(4, ⌈0.3(n_tiles − 1)⌉), my tile excluded) or the professor used the VOTE lexicon in the previous 60 s; else **bid**. Only bids are positives |
| Outcome | `called` (my word-bearing main-room turn starts within offset + 5 s), `withdrawn`, or `open` |

**Gate G2 (Tue 18:00, rules committed Monday).** Precision ≥ 0.90 (Wilson LB ≥ 0.82) on a
stratified sample of detected bids; recall ≥ 0.85 (LB ≥ 0.72, ≥ 40 raises) from pixelated
full-strip clips and 60 clips before fresh speech onsets; wrong tile < 1%; left-censored ≤ 10%.
One offline retune, then a **fresh** confirm sample decides; if it still fails, **T-speech**
(fresh speech onsets, no video) becomes primary under the frozen spec in §9.1.

### 3.1 The target and the guard dial

Decision points sit on a 2-second grid from 120 s into the recording to T_end − H. For a point *t*,
let o* be the first bid with o* ∈ (t + g, t + H]:

$$
y_{H,g}(t)=\begin{cases}
1 & o^*\text{ exists; no onset or glitch in }(t,t+g];\ \text{no vote or censored onset in }(t,o^*];\ (t,o^*]\text{ observed}\\
0 & \text{no onset, vote, censored onset or glitch in }(t,t+H];\ (t,t+H]\text{ observed}\\
\text{undefined} & \text{otherwise}
\end{cases}
$$

**The guard *g* is my dial (A1).** The model must fire at least *g* seconds before the icon.
Headline g = 2 s; the **must-tier sweep is g ∈ {1, 2, 3, 5}** (g = 1 needs re-timing and is
reported as approximate if `retime_failed` ≥ 10%). It tests my own precursors and the clock; the
professor reacting to my hand is caught by the CALLON check and the live logger instead.

**At-risk set.** A point leaves the grid, with the first matching reason logged, when: t < 120 s
(`opening`), t + H > T_end (`end`), inside or < 30 s after a recording gap (`rec_gap`), in a breakout
or break segment, while I am speaking (`me_speaking`: my word ended < 3 s ago), while my hand is
already up by the **live** state (`hand_up`), within 10 s after it comes down or after a glitch,
when my tile is unobservable, inside the guard, or with an onset still pending confirmation.

**Expected balance.** 45–60 at-risk minutes and ≈ 4.5 bids per class, 6.5 positive points per bid:
π ≈ **1.6–2.2%** at H = 15; ≈ 140 QA-passed classes give 190k–250k points and ≈ 630 bids. **The
effective n is bids and classes, not points.** Go threshold: 300 dev bids.

## 4. The causal contract (spec §4)

An input may be used at *t* only if (a) I could have perceived it by *t* (heard it, saw it in the
live Forum UI, or knew it before class), (b) it is computed only from rows whose **availability
time** is ≤ *t*, and (c) it comes from a whitelisted JSON path of a whitelisted raw source.

| Row | Available at |
|---|---|
| word text | `end + ε_w` (ε_w = 0.25 s); a word straddling the cutoff is invisible |
| word onset time (no text) | `start + ε_w` |
| word speaker role | `max(end + ε_w, role_event_start + ε_F)` |
| Forum event | `start + ε_F` (ε_F per class, from calibration residuals; ≤ 1.5 s for the headline) |
| my poll answer | submit time, else poll close, else next segment start; then + ε_F |
| my hand history (`me_hand_*`) | when the lowering is confirmed live; last outcome at offset + 5 s |
| templates, learning outcomes, study guide | before class |
| my PCW answers | class start, only from docs verified as of class start |

| Allowed (examples) | Forbidden (examples) | Why forbidden |
|---|---|---|
| ASR words, masked, lowercase, no punctuation | punctuation and casing in headline text; `has_qmark` outside F-punct | batch ASR chooses them using the audio that follows |
| speaker role per word from Forum voice events | diarization clusters, Scribe's `smooth()` relabels | fit on the whole file; `smooth()` reads 1.2 s ahead |
| current speaker and **elapsed** time | the end or duration of an ongoing voice event | reveals when the floor opens |
| current segment title, `seg_type` | segment end, count or next titles; `actual-length-seconds` | future class structure |
| chat from human actors | system notices; transcript `events`, `participation`, `hand-raise-key` | they *are* the label or precede it |
| polls opened so far; my submitted answers | poll scores, feedback; `outcome-assessments`; analytics | post-class grading |
| attendance so far (`ctx_n_present(t)`) | class-total attendance | post-hoc |
| PCW prompts; my answers from verified pre-class docs | PCW edit times, `pcw_status` | class-level QA only |
| my earlier turns, chat and **hand raises this class** | my words or frames available after *t* | the content of the bid being predicted |
| course, term, minute, session number | `class_id`, date, classmates' ids, `hist_*` from test-fold labels | memorisation and privacy |
| — | CALLON phrases ("go ahead", "I see a hand") and `[NAME]` tokens | call-ons come 1–3 s before called speech |
| — | any video pixels: other hands, the Class Signals counter, my tile, the border | vision is P2; my own hand history is label history, not pixels |

**Enforcement.** `features.py` (fast, `np.searchsorted` on availability times) must equal
`causal.py::CausalView(cid, t)` (slow, physically truncates every table at *t*) on 500 random
(class, t), and planted sentinels (a word inside the margin, a segment at t + 1, my frame at t + 5,
a CALLON phrase, a gap event, a poll answer at t + 3) must change no feature. `test_causal.py`
blocks training; `test_loader_allowlist`, `test_gru_inputs` and `test_calibration_qa` back it up.

## 5. Features (spec §5 + A2, A3)

| Group | What it holds |
|---|---|
| `ctx` | minute (splines), course, area, semester, session number, weekday, local and UTC hour, enrolment, attendance so far, segment type and elapsed time, breakout and poll state |
| `spk` | current speaker role and elapsed time, silence, professor share and student turns over 60 s and 300 s, turns since the professor |
| `me` | my turns, words, chat and breakout talk so far; time since my last turn |
| `me_hand` | **my raises so far this class, time since my last lowering, last outcome** (A2: "I never answer more than 5 times" should be learnable) |
| `hist` | **my bid rate per at-risk minute in this course and overall, from earlier training-fold classes only** (A2, should-tier) |
| `poll` | polls so far, time since close, my answer length, similarity of the room to my answer |
| `q` | professor-question detector on `X_lastprof`: `q_open`, age, invite/check/rhetorical/plain, VOTE lexicon |
| `edu` | Stanford EduBehaviors scores on professor frames: invites participation, question, rhetorical question, whole-class address, calls on a student by name (A3, should-tier, decayed like `q_*`) |
| `text` | TF-IDF of role-prefixed tokens over 15 s and 60 s windows and chat; words per minute, fillers, topic shift, similarity to my own earlier contributions |
| `emb` | bge-small-en-v1.5 embeddings of causal frames, recency-pooled (τ = 15, 60 s); PCA/SVD in-fold for M3 |
| `pcwT`, `pcwA` | similarity of the room to the PCW prompts, learning outcomes and study guide; and to **my answers**: resonance gap Δρ and unspent preparation U(t), z-scored against an earlier-only null |

**Feature sets.** F-struct (`ctx + spk + me + me_hand + poll`, no words); F-text; **F-all**
(headline: F-struct + `q + text + emb + pcwT`, plus `pcwA` if the PCW history branch holds); the
−me_hand twins; F-all+hist; F-all+edu; F-name and F-punct (leak-size ablations); F-pcw; F-min.

## 6. Models (spec §6, ranked by the research report)

The research found **no published model that forecasts a student's hand raise from talk**, so
simple baselines are the yardstick, and the expected result is "AP a few times the base rate, not
near-perfect". Its ranked menu, mapped to On Cue:

| Tier | ID | Model | Why (research report) |
|---|---|---|---|
| MVP | B0, AO, B-min, random | prevalence; always-on; minute-only; **matched-rate random alarms** (A9) | every learned model must beat trivial references on the same metric |
| MVP | R | professor-question rule, no fitting | hand-raising is commonly prompted by a teacher question; wait time ≈ 2 s |
| MVP | B1 | L2 logistic regression on F-struct | plays the report's discrete-time **hazard GLM**: history terms (`me_hand_*`) let it learn a refractory dip and a soft cap |
| MVP | M1 | multinomial naive Bayes on role-prefixed counts | cheap text baseline, Platt-recalibrated |
| MVP | M2 | TF-IDF logistic regression | n-grams beat frozen embeddings in the 18-dataset text+tabular benchmark |
| MVP | **M3** | **HistGradientBoosting** on dense F-* + PCA/SVD text (headline) | non-linear timing without a survival library; sklearn's histogram GBDT because libomp (LightGBM) is absent |
| must | M4 | frozen bge-small embeddings + LR | MTEB classification *is* LR on frozen embeddings; bge-small leads small models |
| should | M3+stack | inner-OOF M2 and M4 probabilities as two extra M3 features (A4) | stacking won the text+tabular benchmark |
| should | M5 | streaming GRU hazard, **10–50k parameters**, or a causal TCN (A9) | does sequence modelling add anything over engineered lags? |
| could | kNN | nearest training-class windows as an **explainer**; metadata only (A5) | explains hits and false alarms without showing classmates' words |
| could | M6 | fine-tuned MiniLM cross-encoder | expected not to beat M4 at a few hundred events |

**Left out:** LLMs (no better than trivial baselines at next-speaker prediction; privacy),
TurnGPT and VAP (no checkpoints; audio-only), neural point processes (blind to the transcript),
DeepSurv-style nets (one event per subject), and any resampling (wrecks calibration).

**Training rules.** No resampling or class weights. HGB n\* and LR C are chosen once per outer
fold on F-all (inner `GroupKFold(3)`) and reused for every ablation. **Platt only, never isotonic**
(A9). Interpretability = **feature-group ablations**; grouped permutation importance is could-tier.

## 7. Conditions (spec §7 + amendments)

Defaults: hand target, H = 15, g = 2, S-group split, F-all, masked normalised text, model M3.

| Tier | Conditions |
|---|---|
| **MVP** (Wed 20:00, G3) | K0 model family (B0, R, B1, M1–M3 + references) · F0 F-struct vs F-all and the −me_hand pair · TG global and local shifts · S-group vs S-rand · EV event metric · P-clock · P-plant · CALLON check |
| **must** (Thu 18:00) | K1 M4 · F1 F-text, PCW contrasts · A-mismatch · **G: g ∈ {1, 2, 3, 5}** (A1) · IH+ · N-name · V vault |
| **should** | **F-all+hist** (A2) · **F-all+edu** (A3) · **M3+stack** (A4) · M5 · F2 (F-pcw, F-all−me, F-name, F-punct) · S-time, S-loco · H ∈ {5, 10, 30, 60} · IH curve · g = 0 · T-speech · A-stream · label-shift retrain · live logger · sensitivities · learning curve |
| **could** | M6 · kNN explainer (A5) · N-stale · A-jitter · A-attrib · turn-end grid · M4-MLP · WER |

**Pre-registered hypotheses** (primary statistic: within-class AP_w, paired cluster bootstrap):

| ID | Claim | Test |
|---|---|---|
| **H1** (primary) | content adds beyond structure, within a class | ΔAP_w(F-all − F-struct) CI > 0 **and** ΔAP_w(F-all−me_hand − F-struct−me_hand) CI > 0 |
| H2 | the model knows the moment, not just the phase | TG_local(F-all) CI > 0 and ΔTG_local(F-all − F-struct) CI > 0 |
| H3 | preparation adds | ΔAP_w(with − without PCW) on the PCW subset (exploratory if < 40 classes or < 150 bids) |
| H4 | learned beats the rule | ΔAP_w(M3 − R) > 0 |
| H5 | embeddings beat TF-IDF | ΔAP_w(M4 − M2) > 0 |

**Multiplicity:** fixed sequence. H1 at α = 0.05; only if it rejects, H2–H5 Holm-corrected.

## 8. Splits and metrics (spec §8)

**Splits.** A hashed **vault**: `sha1(f"{class_id}:oncue-vault") % 5 == 0`, about 20% of classes,
committed before any label exists, opened once on Thu 19:00. Every dev tool asserts it never sees a
vault class. Headline CV: `StratifiedGroupKFold(5)` by class, stratified on course, seed 156.
S-rand (random points) only measures the leakage tax. Every in-fold transform is fit inside the
fold's sklearn `Pipeline`.

**Point metrics.** AP, lift = AP/π, and the primary within-class AP_w:

$$
\text{AP}_w=\frac{\sum_c K_c\,\text{AP}_c}{\sum_c K_c}\qquad(K_c=\text{bids in class }c)
$$

plus pooled AP on Platt-calibrated scores, ROC-AUC (secondary), Brier with the **Brier skill
score** (A9), and ECE on 15 equal-mass bins.

### 8.1 The event metric (novel for the course)

The seizure-prediction convention (Winterhalder et al. 2003). On the causal grid 𝒢 (only causal
exclusions removed), firings are built in time order with a refractory period H − g:

$$
\mathcal F_\tau=\Big\{s\in\mathcal G:\ p(s)\ge\tau\ \wedge\ \nexists\, s'\in\mathcal F_\tau\cap\big(s-(H-g),\,s\big)\Big\}
$$

A firing at *s* claims the occurrence period (s + g, s + H]. It is true if y(s) = 1, false if
y(s) = 0, neutral if undefined.

$$
\text{FAR}(\tau)=\frac{\#\{s\in\mathcal F_\tau: y(s)=0\}}{|\mathcal G_{\text{def}}|\,\Delta/60},\qquad
\text{Recall}(\tau)=\frac1K\sum_k \mathbb 1\big[\mathcal F_\tau\cap[o_k-H,\ o_k-g)\neq\emptyset\big]
$$

$$
\text{lead}_k=o_k-\min\big(\mathcal F_\tau\cap[o_k-H,\ o_k-g)\big),\qquad
\text{TiW}(\tau)=\frac{\big|\{t\in\mathcal G:\ \exists s\in\mathcal F_\tau,\ t\in(s+g,s+H]\}\big|}{|\mathcal G|}
$$

**Chance.** Primary: the same rule on globally shifted predictions, read at the model's realised
FAR, giving EventLift = Recall / Recall_shift. Secondary, analytic: a firing process at rate λ_f
independent of the onsets has expected recall equal to its time in warning,

$$
R_0(\lambda_f)=\min\big(1,\ \lambda_f\,(H-g)/60\big),\qquad \text{EventLift}_0=\text{Recall}/R_0(\lambda_f)
$$

Always-on fires every 14 s (≈ 4.3/min) and scores EventLift ≈ 1, so over-alarming is penalised.
**Headline:** recall (with realised FAR and TiW), EventLift and median lead at λ = **0.1/min** and
**0.5/min**, with thresholds rate-matched on inner out-of-fold predictions, never on test.

### 8.2 Timing decomposition and uncertainty

- **Shifts:** TG = AP − AP_shift, with each class's predictions circularly shifted by 150 to
  n_c − 150 points ("which class", a floor); TG_local shifts by ≥ 30 s within 3-minute
  same-segment blocks, keeping class, segment and minute and breaking only "which moment".
- **Uncertainty:** cluster bootstrap over classes (B = 2000), paired for deltas, conditional on the
  fitted fold models; MDE₈₀ ≈ 2.49·SE so a null reads as "no within-class effect larger than X".
- **Success:** (S1) M3/F-all lift ≥ 2 and TG_local CI > 0; (S2) H1 rejected with its
  co-requirement; (S3) Recall@0.1 CI above both the shifted and the R recall at matched FAR.

## 9. Pre-registration (frozen Wed 14 Oct ≈ 14:00, tag `prereg`)

**Committed Monday night:** the vault rule (`config.in_vault`), the G2 pass rules
(`config.G2_RULES`, `config.g2_pass`) and the calibration QA rule (`config.calibration_qa`).
**Fixed at the tag, before the first full dev CV run:** Δ, H, g and the A1 sweep, g_s, ε_w, the
ε_F rule, calibration/A-V/multi-session QA; at-risk and label rules (asymmetric observability, vote
typing, glitches); masking and normalisation; the PCW branch and its coverage rule; feature sets,
models, grids and the reuse of n\*/C; AP_w, H1–H5 and the fixed sequence; the tiers; S1–S3; the
T-speech spec in §9.1. Any later change is logged as a deviation with its date.
**Branch decisions, recorded here when made:** PCW branch (D8) `TODO` · ASR provider per course
(D10, A8) `TODO` · ROI sizes and HSV thresholds (D2) `TODO`.

### 9.1 T-speech fallback (frozen now, so switching targets is not a forking path)

$$
y_s(t)=\begin{cases}1 & \text{a fresh onset } o^*\in(t+3,\ t+15],\ \text{no onset of mine in }(t,o^*)\\
0 & \text{no onset of mine in }(t,t+15]\ \text{and}\ t+15-\text{last\_me\_end}\ge 30\text{ s}\\
\text{undefined} & \text{otherwise}\end{cases}
$$

At-risk: opening, end, rec_gap, breakout, break, me_speaking, guard 3 s; no hand-derived rule and
no tile requirement. Features: F-all−me_hand. Compared with the hand target only at matched FAR.

## 10. Leakage traps and controls (spec §9)

| Trap | Defence | Check |
|---|---|---|
| words after *t*; turns straddling *t* | word availability `end + ε_w`; words and causal frames only | truncation + sentinel tests (blocking) |
| batch-ASR look-ahead (punctuation, casing, word choice) | headline text has none; F-punct ablation | **A-stream**: re-transcribe [t − 180, t] for 300 points and re-score |
| the professor reacts to my hand ("Brady, go ahead") | hand-up exclusion, 10 s refractory, guard g, CALLON phrases and names deleted | **CALLON check** (MVP), live logger, label-shift retrain |
| key press reaches features via the transcript | event tables from raw `class-events.json` only; never `--hand-log` | `test_loader_allowlist` |
| A/V skew, wrong rendition, calibration drift | one rendition, pts timing, per-class lag, windowed calibration QA | **P-clock** stop rule |
| class-level label rate ("he talks a lot in this course") | no class ids or dates; AP_w primary | TG and TG_local |
| overlapping 2 s windows | grouped CV; `min_df` counted in classes | S-rand tax reported |
| votes as trivially predictable positives | vote typing; votes make windows undefined | vote sensitivities, drop-column AP |
| detector false positives correlated with features | shape test, per-class baseline, tick NA rule | per-class precision, precision by segment |
| PCW edited during or after class | `pcw_status` decided before modelling | funnel by status |
| own-content features read my future words | "so far" cutoffs through the same `searchsorted` | Brady-frame-at-t+5 sentinel |
| GRU learns label-side exclusions | loss mask only, no mask channel | `test_gru_inputs` |
| tuning on test; forking paths | inner CV, rate-matched τ, `prereg` tag, vault | realised FAR reported |
| `hist_*` built from test-fold labels | earlier **training-fold** classes only (A2) | fold-aware unit test |

**Positive controls (stop rules):** P-clock (the border–speech AP(δ) curve must peak in
[−1.0, +0.25] s), P-plant (a token planted 8 s before 30% of bids must reach AP ≥ 0.9), IH+
(letting features see +30/+60 s including my own words must jump AP). **Negative controls:**
TG/TG_local shifts, CALLON rate ratio (CI must include 1), A-mismatch (deranged PCW, retrained ×5),
N-name.

## 11. Day-1 checks (spec §11, D0 corrected by A6)

| # | When | Check | Pass |
|---|---|---|---|
| **D0** | Mon 20:00 | **Access.** *Corrected (A6):* Claude's browser tab already ran in my Minerva session, and every Forum API call worked on 12 Oct (class-events, class_edit_page, PCW, the SLE doc and template). What failed: recordings would not load in the automated tab, and the browser tool deliberately refuses to return signed URLs or cookies. **Route A:** copy the `cookie` header from DevTools into `~/.minerva/.env` (`pipeline1/.env` is a symlink) as `FORUM_COOKIE`; every CLI step uses it. **Route C** (console exporter → `harvest-import`) stays the fallback | `scribe me` prints my name, id and role |
| D1 | Mon night | sync, manifest, raw-field audit; system chat notices; start-time and enrolment field names; the zero-voice section | ≈ 694 classes; fields named |
| D2 | Mon + Tue | one frame per section at 30:00; eras; icon size; tick + hand frame; θ and shape thresholds; URL expiry | icon ≥ 6 px in lowres; strip found on all frames |
| D3 | Tue AM | glyph tracking on 3 pilots in every state and width | NCC ≥ 0.75 on my tile, ≤ 0.6 on others |
| D4 | Tue 18:00 | detector QA = G2 | §3 |
| D5 | Mon / Tue | live Scribe runs (local first; Deepgram only where consented) | windowed, ≥ 6 windows, p95 ≤ 1.5 s on ≥ 2/3 |
| D6 | Tue AM | cross-clock: lag, rendition, slide changes vs segments, bids before speech | median lag in [−0.25, 1.0] s |
| D7 | Tue AM | re-fusion word count equals Scribe's `alignment.n_words` | identical |
| D8 | Mon + Tue | PCW harvest side effects: POST + GET twice, diff metadata; history endpoint? | reads change nothing; branch decided |
| D9 | Tue AM | poll time strings and their reference | open/close/submit times known |
| D10 | Mon night | email Prof. Watson (recommended, not blocking: A8) | written OK, with scope |
| D11 | Tue AM | name masking recall and precision | 0 misses; precision ≥ 0.95 |
| D12 | Tue AM | scan and ASR throughput; does Forum auto-lower the hand when I speak? | scan ≤ 5 min per class |
| D13 | Mon night | local ASR benchmark (faster-whisper vs mlx-whisper) | minutes per class measured |
| D14 | Mon night | multi-session recording files | structure known |
| D15 | Mon night | bid-yield estimate from raw voice events | expected dev bids ≥ 300 |

## 12. Schedule, gates and cut list

| Day | Foreground | Done when |
|---|---|---|
| **Mon 12** | Cookie, `scribe sync`, D10 email, D2 frames, timezone table, logger install. Package skeleton, `config.py` (vault rule, G2 rules), synthetic world and offline tests; causal and metric modules on synthetic tables. Scribe stays an unrecorded editable install until I approve pushing `v3` (A7) | sync done; tests green on synthetic data |
| **Tue 13** | Day-1 checks on non-vault pilots; Phase A scan (36 classes); **G1 12:00** (provider per consent); audits 15:30–17:30; **G2 18:00**; Phase B overnight; Phase C decided 22:00 | detector accepted or fallback chosen |
| **Wed 14** | all data landed; QA dashboards; truncation test green on real data; one M3 fold timed; question-detector labels; **`prereg` tag ≈ 14:00**; MVP runs; **G3 20:00** | MVP table with CIs |
| **Thu 15** | must items (incl. G sweep {1, 2, 3, 5}), then should items; figures; **G4 18:00 freeze**; **vault opened once 19:00** | every must cell filled |
| **Fri 16** | notebook top to bottom from cache with outputs; results bundle; README; privacy scan; **submit by 18:00** | submitted |
| Sat–Sun | buffer, fixes only | |
| Mon 19 | delete glyph bits and audit clips | deleted |

**Gates.** G1 Tue 12:00 pilots transcribed, calibrated and lag-checked, else local ASR. G2 Tue
18:00 detector QA, else retune + fresh confirm by 21:30, then T-speech. G3 Wed 20:00 MVP table with
P-clock and P-plant passing, else drop should/could. G4 Thu 18:00 results freeze.

**Cut list** (drop first → last): M6 → could items (incl. kNN explainer) → M5 → M3+stack → S-loco →
F-all+edu → H-sweep reduced to {5, 30} → full IH curve → T-speech (unless primary) → A-stream
reduced to 60 points → S-time and label-shift → F-all+hist → reliability diagram → Phase B capped
→ vault folded into CV. Re-timing is no longer on the list: the must-tier g = 1 row needs it, and
if it fails, g = 1 is reported as approximate (A1). The `site/` rewrite is done.

**Never cut:** the MVP set; the truncation, sentinel, loader and GRU tests; masking checks; label QA
and the funnel; the privacy scan.

## 13. Privacy, consent and ethics

- Transcripts run with `--privacy ids`; names are masked in memory before anything reaches
  `data/tables/`, and `[NAME]` tokens are deleted from feature text. Harvested data stays in
  gitignored `data/`. The committed `results/` bundle holds OOF predictions and metrics keyed by
  `HMAC(ID_SALT, class_id)[:12]`, my own bids and funnel counts: **no words, no classmate data**.
- Figures are aggregate-only (n-grams need ≥ 20 classes; the hero timeline is synthetic).
  `python -m oncue privacy` fails on any roster name or hashed id in README, notebook outputs,
  `figures/` or `results/`.
- Stated trade-offs: `hand_tiles` keeps 12 × 48-bit name glyphs of every tile, and recall-audit
  clips show the full strip with tile interiors pixelated. Both stay local and are **deleted Mon 19
  Oct**. Precision crops show my own tile only. No face recognition anywhere.
- The kNN explainer outputs metadata only, never text (A5). EduBehaviors weights are downloaded at
  runtime and never committed (licences unstated).

**Consent recommendation (A8).** Consent and third-party ASR are my decisions, so nothing blocks
on them, but I recommend the D10 email to Prof. Watson asking (a) whether using other courses'
recordings to recover *my own* hand label is fine, (b) whether third-party ASR is acceptable
(Deepgram with `mip_opt_out=true`; Groq only if named), and (c) whether other instructors should
be told. `config.ASR_PROVIDER_BY_COURSE` picks the provider per course; **local is the default**.
Final results also need Scribe `v3` pushed at a SHA, which waits on my approval (A7).

## 14. How it reaches a 4

- **Original purpose:** predicting the author's own bids for the floor from three years of seminar
  speech and his own pre-class writing, with labels recovered by computer vision from a UI that
  hides them from every student API.
- **Tools beyond class:** calibrated ASR on one media clock; glyph-NCC tile tracking; HSV + shape
  detection with a causal hysteresis state machine; recency-pooled transformer embeddings with
  null-referenced similarity; gradient boosting; a GRU hazard; seizure-prediction event scoring;
  nested grouped CV, cluster bootstrap, fixed-sequence testing, pre-registration, a hashed vault.
- **Meta-knowledge:** a causal contract enforced by blocking tests; leakage *measured* (S-rand
  tax, F-name, F-punct, A-stream, CALLON); within-class statistics; baselines that can falsify the
  content story; positive controls with stop rules.

**A weak signal is still a finding with CIs:** an H1 null reads "I raise my hand by the rhythm of
class, not by what is said", with the MDE and positive controls showing the pipeline *could* see
text; TG > 0 with TG_local ≈ 0 means it knows the class and phase, not the moment.

## 15. Where things live, and references

`design/` holds the spec and amendments, `reports/` the model-options research, `research_notes/`
five literature notes, `later/` the archived face plan, `site/` the presentation page, `oncue/` the
package (`python -m oncue <stage>`, 21 subcommands) and `tests/` offline tests on a synthetic world
with invented names. `results/` and `figures/` are committed and privacy-scanned; `data/` and
`audit/` are gitignored.

Key references: Singer & Willett 1993 (discrete-time hazard); Truccolo et al. 2005 (point-process
GLM); van Houwelingen 2007 (landmarking); Winterhalder et al. 2003 and Mormann et al. 2007
(seizure-prediction evaluation); Shi et al. 2021 (text+tabular stacking); Xiao et al. 2023 (BGE);
Kapoor & Narayanan 2023 (leakage); Saito & Rehmsmeier 2015 (PR vs ROC); Platt 1999; Böheim et al.
2020 (hand-raising and teacher questions); Bernado et al. 2026 (EduBehaviors). The full list is in
spec §14 and the report.
