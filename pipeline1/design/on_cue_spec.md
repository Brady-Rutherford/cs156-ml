# On Cue: Pipeline 1 design (final, revised after critique)

CS156 First Pipeline · Brady Rutherford · Prof. Watson · design frozen Mon 12 Oct 2026 · due Fri 16 Oct · closes Sun 18 Oct (no late work after Mon 19 Oct 01:00)

This design starts from "rigor" and adds the best ideas from "ship", "rubric" and "nlp". This revision fixes every issue in the critique, and §16 maps each issue to its fix. §1.3 lists the design conflicts and which side was picked. Facts checked against the code on 12 Oct are marked *(verified)*:
- `scribe sync --out`; `transcribe --bundle-dir/--privacy/--formats/--language/--no-debug/--hand-log`.
- `calibrate()` returns these fields:
  - `status ∈ {ok, piecewise, unverified}`
  - `applied`
  - `max_residual_s`
  - `method ∈ {windowed, global}`
  - `windows` (the usable windows)
- With fewer than 3 usable windows, `calibrate()` falls back to a whole-class fit and still returns `status "ok"`, `method "global"` and `max_residual_s = 0.0` (calibrate.py:147–151).
- `--hand-log` entries become `hand-raise-key` events in the transcript's `events` list, with `user_id` = Brady (pipeline.py:239–247).
- `fresh_url()` silently falls back to the other rendition when the requested one is missing (pipeline.py:331–336). The default is `res = "lowres"`.
- `timeline_segments` maps events through `wall_to_video_clamped` (forum_events.py:354), which moves a segment that falls in a recording gap to the restart instant. It also fills `end` from the next segment (forum_events.py:369).
- `fuse.EDGE = 0.25 s`, and `smooth()` reads right-hand neighbours.
- `ForumClient` has only `get`/`try_get`; it has no POST.
- Deepgram params already send `mip_opt_out=true`, and the language defaults to `multi`.
- Scribe v3 is entirely untracked in `../Minervas_scribe`: `pyproject.toml`, `scribe_core/`, `hosts/`, `tests/` and `extension_v3/` all show `??`. Origin has no `v3` branch, and the GitHub repo is **public**.
- libomp is not installed; sklearn is 1.9.0; ffmpeg has videotoolbox. `mmdc` and `dot` are absent; `npx` is present.
- nbstripout is not a git filter.
- `pipeline1/.env` is a symlink to `~/.minerva/.env`.
- `pipeline1/` is untracked (`?? pipeline1/`). `pipeline1/.gitignore` ignores `data/`, `outputs/`, `*.mp4`, `*.ogg`, `*.wav` and `.env`. `*.parquet` is not ignored anywhere.

---

## 1. Project name and question

**On Cue.** In theatre, the cue is the line that tells you it is your turn.

**Question.** Using only what Brady could have heard in class, seen in Forum, or written before class, up to second *t*: can a model predict that he will raise his hand in the next 15 seconds? And does *what is being said*, including how closely it matches his own pre-class work, add anything **within a class** beyond the *structure* of the class (minute, activity, turn-taking)?

| Sub-question | Answered by |
|---|---|
| (a) Content vs structure | within-class ΔAP_w(M3/F-all − M3/F-struct), with the same contrast without detector-derived history (H1) |
| (b) The moment vs the phase of class vs the class | local timing gain TG_local; global TG is the "which class" floor (H2) |
| (c) Preparation | H3 on classes with a verified pre-class workbook; A-mismatch control |
| (d) Beyond the obvious ("he raises when the professor asks") | M3/F-all vs the rule baseline R (H4) |
| (e) Lead | H-sweep, information-horizon curve, event lead times |

**Why it is non-obvious:**
- No student API exposes hand raises. Forum's frontend code shows a PATCH-only endpoint, so the label is recovered from the screen-captured recording.
- The inputs are three years of classroom speech aligned to the second, plus Brady's own pre-class writing.
- The model is designed never to see faces, audio prosody, other students' hands, or the professor reacting to the hand. Each of these is **measured** (§9), not assumed.

### 1.1 Scope reset (Monday evening, 15 min)
- P1 is a text-and-context classifier. Its only computer vision is the label detector, which finds the yellow-green hand icon on Brady's tile.
- Archive the face plan for later work. `pipeline1/` is untracked, so use plain `mv`: `mkdir -p pipeline1/later && mv pipeline1/PLAN.md pipeline1/later/the_tell_face.md`.
- **Keep `pipeline1/research_notes/` where it is.** The five notes (turn-taking prediction, classroom discourse NLP, time-to-event models, small-data text and sequence models, imbalance/calibration/interpretability) support this design: the hazard framing, no resampling, and bge/MiniLM timings on this M4. They are cited in §14 (References and AI use).
- Add a one-line banner to `site/index.html` now: "Superseded: this page describes the earlier face plan; see PLAN.md". Keep any README link to it out until it is rewritten (a cut-list item).
- Update the saved memory note. It should say that P1 is the text-and-context classifier, that the hand icon is only the label, and that the face plan is filed under "later (P2)".
- The new `PLAN.md` holds this design, a "Pre-registration" section (§8.5) and the frozen T-speech fallback spec (§2.6).
- Later pipelines: P2 is vision of Brady's own tile. P3 predicts what he says, reusing the same causal view.

### 1.2 Global constants (`oncue/config.py`; every section refers to these)

| Symbol | Meaning | Value |
|---|---|---|
| Δ | decision stride | 2.0 s |
| H | horizon | 15 s (sweep 5, 10, 15, 30, 60) |
| g | guard: onset must be > t + g | 2.0 s (must sweep {2, 5}; {0, 1} should-tier, needs re-timing) |
| g_s | T-speech guard | 3.0 s |
| ε_w | word margin: word **text** usable iff `end ≤ t − ε_w`; word **onset time** (no text) usable iff `start ≤ t − ε_w` | 0.25 s |
| ε_F | Forum-event margin, per class | max(ε_F,min, p95_resid_c, max_residual_s_c), where ε_F,min = 1.0 s for whole-second 2023 timestamps and 0.5 s otherwise. Headline requires ε_F ≤ 1.5 s. Sensitivity admits up to 3.0 s. Logged in `classes.eps_F_s` |
| ε_F,poll | margin for poll times parsed from strings | max(ε_F, 1.0 s) |
| T_open | skip at the start of the recording | 120 s |
| R_low | refractory after the hand is lowered, and after a glitch | 10 s |
| S_me | exclusion after Brady's last word | 3 s |
| — | turn merge gap / fresh-speech silence | 2 s / 30 s |
| f_scan | hand scan rate | 2 fps |
| θ_on / θ_off | baseline-subtracted icon fraction for up / down (hysteresis; tuned on D2 non-vault frames) | 0.20 / 0.08 |
| shape | icon shape test | circularity ≥ 0.6; white-glyph fraction 0.08–0.45 (set on D2) |
| s1, s2 | corner ROI sides, as a fraction of tile height; both stored | measured on D2 |
| k | samples needed to confirm a state change; the live state uses samples ≤ t only | 2 (1 s) |
| D_pre | confirmed DOWN required before an onset | 2.0 s |
| — | down-gap merge / minimum episode / minimum for `me_hand_*` | 3.0 s / 1.0 s / 2.0 s |
| NCC_min / NCC_gap | tile tracking on binarised name glyphs (12 × 48 px) | 0.75 / 0.10 |
| vote | Δn_others ≥ max(4, ⌈0.3·(n_tiles − 1)⌉), **or** the VOTE lexicon in the professor's words in [o − 60, o] | §2.5 |
| A/V | per-class lag tolerance | abs(lag_c − median lag) ≤ 0.5 s |
| cal QA | calibration gate | `method == "windowed"`, ≥ 6 usable windows, `max_residual_s` ≤ 1.5, residual p95 ≤ 1.5 s and median ≤ 0.5 s |
| frames | pause cut / max length | 0.4 s / 5 s |
| EMB | encoder | `BAAI/bge-small-en-v1.5` |
| κ | "already said" threshold in U(t), in z-units | 2.0 |
| — | alarm refractory | H − g |
| λ | false-firing budgets | 0.1 and 0.5 per at-risk minute (6 and 30 per hour) |
| — | global / local prediction shift | u_c ~ U{150, …, n_c − 150} points (skip n_c < 400) / within 3-min same-segment blocks, u ≥ 15 points |
| SEED, VAULT_SALT | — | 156, `"oncue-vault"` |
| ID_SALT | HMAC key for every id in committed results | secret, in `.env`, never committed |

### 1.3 Conflicts resolved

| Topic | Pick | Why (one line) |
|---|---|---|
| H, g | H = 15, g = 2 (rigor) | 6.5 positives per onset and a 2.25 s information gap; the sweeps cover the rest. |
| Boosting library | sklearn `HistGradientBoostingClassifier` | libomp is absent *(verified)*; same algorithm family, zero installs. |
| Holdout | hashed random 20% **vault** of classes | the F26 lockbox is small and low-participation, and it mixes overfitting with drift. |
| Class selection | whole sections, all recorded classes | no per-class choice correlated with the outcome. The population ("sections where I talk") is declared, and claims are limited to it. |
| Tile tracking | per-frame NCC on **binarised name glyphs**, with a **leave-one-class-out** era template; a class's own anchors only validate | observability and coverage must not depend on Brady's speech in that class, or on his mute, border or hand state. |
| Text unit | causal frames plus recency-pooled embeddings (nlp) | about 200k embeddings once, instead of about 900k window embeddings. |
| Text normalisation | lowercase, no punctuation; CALLON phrases and name tokens deleted; punctuated text is an ablation | batch punctuation and casing encode future silence; name tokens carry call-ons. |
| Speaker role | Forum-only causal attribution; Brady's role only from events that contain words | Scribe's cluster mapping is fit on the whole file, and `smooth()` reads right-hand neighbours *(verified)*. |
| Event tables | built only from raw `class-events.json` with a type allowlist | the transcript JSON can carry `hand-raise-key` and `participation`. |
| Brady's name | `[NAME]` like everyone else, deleted from feature text; `[ME]` only in F-name | removes the call-on leak by default. |
| Class-rate null | global index shift ("which class") **and** local within-segment shift ("which moment") | a global shift is beaten by phase-of-class features alone. |
| Event scoring | seizure-prediction firing convention (refractory H − g) on the full causal grid; FAR per at-risk minute/hour; time-in-warning; shifted-prediction chance as primary | counting alarm episodes was gameable by an always-on predictor. |
| Primary statistic | within-class ΔAP_w with a paired cluster bootstrap; pooled ΔAP is secondary | pooled AP is dominated by *which classes* he talks in. |
| H1 comparator | M3/F-all vs **M3/F-struct** (same model), plus the same pair without `me_hand_*` | separates model family from feature set, and keeps the text claim free of detector inputs. |
| PCW answers | used only from docs verified as of class start; availability is a class-level QA status, never a feature | an edit after class start is future information. |
| Hand vs speech target | descriptive, recall at matched FAR; not a hypothesis | different base rates and eligible sets. |
| Vote rule | other tiles rising together (Δn, Brady excluded) **or** a VOTE-lexicon hit | catches votes in small rooms; the "contested" sensitivity checks the cost in engaged rooms. |
| History priors (`hist_*`) | out of the headline; could-tier, fold-aware | otherwise test-fold labels enter training features. |
| Ambiguity mask; hand-culture filter | dropped | both define the data set from future labels or detector output. |
| LLM baseline | none | the course warns against it; it is not reproducible; privacy. |
| CV seeds | 1 seed × 5 folds | 15 folds triples compute for no decision value. |
| Hyperparameters | n* (HGB) and C (LR) chosen once per outer fold on F-all, then reused for every ablation and sweep | compute; declared in the prereg. |
| Recall audit | full-strip clips (tile interiors pixelated) with the tracker's tile outlined | clips of only the tracked tile hide tracking failures. |
| G2 audit | first audit is the confirm set; any retune triggers a fresh confirm sample | the audit must not become training data. |
| Event threshold | rate-matched τ from inner OOF predictions; realised FAR reported; oracle shown second | no tuning on test. |
| Multiplicity | fixed sequence: H1 first, then Holm over H2–H5 | familywise error rate ≤ 0.05 overall. |

---

## 2. Label (y): the hand-icon detector

### 2.1 Clock
The **master clock** is the media seconds of one rendition r_c of class c. That rendition is the one Scribe's ASR job actually used, recorded as `classes.src_res`. Time zero is the audio stream's `start_time`, because ASR word times count from the first decoded audio sample. Words and icons share this clock only when four conditions hold. Each is **checked per class, not assumed**:

1. **Same file.** Scribe gets a small patch (a new commit) that records `asr_info["source_res"]` and adds `TranscribeOptions(strict_res=True)`, which raises an error instead of the silent fallback at pipeline.py:331. The scan reads `src_res`. If ASR ever ran on highres, the scan uses highres too, or the ASR is re-run on lowres.
2. **Frame times from pts.** Every ffmpeg read uses `-copyts` and a `showinfo` filter. The frame time is `pts_time − a_start`, where `a_start` (the audio start_time) and `v_start` come from ffprobe. Frame times are never computed as index/2. Re-timing seeks with input `-ss` under `-copyts`, so pts stay absolute. The reader asserts that the first pts lies within 0.05 s of the requested start.
3. **Per-class A/V lag.** The lag is measured automatically for every class:
   - `border_any(f)` = 1 when any strip tile is bordered (from `hand_tiles`, §2.2). Cross-correlate it with ASR speech activity (any word) over ±3 s at 40 ms resolution, giving lag_c. The many transitions per class give sub-sample precision.
   - A Brady-only version (his bordered tile vs his ASR words) is computed when he has ≥ 20 s of speech, as a check.
   - lag_c = L_ui + skew_c, where L_ui is Forum's constant speaker-detection latency.
   - A class leaves the headline when abs(lag_c − median lag) > 0.5 s.
   - The median lag must lie in [−0.25, 1.0] s; if it does not, stop and investigate.
   - The any-speaker version is measurable in nearly every class, so this rule does not select on Brady's participation.
4. **Multi-session classes** (more than one `recording-sessions[]` entry) are out of the headline unless all three hold:
   - D14 has established the file structure;
   - the calibration's piecewise boundary falls within ±5 s of the session boundary (or per-session files stitch with no residual jump);
   - lag_c validates on both sides.

Forum events (wall ms) are mapped with `wall_to_video` (not the clamped version). **Events whose wall time falls inside a recording gap are dropped**, then shifted by `scribe_core.calibrate`.

Label lateness relative to the audio (could the professor's reaction to the hand become a feature?) is a separate quantity. It is measured directly by the live logger, the CALLON check and the label-shift retrain (§9).

### 2.2 Locating Brady's tile (no face recognition)

**Pass 1 (strip, anchors, validation; about 30 s per class)**
1. **Strip auto-detection, per class.** Decode 6 frames spread across the class. Search the candidate horizontal bands (expected height ± 30%, from D2). Pick the band whose column-mean V shows periodic dark gutters, i.e. an autocorrelation peak at the tile pitch. Check that `n_tiles` is within ±2 of `n_present` at that time (from `enter-class-time`). If no band qualifies, set `tile_status = "unlocated"`. A fixed y is never used. Store `strip_y`, `strip_h` and `era`.
2. **Tiles.** Gutters are runs of at least 2 px with V < 0.12. Tiles are the spans between gutters whose width is within ±20% of the median. Record `n_tiles(f)`.
3. **Label glyph patch:**
   - Take the label bar (its position comes from D2), inset by 4 px so the speaker-border ring is excluded.
   - Exclude the columns holding mic, mute or signal icons (positions from D2).
   - Otsu-binarise the remaining text, resize it to 12 px high, and take a left-aligned 48 px crop (the first ~6 glyphs, which survive truncation).
   - The patch then depends on the name text only, not on mute state, border, hand, the video under the semi-transparent bar, or tile width. D3 verifies this.
4. **Anchors.** Use Brady's main-room, word-bearing voice events (calibrated) that last ≥ 2 s and overlap no other main-room voice event. Read 3 frames inside each, for up to 12 events. The bordered tile is Brady's. Anchors are used for two things: to build **other** classes' templates, and to validate this class.
5. **Template (leave-one-class-out):**
   - `T_me^(−c)` is the pixelwise median of anchor glyph patches from the **other** classes of the same UI era. Every class is therefore tracked with a template that does not depend on its own speech.
   - Era = semester, unless the D2 per-section frames show a UI change; in that case split there.
   - Validation: the tracked tile equals the bordered tile in ≥ 90% of this class's anchor frames. Otherwise set `tile_status = "tracking_failed"`.
   - Classes with fewer than 3 anchors are `template_validated = False` but are tracked identically.

**Pass 2 (full scan at 2 fps; frames are piped, never written to disk)**

```
ffmpeg -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 30 -copyts -hwaccel videotoolbox \
  -i "$URL_SRC_RES" -vf "crop=iw:STRIP_H:0:STRIP_Y,fps=2,showinfo" -f rawvideo -pix_fmt rgb24 pipe:1
```

On a 403, the scanner fetches a fresh URL, resumes with `-ss` at (last pts − 2 s), and de-duplicates frames by pts.

6. **Per-tile statistics → `hand_tiles`** (local, gitignored). For each frame and tile, store:
   - pts, `tile_idx`, `x0`, `x1`;
   - `frac_hand` and `frac_tick` at ROI sizes s1 and s2;
   - shape score at s1 and s2;
   - border score;
   - `ncc_me`;
   - the packed binary glyph patch (12 × 48 bits).

   That is about 5 MB of float16 statistics plus about 19 MB of glyph bits per class, roughly 4 GB in total. **Tracking, episodes, votes, the A/V lag and any retuning (θ, ROI size, k, shape thresholds, template) then run offline in minutes, with no re-streaming.**
7. **Tracking (offline).** Brady's tile is the argmax of NCC against `T_me^(−c)` if NCC ≥ 0.75 and it beats the runner-up by ≥ 0.10. Otherwise the sample is `NA`. Off-strip time (pagination, on stage) is NA.
8. **Coverage.** `tile_found(f) ∈ {0,1}`. `coverage` is the share of main-room, non-breakout samples with `tile_found = 1`. A class with coverage < 0.30 gets `tile_status = "unlocated"`: it is excluded from the hand target, kept for T-speech, and reported in the funnel.
9. **Tracking diagnostics** (D3 and the funnel). These do not depend on the detector:
   - NCC on Brady's tile by state (muted, unmuted, bordered, hand up) and by tile width.
   - The NA rate and the left-censored share inside [o^s − 60, o^s] around **fresh speech onsets**, compared with random baseline times.
   - If more than 10% of UP runs are left-censored, tracking is fixed **before** G2.

### 2.3 Per-frame hand state
Let R be the top-left square of Brady's tile, of side `s·tile_h`, inset by 3 px so the border ring is excluded. In OpenCV HSV (H in 0–179):

$$
\text{frac}_{\text{hand}}(f)=\frac{1}{|R|}\sum_{p\in R}\mathbb 1\big[H_p\in[26,40]\wedge S_p>115\wedge V_p>115\big]\quad(\text{hue }52\text{–}80^\circ,\ s,v>0.45)
$$

- **Shape test.** The largest connected component of hue-passing pixels in R must be roughly circular, with circularity $4\pi A/P^2\ge0.6$. It must also contain white glyph pixels (S < 0.25, V > 0.8) making up 0.08–0.45 of its enclosing circle. A Hough-circle variant is the fallback, and thresholds are set on D2 frames. This rejects plants, lime shirts, highlighters, avatar colours and chroma bleed from the border.
- **Per-class baseline.** $\text{frac}'_{\text{hand}}=\text{frac}_{\text{hand}}-\operatorname{median}(\text{frac}_{\text{hand}}\mid\text{DOWN samples of this class})$ (two passes, label-side). A class whose DOWN baseline exceeds θ_off/2 is **flagged**, and all of its bids are audited.
- **Poll tick.** `frac_tick` is the same test with H ∈ [60, 82] (120–165°). D2 finds a frame where Brady has both answered a poll and raised his hand:
  - If the tick can hide the hand, any sample with `frac_tick ≥ θ_on` is **NA** for hand state, not DOWN.
  - If the hand wins, the tick is logged only.
  - Either way, bid rates in poll vs non-poll segments are reported with and without the NA rule.
- **Other hands.** `n_others(f)` is the number of tiles **other than Brady's tracked tile** that pass both `frac' ≥ θ_on` and the shape test. It is used only to type votes (§2.5) and is **never** a feature.

### 2.4 Episode state machine (2 fps samples)

| Rule | Value |
|---|---|
| Raw state | UP when `frac' ≥ θ_on` and the shape test passes; DOWN when `frac' < θ_off`; NA when the tile is not tracked (or under the tick rule); otherwise keep the previous state (hysteresis) |
| Confirm | k = 2 consecutive agreeing samples |
| **Live state** (causal) | the confirmed state using only samples ≤ t. A change becomes live at the second agreeing sample, 0.5 s after the first. Used by every causal rule and covered by the mask-causality test |
| NA bridging (label-side) | NA gaps ≤ 1.0 s with the same state on both sides are filled; longer gaps break the episode |
| Onset | first UP sample of a confirmed UP run preceded by ≥ D_pre = 2.0 s of confirmed DOWN. Coarse time = midpoint between the last DOWN and the first UP sample (±0.25 s) |
| Left-censored | an UP run whose preceding 2.0 s contains NA or the recording start: **no onset**, `left_censored = True` |
| Merge | DOWN gaps < 3 s between UP runs form one episode (`n_toggles` logged) |
| Glitch | confirmed episode < 1.0 s: logged as `glitch`. Every label window containing it is **undefined**, and R_low applies after it (`post_glitch`) |
| Offset | first sample of the DOWN run that ends the episode; `offset_live_s` = when that DOWN is confirmed; `dur_s = offset − onset` |
| Re-timing (**should**) | `ffmpeg -copyts -ss (t_up−1.0) -i URL -t 1.25 -vf crop=…,showinfo`, cropped to the tile. Times come from pts and are absolute under `-copyts`. Onset = first frame with `frac' ≥ θ_on` and a shape pass, followed by ≥ 3 positive frames (±1 frame, 33–40 ms). If the seek fails: midpoint, `retime_failed = True` |
| `me_hand_*` eligibility | episodes that pass the shape test on ≥ 2 confirmed samples and have confirmed duration ≥ 2 s |

- **Hand already up** at *t*: the decision has been made, so the point is excluded (§3).
- **Hand lowering**: the point is excluded for R_low = 10 s after `offset_live_s`. That is when "Brady, go ahead" happens.

### 2.5 Episode type and outcome

**Vote vs bid.** Brady's own tile is excluded from Δn:

$$
\Delta n_{\text{others}}(o)=\max_{f\in[o-5,\,o+10]} n_{\text{others}}(f)-n_{\text{others}}(o-5\text{ s})
$$

$$
\text{vote}\iff \Delta n_{\text{others}}(o)\ge\max\big(4,\lceil 0.3\,(n_{\text{tiles}}-1)\rceil\big)\ \ \lor\ \ \text{VOTE lexicon in the professor's words in }[o-60,\,o]
$$

- The lexicon rule is label-side typing, which is legitimate. It catches "raise your hand if…" votes in small rooms.
- **Contested** episodes are Δn-typed votes with no VOTE hit but an INVITE hit or `q_open` in the professor's last 60 s. These are possibly genuine bids in an engaged room.
- **Validation:**
  - the share of Δn-typed votes with a VOTE hit (expect ≥ 0.6);
  - the share of lexicon-typed votes with Δn_others ≥ 2;
  - bids lost to vote typing, reported per course.
- **Sensitivities**, reported next to the headline AP (default = votes undefined):
  1. votes as positives;
  2. Δn-only typing;
  3. contested episodes as bids;
  4. threshold 0.5·(n_tiles − 1).
- **Only bids are positives.** Votes and left-censored onsets make a window undefined: over (t, o*] for a positive, or (t, t + H] otherwise (§3).
- Vote-proneness is checked by drop-column AP (with and without `q_vote_lex`), not by permutation importance, which is unreliable with correlated features.

**Outcome** of each bid:
- `called`: a word-bearing Brady main-room turn starts in [o, offset + 5 s]; `call_latency_s` is recorded;
- `withdrawn`: otherwise;
- `open`: still up at the end of the recording or when the tile is lost.

### 2.6 Speech onsets (second target and cross-check)
1. Take Brady's main-room voice events (`room_signature == "Main Classroom"`, start = `timestamp − duration`, calibrated) that are **word-bearing**, i.e. overlap at least one ASR word attributed to him. Blips with no words (unmuting, breathing, keyboard; p10 duration 0.33 s) are dropped. The AP change when they are kept is reported.
2. Merge events into turns when the gap is < 2 s.
3. A turn is **fresh** when Brady was silent for ≥ 30 s before it. Fresh onsets are $o^s_j$.

They are used four ways:
1. **Label cross-checks**, reported per section as diagnostics and never used as filters:
   - `called_share`;
   - the latency histogram $o^s-o$ (expect a peak at 3–60 s and no mass below 0);
   - `hand_preceded_share`: the share of fresh onsets with a bid in $[o^s-180,\ o^s]$ (the recall proxy; cold-call courses will be low).
2. **T-speech** (should-tier).
3. **Fallback primary target** if G2 fails.
4. The `me_speaking` exclusion and the `me_*` features (past only).

**T-speech spec.** This is written into `PLAN.md` before the prereg, so that switching targets is not a forking-path decision.
- **Target:**

  $$
  y_s(t)=\begin{cases}1 & \text{a fresh onset } o^*\in(t+3,\ t+15],\ \text{no Brady onset in }(t,o^*)\\
  0 & \text{no Brady onset in }(t,t+15]\ \text{and}\ t+15-\text{last\_me\_end}\ge 30\text{ s}\\
  \text{undefined} & \text{any onset in }(t,t+3];\ \text{any non-fresh onset in the window};\ t+15-\text{last\_me\_end}<30\text{ s}\end{cases}
  $$

  A positive automatically satisfies the 30 s rule, so the rule removes only the forced negatives right after Brady speaks.
- **At-risk set:** `opening`, `end`, `rec_gap`, `breakout`, `break`, `me_speaking`, and the guard g_s = 3 s. It has **no hand-derived rule and no tile requirement**.
- **Features:** F-all−me_hand. The CALLON phrases and name tokens are already removed from all feature text (§5.3).
- **Class QA:** calibration and A/V lag only.
- **Comparison with the hand target:** descriptive only, at matched FAR.

### 2.7 Label QA: gate G2 (Tue 18:00; about 100 min of Brady's time)

All material is non-vault, kept local and never committed.

| Audit | Material | Pass |
|---|---|---|
| Precision | `oncue.hands.contact_sheet`, stratified: up to 3 random detected bids per scanned dev class (≈ 300); every bid inside a poll segment (cap 40); every typed vote (cap 40); every bid of a flagged class. Each item shows crops of Brady's tile at o − 1, o, o + 1 s | point estimate ≥ 0.90 **and** Wilson lower bound ≥ 0.82. Per-class precision table. "Classes with audited precision ≥ 0.9" is a sensitivity population |
| Recall (a) | `oncue.hands.audit_clip`: 3 dev classes drawn at random from **all** selected sections. The full strip is shown at 2 fps, rendered at 16 fps (8×). Tile interiors are pixelated to 6 × 6 blocks so faces cannot be recognised; corner squares and name labels stay sharp. The tracker's chosen tile is outlined, or an "NA" banner is shown. Burned-in timecode and a detector-state dot. About 11 min per class. Brady marks **every** raise on his own tile, identified by name, including during NA spans and poll segments | pooled with (b): point estimate ≥ 0.85 **and** Wilson lower bound ≥ 0.72, over ≥ 40 true raises. Reported overall, inside NA spans and inside poll segments. Wrong-tile share of outlined frames < 1% |
| Recall (b) | `oncue.hands.onset_clips`: 60 random fresh speech onsets (dev, all sections), each a full-strip clip over $[o^s-120,\ o^s]$ at 8× (15 s), rendered the same way | pooled with (a) |
| Tracking | left-censored share of UP runs; NA rate around fresh speech onsets vs baseline | ≤ 10% left-censored, or fix tracking first |
| Timing (should) | 10 episodes frame-stepped | median abs(re-timed − manual) ≤ 0.10 s |
| Live ground truth (**should**) | `extension_v3/handraise-logger.js` in the uni profile during **every** live class Tue–Thu. Imported by `python -m oncue hand-log-import` into the separate `hand_log` table, used **only** for timing QA. **Never** `scribe transcribe --hand-log` | key-press → icon lag (median, p95) |

- **Pre-registration of the gate.** The pass rules are committed on Monday night, together with the vault rule.
- **If G2 fails:**
  - Retune θ, the ROI size (s1 or s2), k and the shape thresholds once, offline from `hand_tiles`, by 20:00.
  - After a retune, the first audit counts as **tuning material**. A **fresh confirm sample** (3 new classes, 60 new onsets, a new stratified precision sample) is audited at 20:30–21:15, and G2 is re-checked at 21:30.
  - Reported numbers come from the confirming audit only, with Wilson bounds.
  - If it still fails, **T-speech becomes the primary target** under its frozen spec, and hand results are reported on audited classes only.

---

## 3. Prediction task framing

**Decision grid.** $t\in\{T_{\text{open}}, T_{\text{open}}+\Delta,\dots\}$ in master-clock seconds, stride Δ = 2 s, up to $T_{\text{end}}-H$. A fixed stride makes "false alarms per minute" well defined and avoids favouring utterance boundaries. A turn-end grid is a could-tier variant.

**Target** (headline H = 15, g = 2). Let $o^*$ be the first bid in $(t+g,\ t+H]$. Positives need observation only up to their onset; negatives need the whole window:

$$
y_{H,g}(t)=\begin{cases}
1 & o^*\ \text{exists; no onset or glitch in }(t,t+g];\ \text{no vote or left-censored onset in }(t,o^*];\ (t,o^*]\ \text{observed}\\
0 & \text{no onset, vote, left-censored onset or glitch in }(t,t+H];\ (t,t+H]\ \text{observed}\\
\text{undefined} & \text{otherwise}
\end{cases}
$$

"Observed" means no NA sample after bridging, plus `tile_found = 1` at *t*. The **symmetric rule** (both classes need (t, t + H] observed) is a sensitivity row. This is a discrete-time hazard: $p(t)=P(y_{H,g}(t)=1\mid x_{\le t},\ t\in\mathcal R)$.

**At-risk set 𝓡** (`excl_reason` = the first rule that fires). Causal rules use only information available at *t*. Label-side rules concern observability and label validity only, never feature content.

| Code | Rule | Kind |
|---|---|---|
| `opening` | t < 120 s | design |
| `end` | t + H > T_end (recording end) | label (right-censoring); **no** fixed end skip |
| `rec_gap` | t is inside a gap between recording sessions, measured in **wall time**, or < 30 s after a restart | causal |
| `breakout` | current segment `seg_type == "breakout"`, or Brady's latest voice event in [t − 60, t] has a non-main `room_id` | causal |
| `break` | `seg_type == "break"` (e.g. "Eye rest break") | causal |
| `me_speaking` | a Brady-attributed word ending in (t − S_me, t − ε_w], or an ongoing Brady voice event (start ≤ t − ε_F) that already overlaps an available Brady word | causal |
| `hand_up` | Brady's **live** state (samples ≤ t) is UP, or t is inside a left-censored UP run already observed by t | causal |
| `post_lowering` | t < `offset_live_s` + R_low | causal |
| `post_glitch` | t < live end of a glitch + R_low | causal |
| `unobservable` | `tile_found = 0` at t, or NA in the span the target needs (above) | label |
| `guard` | any onset or glitch in (t, t + g] | label |
| `onset_pending` | an onset in (t − 1.5, t] that is not yet live (confirmation delay plus re-timing jitter) | label |
| `vote_or_censored` / `glitch_window` | a vote, left-censored onset or glitch in the span the target needs | label |
| class QA | decided **before** modelling:<br>• calibration: `applied`, status ∈ {ok, piecewise}, `method == "windowed"` with ≥ 6 usable windows, `max_residual_s` ≤ 1.5 s, residual p95 ≤ 1.5 s and median ≤ 0.5 s (§4.3);<br>• A/V lag within tolerance;<br>• rendition match;<br>• multi-session rule (§2.1);<br>• `tile_status` located, not `tracking_failed`;<br>• ASR present.<br>Sensitivity admits classes with residuals up to 3 s. A global-fallback calibration is **never** headline (unit test) | class |

**Expected balance** (re-measured Tuesday from the Phase A funnel):
- About **45–60 at-risk minutes per class** (breakouts, breaks, hand-up and me-speaking time removed), so 1,350–1,800 points per class.
- About 4.5 bids per class in active sections. Monday's voice-event estimate (D15) refines this.
- Each bid yields (H − g)/Δ = 6.5 positive points. So π ≈ **1.6–2.2%** at H = 15 (**1.0–1.3%** at H = 10; 3.5–4.7% at H = 30).
- For about 140 QA-passed classes: about 190k–250k points, about 630 bids (plausible range 350–1,000), about 4k positives.
- **The effective n is the number of bids and classes, not points.** Every reported number carries `n_classes` and `n_onsets`.

---

## 4. Causal feature contract

**Principle.** An input may be used at *t* only if all three hold:
- **(a)** Brady could have perceived it by *t*: he heard it, saw it in the live Forum UI, or knew it before class.
- **(b)** It is computed only from rows whose **availability time** is ≤ *t*.
- **(c)** It comes from a whitelisted JSON path of a whitelisted raw source (§4.4). **Every event table is built from raw `class-events.json` only**, with the type allowlist {`voice`, `message`, `timeline-segment`}. Nothing in Scribe's transcript JSON feeds features except the `words[]` of `*_asr.json`.

Live-UI visibility counts even when the API writes the row later. For example, the speaker border is visible live, but the voice event is written at its end; in that case only the part of the row up to *t* may be used.

Availability times:

| Row | Available at |
|---|---|
| word text | `end + ε_w` |
| word onset time (no text) | `start + ε_w` |
| word role | `role_avail_s = max(end + ε_w, role_event_start + ε_F)` |
| Forum event | `start + ε_F` (polls: `start + ε_F,poll`) |
| Brady's poll answer | submit time if present, else poll close, else next segment start; then + ε_F |
| `me_hand_*` | `offset_live_s`; `me_hand_last_outcome` at offset + 5 s |
| templates, learning outcomes, study guide | before class (subject to the template-version caveat, §5.6) |
| Brady's PCW answers | class start, **only** for `pcw_status ∈ {asof_start, verified, git}` (§5.6) |

### 4.1 Allowed inputs

| # | Input | Source / exact field | Available at | Why fair |
|---|---|---|---|---|
| A1 | Transcript words | `*_asr.json` `words[]` (`start, end, text, conf, filler`). Names masked (§5.1), then normalised: lowercase, no punctuation, CALLON phrases and name tokens deleted (§5.3). Punctuated, cased text only in F-punct | `end + ε_w` | heard live |
| A2 | Speaker role per word | **Forum-causal role**: the Forum voice event overlapping [start − 0.25, end + 0.25] with the most overlap gives the user, mapped to `prof` (= `teacher-user-id` or staff role), `me` (= `users/self.id`), `student` or `unk` | `role_avail_s` | the speaker border is visible live. Roles only, never classmate identity |
| A3 | Current speaker and elapsed time | `voice`: `actor_id`, start = `timestamp − event_data.duration`, calibrated. Brady's role uses **word-bearing** events only | start + ε_F | visible live. **End and duration are used only once end ≤ t − ε_F** |
| A4 | Class environment | course code, area, term → `semester_idx`; session number from the **full section schedule** (`paginated-classes`, including unrecorded classes); weekday; `ctx_hour_local` (committed semester → IANA timezone table for Brady's city); `ctx_hour_utc` of the scheduled start; `ctx_enrolled` (section roster size) | before class | public schedule |
| A5 | Time in class | `ctx_min = (t + rec_offset_s)/60`, with `rec_offset_s` = recording-started − scheduled start (the start field is named once D1 confirms it); planned remaining = 90 − ctx_min | t | the clock. **Never** `actual-length-seconds` |
| A6 | Current timeline segment | raw `timeline-segment`: `timeline_segment_title`, `timeline_section_title` (masked), `timestamp` = start; unclamped mapping, gap events dropped | start + ε_F | on screen. Current and past only; **no end column** |
| A7 | Breakout membership; Brady's own breakout talk | `/breakouts` (own group only); Brady's voice events with non-main `room_id` | breakout end | he knows his group and what he said |
| A8 | Chat | raw `message`: `event_data.message_text` (masked), role. **Only messages whose `actor_id` is a human in `class-users`**; system, null-actor or notice messages are dropped and counted (D1) | `timestamp + ε_F` | chat panel |
| A9 | Polls opened so far | `/polls` prompt text; start parsed from strings like "started at 0:32:39" against the reference D9 verifies against poll timeline segments | start + ε_F,poll | on screen |
| A10 | Brady's poll answers | `/poll-responses`, own answers only, whitelisted fields | as in the availability table | he wrote it |
| A11 | Attendance so far | `ctx_n_present(t)` = users with `enter-class-time ≤ t` | enter time + ε_F | tiles are visible. **No class-total attendance** (post-hoc) |
| A12 | Learning outcomes, study guide, PCW instructions | `class_edit_page.assessment.focused-outcomes[].learning-outcome.description`, `preparatory-assignment.description` | before class | published in advance |
| A13 | PCW template prompts | sle `GET /api/v1/templates/UUID/` → `blocks[].block_components[].{type,value}` (Quill `ops`); the version current at the class date if versions exist, else flagged `template_asof = "harvest"` | before class | he read them |
| A14 | Brady's PCW answers | sle `GET /api/v1/doc/UUID/` (text, code, upload **filenames**); only docs with a verified pre-class status | class start | written before class |
| A15 | Brady's earlier contributions this class | A1, A3, A8 restricted to role = `me`; frames with `close_s ≤ t`; word-bearing events only | as A1, A3, A8 | he knows |
| A16 | Brady's earlier hand raises this class | eligible `hand_episodes` (§2.4) with `offset_live_s` < t | as stated | **This is the label's own history, as in an autoregressive hazard, not a pixel feature.** It is reported with and without (F-all−me_hand), so the text claim stands without detector-derived inputs |
| A17 | Derived similarities and embeddings | functions of A1, A6, A12–A15 | max of the input availabilities | computed inside the causal view |

### 4.2 Forbidden inputs

| Forbidden | Why |
|---|---|
| Words with `end > t − ε_w`, or words straddling the cutoff; Scribe turns, JSONL utterances, `utterances[]`, `paragraphs` | turns cross *t* and carry future words. Features use words and causal frames only |
| **Punctuation and casing in headline feature text**; `has_qmark` outside F-punct | Deepgram's punctuate/smart_format choose them using the audio that follows, including the pause that opens the floor |
| CALLON phrases and `[NAME]` tokens in feature text (all targets) | call-on cues come 1–3 s before called speech |
| Scribe transcript `events` (including `hand-raise-key`), `participation`, `analytics`; anything imported with `--hand-log`; the `hand_log` table | key press or professor-side counts are the label itself or come before it |
| End or duration of an ongoing voice event | reveals when the floor opens |
| Diarization cluster ids, `cluster_conf`, the cluster→user map, `how`, Scribe `smooth()` relabels | fit on the whole recording; `smooth()` reads words up to 1.2 s later *(verified)* |
| `actual-length-seconds`, `recording-ended`, segment end, count or next titles; class-total attendance from `class-users` | future class structure |
| Whole-class aggregates: manifest `n_voice`/`n_my_voice`, Scribe `speakers{}`, `how_mix`, ASR mean confidence, calibration score/status/offset, `eps_F`, A/V lag | class-level label proxies or whole-file inference |
| `outcome-assessments`, focused-outcome scores, `/analytics/class/CID`, `bookmark-note`; any non-whitelisted JSON path (doc comments, feedback, grades, per-block timestamps; poll scores or feedback; per-user class stats) | post-class grading or annotation |
| PCW edit or modification times, `pcw_status`, any `pcw_edited*` column | class-level QA only; edits after start are future information |
| Brady voice events without words (sub-second blips) in `me_*` or Brady's speaker role | pre-raise mic activity is an "other marker". Sensitivity only |
| Brady's frames, poll answers or words available after *t*, in any own-content feature | the content of the bid being predicted |
| Any video pixels: Class Signals counter, `n_others`, other tiles' icons, Brady's tile, speaker border, slides | the counter includes his own hand; Brady ruled out other markers; vision is P2. (A16 is label history, not pixels) |
| Classmates' hashed ids as categories; `class_id`, date | memorisation and privacy |
| Forum events whose wall time falls inside a recording gap | Scribe clamps them to the restart and creates fake bursts |
| `hist_*` rates built from other classes' labels (headline) | need fold-aware recomputation; could-tier only |
| `y_*`, `next_bid_s`, `excl_reason`, `label_state`, `fold`, `in_vault`, `tile_*`, `coverage` | label machinery, blocked by `FEATURE_ALLOWLIST` |

### 4.3 Timing errors

| Source | Size | Defence |
|---|---|---|
| ASR word boundaries | ±0.1–0.3 s | ε_w = 0.25 |
| Rendition mismatch (highres and lowres are separate encodes) | frames to seconds | `src_res` recorded; scan the same file; `strict_res` (§2.1) |
| Frame timestamps (VFR capture, stream start_time, seek resets) | up to seconds | pts via `-copyts` + `showinfo`; clock origin = audio start_time; assert seek pts |
| A/V lag inside the mp4 | expected < 0.3 s | lag_c for **every** class (§2.1); drop if abs(lag_c − median) > 0.5 s |
| Forum → video calibration | per class | QA requires `method == "windowed"` with ≥ 6 usable windows (the global fallback reports `max_residual_s = 0`); `max_residual_s` ≤ 1.5 s (dropped, not clamped); per-class residuals = abs(calibrated voice start − nearest ASR word start within ±2 s) over all main-room voice events preceded by ≥ 2 s of silence, with **p95** ≤ 1.5 s and median ≤ 0.5 s; ε_F = max(ε_F,min, p95, `max_residual_s`) |
| 2023 whole-second timestamps | ±1 s | ε_F ≥ 1.0 s |
| Forum role edge (EDGE = 0.25 s) | a word can be attributed via an event that starts after it | `role_avail_s`: until the event is available, the role reads `unk` |
| Label timing | ±0.25 s coarse, ±40 ms re-timed | g = 2; re-timing should-tier, needed for g ∈ {0, 1} and H = 5 |
| Label lateness vs audio (key press → icon) | unknown | live logger (should); CALLON check (**must**, MVP); label-shift retrain (should) |
| Batch-ASR look-ahead (punctuation, casing, word choice) | unknown | punctuation and casing stripped from headline text; **A-stream** re-transcribes [t − 180, t] and recomputes every feature; F-punct ablation |
| Recording gaps | 0-length in media time | `rec_gap` in wall time; gap events dropped |
| Robustness | — | A-jitter (could): add N(0, 0.5 s) to all times; AP must stay within its CI |

### 4.4 Enforcement
- **Two paths.**
  - `oncue/features.py` is the fast, vectorised path. It uses `np.searchsorted` on `end + ε_w`, `start + ε_w` (onsets), `role_avail_s`, event `start + ε_F` and frame `close_s`.
  - `oncue/causal.py::CausalView(class_id, t)` is the slow reference path. It physically truncates every table at *t*, clips ongoing intervals at *t* (end = NaN, `ongoing = True`), and rebuilds the frames, the tail, Brady's frames, the word-bearing flags and the live hand state from the truncated data.
- **`tests/test_causal.py`** blocks training on failure. It runs 500 random (class, t) per run, across at least 5 real classes plus synthetic fixtures:
  1. `assert_frame_equal(fast(full, t), reference(truncate(full, t), t))`, with numeric tolerance 1e-6 and string fields equal.
  2. **Sentinels.** Inserting each of the following must change no feature:
     - a word `zqxsentinel` with end in (t − ε_w, t], and another at t + 0.1;
     - a professor voice event starting at t − ε_F/2;
     - a segment starting at t + 1;
     - **a Brady frame at t + 5 whose text equals the current window** (checks `own_sim`, `pcw_unspent`, `poll_res_my` and all `me_*`);
     - **a CALLON phrase and a `[NAME]` token at t − 5** (checks the TF-IDF, embedding and NB inputs);
     - a Forum event inside a recording gap;
     - a Brady poll answer whose submit time is t + 3.
  3. **Straddle:** a word with start < t − ε_w < end must not appear anywhere. A voice event spanning *t* must expose only elapsed time.
  4. **Mask causality:** the causal at-risk rules (including the live hand state, `post_glitch` and the word-bearing flag) are unchanged under truncation.
  5. **Allowlist:** the feature columns are a subset of `FEATURE_ALLOWLIST`. None matches `^(y_|next_|excl|label_|fold|in_vault|tile_|coverage|pcw_edited|pcw_status|cal_|eps_|av_|hand_log|participation|analytics)`. `has_qmark` and cased tokens appear only in F-punct.
- **`tests/test_loader_allowlist.py`:**
  - per-source JSON-path whitelists; unknown keys are dropped and logged to `data/logs/dropped_keys.csv`;
  - every `features` column traces, through `features.PROVENANCE`, to whitelisted paths;
  - event types are limited to {voice, message, timeline-segment};
  - `hand-raise-key`, `participation`, `analytics`, `bookmark-note` and `outcome-assessments` are blocked;
  - the feature build raises if anything reads a transcript `events` key;
  - every chat row's actor is a human in `class-users`.
- **`tests/test_gru_inputs.py`:** M5 input tensors for steps ≤ t are bit-identical when labels are recomputed with every onset after *t* deleted. For the F-all−me_hand variant, they are bit-identical with every onset deleted.
- **`tests/test_calibration_qa.py`:** a calibration with `method == "global"`, or with fewer than 6 windows, gives `qa_pass = False` for the headline.
- **`tests/test_labels.py`:** live state uses samples ≤ t; `onset_pending`; glitch windows are undefined; asymmetric observability; T-speech's 30 s rule.

---

## 5. Representations and features

### 5.1 Masking (in memory, on Deepgram's cased `punctuated_word`, before anything is written to `tables/`)
- **Roster:** `class-users` first and last names, plus Scribe keyterm spellings, plus explicit Brady variants (`Brady`, `Braydon`, `Bradley`, `Braden`, `Rutherford`).
- **Per token:**
  1. Strip surrounding punctuation and a possessive `'s` or `s'`.
  2. **Exact** roster match, case-insensitive → `[NAME]`. Exception: if the name is a common word (in the `wordfreq` top-20k English list or Scribe's `fixup.COMMON_WORDS`, e.g. Will, Hope, Grace, Mark, Case, Park, King, Lane, Young, Bell), mask it only when it is capitalised and not sentence-initial.
  3. **Levenshtein ≤ 1** (`scribe_core.text.levenshtein`) applies only to tokens of ≥ 4 characters that are capitalised mid-sentence and are not common words. Names under 4 characters match exactly only.
  4. Brady → `[NAME]` by default, `[ME]` in F-name.
- Masking is applied to words, chat, timeline titles and poll prompts.
- Casing from Deepgram decides only which tokens become `[NAME]`, and `[NAME]` tokens are deleted from feature text anyway (§5.3). Any residual effect is covered by A-stream.
- **D11 checks** (non-vault pilots and Phase A):
  - Recall: after masking, grep finds 0 occurrences of every non-common roster name and variant.
  - Precision: 200 random masked tokens shown locally with ±5 words of context; Brady judges, target ≥ 0.95.
  - Masked-token rate per class.
  - The 20 most frequent masked surface forms, kept local.
  - **The build fails** if any masked surface form that is not an exact roster name is among the `wordfreq` top-1,000 English words.

### 5.2 Causal text frames (the atomic unit)
- A **frame** is a run of consecutive words with the same Forum-causal speaker (hashed). It is cut at a pause ≥ 0.4 s, a speaker change, or 5 s of length. Expect about 1,200 frames per class, about 200k in total.
- **Speech onset is perceptible.** CausalView may use word **start times** (not text) for words with `start ≤ t − ε_w`.
- **`close_s`** is the time the cut became observable. Both paths implement it identically, and a fixture tests the boundary:
  - **pause cut** after last word ℓ: closed at `end_ℓ + 0.4 + ε_w`, and only if no word starts in (end_ℓ, end_ℓ + 0.4];
  - **speaker change:** `end` of the next speaker's first word + ε_w;
  - **length cut:** `end` of the crossing word + ε_w;
  - a frame is also not closed until every word's `role_avail_s ≤ close_s`.
- At *t*, **closed frames** are those with `close_s ≤ t`. The **tail** is the available words after the last closed frame.
- Closed frames are embedded once. Tails are embedded on demand with a sha1 cache, since consecutive points share tails.

### 5.3 Text normalisation and windows at *t*
**Normalisation order:**
1. Mask names (§5.1, needs casing).
2. Lowercase and strip punctuation.
3. Delete CALLON phrases (§5.5) and every `[name]` token from **feature** text, for every target.

The un-deleted masked text is kept in `words.text_masked` for the CALLON and N-name diagnostics only. F-name keeps `[ME]`/`[NAME]` tokens; F-punct keeps punctuation and casing.

| Window | Content (available, normalised words; role tags `<P> <S> <M> <U>` at speaker changes, `<C>` for chat) | Used by |
|---|---|---|
| `X_all15` | all words in (t − 15, t − ε_w] | TF-IDF, NB |
| `X_all60` | all words in (t − 60, t − ε_w] with role tags | TF-IDF, NB, M6 |
| `X_prof60` / `X_stud60` | professor / other-student words in (t − 60, t − ε_w] | TF-IDF |
| `X_lastprof` | the professor's most recent utterance (split at a pause > 1.2 s or a non-prof word), truncated at t − ε_w | question detector |
| `X_chat60` | chat in (t − 60, t] (human actors only) | TF-IDF |
| `X_me_sofar` | Brady's frames with `close_s ≤ t` | U(t), `own_sim` |

### 5.4 Embeddings: causal recency pooling
`bge-small-en-v1.5` (33M parameters, 384-d, CLS pooling, L2-normalised) via `sentence-transformers` on `mps`, batch 128, float16 cache, on normalised text. Fallback: `all-MiniLM-L6-v2`.

$$
v_\tau(t)=\frac{\sum_{f\in F(t)} e^{-(t-\text{close}_f)/\tau}\,e_f}{\big\lVert\sum_{f\in F(t)} e^{-(t-\text{close}_f)/\tau}\,e_f\big\rVert},\quad F(t)=\{f:\ \text{close}_f\le t,\ t-180<\text{close}_f,\ \text{role}_f\neq\text{me}\}\cup\{\text{tail}\},\ \ \tau\in\{15,60\}
$$

- `e_last`: the most recent non-me frame or tail.
- `e_lastprof`: the embedding of `X_lastprof`.

### 5.5 Professor-question detector and rule baseline R
Runs on `X_lastprof` once ≥ 0.8 s of silence has been observed after its last word (t − ε_w − last_end ≥ 0.8). The silence is measured from word onsets, which are perceptible.

**Question test** (headline, no punctuation): `is_q` holds when either
- the first token is a wh-word or an inverted auxiliary (`what|why|how|which|who|where|when|is|are|do|does|did|can|could|would|should|have|has`) and the utterance has ≥ 4 words; or
- it ends in a tag (`right|yeah|okay|isn't it|don't you think`).

The "ends in `?`" rule exists only in F-punct.

**Lexicons** (case-insensitive, on normalised masked text):

| Lexicon | Pattern | Use |
|---|---|---|
| INVITE | `\b(any(one\|body)\|some(one\|body)\|who (wants\|would like\|'d like\|can\|has)\|any (thoughts\|comments\|questions\|volunteers)\|thoughts on\|what do (you\|we\|people)( all)? think\|would anyone\|does anyone\|can someone\|let'?s hear\|what did your group\|how would you\|why do you think\|anything else\|what else\|build on\|jump in\|feel free)\b` | feature |
| VOTE | `raise your hands? if\|show of hands\|hands up if\|put your hands? up if\|thumbs up\|let'?s vote` | `q_vote_lex`; vote typing (§2.5) |
| CALLON | `go ahead\|i see (a\|your) hand\|you'?re up\|unmute\|take it away` | **deleted from feature text** with all `[name]` tokens; used only for the CALLON check (§9) |

**Outputs:**
- `q_open`: the last professor utterance is a question, and no student frame of ≥ 3 words has come since.
- `q_age_s`, `q_inv_age_s` (∞ → capped at 300).
- `q_kind ∈ {invite, check ("make sense", "any questions"), rhetorical (the professor resumes within 3 s; known only after 3 s), plain}`. There is no cold-call category, because names are not in feature text.
- `q_count_300`, `q_vote_lex`.

**Validation (must, Wed, 40 min).** Brady hand-labels 200 random professor utterances from 10 non-vault dev classes for {question, open invitation}. Report precision and recall for the headline and F-punct detectors. LLM labels are never used.

**R (no training):**

$$
r(t)=\big(0.5+0.5\,q_{\text{open}}(t)\big)\cdot\exp\!\big(-\min(q_{\text{inv\_age}},\,2\,q_{\text{age}})/10\text{ s}\big)
$$

### 5.6 PCW features (candidate novel features)
**Parsing.**
- Prompts $Q_b$ are the concatenated `insert` strings of the template's Quill `ops`, per block.
- Answers $A_b$ come from the doc, keyed by block uuid, from whitelisted paths only: text; code comments and identifiers (split on `_` and camelCase); upload **filenames**. Could-tier: extract text from `.ipynb`/`.py`/`.md` uploads with `nbformat`.
- Answers are split into chunks $a_j$ of ≤ 120 words with 30-word overlap.

**PCW status** is class-level and decided by QA before modelling. It is **never a feature value**.

| `pcw_status` | Meaning |
|---|---|
| `asof_start` | doc rebuilt as of class start from a revision/history endpoint (D8) |
| `verified` | doc last-modified ≤ class start, **and** D8 showed that the POST and GETs do not modify it |
| `git` | CS156 only: the repo's PCW notebook as of `git show $(git rev-list -1 --before=<class start UTC> HEAD):<path>` |
| `empty` | an `asof_start`, `verified` or harvest-created doc with no answered block |
| `none` | no template, or no doc existed before class start (no commit before start, for `git`) |
| `unknown` | modified after start with no history, or the edit time is unreadable or bumped by reading |

**Branches**, decided by D8 on Monday night and frozen in the prereg:
- **Branch H (history endpoint exists).** Every class has an as-of-start doc, so `pcwA` is pre-class information for every class. `pcw_has` = 1 when the as-of-start doc has ≥ 1 answered block. F-all includes `pcwT` and `pcwA`.
- **Branch N (no history).** The headline F-all includes `pcwT` only (prompts, learning outcomes and study guide are known before class for every class). `pcwA` exists only on the **PCW subset** S_pcw, the classes with status ∈ {verified, git} and ≥ 1 answered block. H3, A-mismatch and F-pcw are trained and tested on S_pcw only. `unknown` classes are counted in the funnel and never imputed.
- **If reads bump last-modified and there is no history**, no class can be verified. The `pcwA` analysis becomes an exploratory sensitivity row labelled "edit time unverified", and H3 leaves the family.

**Coverage rule** (pre-registered). H3, A-mismatch and F-all−pcw are evaluated on the `pcw_has = 1` dev classes (Branch H) or on S_pcw (Branch N). If there are fewer than 40 such classes or fewer than 150 bids, H3 is exploratory and leaves the Holm family (H2, H4 and H5 remain), and §14 labels Δρ and U as underpowered. The funnel reports PCW coverage by course.

**Similarity normalisation.** bge-small cosines sit in a compressed band, where unrelated texts often score 0.4–0.6. So every similarity is z-scored against a null:
- For an embedding *u* in class *c*, $\mu_u,\sigma_u$ are the mean and sd of $s(u,r)$ over a reference set $R_c$: 200 chunks (fixed seed) from other courses' PCW prompts and verified answers, **of classes that started before c**.
- $z(u,v)=(s(u,v)-\mu_u)/\sigma_u$; NaN when $|R_c|<50$.
- No labels are used, so there is no fold dependence; using only earlier classes keeps contract rule (a).

**Definitions** (with $x=v_{60}(t)$):

$$
b^*_Q(t)=\arg\max_b z\big(x,e(Q_b)\big),\qquad b^*(t)=\arg\max_b z\big(x,e(Q_b\oplus A_b)\big),\qquad
\rho_A=z\big(x,e(A_{b^*})\big),\ \rho_Q=z\big(x,e(Q_{b^*})\big),\ \Delta\rho=\rho_A-\rho_Q
$$

$$
U(t)=\max_{j\in\text{unspent}(t)} z(x,a_j),\qquad \text{unspent}(t)=\Big\{j:\ \max_{m\in M_{\le t}} z(a_j,m)<\kappa\Big\},\ \ \kappa=2
$$

Here $M_{\le t}$ is Brady's frames with `close_s ≤ t`; every chunk is unspent if he has not spoken.
- **Δρ**, the resonance gap: the room is on what *he* wrote, not just on the question everyone saw.
- **U(t)**, unspent preparation: the room has reached something he prepared and has not yet said.

**Columns:**
- `pcwT` (template side):
  - `pcw_tmpl_has`;
  - `pcw_res_Q` = $\max_b z(x,e(Q_b))$;
  - `pcw_block_pos_Q` = $b^*_Q/B$;
  - `pcw_new_block`: 1 the first time $b^*_Q$ switches to a block whose $\rho_Q$ is at least the **training-fold 90th percentile** of $\rho_Q$;
  - `lo_sim`, `sg_sim`: z-scored maximum similarity to the learning outcomes and to the study-guide paragraphs.
- `pcwA` (answer side):
  - `pcw_has`, `pcw_my_words`, `pcw_frac_answered`, `pcw_n_code`;
  - `pcw_res_A`, `pcw_res_gap` (Δρ), `pcw_unspent` (U);
  - `pcw_res_ref`: $\max_j z(x,a_j)$ minus the mean, over the **5 most recent earlier classes of the same section** with an `asof_start`/`verified` doc, of $\max_{a\in A_{c'}} z(x,a)$; NaN when there are fewer than 3;
  - `pcw_bm25_gap`: the `rank_bm25` lexical twin of Δρ;
  - `pcw_invest` = log(1 + words($A_{b^*}$)); `pcw_block_pos` = $b^*/B$.

**Template version.** If sle exposes template versions or `updated_at`, use the version current at the class date. Otherwise set `template_asof = "harvest"` and run a sensitivity without A13 (F-all−pcwT).

### 5.7 Feature groups

| Group | Columns |
|---|---|
| `ctx` | `ctx_min`; `ctx_min_spl_0..4` (`SplineTransformer`, knots at 10/25/45/65/80); `ctx_course` (categorical; courses with **< 3 training classes** → `other`, using a custom in-fold class-count filter); `ctx_area` (CS/SS/NS/AH/B/other); `ctx_sem_idx`; `ctx_session_no` (full schedule); `ctx_session_frac`; `ctx_dow`; `ctx_hour_local`; `ctx_hour_utc`; `ctx_enrolled`; `ctx_n_present`; `ctx_seg_type` (categorical, regex below); `ctx_seg_elapsed_s`; `ctx_seg_idx`; `ctx_since_breakout_s`; `ctx_after_breakout`; `ctx_in_poll` |
| `spk` (Forum voice, no words) | `spk_cur_role`, `spk_cur_elapsed_s`, `spk_silence_s`, `spk_prof_share_60`, `spk_prof_share_300`, `spk_stud_turns_60`, `spk_stud_turns_300`, `spk_distinct_stud_300`, `spk_turns_since_prof`, `spk_cur_in_my_group`. Brady's role uses word-bearing events only |
| `me` | `me_turns_so_far`, `me_since_turn_s`, `me_words_so_far`, `me_chat_n`, `me_since_chat_s`, `me_breakout_talk_s`; word-bearing events and frames with `close_s ≤ t` only |
| `me_hand` | `me_hand_n_so_far`, `me_hand_since_down_s`, `me_hand_last_outcome` (available at offset + 5 s); eligible episodes only (§2.4) |
| `poll` | `poll_n_so_far`, `poll_since_close_s`, `poll_my_ans_words`, `poll_res_my` (z(x, e(my last available answer)); NaN before it is available) |
| `q` | §5.5 columns (no `?` rule) |
| `text` | **sparse:** TF-IDF of role-prefixed tokens (`p_`, `s_`, `m_`, `u_`, `c_`) over normalised `X_all60`, `X_all15`, `X_chat60`; 1–2-grams; `min_df` = token present in ≥ 5 training classes; `sublinear_tf`; `max_features` 30k. **Dense:** `n_words_15`, `n_words_60`, `wpm_60`, `filler_rate_60`, `frac_unk_60`; `topic_shift` = 1 − s(v over the last 60 s, v over 60–300 s ago); `own_sim` = z(x, mean $e_f$ over Brady frames with `close_s ≤ t`), NaN if none, computed through the same `searchsorted` cutoff that CausalView checks |
| `emb` | M4: [v_15, v_60, e_last, e_lastprof] (1,536-d). M3: PCA-32(v_60) + PCA-16(e_last) + SVD-64(TF-IDF), all fit in-fold |
| `pcwT`, `pcwA` | §5.6 |

**Feature sets:**

| Set | Contents |
|---|---|
| F-struct | `ctx + spk + me + me_hand + poll` (no words) |
| F-text | `text + q + emb + ctx_min` |
| **F-all** (headline) | F-struct + `q + text + emb + pcwT` (+ `pcwA` in Branch H) |
| F-all−me_hand / F-struct−me_hand | without `me_hand` (MVP; H1 co-requirement; T-speech uses F-all−me_hand) |
| F-all−pcw / F-all+pcwA | without `pcwT`/`pcwA` / with `pcwA` (H3, PCW subset only) |
| F-pcw | `ctx + pcwT + pcwA` (PCW subset) |
| F-all−me | without `me` and `me_hand` |
| F-name | F-all with `[ME]`/`[NAME]` tokens kept in feature text |
| F-punct | F-all with punctuated, cased text and the `?` rule (generalises Qnp to every text group) |
| F-min | `ctx_min` splines only (B-min reference) |

**`seg_type` regex** (first match on `section_title + " " + segment_title`):

| `seg_type` | Pattern |
|---|---|
| `breakout_instr` | `breakout instr` |
| `breakout` | `\bbreakouts?\b` |
| `break` | `eye rest\|\bbreak\b` |
| `poll` | `poll\|vote\|quiz` |
| `debrief` | `debrief\|report back\|share out\|discussion` |
| `intro` | `intro\|overview\|recap\|review\|warm` |
| `wrapup` | `wrap\|reflect\|summary\|closing\|conclu` |
| `activity` | `activity\|exercise\|problem\|case` |
| `other` | anything else |

**Encoding** (all fit inside each fold's sklearn `Pipeline`; features stored as float32):
- Durations: capped at 3,600 s, `log1p`, `StandardScaler`.
- `ctx_course`: `MinClassCountCategories(min_classes=3)` maps rare courses to `other` by counting **training classes, not rows**. Then `OneHotEncoder(handle_unknown="ignore")` for linear models, and native categoricals in HGB (`categorical_features="from_dtype"`).
- Other categoricals: one-hot or native.
- Missing values: indicator column plus median fill for linear models; native NaN in HGB.
- NB gets context as pseudo-tokens: `ctx_seg_debrief`, `ctx_min_30_45`, `ctx_sinceme_5_10m`, `ctx_qopen`.
- Section-level constants (`ctx_area`, `ctx_enrolled`, `ctx_sem_idx`) stay in F-struct, so structure can identify classes as well as text can.
- The F-text vs F-struct gap is reported **per course**.

---

## 6. Models

| ID | Model | Input | Library / settings | In class? | Tier |
|---|---|---|---|---|---|
| AO | Always-on | — | p ≡ 1 (FROC reference point) | — | MVP |
| B-min | Minute-only LR | F-min | as B1 (FROC reference curve) | yes | MVP |
| B0 | Prevalence | — | training-fold π | — | MVP |
| R | Question rule | `q_*` | §5.5, no fitting | — | MVP |
| B1 | Structure-only LR | F-struct | `LogisticRegression(penalty="l2", solver="lbfgs", C∈{0.01,0.1,1}, max_iter=2000)` | yes | MVP |
| M1 | Naive Bayes | role-prefixed counts of normalised `X_all60`, `X_all15`, `X_chat60` plus context pseudo-tokens | `MultinomialNB(alpha∈{0.1,0.3,1})` | yes | MVP |
| M2 | Sparse LR | TF-IDF ⊕ scaled dense (F-all minus `emb`) | `LogisticRegression(solver="lbfgs", C∈{0.03,0.1,0.3}, max_iter=3000)`. **C chosen once per outer fold on F-all and reused** for every M2 ablation and sweep | yes | MVP |
| **M3** | **Gradient boosting** (headline) | dense F-* + PCA/SVD text (§5.7) | `HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, max_features=0.8, categorical_features="from_dtype", early_stopping=False, max_iter=n*)`. **n* ≤ 500 chosen once per outer fold on F-all**, on inner `GroupKFold(3)` by `staged_predict_proba` log-loss (training and validation curves are both kept as learning figures), and reused for every ablation and sweep | **no** | MVP |
| M4 | Frozen sentence embeddings + LR | `emb` (1,536) ⊕ all dense groups | `sentence-transformers` ≥ 3 + bge-small on `mps`; LR (lbfgs) with C chosen once per outer fold. Could: `MLPClassifier((128,), alpha=1e-3, early_stopping=False)` | embeddings **no** | must |
| M5 | Streaming GRU hazard | one sequence per class over **every** grid step, with no compaction, zero-filling or mask/at-risk channel. Step input = causal inputs computed exactly as for M3 at that step: PCA-64 of the mean $e_f$ of frames closing in (t − 2, t] ⊕ seconds of speech per role in the last 2 s ⊕ `q_open`, `q_kind` ⊕ `ctx`/`me` scalars. Causal and label-side exclusions enter **only the loss mask** (BCE on at-risk defined steps) | PyTorch `nn.GRU(in, 64)`, dropout 0.3, linear head; truncated BPTT on 300-step (10-min) chunks; AdamW 1e-3, wd 1e-2; ≤ 30 epochs, early stop on inner AP. Causality is checked by `test_gru_inputs`, not assumed | **no** | should |
| M6 | Fine-tuned cross-encoder | `"[{area}] [{seg_type}] [min {m}] [since me {k}m] \|\| {normalised X_all60, left-truncated to 224 tokens} \|\| PCW: {best a_j, ≤ 64 tokens}"` | `transformers` `AutoModelForSequenceClassification("sentence-transformers/all-MiniLM-L6-v2", num_labels=1)`, max_len 320, lr 3e-5, 2 epochs, batch 32, 6% warmup, `mps`; all positives + negatives subsampled at r = 0.2, with prior correction $\text{logit}\mathrel{+}=\log r$ | **no** | could |

**Imbalance.** No resampling and no class weights for B1–M5:
- AP and the false-alarm thresholds are rank- or rate-based;
- SMOTE or undersampling would distort the probabilities that calibration relies on.

Only M6 subsamples negatives, with the prior correction above.

**Calibration.** Platt scaling inside each outer fold, fit on the inner OOF predictions (as `CalibratedClassifierCV(ensemble=False, cv=GroupKFold(3))`). `p_cal` is used for pooling AP across folds and for ECE. The reliability diagram is should-tier.

**Capacity** (about 500 dev bids):
- Dense HGB or LR on about 150 features is fine with `min_samples_leaf=100`.
- 30k TF-IDF features need strong L2.
- Frozen embeddings plus a linear head are the right size.
- M6 is not expected to beat M4. The learning curve (AP vs number of training classes: 20/40/80/all; should-tier) says whether more data would help.

**sklearn hygiene.** Never use internal random validation splits (`validation_fraction`, default `CalibratedClassifierCV(cv=5)`), because they split points within a class. Every inner split is `GroupKFold` by class.

**LLM policy.**
- No generative model produces labels, features, predictions or analysis text.
- Deepgram is used as a measurement instrument, only for courses covered by consent (§10). Its WER is spot-checked on 2 × 3-min hand transcripts with `scribe_core.metrics.wer` (could).
- Groq is used only if the consent names it; otherwise it is not a fallback.
- Encoders (bge, MiniLM) are local and deterministic.
- Claude does engineering only, disclosed in the AI-use section following the repo convention.
- Nothing sends classmates' words to an LLM.

---

## 7. Comparison conditions

Defaults: hand target, H = 15, g = 2, S-group split, F-all, names masked, normalised text, model M3.

**Tiers:**
- **MVP:** finished by Wed 20:00 (gate G3).
- **must:** finished by Thu 18:00 if G3 passed.
- **should**, **could:** in cut order (§12).

| ID | Condition | Varies | Answers | Tier |
|---|---|---|---|---|
| K0 | B0, R, B1, M1, M2, M3 (plus AO and B-min reference curves) | model family | does anything beat the rule? (H4) | **MVP** |
| F0 | F-struct vs F-all; F-struct−me_hand vs F-all−me_hand (M3) | feature groups | H1 and its co-requirement | **MVP** |
| TG | global and local prediction shifts on every row (§8.4) | — | H2: the class vs the phase vs the moment | **MVP** (no compute) |
| S | S-group vs S-rand (M2, M3) | split | leakage tax = AP_rand − AP_group | **MVP** |
| EV | event metric (firing convention) with AO, B-min, shifted and R references | — | S3 | **MVP** |
| P-clock | offset-sensitive clock control (§9.2) | positive control | words, roles and video aligned; **stop rule** | **MVP** (no model) |
| P-plant | token `zqxw` planted 8 s before 30% of bids (training and test), M1/M2 | positive control | features and model can see a planted cue; **stop rule** | **MVP** |
| CALLON | CALLON rate in [o − 5, o) vs matched points | negative control | are labels late against the audio? | **MVP** (no model) |
| K1 | M4 | model | H5 | must |
| F1 | F-text; F-all−pcw and F-all+pcwA on the PCW subset | feature groups | H3 | must |
| A-mismatch | retrain **and** test with deranged PCW (§9.2), ×5 derangements | negative control | does *matching* preparation matter? | must |
| G | g ∈ {2, 5} on M3/F-all | guard | own-precursor and clock test (§9) | must |
| IH+ | information horizon δ ∈ {+30, +60} s, **including Brady's own words**, M2/F-text | positive control | the pipeline sees the signal a called bid leaves | must |
| N-name | descriptive (§9.2) | negative control | name mentions around bids | must |
| V | vault: M3/F-all, M3/F-all−me_hand, M3/F-struct, R, B1, scored once | holdout | untouched estimate | must (cut last) |
| M5 | GRU | model | does sequence modelling help? | should |
| F2 | F-pcw; F-all−me; F-name; **F-punct**; M2 mirrors of F | features | preparation alone; "my own history"; name and punctuation leak size | should |
| S2 | S-time (train sem ≤ 5 = F23–F25, test 6–7 = Sp26, F26); S-loco (leave one course out, courses with ≥ 15 bids) | split | stability over years and courses | should |
| Hs | H ∈ {5, 10, 30, 60} | horizon | how far ahead the cue reaches | should (min {5, 30}) |
| IH | full information horizon δ ∈ {−20, −10, −5, −2, 0, +2, +5, +10} (+5 and +10 descriptive) | cutoff | where the signal arrives (illegal side shaded) | should |
| G01 | g ∈ {0, 1}, only if re-timing ran with `retime_failed` < 10% | guard | own pre-raise behaviour | should |
| T | T-speech vs hand, at matched FAR | target | is volunteering more or less anticipatable than speaking? | should |
| A-stream | streaming re-transcription audit (§9) | ASR causality | how much batch look-ahead inflates AP | should (cut to 60 points late) |
| LS | label-shift retrain: labels 1, 2, 3 s earlier, M2 and M3 | labels | does signal pile up right at the onset? | should |
| Live | live-logger key-press → icon lag | label timing | direct label lateness | should |
| Sens | calibration residuals ≤ 3 s; symmetric observability; votes as positives / Δn-only / contested as bids / 0.5 threshold; `me_*` with no-word blips; classes with audited precision ≥ 0.9; unvalidated-template classes removed; multi-session included; F-all−pcwT (template as of harvest); `hand_preceded_share` ≥ 0.3 sections; poll-tick NA rule on/off | population and label rules | headline robustness | should |
| LC | learning curve on training-class count | data | data limit or ceiling? | should |
| Could | M6; N-stale (text features from t − 600 s); A-jitter; A-attrib (Scribe fused/cluster roles); fold-aware `hist_*`; turn-end grid; M4-MLP; WER | — | breadth | could |

**Pre-registered hypotheses** (frozen in tag `prereg`). The primary statistic is **within-class**: $\text{AP}_w$ (§8.2), with paired cluster-bootstrap CIs.

| ID | Claim | Test |
|---|---|---|
| **H1 (primary)** | content adds beyond structure, within a class | ΔAP_w(M3/F-all − M3/F-struct) has 95% CI lower bound > 0, **and** ΔAP_w(F-all−me_hand − F-struct−me_hand) has 95% CI lower bound > 0 (intersection-union test; no α split). Secondary: pooled ΔAP and ΔTG_local |
| H2 | the model knows the moment, not just the phase of class | TG_local(M3/F-all) CI lower bound > 0 **and** ΔTG_local(F-all − F-struct) CI lower bound > 0 (intersection-union). Global TG is reported as the "which class" floor |
| H3 | preparation adds | ΔAP_w(with − without PCW) on the PCW subset; exploratory if the coverage rule fails (§5.6) |
| H4 | learned beats the rule | ΔAP_w(M3/F-all − R) > 0 |
| H5 | embeddings beat TF-IDF | ΔAP_w(M4 − M2) > 0 |

**Multiplicity: fixed-sequence gatekeeping.** H1 is tested at α = 0.05. Only if H1 rejects are H2–H5 tested, Holm-corrected at α = 0.05 (H3 only if powered). The familywise error rate is ≤ 0.05. Everything else is exploratory. B1 vs M3/F-all is a secondary comparison.

---

## 8. Splits and metrics

### 8.1 Splits
- **Vault.** `in_vault = int(sha1(f"{class_id}:oncue-vault").hexdigest(), 16) % 5 == 0`, about 20% of classes, random across sections and semesters.
  - The rule is committed in `config.py` on Monday night, before any label exists.
  - `select.py` sets `in_vault`. **Every dev tool filters `in_vault == 0` and asserts it** (`assert_no_vault(df)`): pilots, Phase A audits, detector tuning, lexicon development, question-detector labels, masking checks, peri-onset descriptives and QA dashboards.
  - Vault classes are transcribed and scanned like the others, but **only their funnel counts** are looked at before they are opened, once, at Thu 19:00, after the G4 freeze, with models refit on all dev classes.
- **S-group (headline).** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=156)`, `groups=class_id`, stratified on course code so courses are balanced across folds. One seed. The same folds are used for every condition (`points.fold`).
- **Inner.** `GroupKFold(3)` on the training classes. It is used for hyperparameters (once per outer fold, on F-all), HGB `n*`, GRU early stopping, Platt calibration, and event thresholds. The inner OOF predictions at the chosen hyperparameters are kept.
- **S-rand.** `KFold(5, shuffle=True, random_state=156)` over points. Reported only to measure inflation.
- **S-time / S-loco**, as in §7.
- **Fitted in-fold only:** TF-IDF vocabulary and IDF, scalers, splines, the course class-count filter, one-hot levels, PCA, SVD, the `pcw_new_block` percentile, Platt, thresholds, the `pcw_bm25` IDF.

### 8.2 Point-level metrics

$$
\text{AP}=\sum_n (R_n-R_{n-1})P_n,\quad \text{lift}=\text{AP}/\pi,\quad
\text{AP}_w=\frac{\sum_c K_c\,\text{AP}_c}{\sum_c K_c},\quad
\text{Brier}=\tfrac1N\sum(p_i-y_i)^2,\quad \text{ECE}=\sum_{b=1}^{15}\tfrac{n_b}{N}\lvert\bar y_b-\bar p_b\rvert
$$

- **$\text{AP}_w$ (primary for every hypothesis):** within-class AP over classes with ≥ 1 bid, weighted by the class's bid count $K_c$. It ignores which classes Brady talks in, and each class is scored by a single fold model, so fold score scales never mix.
- **Pooled AP:** computed on Platt-calibrated `p_cal` within each fold before pooling, and reported next to the **mean per-fold AP ± sd**.
- Macro-AP; lift; ROC-AUC is secondary (it flatters at π ≈ 2%); ECE uses equal-mass bins on `p_cal`.

### 8.3 Event-level metric (novel for the course)
This follows the seizure-prediction convention (Winterhalder et al. 2003; Mormann et al. 2007).

**Grids.**
- The **causal grid** $\mathcal G$ is every decision point not removed by a *causal* rule (`opening`, `rec_gap`, `breakout`, `break`, `me_speaking`, `hand_up`, `post_lowering`, `post_glitch`) and with t + H ≤ T_end.
- Label-side rules (`guard`, `onset_pending`, `vote_or_censored`, `glitch_window`, `unobservable`) **never** remove points from $\mathcal G$, so alarms are never cut just before a bid.
- $\mathcal G_{\text{def}}\subseteq\mathcal G$ is the set of points with a defined label.

**Firing rule.** Firings are built in time order:

$$
\mathcal F_\tau=\Big\{s\in\mathcal G:\ p(s)\ge\tau\ \wedge\ \nexists\, s'\in\mathcal F_\tau\cap\big(s-(H-g),\,s\big)\Big\}
$$

- A firing at *s* claims the occurrence period $(s+g,\ s+H]$.
- An alarm still above τ after H − g re-arms and counts as a new firing.
- A firing is **true** if y(s) = 1, **false** if y(s) = 0, and **neutral** if y(s) is undefined.

**Metrics.**

$$
\text{FAR}(\tau)=\frac{\#\{s\in\mathcal F_\tau: y(s)=0\}}{|\mathcal G_{\text{def}}|\,\Delta/60}\ \ [\text{per at-risk minute};\ \times 60\text{ per hour}],\qquad
\text{Recall}(\tau)=\frac1K\sum_k \mathbb 1\big[\mathcal F_\tau\cap[o_k-H,\ o_k-g)\neq\emptyset\big]
$$

$$
\text{lead}_k=o_k-\min\big(\mathcal F_\tau\cap[o_k-H,\ o_k-g)\big),\qquad
\text{TiW}(\tau)=\frac{\big|\{t\in\mathcal G:\ \exists s\in\mathcal F_\tau,\ t\in(s+g,s+H]\}\big|}{|\mathcal G|}
$$

- Only **evaluable** bids count: ≥ 50% of the grid in $[o_k-H, o_k-g)$ lies in $\mathcal G$.
- **Time-in-warning (TiW)** is reported next to every recall.

**Chance.**
- **Primary: empirical.** Run the same firing rule on globally shifted predictions (§8.4) and read their recall at the model's realised FAR:

  $$
  \text{EventLift}(\lambda)=\frac{\text{Recall}(\tau_\lambda)}{\text{Recall}_{\text{shift}}(\text{FAR}_{\text{realised}})}
  $$

- **Secondary: analytic.** Because of the refractory period H − g, warning windows are disjoint. So any firing process at rate $\lambda_f$ (all firings per at-risk minute) that is independent of the onsets has expected recall equal to its time-in-warning:

  $$
  R_0(\lambda_f)=\min\big(1,\ \lambda_f\,(H-g)/60\big),\qquad \text{EventLift}_0=\text{Recall}/R_0(\lambda_f)
  $$

  The Poisson form $1-e^{-\lambda(H-g)/60}$ is the no-refractory version and is within 6% for $\lambda_f\le0.5$/min. At H = 15, g = 2: $R_0$ = 0.022 at 0.1/min and 0.108 at 0.5/min.
- **Always-on** fires every Δ⌈(H − g)/Δ⌉ = 14 s, i.e. about 4.3 firings per minute. Its recall is about TiW ≈ 0.93, and its EventLift is about 1. Over-alarming is therefore penalised.
- **Every FROC plot** shows the model curve plus AO (a single point), B-min (`ctx_min` only), globally and locally shifted predictions, and R.

**Headline numbers.** Recall (with realised FAR and TiW), EventLift (vs shift), EventLift_0 and median lead, at λ = **0.1/min** (6/h, about the bid rate) and **0.5/min** (30/h). Also pAUFROC on [0, 0.5]/min.

**Thresholds (rate-matched).**
1. On the inner OOF predictions, find the firing rate $\lambda_f^*$ at which FAR = λ.
2. On the training fold, choose $\tau_\lambda$ so that the **outer model's** firing rate on training-fold $\mathcal G$ equals $\lambda_f^*$.
3. Apply $\tau_\lambda$ to the test fold and report the realised FAR next to recall.

The oracle-τ value read off the test curve is shown second, labelled as such. FROC curves are drawn as the lower envelope over τ.

**`tests/test_metrics.py`:**
1. p ≡ 1 on synthetic contiguous grids: the firing interval is 14 s; FAR = (60/14) × (share of firings that are false), to 1e-9; recall ≤ TiW + 0.02 over 500 random onset placements; EventLift_0 ∈ [0.9, 1.1]; EventLift vs shift = 1.
2. $\mathcal F_\tau$ is identical when labels are recomputed (firing never reads labels).
3. Random p independent of onsets, over 200 synthetic classes: mean EventLift_0 ∈ [0.9, 1.1].
4. An oracle predictor (p = 1 exactly on $[o-H, o-g)$): recall 1, FAR 0.
5. A guard point can fire and is counted as neutral.

### 8.4 Timing decomposition (H2)
- **Global shift ("which class" floor).**
  - Let $\mathcal T_c$ be class *c*'s at-risk defined points in time order, with $n_c$ points.
  - Shift predictions circularly by $u_c\sim U\{150,\dots,n_c-150\}$ **indices** (≈ 5 min at Δ = 2 s), keeping labels fixed. Repeat 20 times.
  - Skip classes with $n_c<400$ and report how many were skipped.
  - $\text{TG}=\text{AP}-\text{AP}_{\text{shift}}$. $\text{AP}_{\text{shift}}>\pi$ is legitimate class-level variation. A global shift breaks alignment with minute and segment, so TG > 0 is expected even for F-struct; it is a floor, not evidence about the moment.
- **Local shift ("which moment").**
  - Partition $\mathcal T_c$ into maximal runs inside one timeline segment, then into consecutive 90-point (3-min) blocks.
  - A block with < 30 points is merged into the previous block of the same segment, or left unshifted if it is alone (the share is reported).
  - Within each block of size $n_b$, shift circularly by $u_b\sim U\{15,\dots,n_b-15\}$ (≥ 30 s). Repeat 20 times.
  - $\text{TG}_{\text{local}}=\text{AP}-\text{AP}_{\text{local}}$. This keeps class, segment type and minute (to within 3 min), and breaks only second-scale alignment.
- **Claim "the moment" only if** $\text{TG}_{\text{local}}(\text{F-all})>0$ **and** $\Delta\text{TG}_{\text{local}}(\text{F-all}-\text{F-struct})>0$ (H2). Within-class versions ($\text{TG}_w$) are reported alongside.

### 8.5 Uncertainty, pre-registration, success criteria
- **Cluster bootstrap** over classes, B = 2000. Percentile 95% CI for AP_w, AP, TG, TG_local, Recall@λ, lead. **The CIs are conditional on the fitted fold models**: they ignore training variability, and the write-up says so.
- **Paired** deltas use the same resampled classes. Report the CI and P(Δ ≤ 0).
- **Detectable effect.** For each within-class Δ, report $\text{MDE}_{80}\approx(z_{0.95}+z_{0.80})\cdot\text{SE}_{\text{boot}}=2.49\,\text{SE}$, so a null reads as "no within-class effect larger than X".
- **Every row** carries `n_classes`, `n_onsets`, π.
- **Pre-registration.** On Wed 14 Oct at about 14:00, before the first full dev CV run, the "Pre-registration" section of `PLAN.md` fixes:
  - Δ, H, g, g_s, ε_w, the ε_F rule, and the calibration, A/V and multi-session QA rules;
  - the at-risk and label rules (asymmetric observability, vote typing, glitches);
  - masking and text normalisation;
  - the PCW branch (H or N) and the coverage rule;
  - feature sets, models, grids, and the reuse of n*/C from F-all;
  - the primary statistic (AP_w), H1–H5, the fixed-sequence procedure, and the MVP/must tiers;
  - the vault rule, the G2 pass rules (already committed Monday), the T-speech fallback spec;
  - the success criteria below.

  It is committed and tagged `prereg` (`git tag prereg && git push --tags`), and the write-up cites the hash. Any later change is logged as a deviation.
- **Success criteria:**
  - (S1) M3/F-all pooled lift ≥ 2 and TG_local CI > 0;
  - (S2) H1 rejected, including its co-requirement;
  - (S3) Recall@0.1 (at its realised FAR) has a CI lower bound above both the shifted-prediction recall at matched FAR and R's recall at matched FAR.

---

## 9. Leakage traps and controls

### 9.1 Traps

| # | How it could cheat | Defence | Check |
|---|---|---|---|
| L1 | words after *t* in a window | word availability `end + ε_w`; reference path | truncation test (blocking) |
| L2 | Scribe turns or utterances straddle *t* | words and causal frames only | straddle unit test |
| L3 | frame boundaries reveal the future (a pause "completes" after *t*) | `close_s` uses word onsets ≤ t − ε_w; tail | sentinel inside (t − ε_w, t]; pause-cut fixture |
| L4 | ongoing voice-event duration | clipped at *t*; elapsed only | unit test |
| L5 | role from an event that starts after the word (EDGE 0.25 s) | `role_avail_s` | sentinel professor event at t − ε_F/2 |
| L6 | `smooth()` and cluster mapping use the future | Forum-causal roles; A-attrib could | allowlist excludes `how` and clusters |
| L7 | batch-ASR look-ahead: punctuation, casing, word choice | headline text has no punctuation or casing. **A-stream**: for 300 sampled points (150 with y = 1, 150 with y = 0), cut the audio [t − 180, t] (`ffmpeg -ss t−180 -t 180` → wav) and transcribe it with `scribe_core.asr_deepgram.DeepgramASR` using the same params (≈ 900 min ≈ $4.50, consented courses only). Splice the stream words into the batch word history in place of the batch words for [t − 180, t], then recompute roles, masking, normalisation, frames and **every** feature. `topic_shift`'s 60–300 s window uses batch words before t − 180, documented. Re-score with the frozen fold model | Spearman(p_full, p_stream); AP_w on the subset both ways with a paired CI; punctuation and casing disagreement rates; F-punct ablation |
| L8 | the professor reacts to the hand ("Brady, go ahead", "I see a hand") | hand-up exclusion; 10 s post-lowering refractory; g = 2; CALLON phrases and name tokens deleted from feature text | **label lateness measured directly:** CALLON check (MVP); live logger (should); label-shift retrain (should). The g-sweep is reframed as an own-precursor and clock test |
| L9 | A/V skew, rendition mismatch or calibration drift | one rendition; pts timing; per-class lag_c; calibration QA (windowed, ≥ 6 windows, p95); ε_F from p95 | P-clock; lag distribution; A-jitter |
| L10 | future class structure | forbidden list; `segments` has no end column | allowlist; grep test for `actual-length-seconds`, `recording-ended`, `end` in `features.py` |
| L11 | post-class artefacts inside allowed files | **per-source JSON-path whitelists**; unknown keys dropped and logged | `test_loader_allowlist` (provenance of every column) |
| L12 | PCW edited during or after class | `pcw_status` decided before modelling; `pcwA` only from verified pre-class docs; no `pcw_edited*` feature | funnel counts by status; sensitivity rows |
| L13 | class-level label rate | no `class_id`, date or class aggregates; `hist_*` excluded; **AP_w primary** | TG, TG_local |
| L14 | overlapping 2 s windows | grouped CV; `min_df` by class | S-rand tax reported |
| L15 | in-fold transforms fit on all data | sklearn `Pipeline` per fold | test that fit indices ⊆ training classes |
| L16 | sklearn internal random validation splits | `early_stopping=False`; manual `GroupKFold` | code review |
| L17 | tuning or thresholds on test; forking paths | inner CV; rate-matched τ; `prereg`; vault | realised FAR reported; tag precedes results |
| L18 | duplicate classes (two recording sessions, re-recordings) | dedupe on `class_id`; flag same section + same date | assertion in `select.py` |
| L19 | selection on the outcome | whole sections; **leave-one-class-out template for every class**; no `n_my_voice` class filter; A/V lag from any speaker | funnel: base rate, AP, coverage by template validation status; speech-onset counts of dropped vs kept classes |
| L20 | observability depends on future speech | glyph NCC tracking independent of state; positives need observation only up to the onset | NA rate and left-censored share around **fresh speech onsets** vs baseline; symmetric-rule sensitivity; recall audit inside NA spans |
| L21 | votes become trivially predictable positives | Δn (Brady excluded) or lexicon typing; windows undefined | vote validation; drop-column AP for `q_vote_lex`; vote sensitivities |
| L22 | detector false positives correlated with features (shirt, plant, border bleed, poll ticks) | shape test; per-class DOWN baseline; ring excluded; tick NA rule | per-class precision; flagged classes; precision by `seg_type`; poll vs non-poll bid rates |
| L23 | course and minute confounds | `ctx_course` (≥ 3 training classes), splines, section constants in F-struct; S-loco | per-course F-text vs F-struct gap |
| L24 | classmate identity learned | roles only; `spk_cur_in_my_group` boolean | allowlist |
| L25 | at-risk mask uses the future | only label-observability rules look ahead; the live state uses samples ≤ t | mask-causality test |
| L26 | key press reaches features via the transcript (`hand-raise-key`, `participation`) | event tables from raw `class-events.json` only; never `--hand-log` for feature transcripts; `hand_log` table is QA-only | `test_loader_allowlist` |
| L27 | Forum system notices in chat ("… raised their hand") | human-actor filter | D1 grep; assertion |
| L28 | own-content features read Brady's future words | "so far" cutoffs through the same `searchsorted` | Brady-frame-at-t+5 sentinel |
| L29 | GRU hidden state learns label-side exclusions | no compaction, zeroing or mask channel; loss mask only | `test_gru_inputs` |
| L30 | pre-raise mic blips as markers | word-bearing events only for Brady | blip sensitivity row |
| L31 | detector errors persist within a class and feed `me_hand_*` | eligible episodes only; F-all−me_hand in MVP; H1 co-requirement | AP on classes with audited precision ≥ 0.9 |
| L32 | vault contamination through dev tooling | `assert_no_vault` in every dev tool | code assertion; vault looked at only via funnel counts |
| L33 | name masking corrupts text (common words) or misses variants | capitalisation and common-word rules; explicit variants | D11 precision and recall; build fails on top-1,000 words |

### 9.2 Controls
**Positive controls:**
- **P-clock (MVP, stop rule).** This replaces the old P3, which succeeded whatever the alignment.
  - Let z(f) = 1 when Brady's tracked tile is bordered at sample f (tracked, main room).
  - Let $s_\delta(f)$ = seconds of Brady-attributed ASR word time in $(t_f+\delta-1,\ t_f+\delta]$.
  - Compute AP(δ) for δ from −2 to +2 s in 0.25 s steps, plus an any-speaker version (`border_any` vs any-word activity).
  - The border can only lag speech, by the UI latency, so the peak δ* must lie in [−1.0, +0.25] s.
  - The peak must be sharp: AP(δ*) − max(AP(δ* − 2), AP(δ* + 2)) ≥ 0.1.
  - The per-class δ*_c must lie within ±0.5 s of the median.
  - A flat curve or a misplaced peak **stops work** until alignment is fixed.
- **P-plant (MVP, stop rule).** Token `zqxw` is inserted as a professor word 8 s before 30% of bids, in training and test. M1 and M2 must reach AP ≥ 0.9 on planted positives vs all negatives. Failure stops work (plumbing).
- **IH+ (must).** δ ∈ {+30, +60} s, with Brady's own words included. Called bids are followed by his own speech, so AP must jump if alignment is right. Failure means investigate, not stop. δ = +5 and +10 are descriptive points on the IH curve (should).

**Negative controls:**
- **TG and TG_local** prediction shifts (MVP, H2).
- **CALLON check (MVP).**
  - Measure the rate of CALLON phrases in the professor's words in [o − 5, o), against 3 matched points per bid: same class, same `seg_type`, |b − o| ∈ [60, 600] s, at-risk, y = 0.
  - Rate ratio with a cluster-bootstrap CI. **Pass if the CI includes 1 (or lies below 1).**
  - Failure means labels are late against the audio. Shift labels earlier by the live-logger median lag (if measured) or move the headline to g = 5, logged as a deviation.
- **A-mismatch (must).**
  - Within each section, derange PCW texts among PCW-subset classes. **Recompute only the similarity columns** (`pcw_res_*`, `pcw_unspent`, `pcw_bm25_gap`, `pcw_new_block`, `pcw_block_pos*`, `pcw_invest`, `lo_sim`, `sg_sim`) against the other class's PCW. Keep each class's own class-level columns (`pcw_has`, `pcw_my_words`, `pcw_frac_answered`, `pcw_n_code`).
  - **Retrain and test** with the deranged features, 5 derangements. Report mean ± sd of AP_w and pooled AP.
  - The PCW gain should vanish.
- **N-name (must, descriptive).**
  - The `[NAME]` rate per minute of professor speech in [o − 15, o − 2] vs [o, o + 10] vs matched non-bid points.
  - The lift of an LR on `[NAME]` counts, with a CI.
  - ΔAP(F-name − F-all) with a CI.
  - **One flag:** the leak is concerning if the CI lower bound of ΔAP(F-name − F-all) is > 0.02 AP. (Professors legitimately name previous speakers, so name mentions alone are not evidence of a leak.)
- **N-stale (could):** the drop toward F-struct measures topic vs moment.
- **Label-shift retrain (should):** labels shifted 1, 2, 3 s earlier, M2 and M3.

**g-sweep (must {2, 5}; should {0, 1}).** This is reframed as an **own-precursor and clock test**. A difference between g = 0/1 and g = 2 reflects Brady's own pre-raise behaviour (unmuting, short voice events, an open mic catching the H key) and clock skew. It cannot detect the professor reacting to the hand; the CALLON check, the live logger and the label-shift retrain do that.

**Audits:** A-stream (should), F-punct (should), A-jitter and A-attrib (could).

---

## 10. Data volume plan

**Selection** (`oncue/select.py`, from `scribe sync` `manifest.csv` + `class.json`):
1. **Population.** Recorded classes in sections where Brady's mean main-room voice events per class is ≥ 5. The write-up states plainly that the population is "the courses where I talk most, two per semester", that this selects on participation, and that **no claim generalises beyond it**.
2. **Headline set.** Within each past semester (F23, Sp24, F24, Sp25, F25, Sp26), take the 2 eligible sections with the highest mean, **whole**: all their recorded classes. Add all recorded CS156 F26 sessions to date (Prof. Watson's course; the live-logger classes).
   - About 12–13 sections, about 165 classes; vault about 33; dev about 130 before QA.
3. **`ctx_session_no`** comes from the full section schedule (`paginated-classes`, including unrecorded classes), so missing recordings do not shift the numbering.
4. **Order inside each phase:** round-robin over sections, random order within each section (seed 156), so a partial run covers every semester.
5. **Excluded:** the unexplained Spring 2024 section with 0 voice events, until D1 explains it (reported).

| Phase | When | Classes | Purpose |
|---|---|---|---|
| Pilot | Mon night (1 class, local ASR) + Tue 08:00 (3 classes) | non-vault: F23 Complex Systems, F24 business, a CS156 F26 session | Day-1 checks |
| A | Tue 10:00 → 15:00 | 36 non-vault (3 per selected section) | detector audit, first labels |
| B | Tue 18:30 → overnight | the rest of the headline set, including the vault | headline |
| C | **decided Tue 22:00**, runs overnight | next-ranked section per semester (+60–80) if (Phase A bids per class × planned dev classes) < 400 | more events; all data lands by Wednesday morning |

**Expected yield.**
- **D15 on Monday night** replaces the earlier extrapolation (576 Brady voice events in 32 classes). It counts fresh speech onsets per class per section from the synced raw voice events, using the 2 s / 30 s rules on events ≥ 0.5 s; no video is needed. That count × an assumed `hand_preceded_share` of 0.5–0.8 gives the expected bids.
- After QA (about 85%): about 140 classes, about **630 bids**, of which about 500 dev and about 130 vault.
- Go threshold: **300 dev bids**.

**Compute, time and cost (M4 laptop, 16 GB)**

| Step | Volume | Estimate |
|---|---|---|
| `scribe sync --out pipeline1/data/scribe` | 694 classes, 8 endpoints | 30–40 min, resumable (Mon night) |
| `class_edit_page` + PCW (POST + 2 GETs) | about 165 × 4 requests at 1 req/s, **once**, cached | about 12 min |
| Deepgram Nova-3 (URL mode, `--language en`, diarize, keyterms, `mip_opt_out`, `strict_res`) | consented courses only; up to 165 × ~88 min ≈ 14,500 min | ≈ $62–72; 6 concurrent; 1–1.5 h |
| Local ASR (courses outside consent) | measured on Monday night (D13) | budget = measured min/class × classes |
| Hand scan pass 1 | 6 strip frames + ≤ 36 anchor frames per class | ≈ 30 s per class |
| Hand scan pass 2 (lowres, strip, 2 fps, videotoolbox) | ≈ 400 MB per class streamed (≈ 66 GB), **0 video stored**; `hand_tiles` ≈ 25 MB per class (≈ 4 GB) | 3–5 min per class per worker; 4 workers → 2.5–3.5 h |
| Highres branch (only if D2 fails in lowres) | keyframe-only decode (`-skip_frame nokey`, if GOP ≤ 4 s), full rate around candidate transitions | capped at 100 classes |
| Re-timing | about 750 × 1.25 s reads | about 15 min |
| Words, events, grid, features | about 190k–250k points | about 20 min (vectorised) |
| Embeddings | about 200k frames + about 150k unique tails + PCW/LO chunks | 10–20 min on MPS; float16 about 0.3 GB |
| CV: hyperparameters once per outer fold | M3: 5 × 3 inner staged fits (max_iter 500, lr 0.1); M2: 5 × 3 × 3 lbfgs fits | ≈ 30 min + 15–30 min |
| CV: each further M3 condition | 5 outer fits with n* reused | ≈ 5–10 min |
| CV: MVP set | K0, F0, S, P-plant | ≈ 2.5–3.5 h at joblib `n_jobs=3`, nothing else running; **re-estimated from one timed M3 fold Wed 09:00** and this row rewritten |
| CV: must set (Thu) | M4, F1, A-mismatch ×5, G, IH+, vault refits | ≈ 2.5 h |
| Bootstrap | 2,000 resamples × about 30 conditions | ≈ 30 min (`n_jobs=3`) |
| M5 / M6 | — | about 30 min / about 2 h (overnight; should / could) |
| A-stream | 300 three-minute clips | about 30 min, ≈ $4.50 |
| Memory | float32 dense ≈ 0.2 GB; TF-IDF ≈ 0.5 GB; M4 dense ≈ 1.2 GB per job | 3 jobs fit in 16 GB with scans and embedding stopped |
| Disk | transcripts + `_asr.json` < 1 GB; tables < 2 GB; `hand_tiles` ≈ 4 GB | well within 43 GB |

**Slow paths:**
- **Deepgram URL mode fails:** Scribe's `audio_fallback` downloads, converts to Opus and uploads automatically.
- **Consent branches (D10):**
  - All courses + third-party ASR approved → Deepgram everywhere.
  - Third-party ASR approved for CS156 only → Deepgram for CS156; local ASR (faster-whisper large-v3-turbo int8, or mlx-whisper with Metal, whichever D13 measures faster) for the rest, with Phase B capped by measured speed.
  - No reply by Tue 12:00 → local ASR for Phase A; Deepgram only after the OK.
  - No reply by Wed 09:00 → stay local and cap at the measured-speed limit (≈ 80 classes at ≈ 6 min per class).
  - "CS156 only, no other courses" → the hand question runs on CS156 F26 only, stated as underpowered; the detector, contract and metric are still delivered.
  - Groq is used only if named in the consent.
- **Scan slower than 6 min per class:** drop to 1 fps (re-timing keeps the precision); scan main-room spans only, by seeking per segment; cap Phase B at 100 classes. Speech labels still cover every transcribed class.
- **Signed URLs expire:** D2 parses `Expires=` and logs the lifetime. Fetch a fresh URL per class at job start (`ForumClient.recording_urls`). ffmpeg runs with `-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 30`. On a 403, re-fetch the URL and resume from the last pts. URLs are never logged or persisted.
- **Behind on Wed 08:00:** freeze at whatever is processed if it holds ≥ 300 dev bids. Otherwise T-speech becomes co-primary under its frozen spec, and the write-up says so.

**Privacy:**
- Transcripts run with `--privacy ids` (HMAC pseudonyms, `scribe_core.privacy.pseudonym`). Names are masked in memory.
- Everything harvested lives under the gitignored `pipeline1/data/`. `pipeline1/.gitignore` adds `audit/`, `*.parquet`, `*.npy`, then `!results/*.parquet`.
- **Committed results bundle** (`pipeline1/results/`, ≤ 50 MB):
  - OOF predictions for MVP and must conditions, with `HMAC(ID_SALT, class_id)[:12]`, `t_s`, `fold`, `y`, `p_raw`, `p_cal`;
  - `results.parquet`;
  - Brady's own bid onsets and outcomes (hashed class id);
  - funnel counts.

  It contains no words and no classmate data, and `ID_SALT` is never committed.
- **Trade-offs, stated in the write-up:**
  - `hand_tiles` keeps binarised name-glyph patches of every tile (classmates' names, 12 × 48 bits), and the recall-audit clips show the full strip with tile interiors pixelated. Both stay local and are **deleted after submission (Mon 19 Oct)**.
  - Precision crops show Brady's own tile only.
- Figures are aggregate-only. N-gram displays require presence in ≥ 20 classes. The hero timeline is a schematic drawn from synthetic data.
- `oncue.privacy.scan` greps the README, notebook outputs, `figures/` and `results/` for every roster name and hashed id, and fails on any hit before pushing.
- Prof. Watson's OK must cover:
  - (a) other courses' recordings;
  - (b) third-party ASR (Deepgram, and Groq if it is to be used);
  - (c) whether other instructors need to be told.

---

## 11. Day-1 checks (blocking unknowns)

| # | When | Check | How | Pass | Fallback |
|---|---|---|---|---|---|
| **D0** | Mon 20:00 | **Access** (the DevTools permission problem) | **Why Claude can't see DevTools:** Claude in Chrome runs in Brady's personal Chrome profile, which has no Forum permission; his Minerva profile does. Nothing in this pipeline needs the browser after one cookie copy. **Route A (default):** in the Chrome profile signed in to Forum (his Minerva account), Brady opens DevTools → Network → any `forum.minerva.edu/api/v1/` request, and copies the full `cookie` request header into `~/.minerva/.env` (the target of the `pipeline1/.env` symlink) as `FORUM_COOKIE=…`, next to `DEEPGRAM_API_KEY=…`. Then `uv run scribe me`. Re-copy whenever `ForumAuthError` appears | `scribe me` prints his uni name, id and role | **Route B:** install or enable Claude in Chrome in the uni profile (or sign in to Forum in the profile that has the extension), so Claude can read requests itself. **Route C (manual, no cookie copied):** Brady pastes `pipeline1/tools/export_snippet.js` into the DevTools console on forum.minerva.edu. It uses same-origin `fetch` over a pasted list of class ids. It gets class-events, class_edit_page, polls, poll-responses, breakouts, the PCW signed-URL POST (with the `X-CSRFToken` header from the `csrftoken` cookie) plus the template and doc, and downloads one `forum_export/{cid}.json` per class. Then `uv run python -m oncue harvest-import ~/Downloads/forum_export/` writes the same `data/raw/{cid}/` layout. Recording URLs from Route C expire, so export them in a separate batch right before scanning. The Scribe v3 extension also exports single-class bundles |
| D1 | Mon night | Sync, manifest, raw-field audit | `scribe sync --out pipeline1/data/scribe`; `scribe census`. Inspect the zero-voice Spring 2024 section. **Grep every synced `message_text` for hand/raise/lower/joined/left patterns, and list null or non-roster actor ids.** Name the class start-time and enrolment fields. Confirm that `paginated-classes` lists unrecorded classes | ≈ 694 classes; system messages identified and filtered; fields named | exclude the zero section (reported); class size from `n_present` only |
| D2 | Mon night + Tue AM | Resolution, strip, icon, eras, URLs | **Mon:** one frame per selected section at 30:00 (`ffmpeg -ss 1800 -frames:v 1` on lowres, 13 frames); define eras (semesters unless the frames differ); parse `Expires=` from a signed URL. **Tue:** resolution, icon size and GOP for lowres **and** highres (size, decode speed); label-bar icon positions; a frame with **both a poll tick and a hand** (which shows?); shape-test and θ thresholds on non-vault frames; is Brady's tile ever off-strip? | icon ≥ 6 px in lowres; tile ≥ 30 px wide; strip auto-detected on all 13 frames | **highres branch:** transcribe and scan the same highres rendition (so there is no cross-encode offset), and confirm on 3 pilots by cross-correlating audio envelopes; keyframe-only scan with full rate around transitions; cap at 100 classes |
| D3 | Tue AM | Glyph tracking | on 3 pilots: anchors; leave-one-class-out template; NCC of Brady's glyph patch in each state (muted, unmuted, bordered, hand up) and at ≥ 3 tile widths | NCC ≥ 0.75 on his tile in every state and width, ≤ 0.6 on others; coverage ≥ 70%; left-censored ≤ 10% | lower NCC_min to 0.65 with margin 0.15; if the strip is in alphabetical order, use roster rank |
| D4 | Tue 18:00 | Detector QA | §2.7 (G2) | precision ≥ 0.90 (LB ≥ 0.82); recall ≥ 0.85 (LB ≥ 0.72) | one retune + fresh confirm sample; then T-speech primary |
| D5 | Mon night / Tue | Live Scribe runs | **Mon:** `scribe transcribe --class CID --bundle-dir … --privacy ids --formats json,jsonl --language en --no-debug --provider local` on 1 non-vault pilot, to expose live bugs overnight without third-party processing. **Tue, after the OK:** Deepgram on 3 pilots. Check `source_res` is recorded | `applied`, status ok/piecewise, `method == "windowed"` with ≥ 6 windows, residual p95 ≤ 1.5 s on ≥ 2/3 | fix in the Scribe repo (new commit + SHA bump). If calibration fails widely, use the **declared fallback**: Forum features by `recording-started` arithmetic with ε_F = 3 s and g = 5, reported as such |
| D6 | Tue AM | Cross-clock | (a) per-class lag_c (any-speaker and Brady-only) on pilots; (b) `src_res` equals the scanned rendition; pts origin = audio start_time; (c) slide-change frame vs `timeline-segment` start (about 10 per pilot); (d) bids before speech in ≥ 95% of `called` | median lag ∈ [−0.25, 1.0] s; per-class deviation ≤ 0.5 s; segment error ≤ 1 s | drop deviating classes; raise g if systematic |
| D7 | Tue AM | Re-fusion | rebuild words from `*_asr.json` with `scribe_core` (`parse_voice_events → calibrate → shift`, own Forum-causal role) and compare the word count with the transcript's `alignment.n_words` | identical | mirror `pipeline.transcribe_bundle` exactly |
| D8 | Mon night + Tue AM | PCW harvest and side effects | **Mon, one past class:** POST `/api/v1/pre_class_work/classes/CID/pre_class_resource_signed_url`, trying `{"iframed":false}` with no saving indicators first and falling back to Forum's body. `ForumClient` has no POST, so this uses `oncue.harvest.post_json`: a `requests.Session` with the same cookie, `X-CSRFToken` and `Referer: https://forum.minerva.edu/`, allowlisted to this one path, **never** PATCH class-users. Follow the link; reproduce the iframe's auth (header or query token seen in DevTools); GET `templates/UUID/` and `doc/UUID/`, recording last-modified, version and any created/started flags. **Wait 2 min, repeat POST + GET, and diff metadata and content.** Run the flow on one class with **no** PCW to see whether a doc gets created. Probe for versions/history endpoints with GETs only. **Tue:** one class per selected section (13). **Never** PUT/PATCH/POST to sle-collaboration and never open websockets; harvest once and cache | text parses; reads change nothing; decide Branch H or N (§5.6) | Route C snippet. If reads bump last-modified: edit time unknown, PCW accepted with a flag, sensitivity row, H3 exploratory. If past terms fail: study guide + learning outcomes + this repo's `PCW/` notebooks as of the last commit before class start (`git rev-list -1 --before=…`); `pcw_status = none` elsewhere |
| D9 | Tue AM | Polls | `/polls` and `/poll-responses` fields; what the "started at 0:32:39" strings are relative to, validated against poll timeline segments | open, close and submit times; reference identified | availability = close or next segment start; ε_F,poll ≥ 1 s |
| D10 | Mon night | Permission | email Prof. Watson: own recordings used only to recover my own hand label; **which courses** (CS156 and about 12 other courses with other instructors); **third-party ASR** (Deepgram with `mip_opt_out`; Groq only if approved); whether other instructors need to be told; nothing harvested committed; aggregate-only reporting; no faces | written OK, with scope | consent branches (§10) |
| D11 | Tue AM | Masking | §5.1: recall grep; precision on 200 masked tokens; masked rate per class; top-20 surface forms (local) | 0 recall misses; precision ≥ 0.95; no top-1,000 word masked | extend variants or common-word list; `--name-fixup` before masking |
| D12 | Tue AM | Throughput and semantics | time one full scan and one ASR job; does Forum auto-lower the hand when Brady speaks? | scan ≤ 5 min; ASR ≤ 3 min | §10 slow paths; outcome rule adjusted |
| D13 | Mon night | Local ASR benchmark | a 10-min clip: faster-whisper large-v3-turbo int8 (CPU) vs mlx-whisper large-v3-turbo (Metal), both with word timestamps | minutes per class measured | Phase B budget set from the faster one; add an `mlx` provider to Scribe if needed |
| D14 | Mon night | Multi-session files | one non-vault pilot with two `recording-sessions[]`: how many files does `generate_recording_urls` return? | structure known | one file per session: scan and transcribe each, stitch with `recording_timeline` offsets |
| D15 | Mon night | Bid-yield estimate | fresh speech onsets per class per section from synced raw voice events (2 s / 30 s rules, events ≥ 0.5 s) | expected dev bids ≥ 300 | plan Phase C early |

---

## 12. Schedule (Mon 12 → Sun 18 Oct)

Two coding agents work in **separate git worktrees** (`git worktree add ../oncue-A`, `../oncue-B`), merged into `main` at each gate:
- **Agent A:** harvest, select, transcribe/words, events, text, pcw, features.
- **Agent B:** hands, labels, audit tools, metrics.

Models come once labels exist. Brady runs the checks and gates and writes the write-up.

| Day | Foreground | Background | Done when |
|---|---|---|---|
| **Mon 12 (eve)** | **Brady:** D0 (10 min); start sync; D10 email with the full consent scope; approve design; review the 13 D2 frames (10 min); commit the semester → timezone table (5 min); install the live logger in the uni profile (5 min). **Agent A:**<br>• §1.1 scope reset (plain `mv`, banner, memory note);<br>• in `../Minervas_scribe`: run the 109 tests, commit v3 (no `.env`, only `.env.example`), add the `source_res`/`strict_res` patch, push branch `v3`;<br>• `uv add` Scribe at the pushed SHA plus dependencies (§13);<br>• package skeleton; `config.py` with the vault rule and the G2 pass rules (commit "freeze vault rule and QA gates");<br>• after sync: `select.py` v0 → `classes.parquet` (`in_vault`, phases); D1 raw-field audit; D15 yield.<br>**Agent B:** `causal.py` + `test_causal.py` on synthetic tables; `metrics.py` + `test_metrics.py` (firing rule, constant-predictor tests); strip auto-detection on the D2 frames | sync (30–40 min); then D5-local on 1 pilot; D13 benchmark; D8 single class + side-effect diff; D14; URL expiry | sync complete; tests green on synthetic data; email sent; PCW branch known or pending only on D8 Tue |
| **Tue 13** | **08:00–10:00:** D1 (zero-voice section), D2 (rest), D3, D6, D7, D8 (13 sections), D9, D11, D12 on non-vault pilots. **10:00:** Phase A scan (36 non-vault; local processing). **12:00 G1** (consent? Deepgram for Phase A+B, or local per the consent branch). **Agent B:** offline tracking, episodes and votes from `hand_tiles`; `contact_sheet`, `audit_clip`, `onset_clips`. **Agent A:** events (raw only), words, frames, labels, at-risk grid, funnel v1, PCW harvest (once, cached). **15:30–17:30:** Brady's audits (≈ 100 min, including masking precision). **18:00 G2.** **20:00** retune deadline; fresh confirm audit 20:30–21:15 if retuned; G2' 21:30. **22:00 Phase C decision** | 12:00 ASR Phase A+B (6 workers or local). 18:30 scan Phase B overnight (4 workers). Phase C ASR + scan overnight if triggered | detector accepted or fallback chosen; ≥ 36 classes labelled; Phase C decided |
| **Wed 14** | **08:00:** all data landed; label QA dashboards (non-vault): funnel, A/V lag, votes, `called_share`, precision by `seg_type`, NA around speech onsets. Features, embeddings, PCW; **truncation test green on real data (blocking)**. **09:00:** time one M3 fold and rewrite the §10 compute rows. **10:00–11:00:** Brady labels 200 professor utterances (40 min). **≈ 14:00: commit and tag `prereg`.** **14:00–20:00 MVP runs:** K0, F0, S, TG, EV, P-clock, P-plant, CALLON. **20:00 G3**: MVP table | overnight: M4 embeddings and LR; must-tier runs; M5 if ahead | MVP table with CIs |
| **Thu 15** | Must items: M4, F1/H3, A-mismatch, G {2, 5}, IH+, N-name. Then should items in reverse cut order (A-stream, S-time, LS, IH full, H-sweep, T-speech, G01, calibration diagram, learning curve, S-loco, M5); figures. **18:00 G4 results freeze.** **19:00 open the vault once.** Evening: write-up draft, diagram | M6 only if ahead | every must cell filled; vault recorded |
| **Fri 16** | Run `pipeline1.ipynb` top to bottom from cache (`RERUN_*=False`, < 30 min), committed with outputs; write the results bundle; README final; privacy scan; confirm the final Scribe SHA; push; **submit by 18:00** | — | submitted |
| Sat 17 – Sun 18 | buffer: fixes only, resubmit before Sunday's close | — | — |
| Mon 19 | delete `hand_tiles` glyph bits and audit clips | — | local deletion done |

**Gates:**
- **G1 Tue 12:00:** pilots transcribed, calibrated (windowed, ≥ 6 windows, p95 ≤ 1.5 s) and lag-checked, under the consented provider. Otherwise local ASR.
- **G2 Tue 18:00:** detector QA passes on the confirm set. Otherwise retune and re-confirm by 21:30, then T-speech primary.
- **G3 Wed 20:00:** MVP table exists, and P-clock and P-plant pass. Otherwise drop all should and could items immediately; must items run only if time remains.
- **G4 Thu 18:00:** results freeze; only the write-up changes after this.

**Brady's time budget (≈ 4.2 h before writing, +45 min if G2 retunes):**

| Task | When | Time |
|---|---|---|
| Cookie, `scribe me`, start sync, logger install | Mon | 20 min |
| Email to Prof. Watson | Mon | 15 min |
| D2 frames + timezone table | Mon | 15 min |
| Day-1 eyeball checks | Tue AM | 30 min |
| Precision sheet (≈ 300 stratified bids) | Tue PM | 25 min |
| Recall clips (3 × 11 min) + onset clips (60 × 15 s) | Tue PM | 55 min |
| Masking precision (200 tokens) | Tue PM | 10 min |
| Timing (should) | Tue PM | 10 min |
| Fresh confirm audit (only if retuned) | Tue eve | 45 min |
| Question-detector labels | Wed | 40 min |
| Results review and vault | Thu | 30 min |
| WER (could) | Thu | 30 min |
| Write-up | Thu eve + Fri | 4–5 h |

**Cut list** (drop first → last):
1. M6
2. could items (N-stale, A-jitter, A-attrib, `hist_*`, turn-end grid, M4-MLP, WER)
3. M5
4. S-loco
5. H-sweep reduced to {5, 30}
6. IH full curve (IH+ at +30/+60 stays)
7. T-speech (unless it is the fallback primary)
8. re-timing (then G01 is dropped and H ≥ 10)
9. A-stream reduced to 60 points
10. S-time; label-shift retrain
11. reliability diagram (ECE stays)
12. `site/` rewrite
13. Phase B capped at Phase A plus whatever is done by Wed 08:00 (≥ 80 classes)
14. vault (folded into CV and stated)

**Never cut:**
- the MVP set: grouped CV vs S-rand; B0, R, B1, M1–M3; F-struct/F-all and the −me_hand pair; TG and TG_local; the event metric with its references; P-clock; P-plant; the CALLON check;
- the truncation, sentinel, loader and GRU tests; masking checks;
- label QA and the funnel;
- the privacy scan.

The must items (M4, F-text, H3, A-mismatch, G {2, 5}, IH+, N-name) are cut only if G3 fails.

---

## 13. Code layout (`cs156-ml/pipeline1/`)

**Dependencies and project setup:**
- **Scribe.** Commit and push branch `v3` on Monday night (the repo is public, so graders can install it). Then:

  `uv add "minervas-scribe @ git+https://github.com/Brady-Rutherford/Minervas_scribe@<sha>"`

  (distribution `minervas-scribe`, import `scribe_core`, CLI `scribe`).
  - The committed `pyproject.toml` and `uv.lock` **never** contain a local path.
  - For quick iteration, `uv pip install -e ../Minervas_scribe` goes into the venv only, unrecorded. No result is kept unless it was produced after `uv sync` at a pushed SHA.
  - Each Scribe fix becomes a new commit and an SHA bump.
  - Notebook §0 prints the installed Scribe SHA (from `direct_url.json`).
  - The repos are never merged.
- `uv add sentence-transformers transformers torch pyarrow opencv-python-headless rank-bm25 wordfreq python-dotenv tqdm requests`, plus `mlx-whisper` if D13 picks it.
- `ffmpeg` (Homebrew, videotoolbox) is already present. The diagram uses `npx -y @mermaid-js/mermaid-cli` (`mmdc` and `dot` are absent), with matplotlib as the fallback.
- `pyproject.toml`: `[tool.hatch.build.targets.wheel] packages = ["src/cs156", "pipeline1/oncue"]`.
- nbstripout is a dependency but **not** a git filter; keep it that way so the notebook is committed with outputs.
- `pipeline1/.gitignore` adds `audit/`, `*.parquet`, `*.npy`, `!results/*.parquet`.

```
pipeline1/
  README.md               write-up (assignment order, §14)
  PLAN.md                 this design + Pre-registration + T-speech fallback spec (tag `prereg`)
  pipeline1.ipynb         run notebook, committed WITH outputs; re-runs metric/figure/table cells from results/
  results/                committed, privacy-safe bundle: oof_*.parquet (HMAC ids), results.parquet, bids.parquet, funnel.csv
  figures/                committed, privacy-scanned SVG/PNG (aggregate; own tile only; schematic timeline)
  research_notes/         kept in place; cited in References and AI use
  later/                  the_tell_face.md (archived face plan, P2)
  tools/export_snippet.js Route C console exporter
  site/index.html         "superseded" banner now; rewrite is a cut item
  oncue/
    __init__.py  __main__.py  CLI: python -m oncue {harvest,harvest-import,select,transcribe,words,events,scan,track,episodes,retime,labels,frames,embed,features,train,evaluate,controls,figures,hand-log-import,results-bundle,privacy} [--phase A|B|C] [--workers N]
    config.py     constants (§1.2), paths, regexes, seg_type map, VAULT_SALT, FEATURE_ALLOWLIST, BLOCKED_RE, RAW_PATH_WHITELIST (per source), EVENT_TYPES, G2 pass rules
    harvest.py    run_sync(out); class_edit_page(client, cid); post_json(path, body) [allowlisted to one path]; pcw_fetch(cid)->(template, doc, meta) [GET-only on sle; cached once]; pcw_side_effect_probe(cid); import_export(dir)
    loaders.py    load_raw(cid, source) -> whitelisted paths only; dropped keys logged; PROVENANCE map
    select.py     build_classes(manifest, raw)->classes.parquet; choose_sections(); vault_flag(cid); assert_no_vault(df); dedupe(); session_no_full_schedule()
    transcribe.py transcribe_many(cids, workers, provider) -> scribe_core.pipeline.transcribe_class(..., options=TranscribeOptions(privacy="ids", formats=("json","jsonl"), language="en", debug_html=False, save_asr=True, strict_res=True, res=SRC_RES)); NEVER passes hand_log
                  words_table(cid): *_asr.json + parse_voice_events(raw) -> calibrate/shift -> forum_role() (causal) + word_bearing flags + fuse() (kept as role_fused) -> words.parquet
    events.py     video_clock(probe, cal, segs) [unclamped; gap events dropped]; voice_table; chat_table (human actors); segments_table (seg_type; NO end); polls_table (parsed reference); presence_table; hand_log_import()
    hands.py      detect_strip(frames); strip_tiles(frame); glyph_patch(tile); border_score; build_era_templates(loco=True); scan(cid, fps=2)->hand_tiles; track(hand_tiles, template)->hand_frames;
                  shape_test; baseline(); episodes(hand_frames)->hand_episodes (state machine, live state, votes, contested, outcomes); av_lag(cid); retime(cid, eps);
                  contact_sheet(sample); audit_clip(cid, pixelate=True); onset_clips(sample)
    labels.py     speech_onsets(voice, words); at_risk(cid); points(cid) with every y_* column (asymmetric observability; symmetric variant)
    text.py       roster_patterns; mask(tokens, roster, common_words); normalise(text); delete_callon(text); frames(words)->frames.parquet (close_s); windows(view_or_arrays, t); question_detector(punct=False); rule_R
    pcw.py        quill_to_text(ops); pcw_status(meta, class_start); blocks(template, doc)->pcw_blocks; chunks(); git_asof(path, start)
    embed.py      Embedder(EMB, device="mps"); encode(texts) with sha1 cache (data/cache/emb/); pooled(v, tau); znull(class) (earlier-only reference set)
    causal.py     truncate(tables, t); CausalView(cid, t) (reference path)
    features.py   build(cid, points) (vectorised, searchsorted); GROUPS; FEATURESETS; PROVENANCE
    splits.py     group_folds; random_folds; time_split; loco; inner_groups; MinClassCountCategories
    models.py     make(id, featureset) -> sklearn Pipeline (AO, B-min, B0, R, B1, M1–M4); tune_once(fold) -> n*, C; gru.py (M5); xenc.py (M6)
    metrics.py    ap, ap_within, macro_ap, ece, brier; firings(p, grid, H, g); far; recall; lead; time_in_warning; froc; r0; tau_rate_matched; shift_global; shift_local; cluster_bootstrap; paired_delta; mde; fixed_sequence_holm
    controls.py   mismatch_pcw (similarity columns only, ×5); cutoff_shift(δ, include_me); name_descriptive; callon_check; p_clock; plant_token; stream_audit(points, n=300, span=180); jitter; label_shift
    experiments.py CONDITIONS registry (§7, with tiers); run(cond) -> oof/{cond}.parquet + results rows (cached by config hash)
    plots.py      one function per figure (§14)
    privacy.py    scan(paths) -> 0 hits required; results_bundle()
  tests/ test_causal.py test_labels.py test_metrics.py test_masking.py test_pcw.py test_loader_allowlist.py test_gru_inputs.py test_calibration_qa.py
  data/ (gitignored)  scribe/{raw/CID, manifest.csv}  transcripts/  tables/  cache/  logs/
  audit/ (gitignored) contact sheets, clips (deleted Mon 19 Oct)
```

**Tables** (`data/tables/*.parquet`; ids hashed; times in master-clock seconds):

| Table | Columns |
|---|---|
| `classes` | class_id, section_id, course_code, course_area, term, semester_idx, session_no, date, weekday, hour_local, hour_utc, sched_start_utc, rec_offset_s, rec_segments_json, n_rec_sessions, T_end_s, prof_hash, enrolled, n_my_voice_main, phase, in_vault, src_res, a_start_s, v_start_s, url_lifetime_s, cal_status, cal_applied, cal_method, cal_n_windows, cal_max_residual_s, resid_med_s, resid_p95_s, eps_F_s, timing_precision, av_lag_s, av_lag_brady_s, av_ok, multi_session_ok, era, strip_y, strip_h, strip_found, template_validated, coverage, tile_status, down_baseline, flagged, pcw_status, template_asof, n_bids, n_votes, n_contested, qa_pass, qa_reason |
| `words` | class_id, w_idx, start_s, end_s, text_masked, text_norm, has_qmark (F-punct only), conf, filler, spk_hash, role, role_event_start_s, role_avail_s, role_fused, how |
| `frames` | class_id, f_idx, start_s, end_s, close_s, role, spk_hash, text_norm, n_words, emb_row |
| `voice` | class_id, spk_hash, role, start_s, end_s, room_main, precise, word_bearing, first_word_avail_s |
| `segments` | class_id, seg_idx, start_s, title_masked, section_masked, seg_type |
| `chat` | class_id, t_s, spk_hash, role, text_masked |
| `polls` | class_id, poll_id, open_s, close_s, prompt_masked, my_answer_text, my_answer_avail_s |
| `pcw_blocks` | class_id, block_uuid, block_order, prompt_text, answer_text, answer_words, has_code, n_files (status lives in `classes`, never in features) |
| `hand_tiles` (local) | class_id, pts_s, tile_idx, x0, x1, frac_hand_s1, frac_hand_s2, frac_tick_s1, frac_tick_s2, shape_s1, shape_s2, border, ncc_me, glyph_bits |
| `hand_frames` | class_id, t_s, n_tiles, tile_found, tile_idx, ncc, ncc_gap, frac_hand_adj, shape_ok, frac_tick, n_others, border_any, raw_state, live_state |
| `hand_episodes` | class_id, ep_id, onset_s, onset_retimed_s, retime_failed, offset_s, offset_live_s, dur_s, n_toggles, ep_type (bid/vote/glitch), delta_n_others, vote_lex, contested, left_censored, me_hand_eligible, outcome (called/withdrawn/open), call_latency_s, seg_type |
| `hand_log` (QA only) | class_id, press_wall_ms, press_s, matched_onset_s, lag_s |
| `speech_onsets` | class_id, onset_s, turn_end_s, silence_before_s, fresh, hand_preceded |
| `points` | class_id, t_s, in_G, at_risk, excl_reason, label_state, y_H15_g2, y_H5_g2, y_H10_g2, y_H30_g2, y_H60_g2, y_H15_g0, y_H15_g1, y_H15_g5, y_H15_g2_sym, y_speech_H15_g3, next_bid_s (eval only), fold, in_vault |
| `features` | class_id, t_s, allowlisted `ctx_* spk_* me_* me_hand_* poll_* q_* pcw_* lo_sim sg_sim n_words_* wpm_60 filler_rate_60 frac_unk_60 topic_shift own_sim`, emb_row |
| `oof/{cond}` | class_id, t_s, fold, y, p_raw, p_cal, inner (bool) |
| `results` | condition, tier, model, featureset, split, target, H, g, fold, n_classes, n_onsets, n_points, base_rate, ap, ap_fold_mean, ap_fold_sd, ap_lift, ap_within, macro_ap, ap_shift, timing_gain, ap_local, timing_gain_local, roc_auc, brier, ece, recall_far01, recall_far05, realised_far01, realised_far05, tiw_far01, tiw_far05, eventlift_shift_far01, eventlift0_far01, lead_med_s, pafroc |

**Run notebook (`pipeline1.ipynb`).** `RERUN_HARVEST`, `RERUN_TRANSCRIBE`, `RERUN_SCAN` and `RERUN_MODELS` all default to `False`. When False, a cell prints the CLI command it would run and loads cached tables or the committed `results/` bundle. The metric, figure and table cells therefore re-run for graders who have no data. Sections:
0. Provenance: git hash, `prereg` hash, Scribe SHA, package versions, row counts, privacy note.
1. Data funnel: 694 → population → selected → transcribed → calibrated (windowed, p95) → A/V lag → scanned → located/validated → QA → dev/vault, with classes, bids and points at each row; PCW coverage by status and course.
2. Label validation: confirm-set audit table with Wilson CIs; per-class precision; recall overall, in NA spans and in polls; votes vs bids, contested episodes and bids lost per course; outcomes; latency histogram; `hand_preceded_share` by section; NA and left-censored share around fresh speech onsets; poll vs non-poll bid rates.
3. Clock: calibration status and residuals, lag_c distribution, P-clock curve, live-logger lag.
4. Causal contract: `pytest tests/` output inline, allowlist and provenance printout.
5. Decision points: base rates by H and g, exclusion breakdown.
6. Descriptives: peri-onset curves (`q_open`, `spk_silence_s`, `pcw_unspent`, −120 to +30 s) against shifted controls; question-detector validation; masking quality.
7. Models with learning curves.
8. Comparison table and forest plot (H1–H5).
9. Event curves (with references), TiW and lead.
10. Timing decomposition (global and local).
11. Ablations and controls (P-plant, CALLON, IH+, N-name, A-mismatch).
12. Splits.
13. Sweeps and information horizon.
14. A-stream and F-punct.
15. Calibration.
16. Vault.
17. Aggregate error analysis.
18. Summary → `figures/summary.json`.

---

## 14. Write-up outline (`README.md`, mapped to the assignment)

| Assignment section | Content |
|---|---|
| High-level idea | 2 paragraphs. Forum records everything except hand raises (frontend code: PATCH-only), so they are recovered from the screen-captured recording. The question: am I cued by *what is said and what I prepared*, or by *when* it is? Image: a **schematic** one-class timeline built from synthetic data and labelled as a schematic, with role lanes, question ticks, p(t), firings, bids and segment shading. No text. |
| Pipeline diagram | `figures/pipeline.svg`, rendered with `npx -y @mermaid-js/mermaid-cli -i pipeline.mmd -o figures/pipeline.svg` (matplotlib fallback): Forum API (raw events, whitelisted paths) + PCW → Scribe (Deepgram or local ASR, calibration) ∥ same-rendition signed mp4 → glyph tracking + hand detector → one media clock (pts, lag check) → labels and at-risk grid → CausalView/features → models → grouped CV → metrics. The clock box is highlighted. |
| Data and ingestion (largest section) | endpoints table and JSON-path whitelists; timestamp traps (voice `timestamp` = end; kebab/snake keys; whole-second 2023; `NOREC`; Scribe's segment `end` and gap clamping; the global-fallback calibration); calibration QA and residual p95; rendition and pts clock; Forum-causal roles; word-bearing events; causal frames; masking and normalisation; PCW via sle-collaboration and `pcw_status`; detector (strip auto-detection, glyph NCC with a leave-one-class-out template, HSV + shape test, baseline, hysteresis, live state, re-timing, vote typing); label QA; causal contract (condensed); funnel |
| GitHub repo | link; `uv sync` (Scribe from a public git SHA); `.env`; CLI stages; notebook with outputs that re-runs from `results/`; why no raw data is committed |
| Models not covered (equations) | each tied to the run that used it, so none describe models that did not run:<br>(1) **Gradient boosting** (M3): $F_m=F_{m-1}+\eta f_m$, leaf weight $w_j^*=-\frac{\sum_{i\in j}g_i}{\sum_{i\in j}h_i+\lambda}$, with $g_i=p_i-y_i$, $h_i=p_i(1-p_i)$.<br>(2) **Transformer sentence encoder** (bge-small, MiniLM; M3 via PCA, M4): $\text{Attn}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$, multi-head, residual + LayerNorm per layer; CLS pooling $e(x)=h^{(L)}_{\text{CLS}}/\lVert h^{(L)}_{\text{CLS}}\rVert$; recency pooling $v_\tau(t)$ (§5.4).<br>(3) **Label model** (always runs): NCC $=\frac{\sum_u(P_u-\bar P)(T_u-\bar T)}{\sqrt{\sum_u(P_u-\bar P)^2\sum_u(T_u-\bar T)^2}}$; the HSV indicator and shape test (§2.3); the hysteresis state machine, with raw state $r_f$ = UP iff $\text{frac}'\ge\theta_{\text{on}}\wedge\text{shape}$, DOWN iff $\text{frac}'<\theta_{\text{off}}$, else $r_{f-1}$, and a confirmed change after k agreeing samples.<br>(4) **Platt scaling** (pooled AP and ECE use it): $p=\sigma(a\,s+b)$, fit by maximum likelihood on inner OOF predictions.<br>(5) **BM25** (if `pcw_bm25_gap` stays): $w(q,d)=\text{IDF}(q)\frac{f(q,d)(k_1+1)}{f(q,d)+k_1(1-b+b|d|/\text{avgdl})}$.<br>(6) **GRU** (if M5 ran): $z_t=\sigma(W_zx_t+U_zh_{t-1})$, $r_t=\sigma(W_rx_t+U_rh_{t-1})$, $\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1}))$, $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$.<br>(7) Cross-encoder (only if M6 ran).<br>(8) Discrete-time hazard framing. |
| Novel metrics and features (equations) | (1) **Event recall at a false-firing budget**: the firing rule with refractory H − g, FAR per at-risk minute/hour, recall, lead, time-in-warning, $R_0(\lambda_f)=\lambda_f(H-g)/60$, EventLift against shifted predictions (§8.3).<br>(2) **Timing gains** TG and TG_local (§8.4).<br>(3) **Unspent preparation** $U(t)$ and the resonance gap Δρ, z-scored (§5.6), labelled underpowered if the coverage rule fails. |
| Learning figures | **must:** (a) HGB staged log-loss, training and inner-validation, per outer fold; (b) LR training vs validation AP across C. Both come from the runs already done. **should:** AP vs number of training classes (M2, M3, M4); M5 training/validation loss and AP per epoch |
| Final performance figures | pooled OOF PR curves (B1, R, M1–M4) with π; per-class AP_w distribution; FROC with AO, B-min, global- and local-shift and R references, lead and TiW annotated at 0.1/min; reliability diagram (should); vault vs CV |
| Comparison | table: rows = §7 conditions; columns = **AP_w [CI]** (primary), pooled AP (calibrated) and mean per-fold AP ± sd, lift, TG, TG_local, ΔAP_w vs F-struct [paired CI], MDE, Recall@0.1 (realised FAR, TiW), EventLift, lead, n_classes, n_onsets. Forest plot of H1–H5 with fixed-sequence/Holm-adjusted p. S-rand tax bars. g/H sweep lines. IH curve (illegal side shaded). A-mismatch (mean ± sd), N-name, CALLON and P-clock panels. Per-course F-text vs F-struct gap. One paragraph per contrast. |
| Discussion | content vs structure vs preparation, within class; what R explains; the class vs the phase vs the moment (TG vs TG_local); what the controls rule out, and what label lateness the CALLON check and the logger bound; A-stream (how much batch ASR inflated results); failure cases by `seg_type`, course and semester; label validity (raise ≠ called; votes and contested episodes; recall in NA spans; per-class precision); n = onsets; population limits; turn-taking literature |
| Privacy and ethics | ids mode, in-memory masking, aggregate-only, own tile only, the full-strip audit and glyph trade-off (local, deleted), HMAC ids in `results/`, consent scope (courses, Deepgram `mip_opt_out`, Groq only if approved) |
| References | Sacks, Schegloff & Jefferson 1974 (*Language*); Skantze 2021 (*CSL*, turn-taking review); Ekstedt & Skantze 2020 (TurnGPT); Rowe 1986 (wait time); Friedman 2001 (*Ann. Stat.*); Ke et al. 2017 (LightGBM histogram GBDT); Vaswani et al. 2017 (attention); Reimers & Gurevych 2019 (Sentence-BERT); Xiao et al. 2023 (C-Pack/BGE); Wang et al. 2020 (MiniLM); Cho et al. 2014 (GRU); Singer & Willett 1993 (discrete-time survival); Winterhalder et al. 2003 and Mormann et al. 2007 (seizure-prediction evaluation); Chakraborty 1989 (FROC); Saito & Rehmsmeier 2015 (PR vs ROC); Platt 1999, Guo et al. 2017 (calibration); Robertson & Zaragoza 2009 (BM25); Lewis 1995 (NCC template matching); Efron & Tibshirani 1993 (bootstrap); Holm 1979; Berger 1982 (intersection-union tests); Wilson 1927; Roberts et al. 2017 (structured CV); Kapoor & Narayanan 2023 (leakage); Pedregosa et al. 2011; Deepgram Nova-3 docs; Minerva's Scribe v3 (SHA); `pipeline1/research_notes/` |
| AI use | Claude for planning, API/frontend investigation and code; the research notes; transcript links per repo convention; no generated analysis text |

---

## 15. How it reaches a 4, and the honest failure story

**Originality.**
- It predicts the author's own decisions to bid for the floor from three years of seminar speech and his own pre-class writing.
- The label is recovered by computer vision from a UI that hides it in every student API.
- Nothing in the course or the reference 3 (Haar + KNN, about 100 hand-sorted photos, random splits) is close. This design answers each of that submission's weaknesses: thousands of auto-labelled events, grouped splits, many models, and a measured leakage tax.

**Tools beyond class, each doing a job:**

| Tool | Job |
|---|---|
| ASR (Deepgram or local Whisper), cross-correlation clock calibration with strict QA, pts-based media clock, per-class A/V lag, Forum-causal speaker attribution | data engineering |
| strip auto-detection, binarised-glyph NCC tracking with a leave-one-class-out template, HSV + shape test with per-class baseline, hysteresis state machine with a causal live state, sub-second re-timing, vote typing | label extraction |
| Frozen transformer embeddings with causal recency pooling; null-referenced z-scored similarity; BM25 | text and preparation features |
| Gradient boosting | strongest tabular model |
| Streaming GRU hazard; fine-tuned cross-encoder (if time) | sequence and end-to-end text |
| Seizure-prediction event scoring (refractory firings, FAR per at-risk hour, time-in-warning) with analytic and shifted-prediction chance | the metric that matches the decision |
| Probability calibration; nested grouped, temporal and leave-course-out CV; cluster and paired bootstrap; intersection-union and fixed-sequence testing; pre-registration; a hashed vault | honest evaluation |

**Meta-knowledge on display:**
- a written causal contract enforced by blocking truncation, sentinel, provenance and GRU-input tests;
- leakage quantified, not just avoided: the S-rand tax, F-name, F-punct, an information-horizon curve, A-stream, and the CALLON check plus live logger for label lateness;
- within-class primary statistics, and a timing decomposition that separates "which class", "which phase" and "which moment";
- a rule baseline and a structure-only model that can falsify the content story, plus a text claim that holds without detector-derived inputs;
- mismatched-PCW (retrained) and name controls; offset-sensitive clock and planted-token positive controls with stop rules;
- votes separated from bids, with sensitivities; no selection on the outcome (leave-one-class-out templates, participation-free A/V check);
- thresholds chosen out of fold; an explicit argument against resampling; capacity reasoning plus a learning curve.

**If the signal is weak, every outcome is still a finding with CIs:**

| Outcome | Reading |
|---|---|
| H1 CI includes 0 (F-all ≈ F-struct ≈ R within class) | "I raise my hand by the rhythm of class (debriefs, open questions, my own refractory period), not by what is said." Backed by the peri-onset curves, the MDE for ΔAP_w, and P-clock / P-plant / IH+ showing that the pipeline *can* see text signal. |
| TG > 0 but TG_local ≈ 0 | The model knows which classes and which phases I'm talkative in, not the moment. The claim is limited to that. |
| H3 null, and A-mismatch ≈ F-all | Preparation does not steer *when* I speak. Discuss written vs spoken register, PCW as uploaded files, AI-assisted PCW, and PCW coverage. |
| H1 holds only with `me_hand_*` | My own earlier raises carry the signal; the text-only claim is not supported, and this is stated. |
| A-stream or F-punct shows a big drop | Part of the apparent signal came from batch-ASR look-ahead. Report the stream-corrected AP as the honest number. |
| CALLON check fails | Labels run late against the audio. The headline moves to g = 5 or to shifted labels, logged as a deviation. |
| S-time fails while S-group works | My participation habits drifted across years or courses. |
| Fewer than 300 dev bids | Headline moves to H = 30, plus T-speech. Wide CIs are stated plainly. |
| G2 fails | T-speech (fresh speech onsets, no video, frozen spec) becomes the headline. The detector audit stays in as a documented partial result, and the hand question moves to a later pipeline with live `H` logging as ground truth. |

**Why a null still scores well.** A clean null with working positive controls, measured leakage and stated detectable effects is a strong result. The engineering (label recovery, one media clock, the causal contract) is graded either way.

**Next pipelines:** P2 adds vision of Brady's own tile. P3 predicts *what* he says, from the same `CausalView`.

---

## 16. Changes from critique

| # | Issue (section, severity) | Resolution |
|---|---|---|
| 1 | §8.3 alarm episodes cut by label-side points; always-on gamed recall/FAR; R_0 assumed single points (blocker) | Episodes replaced by refractory firings on the full causal grid 𝒢, which only causal rules break (§8.3). FAR = false firings per at-risk minute and hour. TiW reported with every recall. AO, B-min, global/local-shift and R curves on every FROC. Shifted-prediction recall at matched FAR is the primary chance level. `test_metrics` asserts a constant p gives EventLift ≈ 1 and recall ≤ TiW. |
| 2 | §5.6/A14 `pcw_edited_after_start` in F-all; `pcw_has` forced by post-start edits (blocker) | Column removed; `^pcw_edited` and `pcw_status` blocked in `test_causal`. PCW availability is a class-level `pcw_status` decided before modelling. Branch H rebuilds docs as of class start from history. Branch N keeps answer features to verified docs on a PCW subset and drops `unknown` classes from H3/A-mismatch with n reported. `pcw_has` = a verified pre-class doc with content. |
| 3 | §2.7 `--hand-log` puts the key press into transcript events (major) | All event tables come from raw `class-events.json` with type allowlist {voice, message, timeline-segment}. `--hand-log` is never used. The logger goes to a QA-only `hand_log` table via `oncue hand-log-import`. `hand-raise-key`, `participation` and `analytics` are blocked, and the build raises if transcript `events` are read. |
| 4 | §5.7 `own_sim` and other own-content features not "so far" (major) | `own_sim`, `U(t)`'s M, `me_*` and `poll_res_my` use only Brady frames with `close_s ≤ t` (NaN if none) through the same `searchsorted` cutoff as CausalView. New sentinel: a Brady frame at t + 5 equal to the current window changes nothing. |
| 5 | §1.2/§4.3 global-fallback calibration passes QA with residual 0; ε_F clamp; median hides tails (major) | QA requires `method == "windowed"` with ≥ 6 usable windows. `max_residual_s` > 1.5 s is dropped, not clamped. Residual p95 ≤ 1.5 s (median ≤ 0.5 s). ε_F = max(ε_F,min, p95, `max_residual_s`). `test_calibration_qa` excludes the global path. |
| 6 | §2.1 "one clock by construction" not guaranteed (major) | Scribe records `source_res` and gets `strict_res` (new commit); the scan reads the same file. Frame times come from pts (`-copyts` + `showinfo`, audio start origin), and seeks keep absolute pts. Per-class A/V lag is automated (any-speaker border vs ASR activity, plus a Brady-only check); deviations > 0.5 s are dropped. Multi-session classes need a validated jump and lag on both sides. |
| 7 | §4.2/L7 punctuation and casing encode future silence (major) | Headline feature text is lowercase and punctuation-free; `q_*` has no `?` rule. F-punct generalises Qnp to every text group. A-stream splices stream words for [t − 180, t] into the history and recomputes every feature. |
| 8 | §6 M5 GRU could learn label-side exclusions (major) | Every grid step is fed its causal inputs unchanged, with no compaction, zeroing or mask channel; exclusions live only in the loss mask. `test_gru_inputs` requires bit-identical inputs for steps ≤ t when onsets after t are deleted. |
| 9 | §7/§8.5 H1 and H3 on pooled AP are class-level (major) | Primary statistic for every hypothesis is ΔAP_w (within-class) with a paired cluster bootstrap. Pooled ΔAP and ΔTG_local are secondary. MDE_80 = 2.49·SE reported. |
| 10 | §8.4 global shift guarantees TG > 0 via phase features (major) | Added TG_local (within-segment, 3-min-block circular shifts of ≥ 30 s). H2 requires TG_local(F-all) > 0 and ΔTG_local(F-all − F-struct) > 0. The global shift is kept as the "which class" floor. |
| 11 | §2.2/§3 label patch changes with state; NA before onsets drops positives selectively (major) | Binarised glyph patches at fixed height, with ring and icons excluded; NCC measured per state and width on D3. Positives need observation of (t, o*] only, negatives the full window; the symmetric rule is a sensitivity. NA and left-censored share around fresh speech onsets reported; > 10% left-censored blocks G2. |
| 12 | §2.7 recall audit shows only the tracked tile; audit classes from top sections (major) | Recall clips show the full strip at 8× with tile interiors pixelated and the tracker's tile outlined. Brady marks every raise by name, including in NA spans. Recall reported overall, in NA spans and in polls. Audit classes and onsets are drawn at random from all selected sections, dev only. |
| 13 | §2.3 yellow-green objects or bleed pass the hue test (major) | Shape test (circular blob with white glyph). Per-class DOWN baseline subtracted. Classes with baseline > θ_off/2 flagged and fully audited. Precision audit stratified to ≤ 3 bids per class, with per-class precision reported. |
| 14 | §2.3/§2.5 poll tick may hide the hand (major) | D2 finds a tick + hand frame. If the tick can hide the hand, `frac_tick ≥ θ_on` samples are NA, not DOWN. Bid rate in poll vs non-poll segments reported with and without the rule. Poll segments included in the recall audit. |
| 15 | §2.5 Δn counts Brady's own hand; small-room votes and engaged-room bids mistyped (major) | Brady's tile excluded (Δn_others). Vote also typed by the VOTE lexicon in [o − 60, o]. "Contested" episodes flagged. AP reported with votes undefined (default), as positives, Δn-only and contested-as-bids; bids lost to vote typing reported per course. Permutation importance replaced by drop-column AP. |
| 16 | §4.1 A16 `me_hand_*` inherits persistent detector errors (major) | F-all−me_hand and F-struct−me_hand are MVP rows, and H1 must hold on both (intersection-union). `me_hand_*` uses only shape-passing episodes with confirmed duration ≥ 2 s. AP on classes with audited precision ≥ 0.9 reported. |
| 17 | §2.6/§7 T-speech forced negatives; detector dependence (major) | Points with a non-fresh onset in the window, or with t + H − last_me_end < 30 s, are undefined. T-speech has no hand-derived exclusions or features (F-all−me_hand). It is compared with the hand target only at matched FAR. |
| 18 | §5.1 Levenshtein masking hits common words; misses possessives/variants (major) | Strip punctuation and possessives first. Distance-1 matches require mid-sentence capitalisation and must not be in top-20k / `COMMON_WORDS`. D11 measures precision on 200 tokens and recall. The build fails if a top-1,000 word is masked. |
| 19 | §4.1/L11 allowed files contain post-class fields; template drift (major) | Per-source JSON-path whitelists; unknown keys dropped and logged. Every feature column traces to whitelisted paths (`PROVENANCE` test). Template version at class date if exposed; else `template_asof = "harvest"` plus an F-all−pcwT sensitivity. |
| 20 | §11 D8 harvest may create or bump docs (major) | D8 re-GETs after POST/GET and diffs last-modified and content, and tests a no-PCW class. `iframed:false` is tried first. Never PUT/PATCH/POST to sle; harvest once and cache. Created flags logged; empty docs → `pcw_status = empty` (pcw_has = 0). |
| 21 | §4.1 A8 chat may include system notices (major) | D1 greps `message_text` for hand/raise/lower/joined/left patterns and for null or system actors. Only messages from human actors in `class-users` are kept, asserted in `test_loader_allowlist`. |
| 22 | §7/L8 g-sweep cannot detect professor reaction (major) | Label lateness measured directly: live logger promoted to should (every live class Tue–Thu); CALLON rate check is MVP with pass "rate-ratio CI includes 1"; label-shift retrain (1, 2, 3 s) promoted to should. g-sweep reframed as an own-precursor and clock test. |
| 23 | §2.2 per-class template makes coverage track participation (major) | Every class is tracked with a leave-one-class-out era template from other classes' anchors; own anchors only validate. Base rate, AP and coverage reported by validation status; speech-onset counts of dropped vs kept classes compared. |
| 24 | §7 A-mismatch swapped class-level columns; test-time-only derangement (major) | Retrain and test with deranged PCW; only similarity columns are recomputed, class-level PCW columns kept. 5 derangements, mean ± sd, on AP_w as well. |
| 25 | §8.1/§10 dev work could touch the vault (minor) | `select.py` sets `in_vault`. Pilots, Phase A, tuning, lexicons, labels, descriptives and QA dashboards all `assert_no_vault`. Vault classes are scanned, but only their funnel counts are viewed. |
| 26 | §2.7 G2 retune on the audit material (minor) | The first audit is the confirm set. Any retune makes it tuning material and triggers a fresh confirm sample; only confirm numbers (with Wilson bounds) are reported. |
| 27 | §4.1 sub-second Brady voice events as pre-raise markers (minor) | `me_*`, Brady's speaker role, `me_speaking` and speech onsets use only word-bearing Brady events. No-word blips dropped; the AP change when they are included is reported. |
| 28 | §5.6 `pcw_res_ref` used later classes (minor) | Uses the 5 most recent **earlier** classes of the same section with verified docs; NaN when fewer than 3. |
| 29 | §11 D8 fallback used HEAD of repo PCW notebooks (minor) | `git show $(git rev-list -1 --before=<class start UTC> HEAD):<path>`; `pcw_status = none` when no commit precedes class start. |
| 30 | §2.4 glitches counted as negatives; live state peeked ahead (minor) | Windows containing a glitch are undefined, and R_low follows glitches (`post_glitch`). The live state uses samples ≤ t only, `onset_pending` covers the confirmation delay, and both are in the mask-causality test. |
| 31 | §4.1 poll time reference; `me_last_outcome` timing; class size post-hoc (minor) | Poll strings parsed against a reference validated on D9; ε_F,poll ≥ 1 s. `me_hand_last_outcome` available at offset + 5 s. Class size = `ctx_enrolled` (section roster) and `ctx_n_present(t)`. |
| 32 | §5.2 pause-cut closure depended on the next word (minor) | Word onsets (not text) are perceptible from start + ε_w. A pause cut closes at end + 0.4 + ε_w only if no word starts in (end, end + 0.4]. Identical in both paths, with a fixture. |
| 33 | §8.2/§8.5 pooled AP mixes fold scales; CIs ignore training variability (minor) | Pooled AP on Platt-calibrated p within each fold, plus mean per-fold AP ± sd; AP_w is primary. CIs are stated as conditional on the fitted fold models. |
| 34 | §7 multiplicity FWER ≈ 0.10 (minor) | Fixed-sequence gatekeeping: H1 at 0.05, then H2–H5 Holm at 0.05 only if H1 rejects; pre-registered. |
| 35 | §9.2 N-name not clean; P3 offset-insensitive (minor) | N-name made descriptive (rates in [o − 15, o − 2], [o, o + 10] and matched points). P3 replaced by P-clock: an offset-sensitive AP(δ) curve with the peak required at the expected border lag. |
| 36 | §3 rec_gap in media time; clamped gap events (minor) | `rec_gap` defined in wall time. Forum events inside gaps are dropped (unclamped `wall_to_video`). The 30 s post-restart exclusion is kept. |
| 37 | §10 population selects on participation; session numbering (minor) | Population stated plainly, with claims limited to it. `ctx_session_no` comes from the full section schedule, including unrecorded classes. |
| 38 | §8.3/§1.3/§8.5/§12 episode FAR gameable; FAR(τ) non-monotone (blocker) | Seizure-prediction firing rule with refractory H − g. A constant predictor fires every 14 s (≈ 4.3/min) and is penalised. Chance is corrected to the exact refractory form $R_0=\lambda_f(H-g)/60$ (Poisson kept as an approximation). TiW reported. Unit test: all-ones FAR = (60/14) × false share. §1.3 row and §14 equation updated; FROC drawn as the lower envelope. |
| 39 | §5.7 `ctx_course` < 10 training classes collapses courses; rows vs classes (major) | Rule set to ≥ 3 training classes, implemented as a custom class-count filter. `ctx_area` and section constants (`ctx_enrolled`, `ctx_sem_idx`) kept. H1 tested within class (AP_w) with ΔTG_local secondary. F-text vs F-struct gap reported per course. |
| 40 | §9.2 IH+ at +5/+10 could fail on a correct pipeline (major) | IH+ rebuilt as δ ∈ {+30, +60} including Brady's own words; +5/+10 are descriptive. The stop rule rests on P-clock and P-plant, both promoted to MVP. |
| 41 | §5.1 masking common words; missing ASR variants (major) | Exact names masked case-insensitively unless the name is a common word (then mid-sentence capitalised only). Distance-1 only for capitalised non-common tokens. Bradley, Braden and possessives added. Masked rate per class and top-20 surface forms reported locally in D11; fail on common words. |
| 42 | §2.2–2.4 pass 2 discards per-tile data; retune needs re-streaming (major) | Pass 2 writes `hand_tiles`: per frame and tile, x-span, frac_hand/frac_tick at two ROI sizes, shape, border, NCC, plus glyph bits (≈ 25 MB per class, local). Tracking, episodes, votes and retuning run offline. Privacy trade-off stated in §10; deleted after submission. |
| 43 | §11 D2 strip "per UI era" undefined (major) | Monday night: one frame at 30:00 per selected section (13 frames); eras = semesters unless the frames differ. Pass 1 auto-detects the strip per class (periodic gutters, n_tiles ≈ n_present); if not found → `unlocated`, never a fixed y. |
| 44 | §11 D2 highres fallback breaks the clock and the budget (major) | D2 records highres size and decode speed. The highres branch transcribes and scans the same highres file and confirms on 3 pilots via audio-envelope cross-correlation. Keyframe-only scan with full rate around transitions; Phase B capped at 100 classes. |
| 45 | §12 risky checks bunched on Tuesday; Phase C too late (major) | Moved to Monday night: a live Scribe run on 1 pilot (local provider pending consent), D2 frames, D8 single plus side-effect diff, D13 local benchmark, D14, D15 yield estimate. Phase C decided Tue 22:00, so data lands by Wed morning. Agents work in separate git worktrees. |
| 46 | §10 CV compute underestimated (major) | n* and C chosen once per outer fold on F-all and reused (declared in prereg). HGB max_iter 500, lr 0.1. lbfgs for LR. float32. `n_jobs=3` with nothing else running. One M3 fold timed Wed 09:00 and §10 rewritten. MVP set defined for Wed 20:00; M4, G, A-mismatch and N-name are must-if-G3. |
| 47 | §5.6/§7/D8 PCW coverage unknown; no rule for small coverage (major) | D8 samples one class per selected section. Pre-registered rule: H3, A-mismatch and F-all−pcw on `pcw_has = 1` (or S_pcw) dev classes. With < 40 classes or < 150 bids, H3 is exploratory and leaves the family, and §14 labels Δρ/U underpowered. Funnel reports coverage by course. |
| 48 | §11 D8/L12 POST may bump last-modified for the whole archive (major) | Metadata read before and after POST/GET and diffed on a past class; history endpoints probed with GETs. Never PUT/PATCH/websocket; harvest once and cache. If reads bump timestamps, edit time is unknown and PCW is accepted with a flag as an exploratory sensitivity row. |
| 49 | §10/D10 consent scope; Groq; local ASR speed (major) | D10 email asks explicitly about other courses' recordings, third-party ASR (Deepgram and Groq) and informing other instructors. Consent branches defined (Deepgram for CS156, local for the rest if needed). Groq used only if named. D13 benchmarks faster-whisper vs mlx-whisper on Monday night, and Phase B is budgeted from it. |
| 50 | §13 Scribe v3 untracked; editable path in pyproject (major) | Monday night: commit and push Scribe `v3` (public repo). Depend on `git+https://…@<sha>` from day one; local editable installs stay out of pyproject/lock. Notebook §0 prints the SHA; each fix bumps the SHA. |
| 51 | §14 missing equations for models not covered (major) | Added attention and CLS pooling, NCC, the HSV indicator and shape test, the hysteresis state machine, Platt, BM25 (if kept), GRU gates (if run) and the cross-encoder (if run). HGB and v_τ kept. Each equation tied to the run that used it. |
| 52 | §1.1 `git mv` on untracked dir; research notes archived; stale site/memory (minor) | Plain `mv` for PLAN.md only. research_notes kept in place and cited in §14. "Superseded" banner on `site/index.html` now; README link withheld. Memory note updated. |
| 53 | §7/§9.2 conflicting N-name criteria (minor) | N-name made descriptive, with lift CI and ΔAP(F-name − F-all). One flag: concerning if the CI lower bound > 0.02 AP. The other criterion deleted. |
| 54 | §7 must G sweep depends on should-tier re-timing (minor) | G (must) = g ∈ {2, 5}. G01 = {0, 1} is should-tier and conditional on re-timing with `retime_failed` < 10%. |
| 55 | §1.2 ε_F clamp vs D5 fallback (minor) | ε_F = max(ε_F,min, p95, `max_residual_s`), headline ≤ 1.5 s, sensitivity up to 3 s, excluded above 3 s. D5 fallback uses ε_F = 3 s explicitly and is reported as such. ε_F logged per class (`eps_F_s`). |
| 56 | §3 base-rate arithmetic (minor) | Corrected to π ≈ 1.0–1.3% at H = 10 and 1.6–2.2% at H = 15, with 45–60 at-risk min per class (≈ 190k–250k points). Re-measured from the Phase A funnel. |
| 57 | §8.4 shift units ambiguous (minor) | Global shift defined in index units over at-risk defined points: u_c ~ U{150, n_c − 150}. Classes with n_c < 400 skipped and counted. |
| 58 | §8.3 τ from inner OOF has a different score scale (minor) | Rate-matched thresholds: the inner-OOF firing rate at FAR = λ is matched by the outer model's firing rate on training-fold 𝒢. Realised FAR reported. |
| 59 | §9 L7 A-stream window shorter than features (minor) | A-stream cuts [t − 180, t] (≈ $4.50). The batch transcript is used only for `topic_shift`'s older window, documented. |
| 60 | §4 A16 vs "no pixels" and Brady's text-only wish (minor) | §4.1 states A16 is the label's own history (autoregressive hazard), not a pixel feature. F-all−me_hand is MVP and part of H1. |
| 61 | §10/§11 URL expiry and multi-session files unmeasured (minor) | D2 parses `Expires=` and logs the lifetime. ffmpeg reconnect flags; re-fetch and resume from last pts on 403. D14 checks a two-session pilot, and per-session files are stitched with `recording_timeline` offsets. |
| 62 | §5.6 compressed bge similarity range (minor) | All similarities z-scored against an earlier-only, label-free null of other-course chunks. `pcw_new_block` threshold = training-fold 90th percentile. U(t) uses z-scores with κ = 2. |
| 63 | §2.7 G2 precision criterion ambiguous (minor) | Pass = point ≥ 0.90 and Wilson LB ≥ 0.82; recall point ≥ 0.85 and LB ≥ 0.72 at n ≥ 40. Committed Monday with the vault rule. |
| 64 | §13/§14 notebook not reproducible; non-aggregate hero image; no diagram tool (minor) | Committed privacy-safe `results/` bundle (HMAC ids, OOF predictions, results, own bids) whitelisted with `!results/*.parquet`, so cells re-run for graders. Hero timeline is a labelled synthetic schematic. Diagram via `npx -y @mermaid-js/mermaid-cli`, matplotlib fallback. |
| 65 | §5.7 `ctx_hour` ambiguous across rotation cities (minor) | `ctx_hour_local` from a committed semester → IANA timezone table, plus `ctx_hour_utc` of the scheduled start. `rec_offset_s` uses the start field named on D1. |
| 66 | §8/§12 required learning figures depend on should items (minor) | Two must-tier plots: HGB staged log-loss (training and inner-validation) per outer fold, and LR training vs validation AP across C. LC and M5 stay should. |