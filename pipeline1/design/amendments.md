# Amendments to `on_cue_spec.md` (12 Oct 2026)

These override the spec where they conflict. They come from Brady's decisions in conversation and
from the model-options research in `reports/Hand raise prediction model options.md`.

## A1. The guard is a dial Brady controls (1, 2, 3 s)
Brady's framing: the model must predict *at least g seconds before* the hand goes up, with g an
easy test parameter. Condition **G** sweeps **g ∈ {1, 2, 3, 5}**, and it is **must**-tier
(previously {2, 5} must and {0, 1} should).
- **g = 1** needs onset re-timing. If `retime_failed` ≥ 10%, report g = 1 as approximate rather
  than dropping it.
- The headline stays H = 15, g = 2.
- `points` gets label columns `y_H15_g1`, `y_H15_g3` alongside the existing ones.

## A2. Brady's own history is first-class
- **Within-class history** (`me_hand_*`: raises so far this class, time since last raise or
  lowering, last outcome) stays in F-all. Brady's example of the pattern he wants learnable: "I
  never answer more than 5 times".
- **Cross-class history** (`hist_*`: his bid rate per at-risk minute in this course and overall,
  from *earlier classes only*) moves from could to **should**. It is computed fold-aware: for a
  row in class c, use only classes with date < date(c) **that are in the training fold** of the
  current split. Under S-time this is natural.
- Condition **F-all+hist** reports the gain. Never compute `hist_*` from test-fold labels.

## A3. Pretrained classroom-discourse scores (should-tier feature group `edu_*`)
Run these Stanford EduBehaviors SetFit classifiers once over professor frames, at runtime:
- `StanfordSCALE/assertion_sentence_invites_participation`
- `StanfordSCALE/assertion_sentence_has_a_question`
- `StanfordSCALE/assertion_sentence_has_a_rhetorical_question`
- `StanfordSCALE/assertion_sentence_addresses_the_whole_class`
- `StanfordSCALE/assertion_sentence_calls_on_student_by_name`

Rules:
- **Caching:** scores are cached per frame and never committed. Licences are unstated, so
  download at runtime only.
- **Input:** score the masked, normalised text (no punctuation, consistent with §4).
- **Join:** join on frame `close_s ≤ t − ε_w`, then decay by time since close, like `q_*`.
- **Condition:** **F-all+edu** reports the gain.
- **Fallback:** if a checkpoint fails to load, the group is skipped and logged, not faked.

## A4. Stacking (should-tier, condition M3+stack)
Feed grouped inner-OOF probabilities from M2 (TF-IDF LR) and M4 (embeddings LR) into M3 as two
extra features. Inner OOF uses `GroupKFold(3)` by class inside each outer training fold.

## A5. kNN as an explainer (could-tier)
- `NearestNeighbors` over pooled window embeddings, drawn from training classes only.
- Output is **metadata only**: segment type, minute, whether a bid followed, distance. Never
  output text, because exemplars would contain classmates' words.
- Used for one aggregate figure: the share of nearest neighbours of true positives versus of
  false alarms that preceded a bid.

## A6. Correction to D0 (access)
The spec says Claude's Chrome profile lacks Forum permission. That is wrong.
- **What worked:** Claude's tab used Brady's Minerva session, and every Forum API call worked on
  12 Oct, including class-events, class_edit_page, PCW and the SLE doc/template.
- **What failed:** recordings would not load in the automated tab, and the browser tool refuses
  to return signed URLs or cookies (a deliberate safeguard).
- **Fix:** the cookie in `~/.minerva/.env` (symlinked as `pipeline1/.env`) is the route for every
  CLI step. Route C (a console exporter) remains a fallback.

## A7. Scribe dependency before Brady approves a push
Branch `v3` of `Minervas_scribe` is local and uncommitted, and the repo is public. Until Brady
approves committing and pushing it:
- use an unrecorded editable install: `uv pip install -e ../Minervas_scribe`;
- `pyproject.toml` and `uv.lock` must **never** contain a local path;
- final results require a pushed SHA, and Brady's approval for that is pending.

## A8. Consent and third-party ASR are Brady's decisions
- **Recommendation, not a blocker:** the D10 email to Prof. Watson (which courses; third-party
  ASR) is still recommended.
- **Provider per course:** the code chooses the ASR provider per course from a config map
  (`deepgram` | `local`), so it never blocks on consent.
- **Default:** local is the default for courses not yet marked approved.

## A9. Research-backed additions
- **Baselines:** add a *matched-rate random alarm* reference to EV, next to AO and the shifted
  prediction.
- **Calibration:** use Platt only, never isotonic, because the effective n is the number of
  bids.
- **Report:** the **Brier skill score** next to Brier.
- **Interpretability:** feature-group ablations are the primary evidence. Permutation importance
  and SHAP are not headline. HistGradientBoosting has no TreeSHAP, so use grouped permutation
  importance only as could-tier, with the correlated-features caveat.
- **Sequence model:** M5 (GRU) is capped at 10–50k parameters, and a causal TCN is an allowed
  alternative.

## A10. The split: per course, 60/20/20, after removing absences (Brady, 13 Oct)
This replaces the hashed 20% vault plus 5 rotating folds as the headline split.
1. **Usable classes:** recorded, Brady on the roster, and Forum's `absent` flag is False.
   Classes with no attendance data (the summer-program sections) are not usable unless Brady
   decides otherwise.
2. **Split within each course** (`splits.course_split`): units (same-day classes grouped by
   `cv_group`) go in a seeded hash order. The first 20% are **test**, the next 20%
   **validation**, and the rest **train**, each at least 1. Courses with fewer than 5 usable
   units are excluded and reported.
3. **How each split is used:**
   - **test:** touched once, at the end;
   - **validation:** hyperparameters, thresholds and Platt calibration;
   - **train:** fitting only.
4. **Uncertainty:** cluster bootstrap over test classes. Rotating folds stay available only as
   a robustness check.
5. **Not yet switched:** `experiments.py` still runs the fold design; switching it is a
   separate change for Brady to review.

## A11. Cleaning and split outputs are committed (Brady, 13 Oct)
`data/02_clean/` and `data/03_split/` are committed as part of the assignment. They hold class
ids, course codes, dates, session titles and Brady's own attendance, and nothing about
classmates.
- **Stays local:** raw Forum records, transcripts, event logs, frames, embeddings and features.
- **Pending decision:** Brady considers all class data fair game under Minerva's terms, so
  whether masked transcripts are also committed is still his to make.

## A12. Forum never logs the professor's speech (verified on real data, 13 Oct)
Across all 717 synced classes with voice events, the teacher (`listing.class.teacher-user-id`)
has **zero** voice events. Forum's voice log covers students only: it is the data behind
Talk Time. The median logged student talk is 28 min per class; most of the rest is the
professor. Two things assumed otherwise and must be fixed:
1. **Scribe v3 fusion:** map the dominant main-room diarization cluster that matches no student
   to `teacher-user-id`, labelled `how = "teacher-cluster"`, instead of leaving it
   "Unattributed".
2. **Pipeline roles:** a word heard while no student voice event is active (after calibration,
   main room) is role `professor`. Every `q_*` and "professor talking" feature depends on this,
   and it is causal.

Both fixes need a test built on the real pattern: a professor with no voice events.

**Implemented (13 Oct).** Testing the fix turned up two more places that assumed professor mic
events: the clock calibration and the speaker features.
- **Calibration** lined up "someone is talking" in the audio with "a student mic is on". The
  professor fills the gaps between students, so every offset looked alike: in a stress test of
  360 simulated classes it found no fit in most classes and was 1.2-14 s off. It now leaves out
  the professor's diarization cluster (`scribe_core.fuse.calibration_words`, used by Scribe and
  by `transcribe.calibrate_class`): 0.03 s median error on ms timestamps, the same as when the
  professor had events. On whole-second (2023) timestamps it lands about 0.5 s from the true
  clock, which is the rounding bias of those timestamps absorbed into the fit, so words still go
  to the right speaker. Groq/local Whisper have no clusters, so their calibration stays
  unreliable: Deepgram is required.
- **Scribe fusion** maps the professor's clusters to `teacher-user-id` first (before the
  Hungarian step, which would otherwise call the professor's cluster a split of whichever
  student their voice bleeds into): the biggest cluster with >= 30 s of speech, >= 60% of it
  while no student mic is on, plus any other such cluster spread over >= 3 five-minute stretches
  (diarizers split long talkers; a video clip or guest is one block and stays "Unattributed").
  Without clusters the v2 rule applies, labelled `no-student-mic`. The teacher id comes from the
  live class record (`/classes/{id}`, now fetched) or the sync's `listing.json`.
- **Pipeline:** words with no main-room mic event are role `prof`, with
  `role_avail_s = end + EDGE + eps_F` (a covering student event would have shown by then).
  The professor's turns are added to the `voice` table as `inferred` rows (`professor_turns`)
  carrying `start_avail_s` / `end_avail_s`, so `spk_cur_role`, `spk_prof_share_*`,
  `spk_turns_since_prof` and silence-after-professor are real again and stay causal. A turn
  reads as ongoing until 1.2 s pass with no word, or the next speaker's word is known.
- **Masking:** "sentence-initial" no longer counts speaker changes. Turn edges are a fraction
  of a second off, and a call-on name glued to the next turn ("... your hand, Mary") escaped
  masking. Punctuation alone over-masks at worst.
- The synthetic world no longer emits professor voice events, so every test runs on the real
  pattern.
