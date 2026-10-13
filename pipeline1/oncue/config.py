"""On Cue constants, paths, regexes, allowlists and pre-registered QA rules.

Every number here comes from `design/on_cue_spec.md` §1.2 (and the sections it points to),
overridden where `design/amendments.md` says so. Values marked PROVISIONAL are placeholders
until a Day-1 check (D1, D2, D9) measures them; change them only through that check.

Secrets are never stored here. `ID_SALT` is read from the environment at call time, never
cached and never logged. Nothing in this module reads `.env`: call `load_env()` explicitly
from a CLI step that needs secrets.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
from pathlib import Path

# --------------------------------------------------------------------------- paths

PACKAGE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PACKAGE_DIR.parent                   # cs156-ml/pipeline1
REPO_DIR = PIPELINE_DIR.parent                      # cs156-ml
ENV_PATH = PIPELINE_DIR / ".env"                    # symlink to ~/.minerva/.env; never read here

DATA_DIR = PIPELINE_DIR / "data"                    # gitignored: everything harvested
SCRIBE_DIR = DATA_DIR / "scribe"                    # `scribe sync --out` root
RAW_DIR = SCRIBE_DIR / "raw"                        # raw/{class_id}/*.json
MANIFEST_PATH = SCRIBE_DIR / "manifest.csv"
ME_PATH = SCRIBE_DIR / "me.json"
SECTIONS_PATH = SCRIBE_DIR / "sections.json"
SCHEDULE_DIR = SCRIBE_DIR / "schedule"              # paginated-classes per section: {sid}.json
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"          # Scribe *_ids.json and *_asr.json
TABLES_DIR = DATA_DIR / "tables"                    # *.parquet (schemas.py)
CACHE_DIR = DATA_DIR / "cache"
EMB_CACHE_DIR = CACHE_DIR / "emb"                   # sha1-keyed float16 embeddings
EDU_CACHE_DIR = CACHE_DIR / "edu"                   # EduBehaviors scores (A3), never committed
LOGS_DIR = DATA_DIR / "logs"
DROPPED_KEYS_LOG = LOGS_DIR / "dropped_keys.csv"    # loaders.py: non-whitelisted keys
OOF_DIR = DATA_DIR / "oof"                          # local OOF predictions (raw class ids)
RESULTS_DIR = PIPELINE_DIR / "results"              # committed, privacy-safe bundle (HMAC ids)
FIGURES_DIR = PIPELINE_DIR / "figures"              # committed, aggregate-only figures
AUDIT_DIR = PIPELINE_DIR / "audit"                  # gitignored: contact sheets, clips


def table_path(name: str) -> Path:
    """`data/tables/{name}.parquet`."""
    return TABLES_DIR / f"{name}.parquet"


def raw_class_dir(class_id) -> Path:
    """`data/scribe/raw/{class_id}/`, the layout written by `scribe sync` (plus harvest files)."""
    return RAW_DIR / str(class_id)


# Files inside raw/{cid}/. The first block is written by `scribe sync` (BUNDLE_FILES in
# Minervas_scribe hosts/cli.py); the second by `oncue harvest` / `harvest-import`.
RAW_SOURCE_FILES = {
    "class": "class.json",
    "events": "events.json",
    "polls": "polls.json",
    "poll_responses": "poll_responses.json",
    "outcome_assessments": "outcome_assessments.json",
    "focused_outcomes": "focused_outcomes.json",
    "breakouts": "breakouts.json",
    "analytics": "analytics.json",
    "listing": "listing.json",
    "class_edit_page": "class_edit_page.json",
    "pcw_template": "pcw_template.json",
    "pcw_doc": "pcw_doc.json",
    "pcw_meta": "pcw_meta.json",
}

# --------------------------------------------------------------------------- secrets


ID_SALT_ENV = ("ONCUE_ID_SALT", "ID_SALT")  # first non-empty wins


def load_env(path: Path | str | None = None) -> bool:
    """Load `pipeline1/.env` into os.environ (no override). Only CLI steps that need a secret
    call this; tests never do. Returns True if a file was loaded."""
    from dotenv import load_dotenv

    p = Path(path) if path is not None else ENV_PATH
    return bool(p.exists() and load_dotenv(p, override=False))


def id_salt() -> str:
    """The HMAC key for ids in committed results, read from the environment on every call."""
    for name in ID_SALT_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        f"No ID salt: set {ID_SALT_ENV[0]} in pipeline1/.env (never commit it).")


def hash_id(value, salt: str | None = None) -> str:
    """`HMAC-SHA256(ID_SALT, str(value))[:12]` for class ids (and anything else) in results/."""
    key = salt if salt is not None else id_salt()
    return hmac.new(key.encode(), str(value).encode(), hashlib.sha256).hexdigest()[:12]


# --------------------------------------------------------------------------- vault, seed

SEED = 156
VAULT_SALT = "oncue-vault"
VAULT_MOD = 5  # one class in five


def in_vault(class_id) -> bool:
    """Pre-registered vault rule (§8.1): sha1(f"{class_id}:oncue-vault") % 5 == 0 (~20%)."""
    digest = hashlib.sha1(f"{class_id}:{VAULT_SALT}".encode()).hexdigest()
    return int(digest, 16) % VAULT_MOD == 0


# --------------------------------------------------------------------------- grid, target

DELTA_S = 2.0                         # decision stride Δ
H_S = 15                              # horizon H (headline)
H_SWEEP = (5, 10, 15, 30, 60)
G_S = 2                               # guard g (headline): onset must be > t + g
G_SWEEP = (1, 2, 3, 5)                # A1: must-tier sweep
G_LABELS = (0, 1, 2, 3, 5)            # every g with a label column (g = 0 kept from the spec)
G1_RETIME_FAILED_MAX = 0.10           # A1: above this share, g = 1 is reported as approximate
G_SPEECH_S = 3                        # T-speech guard g_s
T_OPEN_S = 120.0                      # skip the start of the recording
R_LOW_S = 10.0                        # refractory after lowering and after a glitch
S_ME_S = 3.0                          # exclusion after Brady's last word
REC_RESTART_SKIP_S = 30.0             # rec_gap: < 30 s after a restart
ONSET_PENDING_S = 1.5                 # onset in (t - 1.5, t] not yet live
BREAKOUT_VOICE_LOOKBACK_S = 60.0      # breakout rule: Brady's voice in [t - 60, t]
SPEECH_FRESH_SILENCE_S = 30.0         # T-speech: t + H - last_me_end >= 30


def y_col(H: float, g: float, *, sym: bool = False, speech: bool = False) -> str:
    """Label column name, e.g. y_col(15, 2) == "y_H15_g2"."""
    if speech:
        return f"y_speech_H{int(H)}_g{int(g)}"
    return f"y_H{int(H)}_g{int(g)}" + ("_sym" if sym else "")


HEADLINE_LABEL = y_col(H_S, G_S)                     # "y_H15_g2"
LABEL_COLUMNS = tuple(dict.fromkeys(
    [HEADLINE_LABEL]
    + [y_col(h, G_S) for h in H_SWEEP if h != H_S]   # y_H5_g2 ... y_H60_g2
    + [y_col(H_S, g) for g in G_LABELS if g != G_S]  # y_H15_g0, g1, g3, g5
    + [y_col(H_S, G_S, sym=True), y_col(H_S, G_SPEECH_S, speech=True)]
))

LABEL_POS, LABEL_NEG, LABEL_UNDEF = "pos", "neg", "undef"
LABEL_STATES = (LABEL_POS, LABEL_NEG, LABEL_UNDEF)

# At-risk exclusion codes, in the order the first matching rule wins (§3).
EXCL_CAUSAL = ("opening", "end", "rec_gap", "breakout", "break", "me_speaking", "hand_up",
               "post_lowering", "post_glitch")
EXCL_LABEL = ("unobservable", "guard", "onset_pending", "vote_or_censored", "glitch_window")
EXCL_CLASS = ("class_qa",)
EXCL_REASONS = EXCL_CAUSAL + EXCL_LABEL + EXCL_CLASS
# Causal-grid rules (§8.3): `end` is label-side right-censoring but still removes points from G.
GRID_RULES = tuple(r for r in EXCL_CAUSAL if r != "end")


def alarm_refractory(H: float = H_S, g: float = G_S) -> float:
    """Firing refractory period H - g (§8.3)."""
    return float(H) - float(g)


# --------------------------------------------------------------------------- clock and timing

EPS_W_S = 0.25                        # word margin ε_w
EPS_F_MIN_WHOLE_S = 1.0               # ε_F,min for whole-second (2023) timestamps
EPS_F_MIN_MS_S = 0.5                  # ε_F,min otherwise
EPS_F_HEADLINE_MAX_S = 1.5
EPS_F_SENSITIVITY_MAX_S = 3.0
EPS_F_POLL_MIN_S = 1.0
EPS_F_FALLBACK_S = 3.0                # D5 declared fallback (recording-started arithmetic)
PROF_TURN_GAP_S = 1.2                 # professor words this close form one turn (Scribe's pause)


def eps_f(timing_precision: str | None, resid_p95_s: float = 0.0,
          max_residual_s: float = 0.0) -> float:
    """ε_F = max(ε_F,min, p95 residual, max_residual_s); ε_F,min = 1.0 s for "s" precision."""
    floor = EPS_F_MIN_WHOLE_S if timing_precision == "s" else EPS_F_MIN_MS_S
    vals = [floor] + [float(v) for v in (resid_p95_s, max_residual_s)
                      if v is not None and not math.isnan(float(v))]
    return max(vals)


def eps_f_poll(eps_f_s: float) -> float:
    """ε_F,poll = max(ε_F, 1.0 s) for poll times parsed from strings."""
    return max(float(eps_f_s), EPS_F_POLL_MIN_S)


# Calibration QA (§3 class QA, §4.3). A global-fallback fit is never headline.
CAL_OK_STATUSES = ("ok", "piecewise")
CAL_METHOD_REQUIRED = "windowed"
CAL_MIN_WINDOWS = 6
CAL_MAX_RESIDUAL_S = 1.5
CAL_RESID_P95_MAX_S = 1.5
CAL_RESID_MED_MAX_S = 0.5
CAL_SENSITIVITY_MAX_S = 3.0           # sensitivity admits residuals up to 3 s
RESID_SEARCH_S = 2.0                  # residual = |cal voice start - nearest word start| within ±2 s
RESID_PRE_SILENCE_S = 2.0             # over voice events preceded by >= 2 s of silence


def calibration_qa(cal: dict | None, resid_med_s: float | None = None,
                   resid_p95_s: float | None = None, *, sensitivity: bool = False
                   ) -> tuple[bool, str]:
    """(qa_pass, reason) for a Scribe `calibrate()` result plus measured residuals."""
    if not cal:
        return False, "no_calibration"
    limit = CAL_SENSITIVITY_MAX_S if sensitivity else CAL_MAX_RESIDUAL_S
    p95_max = CAL_SENSITIVITY_MAX_S if sensitivity else CAL_RESID_P95_MAX_S
    if not cal.get("applied"):
        return False, "cal_not_applied"
    if cal.get("status") not in CAL_OK_STATUSES:
        return False, f"cal_status_{cal.get('status')}"
    if cal.get("method") != CAL_METHOD_REQUIRED:
        return False, f"cal_method_{cal.get('method')}"
    if len(cal.get("windows") or []) < CAL_MIN_WINDOWS:
        return False, "cal_few_windows"
    max_res = cal.get("max_residual_s")
    if max_res is None or not float(max_res) <= limit:
        return False, "cal_max_residual"
    if resid_p95_s is None or not resid_p95_s <= p95_max:
        return False, "resid_p95"
    if resid_med_s is None or not resid_med_s <= CAL_RESID_MED_MAX_S:
        return False, "resid_median"
    return True, "ok"


# A/V lag (§2.1)
AV_LAG_TOL_S = 0.5                    # |lag_c - median lag| <= 0.5 s
AV_LAG_MEDIAN_RANGE_S = (-0.25, 1.0)  # outside: stop and investigate
AV_LAG_SEARCH_S = 3.0
AV_LAG_RES_S = 0.04
AV_BRADY_MIN_SPEECH_S = 20.0
MULTI_SESSION_BOUNDARY_TOL_S = 5.0


def av_lag_ok(lag_s: float, median_lag_s: float) -> bool:
    return abs(float(lag_s) - float(median_lag_s)) <= AV_LAG_TOL_S


# --------------------------------------------------------------------------- hand detector

F_SCAN = 2.0                          # samples per second
THETA_ON = 0.20                       # baseline-subtracted icon fraction for UP
THETA_OFF = 0.08                      # ... for DOWN (hysteresis)
DOWN_BASELINE_FLAG = THETA_OFF / 2    # flag a class whose DOWN baseline exceeds this
SHAPE_CIRCULARITY_MIN = 0.6
SHAPE_GLYPH_FRAC = (0.08, 0.45)       # white-glyph share of the enclosing circle
ROI_S1 = 0.30                         # PROVISIONAL (D2): corner ROI side / tile height
ROI_S2 = 0.40                         # PROVISIONAL (D2)
ROI_INSET_PX = 3                      # excludes the speaker-border ring
LABEL_INSET_PX = 4
K_CONFIRM = 2                         # agreeing samples for a confirmed change (1 s)
D_PRE_S = 2.0                         # confirmed DOWN required before an onset
NA_BRIDGE_S = 1.0
DOWN_GAP_MERGE_S = 3.0
MIN_EPISODE_S = 1.0                   # shorter confirmed episode = glitch
ME_HAND_MIN_S = 2.0                   # min duration for me_hand_* eligibility
ME_HAND_MIN_SHAPE_SAMPLES = 2
CALLED_AFTER_OFFSET_S = 5.0           # `called`: Brady turn starts in [o, offset + 5]
ME_HAND_OUTCOME_DELAY_S = 5.0         # me_hand_last_outcome available at offset + 5 s

# OpenCV HSV (H in 0..179, S and V in 0..255)
HAND_H_RANGE = (26, 40)               # hue 52-80 deg, inclusive
HAND_SV_MIN = 115                     # S > 115 and V > 115 (s, v > 0.45)
TICK_H_RANGE = (60, 82)               # poll tick, hue 120-165 deg
WHITE_S_MAX = 0.25 * 255              # white glyph: S < 63.75 and V > 204
WHITE_V_MIN = 0.80 * 255

# Strip and tiles (§2.2)
GUTTER_V_MAX = 0.12                   # V fraction; gutters are runs of >= 2 px below it
GUTTER_MIN_PX = 2
TILE_WIDTH_TOL = 0.20
STRIP_H_TOL = 0.30
N_TILES_TOL = 2
PASS1_N_FRAMES = 6
GLYPH_H, GLYPH_W = 12, 48
NCC_MIN = 0.75
NCC_GAP = 0.10
ANCHOR_MIN_DUR_S = 2.0
ANCHOR_MAX_EVENTS = 12
ANCHOR_FRAMES_PER_EVENT = 3
MIN_ANCHORS = 3
TRACK_VALID_MIN = 0.90
COVERAGE_MIN = 0.30
LEFT_CENSORED_MAX_SHARE = 0.10

# Re-timing (§2.4, should)
RETIME_PRE_S = 1.0
RETIME_SPAN_S = 1.5                   # [t_up - 1.0, t_up + 0.5]: onset + 3 positive frames fit
RETIME_MIN_POS_FRAMES = 3

# Votes (§2.5)
VOTE_MIN_DN = 4
VOTE_FRAC = 0.3
VOTE_DN_WINDOW_S = (-5.0, 10.0)       # Δn over [o - 5, o + 10] vs n(o - 5)
VOTE_LEX_LOOKBACK_S = 60.0            # VOTE lexicon in the professor's words in [o - 60, o]
CONTESTED_LOOKBACK_S = 60.0


def vote_threshold(n_tiles: int, frac: float = VOTE_FRAC) -> int:
    """Δn_others needed for a vote: max(4, ceil(frac * (n_tiles - 1)))."""
    return max(VOTE_MIN_DN, math.ceil(frac * (int(n_tiles) - 1)))


STATE_UP, STATE_DOWN, STATE_NA = "UP", "DOWN", "NA"
EP_BID, EP_VOTE, EP_GLITCH = "bid", "vote", "glitch"
EP_TYPES = (EP_BID, EP_VOTE, EP_GLITCH)
OUTCOMES = ("called", "withdrawn", "open")
TILE_STATUSES = ("located", "unlocated", "tracking_failed")

# --------------------------------------------------------------------------- speech, frames

TURN_MERGE_GAP_S = 2.0
FRESH_SILENCE_S = 30.0
FRAME_PAUSE_S = 0.4                   # pause cut
FRAME_MAX_S = 5.0                     # length cut
LASTPROF_PAUSE_S = 1.2                # X_lastprof splits at a pause > 1.2 s
WIN_SHORT_S = 15.0                    # X_all15
WIN_LONG_S = 60.0                     # X_all60, X_prof60, X_stud60, X_chat60
POOL_HORIZON_S = 180.0                # frames pooled in (t - 180, t]
SPK_WINDOWS_S = (60, 300)

ROLE_PROF, ROLE_ME, ROLE_STUDENT, ROLE_UNK = "prof", "me", "student", "unk"
ROLES = (ROLE_PROF, ROLE_ME, ROLE_STUDENT, ROLE_UNK)
PROF_ROLES = frozenset({"professor", "instructor", "teacher", "faculty", "staff", "ta"})
ROLE_TAGS = {ROLE_PROF: "<P>", ROLE_STUDENT: "<S>", ROLE_ME: "<M>", ROLE_UNK: "<U>",
             "chat": "<C>"}
ROLE_PREFIX = {ROLE_PROF: "p_", ROLE_STUDENT: "s_", ROLE_ME: "m_", ROLE_UNK: "u_", "chat": "c_"}
MAIN_ROOM_SIGNATURE = "Main Classroom"

# --------------------------------------------------------------------------- masking (§5.1)

NAME_TOKEN = "[NAME]"
ME_TOKEN = "[ME]"
BRADY_VARIANTS = ("Brady", "Braydon", "Bradley", "Braden", "Rutherford")
MASK_LEV_MIN_LEN = 4                  # Levenshtein <= 1 only for tokens of >= 4 chars
MASK_LEV_MAX = 1
COMMON_WORDS_TOPN = 20_000            # wordfreq top-20k English = "common word"
MASK_FAIL_TOPN = 1_000                # build fails if a non-roster masked form is in the top 1k
MASK_PRECISION_N = 200
MASK_PRECISION_MIN = 0.95

# --------------------------------------------------------------------------- lexicons (§5.5)
# Applied case-insensitively to normalised, masked text (lowercase, no punctuation). Patterns
# tolerate a dropped apostrophe ("lets", "whod"), because normalisation strips punctuation.

INVITE_RE = re.compile(
    r"\b(?:any(?:one|body)|some(?:one|body)|who(?:\s(?:wants|would like|can|has)|'?d like)"
    r"|any (?:thoughts|comments|questions|volunteers)|thoughts on"
    r"|what do (?:you|we|people)(?: all)? think|would anyone|does anyone|can someone"
    r"|let'?s hear|what did your group|how would you|why do you think|anything else|what else"
    r"|build on|jump in|feel free)\b", re.IGNORECASE)
VOTE_RE = re.compile(
    r"\b(?:raise your hands? if|show of hands|hands up if|put your hands? up if|thumbs up"
    r"|let'?s vote)\b", re.IGNORECASE)
CALLON_RE = re.compile(
    r"\b(?:go ahead|i see (?:a|your) hand|you'?re up|unmute|take it away)\b", re.IGNORECASE)
CHECK_RE = re.compile(r"\b(?:make sense|makes sense|any questions)\b", re.IGNORECASE)

# Question test (headline, no punctuation): first token is a wh-word / inverted auxiliary and
# the utterance has >= 4 words, or it ends in a tag.
Q_FIRST_TOKENS = ("what", "why", "how", "which", "who", "where", "when", "is", "are", "do",
                  "does", "did", "can", "could", "would", "should", "have", "has")
Q_FIRST_RE = re.compile(r"^\s*(?:" + "|".join(Q_FIRST_TOKENS) + r")\b", re.IGNORECASE)
Q_TAG_RE = re.compile(r"\b(?:right|yeah|okay|isn'?t it|don'?t you think)\s*$", re.IGNORECASE)
Q_MIN_WORDS = 4
Q_SILENCE_S = 0.8                     # the detector runs after >= 0.8 s of observed silence
Q_RHETORICAL_RESUME_S = 3.0           # professor resumes within 3 s => rhetorical
Q_OPEN_STUDENT_MIN_WORDS = 3          # q_open ends at a student frame of >= 3 words
Q_AGE_CAP_S = 300.0
Q_COUNT_WINDOW_S = 300.0
Q_KINDS = ("invite", "check", "rhetorical", "plain")
R_DECAY_S = 10.0                      # rule baseline R: exp(-min(q_inv_age, 2 q_age) / 10 s)

# --------------------------------------------------------------------------- seg_type (§5.7)
# First match on `section_title + " " + segment_title`, case-insensitive.

SEG_TYPE_RULES = (
    ("breakout_instr", re.compile(r"breakout instr", re.IGNORECASE)),
    ("breakout", re.compile(r"\bbreakouts?\b", re.IGNORECASE)),
    ("break", re.compile(r"eye rest|\bbreak\b", re.IGNORECASE)),
    ("poll", re.compile(r"poll|vote|quiz", re.IGNORECASE)),
    ("debrief", re.compile(r"debrief|report back|share out|discussion", re.IGNORECASE)),
    ("intro", re.compile(r"intro|overview|recap|review|warm", re.IGNORECASE)),
    ("wrapup", re.compile(r"wrap|reflect|summary|closing|conclu", re.IGNORECASE)),
    ("activity", re.compile(r"activity|exercise|problem|case", re.IGNORECASE)),
)
SEG_TYPES = tuple(name for name, _ in SEG_TYPE_RULES) + ("other",)


def seg_type_of(segment_title: str | None, section_title: str | None = None) -> str:
    """seg_type for a timeline segment (first matching rule, else "other")."""
    text = f"{section_title or ''} {segment_title or ''}"
    for name, pattern in SEG_TYPE_RULES:
        if pattern.search(text):
            return name
    return "other"


# --------------------------------------------------------------------------- class context

PLANNED_CLASS_MIN = 90.0              # planned remaining = 90 - ctx_min
CTX_SPLINE_KNOTS = (10, 25, 45, 65, 80)
CTX_COURSE_MIN_CLASSES = 3            # courses with < 3 training classes -> "other"
DURATION_CAP_S = 3600.0
COURSE_AREAS = ("CS", "SS", "NS", "AH", "B", "other")

TERMS = ("F23", "Sp24", "F24", "Sp25", "F25", "Sp26", "F26")
SEMESTER_IDX = {term: i + 1 for i, term in enumerate(TERMS)}   # F23 = 1 ... F26 = 7
S_TIME_TRAIN_MAX_IDX = 5              # S-time: train F23-F25, test Sp26 and F26
# Semester -> IANA timezone of Brady's city. PROVISIONAL: Brady commits this table (Mon, §12).
TERM_TZ: dict[str, str | None] = {term: None for term in TERMS}


def course_area(course_code: str | None) -> str:
    """CS/SS/NS/AH/B from the course-code prefix, else "other"."""
    m = re.match(r"\s*([A-Za-z]+)", course_code or "")
    prefix = m.group(1).upper() if m else ""
    if prefix in ("CS", "SS", "NS", "AH"):
        return prefix
    if prefix.startswith("B"):
        return "B"
    return "other"


def term_code(term_name: str | None) -> str | None:
    """'Fall 2023' -> 'F23', 'Spring 2026' -> 'Sp26'; a code passes through unchanged."""
    if not term_name:
        return None
    text = str(term_name).strip()
    if text in SEMESTER_IDX:
        return text
    m = re.search(r"(fall|spring)\D*(\d{2,4})", text, re.IGNORECASE)
    if m:
        season, year = m.group(1), m.group(2)
    else:
        m = re.search(r"(\d{4})\D*(fall|spring)", text, re.IGNORECASE)
        if not m:
            return None
        year, season = m.group(1), m.group(2)
    return ("F" if season.lower() == "fall" else "Sp") + year[-2:]


# --------------------------------------------------------------------------- selection (§10)

POP_MIN_MEAN_MAIN_VOICE = 5.0         # sections where Brady's mean main-room voice events >= 5
SECTIONS_PER_SEMESTER = 2
PHASES = ("pilot", "A", "B", "C")
PHASE_A_PER_SECTION = 3
GO_DEV_BIDS = 300
H3_MIN_CLASSES = 40                   # PCW coverage rule (§5.6)
H3_MIN_BIDS = 150

# --------------------------------------------------------------------------- PCW (§5.6)

PCW_STATUSES = ("asof_start", "verified", "git", "empty", "none", "unknown")
PCW_USABLE_STATUSES = frozenset({"asof_start", "verified", "git"})
PCW_CHUNK_WORDS = 120
PCW_CHUNK_OVERLAP = 30
ZNULL_N = 200                         # reference chunks from other courses, earlier classes
ZNULL_MIN = 50                        # z = NaN below this
KAPPA = 2.0                           # "already said" threshold for U(t), z-units
PCW_REF_RECENT = 5
PCW_REF_MIN = 3
PCW_NEW_BLOCK_PCTL = 90.0

# --------------------------------------------------------------------------- embeddings (§5.4)

EMB = "BAAI/bge-small-en-v1.5"
EMB_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
EMB_DEVICE = "mps"
EMB_BATCH = 128
EMB_TAUS_S = (15.0, 60.0)

# A3: EduBehaviors SetFit classifiers, scored at runtime on masked normalised professor frames.
EDU_CHECKPOINTS = {
    "edu_invites": "StanfordSCALE/assertion_sentence_invites_participation",
    "edu_question": "StanfordSCALE/assertion_sentence_has_a_question",
    "edu_rhetorical": "StanfordSCALE/assertion_sentence_has_a_rhetorical_question",
    "edu_whole_class": "StanfordSCALE/assertion_sentence_addresses_the_whole_class",
    "edu_calls_on_name": "StanfordSCALE/assertion_sentence_calls_on_student_by_name",
}
EDU_DECAY_TAU_S = 10.0                # PROVISIONAL: decay by time since frame close (A3)

# A2: cross-class history, fold-aware (earlier classes in the training fold only).
HIST_COLUMNS = ("hist_bid_rate_course", "hist_bid_rate_all")

# A4: stacking features (grouped inner-OOF probabilities from M2 and M4 fed to M3).
STACK_COLUMNS = ("stack_m2", "stack_m4")
STACK_INNER_FOLDS = 3

# A5: kNN explainer, metadata only.
KNN_K = 10
KNN_OUTPUT_COLUMNS = ("seg_type", "minute", "bid_followed", "distance")

# --------------------------------------------------------------------------- ASR (A8)

ASR_PROVIDERS = ("deepgram", "local")
ASR_DEFAULT_PROVIDER = "local"
# course code -> provider. Brady adds a course here ONLY once its consent covers third-party
# ASR (D10). Anything absent runs locally.
ASR_PROVIDER_BY_COURSE: dict[str, str] = {}
SRC_RES = "lowres"
TRANSCRIBE_PRIVACY = "ids"
TRANSCRIBE_FORMATS = ("json", "jsonl")
TRANSCRIBE_LANGUAGE = "en"


def asr_provider(course_code: str | None, mapping: dict[str, str] | None = None) -> str:
    """Provider for a course: the configured map, defaulting to "local" (A8)."""
    table = ASR_PROVIDER_BY_COURSE if mapping is None else mapping
    provider = table.get(str(course_code or "").strip(), ASR_DEFAULT_PROVIDER)
    if provider not in ASR_PROVIDERS:
        raise ValueError(f"unknown ASR provider {provider!r} for {course_code!r}")
    return provider


# --------------------------------------------------------------------------- models, splits

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
B1_C_GRID = (0.01, 0.1, 1.0)
M1_ALPHA_GRID = (0.1, 0.3, 1.0)
M2_C_GRID = (0.03, 0.1, 0.3)
LR_MAX_ITER = {"B1": 2000, "M2": 3000}
HGB_PARAMS = {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 100,
              "l2_regularization": 1.0, "max_features": 0.8,
              "categorical_features": "from_dtype", "early_stopping": False}
HGB_MAX_ITER = 500                    # n* <= 500, chosen once per outer fold on F-all
TFIDF_PARAMS = {"ngram_range": (1, 2), "sublinear_tf": True, "max_features": 30_000}
TFIDF_MIN_CLASSES = 5                 # min_df = token present in >= 5 training classes
PCA_DIMS = {"v60": 32, "e_last": 16, "tfidf_svd": 64}
CALIBRATION_METHOD = "sigmoid"        # A9: Platt only, never isotonic
GRU_HIDDEN = 64
GRU_PARAM_RANGE = (10_000, 50_000)    # A9
GRU_CHUNK_STEPS = 300
GRU_MAX_EPOCHS = 30
M6_NEG_RATE = 0.2

# --------------------------------------------------------------------------- metrics (§8)

FAR_BUDGETS = (0.1, 0.5)              # false firings per at-risk minute
EVALUABLE_MIN_FRAC = 0.5
FROC_FAR_MAX = 0.5
ECE_BINS = 15
N_BOOT = 2000
MDE_FACTOR = 2.49
ALPHA = 0.05
# H1-H5 reject when the two-sided 95% percentile CI excludes 0, i.e. the one-sided bootstrap
# p = P(delta* <= 0) is < ALPHA / 2 (strict): the decision always matches the CI shown.
ALPHA_ONE_SIDED = ALPHA / 2
SHIFT_REPEATS = 20
SHIFT_GLOBAL_MIN = 150                # u_c ~ U{150, ..., n_c - 150} indices
SHIFT_GLOBAL_MIN_N = 400              # skip classes with n_c < 400
SHIFT_LOCAL_BLOCK = 90                # 3-min blocks
SHIFT_LOCAL_MIN_BLOCK = 30
SHIFT_LOCAL_MIN_U = 15

# --------------------------------------------------------------------------- controls (§9)

PLANT_TOKEN = "zqxw"
PLANT_LEAD_S = 8.0
PLANT_FRAC = 0.3
PLANT_AP_MIN = 0.9
P_CLOCK_DELTAS_S = tuple(round(-2.0 + 0.25 * i, 2) for i in range(17))
P_CLOCK_PEAK_RANGE_S = (-1.0, 0.25)
P_CLOCK_SHARPNESS = 0.1
CALLON_WINDOW_S = 5.0
CALLON_MATCHED = 3
CALLON_MATCH_DIST_S = (60.0, 600.0)
N_NAME_FLAG_DAP = 0.02
IH_DELTAS_S = (-20, -10, -5, -2, 0, 2, 5, 10)
IH_PLUS_DELTAS_S = (30, 60)
LABEL_SHIFTS_S = (1, 2, 3)
A_STREAM_N = 300
A_STREAM_SPAN_S = 180.0
A_JITTER_SD_S = 0.5
MISMATCH_N = 5
N_STALE_S = 600.0

# --------------------------------------------------------------------------- G2 (§2.7)
# Pre-registered pass rules, committed with the vault rule.

G2_RULES = {
    "precision": {"point_min": 0.90, "wilson_lb_min": 0.82},
    "recall": {"point_min": 0.85, "wilson_lb_min": 0.72, "n_min": 40},
    "wrong_tile_share_max": 0.01,
    "left_censored_share_max": LEFT_CENSORED_MAX_SHARE,
    "timing_median_abs_max_s": 0.10,  # should-tier
}
G2_SAMPLING = {
    "precision_per_class": 3,
    "precision_poll_cap": 40,
    "precision_vote_cap": 40,
    "recall_clip_classes": 3,
    "recall_onset_clips": 60,
    "onset_clip_span_s": 120.0,
    "audit_speedup": 8,
    "audit_render_fps": 16,
    "pixelate_blocks": 6,
    "timing_episodes": 10,
}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials ((0, 1) when n == 0)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def g2_pass(prec_k: int, prec_n: int, rec_k: int, rec_n: int, *,
            wrong_tile_share: float = 0.0, left_censored_share: float = 0.0
            ) -> tuple[bool, dict]:
    """Apply the G2 gate. Returns (pass, details) with each criterion's value and verdict."""
    r = G2_RULES
    p_hat = prec_k / prec_n if prec_n else 0.0
    q_hat = rec_k / rec_n if rec_n else 0.0
    p_lb, q_lb = wilson_interval(prec_k, prec_n)[0], wilson_interval(rec_k, rec_n)[0]
    checks = {
        "precision_point": (p_hat, p_hat >= r["precision"]["point_min"]),
        "precision_wilson_lb": (p_lb, p_lb >= r["precision"]["wilson_lb_min"]),
        "recall_point": (q_hat, q_hat >= r["recall"]["point_min"]),
        "recall_wilson_lb": (q_lb, q_lb >= r["recall"]["wilson_lb_min"]),
        "recall_n": (rec_n, rec_n >= r["recall"]["n_min"]),
        "wrong_tile_share": (wrong_tile_share, wrong_tile_share < r["wrong_tile_share_max"]),
        "left_censored_share": (left_censored_share,
                                left_censored_share <= r["left_censored_share_max"]),
    }
    return all(ok for _, ok in checks.values()), checks


# --------------------------------------------------------------------------- loader allowlists

# Event tables are built from raw class-events.json only, with this type allowlist (§4).
EVENT_TYPES = frozenset({"voice", "message", "timeline-segment"})
BLOCKED_EVENT_TYPES = frozenset({"hand-raise-key", "participation", "analytics",
                                 "bookmark-note", "outcome-assessments"})
# Keys that must never be read from any source (transcript `events`, analytics, ...).
BLOCKED_KEYS = frozenset({"events", "participation", "analytics", "user-engagement",
                          "hand-raise-key", "bookmark-note", "outcome-assessments",
                          "actual-length-seconds", "utterances", "paragraphs", "speakers",
                          "cluster-map", "clusters", "how-mix"})
# Whole sources nothing may read (post-class grading or staff analytics).
BLOCKED_SOURCES = frozenset({"outcome_assessments", "analytics", "focused_outcomes"})

# What a whitelisted path may be used for:
USE_FEATURE = "feature"  # may feed feature columns (subject to availability-time rules)
USE_TIMING = "timing"    # only places rows on the master clock (calibration, sessions)
USE_LABEL = "label"      # label machinery only (recording end, ...)
USE_META = "meta"        # class-level QA / bookkeeping; never a feature value
USE_MASK = "mask"        # roster names for in-memory masking; never written to tables
PATH_USES = (USE_FEATURE, USE_TIMING, USE_LABEL, USE_META, USE_MASK)

# Per-source JSON-path whitelists. Paths are written AFTER `scribe_core.forum_events.kebab()`
# (snake_case keys become kebab-case). Syntax: dot-separated keys; "[]" = every list element
# (a leading "[]" means the document is a list); "*" = any single dict key (e.g. block uuids).
# Polls, poll responses, breakouts, the study guide and the SLE shapes are PROVISIONAL until
# D8/D9 inspect them. Anything not listed is dropped and logged by loaders.py.
_CU = "class-users[]"
RAW_PATH_WHITELIST: dict[str, dict[str, str]] = {
    "class": {
        "id": USE_META, "title": USE_META, "type": USE_META,
        "date": USE_FEATURE, "start-time": USE_FEATURE,  # scheduled start (named on D1)
        "section.id": USE_META, "section.title": USE_META,
        "section.course.id": USE_META, "section.course.title": USE_META,
        "section.course.course-code": USE_FEATURE,
        "section.course.term": USE_FEATURE, "section.course.term.name": USE_FEATURE,
        "teacher-user-id": USE_FEATURE, "section.teacher-user-id": USE_FEATURE,
        f"{_CU}.role": USE_FEATURE, f"{_CU}.enter-class-time": USE_FEATURE,
        f"{_CU}.id": USE_FEATURE, f"{_CU}.user-id": USE_FEATURE, f"{_CU}.user.id": USE_FEATURE,
        f"{_CU}.absent": USE_META,
        f"{_CU}.first-name": USE_MASK, f"{_CU}.last-name": USE_MASK,
        f"{_CU}.preferred-name": USE_MASK, f"{_CU}.user.first-name": USE_MASK,
        f"{_CU}.user.last-name": USE_MASK, f"{_CU}.user.preferred-name": USE_MASK,
        "recording-sessions[].id": USE_TIMING,
        "recording-sessions[].recording-started": USE_FEATURE,  # rec_offset_s, clock
        "recording-sessions[].recording-ended": USE_LABEL,
    },
    "events": {
        "[].id": USE_META, "[].klass-id": USE_META, "[].private": USE_META,
        "[].event-type": USE_FEATURE, "[].actor-id": USE_FEATURE, "[].actor.id": USE_FEATURE,
        "[].timestamp": USE_FEATURE, "[].start-time": USE_FEATURE, "[].end-time": USE_FEATURE,
        "[].room-id": USE_FEATURE, "[].room-signature": USE_FEATURE,
        "[].event-data.duration": USE_FEATURE,
        "[].event-data.message-text": USE_FEATURE, "[].event-data.message": USE_FEATURE,
        "[].event-data.timeline-segment-title": USE_FEATURE,
        "[].event-data.timeline-section-title": USE_FEATURE,
        "[].event-data.timeline-segment-id": USE_META,
    },
    "polls": {
        "[].id": USE_META, "[].title": USE_FEATURE, "[].prompt": USE_FEATURE,
        "[].choices[].id": USE_META, "[].choices[].text": USE_FEATURE,
        "[].timing-text": USE_FEATURE,  # "started at 0:32:39" strings (reference: D9)
    },
    "poll_responses": {
        "[].id": USE_META, "[].poll-id": USE_FEATURE, "[].user-id": USE_FEATURE,
        "[].choice-id": USE_FEATURE, "[].answer": USE_FEATURE, "[].submitted-at": USE_FEATURE,
    },
    "breakouts": {
        "[].id": USE_META, "[].name": USE_META, "[].room-id": USE_FEATURE,
        "[].users[].id": USE_FEATURE, "[].start-time": USE_FEATURE, "[].end-time": USE_LABEL,
    },
    "listing": {
        "section-id": USE_META, "course-code": USE_FEATURE, "term": USE_FEATURE,
        "class.id": USE_META, "class.title": USE_META, "class.date": USE_FEATURE,
        "class.start-time": USE_FEATURE, "class.teacher-user-id": USE_META,
    },
    "class_edit_page": {
        "id": USE_META,
        "preparatory-assignment.description": USE_FEATURE,
        "assessment.focused-outcomes[].learning-outcome.id": USE_META,
        "assessment.focused-outcomes[].learning-outcome.name": USE_FEATURE,
        "assessment.focused-outcomes[].learning-outcome.description": USE_FEATURE,
        "study-guide": USE_FEATURE, "study-guide.description": USE_FEATURE,
    },
    "pcw_template": {
        "id": USE_META, "title": USE_FEATURE, "version": USE_META, "updated-at": USE_META,
        "blocks[].id": USE_FEATURE, "blocks[].uuid": USE_FEATURE, "blocks[].order": USE_FEATURE,
        "blocks[].block-components[].type": USE_FEATURE,
        "blocks[].block-components[].value": USE_FEATURE,
    },
    "pcw_doc": {
        "id": USE_META, "template": USE_META, "template-id": USE_META, "version": USE_META,
        "created-at": USE_META, "updated-at": USE_META, "modified-at": USE_META,
        "answers.*.type": USE_FEATURE, "answers.*.value": USE_FEATURE,
        "answers.*.files[].name": USE_FEATURE, "answers.*.files[].filename": USE_FEATURE,
    },
    "pcw_meta": {
        "template-id": USE_META, "doc-id": USE_META, "template-updated-at": USE_META,
        "doc-updated-at": USE_META, "created": USE_META, "fetched-at": USE_META,
        "history": USE_META,
    },
    "schedule": {  # paginated-classes for one section (includes unrecorded classes)
        "count": USE_META, "results[].id": USE_FEATURE, "results[].title": USE_META,
        "results[].date": USE_FEATURE, "results[].start-time": USE_FEATURE,
    },
    "me": {"id": USE_FEATURE, "role": USE_META},
    "asr": {  # *_asr.json: only words[] feeds features
        "words[].start": USE_FEATURE, "words[].end": USE_FEATURE, "words[].text": USE_FEATURE,
        "words[].punctuated-word": USE_FEATURE, "words[].conf": USE_FEATURE,
        "words[].filler": USE_FEATURE,
        "words[].cluster": USE_META, "words[].cluster-conf": USE_META,  # A-attrib only
        "provider": USE_META, "model": USE_META, "language": USE_META, "duration": USE_META,
        "diarized": USE_META, "source-res": USE_META,
    },
    "transcript": {  # Scribe *_ids.json: calibration for QA and the clock only
        "class.id": USE_META, "asr.source-res": USE_META, "asr.provider": USE_META,
        "alignment.forum-timing-precision": USE_META, "alignment.n-words": USE_META,
        "alignment.calibration.status": USE_META, "alignment.calibration.applied": USE_META,
        "alignment.calibration.method": USE_META, "alignment.calibration.score": USE_META,
        "alignment.calibration.max-residual-s": USE_META,
        "alignment.calibration.windows": USE_META,
        "alignment.calibration.offset": USE_TIMING, "alignment.calibration.drift": USE_TIMING,
        "alignment.calibration.pieces[].t-from": USE_TIMING,
        "alignment.calibration.pieces[].offset": USE_TIMING,
        "alignment.calibration.pieces[].drift": USE_TIMING,
    },
    "hand_log": {"[].t": USE_META, "[].classId": USE_META, "[].class-id": USE_META},
}
RAW_SOURCES = tuple(RAW_PATH_WHITELIST)


def _path_regex(path: str) -> re.Pattern:
    parts = []
    for seg in path.split("."):
        star_list = seg.endswith("[]")
        key = seg[:-2] if star_list else seg
        if key == "*":
            piece = r"[^.\[\]]+"
        else:
            piece = re.escape(key)
        parts.append(piece + (r"\[\]" if star_list else ""))
    return re.compile(r"\.".join(parts).replace(r"\.\[\]", r"\[\]") + r"$")


_PATH_RES = {src: [(_path_regex(p), use) for p, use in paths.items()]
             for src, paths in RAW_PATH_WHITELIST.items()}


def whitelist_use(source: str, path: str) -> str | None:
    """The allowed use of a flattened JSON path (see RAW_PATH_WHITELIST syntax), or None.

    `path` uses the same syntax with concrete keys, e.g. "answers.6f1c-uuid.value" or
    "[].event-data.duration". Blocked sources and blocked keys always return None.
    """
    if source in BLOCKED_SOURCES or source not in _PATH_RES:
        return None
    keys = {k.rstrip("[]") for k in path.split(".")}
    if keys & BLOCKED_KEYS:
        return None
    for rx, use in _PATH_RES[source]:
        if rx.match(path):
            return use
    return None


# --------------------------------------------------------------------------- feature allowlist

# Feature groups (§5.7 + amendments) as full-match regexes. features.GROUPS builds on these;
# FEATURE_ALLOWLIST is their union. Columns derived in-fold (emb_/txt_) are never stored.
FEATURE_ALLOWLIST: dict[str, str] = {
    "ctx": r"ctx_[A-Za-z0-9_]+",
    "spk": r"spk_[A-Za-z0-9_]+",
    "me": r"me_(?!hand_)[A-Za-z0-9_]+",
    "me_hand": r"me_hand_[A-Za-z0-9_]+",
    "poll": r"poll_[A-Za-z0-9_]+",
    "q": r"q_[A-Za-z0-9_]+",
    "text": r"n_words_(?:15|60)|wpm_60|filler_rate_60|frac_unk_60|topic_shift|own_sim",
    "emb": r"(?:emb|txt)_[A-Za-z0-9_]+",
    "pcwT": r"pcw_tmpl_has|pcw_res_Q|pcw_block_pos_Q|pcw_new_block|lo_sim|sg_sim",
    "pcwA": (r"pcw_has|pcw_my_words|pcw_frac_answered|pcw_n_code|pcw_res_A|pcw_res_gap"
             r"|pcw_unspent|pcw_res_ref|pcw_bm25_gap|pcw_invest|pcw_block_pos"),
    "hist": r"hist_[A-Za-z0-9_]+",   # A2: fold-aware, earlier training-fold classes only
    "edu": r"edu_[A-Za-z0-9_]+",     # A3
    "stack": r"stack_(?:m2|m4)",     # A4
    "punct": r"has_qmark",           # F-punct only
}
FEATURE_GROUP_RES = {g: re.compile(rf"(?:{rx})") for g, rx in FEATURE_ALLOWLIST.items()}
PUNCT_ONLY_COLUMNS = frozenset({"has_qmark"})
PUNCT_FEATURESETS = frozenset({"F-punct"})

# §4.4 rule 5, plus label machinery and pixel-derived columns the contract also forbids.
BLOCKED_RE = re.compile(
    r"^(?:y_|next_|excl|label_|fold|in_vault|tile_|coverage|pcw_edited|pcw_status|cal_|eps_"
    r"|av_|hand_log|participation|analytics"
    r"|in_G$|at_risk$|class_id$|t_s$|date$|emb_row$|n_others|border|ncc|frac_hand|frac_tick"
    r"|glyph"
    r"|bookmark|outcome_assess)")
FEATURE_META_COLUMNS = ("class_id", "t_s", "emb_row")  # keys in the features table, not features


def feature_group(column: str) -> str | None:
    """The feature group a column belongs to, or None if it is not allowlisted."""
    if BLOCKED_RE.match(column):
        return None
    for group, rx in FEATURE_GROUP_RES.items():
        if rx.fullmatch(column):
            return group
    return None


def feature_allowed(column: str, featureset: str | None = None) -> bool:
    """True if `column` may be a model input (in `featureset`, for F-punct-only columns)."""
    group = feature_group(column)
    if group is None:
        return False
    if column in PUNCT_ONLY_COLUMNS:
        return featureset in PUNCT_FEATURESETS
    return True
