# On Cue: can a model hear my turn coming?

**CS156 · First Pipeline · Brady Rutherford · Fall 2026**

> **TEMPLATE.** This is the write-up skeleton, in the order the assignment's *Suggested Format*
> asks for. Every `TODO` block says what goes there, how long it should be, and which run fills it.
> **No result appears in this file until the run produces it.** Design: [`PLAN.md`](PLAN.md)
> (summary), [`design/on_cue_spec.md`](design/on_cue_spec.md) and
> [`design/amendments.md`](design/amendments.md) (authoritative). Delete this box when done.

**Links:** [run notebook](pipeline1.ipynb) · [plan](PLAN.md) · [presentation page](site/index.html) ·
[results bundle](results/) · harvester and transcriber: `Minervas_scribe` v3 at SHA `TODO`

---

## 1. The idea

> **TODO: 2 paragraphs + 1 image.** For a reader who has never seen Forum. Beats to hit:
> - Forum records every class as a screen capture and logs every time anyone speaks, but no
>   student API exposes hand raises: pressing `H` PATCHes a write-only endpoint. So my hand raises
>   are recovered from the recording, by a colour-and-shape test on the icon that appears on my
>   tile. No face recognition.
> - The question: using only what I could have heard, seen in Forum, or written before class up to
>   second *t*, will my hand go up in the next 15 s, at least *g* seconds ahead? Am I cued by *what
>   is said and what I prepared*, or by *when* it is (minute, activity, my own rhythm)?
> - One sentence on the answer, with its CI, once the run exists.
> - **Image:** `figures/hero_timeline.svg`, a **schematic** one-class timeline drawn from synthetic
>   data and labelled as such: role lanes, question ticks, p(t), firings, bids, segment shading.
>   No text, no real class.

## 2. Pipeline

> **TODO: exactly 1 image.** Render `figures/pipeline.svg` with
> `npx -y @mermaid-js/mermaid-cli -i pipeline.mmd -o figures/pipeline.svg` (matplotlib fallback).
> Stages: Forum API (raw events, whitelisted paths) + PCW → Scribe v3 (ASR, calibration) ∥
> same-rendition signed mp4 → glyph tracking + hand detector → **one media clock** (highlighted)
> → labels and at-risk grid → CausalView / features → models → grouped CV → event metric. One
> sentence under it.

![Pipeline](figures/pipeline.svg)

## 3. Data: what it is and how it was ingested

> **TODO: the largest section, about 1–1.5 pages + the funnel table.** Sub-headings:
> 1. **Sources.** The endpoints table from PLAN §2 (my session, cookie auth), with the JSON-path
>    whitelists: event tables come only from raw `class-events.json` with types
>    {voice, message, timeline-segment}; post-class grading, analytics and Scribe transcript
>    `events` are blocked.
> 2. **Timestamp traps.** Voice `timestamp` is the *end*; kebab vs snake keys; whole-second 2023
>    timestamps; `NOREC`; Scribe's segment `end` and gap clamping; the global-fallback calibration
>    that reports a residual of 0.
> 3. **One clock.** Calibration QA (windowed, ≥ 6 windows, residual p95 ≤ 1.5 s); the ASR rendition
>    and pts-based frame times; per-class A/V lag.
> 4. **Words.** Forum-causal speaker roles; word-bearing voice events; causal frames and `close_s`;
>    name masking (D11 precision and recall) and normalisation (no punctuation or casing; CALLON
>    phrases and `[NAME]` deleted).
> 5. **PCW.** Harvest through sle-collaboration; `pcw_status` and the branch D8 chose.
> 6. **The label.** Strip auto-detection; glyph NCC with a leave-one-class-out template; HSV +
>    shape test; per-class baseline; hysteresis and the causal live state; re-timing; vote typing;
>    the G2 audit with Wilson bounds.
> 7. **Causal contract**, condensed: the availability table and the allowed/forbidden examples.

| Stage | Classes | Bids | Points |
|---|---:|---:|---:|
| Recorded sessions (API) | 694 | | |
| Population (sections averaging ≥ 5 of my main-room voice events per class) | TODO | | |
| Selected (2 sections per semester + CS156 F26) | TODO | | |
| Transcribed (provider per course) | TODO | | |
| Calibrated (windowed, p95 ≤ 1.5 s) | TODO | | |
| A/V lag within tolerance | TODO | | |
| Scanned; tile located and validated | TODO | TODO | |
| QA pass | TODO | TODO | TODO |
| Dev / vault | TODO / TODO | TODO / TODO | TODO / TODO |

> PCW coverage by status and course: TODO (from `results/funnel.csv`).

## 4. Code

> **TODO: 4–6 lines.** Repo link; `uv sync` (Scribe from a public git SHA, never a local path);
> `.env` with `FORUM_COOKIE` and `ONCUE_ID_SALT` (never committed); the CLI stages
> (`python -m oncue harvest … evaluate`); `pipeline1.ipynb` is committed **with outputs** and its
> metric, figure and table cells re-run from the committed `results/` bundle with every `RERUN_*`
> flag set to `False`. Say why no raw data is committed (classmates' speech and names).

## 5. Models and methods not covered in class

> **TODO: about 1 paragraph per item, each tied to the run that used it.** Delete any item whose
> run did not happen (M5, M6, BM25).

**Discrete-time hazard framing.** Each 2-second at-risk point is a trial:
$p(t)=P\big(y_{H,g}(t)=1\mid x_{\le t},\ t\in\mathcal R\big)$, a landmark classifier per horizon *H*.

**Gradient boosting (M3).** $F_m=F_{m-1}+\eta f_m$, with leaf weights

$$
w_j^*=-\frac{\sum_{i\in j}g_i}{\sum_{i\in j}h_i+\lambda},\qquad g_i=p_i-y_i,\quad h_i=p_i(1-p_i).
$$

**Transformer sentence encoder** (bge-small in M3 via PCA, and in M4):
$\text{Attn}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$, multi-head, with residual connections
and LayerNorm; CLS pooling $e(x)=h^{(L)}_{\text{CLS}}/\lVert h^{(L)}_{\text{CLS}}\rVert$; and causal
recency pooling over closed frames

$$
v_\tau(t)=\frac{\sum_{f\in F(t)} e^{-(t-\text{close}_f)/\tau}\,e_f}{\big\lVert\sum_{f\in F(t)} e^{-(t-\text{close}_f)/\tau}\,e_f\big\rVert},\qquad \tau\in\{15,60\}\text{ s}.
$$

**Label model (always runs).** Normalised cross-correlation for tile tracking,

$$
\text{NCC}=\frac{\sum_u(P_u-\bar P)(T_u-\bar T)}{\sqrt{\sum_u(P_u-\bar P)^2\sum_u(T_u-\bar T)^2}},
$$

the icon fraction in the corner ROI *R* (OpenCV HSV),

$$
\text{frac}_{\text{hand}}(f)=\frac{1}{|R|}\sum_{p\in R}\mathbb 1\big[H_p\in[26,40]\wedge S_p>115\wedge V_p>115\big],
$$

a shape test (circularity $4\pi A/P^2\ge0.6$, white-glyph share 0.08–0.45), and a hysteresis state
machine: raw state UP iff $\text{frac}'\ge\theta_{\text{on}}\wedge\text{shape}$, DOWN iff
$\text{frac}'<\theta_{\text{off}}$, else the previous state, confirmed after *k* = 2 agreeing samples.

**Platt scaling.** $p=\sigma(a\,s+b)$, fit by maximum likelihood on grouped inner out-of-fold scores.

**BM25** (if `pcw_bm25_gap` survives):
$w(q,d)=\text{IDF}(q)\,\frac{f(q,d)(k_1+1)}{f(q,d)+k_1(1-b+b\,|d|/\text{avgdl})}$.

**GRU** (if M5 ran): $z_t=\sigma(W_zx_t+U_zh_{t-1})$, $r_t=\sigma(W_rx_t+U_rh_{t-1})$,
$\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1}))$, $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$.

> **TODO** if M6 ran: the cross-encoder and its prior correction $\text{logit}\mathrel{+}=\log r$.

## 6. Novel metrics and features

> **TODO: about 1 paragraph per item, plus why frame-level AP does not answer "would it have warned
> me in time?".**

**Event recall at a false-firing budget** (seizure-prediction convention). Firings on the causal
grid $\mathcal G$ with refractory period $H-g$:

$$
\mathcal F_\tau=\Big\{s\in\mathcal G:\ p(s)\ge\tau\ \wedge\ \nexists\, s'\in\mathcal F_\tau\cap\big(s-(H-g),\,s\big)\Big\}
$$

$$
\text{FAR}(\tau)=\frac{\#\{s\in\mathcal F_\tau: y(s)=0\}}{|\mathcal G_{\text{def}}|\,\Delta/60},\qquad
\text{Recall}(\tau)=\frac1K\sum_k \mathbb 1\big[\mathcal F_\tau\cap[o_k-H,\ o_k-g)\neq\emptyset\big]
$$

$$
\text{lead}_k=o_k-\min\big(\mathcal F_\tau\cap[o_k-H,\ o_k-g)\big),\qquad
R_0(\lambda_f)=\min\big(1,\ \lambda_f\,(H-g)/60\big),\qquad
\text{EventLift}=\frac{\text{Recall}(\tau_\lambda)}{\text{Recall}_{\text{shift}}(\text{FAR}_{\text{realised}})}
$$

with time in warning reported next to every recall.

**Timing gains.** $\text{TG}=\text{AP}-\text{AP}_{\text{shift}}$ (whole-class circular shift: "which
class") and $\text{TG}_{\text{local}}=\text{AP}-\text{AP}_{\text{local}}$ (shifts of ≥ 30 s inside
3-minute same-segment blocks: "which moment").

**Within-class AP.** $\text{AP}_w=\sum_c K_c\,\text{AP}_c/\sum_c K_c$, weighted by bids per class.

**Unspent preparation and the resonance gap** (z-scored against an earlier-only null; labelled
underpowered if the PCW coverage rule fails):

$$
U(t)=\max_{j\in\text{unspent}(t)} z(x,a_j),\quad \text{unspent}(t)=\Big\{j:\ \max_{m\in M_{\le t}} z(a_j,m)<\kappa\Big\},\qquad
\Delta\rho=z\big(x,e(A_{b^*})\big)-z\big(x,e(Q_{b^*})\big)
$$

## 7. How the models learned

> **TODO: 2 figures + 1 paragraph.** Must: (a) HGB staged log-loss, training vs inner validation,
> per outer fold, with n\* marked; (b) LR training vs validation AP across C. Should: AP vs number
> of training classes (M2, M3, M4); M5 loss and AP per epoch. Say where each starts to overfit.

## 8. Final performance

> **TODO: 2–3 figures + 1 paragraph.** Pooled OOF precision–recall curves (B1, R, M1–M4) with π
> marked; the per-class AP_w distribution; the FROC curve with always-on, minute-only, globally and
> locally shifted, matched-rate random and R references, lead and time in warning annotated at
> 0.1/min; reliability diagram (should); vault vs CV. Brier with the Brier skill score.

## 9. Comparison conditions

> **TODO: 1 table + figures + 1 paragraph per contrast.** Rows are the PLAN §7 conditions that ran.
> Every number carries a 95% cluster-bootstrap CI and the row's `n_classes` and `n_onsets`.

| Condition | AP_w [CI] | Pooled AP (cal.) | Fold AP ± sd | Lift | TG | TG_local | ΔAP_w vs F-struct [CI] | MDE | Recall@0.1 (FAR, TiW) | EventLift | Lead (s) | Classes | Bids |
|---|---|---|---|---|---|---|---|---|---|---|---|---:|---:|
| B0 prevalence | | | | | | | | | | | | | |
| R question rule | | | | | | | | | | | | | |
| B1 structure LR | | | | | | | | | | | | | |
| M1 naive Bayes | | | | | | | | | | | | | |
| M2 TF-IDF LR | | | | | | | | | | | | | |
| M3 / F-struct | | | | | | | | | | | | | |
| **M3 / F-all** | | | | | | | | | | | | | |
| M3 / F-all−me_hand | | | | | | | | | | | | | |
| M3 / F-all+hist | | | | | | | | | | | | | |
| M3 / F-all+edu | | | | | | | | | | | | | |
| M3+stack | | | | | | | | | | | | | |
| M4 embeddings LR | | | | | | | | | | | | | |
| S-rand (leakage tax) | | | | | | | | | | | | | |
| g = 1 / 3 / 5 | | | | | | | | | | | | | |
| Vault (M3 / F-all) | | | | | | | | | | | | | |

> Figures: forest plot of H1–H5 with fixed-sequence / Holm-adjusted p; S-rand tax bars; g and H
> sweep lines; information-horizon curve with the illegal side shaded; A-mismatch (mean ± sd),
> N-name, CALLON and P-clock panels; per-course F-text vs F-struct gap.
> Paragraphs: H1 (content vs structure, with and without my own hand history); H2 (the moment vs
> the phase); H3 (preparation); H4 (learned vs the rule); H5 (embeddings vs TF-IDF); the guard
> dial; the leakage tax.

## 10. Discussion

> **TODO: 3–4 paragraphs.** Content vs structure vs preparation, within class; what the rule R
> explains; the class vs the phase vs the moment (TG vs TG_local); what the controls rule out and
> what label lateness the CALLON check and live logger bound; how much batch ASR inflated results
> (A-stream, F-punct); failure cases by segment type, course and semester; label validity (a raise
> is not a call; votes and contested episodes; recall in NA spans; per-class precision); n is bids,
> not points; the population limit ("the courses where I talk most"); links to the turn-taking
> literature. If H1 is null, say so plainly and give the MDE.

## 11. Privacy and ethics

> **TODO: 5–8 lines.** Transcripts in ids mode; names masked in memory and deleted from feature
> text; aggregate-only figures (n-grams need ≥ 20 classes); my own tile only in crops; no face
> recognition; the full-strip audit clips and name-glyph bits kept local and deleted Mon 19 Oct;
> HMAC class ids in `results/`; the privacy scan; consent scope (which courses; third-party ASR per
> course, Deepgram with `mip_opt_out`; local ASR by default).

## 12. References

> **TODO.** Keep only what the write-up cites. Starting list (spec §14 and the research report):
> Sacks, Schegloff & Jefferson 1974; Skantze 2021; Ekstedt & Skantze 2020; Rowe 1986; Böheim et al.
> 2020; Singer & Willett 1993; Truccolo et al. 2005; van Houwelingen 2007; Friedman 2001; Ke et al.
> 2017; Vaswani et al. 2017; Reimers & Gurevych 2019; Xiao et al. 2023; Wang et al. 2020; Cho et al.
> 2014; Shi et al. 2021; Bernado et al. 2026; Winterhalder et al. 2003; Mormann et al. 2007;
> Chakraborty 1989; Saito & Rehmsmeier 2015; Platt 1999; Guo et al. 2017; Robertson & Zaragoza 2009;
> Lewis 1995; Efron & Tibshirani 1993; Holm 1979; Berger 1982; Wilson 1927; Roberts et al. 2017;
> Kapoor & Narayanan 2023; Pedregosa et al. 2011; Deepgram Nova-3 docs; Minerva's Scribe v3 (SHA);
> `research_notes/`.

## AI use

> **TODO.** Following the repo convention: what Claude was used for (planning and design critique,
> the Forum API and frontend investigation, the model-options research, code and tests), with
> links to the conversation logs saved next to this file. No generative model produced labels,
> features, predictions or analysis text, and no classmate's words were sent to an LLM.
