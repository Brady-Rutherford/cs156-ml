# Turn-taking, next-speaker and intention-to-speak prediction: what transfers to a text + timing hand-raise predictor

Scope note: researched 12 Oct 2026 for a CS156 pipeline that predicts "hand-raise icon appears within the next H s" (H = 1–20 s) for one student in Minerva Forum, from a causal speaker-tagged ASR transcript window (~60 s, word timestamps) plus class timing/state features. No audio prosody, gaze or face features. Papers that I could only see through search-engine summaries (not the full text) are marked "(snippet only)".

## 1. TurnGPT (Ekstedt & Skantze 2020): what it predicts, how, results, code, and can it be adapted to one listener's entry?

### Takeaway
TurnGPT is a fine-tuned GPT-2 that adds turn-shift/speaker tokens to lowercased, ASR-style text and reads P(turn-shift token next) after each word as a "transition-relevance place" (TRP) probability. It beat LSTM and POS baselines, mainly because it uses the dialogue context. The code is MIT-licensed, but no trained checkpoints are released and the repo is pinned to CUDA. The idea (special event tokens inside an LM) can be adapted, but the 4-day route is to borrow the *concept* as a feature, not to retrain TurnGPT.

### Cited Findings
- Citation: Erik Ekstedt & Gabriel Skantze, "TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog," Findings of ACL: EMNLP 2020. The model predicts turn-shifts and can also *project* when turns will complete — [arXiv 2010.10874](https://arxiv.org/abs/2010.10874)
- Architecture: "a finetuned GPT-2 transformer imbued with special turn-shift tokens". Two backbones were tried, GPT-2-base and DialoGPT-small, and they performed almost identically. Speaker tokens are included in the language-modelling task, and the TRP probability is the maximum speaker-token output probability at each step. Words were lowercased and punctuation was removed to mimic ASR output — [TurnGPT paper PDF](https://arxiv.org/pdf/2010.10874)
- Data: eight datasets in three groups. "Assistant" = Taskmaster, MetaLWoZ, MultiWoZ 2.1, CCPE. "Written social" = PersonaChat, DailyDialog. "Spontaneous spoken" = Map Task (128 dialogs), Switchboard (2.4K dialogs). All were also combined into a "Full" set. Backchannels such as "mm" and "mhm" were removed from the spoken corpora — [TurnGPT paper PDF](https://arxiv.org/pdf/2010.10874)
- Metric: balanced accuracy (bAcc) of turn-shift vs non-shift over all word positions, chosen because the classes are heavily imbalanced ("many more word tokens than turn-shifts") — [TurnGPT paper PDF](https://arxiv.org/pdf/2010.10874)
- Results, trained on the Full set (bAcc on Assistant / Spoken / Written test sets): TurnGPT 0.913 / 0.823 / 0.905; LSTM baseline 0.869 / 0.748 / 0.83; POS-tag baseline 0.750 / 0.675 / 0.732. The spontaneous spoken data is the hardest — [TurnGPT paper PDF](https://arxiv.org/pdf/2010.10874)
- Context ablation: TurnGPT gets better as more preceding turns are given (0 → 4). "The biggest drop in performance happens when going from some context to no context." The LSTM shows the same effect, but less strongly — [TurnGPT paper PDF](https://arxiv.org/pdf/2010.10874)
- Projection: the authors sampled futures autoregressively and counted tokens until a generated speaker token. They presented this as a feasibility demo on one example (1000 samples) and left it to future work. It is not an evaluated method — [TurnGPT paper PDF](https://arxiv.org/pdf/2010.10874)
- Text only: prosody, gaze and timing are explicitly not modelled, and the authors call integrating prosody "an important topic for future work" — [TurnGPT paper PDF](https://arxiv.org/pdf/2010.10874)
- Repo: MIT licence. It fine-tunes GPT-2 or DialoGPT via `TurnGPT(pretrained_model_name_or_path=...)`, uses PyTorch Lightning, and depends on a separate `datasets_turntaking` repo and CUDA 11.3. The README does not distribute trained TurnGPT checkpoints and has no Hugging Face link. A later version added a projection head that predicts turn-shifts in the next N tokens — [GitHub ErikEkstedt/TurnGPT](https://github.com/ErikEkstedt/TurnGPT)
- Later work: LLMs are poor at *within-turn* TRPs (places where a listener could come in mid-turn), as opposed to turn-final ones. This was tested on a new participant-labelled dataset of unscripted conversation. Citation: Umair, Sarathy & de Ruiter, "Large Language Models Know What To Say But Not When To Speak," Findings of EMNLP 2024 — [arXiv 2410.16044](https://arxiv.org/abs/2410.16044)

### Inferences
- **What TurnGPT actually predicts:** "Is the *current speaker's* turn complete after this word?" It does not predict *which listener* will come in, or when a specific listener will bid for the floor. A Forum hand-raise is a bid for the floor that often happens *while someone else is still talking* (students queue). So TurnGPT's target is at best a proxy feature, not the target itself.
- **Direct adaptation (feasible, but risky in 4 days):** insert a `<raise>` token into the word stream at the timestamp of each hand-raise. Fine-tune a small causal LM (GPT-2 124M / distilGPT2) so that P(`<raise>` within the next k tokens) becomes the score. Problems:
  - The target is in *seconds*, not tokens. Silences produce no tokens, so you would need time-tick tokens such as `<pause_2s>`.
  - The LM sees ~1–5% positive events, and fine-tuning on CPU/MPS for millions of tokens is slow.
  - The result is hard to explain in a CS156 write-up.
- **Cheap borrow (recommended):** use an off-the-shelf small LM zero-shot to compute "end-of-utterance-ness" features on the professor's latest words, e.g. P(sentence end / "?" / newline) under GPT-2, or a binary "utterance ends in a question" flag. Feed these, with timing features, into logistic regression or gradient-boosted trees. This keeps TurnGPT's core insight (lexical/pragmatic completeness signals a TRP) without training a transformer.
- **Two transferable lessons:**
  - Context matters most when going from zero to some context, which argues for a window of a few turns (the planned 60 s window is reasonable).
  - Report balanced accuracy or PR-AUC rather than accuracy, because the class imbalance is similar to the student's.

### Gaps
- I found no public TurnGPT checkpoint on Hugging Face. Hugging Face itself was not searched directly; the repo README shows none.
- The repo's last-commit date and whether it still runs with current PyTorch/MPS on Apple silicon were not verified. CUDA 11.3 pinning suggests it needs modification to run on an M4.
- No study was found that adapts TurnGPT to predict a specific listener's entry or a hand-raise.

## 2. Voice Activity Projection (VAP, 2022) and successors: what they model, inputs, and any text-only / text+timing variants

### Takeaway
VAP is a self-supervised, *audio-only* (stereo, one channel per speaker) transformer that, at every frame, predicts the near-future voice activity of both speakers over a 2 s window. Shift/hold/backchannel predictions are read off zero-shot. Its framing of "predict future activity over a window, labelled automatically from the future" maps closely onto "hand up within H s". But every public checkpoint needs audio, and the successors (multilingual, real-time, triadic, multimodal) add audio or video, not text-only. The text+audio fusions exist (Lla-VAP, Wang et al. 2024), but I found no text+timing-only VAP.

### Cited Findings
- Citation: Ekstedt & Skantze, "Voice Activity Projection: Self-supervised Learning of Turn-taking Events," INTERSPEECH 2022. VAP is "a general, self-supervised objective" that models "the dynamics of voice activity of the interlocutors". Four zero-shot tasks target upcoming turn-shifts and backchannels — [arXiv 2205.09812](https://arxiv.org/abs/2205.09812)
- Repo details:
  - Input: stereo waveform (mono plus a silent second channel is supported).
  - Model: CPC audio encoder, then a GPT-like transformer with ALiBi attention, at 50 Hz, on 20 s chunks, with a 2 s projection window and a multitask VAP + VAD loss.
  - A state-dict checkpoint is included in `example/`, under the MIT licence.
  - Training needs private corpora (Switchboard/Fisher/Candor), so the public repo is inference-focused.
  - Lists "How Much Does Prosody Help Turn-taking?" (SIGDIAL 2022 best paper).
  - [GitHub ErikEkstedt/VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection)
- Earlier continuous models (direct ancestors of VAP):
  - Skantze, "Towards a General, Continuous Model of Turn-taking in Spoken Dialogue using LSTM Recurrent Neural Networks," SIGDIAL 2017. The LSTM predicts "upcoming speech activity in a future time window" and beats human observers at predicting turn-shifts at pauses — [ACL Anthology W17-5527](https://aclanthology.org/W17-5527/)
  - Roddy, Skantze & Harte, "Investigating Speech Features for Continuous Turn-Taking Prediction Using LSTMs," INTERSPEECH 2018. Predicts speech activity in each 50 ms frame of a 3 s future window (60 frames) on Map Task (128 dialogues, 96 train / 32 test). Word features were delayed 100 ms "to simulate the conditions of a real-time system" (ASR latency) — [ar5iv 1806.11461](https://ar5iv.labs.arxiv.org/html/1806.11461); [ISCA PDF](https://www.isca-archive.org/interspeech_2018/roddy18_interspeech.pdf)
- **Closest thing to text+timing evidence:** in Roddy et al. 2018, *word features alone* (with timing) reached F ≈ 0.71 on prediction-at-pauses (50 ms), against 0.77 for acoustic features and 0.78 for acoustic+words. POS tags underperformed words overall but did well for onset prediction — [ar5iv 1806.11461](https://ar5iv.labs.arxiv.org/html/1806.11461)
- **Multilingual and real-time VAP:**
  - Inoue et al., "Multilingual Turn-taking Prediction Using Voice Activity Projection," LREC-COLING 2024. Monolingual models transfer poorly across languages; a joint English/Mandarin/Japanese model matches the monolingual ones — [ACL Anthology 2024.lrec-main.1036](https://aclanthology.org/2024.lrec-main.1036/); [arXiv 2403.06487](https://arxiv.org/abs/2403.06487)
  - VAP-Realtime (Inoue, Kyoto University): MIT-licensed code, but "trained models restricted to academic purposes only". Stereo 16 kHz input, 5–20 Hz model variants (English, Japanese, multilingual, noise-robust, backchannel and nodding models). Runs in real time on CPU. Being folded into MaAI — [GitHub inokoj/VAP-Realtime](https://github.com/inokoj/VAP-Realtime); [GitHub maai-kyoto/MaAI](https://github.com/maai-kyoto/maai); [IWSDS 2024 paper](http://sap.ist.i.kyoto-u.ac.jp/EN/bib/intl/INO-IWSDS24a.pdf)
- **Multi-party and multimodal VAP successors (all need audio, some video):**
  - Elmers, Inoue, Lala & Kawahara, "Triadic Multi-party Voice Activity Projection for Turn-taking in Spoken Dialogue Systems," INTERSPEECH 2025 — [ISCA PDF](https://www.isca-archive.org/interspeech_2025/elmers25_interspeech.pdf); [arXiv 2507.07518](https://arxiv.org/abs/2507.07518)
  - "MuVAP: Multimodal Multiparty Voice Activity Projection for Turn-taking Prediction in the Wild" (2026) — [arXiv 2606.16731](https://arxiv.org/abs/2606.16731) (snippet only)
  - "Voice Activity Projection Model with Multimodal Encoders" (2025) — [arXiv 2506.03980](https://arxiv.org/abs/2506.03980) (snippet only)
  - "Exploring Multimodal Turn-Taking Cues in Face-to-Face Conversation using Voice Activity Projection" (2026) — [arXiv 2609.14666](https://arxiv.org/abs/2609.14666) (snippet only)
- **Text + audio fusion:**
  - "Lla-VAP: LSTM Ensemble of Llama and VAP for Turn-Taking Prediction" (arXiv 2412.18061, 2024) ensembles an LLM with VAP. It reports that end-of-turn prediction is much easier than predicting pauses within a turn, and within-turn prediction "remain[s] a significant hurdle". The same source summarises Wang et al. 2024: fusing a neural acoustic model with an LLM consistently beat single-modality baselines — [arXiv 2412.18061](https://arxiv.org/abs/2412.18061) (snippet only)
  - "Prompt-Guided Turn-Taking Prediction," SIGDIAL 2025 — [ACL Anthology PDF](https://aclanthology.org/2025.sigdial-1.9.pdf) (title only; not read)

### Inferences
- **VAP's real contribution here is the *problem framing*:** at every frame t, predict activity in a future window, with labels generated automatically from the future. The student's target ("hand up within H s, labels from future icon timestamps") already has exactly this structure.
- **Borrow the multi-horizon idea:** instead of training separate classifiers for H = 1, 2, 5, 10, 20 s, train one model with several outputs, or a discrete-time hazard model over 2 s bins. Monotone cumulative probabilities then come for free and stay consistent across H.
- From memory of the VAP paper (not re-verified this session): VAP splits the 2 s window into four bins per speaker and predicts one of 256 joint states. A coarse-bin target like this is a reasonable design for the H sweep.
- **Do not plan on running VAP:**
  - The pipeline has no per-speaker audio channels.
  - Forum class audio has more than 2 parties.
  - The VAP-Realtime checkpoints are academic-use only.
  - Fine-tuning needs licensed corpora.
- **Causality:** VAP/LSTM *outputs* at time t are causal. The shift/hold *evaluation events* are defined with future voice activity, which is fine for labels and evaluation but must not leak into features.
- Roddy et al.'s 100 ms word delay is a good precedent: in the student's pipeline, only use words whose *end* timestamp (plus an assumed ASR finalization latency) is ≤ t.

### Gaps
- I found no published VAP variant whose input is only a transcript plus word timing (no audio). Lla-VAP and Wang et al. still use audio.
- Exact numbers for Lla-VAP and Wang et al. (ICASSP 2024) were not retrieved; the claims above are second-hand via Lla-VAP's summary.
- The licence of the MaAI repo was not checked.

## 3. Multi-party turn-taking and next-speaker prediction (gaze-based classics; 2023–2026 work; text- or timing-only)

### Takeaway
Classic multi-party next-speaker work (Ishii, Otsuka et al.) relies on gaze and respiration, which this pipeline lacks. Recent 2024–2026 work shows that *text-only* LLMs given recent transcript context can predict the next speaker in meetings above chance, and sometimes above humans. But trivial heuristics such as "the same speaker continues" are surprisingly strong, and zero-shot LLMs barely beat them. Fine-tuning is what helps. Most of these setups assume the current utterance has *ended*, a mild form of future information compared with a fixed 2 s decision grid.

### Cited Findings
- **Gaze and respiration (Ishii, Otsuka and colleagues):**
  - "Predicting next speaker and timing from gaze transition patterns in multi-party meetings," ICMI 2013. A probabilistic model over all participants' gaze-transition patterns beat models without gaze — [Semantic Scholar](https://www.semanticscholar.org/paper/Predicting-next-speaker-and-timing-from-gaze-in-Ishii-Otsuka/1cdece39829534425fe16ac36693729857237ad6)
  - Follow-ups: "Prediction of Who Will Be the Next Speaker and When Using Gaze Behavior in Multiparty Meetings" (ACM TiiS) — [ACM DL 10.1145/2757284](https://dl.acm.org/doi/abs/10.1145/2757284); "Using Respiration to Predict Who Will Speak Next and When in Multiparty Meetings" (ACM TiiS) — [ACM DL 10.1145/2946838](https://dl.acm.org/doi/10.1145/2946838)
  - "Can Prediction of Turn-management Willingness Improve Turn-changing Modeling?" (Ishii et al., IVA 2020) — [ACM DL 10.1145/3383652.3423907](https://dl.acm.org/doi/10.1145/3383652.3423907) (paywalled, abstract not retrieved)
- **Text-only LLMs on AMI (Fukuda, Arora, Delcroix, Tawara, Ogawa, Chiba, Ando, Chen & Watanabe, "Evaluating Large Language Models Abilities for Addressee, Turn-change, and Next Speaker Prediction in Meetings," arXiv 2606.17542, 2026):**
  - Data: AMI, 10 sessions, 4,051 utterances, 4-speaker meetings.
  - Next-speaker F1: text-only Qwen3-14B 69.4%, humans 60.1%, Gemini 2.5 Pro multimodal 57.7%, majority baseline 25.0%.
  - Turn-change F1: Qwen3-14B 66.4%, humans 71.5%, majority baseline 63.7%.
  - Context: next-speaker prediction keeps improving up to about 40 utterances of history, while turn-change peaks at about 5 utterances.
  - Explicit gaze labels helped; raw audio/video added little over text.
  - [arXiv 2606.17542](https://arxiv.org/html/2606.17542v1)
- **Hilgert & Niehues, "Next Speaker Prediction for Multi-Speaker Dialogue with Large Language Models," ICNLSP 2025, pp. 60–71:**
  - Setup: next-speaker prediction as text completion over the recent utterances, on AMI, DiPCo (dinner party) and MultiLIGHT. Models: Llama 3.2 3B, 3.1 8B and 3.3 70B zero-shot; 3B and 8B also fine-tuned with LoRA.
  - Accuracy on the full AMI test set: "never switch" baseline (last speaker continues) 33.41%; zero-shot 8B 34.88%; zero-shot 70B 35.81%; fine-tuned 8B 47.85%. On the human-annotated AMI sample, humans scored 30.88%.
  - Fine-tuning helped even on unseen DiPCo (38.48% vs 30.94% zero-shot for 8B).
  - [ACL Anthology 2025.icnlsp-1.7](https://aclanthology.org/2025.icnlsp-1.7/); [PDF](https://aclanthology.org/2025.icnlsp-1.7.pdf)
- **Bhagtani, Anand, Xu & Yadav, "Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue," arXiv 2603.11409, 2026 (submitted to Interspeech 2026):**
  - Task: at detected pauses, decide from text context whether an AI participant should speak, across 120K+ labelled conversations from three multi-party corpora.
  - Zero-shot LLMs (eight tested) "failed consistently"; supervised fine-tuning with reasoning traces improved balanced accuracy by up to 23 points.
  - [arXiv 2603.11409](https://arxiv.org/abs/2603.11409)
- Lee, Li & Deng, "A Computational Study on Sentence-based Next Speaker Prediction in Multiparty Conversations," IVA 2024. Sentence-based segmentation beat inter-pausal-unit (IPU) based models; best accuracy 61.39% — [DOI 10.1145/3652988.3673915](https://doi.org/10.1145/3652988.3673915) (snippet only)
- **Other recent multi-party work (audio and/or gaze):**
  - "Online Multimodal End-of-Turn Prediction for Three-party Conversations" (Lee & Deng, ICMI 2024) — [ACM DL](https://dl.acm.org/doi/10.1145/3678957.3685742)
  - "Gaze-Enhanced Multimodal Turn-Taking Prediction in Triadic Conversations" (2025) — [arXiv 2505.13688](https://arxiv.org/html/2505.13688v1)
  - "Predicting Turn-Taking Outcomes in Multi-Party Conversation: Interpretable Modelling of Speech and Gaze Dynamics with Interpersonal Closeness" (2026) — [arXiv 2608.27988](https://arxiv.org/html/2608.27988)
  - "Predicting and Understanding Turn-Taking Behavior in Open-Ended Group Activities in Virtual Reality" (2024) — [arXiv 2407.02896](https://arxiv.org/pdf/2407.02896)
  - (all snippet only)
- **Survey:** "A Survey of Recent Advances on Turn-taking Modeling in Spoken Dialogue Systems," IWSDS 2025 — [ACL Anthology PDF](https://aclanthology.org/2025.iwsds-1.27.pdf) (not read)

### Inferences
- **Baselines:** the Hilgert result (zero-shot Llama 8B ≈ the "last speaker continues" heuristic on AMI) is a strong warning to build naive baselines first. For the hand-raise task these are:
  - the base rate;
  - "hazard as a function of time since his last raise";
  - "the professor just asked a question → raise more likely".
  
  Any learned text model must beat these.
- **What transfers from next-speaker work:** text context *does* carry who-speaks-next signal, but the gains come from *supervised* fine-tuning, and large LLMs are impractical on an M4 in 4 days. A practical substitute: frozen sentence embeddings of the last N seconds (optionally only the professor's last utterance), reduced with PCA, fed to a GBM or logistic regression next to timing features.
- **Causality:** next-speaker papers predict at utterance boundaries, using the complete current utterance, and AMI work often uses ground-truth transcripts. On a 2 s grid during ongoing speech, the student has only a *partial* current utterance. Compute features over finalized words only, and expect lower performance than these papers report.
- Fukuda et al.'s finding that turn-change is best with short context (about 5 utterances) while next-speaker identity benefits from long context suggests two kinds of features:
  - short-window features for "is a floor opportunity coming";
  - longer, class-level features for "is it *him*" (topic relevance to his pre-class answers, his raises/talk time so far).

### Gaps
- Fukuda et al.'s supervised baselines (NB, RF, MLP, SVM) and their feature sets were not extracted, and code availability was not stated.
- Lee, Li & Deng 2024 input modalities were not confirmed (snippet only).
- I found no study that predicts a *specific single participant's* entry, or a non-verbal floor bid like a hand-raise, from transcript text and timing alone. The closest are the next-speaker and speak-or-stay-silent tasks above.

## 4. Predicting student participation, willingness/intention to speak, or hand-raising (any modality): what features mattered

### Takeaway
Education research treats hand-raising as a *stable individual trait* (about half its variance is between students), triggered mainly by teacher questions and shaped by wait time, self-concept and shyness. I found no ML work that forecasts the timing of hand-raises from discourse; the classroom ML work *detects* raised hands in video. "Intention to speak" work (wearables, VR) shows intentions often go unrealized and precede speech by only about 1 s. For the student this means: per-student and per-class participation history and "the professor just asked a question" are the literature-backed features. Minerva's own Talk Time tooling may shape who gets called on.

### Cited Findings
- **Hand-raising as engagement:** Böheim, Knogler, Kosel & Seidel, "Student hand-raising as an indicator of behavioral engagement and its role in classroom learning," Contemporary Educational Psychology, 2020:
  - hand-raising is related to cognitive involvement and to perceived teacher support, and may mediate between teacher support and performance;
  - hand-raising is "commonly prompted by a specific learning task (i.e., teacher question)";
  - [ScienceDirect S0361476X2030059X](https://www.sciencedirect.com/science/article/abs/pii/S0361476X2030059X); [ResearchGate](https://www.researchgate.net/publication/342415351_Student_Hand-raising_As_an_Indicator_of_Behavioral_Engagement_and_Its_Role_in_Classroom_Learning)
  - (via search snippets; full text paywalled)
- **Stability over time:** "A Longitudinal Study of Student Hand Raising: Stability and Reciprocal Dynamics With Cognitive Elaboration and Academic Self-Concept":
  - video of 376 German high-school students over multiple lessons in one school year;
  - "nearly half of the variance observed [is] attributable to individual students, while less is attributable to lessons and classroom teachers";
  - hand-raising is both an antecedent and an outcome of cognitive elaboration;
  - [ResearchGate 376076635](https://www.researchgate.net/publication/376076635_A_Longitudinal_Study_of_Student_Hand_Raising_Stability_and_Reciprocal_Dynamics_with_Cognitive_Elaboration_and_Academic_Self-Concept)
  - (snippet only; authors/venue not confirmed)
- **Shyness and context:** Mundelsee & Jurkowski, "Opening the Gateway to Oral Participation: Exploring Facilitative Contextual Factors in the Association Between Student Shyness and Hand Raising," American Educational Research Journal, 2025 — [SAGE](https://journals.sagepub.com/doi/10.3102/00028312241278585) (title only; abstract not retrieved)
- **Who raises, who gets called:** "Classroom interactions: The role of students' gender, classroom subject, and social relations for student hand raising, teacher calls, and student talk time," International Journal of Educational Research, 2025 — [ScienceDirect S0883035525001843](https://www.sciencedirect.com/science/article/pii/S0883035525001843) (403; title only)
- **Wait time:**
  - Rowe, "Wait Time: Slowing Down May Be a Way of Speeding Up!", Journal of Teacher Education 37:43–50, 1986. Teachers typically wait less than 1 s after a question (wait time 1) and less than 1 s after a student response (wait time 2) — [Semantic Scholar](https://www.semanticscholar.org/paper/Wait-Time:-Slowing-Down-May-Be-A-Way-of-Speeding-Rowe/fe469ce77ef5dd3b09a6ad97bee25a741df4af74); [PDF copy](https://www.teachertoolkit.co.uk/wp-content/uploads/2021/09/Budd-Rowe-1.pdf)
  - Extending pauses to 3+ s produced more student talk, *more voluntary responses*, more students who usually stay silent volunteering, longer answers and fewer "I don't know" answers. Source: the systematic review by Shiau, McWilliams & Williams, "The Role of Wait Time During the Questioning of Children," Trauma, Violence & Abuse, 2024. It defines wait time 1 and 2 following Rowe (1974) and cautions that most evidence is from classrooms — [PMC11545128](https://pmc.ncbi.nlm.nih.gov/articles/PMC11545128/)
  - Doubling wait time from 1.5 s to 3 s increases the number of students volunteering, response length and confidence — [Faculty Focus summary](https://www.facultyfocus.com/articles/effective-teaching-strategies/student-learning-in-3-seconds/) (secondary source)
- **Online synchronous classes:** in Zoom lectures, instructors often invited students to unmute and speak but rarely explicitly directed them to the chat. Citation: "Participation and Performance by Gender in Synchronous Online Lectures: Three Unique Case Studies during Emergency Remote Teaching," J. Microbiology & Biology Education, 2021 — [ASM JMBE](https://journals.asm.org/doi/10.1128/jmbe.00281-21) (snippet only)
- **Minerva Forum specifics:**
  - Talk Time shows instructors each student's speaking share: orange < 20% of average, green 20–80%, blue > 80%. It is designed to help them "decide which student to call on" for equitable participation — [Minerva Help Center: Talk Time](https://help.minervaproject.com/en/articles/4296097-talk-time)
  - Forum logs hand-raises, reactions, chats and per-student talk time — [Minerva blog](https://www.minerva.edu/blog/a-teachers-guide-how-to-use-data-to-measure-student-engagement/)
- **Classroom ML on hand-raising is detection, not forecasting:** e.g. hand-raising gesture detection with spatial context (Computers & Graphics) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0097849322002047); YOLOv7-based classroom behaviour detection — [arXiv 2305.07825](https://arxiv.org/pdf/2305.07825)
- **Intention-to-speak research:**
  - Li, Molhoek & Zhou, "Inferring Intentions to Speak Using Accelerometer Data In-the-Wild" (arXiv 2024). Trained on realized intentions, evaluated on realized and annotated *unrealized* ones; "posture shifts are correlated with intentions to speak, but people also often shift posture without having an intention to speak" — [arXiv 2401.05849](https://arxiv.org/abs/2401.05849)
  - "Sensing the Intentions to Speak in VR Group Discussions," Sensors 24(2):362, 2024. Neural model about 62.79% accuracy; intention onset typically precedes speaking by about 1 s — [MDPI DOI 10.3390/s24020362](https://doi.org/10.3390/s24020362) (snippet only)
  - Related: "The Discontent with Intent Estimation In-the-Wild: The Case for Unrealized Intentions" (CHI EA 2024) — [ACM DL](https://dl.acm.org/doi/fullHtml/10.1145/3613905.3644055) (title only)

### Inferences
- The student's label (hand-raise icon) is an *observable intention to speak*, cleaner than the realized-speech labels used in turn-taking work. Hand-raises that are never called on are still positives. This sidesteps the unrealized-intention labelling problem described in the wearables/VR work.
- **Features the literature supports:**
  1. The professor has just asked a question, or just finished an utterance ending in a question / invitation ("thoughts?", "anyone?", "why…?").
  2. Silence since that question (wait time 1); the probability of volunteers rises over the first ~3 s of silence.
  3. His participation history: raises so far this class, time since last raise, and his per-class base rate across past classes, since hand-raising is about half a stable trait.
  4. Lesson segment (discussion vs lecture vs breakout).
- **New feature suggested by Talk Time:** his cumulative *talk time* this class, computable causally from the speaker-tagged transcript. If he has already spoken a lot, he may be less likely to raise (or less likely to be called on). Treat this as a hypothesis to test, not an established effect.
- Because the trait component is large, the main within-person signal for one student is *situational* (questions, pauses, topic relevance to his pre-class answers). Expect modest discrimination: plausibly PR-AUC a few times the base rate, not near-perfect.

### Gaps
- I found no published model that forecasts *when* a student will raise a hand from classroom discourse or timing. This appears to be an open niche, so the student's task is novel and has no published benchmark to compare against.
- Full texts of the Böheim 2020, longitudinal, Mundelsee & Jurkowski 2025 and IJER 2025 papers were not accessible (paywalls/403). Effect sizes for teacher question type (open vs closed) on hand-raising were not obtained.
- No Minerva-published research on hand-raise dynamics was found, only product documentation.

## 5. Which timing features the literature finds most predictive

### Takeaway
The most consistently predictive timing signals are:
- the current silence or pause duration since the last word, conditioned on whether the preceding utterance is lexically/pragmatically complete (e.g. a question);
- elapsed time since a question (wait time);
- recent voice-activity history of each party.

Human between-speaker gaps cluster around 200 ms in dyadic talk, and answers come faster than non-answers. But turn-taking is less precise than often claimed, and many speaker changes are reactive to what just happened. That favours simple causal "time since X" features. Word or text features alone carry real but weaker signal than acoustics.

### Cited Findings
- Stivers et al., "Universals and cultural variation in turn-taking in conversation," PNAS 106(26):10587–10592, 2009:
  - all 10 languages avoid overlap and minimise silence between turns;
  - language means fall within about 250 ms of the cross-language mean, and the typical gap is about 200 ms;
  - answers to questions come faster than non-answer responses, and confirmations 100–500 ms faster than disconfirmations;
  - [PNAS DOI 10.1073/pnas.0903616106](https://www.pnas.org/doi/10.1073/pnas.0903616106)
- Heldner & Edlund, "Pauses, gaps and overlaps in conversations," Journal of Phonetics 38(4):555–568, 2010:
  - turn-taking is "generally less precise than is often claimed";
  - "the proportion of speaker changes that could potentially be triggered by information immediately preceding the speaker change is large enough for reactive interaction control models to be viable";
  - [KTH/DiVA author PDF](https://www.diva-portal.org/smash/get/diva2:388247/FULLTEXT01.pdf); [DOI 10.1016/j.wocn.2010.08.002](https://doi.org/10.1016/j.wocn.2010.08.002)
- Classroom wait time:
  - teachers typically pause under 1 s after questions (Rowe 1986) — [Semantic Scholar](https://www.semanticscholar.org/paper/Wait-Time:-Slowing-Down-May-Be-A-Way-of-Speeding-Rowe/fe469ce77ef5dd3b09a6ad97bee25a741df4af74)
  - the systematic review reports natural wait time 1 averaging 2.04 s (range 0.97–3.70 s) and wait time 2 averaging 0.69 s across the studies it covers;
  - 3+ s pauses increase volunteering;
  - [PMC11545128](https://pmc.ncbi.nlm.nih.gov/articles/PMC11545128/)
- In continuous models, voice-activity history plus words with ASR-like 100 ms delay give F ≈ 0.71 at pauses, against 0.77 for acoustics. POS features help most at speech onsets — [Roddy et al. 2018](https://ar5iv.labs.arxiv.org/html/1806.11461)
- Turn-final TRPs are much easier to predict than within-turn pauses ([Lla-VAP, arXiv 2412.18061](https://arxiv.org/abs/2412.18061), snippet only). LLMs especially miss within-turn TRPs ([Umair et al. 2024](https://arxiv.org/abs/2410.16044)).
- In TurnGPT, context (previous turns) and pragmatic completeness of the current turn drive TRP prediction — [TurnGPT paper](https://arxiv.org/pdf/2010.10874)
- Intention-to-speak onset precedes actual speaking by about 1 s in VR group discussions — [Sensors 2024](https://doi.org/10.3390/s24020362) (snippet only)

### Inferences
- **Recommended causal timing feature set** (all computable from word timestamps up to t):
  - current silence duration (now minus last finalized word end);
  - whether the last utterance was the professor's and ended in "?" or a wh-/invitation phrase;
  - time since the last professor question (a wait-time proxy);
  - current speaker identity (professor / him / other student / none);
  - length of the current turn so far;
  - number of speaker changes in the last 60 s;
  - his raises so far and time since his last raise;
  - minute of class and lesson segment;
  - his talk time so far.
- **Non-linear effects:** effects such as "probability peaks 1–4 s after a question, then decays" call for gradient-boosted trees or binned/spline features in a logistic model. Bucketing silence and time-since-question into bins is cheap and interpretable.
- **ASR-derived pauses are noisy:** word end-times from ASR and diarization may be off by 100s of ms. With a 2 s decision grid and H ≥ 1 s, this matters less than in sub-second turn-taking work, but alignment between ASR clock and hand-icon clock must be checked.
- **Hand-raise vs speech timing:** the ~200 ms gap norm applies to dyadic *speech* onsets. Hand-raises in a moderated seminar are bids to a chair (the professor) and may be slower and more deliberate. Treat the dyadic timing norms as background, not as the expected hand-raise latency.

### Gaps
- No source quantifies hand-raise latency after a question in online seminars.
- Heldner & Edlund's exact proportions (e.g. the share of gaps within ±200 ms) were not extracted.
- I found no feature-importance study ranking timing features for *multi-party online classes* specifically.

## 6. What is realistic to replicate in 4 days on an M4, risks, licences, and non-causal pitfalls

### Takeaway
Realistic this week:
- a causal feature-engineering pipeline (timing + lexical cues + frozen sentence embeddings) with logistic regression / LightGBM;
- a multi-horizon or hazard target;
- class-grouped evaluation.

TurnGPT-style ideas can enter as cheap zero-shot LM features.

Not realistic: retraining TurnGPT (CUDA-pinned, no checkpoints), running VAP (needs per-speaker audio; checkpoints academic-only), or LoRA fine-tuning 8B+ LLMs.

The main risks are leakage from non-causal ASR/diarization and windowing, and over-claiming from a small number of positive events.

### Cited Findings
- **Licences and availability:**
  - TurnGPT: MIT; no released checkpoints; CUDA 11.3 dependency — [GitHub TurnGPT](https://github.com/ErikEkstedt/TurnGPT)
  - VAP: MIT; one example state-dict included; training needs private corpora — [GitHub VAP](https://github.com/ErikEkstedt/VoiceActivityProjection)
  - VAP-Realtime: code MIT, models "academic purposes only" — [GitHub VAP-Realtime](https://github.com/inokoj/VAP-Realtime)
- **Payoff of fine-tuning:** going from zero-shot to LoRA-fine-tuned Llama 3.1 8B raised AMI next-speaker accuracy from 34.88% to 47.85%, with a trivial baseline at 33.41% — [Hilgert & Niehues 2025](https://aclanthology.org/2025.icnlsp-1.7.pdf). Zero-shot LLMs "failed consistently" at speak-or-stay-silent decisions without fine-tuning — [Bhagtani et al. 2026](https://arxiv.org/abs/2603.11409)
- **Text-only LLMs can match or beat humans** at next-speaker prediction (Qwen3-14B F1 69.4% vs humans 60.1% on AMI). This used ground-truth transcripts and utterance-boundary decision points — [Fukuda et al. 2026](https://arxiv.org/html/2606.17542v1)
- **Real-time word-feature precedent:** delay word features by about 100 ms to simulate ASR latency — [Roddy et al. 2018](https://ar5iv.labs.arxiv.org/html/1806.11461)

### Inferences
**Recommended 4-day plan (ranked):**
1. **Baselines (half a day):**
   - constant base rate;
   - a hazard-by-time-since-last-raise curve;
   - a rule "professor question in last k s".

   Evaluate with PR-AUC / average precision and event-level recall at fixed alert rates. Use GroupKFold or a time-ordered split *by class session*.
2. **Hand-crafted causal timing + lexical features into LightGBM or logistic regression (1–1.5 days).** Use the feature set from Q5, plus keyword/question flags and TF-IDF over the professor's last utterance, plus overlap between the current 60 s window and his pre-class answers (TF-IDF cosine).
3. **Frozen text embeddings (half a day to a day).** Use a small sentence-embedding model run locally on MPS/CPU over the last utterance(s) and the 60 s window, then PCA, then the same GBM. Cache embeddings per decision point. Optionally add cosine similarity between the window embedding and his submitted pre-class answers ("topic relevance to him").
4. **Optional TurnGPT-flavoured feature:** zero-shot GPT-2 (124M) probability that the current professor utterance is complete (P of "?" / "." / end-of-text after the last words). It is cheap to compute and directly echoes TurnGPT's TRP idea without training it.
5. **Multi-horizon (VAP-inspired) output:** fit one model per H, or a discrete-time hazard model on 2 s bins and derive P(raise within H) by cumulating. This guarantees monotonicity across H, unlike independent per-H models.

**Not realistic this week:** retraining TurnGPT (repo needs porting off CUDA 11.3 and its custom dataset package, no checkpoints); fine-tuning a GPT-2 with `<raise>` tokens (doable, but slow to iterate and hard to explain); VAP (no per-speaker audio, more than 2 speakers, model-licence limits); LoRA on 8B LLMs (memory and time on M4, plus Llama-licence terms).

**Non-causal / leakage flags (check each):**
- **(a) Offline ASR and diarization.** Whole-recording ASR uses future audio context, so speaker labels, punctuation and word boundaries can be revised by later audio. Use only words with end time ≤ t − latency. Prefer streaming-style segmentation, or at least document the issue.
- **(b) Utterance boundaries.** Next-speaker papers use the *complete* current utterance, which is unknown at t if the speaker is mid-utterance. Features like "last utterance ended in ?" must use only words already spoken.
- **(c) Sentence segmentation and punctuation** from a punctuation model that looks ahead.
- **(d) Normalisation with whole-class or whole-dataset statistics.** E.g. "his total raises in this class", or z-scoring by a class's mean pause. Fit scalers on training folds only, and use running statistics within a class.
- **(e) Overlapping positives.** Decision points every 2 s with H up to 20 s create highly autocorrelated positives. Random row-level splits leak across adjacent points, so split by class.
- **(f) Post-hoc "lesson segment".** The current segment must come from the lesson plan or schedule as known at t, not inferred from later events.
- **(g) Bidirectional encoders** (BERT-style) are fine *only* if their input window ends at t.
- **(h) VAP/turn-taking evaluation events** (shift/hold) are defined with future activity. They are OK as labels, never as features.

**Other risks:**
- **Few positive events:** a few hundred to a few thousand events over 150–600 classes, so high-variance estimates. Report confidence intervals across class folds.
- **Label noise:** clock alignment between hand-icon timestamps and ASR timestamps; hands lowered and re-raised.
- **Instructor-policy confound:** Talk Time–driven cold calls may reduce his need to raise.
- **Domain shift** across courses and professors.

### Gaps
- Licences for GPT-2 weights, candidate sentence-embedding models, Llama 3 and the AMI corpus were not verified in this session. Check the model cards before use.
- MPS throughput for fine-tuning GPT-2 on several million tokens on an M4 was not benchmarked; this is the main unknown if the student attempts the `<raise>`-token LM.
- No public dataset of online-seminar hand-raises was found for pre-training or transfer. The student's data appears to be unique.
