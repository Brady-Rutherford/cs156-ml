# Classroom-Discourse NLP Resources for Hand-Raise Prediction (datasets, pretrained classifiers, dialogue-act and question detection)

Scope note: this covers off-the-shelf NLP feature extractors and transfer-learning starting points for the speaker-tagged ASR transcript stream. Checked October 2026. Hugging Face (HF) IDs, licences, file sizes and architectures were read directly from the HF Hub API (`https://huggingface.co/api/models/<id>`) on 2026-10-12; links go to the model pages. Almost every education-domain resource below was trained on **K-12 (mostly elementary/middle-school math) human-made transcripts**. None was trained on university seminars or on Deepgram ASR output, so treat all of them as *noisy features* and not as ground truth.

---

## 1. TalkMoves dataset and models (Suresh et al.)

### Takeaway
TalkMoves (LREC 2022) is the largest public K-12 math transcript set with sentence-level accountable-talk labels: 6 teacher moves and 4 student moves. The labs released no official checkpoint. Several third-party checkpoints exist on HF, and the most usable is the ELECTRA-small pair (Apache-2.0) that Edu-ConvoKit loads. The data are CC BY-NC-SA 4.0. That is fine for a class project, but derived data and models inherit the NC-SA terms.

### Cited Findings
- **Paper**: Suresh, Jacobs, Harty, Perkoff, Martin & Sumner (2022), "The TalkMoves Dataset: K-12 Mathematics Lesson Transcripts Annotated for Teacher and Student Discursive Moves", LREC 2022 — [ACL Anthology](https://aclanthology.org/2022.lrec-1.497/) ([PDF](https://aclanthology.org/2022.lrec-1.497.pdf))
- **Size**: 567 transcripts with 174,186 annotated teacher utterances, 59,874 student utterances and 1.8M words (15,830 unique). Most are whole lessons of about 55 min. All were human-transcribed from audio/video, segmented by speaker (teacher vs student), and annotated **at the sentence level** — [LREC PDF](https://aclanthology.org/2022.lrec-1.497.pdf)
- **Teacher talk moves** (6 + none), with the paper's examples:
  - Keeping everyone together: "What did Eliza just say her equation was?"
  - Getting students to relate to another's ideas: "Do you agree with Juan that the answer is 7/10?"
  - Restating
  - Revoicing
  - Pressing for accuracy: "Can you give an example of an ordered pair?"
  - Pressing for reasoning: "Why could I argue that the slope should be increasing?"
  
  Source: [LREC PDF, Table 1](https://aclanthology.org/2022.lrec-1.497.pdf)
- **Student talk moves** (4 + none): Relating to another student, Asking for more info, Making a claim, Providing evidence/reasoning — [LREC PDF, Table 1](https://aclanthology.org/2022.lrec-1.497.pdf)
- **Label distribution**, as a share of teacher utterances: Keeping everyone together 13.1%, Press for accuracy 13.2%, Revoicing 2.3%, Getting students to relate 1.6%, Restating 1.5%, Press for reasoning 1.2%, None 67.2%. Student utterances: Making a claim 30.6%, Providing evidence 13.4%, Relating 11.1%, Asking for more info 3.2%, None 41.7% — [LREC PDF, Table 4](https://aclanthology.org/2022.lrec-1.497.pdf)
- **Human reliability** is high, with Cohen's kappa 0.91–1.0 across teacher moves at the initial and midpoint checks — [LREC PDF, Table 3](https://aclanthology.org/2022.lrec-1.497.pdf)
- **Model performance** (test set, F1 %):
  - Teacher model: RoBERTa-base 76.32 (MCC 0.751), Electra-base 75.77, BERT-base 75.38
  - Student model: BERT-base 73.12, RoBERTa-base 71.58, Electra-base 71.28
  - The teacher input is a "sentence pair": the teacher sentence concatenated with the immediately prior student sentence. The output is a 7-way softmax.
  
  Source: [LREC PDF, Table 6](https://aclanthology.org/2022.lrec-1.497.pdf)
- **Dialogue acts already in the dataset**: every utterance carries an automatic SWBD-DAMSL dialogue-act tag from a Switchboard-trained SGNN. Only 7 of the 42 tags appear. 77.2% are "Statement-non-opinion" and only 0.457% are "Yes-No-Question". The authors suspect this reflects Switchboard training bias — [LREC PDF, Table 5 & §3.3](https://aclanthology.org/2022.lrec-1.497.pdf)
- **ASR in the TalkMoves app**: the deployed TalkMoves application runs uploaded audio through a transcript step, splits it into sentences, labels each as teacher or student, then applies the talk-move models — [LREC PDF §3.4](https://aclanthology.org/2022.lrec-1.497.pdf)
- **Licence**: the GitHub repo states "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License", and the README does not mention released model checkpoints — [SumnerLab/TalkMoves](https://github.com/SumnerLab/TalkMoves). The paper itself is CC-BY-NC-4.0 (ELRA) — [LREC PDF](https://aclanthology.org/2022.lrec-1.497.pdf)
- **HF checkpoints, all third-party**:
  - `YaHi/teacher_electra_small` and `YaHi/student_electra_small`
    - Fine-tuned from `google/electra-small-discriminator`; about 54 MB each; Apache-2.0
    - The card says "unknown dataset" and reports only validation loss 0.2845, no F1
    - Edu-ConvoKit uses them as its teacher and student TalkMoves annotators. Its label map is 0 none, 1 Keeping Everyone Together, 2 Getting Students to Relate, 3 Restating, 4 Revoicing, 5 Pressing for Accuracy, 6 Pressing for Reasoning; student 0 none, 1 Relating, 2 Asking for More Info, 3 Making a Claim, 4 Providing Evidence. The Edu-ConvoKit source notes that labels 4 and 5 are swapped relative to the coding manual.
    
    Sources: [HF teacher](https://huggingface.co/YaHi/teacher_electra_small), [HF student](https://huggingface.co/YaHi/student_electra_small), [Edu-ConvoKit constants.py](https://github.com/EduNLP/edu-convokit/blob/main/edu_convokit/constants.py)
  - `ahoang/electra-base-discriminator-talkmoves-teacher` (7 labels) and `ahoang/electra-base-discriminator-talkmoves-student`
    - ElectraForSequenceClassification, about 110M params, 438 MB
    - Auto-generated card with no data, metrics or licence information
    
    Sources: [HF teacher](https://huggingface.co/ahoang/electra-base-discriminator-talkmoves-teacher), [HF student](https://huggingface.co/ahoang/electra-base-discriminator-talkmoves-student)
  - `StanfordSCALE/tutor_talkmoves_roberta_large_classifier` (355M params, 1.42 GB) and `StanfordSCALE/tutor_talkmoves_modernbert_base_classifier` (149M params)
    - Trained on **tutoring** chats, not classrooms: 1,849 examples, Krippendorff α 0.83
    - Only 3 merged classes + none: Classroom management, Pressing for accuracy or reasoning, Restating or revoicing
    - Input format: `[PRETEXT] {3 previous messages} [TEXT] {target}`, lowercased
    - Test macro-F1 0.703. Per class: Pressing 0.871, Classroom management 0.723, Restating/revoicing 0.286
    - No licence stated
    
    Source: [HF card](https://huggingface.co/StanfordSCALE/tutor_talkmoves_roberta_large_classifier)
- **Unverified figure**: one search-engine snippet attributed 82.4 macro-F1 for tutor talk moves and 76.5 for student talk moves to RoBERTa-base in a tutoring-dialog paper ([arXiv 2410.19231](https://arxiv.org/abs/2410.19231)). This is a different dataset and setup from the LREC figures above, and I did not verify it in the paper text.

### Inferences
- For hand-raise prediction, the teacher moves that invite responses are the most relevant: *Keeping everyone together*, *Getting students to relate*, *Pressing for reasoning/accuracy*. They mark moments when the professor is asking the floor to respond. Student moves (*Making a claim*, *Relating to another student*) on classmates' turns could capture discussion momentum.
- Because the class priors are very skewed, the low-frequency moves (press for reasoning 1.2%, getting students to relate 1.6%) will be noisy in a new domain. Use predicted probabilities as continuous features rather than argmax labels.
- The TalkMoves Switchboard dialogue-act tags put 77% of classroom utterances into "statement-non-opinion", so off-the-shelf Switchboard dialogue-act taggers are likely to under-detect classroom questions without adaptation.
- TalkMoves models expect sentence-level teacher input paired with the previous student sentence. Deepgram utterances (with `utterances`/`paragraphs` segmentation) will need re-segmentation into sentences, for example on smart_format punctuation.

### Gaps
- No official TalkMoves checkpoint from the Sumner Lab or the University of Colorado was found on GitHub or HF. The provenance and test performance of the YaHi and ahoang checkpoints are undocumented on their cards.
- No published evaluation of TalkMoves models on university-level or online-seminar transcripts, or on ASR versus human transcripts, was found.

---

## 2. Conversational uptake (Demszky et al. 2021) and the NCTE transcripts (Demszky & Hill 2023)

### Takeaway
A pretrained uptake model is downloadable as `stanford-nlpxed/uptake-model` (BERT-base, about 438 MB, custom multi-head class). It scores how much a teacher turn builds on the preceding student turn, and its agreement with humans (Spearman ρ = .54) is as good as human inter-rater agreement. The NCTE transcripts (1,660 lessons) are free after a Google Form. Licences conflict: the code is MIT, but the HF model card is CC BY-NC-ND with no-redistribution terms.

### Cited Findings
- **Paper**: Demszky, Liu, Mancenido, Cohen, Hill, Jurafsky & Hashimoto (2021), "Measuring Conversational Uptake: A Case Study on Student-Teacher Interactions", ACL 2021 — [ACL PDF](https://aclanthology.org/2021.acl-long.130.pdf)
- **Method**: uptake is formalised as pointwise Jensen-Shannon divergence (pJSD) between a student utterance S and a teacher reply T. It is estimated by fine-tuning BERT-base with **next-utterance classification** (no human labels needed) on a combination of NCTE classroom transcripts, Switchboard and a tutoring dataset. Pairs with S < 5 tokens or [Inaudible] were filtered out — [ACL PDF](https://aclanthology.org/2021.acl-long.130.pdf)
- **Validation**: 2,246 annotated exchanges. pJSD reaches Spearman ρ = .540 [95% CI .505–.574] against human uptake ratings, about the human upper bound (ρ ≈ .539). It beats word-overlap baselines (%-IN-T ρ = .523, BLEU ρ = .510). Uptake correlates with instruction quality and student satisfaction across three datasets (NCTE, a one-on-one tutoring set, SimTeacher) — [ACL PDF](https://aclanthology.org/2021.acl-long.130.pdf)
- **A text-only cue matters**: in the same paper, exchanges where pJSD beats %-IN-T on predicting student ratings have more teacher question marks (p < .05) and exclamation marks (p < .01). This suggests the model partly keys on follow-up questions — [ACL PDF](https://aclanthology.org/2021.acl-long.130.pdf)
- **Checkpoint**:
  - `stanford-nlpxed/uptake-model`. The older `ddemszky/uptake-model` redirects to it (HTTP 307).
  - Architecture `MultiHeadModel` (bert-base), `pytorch_model.bin` 438 MB, ships its own `handler.py` and `utils.py`
  - Licence tag `cc-by-nc-nd-4.0`. The card adds: "We will not use this model for commercial purposes… will not attempt to recover any of the training / fine-tuning data… will not share or distribute this model… outside of the research team"
  
  Source: [HF card](https://huggingface.co/stanford-nlpxed/uptake-model)
- **Code**: the GitHub repo (`ddemszky/conversational-uptake`) is MIT-licensed. It takes CSV input with a student-utterance column and a teacher-response column, requires the first speaker's utterance to be at least 5 tokens, and links the HF checkpoint — [GitHub](https://github.com/ddemszky/conversational-uptake)
- **NCTE dataset**: Demszky & Hill (2023), "The NCTE Transcripts: A Dataset of Elementary Math Classroom Transcripts", BEA 2023 — [arXiv 2211.11772](https://arxiv.org/abs/2211.11772)
  - 1,660 lessons of 45–60 min, 4th–5th grade math, 2010–2013, 317 teachers, 4 US districts
  - On average 350 turns per transcript; teachers produce 50.2% of turns but 87.7% of words
  - Professional human transcription. Names are replaced with "Student J", "Teacher" and similar. 4% of teacher and 21% of student utterances contain [Inaudible].
  - Timestamps are only available at about 7.5-minute granularity
  
  Source: [arXiv PDF](https://arxiv.org/pdf/2211.11772)
- **NCTE turn-level annotations and RoBERTa 5-fold CV performance**:

  | Annotation | Accuracy | F1 |
  |---|---|---|
  | Student on-task | 0.902 | 0.942 |
  | Teacher on-task | 0.867 | 0.923 |
  | Teacher uptake | 0.768 | 0.688 |
  | Focusing question | 0.856 | 0.501 |
  | Student reasoning | 0.863 | 0.651 |

  The student-reasoning labels cover 2,000 student utterances of at least 8 words, with 90% calibration agreement. Source: [arXiv PDF, Table 4](https://arxiv.org/pdf/2211.11772)
- **Outcome links**: each additional High Uptake per 7.5-min segment goes with +0.12 SD in teacher value-added. Focusing questions go with +0.23 SD and student reasoning with +0.19 SD. Simple *student talk-ratio* baselines did **not** correlate with observation scores or value-added, except student word percentage with "instructional dialogue" — [arXiv PDF, Table 5 & §5](https://arxiv.org/pdf/2211.11772)
- **Access**: fill in a Google Form (https://forms.gle/1yWybvsjciqL8Y9p8) to get the Google Drive folder. Metadata and documentation are on ICPSR (study #36095). The repo is MIT — [GitHub ddemszky/classroom-transcript-analysis](https://github.com/ddemszky/classroom-transcript-analysis). Parents and teachers consented, under Harvard IRB #17768, to de-identified data being shared publicly for research — [arXiv PDF](https://arxiv.org/pdf/2211.11772)

### Inferences
- **Using uptake as a feature**: run it on every (classmate turn → professor turn) pair in the 60 s window. A high score means the professor is *building on* student contributions, which suggests discussion mode and an open floor where more hand raises are likely. A low score after a student turn suggests the professor is moving on or lecturing. The model is directional (student → teacher), so on (professor → classmate) pairs it would be out of distribution.
- **Causality constraint**: uptake needs the teacher reply to be complete. At a decision point you can only score pairs whose professor turn has already finished, so features lag by up to one turn.
- The CC BY-NC-ND card terms ("not share or distribute… outside of the research team") mean the student should **not** commit the weights to a public repo. Download at runtime instead.
- NCTE is useful mainly for pretraining or domain-adaptive fine-tuning on classroom speech. Its 7.5-min timestamps and anonymised, non-persistent student labels make it **unsuitable for training a timing model**.

### Gaps
- I could not confirm whether NCTE speaker labels track individual students consistently across a lesson; the paper shows only anonymised forms like "Student J". The Google Form's exact data-use terms beyond research use were not visible without submitting it.
- No published evaluation of the uptake model on ASR transcripts or on higher-education seminars was found.

---

## 3. Other education-NLP classifiers: focusing/funneling questions, student reasoning, accountable talk, EduBehaviors assertions, LLM-based coding (2023–2026)

### Takeaway
The most directly useful new resource is **EduBehaviors (Stanford SCALE, Sept 2026)**. It is 49 small SetFit classifiers on HF, including `assertion_sentence_invites_participation` (test F1 0.75, ROC-AUC 0.92), `has_a_question`, `calls_on_student_by_name` and `addresses_the_whole_class`. They map almost one-to-one onto "is the professor opening the floor?". Stanford's focusing-question and student-reasoning classifiers are downloadable but have only moderate F1 (0.50 and 0.65). Recent LLM-coding studies show zero-shot LLMs trail fine-tuned encoders on talk moves.

### Cited Findings
**Focusing vs funneling questions**
- Alic, Demszky, Mancenido, Liu, Hill & Jurafsky (2022), "Computationally Identifying Funneling and Focusing Questions in Classroom Discourse", BEA 2022 — [ACL Anthology](https://aclanthology.org/2022.bea-1.27/)
- The data are 2,348 NCTE exchanges labelled funneling / focusing / neither by 3 raters, with filtered and unfiltered subsets. The repo is MIT and links no checkpoint — [GitHub sterlingalic/funneling-focusing](https://github.com/sterlingalic/funneling-focusing)
- **Checkpoint**: `stanford-nlpxed/focusing-questions` (redirected from `ddemszky/focusing-questions`)
  - Config is `BertForSequenceClassification`, 438 MB, licence `cc-by-nc-3.0`
  - The Edu-ConvoKit paper calls it "a fine-tuned RoBERTa classifier", which conflicts with the HF config
  - Edu-ConvoKit warns that it was trained on teacher utterances only
  
  Sources: [HF](https://huggingface.co/stanford-nlpxed/focusing-questions), [Edu-ConvoKit paper](https://arxiv.org/abs/2402.05111), [annotator.py](https://github.com/EduNLP/edu-convokit/blob/main/edu_convokit/annotation/annotator.py)
- Within NCTE its F1 is 0.501, the weakest of the NCTE classifiers — [NCTE paper](https://arxiv.org/pdf/2211.11772)

**Student reasoning**
- `stanford-nlpxed/student-reasoning`: config `BertForSequenceClassification`, 438 MB, licence `gpl`. Edu-ConvoKit only scores utterances of at least 8 words (`STUDENT_REASONING_MIN_NUM_WORDS = 8`) — [HF](https://huggingface.co/stanford-nlpxed/student-reasoning), [constants.py](https://github.com/EduNLP/edu-convokit/blob/main/edu_convokit/constants.py). F1 0.651 on NCTE — [NCTE paper](https://arxiv.org/pdf/2211.11772)

**Edu-ConvoKit (one wrapper for the above)**
- Wang, Demszky et al., NAACL 2024 demo; MIT licence; repo now at `EduNLP/edu-convokit` with 117 stars, last pushed April 2025 — [arXiv 2402.05111](https://arxiv.org/abs/2402.05111), [ACL PDF](https://aclanthology.org/2024.naacl-demo.6.pdf), [GitHub](https://github.com/EduNLP/edu-convokit)
- Its annotators are talk time, math density, student reasoning, focusing questions, conversational uptake, and teacher and student TalkMoves. A stated design goal is to "run on a CPU and support large-scale pre-processing". It needs only `speaker` and `text` columns — [Edu-ConvoKit paper](https://arxiv.org/abs/2402.05111)
- It also lists "teacher launch features" (RoBERTa-large-sized, about 1.4 GB each, no licence or paper yet): `stanford-nlpxed/confusion`, `prior_knowledge`, `norms`, `feelings`, `own_ideas`, `solution`, `elicits_background`, `multimodal_materials`, `na_0.00001_0.9` — [constants.py](https://github.com/EduNLP/edu-convokit/blob/main/edu_convokit/constants.py); sizes from the HF API

**EduBehaviors assertion classifiers (2026)**
- Bernado, Trindade Ribeiro, Beberman & Loeb, "EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues", arXiv 2609.27043, submitted 22 Sept 2026, paper licence CC BY 4.0 — [arXiv](https://arxiv.org/abs/2609.27043)
- **Method and results**:
  - LLMs label low-level observable "assertions" per utterance; the assertions then feed classifiers for higher-level constructs
  - 49 SetFit encoder classifiers are released on HF, trained on LLM annotations of TalkMoves teacher utterances
  - Best pipeline on Teacher TalkMoves: macro-F1 0.673 and Cohen's κ 0.688. For comparison, fine-tuned RoBERTa-base reaches about 0.76 macro-F1, and direct LLM prompting peaks at κ ≈ 0.58
  
  Source: [arXiv HTML](https://arxiv.org/html/2609.27043)
- **Model details**:
  - HF IDs follow `StanfordSCALE/assertion_sentence_<slug>`, all created 2026-09-18
  - Body `sentence-transformers/paraphrase-mpnet-base-v2` (109M params, 438 MB) with a LogisticRegression head
  - Input is the raw single utterance
  - No licence field on the cards
  - pip package `edubehaviors-kit` 0.1.1 (GitHub `scale-nssa/edubehaviors-kit`)
  
  Sources: [HF card example](https://huggingface.co/StanfordSCALE/assertion_sentence_invites_participation), [PyPI](https://pypi.org/project/edubehaviors-kit/)
- **`assertion_sentence_invites_participation`**: base rate 24.3%; test (n = 2,144) precision 0.795, recall 0.709, F1 0.750, ROC-AUC 0.920, AP 0.805. Agreement among the LLM annotators is Krippendorff α 0.610. The card says the "behaviour on student speech is untested" — [HF card](https://huggingface.co/StanfordSCALE/assertion_sentence_invites_participation)
- **Training data**: `StanfordSCALE/assertions_llm_annotated_talkmoves`, about 6.4k teacher utterances with per-assertion splits. Base rate and LLM-annotator α for relevant assertions:

  | Assertion | Base rate | α |
  |---|---|---|
  | has a question | 19.0% | 0.867 |
  | calls on student by name | 5.4% | 0.918 |
  | includes student name | 6.9% | 0.930 |
  | has a directive or instruction | 19.0% | 0.735 |
  | is a declarative statement | 39.3% | 0.815 |
  | answers a question | 6.2% | **0.101** (unreliable) |

  The dataset card shows no licence — [HF dataset](https://huggingface.co/datasets/StanfordSCALE/assertions_llm_annotated_talkmoves)
- **Other released slugs relevant here**: `addresses_the_whole_class`, `has_a_rhetorical_question`, `checks_for_understanding_or_agreement`, `seeks_or_gives_clarification`, `evaluates_a_student_response`, `grants_or_requests_permission`, `has_a_directive_or_instruction`, `summarizes_or_reviews`, `references_task_procedure_or_logistics`, `is_incomplete_or_trails_off`, `has_praise_or_encouragement`, `repeats_or_revoices_prior_speech`, `has_agreement_or_affirmation`, `expresses_confusion_or_requests_help`, `shows_uncertainty` — [HF StanfordSCALE model list](https://huggingface.co/StanfordSCALE)

**Accountable talk in an online course (closest domain match)**
- Kupor, Morgan & Demszky (2023), "Measuring Five Accountable Talk Moves to Improve Instruction at Scale", arXiv 2311.10749 — [arXiv](https://arxiv.org/abs/2311.10749)
- They annotated 2,500 instructor utterances from small-group sections of the **online** CS course Code in Place for adding on, connecting, eliciting, probing and revoicing. They fine-tuned RoBERTa and GPT-3; GPT-3 had higher precision but variable recall.
- Talk-move use correlated with section attendance, ratings and assignment completion, and "connecting student ideas" had the largest positive effect.
- No checkpoint release is indicated.

**LLM-based classroom-discourse coding (2023–2026)**
- Wang & Demszky (2023, BEA), "Is ChatGPT a Good Teacher Coach?": zero-shot ChatGPT feedback lacked novelty, and "82% of the model's suggestions point to places in the transcript where the teacher is already implementing that suggestion" — [arXiv 2306.03090](https://arxiv.org/abs/2306.03090)
- Xu, Liu, Jones, Cohen & Ai (2024, NAACL), "The Promises and Pitfalls of Using Language Models to Measure Instruction Quality in Education":
  - LMs match human agreement on discrete, low-inference items but degrade on complex practices
  - Teacher-only utterances proved effective even for student-centred variables
  - Key obstacles: noisy, long input and imbalanced ratings
  
  Source: [arXiv 2404.02444](https://arxiv.org/abs/2404.02444)
- Ahtisham, Vanacore, Zhou, Lee & Kizilcec (Feb 2026), "LLM Reasoning Predicts When Models Are Right: Evidence from Coding Classroom Discourse": 30,300 teacher utterances coded by several LLMs. A random forest on features of the LLM's reasoning predicts whether a label is correct (F1 0.83, recall 0.854) — [arXiv 2602.09832](https://arxiv.org/abs/2602.09832)
- Santos-Deonizio, Malamut, Martínez & Demszky (Aug 2026), "When Youth Enter The Chat": 8th-grade students contested both the LLM classifications of their talk and the coding scheme. The authors argue that F1 against adult experts misses how students experience participation — [arXiv 2608.23780](https://arxiv.org/abs/2608.23780)
- A 2024 Computers & Education: AI paper, "Evaluating the performance of ChatGPT and GPT-4o in coding classroom discourse data: A study of synchronous online mathematics instruction", is directly on online synchronous classes, but the full text was blocked (HTTP 403) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666920X24001280)

### Inferences
- **Recommended feature stack for "is this professor utterance an open invitation to respond?"**:
  - `assertion_sentence_invites_participation`
  - `assertion_sentence_has_a_question` and `has_a_rhetorical_question`, since a rhetorical question should *not* trigger a hand raise
  - `assertion_sentence_addresses_the_whole_class` vs `calls_on_student_by_name`, since a cold call to someone else probably *suppresses* hand raises
  - `stanford-nlpxed/focusing-questions`
  
  Take probabilities, not labels.
- Calling on a student by name is likely better handled with a regex or fuzzy match against the class roster (and "him" vs others) on Deepgram text than with the SetFit model, given that names in ASR are not anonymised.
- Given the 2023–2026 evidence (Wang & Demszky 2023; EduBehaviors' κ ≈ 0.58 for direct prompting; Xu et al. 2024), using an LLM to label 100k utterances zero-shot is unlikely to beat these encoders on talk-move-style constructs. LLMs are better used to hand-label a few hundred seminar utterances to calibrate or fine-tune a small encoder, which is the EduBehaviors recipe.
- The EduBehaviors data and models derive from TalkMoves (CC BY-NC-SA 4.0), so the NC-SA terms plausibly apply even though the cards state no licence.

### Gaps
- No released checkpoints were found for the Kupor et al. (2023) online-course talk moves, which are the closest domain match.
- I could not access per-category agreement figures for the 2024 synchronous online-math ChatGPT/GPT-4o paper.
- Performance numbers for the HF focusing-question checkpoint itself (as opposed to the paper's CV results) and for the "teacher launch" models are not published on their cards.

---

## 4. Dialogue-act classification models (SwDA, MRDA, DailyDialog, SILICONE) with downloadable checkpoints

### Takeaway
Generic dialogue-act checkpoints exist, but the ones on HF are mostly hobby or small-lab uploads with thin documentation. The best-documented are `diwank/silicone-deberta-pair` (MIT, 11 merged acts including `ask` and `ask_yes_no`) and `WSHAPER/dialogue-act-classifier-fp16` (Apache-2.0, DailyDialog 4-way, macro-F1 0.76). Classic evidence says dialogue-act tagging degrades moderately on ASR text. Switchboard-style taggers misclassify classroom questions as statements.

### Cited Findings
- **SILICONE benchmark**: Chapuis et al. (2020), "Hierarchical Pre-training for Sequence Labelling in Spoken Dialog", Findings of EMNLP 2020. It covers 10 spoken-dialogue datasets including SwDA, MRDA and DailyDialog. Their hierarchical transformer averages 74.3 accuracy against 72.8 for BERT-base — [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.239/), [GitHub eusip/SILICONE-benchmark](https://github.com/eusip/SILICONE-benchmark)
- **Corpus sizes**: SwDA has 1,155 conversations and about 205k utterances. MRDA has about 180k DA tags over 72 h from 75 meetings. MRDA conversations are much longer (about 1,493 utterances on average vs 271 for SwDA) — [Context-aware self-attention DA paper, arXiv 1904.02594](https://arxiv.org/pdf/1904.02594); [MRDA corpus](https://www.researchgate.net/publication/2904172_The_ICSI_meeting_recorder_dialogue_act_MRDA_corpus)
- **`diwank/silicone-deberta-pair`**:
  - deberta-base, `pytorch_model.bin` 557 MB, MIT
  - Trained on the balanced `diwank/silicone-merged` dataset
  - Input is a pair (previous utterance, current utterance)
  - 11 labels: acknowledge, answer, backchannel, reply_yes, exclaim, say, reply_no, hold, **ask**, intent, **ask_yes_no**
  - Loaded via simpletransformers; metrics only in a W&B report linked from the card
  
  Source: [HF card](https://huggingface.co/diwank/silicone-deberta-pair)
- **`WSHAPER/dialogue-act-classifier-fp16`**:
  - DistilBERT fine-tuned on DailyDialog, 4 classes (commissive, directive, inform, **question**), Apache-2.0
  - Macro-F1 0.762, accuracy 83.4%
  - FP16 ONNX 134 MB, with a multilingual variant
  - Latencies on the card are for an NVIDIA GPU only
  
  Source: [HF card](https://huggingface.co/WSHAPER/dialogue-act-classifier-fp16)
- **Other HF dialogue-act uploads without usable cards**: `aditiverma6/meeting-dialogue-act-bert` (BERT-base, 2026; README missing), `pzelasko/longformer-swda-nolower` (token classification) and `ibm-research/roberta-base-vira-dialog-acts-live` (a vaccine-chatbot domain) — from the HF Hub API search on 2026-10-12, [HF](https://huggingface.co/aditiverma6/meeting-dialogue-act-bert)
- **ASR robustness, classic result**: with a recogniser at 41% WER, dialogue-act classification error rose by only about 21% relative to using reference transcripts — Stolcke et al. (2000), "Dialogue Act Modeling for Automatic Tagging and Recognition of Conversational Speech", [arXiv cs/0006023](https://arxiv.org/pdf/cs/0006023)
- **Newer work on dialogue-act classification from ASR output**: "Context-aware Neural-based Dialog Act Classification on Automatically Generated Transcriptions" ([arXiv 1902.11060](https://arxiv.org/pdf/1902.11060)) and RASwDA, a re-aligned Switchboard corpus for dialogue-act prediction (IWSDS 2024, [Columbia PDF](https://www.cs.columbia.edu/speech/PaperFiles/2024/iwsds24_raswda_paper.pdf)). I saw titles only and did not extract numbers.
- **Domain caution**: on TalkMoves classroom transcripts, a Switchboard-trained dialogue-act tagger labelled 77.2% of utterances "statement-non-opinion" and only 0.457% "Yes-No-Question" — [TalkMoves LREC PDF](https://aclanthology.org/2022.lrec-1.497.pdf)

### Inferences
- For this project, a generic dialogue-act model mainly adds an "is a question / directive / backchannel" signal, which the education-specific question and invitation classifiers in §3 and §5 already cover better. `diwank/silicone-deberta-pair` is the most useful generic option because it separates `backchannel` and `acknowledge` from `ask`. That helps filter "mm-hmm" and "okay" turns that reset the 60 s window without inviting responses.
- MRDA (multi-party meetings) is closer in structure to a seminar than SwDA (dyadic phone calls) or DailyDialog (scripted, written). No well-documented MRDA checkpoint was found, though SILICONE includes MRDA if the student wanted to fine-tune.

### Gaps
- No HF dialogue-act checkpoint was found with documented accuracy on **ASR** text or on multi-party meeting or classroom speech.
- The W&B metrics for `diwank/silicone-deberta-pair` were not checked, because they live in an external report.

---

## 5. Question detection robust to ASR transcripts (and Deepgram punctuation)

### Takeaway
There are three complementary text-only signals: (a) Deepgram's own `?` from smart_format; (b) a learned question classifier that does not rely on punctuation, such as Stanford's Switchboard-trained `stanford-nlpxed/question-detection` (BERT-base, about 96.8% validation accuracy on Switchboard, no history); (c) the EduBehaviors `has_a_question` and `invites_participation` classifiers. Classroom ASR work (Kelly et al. 2018) shows that useful question measures survive very noisy ASR at the aggregate level.

### Cited Findings
- **`stanford-nlpxed/question-detection`**:
  - GPL licence, custom `MultiHeadModel` on `bert-base-uncased`, 438 MB
  - Same training run as `ddemszky/supervised_finetuning_hist0_is_question_switchboard_question_detection.json_bs32_lr0.000063` (identical files)
  - `command_args.json`: `dataset_path: data/switchboard_question_detection.json`, `max_history: 0`, head `is_question` (2 classes), 3 epochs, lr 6.25e-5
  - `val_results.txt`: `is_question_accuracy` 0.661 before training, 0.967–0.968 after epochs 1–3
  
  Sources: [HF question-detection](https://huggingface.co/stanford-nlpxed/question-detection), [HF ddemszky version](https://huggingface.co/ddemszky/supervised_finetuning_hist0_is_question_switchboard_question_detection.json_bs32_lr0.000063); files read via HF raw endpoints
- **`mrsinghania/asr-question-detection`**: a "Question vs Statement classifier trained on more than 7k samples which were coming from spoken data in an interview setting". It is a small BERT (44.7 MB), with no licence and no metrics on the card — [HF card](https://huggingface.co/mrsinghania/asr-question-detection)
- **`shahrukhx01/question-vs-statement-classifier`**: about 11.2M params, 44.7 MB, roughly 98k downloads. It was trained to classify Haystack *search queries* as question vs keyword statement, so it is a domain mismatch. There is also `shahrukhx01/bert-mini-finetune-question-detection`. Neither has a licence — [HF card](https://huggingface.co/shahrukhx01/question-vs-statement-classifier)
- **`StanfordSCALE/assertion_sentence_has_a_question`**: base rate 19.0% and LLM-annotator α 0.867 on TalkMoves teacher utterances. A `has_a_rhetorical_question` variant also exists — [HF dataset card](https://huggingface.co/datasets/StanfordSCALE/assertions_llm_annotated_talkmoves), [HF StanfordSCALE](https://huggingface.co/StanfordSCALE)
- **Classroom ASR evidence**: Kelly, Olney, Donnelly, Nystrand & D'Mello (2018), "Automatically Measuring Question Authenticity in Real-World Classrooms", *Educational Researcher* — [NSF PAR PDF](https://par.nsf.gov/servlets/purl/10175684)
  - 132 sessions, 27 classrooms, 45,044 teacher utterances (median 2.26 s)
  - Microsoft Bing Speech reached only 53.3% word accuracy (61.5% ignoring order)
  - Authentic questions made up only 3.6% of utterances
  - Even so, classroom-level correlation between computer- and human-coded authenticity was r = .687 on ASR audio (r = .602 on archival transcripts), against human per-question agreement of 82%
- **Related earlier work**: Blanchard, Donnelly, Olney, Kelly, D'Mello et al., "Identifying Teacher Questions Using Automatic Speech Recognition in Classrooms" (37 class sessions, 11 teachers) — [ResearchGate](https://www.researchgate.net/publication/311990201_Identifying_Teacher_Questions_Using_Automatic_Speech_Recognition_in_Classrooms). Donnelly et al. (2017, LAK), "Words matter: automatic detection of teacher questions in live classroom discourse using linguistics, acoustics, and context", cited in [Demszky et al. 2021](https://aclanthology.org/2021.acl-long.130.pdf). Numbers were not retrieved.
- **Punctuation restoration as a fallback or second opinion**:
  - `1-800-BAD-CODE/punctuation_fullstop_truecase_english`: Apache-2.0, ONNX 209.5 MB, `punctuators` package. It reports `?` precision 76.6, recall 74.6, F1 75.6 on held-out **news** text, and warns it "may not perform well on conversational or informal data" — [HF card](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english)
  - `oliverguhr/fullstop-punctuation-multilang-large`: MIT, XLM-R-large with 559M params, trained on Europarl — [HF](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large)
- **Deepgram**: `smart_format=true` turns on punctuation, so `punctuate=true` is not needed separately — [Deepgram docs, Smart Formatting](https://developers.deepgram.com/docs/smart-format), via search snippet; the page returned HTTP 451 to the fetcher. Deepgram's own blog says punctuation including question marks is predicted from context and audio cues such as rising intonation — [Deepgram blog](https://deepgram.com/learn/speech-to-text-automatic-punctuation-guide). This is vendor material, and no `?` accuracy figure is published.

### Inferences
- Deepgram's `?` uses audio prosody that the student otherwise lacks, so it is probably the single best cheap question cue. It is not reliable for statement-shaped invitations ("Tell me what you think about X", "Someone walk us through…"), which the `invites_participation` and directive classifiers can catch. It also misses questions whose rising intonation is weak.
- The Switchboard question model was trained with `max_history: 0` on Switchboard text. Switchboard transcripts are conventionally lowercase with limited punctuation, so it plausibly relies on lexical and syntactic cues (wh-words, inversion) rather than `?`. That would make it robust to missing ASR punctuation. This is an inference, not verified from the training data.
- A sensible ensemble feature is `max(p_deepgram_qmark, p_question_model, p_has_a_question)`, plus `p_invites_participation`, computed on the professor's most recent completed sentence and decayed by time since it ended.

### Gaps
- No published accuracy for Deepgram smart_format question marks, and no evaluation of any of these question classifiers on Deepgram output, was found.
- Precision and recall for `stanford-nlpxed/question-detection` on a held-out set, beyond the val accuracy, and its class balance are not documented.

---

## 6. How classroom-discourse features have been used to predict student participation or engagement

### Takeaway
I found **no published work that predicts an individual student's hand raise (or turn bid) from transcript text in real time**. The existing literature uses discourse features *descriptively*: it correlates teacher moves with participation, attendance and outcomes, or uses talk-time feedback to change participation. The closest methodological analogue is FairTalk (2025): it predicts turn-grabbing intention in video calls using positive-unlabelled learning with detected turn-takes as positives, which mirrors "hand-raise icon = positive label". Its modalities are not stated in the abstract.

### Cited Findings
- **Talk moves and engagement in an online course**: in Code in Place sections, instructor accountable-talk moves correlated positively with section attendance, ratings and assignment completion, with "connecting" student ideas the strongest — [Kupor et al. 2023, arXiv 2311.10749](https://arxiv.org/abs/2311.10749)
- **Talk-time feedback RCT**: Demszky et al., LAK 2024, "Does Feedback on Talk Time Increase Student Engagement? Evidence from a Randomized Controlled Trial on a Math Tutoring Platform". 742 tutors were in the trial. A "Talk Meter" gave real-time talk-ratio feedback. The student-facing version was more motivating, especially for introverted students, and students started verbalising steps — [ACM DL](https://dl.acm.org/doi/fullHtml/10.1145/3636555.3636924), [ERIC PDF](https://files.eric.ed.gov/fulltext/ED638947.pdf)
- **Uptake feedback RCT**: Demszky, Liu, Hill, Jurafsky & Piech (2024), "Can Automated Feedback Improve Teachers' Uptake of Student Ideas? Evidence From a Randomized Controlled Trial in a Large-Scale Online Course", *Educational Evaluation and Policy Analysis* — [SAGE](https://journals.sagepub.com/doi/abs/10.3102/01623737231169270). Only the title and venue were retrieved; effect sizes were not.
- **Student talk ratio and outcomes**: in NCTE, the share of student turns and words did not predict observation scores or value-added, except student word % with "instructional dialogue". Dialogic moves (uptake, focusing questions, student reasoning) did predict them — [NCTE paper](https://arxiv.org/pdf/2211.11772)
- **Forecasting talk moves**: Ganesh, Palmer & Kann (2021), "What Would a Teacher Do? Predicting Future Talk Moves", Findings of ACL 2021, poses *future* talk-move prediction from dialogue history — [ACL Anthology](https://aclanthology.org/2021.findings-acl.418/), [PDF](https://aclanthology.org/2021.findings-acl.418.pdf). The task details and numbers were not extracted.
- **FairTalk**: Iijima, Yoshida, Hashimoto & Ma (2025), "FairTalk: Facilitating Balanced Participation in Video Conferencing by Implicit Visualization of Predicted Turn-Grabbing Intention". A model trained on web-collected videoconference data with positive-unlabelled learning, using turn-taking detection as automatic positive labels, predicts who wants to speak. The user study suggested better speaking balance but no significant perceived impact — [arXiv 2505.20138](https://arxiv.org/abs/2505.20138)
- **Hand-raising as engagement**: hand-raising frequency has been used as an observable indicator of behavioural engagement and related to self-reported learning activities. These studies use video or skeleton detection — [ResearchGate: "Student Hand-raising As an Indicator of Behavioral Engagement and Its Role in Classroom Learning"](https://www.researchgate.net/publication/342415351_Student_Hand-raising_As_an_Indicator_of_Behavioral_Engagement_and_Its_Role_in_Classroom_Learning), via search snippet only
- **Minerva Forum context**: Forum logs "overall student-instructor talk time, reactions (emojis), hand-raises, chats, and individual student talk-time". Instructors use them "to identify students who may be struggling to participate". Forum "can automatically record classes, generate transcripts" — [Minerva blog](https://www.minerva.edu/blog/a-teachers-guide-how-to-use-data-to-measure-student-engagement/)

### Inferences
- Because no prior text-based hand-raise predictor exists, the education classifiers above are best framed as *engineered covariates* inside the student's own temporal model (GBM, logistic or sequence model), not as transfer targets. Fine-tuning a TalkMoves or EduBehaviors encoder end-to-end on "hand raise within the next k seconds" is feasible, but with a few hundred to a few thousand positives a frozen-feature approach is safer.
- The Forum TalkTime context suggests a confound: professors may cold-call low-talk-time students, which would change the hand-raise base rate. The student's own cumulative talk time in the session, available from speaker tags, is probably a strong non-NLP feature, consistent with the participation literature above.

### Gaps
- No study was found linking real-time teacher-question detection to *subsequent* student volunteering or hand raises at second-level resolution.
- The FairTalk modalities and performance were not available from the abstract.

---

## 7. Practical: model sizes, CPU/Apple-MPS throughput for ~100k utterances, licence restrictions

### Takeaway
Every recommended feature model is a BERT-base-class encoder (about 110M params, about 440 MB) or smaller. At published Apple-Silicon GPU throughputs (thousands of words per second), 100k utterances take minutes per model on an M-series Mac, so throughput is not the bottleneck. Licences are mostly non-commercial or copyleft, which is fine for coursework. The uptake model's no-redistribution clause and the NC-SA data terms mean: download weights at runtime, don't commit them, and attribute.

### Cited Findings
**Inventory** (sizes and licences from the HF API, 2026-10-12; links in earlier sections)

| Model ID | What it gives you | Arch / size | Licence (as stated) |
|---|---|---|---|
| `StanfordSCALE/assertion_sentence_invites_participation` (+48 sibling assertions) | P(utterance invites participation); also has_a_question, calls_on_student_by_name, addresses_the_whole_class… | SetFit, mpnet-base, 109M, 438 MB each | none stated (data from CC BY-NC-SA TalkMoves) |
| `stanford-nlpxed/question-detection` | P(is question), Switchboard-trained, no context | BERT-base MultiHeadModel, 438 MB | GPL |
| `stanford-nlpxed/focusing-questions` | P(teacher focusing question) | BERT-base (config), 438 MB | CC BY-NC 3.0 |
| `stanford-nlpxed/uptake-model` | uptake score for a (student → teacher) pair | BERT-base MultiHeadModel, 438 MB | CC BY-NC-ND 4.0 + no-share terms |
| `stanford-nlpxed/student-reasoning` | P(student utterance gives reasoning), ≥8 words | BERT-base (config), 438 MB | GPL |
| `YaHi/teacher_electra_small` / `YaHi/student_electra_small` | 7-way / 5-way TalkMoves | ELECTRA-small, 54 MB | Apache-2.0 |
| `ahoang/electra-base-discriminator-talkmoves-{teacher,student}` | TalkMoves (undocumented) | ELECTRA-base, 110M, 438 MB | none stated |
| `StanfordSCALE/tutor_talkmoves_{roberta_large,modernbert_base}_classifier` | tutoring talk moves (3 + none) | 355M / 149M | none stated |
| `diwank/silicone-deberta-pair` | 11 generic dialogue acts (ask, ask_yes_no, backchannel…) | DeBERTa-base, 557 MB | MIT |
| `WSHAPER/dialogue-act-classifier-fp16` | 4 dialogue acts incl. question (DailyDialog) | DistilBERT FP16 ONNX, 134 MB | Apache-2.0 |
| `mrsinghania/asr-question-detection` | question vs statement (spoken interview data) | small BERT, 44.7 MB | none stated |
| `shahrukhx01/question-vs-statement-classifier` | question vs keyword query (search) | BERT-mini, 11M | none stated |
| `1-800-BAD-CODE/punctuation_fullstop_truecase_english` | punctuation incl. `?` (news-trained) | ONNX, 209.5 MB | Apache-2.0 |

- **Throughput anchors**:
  - Explosion (spaCy) measured transformer-pipeline inference on an M1 Max at up to **8,648 words/s on the Apple GPU (Metal/MPS) vs 1,821 words/s on CPU**, a speed-up of up to 4.7x — [Explosion blog](https://explosion.ai/blog/metal-performance-shaders)
  - An MLX benchmark on an M1 MacBook Pro measured single-input BERT-base-uncased latency of 76 ms on the GPU vs 227 ms on the CPU for 50-character inputs, and 99 ms vs 286 ms for 100-character inputs — [arXiv 2510.18921](https://arxiv.org/abs/2510.18921), via search snippet
- **Edu-ConvoKit** was designed to run on CPU and calls each HF model with a minimum-word filter: uptake needs ≥5 words in the student turn, student reasoning ≥8, focusing 0 — [Edu-ConvoKit paper](https://arxiv.org/abs/2402.05111), [constants.py](https://github.com/EduNLP/edu-convokit/blob/main/edu_convokit/constants.py)
- **Loading quirks**: the uptake and question-detection checkpoints use a custom `MultiHeadModel` class, so they need the repo's `utils.py`/`handler.py` (Edu-ConvoKit's `uptake_utils.py` vendors this) and do not load with `AutoModelForSequenceClassification`. The SetFit models need the `setfit` package or `edubehaviors-kit` — [HF uptake files](https://huggingface.co/stanford-nlpxed/uptake-model/tree/main), [Edu-ConvoKit repo](https://github.com/EduNLP/edu-convokit)
- **Data licences**: TalkMoves is CC BY-NC-SA 4.0 ([GitHub](https://github.com/SumnerLab/TalkMoves)). NCTE needs a Google Form for access, the repo is MIT, and the metadata are on ICPSR #36095 ([GitHub](https://github.com/ddemszky/classroom-transcript-analysis)). The funneling/focusing data are MIT ([GitHub](https://github.com/sterlingalic/funneling-focusing)).

### Inferences
- **Throughput estimate (mine, not measured; torch is not installed in the project's `.venv`, so I could not benchmark on the M4)**:
  - 100k utterances × about 15–25 words is about 1.5–2.5M words
  - At the M1 Max GPU rate (about 8.6k words/s) that is about 3–5 min per BERT-base-class model on MPS, and about 15–25 min on CPU (about 1.8k words/s)
  - An M4 should be at least comparable
  - Running 5–8 feature models is therefore well under an hour total and only needs doing once (cache the per-utterance scores to parquet)
- Running all 49 EduBehaviors models costs about 49 forward passes, because each SetFit body is separately fine-tuned. Pick the 5–10 relevant ones. Alternatively, embed once with the base `paraphrase-mpnet-base-v2` and train a light head on the student's own labels.
- **Leakage and causality**: every feature must be computed only on utterances (or sentence prefixes) **completed before** the decision time t. Pair-based models such as uptake and TalkMoves sentence pairs must use only already-finished pairs. Per-utterance scores can be precomputed offline once and then joined on end-timestamp ≤ t, so this adds no online cost.
- **Licence handling for a CS156 submission**: non-commercial academic use is allowed under all of the above. Do not redistribute the uptake weights (ND plus the explicit no-share clause), and avoid committing any weights. GPL on model weights is legally ambiguous; citing and downloading at runtime sidesteps it.

### Gaps
- No direct CPU or MPS benchmark of these specific checkpoints on an Apple M4 was found, and I could not run one locally because torch is not installed in the project environment.
- The licences of the EduBehaviors models and dataset, the `ahoang` and `StanfordSCALE` talk-move models, and several question classifiers are **unstated** on HF. The intended terms would need confirming with the authors.
