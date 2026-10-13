# Time-to-event, survival and temporal point process models for predicting the next hand raise

Scope: methods for predicting when one student raises his hand in a live seminar, scored at decision points about every 2 s using only information available at that moment. Covers discrete-time hazards, Cox and recurrent-event models, Hawkes and neural TPPs, early detection, evaluation, and what can realistically be built in 4 days (deadline Fri 16 Oct 2026, Apple M4). Checked 12 Oct 2026. Library facts come from the PyPI JSON API and the GitHub REST API, queried that day.

## Q1. Discrete-time hazard models: how they relate to the "within H seconds" classifier, and how they handle recurrent events

### Takeaway
A person-period (here class-period) logistic regression on the 2-s decision grid is a discrete-time hazard model. If it also includes features of the student's own past raises, it is a discrete-time self-exciting or self-inhibiting point process with covariates (the "point-process GLM" from neuroscience). The windowed "raise within H s" classifier with H = 2 s (one step) is the same model. With H > 2 s it becomes the *landmarking* approach to dynamic prediction (van Houwelingen 2007), which is a legitimate, published alternative to the hazard model. It is not a hack.

### Cited Findings
- Singer & Willett (1993) gave a single-spell discrete-time survival analysis. It defines hazard and survivor probabilities, derives the discrete-time likelihood, and fits it by maximum likelihood using standard logistic regression. Singer & Willett (2003, *Applied Longitudinal Data Analysis*) covers both discrete- and continuous-time models. — [Willett's discrete-time survival paper list](https://gseacademic.harvard.edu/~willetjo/dsta.htm); [ALDA discrete-time chapter page](https://alda.gse.harvard.edu/discrete-time-survival-analysis)
- The person-period method: convert one-record-per-subject data into one record per subject per period, then model the log-odds of the event in each period with logistic regression, because the outcome is binary. Tutz & Schmid, *Modeling Discrete Time-to-Event Data*, deals only with discrete-time models. — [USU graduate report summarising the literature](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2539&context=gradreports); [ERIC: Interpreting significant discrete-time periods (1994)](https://files.eric.ed.gov/fulltext/ED383767.pdf)
- Truccolo, Eden, Fellows, Donoghue & Brown (2005, *J. Neurophysiol.* 93:1074–1089) model the conditional intensity of a spike train as a function of three things: the neuron's own spiking history, ensemble activity, and extrinsic covariates. The log conditional intensity is linear in functions of the covariates. Fitting uses the *discrete-time* point-process likelihood with local Bernoulli or Poisson models, inside the standard GLM framework. — [Truccolo et al. 2005 (PDF)](https://web.math.ku.dk/~susanne/BSc/TruccoloEdenFellowsDonoghueBrown2005.pdf); [Springer encyclopedia entry on GLMs for point processes](https://link.springer.com/rwe/10.1007/978-1-4614-7320-6_393-1)
- Landmarking (van Houwelingen 2007, *Scand. J. Stat.* 34(1):70–85) adjusts predictions during follow-up by fitting models directly to the individuals still at risk at each landmark time. This lets simple proportional-hazards models handle time-varying effects and time-dependent covariates. — [Wiley abstract](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9469.2006.00529.x)
- The landmark "supermodel" fixes a prediction window w, picks a set of landmark (assessment) points, and stacks them into one prediction dataset. — [dynamicLM tutorial: landmark supermodel theory](https://github.com/thehanlab/dynamicLM/blob/main/tutorials/theory-landmark-supermodel.md); [comparative study of dynamic survival methods (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11876041/)
- For neural discrete-time survival, Kvamme & Borgan (2019) found that parameterising the discrete-time *hazard* performed slightly better than parameterising the PMF. For smaller datasets, cutting the time grid at quantiles of the event-time distribution beat an equidistant grid. — [arXiv 1910.06724](https://arxiv.org/abs/1910.06724)
- Nnet-survival (Gensheimer & Narasimhan, arXiv 2018) is a discrete-time survival model for neural networks. It is trained by maximum likelihood with minibatch SGD, and both the baseline hazard and covariate effects can vary with follow-up time. — [arXiv 1805.00917](https://arxiv.org/abs/1805.00917)

### Inferences
- **Exact relationship.** Let h_k = P(raise in step k | no raise yet in the current spell, information up to k). Then P(raise within the next H s) = 1 − ∏_{j=1..H/Δ}(1 − h_{t+j}). This identity needs the covariates for steps t+1 … t+H, which include the future transcript, so **computing it exactly is non-causal**. There are two causal options:
  - (a) Freeze the covariates at their time-t values and step the history features forward deterministically, assuming no new raise. This is an approximation.
  - (b) Fit a direct landmark model for each H. That is the windowed classifier.
- The windowed classifier is therefore a landmark model with horizon H. Its rows at consecutive 2-s landmarks share overlapping labels, so they are heavily autocorrelated. Treat each class as the unit for cross-validation (GroupKFold by class) and for any standard errors. The hazard (H = 2 s) dataset has no label overlap.
- **Recurrent raises fit naturally.** The student stays in the risk set after each raise, so each class is one long series of 2-s rows. History features carry the recurrence structure: log time since last raise, raises so far this class, exponentially decayed sums of past raises at several time constants, and his usual rate computed only from *earlier* classes. This is the Truccolo-style point-process GLM. Unlike a classical Hawkes model it allows inhibition (negative coefficients, e.g. a refractory dip after a raise) and a soft cap (a negative count effect).
- Rows inside a raise episode: drop decision points while the hand is still up (he is not "at risk" of a new raise then). Rows right after the professor calls on him may also need special handling. Neither point is covered by the sources above; they are design decisions.
- With 2-s bins and roughly 1–5% positives, the logit, complementary log-log and Poisson links will give nearly the same results. The cloglog link is the textbook grouped-time proportional-hazards link. That is standard theory, not re-verified in this session.

### Gaps
- I did not get the Tutz & Schmid book page itself. It is cited secondhand, and the USU report gives the year as 2018, which may not match the Springer publication year. Check before citing a year.
- I found no paper that formally compares a stacked landmark classifier with a discrete-time hazard GLM on high-frequency recurrent behavioural events like this one. Evidence on which works better here is thin and will have to be settled empirically.

## Q2. Survival models with covariates (Cox with time-varying covariates, Andersen–Gill and PWP, gradient-boosted survival, DeepSurv and DeepHit): do they fit recurrent events within a session?

### Takeaway
The classical recurrent-event Cox models (Andersen–Gill, PWP) describe the structure correctly: counting-process rows, gap time or total time, stratification by event number. They are built for *inference about covariate effects*, though, not online forecasting. lifelines' time-varying Cox model explicitly cannot forecast forward without future covariate values. Off-the-shelf ML survival models (scikit-survival gradient boosting, DeepSurv, pycox) assume one row and at most one event per subject, so they do not fit this problem without reformulation. Dynamic-DeepHit handles longitudinal covariates but is a research codebase and too heavy for 4 days.

### Cited Findings
- Amorim & Cai (2015, *Int. J. Epidemiol.* 44(1):324–333) is a tutorial on recurrent time-to-event data. It covers the Andersen–Gill (AG) model, PWP total-time (PWP-TT) and PWP gap-time (PWP-GT), marginal means/rates models, frailty models and multi-state models. AG extends the Cox model to repeated events. A frailty adds a random effect that captures unmeasured heterogeneity between units. — [Tutorial (academia.edu copy)](https://www.academia.edu/45059330/Modelling_recurrent_events_a_tutorial_for_analysis_in_epidemiology); [Mendeley record](https://www.mendeley.com/catalogue/b370d290-c469-380c-affd-d4d82a259d5b/)
- lifelines `CoxTimeVaryingFitter` takes long-format data (id, start, stop, event, one row per covariate change). The model is h(t|x) = b0(t)·exp(Σβi(xi(t) − x̄i)). Helpers include `to_long_format`, `add_covariate_to_timeline` (with optional cumulative sums and a *delay* option to reduce reverse causation) and `covariates_from_event_matrix`. — [lifelines docs: time-varying survival regression](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html)
- On prediction, lifelines says: "To predict, we would need to know the covariates values beyond the observed times, but if we knew that, we would also know if the subject was still alive or not!" It can compute hazards at observed times but cannot forecast survival forward. — [lifelines docs](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html)
- The lifelines docs warn that covariates can be affected by the upcoming event (reverse causation) and recommend lagging covariates. — [lifelines docs](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html)
- The scikit-survival evaluation guide does not mention time-varying covariates or recurrent events. Its estimators and metrics are written for one survival time per sample. — [scikit-survival user guide: evaluating survival models](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)
- DeepSurv (Katzman et al., arXiv 2016, ICML Computational Biology workshop) is a Cox proportional-hazards deep network for static patient covariates, aimed at treatment recommendation. — [arXiv 1606.00931](https://arxiv.org/abs/1606.00931)
- Kvamme, Borgan & Scheel (2019, *JMLR* 20(129)) extend Cox regression with neural networks, including non-proportional variants, using a nested case-control loss. The code is the pycox package. — [arXiv 1907.00825](https://arxiv.org/abs/1907.00825)
- Dynamic-DeepHit (Lee, Yoon & van der Schaar, *IEEE TBME* 67(1):122–133, 2019) uses the full history of repeated longitudinal measurements to issue dynamically updated survival predictions under competing risks. It was demonstrated on yearly follow-ups of 5,883 cystic-fibrosis patients. — [Semantic Scholar](https://www.semanticscholar.org/paper/Dynamic-DeepHit:-A-Deep-Learning-Approach-for-With-Lee-Yoon/293d91a1d4b3f296cb040a766bf7ab31197b1a8f); [code](https://github.com/chl8856/Dynamic-DeepHit)

### Inferences
- **AG maps to this problem as follows.** The "subject" is a class session, rows are 2-s start/stop intervals, the event is a raise, and history covariates (count so far, time since last raise) carry the dependence. PWP-GT is the closer analogue to the "time since last raise" clock: it stratifies the baseline hazard by raise number (1st, 2nd, 3rd…). A per-class frailty would absorb day-to-day variation in his engagement. In discrete time, all of these reduce to the logistic hazard GLM from Q1 with suitable features. That is the shippable form.
- Use lifelines `CoxTimeVaryingFitter` at most as an interpretable cross-check on effect directions, e.g. "does a raise in the last 60 s lower the hazard?". It is not the prediction engine, because of the forecasting caveat quoted above.
- Gradient-boosted survival in scikit-survival, DeepSurv and DeepHit predict a whole survival curve from one static covariate vector. For a stream they would need a landmark reformulation (one sample per decision point, censored at the horizon), and at that point a plain binary classifier or boosted hazard is simpler and equivalent. A boosted *per-bin hazard* (XGBoost or LightGBM `binary:logistic` on the person-period table) gets the nonlinearity without any survival library.
- **Non-causal trap specific to this project.** The transcript right after the icon appears ("Brady, go ahead") is caused by the raise. It must never leak into features for decision points at or before the raise. Check the end timestamps of transcript segments against the label time, including ASR timestamp latency. This is exactly the reverse-causation issue lifelines warns about.

### Gaps
- Dynamic-DeepHit's code is a 2019-era research repo. I did not check whether it runs on current Python or PyTorch, or on Apple Silicon.
- I found no source testing AG, PWP or frailty models on within-session behavioural events (e.g. classroom participation). How well they transfer is unverified.

## Q3. Hawkes processes and neural TPPs, and the state of their libraries in 2026 (maintenance, Apple Silicon)

### Takeaway
Classical Hawkes models can only *excite*. Their positivity constraints rule out the inhibition and soft cap expected here. Neural TPPs (Neural Hawkes, THP, intensity-free) are flexible but are built and benchmarked on event-history-only sequences. Library support in Oct 2026 is better than expected: tick was revived in Apr–May 2026 with macOS arm64 wheels, and EasyTPP is active and PyTorch-only. Several original reference repos are archived or dormant.

### Cited Findings
- **Neural Hawkes** (Mei & Eisner, NIPS 2017; arXiv Dec 2016): a continuous-time LSTM drives event intensities, so past events can excite *or inhibit* future ones. The paper says: "The positivity constraints in the Hawkes process limit its expressivity. First, the positive interaction parameters αj,k fail to capture inhibition effects…" It also proposes a decomposable "Hawkes process with inhibition", which allows negative α and passes the total activation through a softplus. Between events, intensities decay toward a steady state. — [arXiv 1612.09328](https://arxiv.org/abs/1612.09328)
- **Transformer Hawkes Process** (Zuo, Jiang, Li, Zhao, Zha; ICML 2020) uses self-attention for long-range dependencies and reports better likelihood and event-prediction accuracy than RNN-based TPPs. — [arXiv 2002.09291](https://arxiv.org/abs/2002.09291)
- **Intensity-free TPP** (Shchur, Biloš, Günnemann; ICLR 2020) models the conditional inter-event time distribution directly with normalising flows or a log-normal mixture. The mixture allows closed-form sampling and moments. — [arXiv 1909.12127](https://arxiv.org/abs/1909.12127)
- **Review** (Shchur, Türkmen, Januschowski, Günnemann; IJCAI 2021). A neural TPP has three parts: event feature vectors (time features plus mark embeddings), a history encoder (RNN or self-attention), and a conditional distribution (intensity, density, CDF, survival function or cumulative hazard). The review notes that claims about self-attention encoders "need more thorough evaluation". — [IJCAI 2021 PDF](https://www.ijcai.org/proceedings/2021/0623.pdf); [arXiv 2104.03528](https://arxiv.org/abs/2104.03528)
- **EasyTPP** (Xue et al., ICLR 2024) is a unified benchmark and library. Models implemented: RMTPP (KDD'16), NHP (NeurIPS'17), FullyNN (NeurIPS'19), SAHP (ICML'20), THP (ICML'20), IntensityFree (ICLR'20), ODETPP (ICLR'21), AttNHP (ICLR'22), WSMTHP (listed as NeurIPS'24) and S2P2 (NeurIPS'25). TensorFlow support was removed in Nov 2025, leaving PyTorch only. The README lists a news item dated "04-10-2026" adding WSMTHP (date format ambiguous). Data uses the "Gatech" format (time since start, time since last event, event type), and the README does not mention per-event external covariates or Apple MPS. — [GitHub README](https://github.com/ant-research/EasyTemporalPointProcess); [arXiv 2307.08097](https://arxiv.org/abs/2307.08097)
- **easy-tpp on PyPI**: latest 0.3.0, uploaded 2026-07-13, pure-Python wheel. GitHub repo last pushed 2026-09-29, 4 open issues, Apache-2.0. — [PyPI easy-tpp](https://pypi.org/project/easy-tpp/); [GitHub](https://github.com/ant-research/EasyTemporalPointProcess)
- **tick** (Hawkes, survival and GLM tools in C++/Python) had no release from 0.7.0.1 (May 2020) until 0.8.0.0 (2026-04-18). v0.8.0.2 (2026-05-04) ships wheels for "macOS arm64 + x86_64… for CPython 3.11/3.12/3.13/3.14" and requires Python ≥ 3.11. Repo last pushed 2026-06-15, 86 open issues. — [tick releases](https://github.com/X-DataInitiative/tick/releases); [PyPI tick](https://pypi.org/project/tick/)
- **tick Hawkes learners**: `HawkesExpKern`, `HawkesSumExpKern`, `HawkesADM4`, `HawkesSumGaussians` (parametric) and `HawkesEM`, `HawkesBasisKernels`, `HawkesConditionalLaw`, `HawkesCumulantMatching` (non-parametric). Simulators include `SimuHawkes` and `SimuInhomogeneousPoisson`. Parametric exponential and sum-of-exponential kernels support least-squares and log-likelihood goodness of fit. The module docs do not mention covariates or exogenous inputs. — [tick Hawkes docs](https://x-datainitiative.github.io/tick/modules/hawkes.html)
- **Status of other libraries as of 2026-10-12 (PyPI and GitHub APIs)**:
  - lifelines 0.30.3 (2026-03-05; pure Python; Python ≥ 3.11; repo pushed 2026-03-07; 305 open issues). — [PyPI](https://pypi.org/project/lifelines/); [GitHub](https://github.com/CamDavidsonPilon/lifelines)
  - scikit-survival 0.28.0 (2026-07-05; macOS arm64 wheels for cp311–cp314; repo pushed 2026-10-11). — [PyPI](https://pypi.org/project/scikit-survival/); [GitHub](https://github.com/sebp/scikit-survival)
  - pycox 0.3.0 (2024-09-04), with the repo last pushed the same day, so dormant for about 2 years; 96 open issues. — [PyPI](https://pypi.org/project/pycox/); [GitHub](https://github.com/havakv/pycox)
  - torchsurv 0.2.0 (Novartis; 2026-06-17; repo pushed 2026-08-27). — [PyPI](https://pypi.org/project/torchsurv/); [GitHub](https://github.com/Novartis/torchsurv)
  - auton-survival 0.1.0 (2023-10; repo pushed 2024-04), dormant. — [PyPI](https://pypi.org/project/auton-survival/)
  - statsmodels 0.15.0 (2026-08-30; arm64 wheels). — [PyPI](https://pypi.org/project/statsmodels/)
  - xgboost 3.4.1 (2026-08-15; macOS arm64 wheel; Python ≥ 3.12). — [PyPI](https://pypi.org/project/xgboost/)
  - Dead Hawkes packages: pyhawkes 0.3.1 (2017, source only) and hawkeslib 0.2.2 (2019; macOS x86_64 cp36 wheel only). — [PyPI pyhawkes](https://pypi.org/project/pyhawkes/); [PyPI hawkeslib](https://pypi.org/project/hawkeslib/)
- **Original reference implementations**: ifl-tpp (intensity-free) is archived, last push 2021-02. neurawkes last pushed 2021-09. The THP repo last pushed 2024-04. anhp-andtt (AttNHP) is archived, last push 2024-12. — [ifl-tpp](https://github.com/shchur/ifl-tpp); [neurawkes](https://github.com/hongyuanmei/neurawkes); [THP](https://github.com/SimiaoZuo/Transformer-Hawkes-Process); [anhp-andtt](https://github.com/yangalan123/anhp-andtt)
- **Local check on this machine**: the project's `cs156-ml` Jupyter kernel runs CPython 3.12.4 on arm64. That satisfies the Python ≥ 3.11 or ≥ 3.12 requirements of tick 0.8, lifelines, scikit-survival and xgboost 3.4. (Read from `/Users/brady/Library/Jupyter/kernels/cs156-ml/kernel.json` and the venv interpreter; no web source.)

### Inferences
- Version and wheel data suggest tick will install from a binary wheel on an M4 under Python 3.12. I did **not** test-install it, so allow 15 minutes for that.
- A tick `HawkesExpKern` fitted on raise times only (one realization per training class) is a cheap "history-only" baseline. It cannot represent the expected refractory period or soft cap, because its kernels are excitatory. The discrete-time GLM with signed history coefficients covers the same ground and also allows inhibition.
- EasyTPP is the realistic way to try THP, NHP or intensity-free models without porting archived code. On a single student with one mark type, though, these models collapse to "model the gap distribution given past gaps", and they ignore the transcript (see Q4). They may gain little over a gap-time hazard while costing a day of data wrangling. The PyTorch models should run on CPU on an M4. I found no confirmation that EasyTPP supports the MPS backend.

### Gaps
- No source states whether tick's Hawkes learners enforce non-negative kernels in code. The module docs are silent. Mei & Eisner's critique applies to the classical Hawkes model in general.
- EasyTPP's README does not report dataset sizes or minimum data needs, so I could not establish how many sequences it needs for stable training.
- I did not install or import any of these packages on the M4. Apple Silicon compatibility here is inferred from wheel tags (tick, scikit-survival, statsmodels, xgboost) or from being pure Python on PyTorch (easy-tpp, pycox, lifelines, torchsurv).

## Q4. How TPP models bring in external covariates such as text embeddings

### Takeaway
Mainstream neural TPPs (RMTPP, NHP, THP, intensity-free) condition only on *past events*: their times plus categorical or real-valued marks. Between events the intensity evolves deterministically, so a signal like the live transcript, which changes between raises, is invisible to them unless engineered in. "Covariate TPP" and LLM-based TPP work exists (2024–2025) but is research code. The LLM-TPP papers use text *describing events*, not exogenous text streams. For this problem, a discrete-time hazard or GLM that reads the current covariates at every 2-s step is the simplest correct way to combine history effects with transcript features.

### Cited Findings
- In the neural-TPP framework, each event (t_j, m_j) becomes a feature vector from time features (inter-event time, its log, or trigonometric encodings) and mark features (embedding layer for categorical marks). An encoder turns past events into a history embedding. — [Shchur et al. 2021 review](https://www.ijcai.org/proceedings/2021/0623.pdf)
- In Neural Hawkes, the continuous-time LSTM state decays toward steady-state values between events. Intensities change only through event-triggered updates and those deterministic decays. — [arXiv 1612.09328](https://arxiv.org/abs/1612.09328)
- TransFeat-TPP (Meng et al., arXiv Jul 2024) argues that "occurrence time may not be the only relevant factor, other contextual data, termed covariates, may also impact the event evolution". It proposes a Transformer-based covariate TPP that also ranks covariate importance. — [arXiv 2407.16161](https://arxiv.org/abs/2407.16161)
- A 2025 survey (Zhou et al., arXiv 2501.14291) covers Bayesian, neural and LLM approaches to TPPs. The search summary mentions dual-LSTM designs in which one LSTM models the event sequence and another a time-series covariate, with the two history embeddings fused. I took this from the search summary and did not verify it in the full text. — [arXiv 2501.14291](https://arxiv.org/abs/2501.14291)
- TPP-LLM (arXiv Oct 2024) fine-tunes an LLM with parameter-efficient methods plus temporal embeddings, using the *textual descriptions of event types* in place of categorical marks. — [arXiv 2410.02062](https://arxiv.org/abs/2410.02062); [code](https://github.com/zefang-liu/TPP-LLM)
- Language-TPP (arXiv Feb 2025) turns continuous time intervals into byte tokens so a standard LLM can model event sequences, including generating event descriptions. — [arXiv 2502.07139](https://arxiv.org/abs/2502.07139)
- Truccolo et al. (2005) handle history and "extrinsic covariates" together in one discrete-time GLM intensity. — [Truccolo et al. 2005](https://web.math.ku.dk/~susanne/BSc/TruccoloEdenFellowsDonoghueBrown2005.pdf)

### Inferences
- There are three causal ways to feed the transcript to a TPP:
  1. Discretise time and use a GLM, logistic hazard or boosted hazard with transcript features at each step. This is easiest and best matched to the data.
  2. Treat each transcript utterance (or each 2-s tick) as a non-target "context event" of a second mark type that carries a text-embedding vector. The history encoder then updates whenever something is said. This needs a custom marked TPP and is not supported out of the box by EasyTPP's Gatech format.
  3. Use a covariate TPP such as TransFeat-TPP, which is research code.
  
  Options 2 and 3 are not realistic in 4 days.
- **Leak risk with embeddings.** Use only text whose end timestamp is before the decision time. Do not run a bidirectional encoder over a window that extends past t. Fit any PCA or normalisation of embeddings on training classes only.

### Gaps
- I found no paper evaluating neural TPPs with streaming *exogenous* text (as opposed to text marks on the events themselves) in live conversation or classroom data.
- I did not check whether TransFeat-TPP or TPP-LLM code is maintained or runs on Apple Silicon.

## Q5. Early event detection and early classification: literature and lead-time metrics

### Takeaway
"Early classification of time series" (ECTS) and "early event detection" (MMED) mostly assume either a fixed-length series that gets one label, or an event with duration that is detected before it ends. Neither matches an instantaneous event in an open-ended stream. The closest frameworks for this problem are **event forecasting with an alarm window**, as in seizure prediction (prediction horizon, occurrence period, false predictions per hour), and streaming-detection scoring (NAB, range-based precision and recall). Conversational turn-taking prediction (Voice Activity Projection) is the nearest behavioural analogue.

### Cited Findings
- ECTS: predict as early as possible to avoid a time penalty, but not so early that misclassification becomes likely. Renault, Bondu, Cornuéjols & Lemaire (TMLR 2025) split an ECTS system into a *prediction* component and a *decision* (when to trigger) component, give a principle-based taxonomy, and benchmark nine algorithms with an open-source library. — [arXiv 2406.18332](https://arxiv.org/abs/2406.18332)
- Cost-based ECTS casts the problem as minimising the expected misclassification cost at decision time plus a cost for delay. — [Achenchabe et al., arXiv 2005.09945](https://arxiv.org/pdf/2005.09945); [MLJ 2021 article](https://link.springer.com/article/10.1007/s10994-021-05974-z)
- Max-Margin Early Event Detectors (Hoai & De la Torre, CVPR 2012) use a structured-output SVM trained on partial events, simulating sequential arrival of the data, to detect events such as facial expressions and gestures before they complete. The authors describe it as the first learning formulation for early event detection in computer vision. — [paper PDF](https://www3.cs.stonybrook.edu/~minhhoai/papers/MMED_CVPR12.pdf); [project page](https://www3.cs.stonybrook.edu/~minhhoai/projects/mmed.html)
- The seizure prediction characteristic (Winterhalder et al., *Epilepsy & Behavior* 2003) is built on an alarm, a **seizure prediction horizon** (SPH: minimum time between alarm and the start of the occurrence window) and a **seizure occurrence period** (SOP: the window in which the event must fall). A prediction counts as correct only if the seizure occurs during the SOP and not during the SPH. Sensitivity is correct predictions divided by all seizures. Specificity is measured as a **false prediction rate per hour**, and methods are tuned for maximum sensitivity under a cap on that rate. — [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S1525505003001057); [author PDF](https://www.fdm.uni-freiburg.de/publications-preprints/papers/winterhalder_printed.pdf)
- The Numenta Anomaly Benchmark (Lavin & Ahmad, ICMLA 2015) scores streaming detectors so that the ideal detector finds every anomaly as soon as possible and raises no false alarms. It rewards early detection inside labelled windows and penalises false positives. — [arXiv 1510.03336](https://arxiv.org/abs/1510.03336)
- Range-based precision and recall (Tatbul et al., NeurIPS 2018) extend precision and recall from points to ranges, with tunable preferences such as rewarding early overlap. — [arXiv 1803.03639](https://arxiv.org/abs/1803.03639)
- Voice Activity Projection (Ekstedt & Skantze, Interspeech 2022) frames turn-taking as self-supervised prediction of the participants' near-future voice activity. It supports zero-shot prediction of upcoming turn shifts and backchannels, and argues that the dependency of events within the projection window must be modelled. — [ISCA archive](https://www.isca-archive.org/interspeech_2022/ekstedt22_interspeech.html); [multilingual follow-up, arXiv 2403.06487](https://arxiv.org/pdf/2403.06487)

### Inferences
- **Map the seizure framework onto this problem.**
  - Alarm: fires when p_H(t) or the hazard crosses a threshold.
  - SPH: the minimum useful lead time, which could be 0 s or a few seconds depending on the use case.
  - SOP: H.
  - Report: event-level sensitivity (fraction of raises preceded by an alarm within [t+SPH, t+SPH+SOP]), false alarms per class-hour, and the distribution of lead times for detected raises.
  
  This addresses "detect before it happens" more directly than per-decision-point AUROC, and it can be computed the same way for any model.
- "Earliness" in ECTS is defined relative to a series end. Here the analogue is lead time before the raise, which the alarm framework measures directly.
- Hand raises usually have no observable onset phase in the given features (no video or audio of the student), so MMED's partial-event idea applies only if precursors exist, such as unmuting or chat activity. These are not in the stated feature set.

### Gaps
- I did not verify VAP's exact projection-window length or bin structure.
- I found no published work on predicting online-classroom hand raises specifically. Evidence for the domain is thin.

## Q6. Evaluation: log-likelihood, time-dependent concordance, Brier score, event-level metrics, and fair comparison with a classifier

### Takeaway
Compare models on a **common output**: p_H(t), the probability of a raise within H s, at every decision point. Pair it with the per-2-s-bin log-likelihood or Brier score (the discrete point-process likelihood) and event-level alarm metrics. TPP-style NLL is unreliable on 2-s-gridded times, and MAE on next-event time is unfair to probabilistic models. Survival C-index and IPCW metrics assume one event per subject and need adapting for recurrent raises.

### Cited Findings
- NLL is the default training objective and a popular metric for TPPs. A single NLL mixes the time and mark components. It "can be 'fooled' on datasets where the arrival times are measured in discrete units, such as seconds", because a flexible model can put narrow density spikes on the discrete values. The review also calls NLL "mostly irrelevant as a measure of error in real-world applications". — [Shchur et al. 2021 review](https://www.ijcai.org/proceedings/2021/0623.pdf)
- MAE or MSE on the next inter-event time and mark accuracy "are even less suited" for TPPs. They score a single point estimate, which a simple point-estimate baseline can optimise directly, so "comparing them to point-estimate baselines using above metrics is unfair". The real advantage of a TPP is sampling whole future trajectories. — [Shchur et al. 2021 review](https://www.ijcai.org/proceedings/2021/0623.pdf)
- scikit-survival metrics:
  - `concordance_index_censored` (Harrell's C), which "has been shown that it is too optimistic with increasing amount of censoring".
  - `concordance_index_ipcw` (Uno's C).
  - `cumulative_dynamic_auc` (time-dependent AUC).
  - `brier_score` and `integrated_brier_score`, which need a predicted survival function.
  
  Evaluation time points must lie inside the observed range; the docs suggest an upper bound around the 80th percentile so censoring probabilities stay non-zero. — [scikit-survival docs](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)
- Kvamme et al. (2019) evaluate survival models mainly by Brier score and binomial log-likelihood over time. — [arXiv 1907.00825](https://arxiv.org/abs/1907.00825)
- Bosser & Ben Taieb (TMLR 2023) ran a large unified study of neural TPP predictive accuracy, varying event encoding, history encoder and decoder, and found that neural TPP *mark* distributions are often miscalibrated. — [arXiv 2306.17066](https://arxiv.org/abs/2306.17066)
- Goodness of fit for point-process GLMs can use time-rescaling, thinning or complementing procedures. — [Gerhard & Gerstner, arXiv 1011.4188](https://arxiv.org/html/1011.4188)
- Event-level and lead-time metrics: seizure-prediction sensitivity and false predictions per hour with SPH/SOP ([Winterhalder et al. 2003](https://www.sciencedirect.com/science/article/abs/pii/S1525505003001057)), NAB early-detection scoring ([arXiv 1510.03336](https://arxiv.org/abs/1510.03336)), and range-based precision and recall ([arXiv 1803.03639](https://arxiv.org/abs/1803.03639)).

### Inferences
- **Recommended protocol (my synthesis).**
  1. Every model outputs p_H(t) at each decision point, for each H in the sweep:
     - Windowed classifier: directly.
     - Hazard model: 1 − ∏(1 − ĥ) with frozen covariates.
     - Hawkes or TPP: 1 − exp(−∫ λ) assuming no new event, or by simulation.
  2. Score p_H with log loss, Brier score, PR-AUC (with ~1–5% positives) and a reliability curve, pooled over held-out classes and grouped by class for confidence intervals.
  3. Score the H = 2 s (single-bin) prediction with the per-bin Bernoulli log-likelihood. This is the discrete point-process likelihood, a proper score that every model can produce.
  4. Report event-level sensitivity at a fixed false-alarm rate per class-hour, plus lead-time distributions.
- **Fair-comparison caveats.**
  - Rows with overlapping H windows inflate apparent sample size, so all splits must be by class (or by date, if the classes have a time order).
  - The hazard model's p_H for H > 2 s uses frozen covariates, which handicaps it relative to a classifier trained directly for that H. Report both.
  - Thresholds must be chosen on validation classes, not test classes.
- C-index and time-dependent AUC fit poorly for recurrent events in an open stream. If used at all, apply them per gap: each inter-raise spell is one "subject", censored at class end, with risk scores fixed at the spell start. Otherwise prefer per-bin AUC.

### Gaps
- No single source gives an agreed protocol for comparing TPPs, hazard models and windowed classifiers on the same streaming task. The protocol above is my synthesis.
- I did not find quantitative evidence on how bad the frozen-covariate approximation is for p_H with exogenous text features.

## Q7. Small-data risks, and what is realistically shippable in 4 days

### Takeaway
With a few hundred to a few thousand raises, the evidence favours simple, well-specified hazard models over neural TPPs. The neural-TPP literature itself notes that simple history-independent baselines "solve" some standard benchmarks, that evaluation is inconsistent, and that mark distributions are miscalibrated. The realistic plan: a logistic discrete-time hazard GLM with history kernels and transcript features, a boosted-tree version of the same, and optionally a tick Hawkes baseline on event times only, all scored against the windowed classifier on the shared p_H protocol.

### Cited Findings
- The review notes "Lack of standardized experimental setups and high-quality benchmark datasets" and inconsistent baselines. Enguehard et al. showed two popular datasets (MIMIC-II, StackOverflow) "can be 'solved' by a simple history-independent baseline", and treating training sequences as i.i.d. may be inappropriate. — [Shchur et al. 2021 review](https://www.ijcai.org/proceedings/2021/0623.pdf); [Enguehard et al., ML4H@NeurIPS 2020, arXiv 2007.13794](https://arxiv.org/abs/2007.13794)
- Neural TPP mark distributions are frequently miscalibrated. — [Bosser & Ben Taieb, TMLR 2023](https://arxiv.org/abs/2306.17066)
- For smaller datasets in discrete-time neural survival, quantile-based time grids beat equidistant ones, and hazard parameterisation is slightly better than PMF. — [Kvamme & Borgan 2019](https://arxiv.org/abs/1910.06724)
- The point-process GLM framework fits history-dependent intensity with ordinary GLM software. — [Truccolo et al. 2005](https://web.math.ku.dk/~susanne/BSc/TruccoloEdenFellowsDonoghueBrown2005.pdf)
- Software that is maintained and has arm64 support in Oct 2026: statsmodels 0.15.0, xgboost 3.4.1, scikit-survival 0.28.0, tick 0.8.0.2, lifelines 0.30.3, easy-tpp 0.3.0. — [PyPI statsmodels](https://pypi.org/project/statsmodels/); [PyPI xgboost](https://pypi.org/project/xgboost/); [PyPI scikit-survival](https://pypi.org/project/scikit-survival/); [PyPI tick](https://pypi.org/project/tick/); [PyPI lifelines](https://pypi.org/project/lifelines/); [PyPI easy-tpp](https://pypi.org/project/easy-tpp/)

### Inferences
- **Small-data risks.**
  - With ~1–5% positives at 2-s steps, the effective sample size is the number of raises (hundreds to low thousands), not the number of rows. Keep the hazard GLM to a few dozen effective parameters: a spline or a few bins for log time-since-last-raise, 2–3 exponential-decay history kernels, count so far (capped), minute of class or segment, his usual rate from prior classes only, and a low-dimensional transcript projection (e.g. PCA to ~10–20 dimensions) or a handful of interpretable cues. Use L2 regularisation.
  - All model selection should use GroupKFold by class.
  - High-dimensional text embeddings fed raw into a boosted model will overfit at this scale.
- **Four-day plan (12–16 Oct).**
  - **Day 1.** Build the class-period table on the 2-s grid with causal features and labels at H = 2 s.
    - Baselines: (a) constant rate; (b) history-only logistic GLM, which is the discrete-time Hawkes or self-inhibiting analogue.
    - Check timestamp alignment of transcript against label so nothing leaks.
  - **Day 2.** Add transcript features to the GLM. Fit a LightGBM/XGBoost per-bin hazard on the same table with early stopping on grouped validation.
  - **Day 3.**
    - Convert to p_H for the H sweep with frozen covariates, and compare against the direct windowed classifiers (landmark models) on the shared metrics and the alarm or lead-time metrics.
    - Optional (≤ 3 h): tick `HawkesExpKern` on raise times only, fitted on training classes.
    - Optional (≤ 1 h): lifelines `CoxTimeVaryingFitter` on start/stop rows, to report interpretable history hazard ratios.
  - **Day 4.** Write-up.
- **Skip for this deadline:**
  - EasyTPP neural TPPs: no native exogenous covariates, extra data-format work, and expected small-data instability.
  - DeepSurv, DeepHit, Dynamic-DeepHit and pycox: single-event and static-covariate designs, a dormant library, or research code.
  - LLM-TPP methods.
  
  The write-up can mention them as future work, with the reasons above.
- **Non-causal checklist for the write-up.**
  1. Transcript segments must end before t.
  2. "Usual rate" and normalisers come from training classes only (strictly earlier classes, if the evaluation is chronological).
  3. Hazard-based p_H must not use future covariates.
  4. Hawkes or TPP parameters must not be fitted on the test class.
  5. Rows while the hand is already up should be excluded or flagged.

### Gaps
- I found no quantitative source on the minimum number of events needed to train neural TPPs reliably, or on how they compare with discrete-time GLMs at a few hundred to a few thousand events. The recommendation leans on the review's critique and general small-sample reasoning, not a head-to-head benchmark at this scale.
- I did not find events-per-variable guidance specific to discrete-time hazard models in the sources consulted.
