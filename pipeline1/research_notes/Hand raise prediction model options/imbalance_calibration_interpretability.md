# Imbalance handling, calibration, uncertainty, rare-event evaluation and interpretability for the hand-raise predictor

*Scope: decision points every ~2 s, 1–5% positive, a few hundred to a few thousand positive events, strongly autocorrelated neighbouring points, splits grouped by class session plus forward-chaining by date. Library versions checked on PyPI on 2026-10-12: scikit-learn 1.9.1 (released 2026-09-10), MAPIE 1.5.0 (2026-08-05), SHAP 0.53.0 (2026-10-09), LightGBM 4.7.0 (2026-07-18), imbalanced-learn 0.14.2 (2026-06-07) — [scikit-learn](https://pypi.org/project/scikit-learn/), [MAPIE](https://pypi.org/project/MAPIE/), [shap](https://pypi.org/project/shap/), [lightgbm](https://pypi.org/project/lightgbm/), [imbalanced-learn](https://pypi.org/project/imbalanced-learn/).*

## 1. Imbalance: class weights vs resampling (SMOTE etc.) vs focal loss vs threshold moving

### Takeaway
The best evidence says: train on the natural class distribution, rank by score, and pick the operating point afterwards (threshold moving to a target false-alarm rate). With strong learners like LightGBM, rebalancing does not improve discrimination. Every rebalancing trick (resampling, class weights, `scale_pos_weight`, focal loss) distorts the probabilities, so if one is used it must be followed by recalibration on untouched held-out sessions. SMOTE is a poor fit for text and sequence inputs, and it is a leakage hazard here.

### Cited Findings
- **van den Goorbergh et al. 2022 (JAMIA 29(9):1525–1534).** Tested random undersampling (RUS), random oversampling (ROS) and SMOTE with standard and ridge logistic regression, in Monte Carlo simulations and an ovarian-cancer case study. Abstract: "all imbalance correction methods led to poor calibration (strong overestimation of the probability to belong to the minority class), but not to better discrimination in terms of the area under the receiver operating characteristic curve. Imbalance correction improved classification in terms of sensitivity and specificity, but similar results were obtained by shifting the probability threshold instead. Our study shows that outcome imbalance is not a problem in itself." — [arXiv 2202.09101](https://arxiv.org/abs/2202.09101); [JAMIA](https://academic.oup.com/jamia/article/29/9/1525/6605096)
  - In the simulations, models on unadjusted data had median calibration intercepts between −0.05 and 0.03. With imbalance correction the median intercepts were "−4.5 or lower for scenarios with a 1% event fraction, −2.1 or lower for … 10%". — [arXiv 2202.09101](https://arxiv.org/pdf/2202.09101)
  - SMOTE, and to a lesser extent ROS, gave "probability estimates that were too extreme" (median calibration slopes below 1). This was worse at lower event fraction, smaller training sets and more predictors. — [arXiv 2202.09101](https://arxiv.org/pdf/2202.09101)
  - Recalibrating on the original (imbalanced) data brought the median intercepts back to between −0.07 and 0.03. The authors conclude that "class imbalance correction followed by recalibration is only worth the effort if imbalance correction leads to better discrimination." — [arXiv 2202.09101](https://arxiv.org/pdf/2202.09101)
  - At a 1% event fraction with 2,500 rows, RUS left ~25 events and 25 non-events, and the unpenalised LR failed to converge. A 0.5 threshold on unadjusted data gave a median sensitivity of 0%, which shows the problem is the threshold, not the model. — [arXiv 2202.09101](https://arxiv.org/pdf/2202.09101)
  - The authors flag that Random Forests and neural networks "remain to be investigated". They *anticipate* (speculation, untested) that miscalibration will persist "regardless of type of model or imbalance correction technique, unless the models are recalibrated". — [arXiv 2202.09101](https://arxiv.org/pdf/2202.09101)
- **Elor & Averbuch-Elor 2022, "To SMOTE, or not to SMOTE?"** (Amazon; 73 datasets). With strong classifiers (LightGBM, XGBoost, CatBoost), balancing did not help, and strong classifiers without balancing beat weak classifiers with balancing. For strong classifiers, balancing and tuning the decision threshold gave similar quality; the authors recommend threshold tuning because it is simpler and cheaper. When balancing, plain random oversampling was as good as SMOTE-like methods. Balancing could help weak classifiers. — [arXiv 2201.08528](https://arxiv.org/abs/2201.08528)
- **Buda, Maki & Mazurowski 2018 (Neural Networks 106:249–259).** For CNNs, oversampling was the dominant method in almost all scenarios and should remove the imbalance completely. Thresholding was also worth considering. (Image benchmarks, not text.) — [arXiv 1710.05381](https://arxiv.org/abs/1710.05381); [PubMed](https://pubmed.ncbi.nlm.nih.gov/30092410/)
- **Focal loss (Lin et al. 2017, ICCV).** FL(p_t) = −α_t(1−p_t)^γ log(p_t) down-weights easy, well-classified examples, so training focuses on hard ones. It was designed for dense one-stage object detection, with extreme foreground–background imbalance across ~100k candidate boxes per image. γ=2, α=0.25 worked best in their experiments. — [arXiv 1708.02002](https://arxiv.org/abs/1708.02002)
- Focal loss is **not strictly proper**: the scores it learns "do not match the true class-posterior probability", and it can produce both under- and over-confident classifiers. A closed-form transform recovers the posteriors. — [Charoenphakdee et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Charoenphakdee_On_Focal_Loss_for_Class-Posterior_Probability_Estimation_A_Theoretical_Perspective_CVPR_2021_paper.html)
- Focal loss combined with temperature scaling gave state-of-the-art calibration on (mostly balanced, multiclass) vision and NLP benchmarks without hurting accuracy. — [Mukhoti et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/aeb7b30ef1d024a76f21a1d40e30c302-Abstract.html)
- **LightGBM docs** for both `is_unbalance` and `scale_pos_weight`: "while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities." Balanced bagging (`pos_bagging_fraction`/`neg_bagging_fraction`) requires `bagging_freq` and overrides `bagging_fraction`. — [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- **Undersampling correction (Dal Pozzolo et al. 2015).** Undersampling negatives at rate β biases the posterior but keeps the ranking. The bias can be undone analytically with p = βp_s / (βp_s − p_s + 1). — [Semantic Scholar entry](https://www.semanticscholar.org/paper/Calibrating-Probability-with-Undersampling-for-Pozzolo-Caelen/e36bb7fbe1b4f7c521608e93a2215e2062dae5b1); formula as restated in [arXiv 2410.18144](https://arxiv.org/pdf/2410.18144)
  - Contradicted for trees: Phelps, Lizotte & Woolford (2024) found that analytical correction of undersampled random forests gives prevalence estimates that depend on the number of features tried per split. They also found decision trees biased toward the minority class, and conclude that "tree-based models trained on undersampled data should not be calibrated analytically". They favour learned recalibration such as beta calibration. — [arXiv 2412.16209](https://arxiv.org/abs/2412.16209)
- **Unreviewed preprint (Liu, June 2026).** On five datasets (imbalance ratio 1.9–70) with RF and gradient boosting:
  - SMOTE only modestly hurt calibration (ECE +0.009).
  - RUS damage "grows sharply with imbalance", inflating ECE from 0.008 to 0.395 on the most imbalanced dataset.
  - Platt or isotonic recalibration cut ECE by up to 66% at AUC −0.002.
  - The analytic prior-shift fix does not transfer to SMOTE, because SMOTE "distorts the class-conditional density".

  Single author, not peer-reviewed. Treat as weak evidence. — [arXiv 2606.29720](https://arxiv.org/abs/2606.29720)
- **Rare-events logistic regression (King & Zeng 2001, Political Analysis 9(2)).** In small samples, maximum-likelihood logistic regression underestimates rare-event probabilities. The paper gives prior-correction and weighting estimators, and argues that sampling all events plus a fraction of non-events is efficient. — [King & Zeng 2001 PDF](https://gking.harvard.edu/files/0s.pdf)
- **Cost-sensitive theory (Elkan 2001, IJCAI).** The Bayes-optimal threshold depends only on the misclassification costs, and changing the class proportions in training is equivalent to moving the threshold. This is the theoretical basis for "threshold moving ≈ rebalancing". — [Elkan 2001](https://cseweb.ucsd.edu/~elkan/rescale.pdf)
- Oversampling **before** the train/test split is named as leakage type L1.2 ("pre-processing on training and test set") by Kapoor & Narayanan. — [arXiv 2207.07048](https://arxiv.org/abs/2207.07048)

### Inferences
- **Default for every model:** no resampling. Use a proper loss (log-loss / BCE) on the natural distribution. Choose the alarm threshold on validation sessions to hit a target false-alarm rate per minute. This is exactly threshold moving, which the cost-sensitive view (Elkan) and both empirical studies support.
- **Logistic regression and naive Bayes:** `class_weight=None` plus L2/C tuning. If `class_weight='balanced'` helps the optimiser, keep it but recalibrate with Platt on held-out sessions. Expect average precision (AP) to barely change while the raw probabilities inflate.
- **LightGBM:** treat `scale_pos_weight ∈ {1, ~sqrt(neg/pos), neg/pos}` as a hyperparameter scored by grouped-CV AP. Do not read raw probabilities without recalibrating. With only hundreds of positives, `min_child_samples` / `min_sum_hessian_in_leaf` and regularisation probably matter more than the weighting. This is my inference, not tested in the sources.
- **Neural nets (TextCNN, GRU, embedding MLP):** use weighted BCE (`pos_weight` in PyTorch `BCEWithLogitsLoss`) or session-aware minibatch oversampling. Add focal loss only as an ablation. Always temperature- or Platt-scale on held-out sessions afterwards.
- **SMOTE:** effectively inapplicable to raw transcripts and token sequences. Interpolating in embedding space is possible, but the minority neighbours are near-duplicates (positive windows 2 s apart), so SMOTE mostly makes copies. If it is tried at all, do it only inside the training fold (an `imblearn.pipeline.Pipeline`), never before splitting.
- **Effective sample size is the number of hand-raise *events* (and sessions), not the number of positive 2-s windows.** How many windows before onset count as positive, and whether windows after onset are masked, will move results more than any rebalancing choice.

### Gaps
- I found no controlled study of class weighting or focal loss for small-data *text* or streaming event prediction with autocorrelated windows. The neural-net evidence (Buda et al.) comes from image CNNs.
- Evidence on how calibration interacts with imbalance correction for *gradient-boosted trees* is thin: one 2024 arXiv paper and one unreviewed 2026 preprint. The JAMIA study covers only logistic regression.

## 2. Metrics for rare events: AP vs ROC-AUC, PR-gain, event-level metrics, and confidence intervals with correlated samples

### Takeaway
Report frame-level AP next to its base rate (prevalence = AP of a random ranker), with ROC-AUC as a secondary metric. Make the event-level analysis the headline: onset recall against false alarms per minute (an AMOC/FROC-style curve), plus lead time, scored with explicit tolerance, merging and refractory rules like seizure-detection frameworks use. Never use point-adjusted scores. Get confidence intervals by resampling whole sessions (cluster bootstrap), not windows.

### Cited Findings
- **Saito & Rehmsmeier 2015 (PLOS ONE 10(3):e0118432).** On imbalanced data ROC plots can be visually deceptive, while precision–recall plots are more informative. The baseline of a PR plot is y = P/(P+N), the prevalence. — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
- **Counterpoint, McDermott et al. 2024 (NeurIPS).** AUPRC is "not generally superior in cases of class imbalance". It "can unduly favor model improvements in subpopulations with more frequent positive labels". In a review of ~1.5M papers, the AUPRC-is-better claim was often uncited, misattributed or overgeneralised. — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4df3510ad02a86d69dc32388d91606f8-Abstract-Conference.html); [arXiv 2401.06091](https://arxiv.org/abs/2401.06091)
- **Interpolation.** One curve dominates another in ROC space if and only if it dominates in PR space. Linear interpolation between PR points is wrong (overly optimistic). — [Davis & Goadrich, ICML 2006](https://dl.acm.org/doi/10.1145/1143844.1143874). scikit-learn's `average_precision_score` is a step-wise, non-interpolated sum, unlike trapezoidal PR-AUC. — [sklearn average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- **Precision–Recall–Gain (Flach & Kull, NIPS 2015).** Area under a PR curve takes an arithmetic mean of precisions, whereas F-scores use harmonic means, which is incoherent. PRG curves plot precision gain = (prec − π)/((1 − π)·prec) against recall gain (defined the same way, with π the prevalence). In gain space the baseline is the minor diagonal, F1 isometrics are parallel, and AUPRG relates to expected F1. The PyPI package `precision-recall-gain` implements it. — [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf); [PyPI](https://pypi.org/project/precision-recall-gain/)
- **Prevalence-dependence.** AUC-PR and F1 "are heavily dependent on the class prior", which makes it hard to compare across subpopulations or periods. Siblini et al. propose "calibrated" precision, computed as if the positive ratio equalled a reference ratio. — [Siblini et al., IDA 2020](https://link.springer.com/chapter/10.1007/978-3-030-44584-3_36); [arXiv 1909.02827](https://ui.adsabs.harvard.edu/abs/2019arXiv190902827S/abstract)
- **Event-based scoring in seizure detection (SzCORE, Dan et al., Epilepsia, published 2024–25).**
  - Event-based scoring is the primary mode because it counts detected events and false alarms the way clinicians experience them.
  - Metrics are sensitivity, precision, F1 and false positives per 24 h.
  - Rules: any overlap counts as a detection; 30 s pre-event and 60 s post-event tolerance; events closer than 90 s are merged; events longer than 5 min are split into 5-min segments.
  - Sample-based scoring uses 1 Hz labels, with a sample labelled positive if at least 50% covered.
  - No explicit latency requirement.

  — [SzCORE framework page](https://epilepsybenchmarks.com/framework/); [Epilepsia article](https://onlinelibrary.wiley.com/doi/full/10.1111/epi.18113)
- **Activity monitoring / AMOC (Fawcett & Provost, KDD 1999).** Introduced the Activity Monitoring Operating Characteristic: false-alarm rate against time-to-detection, for problems where alarms on a positive episode are not equally valuable early and late. This is the closest classic analogue of "onset recall + lead time vs false alarms per minute". — [ACM DL](https://dl.acm.org/doi/10.1145/312129.312195); description of AMOC (FAR vs TTD) in [Mach Learn 2010 PDF](https://link.springer.com/content/pdf/10.1007/s10994-009-5141-7.pdf)
- **Range-based precision/recall (Tatbul et al., NeurIPS 2018).** Range-based precision and recall add an existence reward plus overlap size, cardinality and a *positional bias* term (e.g. front-end bias that rewards early detection inside the range). — [NeurIPS 2018 PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/8f468c873a32bb0619eaeb2050ba45d1-Paper.pdf)
- **Numenta Anomaly Benchmark (Lavin & Ahmad 2015).** Scores detections inside windows around each anomaly, with a scaled-sigmoid reward that favours earlier detection, penalties for false positives, and "application profiles" that weight FP/FN differently. — [arXiv 1510.03336](https://arxiv.org/abs/1510.03336)
- **Point-adjust warning (Kim et al., AAAI 2022).** Point adjustment marks a whole anomaly segment as detected if any point in it is flagged. It "has a great possibility of overestimating detection performance; even a random anomaly score can easily turn into a state-of-the-art" detector. — [AAAI 2022 PDF](https://cdn.aaai.org/ojs/20680/20680-13-24693-1-2-20220628.pdf)
- **Keyword spotting convention.** Fix false alarms per hour and report the false-reject rate at that operating point, as in Sainath & Parada (Interspeech 2015, CNN keyword spotting). — [ISCA archive](https://www.isca-archive.org/interspeech_2015/sainath15b_interspeech.html). The exact operating points per paper were not verified here.
- **Correlated samples and CIs:**
  - Field & Welsh (2007, JRSS-B 69(3):369–390) analyse bootstraps for clustered data: resample at the cluster level, with validity depending on the assumed cluster model. — [DOI](https://doi.org/10.1111/j.1467-9868.2007.00593.x)
  - Obuchowski (1997, Biometrics 53(2):567–578) gives nonparametric ROC-AUC analysis for clustered data that accounts for intracluster correlation. — [JSTOR DOI](https://dx.doi.org/10.2307/2533958)
  - Roberts et al. (2017, Ecography) show that ignoring temporal or hierarchical dependence in cross-validation gives optimistic error estimates, and recommend blocked CV. — [DOI](https://doi.org/10.1111/ecog.02881)
  - Bates, Hastie & Tibshirani (JASA 2024): standard CV CIs are miscalibrated because fold estimates are correlated, so the naive variance is too small. Their nested CV gives roughly correct coverage, with intervals typically 20–80% wider. — [arXiv 2104.00673](https://arxiv.org/abs/2104.00673); [PDF](https://tibshirani.su.domains/ftp/NCV.pdf)

### Inferences
- **Event definition to write into `metrics.py`.** Onset t₀ = the moment of the hand-raise. An alarm at time a is a *hit* for that onset if t₀ − H ≤ a ≤ t₀ + ε, with H the maximum useful lead (e.g. 20–30 s) and ε a small post-onset tolerance (e.g. 2 s, one decision step).
  - **Refractory period:** after an alarm, suppress further alarms for R seconds (e.g. 10 s). Without it, one sustained high score counts as many false alarms.
  - **False alarm:** an alarm that does not hit any onset.
  - **FA/min denominator:** at-risk class minutes, excluding time when he is already speaking, already has his hand up, or the class is in breakout rooms.
  - **Lead time:** t₀ minus the first alarm inside the hit window. Report the median and IQR over hits.

  Sweep the threshold to get a recall-vs-FA/min curve (a free-response/AMOC curve). Report recall and median lead time at fixed points (e.g. 0.25, 0.5 and 1 FA/min) and the partial area up to 1 FA/min. Report the same numbers for a "base-rate" alarmer (random alarms at the same rate) and a timing-only heuristic.
- **Frame-level:** report AP with prevalence π and the lift AP/π for every fold. Report ROC-AUC as secondary, citing McDermott et al. for why both are shown. Optionally add AUPRG or prevalence-calibrated AP when comparing folds or sessions whose base rates differ.
- **Never** count each positive 2-s window as a separate detection, and never apply point adjustment.
- **Session cluster bootstrap for CIs.** Resample *sessions* with replacement (B ≈ 2,000), recompute the pooled metric each time (AP, recall@0.5 FA/min, median lead), and take percentile intervals. For model comparisons, bootstrap the *paired difference* on the same resampled sessions. With fewer than ~20 test sessions, say that percentile intervals may undercover.

### Gaps
- I found no peer-reviewed paper on cluster-bootstrap CIs for *average precision* specifically. The clustered-ROC (Obuchowski) and general cluster-bootstrap (Field & Welsh) literatures are the nearest support.
- The exact seizure-latency metrics in SzCORE were not available, and the Epilepsia full text returned 403. The framework page says no latency threshold is specified.
- I did not find a standard published metric for "anticipatory" (pre-onset) detection of a single person's discrete social action. The AMOC / range-based / NAB family is the closest.

## 3. Calibration: Platt, isotonic, temperature; reliability diagrams, ECE, Brier; imbalance; CalibratedClassifierCV with grouped data

### Takeaway
Calibrate on held-out *sessions*, never on windows from training sessions. Platt (sigmoid) is the safe default with a few hundred events. Isotonic needs ≫1,000 calibration samples, and here the effective count is events. Temperature scaling suits the neural nets. Evaluate with quantile-binned reliability diagrams, Brier skill score, the stratified (per-class) Brier score and log-loss. Plain fixed-bin ECE is close to meaningless at 1–5% prevalence.

### Cited Findings
- **Niculescu-Mizil & Caruana 2005 (ICML).**
  - Max-margin methods such as boosted trees and SVMs push predictions away from 0 and 1, a sigmoid-shaped distortion. Naive Bayes pushes them toward 0 and 1.
  - Platt scaling works well when calibration data are scarce. Isotonic regression is more flexible but overfits small calibration sets.
  - After calibration, boosted trees gave the best probabilities.

  — [Paper PDF](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- **Guo et al. 2017 (ICML).** Modern deep networks are poorly calibrated, typically overconfident. Temperature scaling, one scalar T applied to the logits and fit on validation NLL, is a surprisingly effective fix, and it leaves rankings and accuracy unchanged. — [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
- **scikit-learn 1.9.1 `CalibratedClassifierCV`:**
  - Methods `'sigmoid'` (default, Platt), `'isotonic'` and `'temperature'` (added in 1.8).
  - "Isotonic calibration is not recommended when the number of calibration samples is too low (≪1000) since it then tends to overfit."
  - "For very uncalibrated classifiers on very imbalanced datasets, sigmoid calibration might be preferred because it fits an additional intercept parameter."
  - `cv` accepts an int, a splitter or **an iterable of (train, test) index pairs**.
  - `ensemble=True` fits a model/calibrator pair per fold and averages them. `ensemble=False` uses `cross_val_predict` to get unbiased scores and fits one calibrator on top of one model refit on all data.
  - `cv="prefit"` was deprecated in 1.6. To calibrate an already-fitted model, wrap it as `CalibratedClassifierCV(FrozenEstimator(model))`; "the user must manually ensure that data for model fitting and calibration are disjoint." `ensemble='auto'` resolves to False for a FrozenEstimator.

  — [CalibratedClassifierCV docs](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html); [FrozenEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html); [1.6 changelog](https://scikit-learn.org/stable/whats_new/v1.6.html)
- **Wallace & Dahabreh (ICDM 2012; KAIS 2014).** Probability estimates for the minority class are unreliable, and the overall Brier score *masks* this. They propose the **stratified Brier score**, computed per class, and bagging of estimators calibrated on balanced bootstraps as a fix. — [ResearchGate (ICDM 2012)](https://www.researchgate.net/publication/261040867_Class_Probability_Estimates_are_Unreliable_for_Imbalanced_Data_and_How_to_Fix_Them); [KAIS 2014](https://dl.acm.org/doi/abs/10.1007/s10115-013-0670-6)
- **ECE estimation problems:**
  - Nixon et al. (CVPR-W 2019): standard fixed-bin ECE is an inefficient estimator. They propose adaptive (equal-mass) binning (ACE) and thresholded variants. — [CVF PDF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf)
  - Binned ECE estimators are biased. Kumar, Liang & Ma show binned estimates can understate the miscalibration of scaling methods. — [Kumar et al., NeurIPS 2019](https://arxiv.org/abs/1909.10155)
  - Roelofs et al. propose ECE_sweep to reduce estimation bias. — [Roelofs et al., AISTATS 2022](https://proceedings.mlr.press/v151/roelofs22a.html)
- **van den Goorbergh et al.** use the calibration intercept (calibration-in-the-large) and calibration slope, plus flexible calibration curves. These are the standard clinical-prediction calibration metrics and suit a written report. — [arXiv 2202.09101](https://arxiv.org/pdf/2202.09101)
- **Calibration after rebalancing on trees:** prefer learned recalibration such as beta calibration over analytic prior correction. — [Phelps et al. 2024](https://arxiv.org/abs/2412.16209)
- **MAPIE alternative:** MAPIE 1.3.0 (Feb 2026) added Venn-Abers calibration for binary and multiclass classification. — [MAPIE releases](https://github.com/scikit-learn-contrib/MAPIE/releases); dates per [PyPI](https://pypi.org/project/MAPIE/)
- **Neural nets:** focal loss plus temperature scaling is well calibrated. — [Mukhoti et al. 2020](https://proceedings.neurips.cc/paper/2020/hash/aeb7b30ef1d024a76f21a1d40e30c302-Abstract.html)

### Inferences
- **Grouped CV recipe:**
  ```python
  splits = list(GroupKFold(n_splits=5).split(X_tr, y_tr, groups=session_tr))
  cal = CalibratedClassifierCV(model, method="sigmoid", cv=splits, ensemble=False)
  ```
  Passing explicit index pairs avoids depending on metadata routing for `groups`, and the calibrator is fit on out-of-session predictions.
- **Forward-chaining recipe:** train on sessions ≤ d−2, calibrate with `FrozenEstimator` on the date block d−1, and test on d. This mirrors deployment, where the calibrator is always older than the test data.
- **Isotonic** is defensible only when the calibration fold holds ≫1,000 points *and* enough positive events. With a few hundred events and correlated windows, use sigmoid (LR, NB, LightGBM, kNN) or temperature (TextCNN, GRU).
- **Monotone calibration does not change ranking metrics.** Platt and temperature scaling leave AP, ROC-AUC and recall@FA/min unchanged within a pooled test set. Isotonic can create ties and shift AP slightly. Calibration matters for the report only if probabilities are shown (e.g. "P(raise within 10 s)"), if one threshold is carried across sessions, or if a Brier or log-loss comparison is made. Say this explicitly so readers don't expect calibration to raise AP.
- **Report:**
  1. A reliability diagram with `CalibrationDisplay.from_predictions(..., strategy="quantile", n_bins=10)`. Uniform bins put almost every point in the first bin at 2% prevalence. Consider a zoomed or log-x panel for 0–0.2.
  2. Brier skill score = 1 − Brier/[π(1−π)]. The constant-prevalence forecaster scores π(1−π), only ≈0.0196 at π=2%, so a raw Brier score looks deceptively good.
  3. Stratified Brier (positives and negatives separately).
  4. Log-loss.
  5. Calibration intercept and slope (fit logit(y) ~ logit(p)).

  Put any ECE in a footnote and label it as adaptive-binned.

### Gaps
- I found no study of calibration specifically for *autocorrelated* streaming predictions. Calibrators assume i.i.d. calibration points, and the effective calibration sample size here is unknown (somewhere between the number of events and the number of windows).
- Whether `CalibratedClassifierCV` automatically routes `groups` to a grouped splitter depends on enabling metadata routing (`sklearn.set_config(enable_metadata_routing=True)`). I did not verify the exact behaviour in 1.9.1, so the explicit-splits approach above is safer.

## 4. Conformal prediction for classification and streams (MAPIE, conformal risk control): meaningful here?

### Takeaway
Plain marginal conformal prediction sets are nearly useless for a 2%-positive binary label: the set {0} for every point already gives ~98% coverage. The meaningful options are (a) class-conditional (Mondrian) coverage, and (b) **risk control of the alarm threshold** (Learn-Then-Test / conformal risk control). The latter gives a statistically guaranteed bound on precision, false-positive rate or recall over the calibration distribution. Exchangeability holds at best between *sessions*, not between 2-s windows, and drift across the semester weakens it further. Present conformal as an optional, clearly caveated "uncertainty" section, not as a headline result.

### Cited Findings
- **Split conformal (Angelopoulos & Bates).** Gives 1−α *marginal* coverage under exchangeability of calibration and test points, not conditional coverage. Class-conditional coverage requires calibrating separately within each class. — [A Gentle Introduction to Conformal Prediction, arXiv 2107.07511](https://arxiv.org/abs/2107.07511)
- **MAPIE v1 API** (1.0 released 2025; 1.5.0 current, Aug 2026):
  - `SplitConformalClassifier` and `CrossConformalClassifier`, with methods `fit` / `conformalize` / `predict_set`.
  - Conformity scores `"lac"`, `"top_k"`, `"aps"`, `"raps"`, or a custom `BaseClassificationScore`.
  - `alpha` was replaced by `confidence_level` (= 1 − α).

  — [MAPIE classification API](https://mapie.readthedocs.io/en/stable/api/classification/); [v1 release notes](https://mapie.readthedocs.io/en/stable/content/getting-started/v1-release-notes/)
- **MAPIE `BinaryClassificationController`** (added in 1.1.0, Sep 2025):
  - Finds decision thresholds that "statistically guarantee a risk to be below a target level".
  - Risks: `"precision"`, `"recall"`, `"accuracy"`, `"fpr"`, `"predicted_positive_fraction"`, or a custom `BinaryClassificationRisk`.
  - Uses Learn-Then-Test with binary Hoeffding–Bentkus p-values and family-wise error control (`fwer_method` defaults to `"bonferroni_holm"`); `confidence_level` defaults to 0.9.
  - **Assumes exchangeability** of calibration and test data.
  - In the docs example, targeting precision ≥ 0.8 at 0.9 confidence: the naive threshold gave test precision 0.763, the risk-controlled threshold 0.857.

  — [MAPIE risk-control API](https://mapie.readthedocs.io/en/stable/api/risk-control/); [precision-control example](https://mapie.readthedocs.io/en/latest/examples_risk_control/1-quickstart/plot_risk_control_binary_classification.html)
- **Later MAPIE releases:**
  - 1.3.0 (Feb 2026): Venn-Abers calibration and the `predicted_positive_fraction` risk.
  - 1.4.0 (Apr 2026): FWER methods, exchangeability tests (permutation and online martingale) and experimental risk monitoring.
  - 1.5.0 (Aug 2026): conditional conformal (`ConditionalSplitConformalClassifier`, `ConditionalExpectedRiskController`) and a classwise ECE metric.

  The GitHub release page as rendered by my fetch tool showed years two years too early. The PyPI upload dates (2025-11-17 for 1.2.0 through 2026-08-05 for 1.5.0) are authoritative. — [MAPIE releases](https://github.com/scikit-learn-contrib/MAPIE/releases); [PyPI](https://pypi.org/project/MAPIE/)
- **Learn-Then-Test** (Angelopoulos, Bates, Candès, Jordan, Lei): risk control by multiple hypothesis testing over a grid of thresholds. — [arXiv 2110.01052](https://arxiv.org/abs/2110.01052)
- **Conformal risk control** (Angelopoulos, Bates, Fisch, Lei, Schuster; ICLR 2024): bounds the expected value of any monotone loss, such as the false-negative rate. — [arXiv 2208.02814](https://arxiv.org/abs/2208.02814)
- **Non-exchangeable / streaming data:**
  - Barber, Candès, Ramdas & Tibshirani (Annals of Statistics 51(2):816–845, 2023) weight calibration points, e.g. recent sessions more heavily, and bound the coverage gap under drift. — [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.full); [arXiv 2202.13415](https://arxiv.org/abs/2202.13415)
  - Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) updates α_t online to keep long-run coverage under arbitrary shift. — [arXiv 2106.00170](https://arxiv.org/abs/2106.00170)
- **Hierarchical / grouped data (Dunn, Wasserman & Ramdas, JASA).** Exchangeability fails when groups (e.g. patients, here sessions) contribute many correlated observations. The paper proposes CDF pooling, single subsampling and repeated subsampling (one point per group), recommending repeated subsampling when a coverage guarantee is required. — [arXiv 1809.07441](https://arxiv.org/abs/1809.07441); [code](https://github.com/RobinMDunn/ConformalTwoLayer)

### Inferences
- **The most defensible use here is a session-level risk-control demonstration.** On calibration sessions, choose the alarm threshold so that a risk stays below a target with ~90% confidence; then check on test sessions whether the bound held. The risk can be window-level FPR or precision via MAPIE's `BinaryClassificationController`.
- **The metric that matters (false alarms per minute) is event-level, not per-window**, so MAPIE's built-in risks do not express it directly. A pragmatic LTT-style substitute treats each calibration session as one exchangeable unit. For each candidate threshold, compute per-session FA/min, then pick the lowest threshold whose upper confidence bound across sessions (e.g. a one-sided t or Hoeffding bound on the session mean) is ≤ the target. This is my inference, not a published recipe.
- **If prediction sets are shown at all,** use class-conditional (Mondrian) calibration and report coverage *among positives* separately. Marginal coverage will be met trivially, and the report should say so.
- **Forward-chaining violates exchangeability through drift** (topic, time of semester). Either weight recent calibration sessions (Barber et al.) or describe the guarantee as approximate. ACI is the principled tool for within-session streaming, but it is probably out of scope for an Oct 16 deadline.
- **Priority:** low. One paragraph plus one figure (threshold vs realised FA/min on calibration and test sessions, with the risk-controlled threshold marked) is enough to show #uncertainty awareness.

### Gaps
- I found no published application of conformal prediction or risk control to *event-level* false-alarm rates in streaming detection with session-clustered data.
- I could not confirm whether MAPIE's custom `BinaryClassificationRisk` can express a per-session, per-minute risk. It appears built around per-sample risks.

## 5. Statistical comparison of models across grouped folds

### Takeaway
Per-fold scores share training data and are correlated, so a naive paired t-test (or any test treating 2-s windows as independent) is badly anti-conservative. Use one of:
- the Nadeau–Bengio corrected resampled t-test on per-fold differences;
- its Bayesian counterpart, the correlated t-test with a region of practical equivalence (ROPE);
- a paired session-level bootstrap of the metric difference.

Pre-specify one primary metric and a few primary comparisons, apply Holm/Bonferroni to the frequentist tests, and label everything else exploratory.

### Cited Findings
- **Corrected t-test (sklearn example).** The scikit-learn example computes t = mean(x) / sqrt((1/(k·r) + n_test/n_train)·σ̂²), with x the per-fold score differences, k folds and r repetitions. In its demo the uncorrected p was 0.005 and the corrected p 0.227. The same corrected variance serves as the scale of a Student-t posterior for Bayesian comparison: P(model A better), P(inside a ROPE, e.g. ±0.01) and credible intervals. For pairwise frequentist tests it applies Bonferroni and states that the Bayesian approach needs no multiple-comparison correction (citing Benavoli et al. 2017). — [sklearn: Statistical comparison of models using grid search](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html)
- **Nadeau & Bengio** (Machine Learning 52:239–281, 2003) is the original variance correction for resampled train/test estimates. — [Springer](https://link.springer.com/article/10.1023/A:1024068626366)
- **Benavoli, Corani, Demšar & Zaffalon 2017** (JMLR 18(77):1–36) argue for replacing null-hypothesis tests with Bayesian analysis (correlated t-test, ROPE) when comparing classifiers. — [JMLR](https://www.jmlr.org/papers/v18/16-305.html)
- **CV confidence intervals undercover** because fold estimates are correlated; nested CV fixes the variance estimate. — [Bates, Hastie & Tibshirani, arXiv 2104.00673](https://arxiv.org/abs/2104.00673)
- **Fold and cluster structure:** blocked or grouped CV is required when data are structured, and ignoring it is optimistic. — [Roberts et al. 2017](https://doi.org/10.1111/ecog.02881)

### Inferences
- **Grouped CV:** use per-fold metric differences (AP, recall@0.5 FA/min) with n_test/n_train ≈ 1/(k−1). With 5 folds and 1 repeat that leaves 4 degrees of freedom and little power. Repeating GroupKFold with different session-to-fold assignments (r > 1) helps.
- **Forward-chaining:** the training size grows fold by fold, so the Nadeau–Bengio correction is only heuristic. A paired session bootstrap of the metric difference on the forward-chained test blocks (Section 2) is easier to defend.
- **Never** run McNemar or DeLong-style tests that treat every 2-s window as independent. If a test needs independent units, use sessions as the units, e.g. a Wilcoxon signed-rank on per-session metric differences when there are ≥ ~10 test sessions.
- **Multiplicity:** with ~8–10 models × several metrics × ablations, choose the primary comparisons up front, e.g. (i) best model vs the logistic-regression baseline and (ii) full model vs model without transcript features. Apply Holm to those and present the rest descriptively, with CIs.
- **Hyperparameter tuning** also inflates results: tune inside the training folds (nested) or keep the last date block untouched as a final test.

### Gaps
- I found no published guidance on the Nadeau–Bengio correction for *grouped* or forward-chaining splits with unequal fold sizes. The correction was derived for random resampling.
- I did not verify which current Python packages (e.g. `baycomp`) are maintained as of Oct 2026. The sklearn example's code is self-contained and enough.

## 6. Interpretability: TreeSHAP, grouped permutation importance, LR coefficients, nearest neighbours, attention, feature-group ablations, correlated features

### Takeaway
For the report, use:
- signed LR coefficients for text n-grams, checked for stability across folds;
- TreeSHAP for LightGBM, computed on held-out sessions, stating the perturbation mode and output scale;
- **feature-group ablations by retraining** (transcript / timing-state / pre-class) as the main "what matters" evidence, because permutation importance and SHAP both mislead when features are correlated;
- nearest-neighbour exemplars for the embedding/kNN models.

Do not present GRU or TextCNN attention or saliency as explanations without caveats.

### Cited Findings
- **SHAP** unifies additive feature-attribution methods under Shapley values with local accuracy and consistency properties. — [Lundberg & Lee, NeurIPS 2017, arXiv 1705.07874](https://arxiv.org/abs/1705.07874). TreeSHAP computes exact Shapley values for tree ensembles in polynomial time and supports interaction values and global summaries. — [Lundberg et al., Nature Machine Intelligence 2020](https://www.nature.com/articles/s42256-019-0138-9)
- **SHAP `TreeExplainer` docs:**
  - `feature_perturbation="interventional"` needs a background dataset (100–1,000 samples recommended) and handles dependencies "according to the rules dictated by causal inference".
  - `"tree_path_dependent"` uses the training cover stored in the trees, with no background needed.
  - `"auto"` (added in 0.47) picks interventional when background data is given.
  - `model_output="probability"` / `"log_loss"` work only with interventional; the default explains raw log-odds.
  - Current SHAP is 0.53.0.

  — [SHAP TreeExplainer docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html); [PyPI](https://pypi.org/project/shap/)
- **LightGBM native TreeSHAP:** `Booster.predict(..., pred_contrib=True)` returns per-feature SHAP contributions plus the expected value, with no extra dependency. — [LightGBM Booster API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html)
- **Problems with Shapley-value feature importance** (Kumar et al., ICML 2020): the choice of value function and correlated features can produce misleading attributions. — [PMLR](https://proceedings.mlr.press/v119/kumar20e.html)
- **sklearn 1.9.1 permutation importance:**
  - Correlated features: "When two features are correlated and one of the features is permuted, the model still has access to the latter through its correlated feature. This results in a lower reported importance value for both features." The suggested fix is to "cluster features that are correlated and only keep one feature from each cluster."
  - Compute on held-out data to measure generalisation. Importance in a weak model says little about a good model.
  - Impurity-based (MDI) importance is biased toward high-cardinality features.

  — [sklearn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- **Permute-and-predict extrapolates (Hooker, Mentch & Zhou 2021).** Under feature dependence, permuting creates unrealistic inputs, so permute-and-predict importance "can be highly misleading". Reliable variable importance "requires at least one more model" (refit or conditional approaches). — [Statistics and Computing 31:82](https://link.springer.com/article/10.1007/s11222-021-10057-z)
- **SAGE (Covert, Lundberg & Lee, NeurIPS 2020)** gives global, loss-based Shapley importance and supports feature *groups*. — [arXiv 2004.00668](https://arxiv.org/abs/2004.00668)
- **Linear-model weights (Haufe et al., NeuroImage 2014).** Weights of discriminative ("backward") linear models are not directly interpretable as relationships between the signal and the class: correlated or noise-suppressing features can get large weights. Transform them to "activation patterns" (covariance × weights). — [DOI](https://doi.org/10.1016/j.neuroimage.2013.10.067)
- **Attention:**
  - Jain & Wallace (NAACL 2019): attention weights often do not correlate with gradient-based importance, and very different attention distributions can give the same predictions. — [ACL Anthology](https://aclanthology.org/N19-1357/)
  - Wiegreffe & Pinter (EMNLP 2019) counter that whether attention explains depends on the definition and on testing. — [ACL Anthology](https://aclanthology.org/D19-1002/)
- **Nearest-neighbour explanations (Deep kNN, Papernot & McDaniel 2018).** Retrieving the training examples nearest to a test input in representation space gives example-based explanations and a confidence signal. — [arXiv 1803.04765](https://arxiv.org/abs/1803.04765)
- **Interpretability as a leakage detector.** Leakage type L2 (illegitimate features, e.g. proxies for the outcome) is often found by inspecting which features drive predictions. — [Kapoor & Narayanan, arXiv 2207.07048](https://arxiv.org/abs/2207.07048)

### Inferences
- **Feature-group ablation table** (the key interpretability figure). Rows: timing/state only; transcript only; pre-class only; timing + transcript; all; all minus each group. Columns: AP (session-bootstrap CI), lift over π, recall@0.5 FA/min (CI), median lead time. Retrain each configuration on identical folds and report paired differences against "all" (Section 5). Retraining avoids the extrapolation problem that Hooker et al. identify in permute-and-predict.
- **If permutation importance is used:** compute it on held-out sessions and permute *whole feature groups* jointly, so correlated members cannot cover for each other. Use `n_repeats ≥ 10` and report the spread. Hierarchically cluster numeric timing features (Spearman) before per-feature importance, as the sklearn docs suggest.
- **LR on text:** standardise or TF-IDF the features. Show the top ±20 n-grams by coefficient, along with how often each appears in the top-k across folds (stability) and its document frequency. Caveat (Haufe; regularisation): coefficients reflect conditional, regularised contributions, not marginal associations. Show a raw "phrase → hand-raise rate within H s" table alongside.
- **TreeSHAP for LightGBM:** compute on held-out sessions with `pred_contrib=True` (path-dependent, log-odds). Show a beeswarm plus 2–3 dependence plots (e.g. time since the instructor's last question, minutes into class). State the scale (log-odds) and the perturbation mode. Note that for correlated features, credit is split in ways that depend on that choice.
- **kNN / embedding models:** for a few correct hits and false alarms, show the k nearest past transcript snippets and whether each was followed by a raise. This is honest, easy to read, and matches how the model actually works.
- **GRU / TextCNN:** if saliency or attention is shown, label it as descriptive only and cite Jain & Wallace. Do not draw causal conclusions from it.

### Gaps
- No source compared these interpretation methods on small, correlated, streaming text-plus-timing data. The recommendations above are synthesised from general-purpose sources.
- I did not verify whether sklearn 1.9.1's `permutation_importance` natively supports feature groups. Joint group permutation may need a short custom loop.

## 7. Leakage pitfalls for time series / grouped data and how to audit them

### Takeaway
The biggest threats are:
- **L3.2 non-independence:** windows from the same session in both train and test, which the planned random-split condition is meant to quantify;
- **L3.1 temporal leakage:** inputs that include words or timing after the decision time, or session-level statistics computed over the whole session;
- **L1.2/L1.3:** vocabulary, TF-IDF, scalers, embedding PCA, feature selection or resampling fit outside the training fold;
- **L2:** illegitimate proxy features (e.g. the instructor acknowledging him by name after the hand is already up).

Audit with an explicit timestamp check, a random-vs-grouped split gap, label-permutation and time-shift sanity tests, and a look at the top features.

### Cited Findings
- **Kapoor & Narayanan** ("Leakage and the reproducibility crisis in ML-based science", Patterns 2023; arXiv 2207.07048):
  - The arXiv version reports a survey of 20 papers across 17 fields, collectively affecting 329 papers; the project site now lists a larger, updated count (648 papers, 30 fields).
  - In civil-war prediction, claims that complex models beat logistic regression all failed to reproduce once leakage was fixed.
  - They propose "model info sheets" that argue for the absence of each leakage type.

  — [arXiv 2207.07048](https://arxiv.org/abs/2207.07048); [project site](https://reproducible.cs.princeton.edu/)
- **Taxonomy (verbatim labels from the paper):**
  - **L1.1** No test set.
  - **L1.2** Pre-processing on training and test set, including "over/under sampling" before splitting.
  - **L1.3** Feature selection on training and test set.
  - **L1.4** Duplicates in datasets.
  - **L2** Model uses features that are not legitimate, e.g. proxies for the outcome.
  - **L3** Test set not drawn from the distribution of scientific interest:
    - **L3.1 Temporal leakage**: "the test set should not contain any data from a date before the training set".
    - **L3.2 Nonindependence between train and test samples**: "in the extreme (but unfortunately common) case, train and test samples come from the same people or units"; remedies such as "block cross validation".
    - **L3.3** Sampling bias in test distribution.

  — [arXiv PDF](https://arxiv.org/pdf/2207.07048)
- **Blocked CV** for temporal and hierarchical structure (Roberts et al. 2017), cited by Kapoor & Narayanan as a remedy for L3.2. — [DOI](https://doi.org/10.1111/ecog.02881)
- **Permutation test for classifier performance** (Ojala & Garriga, JMLR 2010): retraining with permuted labels gives a null distribution of the score, so you can check that a model is learning real signal. — [JMLR](https://jmlr.org/papers/v11/ojala10a.html)
- **Inflated time-series evaluation protocols** (point adjustment) make even random scores look state-of-the-art. — [Kim et al., AAAI 2022](https://cdn.aaai.org/ojs/20680/20680-13-24693-1-2-20220628.pdf)

### Inferences
Audit checklist for the report, as a "model info sheet"-style table:
1. **Split gap:** report the same model under a random-window split vs a session-grouped split vs forward-chaining. The size of the gap measures L3.2 leakage.
2. **Causality assertion (L3.1):** a unit test that, for every decision point t, every input token or utterance has an end time ≤ t − processing latency. Note that ASR segment end times and sentence-level punctuation can carry future words into the present. Rolling features must be trailing only; session-normalised features (z-scores, "fraction of class elapsed" from the actual end time) must use only information known at t, e.g. the scheduled end time, not the actual one.
3. **Fit-inside-fold (L1.2/L1.3):** vocabulary, TF-IDF IDF weights, scalers, PCA/UMAP on embeddings, feature selection, resampling and the calibrator must all be fit within the training fold (sklearn/imblearn `Pipeline`). Pre-trained embedding models are fine, since they are not fit on these labels.
4. **Label-permutation sanity test:** shuffle event times within each session (or shift labels by a random offset), retrain, and confirm AP ≈ base rate and recall@FA/min ≈ chance.
5. **Time-shift test:** move the feature window forward by +5–10 s, deliberately peeking. If performance jumps, the "real" model may also be partly peeking. If it barely changes, temporal leakage is unlikely to explain the results.
6. **Illegitimate proxies (L2):** inspect the top n-grams and SHAP features for post-hoc cues, e.g. the student's name, "go ahead", "I see a hand", or the platform's hand-raise notification text. Mask windows after onset, plus a short guard band before it if annotation timing is uncertain.
7. **Duplicates (L1.4):** pre-class content reused across sections or sessions (same reading, same slides) can appear in both train and test. Check exact and near-duplicate text overlap, or group splits by course unit.
8. **Hyperparameters and model selection:** tune on inner folds or a validation date block. Report the final forward-chained block once.

### Gaps
- The published Patterns version could not be fetched (HTTP 403), so I used the arXiv version. Paper and field counts may differ slightly in the journal version.
- I found no leakage audit specific to live-transcript or ASR-timing pipelines. The timestamp-causality checks above are my synthesis.
