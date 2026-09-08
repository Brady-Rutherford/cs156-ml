# Probabilistic Machine Learning, Chapter 1: Condensed Reading

*Source: Kevin P. Murphy, Probabilistic Machine Learning: An Introduction (MIT Press, 2022). Covers printed pages 1 to 33, which is PDF pages 31 to 63. Add 30 to any printed page number to get the PDF page.*

---

## 1. The frame: what ML is, and why probabilistic

Mitchell's definition sets the vocabulary the whole chapter runs on. A program learns from experience $E$ on tasks $T$ measured by performance $P$ if its performance at $T$ improves with $E$. Every ML setup you meet later is just a different filling of those three slots.

Murphy's specific commitment is that all unknown quantities are treated as **random variables** with probability distributions over their possible values. Unknown here means both future predictions and model parameters. Two justifications are given: probability is the optimal calculus for decision-making under uncertainty (developed in Ch. 5), and it is the shared language of statistics, control theory, information theory, econometrics and statistical physics, so it lets ML connect outward rather than sitting in its own silo.

**Supervised learning** is then defined precisely: learn a mapping $f: \mathcal{X} \to \mathcal{Y}$ from inputs to outputs.

- Inputs $\boldsymbol{x}$: features, covariates, or predictors. Usually $\mathcal{X} = \mathbb{R}^D$.
- Output $y$: label, target, or response.
- Experience $E$: the training set $\mathcal{D} = \{(\boldsymbol{x}_n, y_n)\}_{n=1}^N$, with $N$ the sample size.

Small feature datasets are stored as an $N \times D$ **design matrix**: rows are examples, columns are features. The Iris data (150 flowers, 4 measurements each, 3 species) is $150 \times 4$. Worth internalizing the shape vocabulary: $N \gg D$ is tall and skinny, $D \gg N$ (genomics) is short and fat, big data means large $N$, wide data means large $D$.

Variable-size inputs such as text or graphs do not fit a design matrix natively, so they get **featurized** into fixed-length vectors first. That is the whole motivation for section 1.5.4 later.

**Exploratory data analysis** comes before modeling. For low-dimensional tabular data, use a pair plot (scatter of features $i$ vs $j$ off-diagonal, marginal density on the diagonal, colored by class). For high-dimensional data, reduce dimensionality first, then plot in 2d or 3d.

---

## 2. The core machinery (this is the part that matters)

### 2.1 A model is a parameterized function

The decision rule for Iris:

$$f(\boldsymbol{x}; \boldsymbol{\theta}) = \begin{cases} \text{Setosa} & \text{if petal length} < 2.45 \\ \text{Versicolor or Virginica} & \text{otherwise} \end{cases}$$

Read this as: $f$ of $\boldsymbol{x}$ given $\boldsymbol{\theta}$. Everything before the semicolon is data, everything after defines the rule. Here $\boldsymbol{\theta}$ stores which feature to split on and the threshold value. Stacking such rules recursively gives a **decision tree**, and $\boldsymbol{\theta}$ becomes the full set of (feature index, threshold) pairs across internal nodes.

The structural point: the shape of the model is fixed by you, and learning is a search over $\boldsymbol{\theta}$.

### 2.2 Loss, then empirical risk

Misclassification rate on the training set:

$$L(\boldsymbol{\theta}) \triangleq \frac{1}{N}\sum_{n=1}^{N} \mathbb{I}(y_n \neq f(\boldsymbol{x}_n; \boldsymbol{\theta}))$$

where $\mathbb{I}(e)$ returns 1 if $e$ is true and 0 otherwise.

This treats all errors as equally bad, which is often wrong. If Setosa and Versicolor are edible and Virginica is poisonous, misclassifying a Virginica as edible should cost far more than the reverse. So generalize to an arbitrary loss $\ell(y, \hat{y})$ and define **empirical risk**:

$$L(\boldsymbol{\theta}) \triangleq \frac{1}{N}\sum_{n=1}^{N} \ell(y_n, f(\boldsymbol{x}_n; \boldsymbol{\theta}))$$

Misclassification rate is the special case where $\ell$ is **zero-one loss**, $\ell_{01}(y, \hat{y}) = \mathbb{I}(y \neq \hat{y})$.

**Empirical risk minimization** is then the definition of training:

$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \frac{1}{N}\sum_{n=1}^{N} \ell(y_n, f(\boldsymbol{x}_n; \boldsymbol{\theta}))$$

Murphy immediately flags the catch: the real objective is expected loss on unseen data, not training loss. Hold that thought for section 4 below.

### 2.3 Two kinds of uncertainty

- **Epistemic** (model) uncertainty: ignorance of the true input-output mapping. Reducible with more data.
- **Aleatoric** (data) uncertainty: intrinsic randomness in the mapping. Not reducible. A fair coin has zero epistemic and maximal aleatoric uncertainty.

This distinction is load-bearing for active learning: high predictive entropy is only worth chasing with more data if it comes from the epistemic component.

### 2.4 Turning a function into a distribution

Instead of outputting a hard label, output a conditional distribution:

$$p(y = c \mid \boldsymbol{x}; \boldsymbol{\theta}) = f_c(\boldsymbol{x}; \boldsymbol{\theta})$$

which requires $0 \le f_c \le 1$ and $\sum_{c=1}^{C} f_c = 1$. Rather than constrain the network to satisfy that, let it emit unnormalized log-probabilities called **logits** and normalize with the **softmax**:

$$\text{softmax}(\boldsymbol{a}) \triangleq \left[\frac{e^{a_1}}{\sum_{c'} e^{a_{c'}}}, \dots, \frac{e^{a_C}}{\sum_{c'} e^{a_{c'}}}\right]$$

giving the model $p(y = c \mid \boldsymbol{x}; \boldsymbol{\theta}) = \text{softmax}_c(f(\boldsymbol{x}; \boldsymbol{\theta}))$.

When $f$ is affine, $f(\boldsymbol{x};\boldsymbol{\theta}) = b + \boldsymbol{w}^\top \boldsymbol{x}$, this is **logistic regression**. Terminology map: statisticians say regression coefficients $\boldsymbol{\beta}$ and intercept; ML says weights $\boldsymbol{w}$ and bias $b$. Absorbing $b$ into $\boldsymbol{w}$ via $\tilde{\boldsymbol{w}} = [b, w_1, \dots, w_D]$ and $\tilde{\boldsymbol{x}} = [1, x_1, \dots, x_D]$ turns the affine function into a linear one, so you can just write $f(\boldsymbol{x}; \boldsymbol{w}) = \boldsymbol{w}^\top \boldsymbol{x}$ throughout.

### 2.5 Maximum likelihood estimation

Pick the loss to be negative log probability, $\ell(y, f(\boldsymbol{x};\boldsymbol{\theta})) = -\log p(y \mid f(\boldsymbol{x};\boldsymbol{\theta}))$. Averaged over the training set this is the **negative log likelihood**:

$$\text{NLL}(\boldsymbol{\theta}) = -\frac{1}{N}\sum_{n=1}^{N} \log p(y_n \mid f(\boldsymbol{x}_n; \boldsymbol{\theta}))$$

and $\hat{\boldsymbol{\theta}}_{\text{mle}} = \arg\min_{\boldsymbol{\theta}} \text{NLL}(\boldsymbol{\theta})$. Intuition: a good model assigns high probability to the labels that actually occurred.

**This is the chapter's central move.** Model plus loss plus optimizer, with the loss derived from a probability model, is the template every later chapter instantiates.

---

## 3. Regression: same skeleton, different loss

For $y \in \mathbb{R}$, use quadratic (or $\ell_2$) loss $\ell_2(y,\hat{y}) = (y - \hat{y})^2$, whose empirical risk is the **mean squared error**. It penalizes large residuals disproportionately, which is bad under outliers; $\ell_1$ loss is the robust alternative.

The probabilistic version assumes Gaussian output noise, $p(y_n \mid \boldsymbol{x}_n; \boldsymbol{\theta}) = \mathcal{N}(y_n \mid f(\boldsymbol{x}_n;\boldsymbol{\theta}), \sigma^2)$. Working through the NLL with fixed $\sigma^2$:

$$\text{NLL}(\boldsymbol{\theta}) = \frac{1}{2\sigma^2}\text{MSE}(\boldsymbol{\theta}) + \text{const}$$

So least squares is not an arbitrary convention. It is exactly maximum likelihood under a Gaussian noise assumption. If you retain one derivation from the chapter, retain this one.

Model ladder, all built on the same loss:

1. **Linear regression**: $f(x;\boldsymbol{\theta}) = b + wx$, extending to $b + \boldsymbol{w}^\top \boldsymbol{x}$ for multiple inputs.
2. **Polynomial regression**: $f(\boldsymbol{x};\boldsymbol{w}) = \boldsymbol{w}^\top \boldsymbol{\phi}(\boldsymbol{x})$ with $\boldsymbol{\phi}(x) = [1, x, x^2, \dots, x^D]$. This is hand-designed **feature engineering**. Critically, the model is still linear in $\boldsymbol{w}$ even though it is nonlinear in $x$, which is what guarantees the MSE surface has a unique global optimum.
3. **Deep neural networks**: stop hand-designing $\boldsymbol{\phi}$ and give it its own parameters, $f(\boldsymbol{x};\boldsymbol{w},\mathbf{V}) = \boldsymbol{w}^\top \boldsymbol{\phi}(\boldsymbol{x};\mathbf{V})$, then decompose $\boldsymbol{\phi}$ into $L$ nested layers, $f(\boldsymbol{x};\boldsymbol{\theta}) = f_L(f_{L-1}(\cdots f_1(\boldsymbol{x})))$. The final layer stays linear; everything before it is a learned feature extractor.

---

## 4. Overfitting, generalization, and no free lunch

Rewrite empirical risk to make the dataset explicit, $L(\boldsymbol{\theta}; \mathcal{D}_{\text{train}})$. A flexible enough model drives this to zero by memorizing. A degree-20 polynomial through 21 points interpolates perfectly and predicts terribly.

Define **population risk** against the true data-generating distribution $p^*$:

$$L(\boldsymbol{\theta}; p^*) \triangleq \mathbb{E}_{p^*(\boldsymbol{x},y)}[\ell(y, f(\boldsymbol{x};\boldsymbol{\theta}))]$$

The **generalization gap** is $L(\boldsymbol{\theta}; p^*) - L(\boldsymbol{\theta}; \mathcal{D}_{\text{train}})$. Large gap means overfitting. Since $p^*$ is unknown, approximate it with **test risk** on held-out data.

The characteristic picture: training error decreases monotonically in model complexity; test error is U-shaped. Left of the minimum is underfitting, right is overfitting.

Practical consequence, and a common student error: you need **three** splits, not two. Train fits parameters, validation selects the model, test estimates future performance and is touched by neither of the first two.

**No free lunch theorem**: no single model is optimal across all problems, because a set of assumptions (an **inductive bias**) that helps in one domain hurts in another. Hence the practical advice to carry a large toolbox and select via cross-validation or Bayesian methods.

---

## 5. Unsupervised learning

Data is $\mathcal{D} = \{\boldsymbol{x}_n\}$ with no labels. Probabilistically, fit an unconditional model $p(\boldsymbol{x})$ rather than a conditional $p(y \mid \boldsymbol{x})$.

Three arguments in its favor: labels are expensive; many labeling tasks are genuinely ill-defined (Murphy's example is deciding exactly when the action of drinking begins in a video, where human annotators disagree); and modeling high-dimensional inputs forces richer world models than modeling low-dimensional outputs. The Hinton quotation makes the bandwidth argument: supervision delivers a few bits per second, the input itself delivers orders of magnitude more.

Main varieties covered:

- **Clustering**: partition inputs into regions of similar points. There is no correct $K$; you trade model complexity against fit.
- **Latent factor discovery**: assume each observed $\boldsymbol{x}_n \in \mathbb{R}^D$ was generated by hidden $\boldsymbol{z}_n \in \mathbb{R}^K$ with $K \ll D$. Linear Gaussian version is factor analysis, $p(\boldsymbol{x}_n \mid \boldsymbol{z}_n;\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{x}_n \mid \mathbf{W}\boldsymbol{z}_n + \boldsymbol{\mu}, \boldsymbol{\Sigma})$; setting $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ gives probabilistic PCA. Replacing the linear map with a neural net gives variational autoencoders.
- **Self-supervised learning**: manufacture proxy supervised tasks from unlabeled data (colorize a grayscale image, predict masked words) so that standard supervised machinery learns transferable features without ever inferring true latent factors.

**Evaluation is the hard part**, since there is no ground truth. Options: density estimation quality via unconditional NLL $L(\boldsymbol{\theta};\mathcal{D}) = -\frac{1}{|\mathcal{D}|}\sum_{\boldsymbol{x}} \log p(\boldsymbol{x}\mid\boldsymbol{\theta})$, which is difficult in high dimensions and rewardable by memorization; or **sample efficiency** on a downstream supervised task, which is more practical. In scientific settings the actual goal is often interpretability and understanding, not benchmark performance.

---

## 6. Reinforcement learning

The agent learns a policy $a = \pi(\boldsymbol{x})$ mapping states to actions. The distinguishing feature is not the presence of sequential structure but the **weakness of the training signal**: the system is never told which action was correct, only given occasional reward. Learning with a critic, not a teacher.

Two difficulties follow: reward is sparse, and credit assignment is ambiguous (which of a hundred chess moves caused the loss?). Hence the practice of supplementing with expert demonstrations or unsupervised structure learning. LeCun's cake analogy captures the information budget: unsupervised learning is the sponge, supervised is the icing, RL is the cherry.

---

## 7. Data and preprocessing

**Image benchmarks**: MNIST (60k/10k, 28x28 grayscale digits, now considered too easy since many digit pairs separate on a single pixel), EMNIST (62 classes including letters, genuinely ambiguous cases like 1 vs lowercase l), Fashion-MNIST (identical shape, harder), CIFAR (60k, 32x32x3, 10 or 100 classes), ImageNet (~14M images, 256x256x3, 20k classes; the ILSVRC subset used 1.3M images and 1000 classes).

Two notes worth carrying: CIFAR-100 has roughly 5.85% base label error, so any reported accuracy above about 94% deserves suspicion, and 10% of its training images are duplicated in the test set. And ImageNet superhuman performance in 2015 does not mean CNNs see better than humans; it mostly reflects fine-grained distinctions (tiger vs tiger cat) that humans find unintuitive.

**Text benchmarks**: IMDB (25k/25k, binary sentiment), WMT (English-German translation), SQuAD (question answering framed as seq2seq). Language modeling, $p(x_1,\dots,x_T)$, is unconditional generation and therefore a form of unsupervised learning.

**Preprocessing discrete inputs**:

- **One-hot encoding**: $\text{one-hot}(x) = [\mathbb{I}(x=1),\dots,\mathbb{I}(x=K)]$.
- **Feature crosses**: concatenated one-hots capture main effects only. To capture interactions (US trucks being less efficient than Japanese trucks, beyond the separate effects of truck and US), build composite features with $3 \times 2$ values. This blows the design matrix out into wide format.

**Preprocessing text** faces three problems: variable document length, high-dimensional one-hot word vectors with no similarity structure, and unseen words at test time.

- **Bag of words**: ignore order, count tokens. $\tilde{x}_{nv} = \sum_{t=1}^{T} \mathbb{I}(x_{nt} = v)$. Documents become vectors in $\mathbb{R}^D$, the vector space model.
- **TF-IDF**: raw counts overweight frequent, low-content words. Log-transform within documents and downweight across them with $\text{IDF}_i = \log \frac{N}{1 + \text{DF}_i}$, giving $\text{TFIDF}_{ij} = \log(\text{TF}_{ij}+1) \times \text{IDF}_i$.
- **Word embeddings**: TF-IDF still leaves man and woman no closer than man and banana, which breaks the implicit assumption that nearby inputs have similar outputs. Map one-hot $\boldsymbol{x}_{nt}$ to dense $\boldsymbol{e}_{nt} = \mathbf{E}\boldsymbol{x}_{nt}$ with $\mathbf{E} \in \mathbb{R}^{K \times V}$ learned so semantic neighbors are geometric neighbors. Sum or average to get a fixed-length document vector, then $p(y=c\mid\boldsymbol{x}_n,\boldsymbol{\theta}) = \text{softmax}_c(\mathbf{W}\mathbf{E}\tilde{\boldsymbol{x}}_n)$.
- **Out-of-vocabulary words**: replacing with UNK discards information. Better to exploit subword structure via wordpieces built with byte-pair encoding.

**Missing data**: let $M_{nd} = 1$ if feature $d$ of example $n$ is missing.

- MCAR: $p(\mathbf{M} \mid \mathbf{X}_v, \mathbf{X}_h, \mathbf{Y}) = p(\mathbf{M})$. Missingness depends on nothing.
- MAR: $p(\mathbf{M} \mid \mathbf{X}_v, \mathbf{X}_h, \mathbf{Y}) = p(\mathbf{M} \mid \mathbf{X}_v, \mathbf{Y})$. Depends on observed features only.
- NMAR: neither holds, so the missingness mechanism itself must be modeled because the absence is informative (someone declining to answer a sensitive survey question).

Under MCAR and MAR you can ignore the mechanism. The book assumes MAR throughout. Simplest fix is mean value imputation; better is fitting a generative model to fill values in.

---

## 8. Discussion and caveats

**Neighboring fields**: predictive analytics is supervised learning with a business framing; data mining spans both but emphasizes structured commercial databases; data science adds integration, visualization and domain-expert loops. Friedman's remark is that if statistics had adopted computation as a foundational tool rather than a convenience, ML might never have needed to exist as a separate field. Classical AI assumed intelligence could be hand-programmed and largely failed because encoding all needed knowledge proved intractable, which is what renewed interest in learned knowledge.

**Caveats**: specifying a loss function that captures all your preferences is hard, and optimizing a misspecified one produces **reward hacking**. This is an instance of the **alignment problem**, the gap between what you ask an algorithm to optimize and what you actually want. Russell's proposed remedy is inverse reinforcement learning: do not specify the reward, infer it from human behavior, formalized as a two-player cooperative assistance game where an uncertain machine asks rather than acting. Set against the AGI framing is **augmented intelligence** (IA), where AI builds tools that keep a human in the decision loop, closer in kind to autopilot or adaptive cruise control than to an autonomous agent.

---

## 9. Where to spend your reading time

**The critical cluster: printed pages 5 to 9 (PDF pages 35 to 39).**

Sections 1.2.1.3 through 1.2.2 inclusive. This is where the chapter stops describing and starts defining. In five pages you get the parameterized model, empirical risk minimization, asymmetric loss, the epistemic/aleatoric split, softmax and logits, logistic regression, maximum likelihood estimation, and the MSE-equals-Gaussian-NLL result. Every subsequent chapter of the book is a variation on this material. If you only read five pages, read these, and read them slowly enough to reproduce equations 1.4, 1.6, 1.14 and 1.21 from memory.

**Second priority: printed pages 12 to 13 (PDF pages 42 to 43).**

Section 1.2.3 on overfitting and generalization, plus 1.2.4 on no free lunch. Short, and it is the conceptual correction to everything in the critical cluster. The three-way train/validation/test split and the U-shaped test error curve are the two things practitioners get wrong most often.

**Skimmable: printed pages 19 to 26 (PDF pages 49 to 56).**

Section 1.5 on datasets and preprocessing is reference material. Know that TF-IDF, feature crosses, byte-pair encoding and the MCAR/MAR/NMAR taxonomy exist, and come back when you need them.

**Read once, do not labor over: printed pages 14 to 18 and 27 to 29.**

Sections 1.3, 1.4 and 1.6 are motivational and taxonomic. Useful for orientation, expanded properly in Parts V and in the sequel volume.

**Pages 33 onward** begin Chapter 2 with the frequentist/Bayesian split and the basic probability axioms. Murphy adopts the Bayesian interpretation because it handles one-off events, but notes the rules are identical either way.
