# Small-data text classification, embedding models, tabular foundation models and sequence models on an Apple M4 (as of 12 Oct 2026)

Scope note: written for a hand-raise predictor. It has decision points every ~2 s, 100k–1M rows, about 1–5% positives, 150–600 class sessions, and runs on an Apple M4 laptop with PyTorch MPS.

**Local M4 measurements.** Several findings below cite measurements I took on the target machine. The machine is an Apple M4 with 16 GB unified memory, running torch 2.14.1 and transformers 5.19.0. I used random-initialised weights, because speed does not depend on weight values. Every run was fp32 and single-shot, and tokenisation time is excluded. Treat the numbers as ±20% order-of-magnitude figures, not a formal benchmark. The script is `bench.py` in the session scratchpad. It is reproducible in about 5 minutes with `uv pip install torch transformers`. These figures are cited as "[local M4 benchmark]".

| Model (HF id) | Params | Infer MPS, 32 tok (sent/s) | Infer MPS, 256 tok | Infer CPU, 32 tok | Infer CPU, 256 tok | Train MPS fwd+bwd+AdamW, 128 tok, bs16 (samples/s) | Train MPS, 256 tok, bs16 | MPS driver memory while training (128 / 256 tok) |
|---|---|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 23M | 2,921 | 385 | 1,151 | 128 | 147 | 67 | 1.1 / 2.1 GB |
| BAAI/bge-small-en-v1.5 (12 layers, 384-d) | 33M | 1,491 | 188 | 584 | 64 | 79 | 34 | 2.2 / 4.1 GB |
| BAAI/bge-base-en-v1.5 (BERT-base architecture) | 109M | 471 | 59 | 188 | 23 | 29 | 14 | 4.4 / 7.1 GB |
| distilbert/distilbert-base-uncased | 66M | 927 | 116 | 371 | 46 | 56 | 28 | 3.0 / 4.2 GB |
| answerdotai/ModernBERT-base | 149M | 293 | 34 | 129 | 15 | 19 | 9 | 5.5 / 8.2 GB |
| microsoft/deberta-v3-small | 141M | 525 | 24 | 180 | 22 | 28 | 11 | 6.0 / 8.0 GB |
| google/embeddinggemma-2 (text tower only) | 271M | 164 | 19 | – | – | – | – | – |

Mixed precision gave almost nothing on MPS. DistilBERT training at 256 tokens ran at 28.9 samples/s in fp32, 29.7 with bf16 autocast and 29.8 with fp16 autocast [local M4 benchmark].

## 1. Which small sentence-embedding models are strongest in 2026, and which suit short conversational/ASR text?

### Takeaway
No benchmark covers classroom ASR text, so pick by the MTEB *Classification* column and by speed.

- **Best quality per millisecond:** `BAAI/bge-small-en-v1.5` (MIT licence, 33M parameters, 384-d).
- **Fastest:** `all-MiniLM-L6-v2`, which is already planned.
- **Strongest under 500M parameters:** EmbeddingGemma. The original `google/embeddinggemma-300m` is gated under the Gemma licence. The new `google/embeddinggemma-2` (Sept 2026) is Apache 2.0. On the M4, the EmbeddingGemma-2 text tower measured about 9–10× slower than bge-small and about 18–20× slower than MiniLM.

Because embedding cost is dominated by sequence length, embed each utterance once and cache it, rather than re-embedding every 60 s window.

### Cited Findings

**How MTEB scores relate to this project**
- MTEB scores its Classification tasks by embedding the train and test sets and training "a logistic regression classifier with 100 maximum iterations" on the frozen train embeddings. The MTEB Classification column is therefore a direct proxy for the planned "frozen embedding + linear head" approach. — [Muennighoff et al., MTEB, EACL 2023, arXiv:2210.07316](https://arxiv.org/abs/2210.07316)
- MTEB's classification tasks include short-utterance and conversational sets: Banking77, MassiveIntent, MassiveScenario, Emotion, ToxicConversations and TweetSentimentExtraction. The paper finds "no single best solution, with different models dominating different tasks." — [MTEB paper](https://arxiv.org/abs/2210.07316)
- The Sentence-Transformers docs advise using the MTEB leaderboard for inspiration but warn that "models that perform well on the leaderboard do not necessarily do well on your tasks." The docs also state that all-mpnet-base-v2 gives the best quality of the original models, while all-MiniLM-L6-v2 "is 5 times faster and still offers good quality." — [SBERT pretrained models docs](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

**BGE, GTE and E5 (MTEB English v1)**
- The BGE model-card table reports MTEB average / Classification as follows. — [BAAI/bge-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)

  | Model | Dimension | MTEB average | Classification |
  |---|---|---|---|
  | bge-small-en-v1.5 | 384 | 62.17 | 74.14 |
  | bge-base-en-v1.5 | 768 | 63.55 | 75.53 |
  | bge-large-en-v1.5 | 1024 | 64.23 | 75.97 |
  | gte-base | 768 | 62.39 | 73.01 |
  | e5-small-v2 | 384 | 59.93 | 72.94 |
  | e5-base-v2 | 768 | 61.5 | 73.84 |

  bge-small has 33.4M parameters and an MIT licence.

**EmbeddingGemma-300m (MTEB English v2)**
- EmbeddingGemma-300m (308M parameters) has the following scores on MTEB(Eng, v2). — [EmbeddingGemma paper, arXiv:2509.20354](https://arxiv.org/abs/2509.20354)
  - At 768-d: Mean(Task) 69.7, Classification 87.6, PairClassification 87.3, STS 83.6.
  - It still reaches Classification 85.9 when truncated to 128-d.
  - The next-best models under 500M are GIST-large and mxbai-embed-large-v1 (335M). Both score Mean(Task) 66.3, with Classification 78.9 and 79.1.
  - It uses task prefixes such as `task: classification | query:`.
- Google reports EmbeddingGemma at MTEB Multilingual v2 61.15 and English v2 69.67, calling it the top multilingual model under 500M. It can run in under 200 MB RAM when quantised. — [Google Developers Blog](https://developers.googleblog.com/en/introducing-embeddinggemma/)
- `google/embeddinggemma-300m` is gated (manual acceptance) under the `license:gemma` tag. It was created on HF 2025-07-17. — [HF API](https://huggingface.co/google/embeddinggemma-300m)

**EmbeddingGemma 2 (released Sept 2026)**
- `google/embeddinggemma-2` was created 2026-09-14 under Apache 2.0 and is not gated. It is a 740M-parameter multimodal model, but loading text only (`config_kwargs={"vision_config": None, "audio_config": None}`) gives 270M parameters. — [embeddinggemma-2 model card](https://huggingface.co/google/embeddinggemma-2)
  - Context is 8K tokens, with Matryoshka truncation to 512, 256 or 128 dimensions.
  - At 768-d it scores MTEB(eng, v2) Mean(Task) 68.46. That is slightly below EmbeddingGemma-1's 69.67; the gains are in code and multimodal tasks.
  - It has a "Classification" prompt: `task: classification | query: {content}`.
  - The card warns "Do not use `float16`" because the activations overflow and return NaN or silently degraded embeddings. Use bf16 or fp32.
- On the M4, the EmbeddingGemma-2 text tower encoded about 164 sentences/s at 32 tokens and 19/s at 256 tokens. That is about 18× (32 tokens) and 20× (256 tokens) slower than all-MiniLM-L6-v2. [local M4 benchmark]

**ModernBERT-based embedders**
- `Alibaba-NLP/gte-modernbert-base` (Apache-2.0, Jan 2025) scores MTEB average 64.38. `nomic-ai/modernbert-embed-base` (Apache-2.0, Dec 2024) scores 62.62. Both have 149M parameters and 768-d output. — [gte-modernbert-base model card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base); [modernbert-embed-base model card](https://huggingface.co/nomic-ai/modernbert-embed-base)
- On the M4, ModernBERT-base was the slowest encoder tested: 293 sentences/s on MPS at 32 tokens, against 1,491/s for bge-small. [local M4 benchmark] The reason is that its speed advantage depends on FlashAttention unpadding, which MPS lacks (see Q3).

**Nomic, Granite and all-MiniLM**
- `nomic-ai/nomic-embed-text-v1.5` (137M, Apache-2.0, Feb 2024) scores MTEB 62.28 at 768-d and 61.04 at 256-d. It requires the prefix `classification: ` for classifier features. Its card loads it with `trust_remote_code=True`. — [nomic-embed-text-v1.5 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- IBM's Granite R2 paper reports the following on MTEB-v2 (Apache 2.0). Granite-small encodes about 38% faster on an H100. — [Granite Embedding R2, arXiv:2508.21085](https://arxiv.org/abs/2508.21085)

  | Model | Params | Average | Classification |
  |---|---|---|---|
  | granite-embedding-small-english-r2 | 47M | 61.1 | 68.2 |
  | e5-small-v2 | 33M | 61.3 | 74.6 |
  | bge-small-en-v1.5 | 33M | 64.3 | 76.6 |

- `sentence-transformers/all-MiniLM-L6-v2` has 23M parameters, 384-d output and a 256-token maximum. It was released Aug 2021 under Apache-2.0. — [MTEB leaderboard model page](https://leaderboard.mteb.org/models/sentence-transformers/all-MiniLM-L6-v2) Its card states that "input text longer than 256 word pieces is truncated." — [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**Licences and dates of other 2025–26 checkpoints (scores not verified)**
These come from HF API licence tags and creation dates. — [Hugging Face Hub API](https://huggingface.co/api/models?pipeline_tag=feature-extraction&sort=downloads)

| Model | Licence | HF creation date |
|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | 2025-06 |
| microsoft/harrier-oss-v1-0.6b | MIT | 2026-03 |
| voyageai/voyage-4-nano | Apache-2.0 | 2026-01 |
| perplexity-ai/pplx-embed-v1-0.6b | MIT | 2026-01 |
| Snowflake/snowflake-arctic-embed-s / -m-v2.0 | Apache-2.0 | 2024 |
| mixedbread-ai/mxbai-embed-xsmall-v1 | Apache-2.0 | 2024 |
| jinaai/jina-embeddings-v3 | CC-BY-NC-4.0 (non-commercial) | – |
| jinaai/jina-embeddings-v5-omni-small | CC-BY-NC-4.0 (non-commercial) | 2026-04 |

- Library status: sentence-transformers 6.1.0 is the current PyPI release. — [PyPI](https://pypi.org/project/sentence-transformers/)

### Inferences

**Compute estimates on the M4 (from the local benchmark)**
- Per-utterance approach: 600 classes × about 1,000 ASR utterances is about 600k short texts at about 32 tokens. That takes about 3.5 min with MiniLM-L6, 7 min with bge-small, 21 min with bge-base, and about 1 h with EmbeddingGemma-2 on MPS.
- Per-decision-window approach: 1M windows at 256 tokens takes about 45 min with MiniLM, 1.5 h with bge-small, 4.7 h with bge-base, and about 15 h with EmbeddingGemma-2.
- Recommendation: embed each utterance once, cache it to disk, and build window features by (time-decayed) mean-pooling the cached utterance vectors. A window only changes when a new utterance arrives, so most 2 s decision points can reuse the previous vector.

**Model choice**
- **Default:** `BAAI/bge-small-en-v1.5`. It scores higher on MTEB Classification than MiniLM-L6, e5-small-v2 and granite-small. It is MIT licensed, needs no prefix and is about 2× MiniLM's cost.
- **Comparison point:** keep `all-MiniLM-L6-v2` as the planned baseline.
- **Answer-similarity feature:** for semantic similarity between the current utterance and his submitted answers, compute once per utterance with the strongest affordable model. That is `google/embeddinggemma-2` with the `SentenceSimilarity` prompt, or `bge-base-en-v1.5`. The answer set is tiny, so quality matters more than speed here.
- **Testing:** because MTEB shows no universal winner, test 2–3 embedders under the same grouped-by-class CV split, not just one.
- **Prefix models:** for EmbeddingGemma and nomic, the prefix is part of the model contract. Forgetting it silently degrades features.
- **ASR noise:** ASR text is lower-cased, disfluent and lacks punctuation. MTEB includes noisy social text (TweetSentimentExtraction, ToxicConversations), but no ASR transcripts, so robustness to ASR noise is untested for all of these models.
- **Avoid for this project:** CC-BY-NC models (Jina) are fine for a class project but not reusable commercially. Gated Gemma-licence weights need an HF login and acceptance step. EmbeddingGemma-2 avoids that.

### Gaps
- I found no benchmark of embedding models on ASR or lecture/seminar transcripts, or on predicting turn-taking or participation from text.
- I could not get MTEB(eng, v2) Classification scores for all-MiniLM-L6-v2, Qwen3-Embedding-0.6B, harrier-oss-v1-0.6b, voyage-4-nano or pplx-embed from a primary page. The MTEB leaderboard page rendered "Loading…".
- MTEB v1 and v2 numbers are not directly comparable (different task sets). Mixing columns from the BGE card (v1) and the EmbeddingGemma paper (v2) is only indicative.
- The M4 speeds are my own single-run fp32 measurements with random weights on a 10-core-GPU base M4. They are not a published benchmark. Real text adds tokenisation and padding overhead.

## 2. SetFit: how it works, data efficiency, library status, imbalanced binary tasks, MPS, and training time

### Takeaway
SetFit is designed for few-shot data (8–64 examples per class). This project has a few hundred to a few thousand positives and 100k+ negatives, so SetFit's contrastive pair stage brings little and must be capped or subsampled. With default settings on extreme imbalance, one user reported F1 of only about 0.18. Use it as a cheap "contrastively tuned embedder + LR head" variant on a balanced subsample, not as the main model.

### Cited Findings
- SetFit (Tunstall, Reimers, Jo, Bates, Korat, Wasserblat, Pereg; arXiv 22 Sep 2022) works in two stages. It contrastively fine-tunes a Sentence Transformer on sentence pairs, then fits a classification head. It "achieves high accuracy with orders of magnitude less parameters" and is "an order of magnitude faster to train" than PEFT/PET. — [arXiv:2209.11055](https://arxiv.org/abs/2209.11055)
- Library status: `setfit` v1.2.0 was released 2026-09-04. Earlier releases were v1.1.3 (2025-08-05), v1.1.2 (2025-04-03), v1.1.1 (2025-01-13, ModernBERT support) and v1.1.0 (2024-09-19, Sentence-Transformers became the fine-tuning backend). — [GitHub releases](https://github.com/huggingface/setfit/releases); [PyPI](https://pypi.org/project/setfit/)
  - v1.2.0 supports transformers v4.41–5.x, sentence-transformers v3.0–6.0 and Python 3.9–3.13.
  - On PyPI it requires `sentence-transformers[train]>=3` and `transformers>=4.41.0`.
- Pair sampling: `sampling_strategy` defaults to `"oversampling"`, which "draws even number of positive/negative sentence pairs until every sentence pair has been drawn." `"undersampling"` and `"unique"` are the alternatives. `num_iterations` or `max_steps` cap the number of pairs. — [SetFit TrainingArguments docs](https://huggingface.co/docs/setfit/en/reference/trainer)
- Training defaults: `batch_size=(16, 2)`, `num_epochs=(1, 16)`, `body_learning_rate=(2e-5, 1e-5)`, `head_learning_rate=0.01`, `end_to_end=False`, `use_amp=False`, loss `CosineSimilarityLoss`. — [SetFit TrainingArguments docs](https://huggingface.co/docs/setfit/en/reference/trainer)
- On an extremely imbalanced binary dataset (about 99.2% / 0.8%), a user reported F1 of about 0.18 with default `CosineSimilarityLoss`. — [setfit issue #161](https://github.com/huggingface/setfit/issues/161)
- Frozen sentence embeddings work as classifier inputs but are "limited compared to standard full fine-tuning". SetFit "significantly improves few-shot classification" by contrastive tuning before the head. — [AdaSent, arXiv:2311.00408](https://arxiv.org/abs/2311.00408)

### Inferences
- **Scale problem:** with the default oversampling and 100k+ texts, the number of possible pairs is astronomically large. Set `max_steps` (for example 2k–5k steps of batch 16) or train on a balanced subsample. For example, use all positives plus 2–5× as many hard negatives, drawn from windows near raises and from windows ending in a question.
- **Imbalance handling:** fix the imbalance outside SetFit. Calibrate the LR head with `class_weight="balanced"`, or fit your own `sklearn.linear_model.LogisticRegression` on the SetFit-tuned embeddings. Then tune the decision threshold on grouped CV for PR-AUC or F1.
- **MPS:** SetFit's body training now runs through sentence-transformers' Trainer, which is built on the HF Trainer. That Trainer picks MPS automatically on Apple Silicon, but I did not find SetFit-specific MPS docs.
- **Rough training time (bge-small body on M4 MPS):** from the local benchmark, roughly 30–40 pairs/s at 128 tokens, so 5k steps × 16 pairs is about 40 min. With MiniLM-L6 it is about half that. These are estimates, not measurements of SetFit itself.
- **Label noise:** the label is "he raised his hand in the next k seconds", and it is driven by timing more than by semantic content. Contrastive pairs built from such weak, noisy labels may teach the embedder little. Expect gains over frozen bge-small + LR to be small or zero. Report this honestly as a negative result if it happens.

### Gaps
- I found no published evaluation of SetFit at thousands of examples with 1–5% positives. The SetFit paper focuses on N ≤ 64 per class. I could not verify its learning curves at large N from the abstract.
- I found no SetFit-specific documentation of MPS behaviour or speed.

## 3. Fine-tuning small encoders (DistilBERT, ModernBERT-base, DeBERTa-v3-small) on Apple MPS: throughput, issues, memory, and value vs frozen embeddings

### Takeaway
Fine-tuning is feasible on an M4 but slow. Measured training speeds at 256 tokens with batch 16:

| Model | Samples/s | Peak MPS memory |
|---|---|---|
| DistilBERT | about 28 | about 4 GB |
| DeBERTa-v3-small | about 11 | about 8 GB |
| ModernBERT-base | about 9 | about 8 GB |

So one epoch over a 33k-row negative-downsampled set takes about 20 min with DistilBERT and about 1 h with ModernBERT. ModernBERT loses its speed advantage on MPS (no FlashAttention unpadding), and mixed precision gives under 5%. With a few hundred to a few thousand weak positives, start with frozen embeddings + linear/GBDT. Treat DistilBERT fine-tuning as a stretch ablation.

### Cited Findings
- ModernBERT (Warner et al., arXiv 18 Dec 2024) was trained on 2T tokens with a native 8192-token context. It claims SOTA on classification and retrieval, and calls itself "the most speed and memory efficient encoder... designed for inference on common GPUs." — [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)
- A ModernBERT maintainer explained that ModernBERT "uses Flash Attention to support full model unpadding, so any backend which doesn't have a Flash Attention equivalent will require padding tokens and won't benefit from those speedups." On an unspecified Apple Silicon Mac, a user measured about 120 docs/s on MPS for ModernBERT against 150 docs/s for DeBERTa-v3. — [HF discussion #47, ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base/discussions/47)
- A tutorial fine-tunes ModernBERT for classification on Apple Silicon with `use_mps_device=True`. Others report "RuntimeError: Placeholder storage has not been allocated on MPS device" when tensors and model are on different devices. — [HF blog: fine-tune ModernBERT on synthetic data](https://huggingface.co/blog/davidberenstein1957/fine-tune-modernbert-on-synthetic-data); [HF forum thread](https://discuss.huggingface.co/t/runtimeerror-placeholder-storage-has-not-been-allocated-on-mps-device/42999)
- MPS autocast was first fp16-only. The issue requesting bf16 autocast on MPS was closed 2024-11-20, with linked PRs for bf16 and fp16 AMP on MPS. — [pytorch/pytorch #139386](https://github.com/pytorch/pytorch/issues/139386); [PR #99272](https://github.com/pytorch/pytorch/pull/99272)
- On the M4 (torch 2.14.1), bf16 and fp16 autocast raised DistilBERT training throughput only from 28.9 to 29.7–29.8 samples/s. [local M4 benchmark]
- An Apple Silicon profiling study found Apple GPUs about 3–4× slower than similarly priced NVIDIA GPUs when the model fits in VRAM. — [Feng et al., "Profiling Apple Silicon Performance for ML Training", arXiv:2501.14925](https://arxiv.org/abs/2501.14925)
  - Memory use during training "shows a gradual increase, rather than maintaining relatively stable like NVIDIA GPUs."
  - At the time, "Torch AMP lacks support for Apple silicon SoCs."
  - FP16 gave MLX only about 20–30% benefit over FP32.
- Measured M4 training throughput (fwd + bwd + AdamW, batch 16) and peak MPS driver memory: [local M4 benchmark]

  | Model | 128 tokens (samples/s) | 256 tokens (samples/s) | Peak memory |
  |---|---|---|---|
  | DistilBERT | 56 | 28 | 4.2 GB |
  | DeBERTa-v3-small | 28 | 11.4 | 8.0 GB |
  | ModernBERT-base | 19 | 9.1 | 8.2 GB |
  | bge-small | 79 | 34 | 4.1 GB |
  | MiniLM-L6 | 147 | 67 | 2.1 GB |

  DeBERTa-v3-small inference at 256 tokens was barely faster on MPS (24/s) than on CPU (22/s).
- An old PyTorch issue reports "bad performance metrics for BERT model training" on M1 MPS (2022). That reflects early-MPS correctness problems, so sanity-check MPS against CPU on a small batch. — [pytorch/pytorch #82707](https://github.com/pytorch/pytorch/issues/82707)
- Licences and dates: DistilBERT (Apache-2.0), ModernBERT-base (Apache-2.0, Dec 2024) and DeBERTa-v3-small (MIT). — [HF API: distilbert](https://huggingface.co/distilbert/distilbert-base-uncased), [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), [deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small)

**Frozen embeddings vs fine-tuning**
- In AWS's multimodal-AutoML benchmark (18 tabular+text datasets), text featurisation scored as follows. The authors were "surprised to find both N-Grams and word2vec here provide superior text featurization than Pre-Embedding." — [Shi, Mueller, Erickson, Li, Smola, arXiv:2111.02705, NeurIPS 2021 Datasets & Benchmarks](https://arxiv.org/abs/2111.02705)

  | Featurisation | Average score |
  |---|---|
  | Frozen "Pre-Embedding" | 0.593 |
  | Fine-tuned "Text-Embedding" | 0.638 |
  | "Multimodal-Embedding" | 0.644 |

- With very few samples, frozen sentence embeddings beat fine-tuning. Concatenating pretrained embeddings with a small task-trained model beats both. — [Beyond Fine-tuning: Few-Sample Sentence Embedding Transfer, AACL 2020](https://aclanthology.org/2020.aacl-main.47.pdf)

### Inferences

**When fine-tuning pays off**
- The fine-tuning advantage in Shi et al. came from 2021-era frozen embeddings, not modern contrastive embedders like bge or EmbeddingGemma. Lexical n-grams already beat those frozen embeddings. TF-IDF + LR and modern frozen embedders may therefore close most of the gap here.
- The deciding factor is the effective number of positives (hundreds to a few thousand), not 100k–1M rows. The label is also partly non-textual: timing, his own state, and the instructor's prompts. In that regime a 66–149M-parameter fine-tune risks fitting class-specific vocabulary. Fold-to-fold variance under grouped-by-class CV may exceed the average gain.

**Practical recipe if you do fine-tune**
- Use `distilbert-base-uncased` or `bge-small` as the body, at 128 tokens (about the last 20–30 s of speech).
- Downsample negatives to about 10:1 and use a class-weighted loss.
- Run 2–3 epochs, early-stop on grouped validation PR-AUC, and train in fp32.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` only if an op is missing. Check one batch on CPU vs MPS.

**Choice of model**
- ModernBERT-base is the wrong choice on an M4. It is about 3× slower than DistilBERT here, and its long context is not needed for 60 s windows.
- DeBERTa-v3-small is often the strongest small fine-tuner on GPUs, but on MPS it is slow at 256 tokens and memory-heavy.

### Gaps
- I found no published, controlled M4 (or M-series) benchmark of BERT-family fine-tuning throughput. The table above is my own rough measurement.
- I found no study comparing modern frozen embedders (bge, EmbeddingGemma) + LR against fine-tuned DistilBERT or ModernBERT at about 1k positives and severe imbalance.
- Whether PyTorch ≥2.13's FlashAttention-on-MPS path (used by TabPFN, see Q4) helps HF ModernBERT unpadding is unverified.

## 4. Tabular foundation models (TabPFN v2 → 2.5 → 3 → 3.5, TabICL): limits, Apple Silicon/CPU, licence, use on 100k+ rows, and comparison with LightGBM/CatBoost

### Takeaway
The field moved fast. TabPFN-3.5 (Sept 2026) accepts up to 1M rows and 20k features, and runs on Apple Silicon via MPS (PyTorch ≥2.13 recommended). On CPU it is capped at 5,000 samples by default, and its weights are non-commercial and need a Prior Labs login. TabICLv2 (v2.2.0, Sept 2026) is the fully open alternative: BSD-3 code, MPS support, CPU and disk offload, tested to about 600k rows, best at 100 or fewer features.

On a 16 GB M4 with about 1–5% positives, the realistic use is:
1. Downsample negatives to a 10k–50k-row context.
2. Ensemble over several subsamples.
3. Fix the threshold with grouped CV.
4. Compare against LightGBM.

Do not assume it beats LightGBM. All "beats tuned GBDT" claims are vendor-reported and measured on H100s.

### Cited Findings

**TabPFN v2 (Nature, Jan 2025)**
- Hollmann et al., "Accurate predictions on small data with a tabular foundation model", *Nature* (Jan 2025). It is pretrained on millions of synthetic datasets and predicts by in-context learning in one forward pass. — [Nature, doi:10.1038/s41586-024-08328-6](https://www.nature.com/articles/s41586-024-08328-6); [TabPFN-TS paper describing v2](https://arxiv.org/abs/2501.02945)

**Checkpoint limits (Prior Labs models page)**
Source: [Prior Labs models docs](https://docs.priorlabs.ai/models)

| Checkpoint | Released | Max rows | Max features | Max classes | Licence |
|---|---|---|---|---|---|
| TabPFNv2 | 2025 | 10,000 | 500 | 10 | Prior Labs License |
| TabPFN-2.5 | Nov 2025 | 100,000 | 2,000 | 10 | TABPFN-2.5 License v1.1 |
| TabPFN-2.6 | Apr 2026 | 100,000 | 2,000 | 10 | – |
| TabPFN-3 / 3-Plus | May 2026 | 1,000,000 | 2,000 | 160 | – |
| TabPFN-3-Thinking | May 2026 | 200,000 | – | – | – |
| TabPFN-3.5 / 3.5-Plus / 3.5-Fast | Sep 2026 | 1,000,000 | 20,000 | 160 | TABPFN-3.5 License v1.0 |
| TabPFN-3.5-Thinking | Sep 2026 | 200,000 | – | – | – |

**TabPFN package, README and licensing**
- The `tabpfn` package is v9.2.0 on PyPI (released 2026-10-12) and requires `torch>=2.5`. TabPFN-3.5 is the default. — [GitHub releases](https://github.com/PriorLabs/TabPFN/releases); [PyPI](https://pypi.org/project/tabpfn/)
- The README says:
  - "On CPU, only moderate datasets are feasible (TabPFN-3, TabPFN-3.5 and TabPFN-3.5-Fast allow up to 5000 samples; older versions up to 1000)." Override with `TABPFN_ALLOW_CPU_LARGE_DATASET=true`.
  - "On macOS: GPU support is automatically included for Apple Silicon Macs. For best performance, ensure you are using PyTorch 2.13 or newer." PR #949, "Add support for FlashAttn on MPS", was merged 2026-05-13.
  - `TABPFN_MPS_MEMORY_FRACTION` defaults to 0.7 "to prevent macOS system crashes."
  - Above the limits: "subsample, set `ignore_pretraining_limits=True`... or upgrade."
  - Batch your predictions: "Each predict call recomputes the training set". Also: "Do not apply data scaling or one-hot encoding."
  - Missing values are handled natively.
  - Code is Apache-2.0. The TabPFN-2.5/2.6/3/3.5 weights are non-commercial. TabPFN-2 weights use the Prior Labs License (Apache 2.0 plus attribution).
  - First use opens a browser login to accept the licence; headless use needs `TABPFN_TOKEN`.

  — [TabPFN README](https://github.com/PriorLabs/TabPFN/blob/main/README.md); [PR #949](https://github.com/PriorLabs/TabPFN/pull/949)
- Current PyTorch on PyPI is 2.14.1, so the ≥2.13 recommendation is installable. — [PyPI torch](https://pypi.org/project/torch/)
- TabPFN-2.5's licence forbids using the model or its outputs for commercial or business decisions. — [TabPFN-2.5 paper, arXiv:2511.08667](https://arxiv.org/abs/2511.08667)

**Vendor performance claims**
- TabPFN-2.5 "outperforms any other model in a forward pass" on TabArena-lite. Results to about 100k training rows were validated, and up to about 160k × 500 fit on an 80 GB H100 in exploratory runs. — [TabPFN-2.5 paper, arXiv:2511.08667](https://arxiv.org/html/2511.08667v2)
- TabPFN-3 (technical report, 13 May 2026) is "up to 20x faster than TabPFN-2.5". It "beats 8-hour-tuned gradient-boosted-tree baselines on datasets up to 1M training rows and 200 features", and scales "to 1M rows on one H100." — [arXiv:2605.13986](https://arxiv.org/abs/2605.13986)
- TabPFN-3.5 (22 Sep 2026) claims to handle "non-i.i.d. data, temporal/grouped splits, multimodal inputs (strings, text, images)". Its Fast variant is "up to 3x faster than TabPFN-3." — [TabPFN-3.5 technical report, arXiv:2609.17895](https://arxiv.org/abs/2609.17895)

**Class imbalance in TabPFN-style models**
- PFNs' "performance can suffer under the effect of class imbalance, resulting in poor performance for rare classes". Loss-reweighting does not apply to in-context learning. Thresholding "performs exceptionally well because of the calibration characteristics of PFNs". Downsampling "performs comparably... with the additional benefit of reduced computation cost." — [McDowell, Stromberg, Sankar, arXiv:2605.21742 (May 2026)](https://arxiv.org/abs/2605.21742)
- A separate evaluation reports TabPFN v2 biased toward majority classes on imbalance-sensitive metrics. This is from a search snippet; I did not verify it in the full text. — [Realistic Evaluation of TabPFN v2 in Open Environments, arXiv:2505.16226](https://arxiv.org/abs/2505.16226)

**TabICL**
- TabICL v1 (ICML 2025) uses column-then-row attention plus an ICL transformer. It was pretrained on datasets of up to 60k rows and handles up to 500k samples. "On 53 datasets with over 10K samples, TabICL surpasses both TabPFNv2 and CatBoost." — [arXiv:2502.05564](https://arxiv.org/abs/2502.05564)
- TabICLv2 (ICML 2026) details: — [soda-inria/tabicl GitHub](https://github.com/soda-inria/tabicl); [GitHub releases](https://github.com/soda-inria/tabicl/releases); [PyPI](https://pypi.org/project/tabicl/)
  - The `tabicl` package is v2.2.0, released 2026-09-02, with a BSD-3-Clause licence. Install with `pip install tabicl` and use `from tabicl import TabICLClassifier`.
  - Supports MPS, CPU and disk offloading, and was tested up to about 600k rows.
  - Pretrained on 2–100 columns, with "performance degrades beyond 100".
  - Claims it "outperforms heavily tuned XGBoost, CatBoost, or LightGBM on TabArena on ~80% of datasets" and is "10x faster than TabPFN-2.5" on larger data.

### Inferences

**Fit to this project**
- The class-state tabular block (10–30 features) is squarely in the TabPFN/TabICL sweet spot. Adding 384–768 raw embedding dimensions is not: TabICL degrades above about 100 features. PCA the text embeddings to 16–32 dimensions, or pass out-of-fold text-model scores instead.
- On a 16 GB M4, the MPS memory cap (70% by default) and H100-validated limits mean ~1M-row contexts are impractical. A sound protocol:
  1. Use all positives plus a random 10–20× negative sample as context (5k–50k rows).
  2. Predict the validation fold in one batched call.
  3. Repeat over 3–5 negative subsamples and average.
  4. Choose the threshold on grouped validation data.
- Downsampling shifts predicted probabilities. Re-calibrate (prior correction or isotonic) before reporting calibration or Brier score.

**Time structure and alternatives**
- The data are temporally dependent within a class. The i.i.d. assumption behind TabPFN v2 and TabICL makes leakage easy if the context contains rows from the same class session as the query. Keep grouped-by-class splits for the context too. TabPFN-3.5 claims support for grouped/temporal splits, but how that works in the local package is unverified.
- LightGBM remains the low-risk main model: fast on CPU, handles 1M rows, uses `scale_pos_weight`, and has no licence or login friction. TabICLv2 is the best "foundation model" comparison to add. It is open and pip-installable, and runs on MPS without an account. TabPFN-3.5 is worth one run if the login and non-commercial licence are acceptable for coursework.

### Gaps
- I found no independent (non-vendor) benchmark of TabPFN-3/3.5 or TabICLv2 at 100k+ rows with 1–5% positives, and no Apple Silicon timing data for either.
- I did not verify the TabICLv2 checkpoint licence separately from the BSD-3 code licence.
- I did not find whether TabPFN-3.5's text-column support works locally or only through the Prior Labs API.

## 5. Sequence models over utterance embeddings (hierarchical RNN/transformer, TCN, streaming/causal): data needs and overfitting risk

### Takeaway
The architectures are well established:
- Hierarchical encoders over utterances (HRED).
- Causal TCNs, which match or beat LSTMs/GRUs on generic sequence benchmarks.
- Continuous, frame-level turn-taking predictors (Voice Activity Projection).

However, each class session is one sequence, so there are only 150–600 independent sequences and a few thousand positive events. A small causal GRU or TCN over compressed inputs is defensible:
- Inputs: PCA'd utterance embeddings plus tabular state.
- Size: hidden 32–64, about 10–50k parameters.
- Regularisation: dropout and early stopping on grouped CV.

A transformer over utterances is high-risk. A LightGBM with hand-built lag and rolling features is the bar it must beat.

### Cited Findings
- Bai, Kolter & Koltun (2018): "a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets". This is the reference generic causal TCN. — [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)
- Serban et al. (2015/2016) extended the hierarchical recurrent encoder-decoder (HRED) to dialogue. It is a word-level encoder per utterance, followed by an utterance-level RNN over the conversation, which is the template for a GRU over utterance embeddings. — [arXiv:1507.04808](https://arxiv.org/abs/1507.04808)
- Ekstedt & Skantze (2022), Voice Activity Projection (VAP), frame turn-taking as continuous prediction of upcoming voice activity. It is trained self-supervised "without the need of labeled data". This is the closest analogue to "will he speak or raise his hand in the next k seconds". — [arXiv:2205.09812](https://arxiv.org/abs/2205.09812)
- Recent multimodal and multi-party turn-taking work uses transformer models over interlocutor states and gaze, and notes overfitting risk on small datasets. — [Multimodal Turn Analysis and Prediction for Multi-party Conversations (ICMI 2023)](https://dl.acm.org/doi/fullHtml/10.1145/3577190.3614139); [Exploring Multimodal Turn-Taking Cues… using VAP, arXiv:2609.14666](https://arxiv.org/html/2609.14666)
  - One example from search snippets: models trained on the 7-hour French subset of NoXi were competitive, "though the small dataset limits generalisation." I did not verify this in full text.

### Inferences

**Data budget**
- Effective sample size: 150–600 sessions × about 1–5% positive steps gives a few hundred to a few thousand positive events, clustered within sessions. That supports models with thousands to tens of thousands of parameters, not millions.

**Concrete low-risk design**
1. Inputs per step: cached utterance embedding (PCA → 32 dimensions), plus 10–30 tabular features, plus a speaker-change flag.
2. A 1-layer causal GRU (hidden 32–64), or a 3–4-block causal TCN with dilations 1, 2, 4, 8. At 2 s steps, a 4-block TCN with kernel 3 has a receptive field of about 1 min, which matches the window.
3. Train on chunks of about 5 min (truncated BPTT) sampled from training sessions.
4. Use per-step BCE with `pos_weight` or focal loss; use dropout 0.3 and weight decay.
5. Early-stop on grouped-by-class validation PR-AUC.

On an M4, training takes minutes even on CPU.

**Causality checks**
- Only use utterances whose ASR end-time is before the decision time.
- Never use bidirectional RNNs.
- Never normalise with statistics from the whole session, including future steps.

**Comparison baseline**
- A LightGBM on the same inputs plus engineered history features (time since last raise, rolling counts, decayed text-score averages) often captures most of what a GRU learns. If the GRU cannot beat it on grouped CV, that is a valid finding.
- A per-class "transformer over utterances" with 4+ layers and 384-d inputs has more than 1M parameters. With a few hundred sequences it will likely overfit to session identity. Use it only with heavy regularisation and as an ablation.

### Gaps
- I found no published work predicting hand-raises or student participation in online seminars from transcripts, so there are no direct data-size or accuracy anchors.
- I found no quantitative rule for how many sequences a GRU or TCN needs in this regime. The sizing advice above is inference.

## 6. TextCNN (Kim 2014) today: when does it still help vs pretrained embeddings?

### Takeaway
TextCNN is a cheap, fast n-gram cue detector. It can learn phrases like "any questions", "raise your hand" or the student's name from data. With limited labels it usually does no better than TF-IDF + LR, and simple models can beat it outright. Keep it as a didactic, interpretable middle rung between TF-IDF and transformer embeddings, not as an expected winner.

### Cited Findings
- Kim (2014): "a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks". Fine-tuning the word vectors gives further gains, and it improved SOTA on 4 of 7 tasks. — [arXiv:1408.5882](https://arxiv.org/abs/1408.5882)
- With very limited training data, simple models generally beat CNN and BERT. On 20 Newsgroups with 320 training examples, CNN macro-F1 was 43.9 against 61.7 for a simple controlled-generalisation classifier. I took these numbers from a search snippet and did not check them against the full text. — [Text Classification with Few Examples using Controlled Generalization, arXiv:2005.08469](https://arxiv.org/abs/2005.08469)
- N-gram features beat frozen pretrained embeddings as text features in a tabular+text benchmark. This supports the view that lexical cue detection carries much of the signal. — [Shi et al., arXiv:2111.02705](https://arxiv.org/abs/2111.02705)
- A CNN-based semi-supervised model (FLiText) matched BERT on IMDb and Yelp-5 with about 2.7% of the compute. — [FLiText, arXiv:2110.11869](https://arxiv.org/abs/2110.11869)

### Inferences

**Where TextCNN might help**
- TextCNN helps where cue phrases matter more than overall meaning, as with instructor prompts that invite participation. A frozen-mean-pooled sentence embedding can blur such phrases across a 60 s window.
- Feeding it frozen GloVe/fastText vectors (Kim's "static" setting) keeps parameters small.

**Risks**
- With a few thousand positives, expect performance close to TF-IDF + LR with word 1–3-grams.
- It is sensitive to ASR spelling variants.
- On MPS it trains in seconds to minutes.

**Interpretability**
- Max-pooled filter activations show which n-grams triggered a prediction. That is useful for the write-up.

### Gaps
- I found no recent (2024–26) controlled comparison of TextCNN against modern frozen sentence embedders + LR on small, imbalanced conversational data.

## 7. Combining text embeddings with tabular features: concatenation, stacking, late fusion

### Takeaway
The best-evidenced recipe is stacking. Train text-only models (TF-IDF LR, embedding LR, optionally a fine-tuned encoder), produce **out-of-fold** probabilities grouped by class session, and feed those scores plus the raw tabular features into a GBDT. Late fusion beat early fusion in the main benchmark, and lexical n-grams beat frozen embeddings as text features. If you concatenate raw embeddings into LightGBM, reduce them first (PCA to about 16–50 dimensions). Add hand-built similarity scalars, such as cosine similarity to his submitted answers.

### Cited Findings
- AWS benchmark of 18 datasets: — [Shi, Mueller, Erickson, Li, Smola, arXiv:2111.02705](https://arxiv.org/html/2111.02705)
  - "Stack ensembling a multimodal Transformer with tree models" was best, with an average score of 0.684 against 0.676 for a weighted ensemble.
  - Among network fusions, Fuse-Late (0.664) beat Fuse-Early (0.658) and All-Text (0.662). "Newer ideas to fuse modalities in early layers... are not necessarily superior to older multi-tower Fuse-Late architectures."
  - N-grams and word2vec beat frozen pretrained embeddings as text features. Fine-tuned text networks beat frozen ones (0.638 vs 0.593).
- In a tabular foundation-model paper, gradient-boosting baselines were given text embeddings reduced to 50 dimensions via PCA. Text columns with fewer than 50 unique values were treated as categorical. — [PORTAL, arXiv:2410.13516](https://arxiv.org/abs/2410.13516) (method detail from a search snippet)
- TabSTAR (Arazi, Shapira, Reichart; May 2025) is a tabular foundation model for tables with free-text fields. It argues that "static, target-agnostic textual representations" limit performance, and learns target-aware text representations instead. — [arXiv:2505.18125](https://arxiv.org/abs/2505.18125)
- TabPFN-3.5 claims native support for "multimodal inputs (strings, text, images)". — [arXiv:2609.17895](https://arxiv.org/abs/2609.17895)

### Inferences

**Recommended layering (cheap to expensive)**
1. **Scalar text features into LightGBM.** Build cosine similarity between the current utterance or window embedding and (i) each of his submitted answers (max and mean) and (ii) the pre-class questions. Add simple lexical flags: a question mark or WH-word from the instructor, his name mentioned, "any questions". These are low-dimensional, interpretable and well suited to GBDTs.
2. **Reduced embeddings.** Concatenate a PCA-32 version of the window embedding. Fit the PCA on training folds only.
3. **Stacking.** Train text-only models on grouped folds and pass their out-of-fold logits as LightGBM features. Never pass in-fold predictions, which leak because the text model has seen those rows.
4. **Late fusion.** Average calibrated probabilities from the tabular GBDT and the text model, or fit a small LR meta-learner on out-of-fold scores. This is robust when one modality is weak.
5. **Neural early fusion.** Concatenate the tabular vector with the embedding inside an MLP or GRU (Q5). Use this only as an ablation, since the evidence favours late fusion and stacking.

**Leakage rules**
- Per-class normalisation, PCA, TF-IDF vocabularies and threshold selection must all be fitted inside the training folds.
- "His answers" and the pre-class questions are static per class and known beforehand, so they are legitimate inputs. Answers submitted after class must be excluded.

### Gaps
- The main fusion benchmark (Shi et al., 2021) predates modern contrastive embedders and tabular foundation models. I found no 2025–26 independent benchmark comparing these fusion strategies with bge- or EmbeddingGemma-class embeddings plus LightGBM, TabPFN-3.5 or TabICLv2.
- I did not fetch Grinsztajn et al.'s study of string-entry vectorisation for tables, or MulTaBench (arXiv:2605.10616), both of which could update these conclusions.
