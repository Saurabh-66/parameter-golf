# Training an LLM from Scratch — End-to-End Guide

**A complete walkthrough: FineWeb dataset → decoder-only transformer → Muon optimizer → val_bpb → checkpoint → inference → SOTA comparison**

> This repository trains an 18.1M-parameter GPT-style language model from scratch on the FineWeb dataset using the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) evaluation protocol. Every component is documented below — architecture, data pipeline, training loop, evaluation metric, inference, and how it compares to production-scale models like GPT-4, Claude, and LLaMA.

---

## Table of Contents

1. [The Dataset — FineWeb](#1-the-dataset--fineweb)
2. [The Tokenizer — SentencePiece BPE](#2-the-tokenizer--sentencepiece-bpe)
3. [Model Architecture — Complete Component Guide](#3-model-architecture--complete-component-guide)
4. [Training Pipeline](#4-training-pipeline)
5. [The Muon Optimizer](#5-the-muon-optimizer)
6. [Evaluation Metric — val_bpb](#6-evaluation-metric--valbpb)
7. [Training Script Walkthrough](#7-training-script-walkthrough)
8. [Inference](#8-inference)
9. [Logs and Monitoring](#9-logs-and-monitoring)
10. [This Run — Complete Results Analysis](#10-this-run--complete-results-analysis)
11. [SOTA Comparison — Market Context](#11-sota-comparison--market-context)
12. [Hardware and Slurm Cluster Guide](#12-hardware-and-slurm-cluster-guide)
13. [Known Issues and Fixes Applied](#13-known-issues-and-fixes-applied)
14. [Repository Layout](#14-repository-layout)
15. [Quick Start](#15-quick-start)
16. [Glossary of Abbreviations](#appendix-glossary-of-abbreviations)

---

## 1. The Dataset — FineWeb

### What is FineWeb?

**FineWeb** is a 15-trillion-token English web text dataset created by HuggingFace in April 2024. It is one of the highest-quality publicly available pretraining corpora and is used by many open-source models.

- **Source:** CommonCrawl web crawls from 2013–2024 (96 snapshots)
- **Processing pipeline:** URL filtering → language detection (fastText) → quality heuristics → MinHash deduplication (5-gram, 14×8 hash) → PII (Personally Identifiable Information) scrubbing
- **Size:** 18.5 trillion tokens (original), ~15T after filtering
- **Format:** English-only, diverse domains — news, Wikipedia, forums, code, academic text, recipes

### Why This Dataset for LLM Training?

Unlike domain-specific datasets, FineWeb contains the full breadth of human writing on the internet. A language model trained on it learns general English — not just encyclopedic prose or programming syntax. This diversity is what makes models generalise to arbitrary prompts.

### The Fixed Validation Set

The parameter-golf challenge freezes the **first 50,000 documents** of the FineWeb validation split into binary shards. This never changes. Every participant evaluates on the exact same bytes, making val_bpb (bits per byte) a fair apples-to-apples comparison regardless of tokenizer or architecture.

### Binary Shard Format

Each `.bin` file stores pre-tokenized FineWeb:

```
┌──────────────────────────────────────────────────────────┐
│  HEADER: 256 × int32  (1024 bytes total)                 │
│    [0]  = 20240520   ← magic number (file identification)│
│    [1]  = 1          ← version                          │
│    [2]  = N          ← number of tokens in this file    │
│    [3..255] = 0      ← reserved padding                 │
├──────────────────────────────────────────────────────────┤
│  DATA:  N × uint16   (2 bytes per token ID, range 0–1023)│
└──────────────────────────────────────────────────────────┘
Each shard: 100 million tokens ≈ 190 MB
```

### Downloading the Data

```bash
# From repo root — downloads exactly what every leaderboard submission uses
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4

# Verify download integrity
python3 data/cached_challenge_fineweb.py --variant sp1024 --check
```

This fetches pre-tokenized shards from `kevclark/parameter-golf` on HuggingFace — the official mirror maintained by OpenAI. No tokenization step needed.

---

## 2. The Tokenizer — SentencePiece BPE

### What is Tokenization?

Language models cannot process raw text — they operate on integers. The tokenizer converts text to **token IDs** (integers) and back. A vocabulary of size V means all text is expressed using only V distinct symbols.

**BPE (Byte Pair Encoding)** builds the vocabulary by:
1. Starting with all individual bytes (256 initial tokens)
2. Repeatedly merging the most frequent adjacent token pair into a new token
3. Stopping when the vocabulary reaches the target size (1024 here)

### The SP1024 Tokenizer

This run uses `fineweb_1024_bpe.model` — a SentencePiece BPE tokenizer with **1024 tokens**, trained specifically on FineWeb.

```
"artificial intelligence" → [842, 71, 318, 892]
"The history"             → [256, 183, 847]
```

With 1024 tokens, each token covers roughly **1.5 bytes** of English text on average. Common words are single tokens; rare words split into multiple sub-word pieces. This low vocabulary is intentional — it keeps the embedding matrix small enough to fit in the 16 MB artifact limit.

**Why 1024 instead of 32,000 (GPT-4) or 100K (LLaMA-3)?** Smaller vocabulary = smaller embedding matrix = fewer parameters = more room for transformer depth within the size budget.

---

## 3. Model Architecture — Complete Component Guide

### High-Level Structure

The model is a **decoder-only transformer** — the same family as GPT-2, GPT-3, GPT-4, LLaMA-3, Claude, and Gemma. It reads a sequence of token IDs and predicts the probability distribution over the next token at every position simultaneously.

```
                    ╔══════════════════════════════════════════╗
                    ║         MODEL ARCHITECTURE               ║
                    ╚══════════════════════════════════════════╝

  Input IDs:  [42, 871, 3, 201, 956, ...]   shape: (Batch × SeqLen)

                              │
                    ┌─────────▼─────────┐
                    │  Token Embedding   │   1024 vocab × 384 dims
                    │  (lookup table)    │   shape: (B × T × 384)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  RMS Normalise    │   stabilise embedding scale
                    │  (no params)      │
                    └─────────┬─────────┘
                              │
            ┌─────────────────▼──────────────────┐
            │         TRANSFORMER BLOCK           │  repeated × 8
            │  ┌──────────────────────────────┐  │
            │  │ 1. RMSNorm (pre-norm)        │  │
            │  │    scale: 384 learnable params│  │
            │  └──────────────┬───────────────┘  │
            │                 │                   │
            │  ┌──────────────▼───────────────┐  │
            │  │ 2. Grouped Query Attention   │  │
            │  │    Q-proj: 384→384  147K prm │  │
            │  │    K-proj: 384→192   74K prm │  │
            │  │    V-proj: 384→192   74K prm │  │
            │  │    O-proj: 384→384  147K prm │  │
            │  │    + QK-Norm (per head)      │  │
            │  │    + Partial RoPE (32 dims)  │  │
            │  │    + FlashAttention-3        │  │
            │  └──────────────┬───────────────┘  │
            │                 │                   │
            │        x = x + attn_output          │  ← residual
            │                 │                   │
            │  ┌──────────────▼───────────────┐  │
            │  │ 3. RMSNorm (pre-norm)        │  │
            │  └──────────────┬───────────────┘  │
            │                 │                   │
            │  ┌──────────────▼───────────────┐  │
            │  │ 4. SwiGLU MLP               │  │
            │  │    gate+up: 384→3072 1.18M p│  │
            │  │    down:   1536→384   590K p│  │
            │  └──────────────┬───────────────┘  │
            │                 │                   │
            │         x = x + mlp_output          │  ← residual
            └─────────────────┬──────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Final RMSNorm    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Tied Output Proj │   shares weights with embedding
                    │  384 → 1024       │   shape: (B × T × 1024)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Logit Softcap    │   30 × tanh(logit/30)
                    │  tanh × 30        │   clips to (-30, +30)
                    └─────────┬─────────┘
                              │
  Output probs: P(next_token) for every position   shape: (B × T × 1024)
```

### 3.1 Token Embedding

The embedding layer is a lookup table: each of the 1024 token IDs maps to a learned 384-dimensional vector.

```
Parameter count: 1024 × 384 = 393,216 parameters
```

**Tied embeddings:** The same weight matrix is used for both input lookup (embedding) and output projection (`F.linear(hidden, emb_weight)`). This saves 393,216 parameters at no quality cost. The input and output spaces are naturally aligned — what the model "means" by token 42 going in is the same as what it means going out.

**Why normalise embeddings at startup?** `F.rms_norm` is applied to the raw looked-up embeddings before the first transformer block. This prevents large variation in embedding norms (which vary across vocabulary items) from destabilising the first layer.

### 3.2 RMSNorm — Root Mean Square Normalisation

Applied before every sublayer (pre-norm style). Formula:

```
output = (x / sqrt(mean(x²) + ε)) × scale
```

where `scale` is a learnable vector of shape `(dim,)`.

| | LayerNorm (GPT-2) | RMSNorm (this model, LLaMA, Gemma) |
|--|------------------|------------------------------------|
| Subtracts mean? | Yes | No |
| Has bias? | Yes | No |
| Parameters | 2 × dim (scale + bias) | 1 × dim (scale only) |
| Speed | baseline | ~10% faster |

The mean subtraction in LayerNorm turns out to be unnecessary for training stability — the scale parameter does the important work.

**Pre-norm vs post-norm:** Pre-norm (normalise before the sublayer) provides better gradient flow through deep networks and is used in all modern LLMs.

### 3.3 Grouped Query Attention (GQA) with RoPE and QK-Norm

Attention is the core mechanism of transformers. It answers: *for each token, which other tokens in the sequence are most relevant?*

#### Standard vs Grouped Query Attention

In standard MHA (Multi-Head Attention), every head has its own independent Q, K, V projections. GQA reduces the K and V heads while keeping the full Q heads:

```
Standard MHA (6 heads):
  Q₁K₁V₁   Q₂K₂V₂   Q₃K₃V₃   Q₄K₄V₄   Q₅K₅V₅   Q₆K₆V₆
  6 Q heads, 6 K heads, 6 V heads

GQA (this model: 6Q heads, 3KV heads):
  Q₁  Q₂  │  Q₃  Q₄  │  Q₅  Q₆
       K₁V₁         K₂V₂         K₃V₃
  (Q₁,Q₂ share K₁V₁) (Q₃,Q₄ share K₂V₂) (Q₅,Q₆ share K₃V₃)
```

GQA is used in LLaMA-2, LLaMA-3, Mistral, Gemma, and Falcon. It gives the same attention quality with fewer KV parameters and a smaller KV cache during inference.

#### Projection Dimensions

```
head_dim = model_dim / num_heads = 384 / 6 = 64

Q projection:  384 → 6×64 = 384     (147,456 params)
K projection:  384 → 3×64 = 192      (73,728 params)
V projection:  384 → 3×64 = 192      (73,728 params)
O projection:  384 → 384             (147,456 params)
─────────────────────────────────────
Attention params per layer:           442,368
```

#### QK-Norm

Before applying RoPE, both Q and K are RMS-normalised per head. This prevents attention logits from growing unboundedly, a known source of instability in models trained at higher precision or with aggressive learning rates. Used in Google's PaLM-2 and the parameter-golf SOTA.

#### RoPE — Rotary Position Embeddings

Traditional transformers add learned position embeddings to token embeddings. **RoPE** instead rotates Q and K vectors by a position-dependent angle, so the dot product between positions i and j encodes only their *relative* distance (i − j).

```
Standard:   embedding = token_embedding + position_embedding[position]

RoPE:       Q[pos] = Rotate(Q, pos × frequency_matrix)
            K[pos] = Rotate(K, pos × frequency_matrix)
            Score(i,j) = Q[i] · K[j] = f(token_i, token_j, i-j)  ← relative only
```

RoPE generalises better to sequence lengths longer than seen during training, and encodes relative not absolute position. Used in LLaMA, Mistral, Gemma, ChatGLM, Falcon, and virtually every modern open-source LLM.

**Partial RoPE (this model):** Only the first 32 of 64 head dimensions are rotated. The remaining 32 dims attend without position encoding, learning position-invariant patterns (e.g., token type regardless of where it appears). Used in LLaMA-3 and the parameter-golf SOTA.

#### FlashAttention

The attention computation is computed using PyTorch's `F.scaled_dot_product_attention`. On Hopper architecture GPUs (H100, H200), this automatically dispatches to the **FlashAttention-3** kernel, which:
- Fuses the entire attention computation into one GPU kernel (no materialising the full N×N matrix in HBM)
- Reduces memory from O(N²) to O(N)
- Provides 2–3× speedup and ~10× memory reduction vs naive implementation

### 3.4 SwiGLU MLP (Multi-Layer Perceptron / Feed-Forward Network)

After attention each token passes independently through a feed-forward network. **SwiGLU** formula:

```
FFN(x) = down_proj( SiLU(gate_proj(x)) ⊗ up_proj(x) )

where SiLU(x) = x × sigmoid(x)   (Sigmoid Linear Unit)
      ⊗ = element-wise multiplication
```

The `gate_proj` output (after SiLU) acts as a learned filter: values near 0 gate out the corresponding `up_proj` dimensions. This selective gating gives ~5–10% lower loss than a standard GELU (Gaussian Error Linear Unit) MLP at the same parameter count. Used in LLaMA, PaLM, Gemma, Mistral, and most modern models.

```
Dimensions:
  gate+up (fused): 384 → 3072     (384 × 2 × 1536 = 1,179,648 params)
  down:           1536 → 384     (1536 × 384 = 589,824 params)
  MLP params per layer: 1,769,472
```

### 3.5 Residual Connections

Every sublayer (attention and MLP) uses `x = x + sublayer(norm(x))`. This provides:
1. Direct gradient paths during backpropagation (prevents vanishing gradients in deep networks)
2. An "identity shortcut" — early in training when sublayers are random noise, the residual stream passes information cleanly through

Without residual connections, transformers deeper than ~4 layers become effectively untrainable.

### 3.6 Logit Softcap

Before the final softmax, all logits are passed through:

```
logit_final = 30 × tanh(logit / 30)
```

This clips all values into (−30, +30). Without this, logits can grow to extreme values during training, causing extremely peaked distributions that make training unstable (near-infinite gradients from softmax). Used in Gemma and the parameter-golf SOTA.

### 3.7 EMA — Exponential Moving Average of Weights

```
ema_weight = 0.999 × ema_weight + 0.001 × current_weight
```

Instead of using the final training step's weights, a running average is maintained throughout training. The **final checkpoint uses EMA weights**, not the last-step weights. This provides ~0.002–0.005 val_bpb improvement at zero training cost — the EMA trajectory is smoother and converges to a flatter minimum, which also quantizes better (important for the 16 MB limit).

### 3.8 Parameter Count Summary

```
Component                            Parameters
────────────────────────────────────────────────
Token Embedding  (1024 × 384)          393,216
  (shared with output projection)

Per Transformer Block (× 8 total):
  Q projection   (384 × 384)           147,456
  K projection   (384 × 192)            73,728
  V projection   (384 × 192)            73,728
  O projection   (384 × 384)           147,456
  QK-Norm scales (heads × head_dim)      ~1,152
  gate+up proj   (384 → 3072)        1,179,648
  down proj      (1536 → 384)          589,824
  attn_norm      (384)                      384
  mlp_norm       (384)                      384
  Per block:                        ~2,213,184
  × 8 blocks:                      ~17,705,472

Final RMSNorm   (384)                      384
────────────────────────────────────────────────
TOTAL                               18,095,488
Model size (BF16 = 2 bytes/param):      36.2 MB
```

### 3.9 GPT-2 vs This Model: Design Evolution

| Feature | GPT-2 (2019) | This model | Why changed |
|---------|-------------|------------|-------------|
| Normalisation | LayerNorm post-residual | RMSNorm pre-residual | Faster, better gradient flow |
| Position encoding | Learned absolute | Partial RoPE (32/64 dims) | Relative position, no extra params |
| Attention | Multi-Head (all independent) | Grouped Query (6Q/3KV) | Fewer KV params, same quality |
| MLP activation | GELU | SwiGLU | ~5–10% lower loss |
| Output logits | Uncapped | tanh softcap ×30 | Prevents instability |
| Embeddings | Separate in/out | Tied (shared) | Saves 393K params |
| Weight averaging | None | EMA decay=0.999 | Better final checkpoint |

---

## 4. Training Pipeline

### 4.1 Data Loading

The `DataLoader` loads all 4 training shards into RAM at startup, pre-computes all valid start positions (shuffled), and streams batches during training:

```
Shard 1: [tok0, tok1, tok2, ..., tok99M]
Shard 2: [tok0, tok1, tok2, ..., tok99M]   → concatenated in RAM: 400M tokens
Shard 3: [...]
Shard 4: [...]

Batch:  x = tokens[start   : start+1024]   (input)
        y = tokens[start+1 : start+1025]   (target = x shifted right by 1)
        batch_size = 32 sequences
        total batch tokens = 32 × 1024 = 32,768
```

**Autoregressive training:** The model learns to predict token[t+1] given tokens[0..t]. The same batch generates 1024 training signals simultaneously (one per position). This is the universal pretraining objective used by GPT-2, GPT-3, GPT-4, LLaMA, Claude, Gemma, and essentially every modern LLM.

### 4.2 The Forward and Backward Pass

```
Forward:
  loss = model(x, y)
       = cross_entropy(model_logits(x), y)
       = -mean[log P(y[t] | x[0..t]) for all t]

Backward:
  loss.backward()   ← PyTorch autograd computes ∂loss/∂weight for every param
```

### 4.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the L2 norm of all gradients exceeds 1.0, they are scaled down proportionally. This prevents rare "gradient explosions" (caused by unusual training examples) from making extreme weight updates. In this run: 2 clipping events in 1307 steps — both during LR warmup. Healthy training has <5% clipping rate.

### 4.4 Mixed Precision Training — BF16

```
FP32:  1 sign + 8 exponent + 23 mantissa = 32 bits  (full precision)
BF16:  1 sign + 8 exponent +  7 mantissa = 16 bits  (half memory, same range)
FP16:  1 sign + 5 exponent + 10 mantissa = 16 bits  (different: narrower range)
```

All forward and backward passes use BF16 via `torch.autocast`. BF16 has the same dynamic range as FP32 (same exponent bits) but less precision — acceptable for neural network training because gradient variance dominates anyway. Benefits: 2× less GPU memory, 2–4× higher throughput on tensor cores.

### 4.5 Learning Rate Schedule — Trapezoidal

```
Learning Rate
  3e-3 │         ╔══════════════════════════════════════╗
       │        ╔╝                                      ╚╗
       │       ╔╝                                        ╚╗
       │      ╔╝                                          ╚╗
       │     ╔╝                                            ╚╗
     0 └────╔╝──────────────────────────────────────────────╚╗──────►
            0   100                                    1210 1307  step
            │    │                                      │
       Warmup    Peak (constant)                    Warmdown
       (linear   (lr = 3e-3)                       (linear → 0)
        0→3e-3)
```

- **Warmup (0→100):** LR rises linearly. Prevents large destructive updates when weights are initialised randomly.
- **Constant (100→1210):** Maximum learning rate, fastest learning.
- **Warmdown (1210→1307):** LR decays toward 0. Encourages convergence to a flat minimum — important for quantization-robustness (flatter minima maintain quality under low-bit compression).

---

## 5. The Muon Optimizer

### Standard AdamW

AdamW (Adam with decoupled Weight decay) adapts the learning rate per parameter using estimates of first and second gradient moments. It's the standard optimizer for most deep learning.

### Muon — Momentum Orthogonalized by Newton-Schulz

**Muon** (Keller Jordan, 2024) is used for all 2D weight matrices (Q/K/V/O projections, MLP weights) — 17.69M out of 18.10M parameters.

**Core idea:** Instead of updating weights in the raw gradient direction, orthogonalize the gradient update first so all singular values become 1. This means every direction in weight space gets an equal-magnitude update — no direction dominates.

```
Step 1: Nesterov momentum
    buf = momentum × buf + gradient
    g   = gradient + momentum × buf           ← lookahead update

Step 2: Orthogonalize via Newton-Schulz (5 iterations)
    g_ortho ≈ U × I × V^T   (polar decomposition approximation)
    where U, V are from SVD of g, and all singular values become 1

Step 3: Update
    W ← W − lr × g_ortho
```

**Newton-Schulz iteration** (avoids expensive SVD):

```python
a, b, c = 3.4445, -4.7750, 2.0315
X = G / (G.norm() + 1e-7)       # normalise to unit Frobenius norm
for _ in range(5):               # 5 iterations sufficient
    A = X @ X.T
    X = a*X + (b*A + c*(A@A)) @ X
# X now has all singular values ≈ 1
```

**Why this works:** Standard gradient descent under Euclidean norm treats all parameter dimensions equally by magnitude. Muon's orthogonalized update treats all *directions* equally by the spectral norm — a more natural norm for weight matrices that encode transformations. Empirically converges ~35% faster than AdamW for transformer weight matrices.

**AdamW is still used** for embeddings and RMSNorm scales (1D parameters, or parameters where orthogonalization doesn't make geometric sense).

---

## 6. Evaluation Metric — val_bpb

### What is val_bpb?

**val_bpb = validation bits per byte** — how many bits the model needs to encode each byte of the validation text. This is a tokenizer-agnostic compression quality metric. Lower is always better.

### Derivation from Cross-Entropy Loss

The model outputs `P(next_token | context)` at each position. **Cross-entropy loss** (NLL — Negative Log-Likelihood) measures the average surprise:

```
loss_nats = −mean[ ln P(correct_next_token | context) ]    (in nats, base e)
```

Converting to bits and normalising by bytes:

```
val_bpb = (loss_nats / ln 2) × (tokens_scored / bytes_in_those_tokens)
        = loss_bits × (tokens/bytes)
```

The `tokens/bytes` factor accounts for tokenizer efficiency — a model using a 1024-vocab tokenizer and one using a 32000-vocab tokenizer can produce the same val_bpb if they compress text equally well, even though their raw losses differ.

### Reference Scale

| val_bpb | What the model has learned |
|---------|---------------------------|
| ~8.0 | Uniform random (no learning at all) |
| 4.0–5.0 | Basic character/byte statistics |
| 3.0–4.0 | Word-level statistics, common words |
| 2.5–3.5 | Word collocations, basic grammar **(this run: 3.567)** |
| 2.0–2.5 | Sentence structure, topic coherence |
| 1.5–2.0 | Deep grammar, factual associations |
| 1.224 | OpenAI param-golf naive baseline (8×H100, 10min) |
| **1.081** | **Parameter-golf SOTA (Apr 2026)** |
| ~1.0 | Shannon entropy of English — theoretical minimum |

### Sliding Window Evaluation

Naively chunking the validation set gives the first token of each chunk zero context, artificially inflating the loss.

```
Naive (non-overlapping chunks of 1024):
  Chunk 1: [tok0  ... tok1023]   ← tok0 has 0 context, scored poorly
  Chunk 2: [tok1024 ... tok2047] ← tok1024 has 0 context again

Sliding window (stride=64, this model):
  Window 1: tok0   → tok1023    score tok960–tok1023    (960 tokens of context)
  Window 2: tok64  → tok1087    score tok1024–tok1087   (960 tokens of context)
  Window 3: tok128 → tok1151    score tok1088–tok1151   (960 tokens of context)
```

Every scored token gets 960 tokens of prior context. This is ~0.01–0.03 bpb better than naive chunking and is the standard used in all parameter-golf submissions.

---

## 7. Training Script Walkthrough

`train_llm_scratch.py` implements the complete training pipeline in a single file (~1160 lines). Execution flow:

```
train(cfg)
  │
  ├── 1. Setup
  │       device = cuda / cpu
  │       torch.manual_seed(42)
  │       torch.set_float32_matmul_precision("high")   ← use TF32 on Ampere+
  │
  ├── 2. Load tokenizer
  │       sp = SentencePieceProcessor("fineweb_1024_bpe.model")
  │       vocab_size = 1024
  │
  ├── 3. Build byte LUTs for val_bpb
  │       base_bytes_lut[token_id] = UTF-8 byte count
  │       has_lead_space_lut[token_id] = True/False
  │       is_boundary_lut[token_id] = True/False
  │
  ├── 4. Load training data into RAM
  │       DataLoader(glob("fineweb_train_*.bin"))
  │       → load_shard() × 4 → torch.cat → 400M tokens
  │
  ├── 5. Load validation data (capped at 2M tokens for eval speed)
  │
  ├── 6. Build model: GPT(cfg).cuda().bfloat16()
  │       8 × TransformerBlock(GroupedQueryAttention + SwiGLU)
  │       18.10M parameters
  │
  ├── 7. compiled_model = model
  │       (torch.compile skipped — hung on MIG slice)
  │
  ├── 8. Setup optimizers
  │       matrix_params → Muon(lr=3e-3, momentum=0.95, ns_steps=5)
  │       scalar_params → AdamW(lr=6e-4, betas=(0.9,0.95))
  │
  ├── 9. EMA(model, decay=0.999)
  │
  ├── 10. Training loop  (while elapsed < 1800s)
  │        │
  │        ├── get_lr(step, total_steps, cfg) → update optimizer LR
  │        ├── x, y = train_loader.next_batch(device)
  │        ├── with torch.autocast("cuda", bfloat16):
  │        │       loss = compiled_model(x, y)
  │        ├── loss.backward()
  │        ├── clip_grad_norm_(params, max_norm=1.0)
  │        ├── muon_opt.step()    ← Newton-Schulz + update 17.69M params
  │        ├── adam_opt.step()    ← AdamW update 0.40M params
  │        ├── zero_grad()
  │        ├── ema.update(model)
  │        └── if step % val_every_steps == 0:
  │                orig = save_weights(model)
  │                ema.apply_to(model)      ← evaluate EMA weights
  │                val_metrics = evaluate() ← sliding window over 2M val tokens
  │                restore_weights(model, orig)
  │                log val_bpb to JSONL
  │
  ├── 11. ema.apply_to(model)    ← switch to EMA weights for final eval
  │
  ├── 12. Final evaluate() → val_bpb
  │
  ├── 13. torch.save(checkpoint)  → checkpoints/final_model.pt
  │
  └── 14. Inference demo (3 prompts, autoregressive generation)
```

### Key Configuration

```python
@dataclass
class Config:
    # Architecture
    num_layers:     8       # transformer depth
    model_dim:      384     # hidden dimension D
    num_heads:      6       # query heads
    num_kv_heads:   3       # key/value heads (GQA ratio 2:1)
    mlp_mult:       4.0     # MLP hidden dim = 4 × D = 1536
    rope_dims:      32      # head dims getting RoPE (out of 64)
    logit_softcap:  30.0    # tanh clip

    # Training
    max_train_seconds: 1800.0   # 30 minutes hard stop
    batch_tokens:      32768    # 32 seqs × 1024 tokens per step
    grad_clip:         1.0

    # LR schedule
    peak_lr:        3e-3    # Muon peak LR
    embed_lr:       6e-4    # AdamW LR
    warmup_steps:   100
    warmdown_frac:  0.30    # last 30% of training: linear decay

    # Evaluation
    val_every_steps: 100    # validate every N steps
    val_stride:      64     # sliding window stride
    ema_decay:       0.999
```

---

## 8. Inference

### The Checkpoint File

`checkpoints/final_model.pt` contains:

```python
{
    "model_state_dict": {...},   # all EMA weight tensors
    "config":           {...},   # Config dataclass serialised to dict
    "step":             1307,    # gradient steps completed
    "val_bpb":          3.5668,  # final validation score
    "val_loss":         5.9525,  # final validation loss in nats
}
```

### Loading and Generating

```python
import torch
from train_llm_scratch import GPT, Config

ckpt  = torch.load("checkpoints/final_model.pt", weights_only=False)
cfg   = Config(**ckpt["config"])
model = GPT(cfg).cuda().bfloat16()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"val_bpb: {ckpt['val_bpb']:.4f}")
```

### Autoregressive Generation

The `generate()` function in `inference.py` implements autoregressive decoding — the same method used by ChatGPT, Claude, and every GPT-style model:

```
1. Encode prompt: "The history of AI" → [183, 847, 71, 842]
2. Forward pass → logits for position after last token
3. Scale by temperature:  logits /= temperature
4. Top-k filter:          zero all but top-k logits
5. Softmax → probability distribution over 1024 tokens
6. Sample one token id from the distribution
7. Append token id to sequence
8. Repeat from step 2 until max_new_tokens reached
9. Decode token IDs → text
```

**Temperature** controls randomness:
- `temperature = 0.1` → near-greedy, picks highest-probability token
- `temperature = 1.0` → samples from true model distribution
- `temperature = 1.5` → more creative/diverse (wider distribution)

### Usage

```bash
# Interactive REPL
python inference.py checkpoints/final_model.pt

# Single prompt
python inference.py checkpoints/final_model.pt --prompt "Scientists discovered"

# Show model info
python inference.py checkpoints/final_model.pt --info

# Compute perplexity on a text passage
python inference.py checkpoints/final_model.pt --ppl "The quick brown fox"

# CPU inference (no GPU needed)
python inference.py checkpoints/final_model.pt --cpu
```

### Why Inference Output Looks Repetitive

```
Prompt:    "The history of artificial intelligence"
Generated: "enceenceenceenceenceenceence..."
```

This is **expected** for a model trained on only 42.8M tokens. Without a repetition penalty, small undertrained models loop on high-frequency token patterns. Despite the repetition, the model demonstrates real learning — "machine learning has → has also been" correctly captures a frequent English construction. For coherent generation, this model needs ~100–1000× more training.

---

## 9. Logs and Monitoring

### JSONL Log Format

`logs/training_log_JOBID.jsonl` — one JSON object per line, written immediately (no buffering).

```jsonl
{"type":"header","note":"JSON Lines log. Each line is one metric event."}
{"type":"config","num_layers":8,"model_dim":384,...}
{"type":"model_info","n_params":18095488,...}
{"type":"train","step":100,"train_loss":5.13,"tok_per_sec":98304,"mfu":0.485,"lr":0.003,...}
{"type":"val","step":100,"val_loss":7.95,"val_bpb":4.767,"elapsed_min":2.4,...}
{"type":"final","val_bpb":3.567,"total_tokens_M":42.8,"total_time_min":30.0,...}
{"type":"inference","prompt":"...","generated":"..."}
```

### Monitoring Commands

```bash
# Watch val_bpb in real time
watch -n 60 'grep "\"val\"" logs/training_log_JOBID.jsonl | tail -5'

# Stream live updates
tail -f logs/training_log_JOBID.jsonl

# Check job is running
squeue -u $USER

# Check what the Python process is doing (from a srun --pty bash session)
ps aux | grep train_llm_scratch        # see CPU%, memory
cat /proc/PID/status | grep VmRSS      # RAM in use
cat /proc/PID/io                       # bytes read from disk
```

### Generating Plots

```bash
# Step-based x-axis
python plot_training.py logs/training_log_JOBID.jsonl --out plots/run_JOBID

# Time-based x-axis (minutes)
python plot_training.py logs/training_log_JOBID.jsonl --out plots/run_JOBID --xaxis time
```

### Understanding the Plots

**training_dashboard_step.png** (6 panels):

| Panel | What to look for | Healthy range |
|-------|-----------------|---------------|
| Loss (nats) | Monotonically decreasing train + val loss | Smooth downward, no NaN spikes |
| val_bpb | Primary metric — lower is better | Steady decline throughout |
| Throughput | Tokens/sec — ramps then plateaus | Steady ~98K tok/s on saxa |
| MFU | % of peak FLOPS used | 40–55% is production quality |
| Gradient Norm | L2 norm of all gradients | Stay 0.3–1.0, few spikes above 1.5 |
| LR Schedule | Three phases: warmup / constant / warmdown | Warmup completes, warmdown visible |

### Why .log Shows Everything at the End

Python buffers `stdout` when piped to a file. Fix by adding `-u` flag:
```bash
# In train_saxa.slurm:
python -u train_llm_scratch.py ...
```
The JSONL file is always written immediately and is always current.

---

## 10. This Run — Complete Results Analysis

### Run Specification

| Parameter | Value |
|-----------|-------|
| Job ID | 2262656 |
| Node | saxa — NVIDIA H200 MIG 1g.18gb |
| Date | Thu 16 Apr 2026, 09:47–10:20 BST |
| Model | 8L × 384d, 18.10M params, 36.2 MB (BF16) |
| Tokenizer | SP1024 (1024 vocab) |
| Dataset | FineWeb 10B — 4 train shards (400M tokens), val capped at 2M |
| Batch | 32 seqs × 1024 tokens = 32,768 tokens/step |
| torch.compile | Disabled (hung on MIG; see Known Issues) |

### Training Results

| Metric | Value |
|--------|-------|
| Total gradient steps | 1,307 |
| Training time | 30.00 min (hard stop) |
| Tokens consumed | 42.8M |
| Data passes | 10.7% (< 1 full pass through 400M token dataset) |
| Average step time | ~334ms |
| Tokens/sec (steady state) | ~98,116 |
| MFU (Model FLOP Utilisation) | 48.4% of estimated 22 TFLOPS peak |
| Gradient clipping events | 2 (steps 50 and 60 — normal during warmup) |

### val_bpb Progression

| Step | val_bpb | Elapsed | Drop |
|------|---------|---------|------|
| 100 | 4.7666 | 2.4 min | — |
| 200 | 4.7121 | 4.7 min | −0.054 |
| 300 | 4.6299 | 7.0 min | −0.082 |
| 400 | 4.4883 | 9.3 min | −0.142 |
| 500 | 4.2823 | 11.6 min | −0.206 |
| 600 | 4.0970 | 13.9 min | −0.185 |
| 700 | 3.9619 | 16.2 min | −0.135 |
| 800 | 3.8654 | 18.5 min | −0.097 |
| 900 | 3.7840 | 20.8 min | −0.081 |
| 1000 | 3.7138 | 23.1 min | −0.070 |
| 1100 | 3.6525 | 25.4 min | −0.061 |
| 1200 | 3.6020 | 27.7 min | −0.050 |
| 1300 | 3.5683 | 30.0 min | −0.034 |
| **Final (EMA)** | **3.5668** | 30.0 min | — |

**Total val_bpb drop: 4.767 → 3.567 = 1.200 bpb in 30 minutes.** The rate of improvement is slowing but still positive — the model has not converged.

### Critical Finding: Training Efficiency

```
Total job time:        30.0 minutes
Time in validation:    13 evals × ~1.74 min = 22.6 min  ← 75% of the job!
Actual training time:  7.4 minutes  ← only 25% of the job!
```

With `--val_every_steps 100`, validation consumed 3× more time than training. Reducing to `--val_every_steps 500` would give ~4,500 training steps (vs 1,307) and an estimated val_bpb of ~2.5–2.8, at zero other cost.

### LR Schedule Anomaly

Warmdown should have started at step 915 (70% of training). It started at step 1210 (93%). Cause: `estimated_step_ms = 200` in config vs actual 334ms, causing the dynamic total_steps estimate to overshoot. Fix: add `--estimated_step_ms 334` to the slurm script.

### What the Model Learned

The val_bpb of 3.567 (vs 4.767 at first evaluation, vs ~8.0 for a random model) demonstrates genuine language learning:
- High-frequency English vocabulary and word frequencies
- Basic subject-verb-object grammatical patterns
- Common phrase constructions ("has also been", "was discovered that")
- Topic clustering (AI-related prompts activate AI vocabulary)

For reference, GPT-2's smallest model (124M params) trained on 40B tokens achieves ~1.9 bpb. This model is 7× smaller and trained on 1000× less data.

---

## 11. SOTA Comparison — Market Context

### This Run vs Production Systems

| System | val_bpb | Params | Training tokens | Hardware | Training time | Use |
|--------|---------|--------|-----------------|----------|---------------|-----|
| **This run** | **3.567** | **18.1M** | **42.8M** | **1× H200 MIG** | **30 min** | Teaching |
| GPT-2 124M (OpenAI, 2019) | ~1.9† | 124M | ~40B | 32× V100 | ~1 week | Research |
| Param-golf baseline (2026) | 1.224 | 18.5M | ~3.6B | 8× H100 | 10 min | Challenge |
| Param-golf SOTA (Apr 2026) | 1.081 | ~36M | ~3.6B | 8× H100 | 10 min | Challenge |
| LLaMA-3 8B (Meta, 2024) | ~0.85† | 8B | 15T | ~50K H100-hrs | weeks | Open source |
| Gemma 2 9B (Google, 2024) | ~0.80† | 9B | 8T | TPU v5e | weeks | Open source |
| GPT-4 (OpenAI, est.) | ~0.60† | ~1.8T (MoE) | ~13T | ~25K A100-days | months | Production |
| Claude 3.5 (Anthropic, est.) | ~0.62† | unknown | unknown | TPU v4/v5 | months | Production |

*† Approximate — different evaluation sets, not directly comparable to parameter-golf val_bpb.*

The difference between this run (3.567) and production models (~0.6) is almost entirely compute:

```
This run:       42.8M tokens × 18.1M params ≈ 7.7 × 10¹⁴ FLOPs
Param-golf:     3.6B tokens  × 35M params   ≈ 1.3 × 10¹⁷ FLOPs  (163× more)
LLaMA-3 8B:     15T tokens   × 8B params    ≈ 7.2 × 10²³ FLOPs  (900M× more)
GPT-4 (est.):   13T tokens   × ~1.8T params ≈ ~10²⁷ FLOPs       (10¹² × more)
```

### Parameter-Golf Leaderboard — 4-Week Evolution

| Date | val_bpb | Key innovation |
|------|---------|----------------|
| Mar 18 | 1.2244 | OpenAI naive baseline (9L×512d, int8+zlib) |
| Mar 19 | 1.2014 | seq_len = 4096 (4× more context per step) |
| Mar 22 | 1.1233 | 11 layers + EMA + GPTQ-lite int6 + QAT (Quantization-Aware Training) |
| Mar 30 | ~1.119 | XSA (Cross-Sequence Attention) + VE128 + Partial RoPE |
| **Apr 9** | **1.081** | **SP8192 tokenizer + depth recurrence + TTT + full Hessian GPTQ** |
| Apr 16 | ~1.108 | XSA-all + selective pruning + GPTQ |

### SOTA Techniques Breakdown (1.081 BPB submission)

| Technique | Description | BPB gain |
|-----------|-------------|----------|
| SP8192 tokenizer | 8× larger vocab → more efficient encoding | ~0.040 |
| 3-layer depth recurrence | Re-use 3 layers 3× (17 virtual layers from 11 physical) | ~0.030 |
| Parallel residuals (GPT-J style) | Attention + MLP run in parallel for layers 7–11 | ~0.015 |
| Full Hessian GPTQ int6 + Brotli | Optimal quantization using weight Hessian info | ~0.030 |
| MuonEq-R | Row-normalised Muon optimizer | ~0.010 |
| Score-first TTT | SGD on already-evaluated tokens during val (3 epochs/32K) | ~0.020 |
| QK-gain = 5.25 | Higher QK projection gain | ~0.005 |

### What Production Companies Use

**OpenAI (ChatGPT, GPT-4o):** Transformer-based, likely MoE (Mixture of Experts — many specialized sub-networks, only a few activated per token). Training involves pretraining on ~13T tokens then RLHF (Reinforcement Learning from Human Feedback) fine-tuning for instruction following. Infrastructure: custom Microsoft Azure clusters with H100 GPUs.

**Anthropic (Claude 3.5, Claude 4):** Dense (non-MoE) transformer with Constitutional AI (CAI) safety training layered on top of standard pretraining + RLHF. Infrastructure: AWS Trainium + Google TPU v5.

**Meta AI (LLaMA-3):** Dense decoder-only transformer with the **same component choices as this model** — RMSNorm, GQA, RoPE, SwiGLU, tied embeddings — just 8B–70B parameters trained on 15T tokens on ~16,000 H100 GPUs. Fully open-source weights.

**Google DeepMind (Gemma 2, Gemini):** Gemma 2 specifically uses the logit softcap (tanh×30) — **identical to this model**. Pretraining + instruction fine-tuning + safety tuning. Infrastructure: custom TPU v4/v5 clusters.

**The architectural components in this teaching model** (RMSNorm, GQA, RoPE, SwiGLU, tied embeddings, logit softcap, EMA) are exactly the same components used by Meta, Google, and Mistral in their frontier open-source models. The only differences are scale and training compute.

---

## 12. Hardware and Slurm Cluster Guide

### The MLP Teaching Cluster at Edinburgh

| Node | GPU | VRAM | Architecture | CUDA | Notes |
|------|-----|------|-------------|------|-------|
| saxa | H200 MIG 1g.18gb | ~16 GB | Hopper SM 9.0 | 13.0 | 1/9 of full H200 |
| damnii07–12 | RTX 2080 Ti | 11 GB | Turing SM 7.5 | 12.x | Full GPU |

**MIG (Multi-Instance GPU):** NVIDIA technology that partitions a single GPU into isolated slices with guaranteed compute and memory resources. The saxa H200 has 9 MIG instances of profile `1g.18gb` — each gets 1/9 of the SMs (16 out of 144) and ~16 GB of HBM3 (High Bandwidth Memory 3).

**Why torch.compile hung on MIG:** Triton JIT (Just-In-Time) kernel compilation is CPU-bound and requires significant compilation infrastructure. On a 1g.18gb MIG slice, both `fullgraph=True` and `dynamic=True` modes hung indefinitely (45+ minutes tested, no output). Resolution: skip compilation entirely. Despite this, the H200's HBM3 memory bandwidth provides 48.4% MFU in eager mode via the FlashAttention-3 backend in PyTorch SDPA.

**Choosing the right node for your batch size:**

| Node | Max batch_tokens | Why |
|------|-----------------|-----|
| saxa MIG 1g.18gb | 32,768 | 16 GB VRAM |
| damnii RTX 2080 Ti | 8,192 | 11 GB VRAM |

### QoS (Quality of Service) Limits — Teaching Partition

From `sacctmgr show qos Teaching`:

```
MaxTRES: cpu=2, gres/gpu=1
```

Maximum 2 CPUs and 1 GPU per job. This is why `--cpus-per-task=4` was rejected — must use `--cpus-per-task=2`.

### Slurm Reference

**Slurm** (Simple Linux Utility for Resource Management) is the job scheduler. You submit scripts from the head node (hastings), the scheduler queues them, and they run on compute nodes when resources are available.

```
hastings (head node)
    │
    │  sbatch train_saxa.slurm
    │
    ▼
Slurm queue → waits for saxa GPU to be free
    │
    ▼
saxa (compute node) ← job runs here, output → logs/train_JOBID.log
```

**Key commands:**

```bash
sbatch train_saxa.slurm           # submit job
squeue -u $USER                   # check job status (R=running, PD=pending)
scancel JOBID                     # kill a job
sinfo -p Teaching                 # see node availability (idle/alloc/drain)
sacctmgr show qos Teaching        # see resource limits
sstat -j JOBID --format=JobID,AveCPU,AveRSS    # job resource usage
srun -p Teaching -w saxa --gres gpu:1 --mem=32G -t 00:30:00 --pty bash  # interactive session
```

**Critical: sbatch extra args don't reach Python.** Arguments after the script name go to the shell as `$1`, `$2` — NOT to the Python command:

```bash
# WRONG — --batch_tokens is silently ignored by Python:
sbatch train_saxa.slurm --batch_tokens 8192

# CORRECT — edit the python call directly inside train_saxa.slurm:
python -u train_llm_scratch.py \
    --batch_tokens 8192 \
    ...
```

### This Run vs Challenge Hardware

| Metric | This run | Param-golf SOTA | Ratio |
|--------|----------|----------------|-------|
| GPUs | 1× H200 MIG (1/9) | 8× H100 SXM | ~72× |
| Total VRAM | ~16 GB | 640 GB | 40× |
| SMs | 16 | 8×132 = 1,056 | 66× |
| Peak BF16 | ~22 TFLOPS | ~7,912 TFLOPS | ~360× |
| Batch tokens/step | 32,768 | ~524,288 | 16× |
| Step time | ~334ms | ~83ms | 4× faster |
| Steps in time budget | 1,307 (30min) | ~4,550 (10min) | 3.5× |
| Tokens seen | 42.8M | ~3.6B | 84× |

---

## 13. Known Issues and Fixes Applied

### Bug 1 — RoPE Shape Mismatch (all early jobs crashed immediately)

**Error:** `TorchRuntimeError: Attempting to broadcast dimension of length 32 at -1. Mismatching: size=(1, 1024, 1, 32) vs broadcastable to (32, 1024, 6, 16)`

**Root cause:** `get_cos_sin()` used `torch.cat([freqs, freqs])` creating `cos/sin` of shape `(1, T, 1, rope_dims=32)`, but `forward()` sliced `x1` to `(B, T, H, rope_dims/2=16)`. Last dim 32 ≠ 16.

**Fix:** Removed `cat()` — return `freqs.cos()[None, :, None, :]` with shape `(1, T, 1, rope_dims/2)`.

### Bug 2 — `torch.compile(fullgraph=True)` hung for 45+ minutes

**Symptom:** Single warning line `Not enough SMs to use max_autotune_gemm mode`, then nothing for 45+ minutes until Slurm time limit killed the job.

**Fix:** `compiled_model = model`. Skip compilation entirely. Still achieves 48.4% MFU via FlashAttention-3 in eager mode.

### Bug 3 — `@torch.compile` decorator on `newton_schulz_5` spawned hanging workers

**Symptom:** Even after removing the main compile block, jobs hung for 20+ minutes showing `torch/_inductor/compile_worker` subprocesses.

**Fix:** Removed `@torch.compile` decorator from the `newton_schulz_5` function.

### Bug 4 — Step-0 validation timed out (120+ min evaluation)

**Symptom:** After data loading, training appeared to hang — actually running evaluation over 115M validation tokens with batch_size=1.

**Fix:** Skip step-0 eval. Cap val tokens to 2M. Batch the sliding window loop (batch_seqs=32).

### Bug 5 — Warmdown started too late (LR anomaly)

**Symptom:** LR stayed at peak 3e-3 until step 1210 instead of starting at step ~915.

**Root cause:** `estimated_step_ms = 200` vs actual 334ms. Dynamic `total_steps` overestimated.

**Fix for next run:** Add `--estimated_step_ms 334` to slurm script.

### Bug 6 — Python stdout buffering (.log shows nothing until job ends)

**Fix:** Add `-u` flag: `python -u train_llm_scratch.py`

### Bug 7 — Wrong dataset (FineWeb-Edu instead of FineWeb)

**Root cause:** Custom download script used `HuggingFaceFW/fineweb-edu` (filtered educational subset) instead of `HuggingFaceFW/fineweb` (the actual challenge dataset).

**Fix:** Replaced with official `cached_challenge_fineweb.py` from `github.com/openai/parameter-golf`.

---

## 14. Repository Layout

```
parameter-golf/
│
├── data/
│   ├── cached_challenge_fineweb.py    ← official OpenAI download script
│   ├── tokenizers/
│   │   ├── fineweb_1024_bpe.model     ← SentencePiece model (binary, 249 KB)
│   │   └── fineweb_1024_bpe.vocab     ← human-readable vocabulary companion
│   └── datasets/
│       └── fineweb10B_sp1024/         ← binary shards (NOT in git, ~1 GB)
│           ├── fineweb_train_000000.bin    (191 MB, 100M tokens)
│           ├── fineweb_train_000001.bin
│           ├── fineweb_train_000002.bin
│           ├── fineweb_train_000003.bin
│           └── fineweb_val_000000.bin     (119 MB, fixed validation set)
│
├── train_llm_scratch.py    ← complete training pipeline (~1160 lines)
├── inference.py            ← load checkpoint, generate text, perplexity
├── plot_training.py        ← JSONL log → 4 PNG training plots
│
├── train_saxa.slurm        ← Slurm job for saxa H200 MIG (batch_tokens=32768)
├── train_damnii.slurm      ← Slurm job for damnii RTX 2080 Ti (batch_tokens=8192)
│
├── logs/
│   ├── train_2262656.log              ← Slurm stdout (buffered — appears at end)
│   └── training_log_2262656.jsonl     ← JSONL metrics (written live, use this)
│
├── checkpoints/
│   └── final_model.pt                 ← EMA weights + config + val_bpb
│
├── plots/
│   └── run_2262656/
│       ├── training_dashboard_step.png    ← 6-panel overview (main plot)
│       ├── val_bpb_step.png               ← primary metric vs SOTA/baseline
│       ├── loss_curves_step.png           ← train + val loss curves
│       └── optimizer_diagnostics_step.png ← LR schedule + gradient norm
│
└── .gitignore    ← excludes data/datasets/, *.bin, __pycache__/
```

---

## 15. Quick Start

### Prerequisites

```bash
# Check environment
conda activate llm_training
python -c "import torch, sentencepiece, numpy, matplotlib; print('OK')"
```

### Step 1 — Download Data (from hastings or a compute node)

```bash
cd ~/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4
```

### Step 2 — Smoke Test (2 min, confirms no crash)

```bash
# Get interactive session
srun -p Teaching -w damnii08 --gres gpu:1 --mem=32G -t 00:10:00 --pty bash

source /opt/conda/etc/profile.d/conda.sh && conda activate llm_training
cd ~/parameter-golf

python train_llm_scratch.py \
    --max_train_seconds 60 \
    --batch_tokens      8192 \
    --val_every_steps   500
# Expected: loss drops from ~6.9, no crash, clean exit
exit
```

### Step 3 — Full 30-Minute Training Run

```bash
# From hastings:
sbatch train_saxa.slurm

# Watch metrics live:
tail -f logs/training_log_$(squeue -u $USER -h -o %i).jsonl
```

### Step 4 — After Training

```bash
# Run inference
python inference.py checkpoints/final_model.pt --prompt "The history of AI"

# Commit results
git add logs/ plots/ checkpoints/
git commit -m "run: saxa H200 MIG 30min val_bpb=3.567 1307steps 42.8Mtokens"
git push
```

### Recommended Changes for Better val_bpb

In `train_saxa.slurm`, make two changes:

```bash
# 1. Reduce validation frequency (MOST IMPORTANT — gives ~3.4× more training steps)
--val_every_steps    500    # was 100

# 2. Unbuffered Python output (fixes .log file)
python -u train_llm_scratch.py \

# 3. Fix LR warmdown timing (optional)
--estimated_step_ms  334    # was 200 in Config default
```

Expected result with these changes: ~4,500 steps, val_bpb ~2.5–2.8 (vs 3.567).

---

## Appendix: Glossary of Abbreviations

| Abbreviation | Full Form |
|-------------|-----------|
| LLM | Large Language Model |
| BPE | Byte Pair Encoding |
| SP / SP1024 | SentencePiece / SentencePiece with 1024 vocab |
| GQA | Grouped Query Attention |
| MHA | Multi-Head Attention |
| MQA | Multi-Query Attention |
| RoPE | Rotary Position Embeddings |
| RMSNorm | Root Mean Square Normalisation |
| SwiGLU | Swish-Gated Linear Unit |
| SiLU | Sigmoid Linear Unit (the activation inside SwiGLU) |
| MLP | Multi-Layer Perceptron (the feed-forward sublayer in transformers) |
| FFN | Feed-Forward Network (same as MLP in transformer context) |
| BF16 | Brain Float 16 (Google's 16-bit float format with FP32 dynamic range) |
| FP32 | 32-bit floating point (single precision, standard IEEE 754) |
| FP16 | 16-bit floating point (half precision, narrower range than BF16) |
| MFU | Model FLOP Utilisation (fraction of theoretical peak FLOPs achieved) |
| FLOP | Floating Point Operation |
| TFLOPS | Tera (10¹²) FLOPs per second |
| EMA | Exponential Moving Average |
| NLL | Negative Log-Likelihood |
| BPB | Bits Per Byte |
| val_bpb | Validation Bits Per Byte (primary eval metric) |
| LR | Learning Rate |
| SDPA | Scaled Dot-Product Attention (PyTorch's built-in attention function) |
| FA3 | FlashAttention-3 (Hopper-optimised attention kernel by Tri Dao et al.) |
| HBM | High Bandwidth Memory (stacked GPU memory technology) |
| HBM3 | Third generation High Bandwidth Memory (used in H100/H200) |
| TTT | Test-Time Training (updating model weights during inference/evaluation) |
| GPTQ | Generalised Post-Training Quantisation (optimal weight compression) |
| QAT | Quantisation-Aware Training (training model to tolerate quantization) |
| MoE | Mixture of Experts (architecture with multiple specialist sub-networks) |
| RLHF | Reinforcement Learning from Human Feedback |
| CAI | Constitutional AI (Anthropic's safety training method) |
| MIG | Multi-Instance GPU (NVIDIA's GPU partitioning technology) |
| SM | Streaming Multiprocessor (GPU's fundamental execution unit) |
| JIT | Just-In-Time compilation |
| AFS | Andrew File System (network filesystem on this cluster) |
| Slurm | Simple Linux Utility for Resource Management |
| NVLink | NVIDIA's high-speed GPU-to-GPU interconnect |
| SXM | Server PCI-Express Module (high-performance GPU board format) |
| VRAM | Video RAM (GPU memory) |
| BPB | Bits Per Byte |
| XSA | Cross-Sequence Attention (removes self-value bias in attention) |
| QoS | Quality of Service (Slurm resource limit policy) |
| EMA | Exponential Moving Average |
| MFU | Model FLOP Utilisation |
| JSONL | JSON Lines (one JSON object per line, streaming log format) |
