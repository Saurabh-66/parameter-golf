"""
train_llm_scratch.py
====================
Train a small GPT-style language model from scratch on the FineWeb dataset.
Designed for: NVIDIA H200 MIG (1g.16gb slice) on the MLP teaching cluster (saxa node).
Also runs on any CUDA GPU >= Ampere with 8GB+ VRAM.

Architecture: Decoder-only Transformer with modern improvements:
  - Grouped Query Attention (GQA)
  - Rotary Position Embeddings (RoPE)
  - RMSNorm (faster than LayerNorm)
  - SwiGLU activation (used in LLaMA, PaLM, etc.)
  - FlashAttention-2 via PyTorch scaled_dot_product_attention
  - Muon optimizer (orthogonal gradient updates via Newton-Schulz)
  - Cosine LR schedule with linear warmdown
  - EMA of weights for better final checkpoint quality

Dataset: FineWeb 10B subset, tokenized with 1024-vocab BPE (parameter-golf tokenizer)
Metric:  val_bpb = bits per byte (tokenizer-agnostic compression quality)

Author: training demo for parameter-golf challenge
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import glob
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir:       str  = "./data"
    log_file:       str  = "training_log.jsonl"
    checkpoint_dir: str  = "checkpoints"

    # ── Tokenizer / Dataset ────────────────────────────────────────────────
    vocab_size:     int  = 1024          # FineWeb SP1024 vocabulary
    # Tokenizer path and dataset path are derived from data_dir at runtime

    # ── Model Architecture ─────────────────────────────────────────────────
    num_layers:     int  = 8             # transformer blocks
    model_dim:      int  = 384           # embedding / hidden dimension
    num_heads:      int  = 6             # query heads
    num_kv_heads:   int  = 3             # key/value heads (GQA: ratio 2:1)
    mlp_mult:       float = 4.0          # MLP hidden = mlp_mult * model_dim
    rope_dims:      int  = 32            # how many head dims get RoPE (partial RoPE)
    rope_base:      float = 10000.0      # RoPE theta base
    logit_softcap:  float = 30.0         # tanh softcap on logits (Gemma-style)

    # ── Training ───────────────────────────────────────────────────────────
    max_train_seconds: float = 1800.0    # hard stop: 30 minutes
    seed:           int  = 42
    train_seq_len:  int  = 1024          # sequence length during training
    batch_tokens:   int  = 32768         # tokens per gradient step (32 seqs × 1024)
    grad_clip:      float = 1.0          # gradient clipping norm

    # ── Learning Rate Schedule ─────────────────────────────────────────────
    # Cosine schedule: warmup → peak → warmdown to 0
    peak_lr:        float = 3e-3         # peak learning rate (Muon)
    embed_lr:       float = 6e-4         # embedding / output head LR (Adam)
    warmup_steps:   int  = 100           # linear LR warmup
    warmdown_frac:  float = 0.30         # last 30% of training = linear LR decay to 0
    weight_decay:   float = 0.1          # AdamW weight decay for embeddings

    # ── Muon Optimizer ─────────────────────────────────────────────────────
    muon_momentum:  float = 0.95         # Nesterov momentum
    muon_ns_steps:  int  = 5             # Newton-Schulz iterations (orthogonalization)
    muon_wd:        float = 0.01         # Muon weight decay

    # ── Evaluation ─────────────────────────────────────────────────────────
    val_every_steps:   int = 100         # validate every N steps
    val_seq_len:       int = 1024
    val_stride:        int = 64          # sliding window stride for val_bpb
    log_every_steps:   int = 10          # log train metrics every N steps

    # ── EMA ────────────────────────────────────────────────────────────────
    ema_decay:      float = 0.999        # exponential moving average decay

    # ── MFU calculation ────────────────────────────────────────────────────
    # H200 MIG 1g.16gb: roughly 1/9 of H200 peak = ~22 BF16 TFLOPS effective
    peak_tflops:    float = 22.0         # estimated peak TFLOPS for this MIG slice

    def __post_init__(self):
        self.datasets_dir = os.path.join(self.data_dir, "datasets", "fineweb10B_sp1024")
        self.tokenizer_path = os.path.join(self.data_dir, "tokenizers", "fineweb_1024_bpe.model")
        self.train_files = os.path.join(self.datasets_dir, "fineweb_train_*.bin")
        self.val_files   = os.path.join(self.datasets_dir, "fineweb_val_*.bin")
        assert self.model_dim % self.num_heads == 0, "model_dim must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"


# ─────────────────────────────────────────────────────────────────────────────
# Logging  (every metric written as JSON Lines for easy plotting)
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    """Writes one JSON object per line to a log file.  Easy to read with pandas."""

    def __init__(self, path: str):
        self.path = path
        # Write header comment
        Path(path).write_text(
            '{"type":"header","note":"JSON Lines log. Each line is one metric event."}\n'
        )

    def log(self, **kwargs):
        with open(self.path, "a") as f:
            f.write(json.dumps(kwargs) + "\n")

    def print_and_log(self, msg: str, **kwargs):
        print(msg)
        if kwargs:
            self.log(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_shard(path: str) -> torch.Tensor:
    """Load a .bin shard file (FineWeb format: uint16 token IDs)."""
    header_ints = 256
    header = np.fromfile(path, dtype="<i4", count=header_ints)
    assert header[0] == 20240520, f"Bad magic in {path}"
    assert header[1] == 1,        f"Bad version in {path}"
    n_tokens = int(header[2])
    offset   = header_ints * 4    # bytes to skip past header
    tokens   = np.fromfile(path, dtype="<u2", count=n_tokens, offset=offset)
    return torch.from_numpy(tokens.astype(np.int32))   # int32 for embedding lookup


class DataLoader:
    """Infinitely iterate over sharded token files, yielding (x, y) batches."""

    def __init__(self, file_pattern: str, seq_len: int, batch_tokens: int, seed: int = 42):
        self.seq_len      = seq_len
        self.batch_size   = batch_tokens // seq_len   # sequences per step
        assert self.batch_size >= 1, "batch_tokens must be >= seq_len"

        files = sorted(glob.glob(file_pattern))
        assert files, f"No files found: {file_pattern}"
        print(f"  DataLoader: found {len(files)} shards, batch_size={self.batch_size} seqs")

        # Load all shards and concatenate (fine for 10B token dataset with 1 worker)
        # For very large datasets you'd stream; here we keep it simple for teaching
        all_tokens = torch.cat([load_shard(f) for f in files])
        self.tokens = all_tokens
        print(f"  Total tokens: {len(self.tokens):,}")

        self.rng = np.random.default_rng(seed)
        self._build_positions()

    def _build_positions(self):
        """Pre-compute all valid start positions and shuffle them."""
        n = len(self.tokens) - self.seq_len   # last valid start
        self.positions = np.arange(0, n, self.seq_len, dtype=np.int64)
        self.rng.shuffle(self.positions)
        self.pos_idx = 0

    def next_batch(self, device: torch.device):
        """Return (x, y) tensors of shape (batch_size, seq_len)."""
        bs = self.batch_size
        needed = bs

        # Wrap around if we run out of positions
        if self.pos_idx + needed > len(self.positions):
            self._build_positions()   # reshuffle for next epoch

        starts = self.positions[self.pos_idx : self.pos_idx + needed]
        self.pos_idx += needed

        x = torch.stack([self.tokens[s     : s + self.seq_len    ] for s in starts])
        y = torch.stack([self.tokens[s + 1 : s + self.seq_len + 1] for s in starts])
        return x.to(device=device, dtype=torch.long), y.to(device=device, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer utilities for val_bpb
# ─────────────────────────────────────────────────────────────────────────────

def build_byte_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    """
    Build lookup tables for byte-counting needed to compute val_bpb.

    val_bpb requires knowing how many UTF-8 bytes each token corresponds to.
    SentencePiece tokens can be:
      - Regular tokens: "▁hello" → has leading space + "hello" = 6 bytes
      - Byte tokens:    "<0xE2>" → exactly 1 byte
      - Control tokens: <s>, </s> → 0 bytes (ignored)

    Returns three tensors of shape (vocab_size,):
      base_bytes:      number of UTF-8 bytes in the token's text
      has_lead_space:  whether the token has a leading space (▁)
      is_boundary:     whether this token is a control/unknown token
    """
    n = int(sp.vocab_size())
    base_bytes_arr    = np.zeros(max(n, vocab_size), dtype=np.int16)
    has_lead_space_arr = np.zeros(max(n, vocab_size), dtype=bool)
    is_boundary_arr   = np.ones( max(n, vocab_size), dtype=bool)   # default: boundary

    for tid in range(n):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_arr[tid] = False

        if sp.is_byte(tid):
            base_bytes_arr[tid] = 1
            continue

        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_lead_space_arr[tid] = True
            piece = piece[1:]
        base_bytes_arr[tid] = len(piece.encode("utf-8"))

    return (
        torch.tensor(base_bytes_arr,    dtype=torch.int16, device=device),
        torch.tensor(has_lead_space_arr, dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_arr,   dtype=torch.bool,  device=device),
    )


def load_val_tokens(file_pattern: str, seq_len: int) -> torch.Tensor:
    """Load and concatenate validation tokens."""
    files  = sorted(glob.glob(file_pattern))
    tokens = torch.cat([load_shard(f) for f in files])
    usable = (len(tokens) - 1) // seq_len * seq_len
    return tokens[: usable + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Faster than LayerNorm: no mean subtraction, no bias, just scale.
    Used in LLaMA, Mistral, Gemma, etc.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.scale


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Instead of adding position embeddings, rotates Q and K vectors by a
    position-dependent angle. This encodes RELATIVE position implicitly.

    Partial RoPE: only apply to the first `rope_dims` dimensions of each head.
    Remaining dims attend without position bias (position-invariant patterns).
    """
    def __init__(self, head_dim: int, rope_dims: int, base: float = 10000.0):
        super().__init__()
        self.rope_dims = min(rope_dims, head_dim)
        # Frequency for each pair of dimensions: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2).float() / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t     = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)   # (T, rope_dims/2)
        # Return cos/sin of shape (1, T, 1, rope_dims/2) — do NOT cat([freqs,freqs]).
        # In forward(), x1 and x2 are each sliced to rope_dims/2 dims, so cos/sin
        # must also have rope_dims/2 in the last dimension to broadcast correctly.
        # (The cat trick is only needed if you store the full rotation in one tensor;
        # here we reconstruct [x1*cos-x2*sin, x1*sin+x2*cos] explicitly.)
        return freqs.cos()[None, :, None, :].to(dtype), freqs.sin()[None, :, None, :].to(dtype)
        # shapes: (1, T, 1, rope_dims/2) — broadcast over batch and heads

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to x (B, T, H, head_dim).
        Only the first rope_dims dimensions are rotated; the rest pass through unchanged.
        """
        r    = self.rope_dims
        x_r  = x[..., :r]              # dimensions to rotate
        x_p  = x[..., r:]              # dimensions to pass through (partial RoPE)
        half = r // 2
        x1, x2 = x_r[..., :half], x_r[..., half:]
        # Rotation: [x1, x2] → [x1·cos - x2·sin,  x1·sin + x2·cos]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return torch.cat([rotated, x_p], dim=-1)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Standard Multi-Head Attention: every head has its own Q, K, V.
    Multi-Query Attention (MQA): all heads share ONE K, V pair.
    Grouped Query Attention: heads are grouped, each group shares K, V.

    Benefits:
      - Fewer KV parameters → smaller model, less KV cache memory
      - Same quality as MHA at smaller model sizes
      - Used in LLaMA-2, Mistral, Gemma, etc.

    Here: num_heads=6, num_kv_heads=3 → 2 heads per KV group (ratio 2:1)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.num_heads    = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim     = cfg.model_dim // cfg.num_heads
        self.kv_dim       = cfg.num_kv_heads * self.head_dim

        # Projections
        self.q_proj = nn.Linear(cfg.model_dim, cfg.model_dim,  bias=False)
        self.k_proj = nn.Linear(cfg.model_dim, self.kv_dim,    bias=False)
        self.v_proj = nn.Linear(cfg.model_dim, self.kv_dim,    bias=False)
        self.o_proj = nn.Linear(cfg.model_dim, cfg.model_dim,  bias=False)

        # Q and K normalisation (improves training stability, from QK-Norm paper)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Rotary embeddings
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_dims, cfg.rope_base)

        # Zero-initialize output projection (so residual stream starts clean)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, T, self.num_heads,    self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        # QK-Norm: normalise per head before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply Rotary embeddings
        q = self.rope(q, cos, sin)
        k = self.rope(k, cos, sin)

        # GQA: expand K and V to match number of Q heads
        # Each KV head is repeated (num_heads // num_kv_heads) times
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=2)   # (B, T, num_heads, head_dim)
        v = v.repeat_interleave(groups, dim=2)

        # Rearrange to (B, H, T, head_dim) for PyTorch SDPA
        q = q.transpose(1, 2)   # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention with FlashAttention kernel (PyTorch ≥ 2.0)
        # is_causal=True handles the causal mask automatically and efficiently
        # On Hopper (H200), this uses the FlashAttention-3 backend automatically
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, head_dim)

        # Reassemble heads and project
        y = y.transpose(1, 2).reshape(B, T, D)   # (B, T, D)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Used in LLaMA, PaLM, Gemma — empirically better than GELU/ReLU for LMs.

    Formula: FFN(x) = (W1·x ⊙ σ(W3·x · β)) · W2
    where σ(x·β) is a smooth gating function (Swish = SiLU when β=1).

    Implementation trick: fuse W1 and W3 into one matrix, split after projection.
    This is slightly more efficient (one GEMM instead of two).
    """
    def __init__(self, model_dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(model_dim * mlp_mult)
        # Make hidden divisible by 8 for tensor core alignment
        hidden = (hidden + 7) // 8 * 8

        # gate_proj and up_proj fused into one matrix
        self.gate_up = nn.Linear(model_dim, 2 * hidden, bias=False)
        self.down    = nn.Linear(hidden,    model_dim,  bias=False)

        nn.init.zeros_(self.down.weight)   # zero-init down proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)  # SwiGLU: silu(gate) * up


class TransformerBlock(nn.Module):
    """
    One transformer block: Attention + FFN with pre-norm (norm before each sub-layer).

    Pre-norm (norm → attention → residual) is more stable than post-norm
    and is used by all modern LLMs (GPT-3, LLaMA, Mistral, etc.)
    """
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.model_dim)
        self.attn      = GroupedQueryAttention(cfg)
        self.mlp_norm  = RMSNorm(cfg.model_dim)
        self.mlp       = SwiGLU(cfg.model_dim, cfg.mlp_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer: pre-norm, residual connection
        x = x + self.attn(self.attn_norm(x), cos, sin)
        # FFN sub-layer: pre-norm, residual connection
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """
    Decoder-only GPT-style language model.

    Token IDs → Embeddings → N × TransformerBlock → RMSNorm → Logits

    Design choices vs vanilla GPT-2:
      - Tied input/output embeddings (saves vocab_size × dim parameters)
      - RMSNorm instead of LayerNorm
      - RoPE instead of learned absolute positional embeddings
      - GQA instead of MHA
      - SwiGLU instead of GELU
      - Logit softcap (tanh clipping) to prevent logit explosion
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.tok_emb    = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.blocks     = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm_final = RMSNorm(cfg.model_dim)
        self.rope       = RotaryEmbedding(cfg.model_dim // cfg.num_heads, cfg.rope_dims, cfg.rope_base)
        # Note: we use a single rope instance; cos/sin are recomputed per forward pass

        self.logit_softcap = cfg.logit_softcap
        self._init_weights()

    def _init_weights(self):
        """
        Careful weight initialisation.
        - Embeddings: small normal (std = 0.02, as in GPT-2)
        - Linear projections: orthogonal init for matrix params
        - Output projections already zero-inited in their modules
        """
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.data.abs().max() > 0:
                if module.weight.shape[0] >= 16 and module.weight.shape[1] >= 16:
                    nn.init.orthogonal_(module.weight, gain=0.5)

    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None):
        """
        input_ids:  (B, T)  — token indices
        target_ids: (B, T)  — next-token targets (optional; if None, return logits)

        Returns: loss (scalar) if target_ids given, else logits (B, T, V)
        """
        B, T = input_ids.shape

        # Token embeddings + RMS normalisation of embeddings (stabilises early training)
        x = self.tok_emb(input_ids)          # (B, T, D)
        x = F.rms_norm(x, (x.size(-1),))     # normalise embedding scale

        # Compute RoPE cos/sin for this sequence length (cached after first call)
        cos, sin = self.rope.get_cos_sin(T, input_ids.device, x.dtype)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm_final(x)

        # Tied output projection: use same weight matrix as input embedding (transposed)
        logits = F.linear(x, self.tok_emb.weight)   # (B, T, V)

        # Logit softcap: prevents logit values from growing unboundedly
        # logit = cap * tanh(logit / cap)  →  keeps values in (-cap, cap)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if target_ids is None:
            return logits

        # Cross-entropy loss: average over all token positions
        # Reshape to (B*T, V) and (B*T,) for F.cross_entropy
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )
        return loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimate_mfu(self, batch_tokens: int, step_time_s: float, peak_tflops: float) -> float:
        """
        Model FLOP Utilization (MFU) — fraction of peak FLOPs actually used.

        For a transformer, FLOPs per forward+backward ≈ 6 * N * T
        where N = number of parameters, T = sequence length (batch tokens).
        (Factor of 6: 2 for matmul forward + 2 for backward activations + 2 for backward weights)
        """
        N = self.count_params()
        flops_per_step = 6 * N * batch_tokens
        achieved_tflops = flops_per_step / step_time_s / 1e12
        return achieved_tflops / peak_tflops


# ─────────────────────────────────────────────────────────────────────────────
# Muon Optimizer
# ─────────────────────────────────────────────────────────────────────────────

@torch.compile
def newton_schulz_5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the orthogonal factor of a matrix.

    Given gradient matrix G, we want the update direction U such that:
      - U has orthonormal rows/columns (spectral norm ≈ 1)
      - U is the "nearest orthogonal matrix" to G in Frobenius norm

    This is the Polar Decomposition: G = U @ S @ V^T  (SVD)
    Newton-Schulz approximates U without the expensive full SVD.

    5th-order polynomial: converges in 5 iterations for well-conditioned matrices.
    Coefficients (a,b,c) chosen to maximise convergence radius.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)   # normalise to unit Frobenius norm

    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-schulz).

    An optimizer designed for transformer weight matrices.
    Instead of updating W ← W - lr * g  (like SGD),
    Muon updates: W ← W - lr * orthogonalize(nesterov_momentum(g))

    Why orthogonalize? The gradient update direction is "cleaned up" so each
    singular value is treated equally — equivalent to steepest descent under
    the spectral norm rather than the Euclidean norm.

    Empirically: ~35% faster training than AdamW on LMs (Keller Jordan, 2024).

    In this implementation:
      - 2D weight matrices (Q, K, V, O, gate, up, down projections) → Muon
      - 1D parameters (RMSNorm scales, biases) → AdamW
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        nesterov: bool = True,
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr          = group["lr"]
            momentum    = group["momentum"]
            ns_steps    = group["ns_steps"]
            wd          = group["weight_decay"]
            nesterov    = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialise momentum buffer
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)

                buf = state["buf"]
                buf.mul_(momentum).add_(g)                   # buf = momentum * buf + g

                if nesterov:
                    g = g + momentum * buf                   # Nesterov lookahead
                else:
                    g = buf

                # Only orthogonalize 2D weight matrices
                if g.ndim == 2:
                    g = newton_schulz_5(g, steps=ns_steps)
                    # Scale to preserve update magnitude (compensate for orthogonalization)
                    g = g * max(1, g.shape[0] / g.shape[1]) ** 0.5

                # Weight decay applied directly to parameters (decoupled)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                p.data.add_(g.to(p.dtype), alpha=-lr)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Learning Rate Schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, total_steps: int, cfg: Config) -> float:
    """
    Trapezoidal LR schedule:
      1. Linear warmup:   0 → peak_lr  over warmup_steps
      2. Constant:        peak_lr  (middle phase)
      3. Linear warmdown: peak_lr → 0  over last warmdown_frac * total_steps

    This is the schedule used in Chinchilla, GPT-4, and the parameter-golf SOTA.
    The warmdown phase encourages the optimizer to find a flat, quantization-friendly
    minimum (weights with small norms compress better).
    """
    warmdown_start = total_steps - int(total_steps * cfg.warmdown_frac)

    if step < cfg.warmup_steps:
        # Linear warmup
        return cfg.peak_lr * (step + 1) / cfg.warmup_steps
    elif step < warmdown_start:
        # Constant phase
        return cfg.peak_lr
    else:
        # Linear warmdown to 0
        remaining = total_steps - step
        warmdown_len = total_steps - warmdown_start
        return cfg.peak_lr * remaining / warmdown_len


# ─────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.

    Instead of using the final checkpoint weights, we maintain a running
    average: ema_w = decay * ema_w + (1-decay) * current_w

    Why? Training loss landscapes are noisy. The EMA trajectory is smoother
    and often lands in a better generalisation minimum. Empirically gives
    ~0.002-0.005 BPB improvement for free.

    Decay=0.999 means the EMA represents roughly the last 1/(1-0.999)=1000 steps.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Store a copy of weights (detached, on CPU to save GPU memory)
        self.shadow = {name: p.data.clone().cpu() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data.cpu(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        """Copy EMA weights into the model (in-place)."""
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name].to(p.device))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: val_bpb with sliding window
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_lead_space_lut: torch.Tensor,
    is_boundary_lut: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> dict:
    """
    Compute val_loss (nats) and val_bpb on the validation set.

    Uses a SLIDING WINDOW evaluation:
      - Instead of non-overlapping chunks (naive), we slide with stride=64
      - Each token is scored with maximum available context (seq_len-1 tokens)
      - Only the last `stride` tokens of each window count toward the metric
        (the earlier tokens served as context, not scored)
      - This better estimates the model's actual compression ability

    Why sliding window matters:
      Naive chunking: tokens at position 0 of each chunk have ZERO context.
      Sliding window: every scored token has ~960 tokens of context.
      Improvement: ~0.01-0.03 BPB typically.
    """
    model.eval()
    seq_len    = cfg.val_seq_len
    stride     = cfg.val_stride
    total_toks = val_tokens.numel() - 1   # last token has no target

    # Positions where we start a window
    window_starts = range(0, total_toks - seq_len + 1, stride)

    total_loss   = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_tokens = torch.tensor(0,   dtype=torch.int64,   device=device)
    total_bytes  = torch.tensor(0.0, dtype=torch.float64, device=device)

    context_size = seq_len - stride   # tokens used as context (not scored)

    for ws in window_starts:
        we   = ws + seq_len
        toks = val_tokens[ws : we + 1].to(device=device, dtype=torch.long)
        x    = toks[:-1].unsqueeze(0)   # (1, seq_len)
        y    = toks[1:].unsqueeze(0)    # (1, seq_len)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)            # (1, seq_len, vocab_size)

        # Only score tokens after the context prefix
        score_start = 0 if ws == 0 else context_size
        scored_logits = logits[0, score_start:, :]         # (scored, V)
        scored_tgts   = y[0, score_start:]                 # (scored,)
        scored_prev   = x[0, score_start:]                 # (scored,) — for byte counting

        # Per-token NLL in nats
        nll = F.cross_entropy(
            scored_logits.float(), scored_tgts, reduction="none"
        )                                                   # (scored,)

        total_loss   += nll.sum().double()
        total_tokens += scored_tgts.numel()

        # Count bytes: base bytes + 1 if token has leading space AND prev is not boundary
        tb = base_bytes_lut[scored_tgts].double()
        tb += (has_lead_space_lut[scored_tgts] & ~is_boundary_lut[scored_prev]).double()
        total_bytes += tb.sum()

    val_loss = (total_loss / total_tokens).item()      # nats per token
    val_bpb  = (val_loss / math.log(2)) * (total_tokens / total_bytes).item()

    model.train()
    return {"val_loss": val_loss, "val_bpb": val_bpb}


# ─────────────────────────────────────────────────────────────────────────────
# Inference (next-token generation)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Autoregressive text generation.

    The model generates one token at a time:
      1. Encode the prompt to token IDs
      2. Run a forward pass → get probability distribution over next token
      3. Sample from the distribution (temperature + top-k filtering)
      4. Append the sampled token to the sequence
      5. Repeat until max_new_tokens reached

    Temperature: scales logit magnitudes before softmax
      - temperature → 0:  greedy decoding (always pick highest-prob token)
      - temperature = 1:  sample from the true model distribution
      - temperature > 1:  more random / creative output

    Top-k: only consider the k most likely tokens at each step
      - Prevents sampling very unlikely tokens
      - top_k=1 is greedy decoding
    """
    model.eval()
    ids = sp.encode(prompt)
    x   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        # Truncate context to the last seq_len tokens if needed
        x_ctx = x[:, -model.cfg.train_seq_len:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x_ctx)   # (1, T, V)

        # Take the logits for the LAST position (next token prediction)
        logits = logits[0, -1, :] / temperature   # (V,)

        # Top-k filtering: zero out all logits except the top-k
        if top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[-1]] = float("-inf")

        # Sample from the filtered distribution
        probs  = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # (1,)

        x = torch.cat([x, next_id.unsqueeze(0)], dim=1)

    # Decode all generated tokens (skip the prompt)
    generated_ids = x[0, len(ids):].tolist()
    return sp.decode(generated_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    # ── Setup ───────────────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")   # use TF32 on Ampere+

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  LLM Training from Scratch")
    print(f"{'='*60}")
    print(f"  Device:   {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  CUDA:     {torch.version.cuda}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  Max time: {cfg.max_train_seconds/60:.0f} minutes")
    print(f"{'='*60}\n")

    # ── Logging ─────────────────────────────────────────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    logger = Logger(cfg.log_file)
    logger.log(type="config", **asdict(cfg))

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    assert int(sp.vocab_size()) == cfg.vocab_size, \
        f"Tokenizer vocab {sp.vocab_size()} != cfg.vocab_size {cfg.vocab_size}"
    print(f"  Vocab size: {cfg.vocab_size}")

    # ── Byte LUTs for val_bpb ────────────────────────────────────────────────
    base_bytes_lut, has_lead_space_lut, is_boundary_lut = build_byte_luts(sp, cfg.vocab_size, device)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading training data...")
    train_loader = DataLoader(cfg.train_files, cfg.train_seq_len, cfg.batch_tokens, cfg.seed)

    print("\nLoading validation data...")
    val_tokens = load_val_tokens(cfg.val_files, cfg.val_seq_len).to(device)
    print(f"  Val tokens: {val_tokens.numel():,}")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = GPT(cfg).to(device).to(torch.bfloat16)
    n_params = model.count_params()
    print(f"  Parameters: {n_params/1e6:.2f}M")
    print(f"  Architecture: {cfg.num_layers}L × {cfg.model_dim}d × {cfg.num_heads}H/{cfg.num_kv_heads}KVH")
    print(f"  Model size (BF16): {n_params * 2 / 1e6:.1f} MB")

    logger.log(type="model_info", n_params=n_params, layers=cfg.num_layers,
               dim=cfg.model_dim, heads=cfg.num_heads, kv_heads=cfg.num_kv_heads)

    # ── Compile (torch.compile for Hopper speedup) ───────────────────────────
    # torch.compile() here only *registers* the compiled function.
    # The actual JIT compilation (Triton kernel generation) happens on the
    # FIRST forward pass and takes 60-120s on a Hopper GPU.
    # We trigger a full warmup step HERE, before train_start, so compilation
    # time does NOT eat into the 30-minute training budget.
    print("\nCompiling + warming up torch.compile (first run triggers JIT, ~60-120s)...")
    _init_state = {n: p.data.clone() for n, p in model.named_parameters()}
    compiled_model = torch.compile(model, dynamic=True)
    # Run one full warmup step (forward + backward + optimizer step)
    _wx, _wy = train_loader.next_batch(device)
    _all_params = list(model.parameters())
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _wloss = compiled_model(_wx, _wy)
    _wloss.backward()
    torch.nn.utils.clip_grad_norm_(_all_params, cfg.grad_clip)
    # Minimal optimizer step just to trigger compilation of the optimizer graph too
    for _p in _all_params:
        if _p.grad is not None:
            _p.data.add_(_p.grad, alpha=-1e-10)   # near-zero update, just triggers kernels
            _p.grad = None
    torch.cuda.synchronize()
    # Restore the original untrained weights (undo the warmup step)
    for n, p in model.named_parameters():
        p.data.copy_(_init_state[n])
    del _init_state, _wx, _wy, _wloss, _all_params
    print("  Compilation warmup done.")

    # ── Optimizers ───────────────────────────────────────────────────────────
    # Separate parameter groups:
    #   - 2D weight matrices → Muon (attention projections, MLP weights)
    #   - Everything else   → AdamW (embeddings, RMSNorm scales, 1D params)
    matrix_params = [p for n, p in model.named_parameters()
                     if p.ndim == 2 and "tok_emb" not in n]
    scalar_params  = [p for n, p in model.named_parameters()
                     if p.ndim != 2 or "tok_emb" in n]

    print(f"\nOptimizer setup:")
    print(f"  Muon  params: {sum(p.numel() for p in matrix_params)/1e6:.2f}M (2D weight matrices)")
    print(f"  AdamW params: {sum(p.numel() for p in scalar_params)/1e6:.2f}M (embeddings, norms)")

    muon_opt  = Muon(matrix_params,  lr=cfg.peak_lr,   momentum=cfg.muon_momentum,
                     ns_steps=cfg.muon_ns_steps, weight_decay=cfg.muon_wd)
    adam_opt  = torch.optim.AdamW(scalar_params, lr=cfg.embed_lr, betas=(0.9, 0.95),
                                   weight_decay=cfg.weight_decay, eps=1e-8)

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=cfg.ema_decay)

    # ── Training ─────────────────────────────────────────────────────────────
    # Estimate total steps from time budget
    # We'll run until time runs out; the step count is just for LR schedule
    estimated_step_ms = 200    # conservative estimate for H200 MIG with this model
    estimated_steps   = int(cfg.max_train_seconds * 1000 / estimated_step_ms)
    print(f"\nEstimated steps in {cfg.max_train_seconds/60:.0f} min: ~{estimated_steps:,}")
    print(f"Training tokens: ~{estimated_steps * cfg.batch_tokens / 1e6:.0f}M\n")
    print(f"{'─'*60}")
    print(f"  {'step':>6}  {'train_loss':>10}  {'val_loss':>10}  {'val_bpb':>8}  {'tok/s':>8}  {'MFU':>6}  {'lr':>8}")
    print(f"{'─'*60}")

    # Run initial validation (step 0, untrained model)
    val_metrics = evaluate(model, val_tokens, base_bytes_lut, has_lead_space_lut,
                           is_boundary_lut, cfg, device)
    print(f"  {'0':>6}  {'—':>10}  {val_metrics['val_loss']:>10.4f}  "
          f"{val_metrics['val_bpb']:>8.4f}  {'—':>8}  {'—':>6}  {'—':>8}   [initial]")
    logger.log(type="val", step=0, **val_metrics)

    step           = 0
    train_start    = time.perf_counter()
    step_times     = []
    loss_history   = []

    model.train()

    while True:
        step_start = time.perf_counter()
        elapsed    = step_start - train_start

        # Hard stop at time limit
        if elapsed >= cfg.max_train_seconds:
            print(f"\n  Time limit reached ({cfg.max_train_seconds/60:.0f} min). Stopping.")
            break

        # Dynamically compute total_steps for LR schedule based on elapsed fraction
        # We recalibrate every step as we learn the actual step time
        if step_times:
            avg_step_s   = sum(step_times[-20:]) / len(step_times[-20:])
            remaining_s  = cfg.max_train_seconds - elapsed
            total_steps  = step + max(1, int(remaining_s / avg_step_s))
        else:
            total_steps = estimated_steps

        # ── LR update ───────────────────────────────────────────────────────
        lr = get_lr(step, total_steps, cfg)
        for pg in muon_opt.param_groups:
            pg["lr"] = lr
        for pg in adam_opt.param_groups:
            pg["lr"] = lr * (cfg.embed_lr / cfg.peak_lr)  # scale proportionally

        # ── Forward + backward ─────────────────────────────────────────────
        x, y = train_loader.next_batch(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled_model(x, y)

        loss.backward()

        # Gradient clipping (prevents spikes from ruining training)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        muon_opt.step()
        adam_opt.step()
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

        # ── EMA update ──────────────────────────────────────────────────────
        ema.update(model)

        # ── Timing and logging ──────────────────────────────────────────────
        torch.cuda.synchronize()
        step_time_s = time.perf_counter() - step_start
        step_times.append(step_time_s)

        tok_per_sec = cfg.batch_tokens / step_time_s
        mfu         = model.estimate_mfu(cfg.batch_tokens, step_time_s, cfg.peak_tflops)
        loss_val    = loss.item()
        loss_history.append(loss_val)

        if step % cfg.log_every_steps == 0:
            elapsed_min = (time.perf_counter() - train_start) / 60
            logger.log(
                type="train", step=step, train_loss=loss_val,
                tok_per_sec=tok_per_sec, grad_norm=grad_norm.item(),
                lr=lr, mfu=mfu, elapsed_min=elapsed_min,
            )

        # ── Periodic validation ──────────────────────────────────────────────
        if step > 0 and step % cfg.val_every_steps == 0:
            # Save current weights, apply EMA for validation
            orig_state = {n: p.data.clone() for n, p in model.named_parameters()}
            ema.apply_to(model)

            val_metrics = evaluate(model, val_tokens, base_bytes_lut, has_lead_space_lut,
                                   is_boundary_lut, cfg, device)

            # Restore training weights
            for n, p in model.named_parameters():
                p.data.copy_(orig_state[n])

            elapsed_total = time.perf_counter() - train_start
            avg_step_ms   = 1000 * sum(step_times[-cfg.val_every_steps:]) / cfg.val_every_steps

            print(f"  {step:>6}  {loss_val:>10.4f}  {val_metrics['val_loss']:>10.4f}  "
                  f"{val_metrics['val_bpb']:>8.4f}  {tok_per_sec:>8.0f}  {mfu:>6.3f}  "
                  f"{lr:>8.2e}  [{elapsed_total/60:.1f}min, {avg_step_ms:.0f}ms/step]")

            logger.log(type="val", step=step, **val_metrics,
                       elapsed_min=elapsed_total / 60, avg_step_ms=avg_step_ms)

        step += 1

    # ── Final evaluation with EMA weights ────────────────────────────────────
    total_time = time.perf_counter() - train_start
    print(f"\n{'='*60}")
    print(f"  Training complete: {step} steps in {total_time/60:.1f} min")
    print(f"  Avg step time: {1000*total_time/step:.0f}ms")
    print(f"  Total tokens seen: {step * cfg.batch_tokens / 1e6:.1f}M")
    print(f"{'='*60}")

    print("\nApplying EMA weights and running final evaluation...")
    ema.apply_to(model)

    final_val = evaluate(model, val_tokens, base_bytes_lut, has_lead_space_lut,
                         is_boundary_lut, cfg, device)
    print(f"\n  FINAL val_loss: {final_val['val_loss']:.6f} nats")
    print(f"  FINAL val_bpb:  {final_val['val_bpb']:.6f}")

    # Context: SOTA is 1.081, naive baseline was 1.224
    # Our small model will be higher (worse), around 2.5-3.5 depending on training time
    print(f"\n  Context:")
    print(f"    Random model:      ~8.0 bpb  (no learning)")
    print(f"    Our model:          {final_val['val_bpb']:.3f} bpb")
    print(f"    Param-golf baseline: 1.224 bpb (9L-512d, 10min 8xH100)")
    print(f"    Param-golf SOTA:     1.081 bpb (11L-512d, 10min 8xH100 + tricks)")

    logger.log(type="final", step=step, **final_val,
               total_steps=step, total_time_min=total_time / 60,
               total_tokens_M=step * cfg.batch_tokens / 1e6)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg.checkpoint_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           asdict(cfg),
        "step":             step,
        "val_bpb":          final_val["val_bpb"],
        "val_loss":         final_val["val_loss"],
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    # ── Inference demo ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  INFERENCE DEMO")
    print(f"{'='*60}")

    model.eval()
    prompts = [
        "The history of artificial intelligence",
        "In recent years, machine learning has",
        "Scientists have discovered that",
    ]

    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")
        generated = generate(model, sp, prompt, max_new_tokens=100,
                             temperature=0.8, top_k=50, device=device)
        print(f"  Generated: {generated}")
        logger.log(type="inference", prompt=prompt, generated=generated, step=step)

    print(f"\n{'='*60}")
    print(f"  Log file: {cfg.log_file}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")

    return final_val


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a small LLM from scratch on FineWeb")
    parser.add_argument("--data_dir",           default="./data")
    parser.add_argument("--log_file",           default="training_log.jsonl")
    parser.add_argument("--checkpoint_dir",     default="checkpoints")
    parser.add_argument("--max_train_seconds",  type=float, default=1800.0)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--num_layers",         type=int,   default=8)
    parser.add_argument("--model_dim",          type=int,   default=384)
    parser.add_argument("--num_heads",          type=int,   default=6)
    parser.add_argument("--num_kv_heads",       type=int,   default=3)
    parser.add_argument("--batch_tokens",       type=int,   default=32768)
    parser.add_argument("--peak_lr",            type=float, default=3e-3)
    parser.add_argument("--val_every_steps",    type=int,   default=100)
    args = parser.parse_args()
    cfg  = Config(**{k: v for k, v in vars(args).items()})
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
