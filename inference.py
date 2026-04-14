"""
inference.py
============
Load a trained model checkpoint and run interactive text generation.

This is the script you use AFTER training to actually talk to your model.

Usage:
------
# Interactive REPL (type prompts, get responses):
    python inference.py checkpoints/final_model.pt

# Single prompt via command line:
    python inference.py checkpoints/final_model.pt --prompt "The history of AI is"

# Batch mode (multiple prompts from a file):
    python inference.py checkpoints/final_model.pt --prompt_file prompts.txt

# Show what's inside the checkpoint:
    python inference.py checkpoints/final_model.pt --info

What's in final_model.pt?
--------------------------
It's a Python dict saved by torch.save(), containing:
  - "model_state_dict" : all learned weights (the actual trained parameters)
  - "config"           : the Config dataclass as a dict (architecture hyperparams)
  - "step"             : how many gradient steps were taken
  - "val_bpb"          : final validation bits-per-byte score
  - "val_loss"         : final validation loss in nats

The model_state_dict is a dict mapping parameter names → tensors, e.g.:
  "tok_emb.weight"                → shape (1024, 384)
  "blocks.0.attn.q_proj.weight"   → shape (384, 384)
  "blocks.0.mlp.gate_up.weight"   → shape (3072, 384)
  ...

Nothing else is needed to run inference: just the config and state dict.

Production Deployment Options:
-------------------------------
For research/demos (what we're doing here):
  → Load the .pt file directly in Python, call generate()
  → Serve via a simple Flask/FastAPI endpoint

For real production:
  → Export to ONNX: torch.onnx.export(model, ...)
  → Compile with TorchScript: torch.jit.script(model)
  → Quantize: torch.quantization.quantize_dynamic(model)
  → Serve with TorchServe, Triton Inference Server, or vLLM
  → For edge/browser: export to GGUF format (llama.cpp compatible)

For the parameter-golf challenge specifically:
  → The final submission IS the deployment: train_gpt.py loads the compressed
    checkpoint inline during eval and runs inference on the validation set.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm


# ─────────────────────────────────────────────────────────────────────────────
# We import the model from the training script so we don't duplicate code.
# If you prefer a standalone file, copy the model classes here.
# ─────────────────────────────────────────────────────────────────────────────

# Add the parent directory to path so we can import from train_llm_scratch
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_llm_scratch import GPT, Config
except ImportError:
    print("ERROR: Could not import from train_llm_scratch.py")
    print("Make sure inference.py is in the same directory as train_llm_scratch.py")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_checkpoint(path: str):
    """Print everything stored in the checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {path}")
    print(f"{'='*60}")

    # File size
    size_mb = Path(path).stat().st_size / 1e6
    print(f"\n  File size: {size_mb:.1f} MB")

    # Training stats
    print(f"\n  Training stats:")
    print(f"    Steps trained:  {ckpt.get('step', 'unknown'):,}")
    print(f"    Final val_loss: {ckpt.get('val_loss', 'unknown'):.6f} nats")
    print(f"    Final val_bpb:  {ckpt.get('val_bpb',  'unknown'):.6f}")

    # Config (architecture)
    cfg_dict = ckpt.get("config", {})
    print(f"\n  Architecture:")
    for key in ["num_layers", "model_dim", "num_heads", "num_kv_heads",
                "mlp_mult", "vocab_size", "train_seq_len"]:
        if key in cfg_dict:
            print(f"    {key}: {cfg_dict[key]}")

    # Parameter count
    state = ckpt["model_state_dict"]
    n_params = sum(v.numel() for v in state.values())
    print(f"\n  Parameters: {n_params/1e6:.2f}M")

    # List all tensors (parameter names and shapes)
    print(f"\n  Parameter tensors ({len(state)} total):")
    for name, tensor in sorted(state.items()):
        print(f"    {name:55s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> tuple:
    """
    Load a trained model and its tokenizer from a checkpoint file.

    Returns: (model, sp, cfg)
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct config from saved dict
    cfg_dict = ckpt["config"]
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in Config.__dataclass_fields__})

    # Build model and load weights
    model = GPT(cfg).to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = model.count_params()
    print(f"  Model: {cfg.num_layers}L × {cfg.model_dim}d  ({n_params/1e6:.2f}M params)")
    print(f"  Trained for: {ckpt.get('step', '?'):,} steps")
    print(f"  val_bpb: {ckpt.get('val_bpb', '?'):.4f}  |  val_loss: {ckpt.get('val_loss', '?'):.4f}")

    # Load tokenizer
    print(f"  Tokenizer: {cfg.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)

    return model, sp, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model:          nn.Module,
    sp:             spm.SentencePieceProcessor,
    prompt:         str,
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_k:          int   = 50,
    top_p:          float = 0.95,
    device:         torch.device = torch.device("cpu"),
    show_tokens:    bool  = False,
) -> str:
    """
    Autoregressive token-by-token text generation.

    How it works:
      1. Encode the prompt text to token IDs using the SentencePiece tokenizer
      2. Feed the token IDs through the model → logits over the vocabulary
      3. Sample the next token using temperature + top-k + top-p (nucleus) filtering
      4. Append the sampled token and repeat from step 2
      5. Decode all generated token IDs back to text

    Sampling parameters:
      temperature (float, default 0.8):
        Scales logits before softmax. Lower = more deterministic, higher = more random.
        0.0 → always pick the most likely token (greedy)
        1.0 → sample from the true model distribution
        >1.0 → more creative but potentially incoherent

      top_k (int, default 50):
        At each step, only consider the top-k most likely tokens.
        Prevents sampling from the long tail of unlikely tokens.
        1 = greedy decoding

      top_p (float, default 0.95):
        Nucleus sampling: only keep tokens whose cumulative probability ≤ top_p.
        More adaptive than top-k: uses fewer candidates when distribution is peaked.
        Combined with top-k, this gives very controlled yet creative outputs.
    """
    model.eval()

    # Tokenise the prompt
    input_ids = sp.encode(prompt)
    if show_tokens:
        pieces = [sp.id_to_piece(i) for i in input_ids]
        print(f"  Prompt tokens ({len(input_ids)}): {pieces}")

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    t_start = time.perf_counter()
    generated_ids = []

    for step in range(max_new_tokens):
        # Only feed the last seq_len tokens (context window limit)
        x_ctx = x[:, -model.cfg.train_seq_len:]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x_ctx)    # (1, T, vocab_size)

        # Next-token logits (last position)
        logits = logits[0, -1, :].float()   # (vocab_size,)

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy: pick argmax directly
            next_id = logits.argmax().item()
            generated_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            continue

        # Top-k filtering: zero out all logits except top-k
        if top_k > 0 and top_k < logits.size(-1):
            top_vals, _ = torch.topk(logits, top_k)
            logits[logits < top_vals[-1]] = float("-inf")

        # Top-p (nucleus) filtering: keep smallest set of tokens summing to p
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens once cumulative prob exceeds top_p
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()   # shift right: always keep at least one
            remove[0]  = False
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

        # Sample from the filtered distribution
        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    elapsed    = time.perf_counter() - t_start
    tok_per_s  = max_new_tokens / elapsed

    if show_tokens:
        gen_pieces = [sp.id_to_piece(i) for i in generated_ids]
        print(f"  Generated tokens: {gen_pieces}")
        print(f"  Speed: {tok_per_s:.0f} tok/s")

    return sp.decode(generated_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity evaluation on a text string
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model: nn.Module, sp, text: str, device: torch.device) -> dict:
    """
    Compute perplexity of a text string under the model.

    Perplexity = exp(cross_entropy_loss) = exp(average NLL per token)

    A perplexity of N means the model is as confused as if it had to choose
    uniformly among N equally likely options at each position.

    Lower perplexity = model finds the text more predictable.

    Note: perplexity is token-level (depends on tokenizer), while val_bpb is
    byte-level (tokenizer-agnostic). For comparison:
      perplexity = exp(val_loss)
      val_bpb    = val_loss / ln(2) × tokens_per_byte
    """
    model.eval()
    ids = sp.encode(text)
    if len(ids) < 2:
        return {"error": "Text too short (need at least 2 tokens)"}

    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:],  dtype=torch.long, device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(x)   # (1, T, vocab_size)

    loss = F.cross_entropy(logits[0].float(), y, reduction="mean")
    ppl  = torch.exp(loss).item()

    return {
        "text_length_chars":  len(text),
        "num_tokens":         len(ids),
        "loss_nats":          loss.item(),
        "perplexity":         ppl,
        "bits_per_token":     loss.item() / torch.log(torch.tensor(2.0)).item(),
        "chars_per_token":    len(text) / len(ids),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def interactive_repl(model, sp, cfg, device, args):
    """Interactive prompt loop — type prompts, receive generated text."""
    print(f"\n{'='*60}")
    print("  Interactive Inference")
    print(f"  Model: {cfg.num_layers}L × {cfg.model_dim}d  |  val_bpb: see checkpoint")
    print(f"  Temperature: {args.temperature}  |  top_k: {args.top_k}  |  top_p: {args.top_p}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"{'='*60}")
    print("  Commands:")
    print("    Type any text prompt and press Enter to generate")
    print("    :temp 0.9     — change temperature")
    print("    :topk 30      — change top_k")
    print("    :len 300      — change max_new_tokens")
    print("    :ppl <text>   — compute perplexity of text")
    print("    :quit or Ctrl-C to exit")
    print(f"{'='*60}\n")

    temperature    = args.temperature
    top_k          = args.top_k
    top_p          = args.top_p
    max_new_tokens = args.max_new_tokens

    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue

        # Handle commands
        if prompt.startswith(":quit"):
            break
        elif prompt.startswith(":temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"  Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("  Usage: :temp 0.9")
            continue
        elif prompt.startswith(":topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"  top_k set to {top_k}")
            except (IndexError, ValueError):
                print("  Usage: :topk 50")
            continue
        elif prompt.startswith(":len "):
            try:
                max_new_tokens = int(prompt.split()[1])
                print(f"  max_new_tokens set to {max_new_tokens}")
            except (IndexError, ValueError):
                print("  Usage: :len 200")
            continue
        elif prompt.startswith(":ppl "):
            text = prompt[5:].strip()
            if text:
                metrics = compute_perplexity(model, sp, text, device)
                print(f"\n  Perplexity metrics:")
                for k, v in metrics.items():
                    print(f"    {k:25s}: {v:.4f}" if isinstance(v, float) else f"    {k:25s}: {v}")
                print()
            continue

        # Generate
        t0 = time.perf_counter()
        output = generate(
            model, sp, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        elapsed   = time.perf_counter() - t0
        tok_speed = max_new_tokens / elapsed

        print(f"\n{'-'*40}")
        print(f"{prompt}{output}")
        print(f"{'-'*40}")
        print(f"  [{max_new_tokens} tokens in {elapsed:.1f}s = {tok_speed:.0f} tok/s]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a trained LLM checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat
  python inference.py checkpoints/final_model.pt

  # Single prompt
  python inference.py checkpoints/final_model.pt --prompt "The future of AI"

  # Multiple prompts from file (one per line)
  python inference.py checkpoints/final_model.pt --prompt_file prompts.txt

  # Show checkpoint contents
  python inference.py checkpoints/final_model.pt --info

  # Compute perplexity of a string
  python inference.py checkpoints/final_model.pt --ppl "The quick brown fox"
        """
    )
    parser.add_argument("checkpoint",        help="Path to final_model.pt checkpoint")
    parser.add_argument("--prompt",          type=str, default=None,
                        help="Single text prompt for generation")
    parser.add_argument("--prompt_file",     type=str, default=None,
                        help="File with one prompt per line")
    parser.add_argument("--ppl",             type=str, default=None,
                        help="Text to compute perplexity on")
    parser.add_argument("--info",            action="store_true",
                        help="Print checkpoint contents and exit")
    parser.add_argument("--max_new_tokens",  type=int,   default=200)
    parser.add_argument("--temperature",     type=float, default=0.8,
                        help="Sampling temperature (0=greedy, 1=model dist, >1=random)")
    parser.add_argument("--top_k",           type=int,   default=50,
                        help="Top-k filtering (0 to disable)")
    parser.add_argument("--top_p",           type=float, default=0.95,
                        help="Nucleus sampling threshold (1.0 to disable)")
    parser.add_argument("--show_tokens",     action="store_true",
                        help="Print individual token pieces during generation")
    parser.add_argument("--cpu",             action="store_true",
                        help="Force CPU inference (useful if no GPU available)")
    args = parser.parse_args()

    # ── Checkpoint info only ─────────────────────────────────────────────────
    if args.info:
        inspect_checkpoint(args.checkpoint)
        return

    # ── Device ───────────────────────────────────────────────────────────────
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Running on CPU (slow for generation, but works)")
    else:
        device = torch.device("cuda")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ───────────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, sp, cfg = load_model(args.checkpoint, device)

    # ── Perplexity mode ───────────────────────────────────────────────────────
    if args.ppl:
        print(f"\nComputing perplexity for: \"{args.ppl}\"")
        metrics = compute_perplexity(model, sp, args.ppl, device)
        print(f"\n  Perplexity metrics:")
        for k, v in metrics.items():
            print(f"    {k:25s}: {v:.4f}" if isinstance(v, float) else f"    {k:25s}: {v}")
        return

    # ── Single prompt ─────────────────────────────────────────────────────────
    if args.prompt:
        print(f"\nPrompt: \"{args.prompt}\"")
        output = generate(
            model, sp, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            show_tokens=args.show_tokens,
        )
        print(f"\n{args.prompt}{output}\n")
        return

    # ── Prompt file ───────────────────────────────────────────────────────────
    if args.prompt_file:
        prompts = [line.strip() for line in Path(args.prompt_file).read_text().splitlines()
                   if line.strip() and not line.startswith("#")]
        print(f"\nGenerating for {len(prompts)} prompts from {args.prompt_file}\n")
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] Prompt: \"{prompt}\"")
            output = generate(
                model, sp, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
            print(f"Generated: {prompt}{output}")
            print(f"{'─'*50}")
        return

    # ── Default: interactive REPL ─────────────────────────────────────────────
    interactive_repl(model, sp, cfg, device, args)


if __name__ == "__main__":
    main()
