"""
cached_challenge_fineweb.py
============================
Download the exact FineWeb dataset used by the OpenAI Parameter Golf challenge.

This downloads PRE-TOKENIZED binary shards from HuggingFace.
No tokenization needed — OpenAI/kevclark already ran it for you.

The data is identical to what every SOTA submission trains and evaluates on,
so your val_bpb scores are directly comparable to the leaderboard.

WHAT GETS DOWNLOADED
--------------------
  data/datasets/fineweb10B_sp1024/
      fineweb_train_000000.bin  ← training shards (100M tokens each, ~190MB each)
      fineweb_train_000001.bin
      ...
      fineweb_val_000000.bin    ← validation shard (FIXED, same for every submission)
      fineweb_val_000001.bin
      ...
  data/tokenizers/
      fineweb_1024_bpe.model    ← SentencePiece tokenizer (1024 vocab)

USAGE
-----
  # Minimum for a teaching run (4 train shards ~760MB + full val ~380MB):
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4

  # Official challenge default (80 train shards = ~8B tokens, ~15GB):
  python3 data/cached_challenge_fineweb.py --variant sp1024

  # Absolute minimum smoke test (1 train shard):
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

BINARY FORMAT (for reference, matches train_llm_scratch.py exactly)
---------------------------------------------------------------------
  Header:  256 × int32  (little-endian)
    [0]  = 20240520  (magic number)
    [1]  = 1         (version)
    [2]  = N         (number of tokens in this shard)
    [3..255] = 0     (reserved / padding)
  Data:    N × uint16  (token IDs, 0–1023 for sp1024)

WHY THESE FILES MATCH THE CHALLENGE
-------------------------------------
  Source:    HuggingFace repo "kevclark/parameter-golf"
             This is the official pre-tokenized mirror maintained by the challenge.
  Dataset:   HuggingFaceFW/fineweb  (NOT fineweb-edu — that's a different distribution)
  Tokenizer: SentencePiece BPE, 1024 vocab, trained on FineWeb
  Val set:   Fixed first 50,000 documents of FineWeb — NEVER changes across runs.
             This is what val_bpb is measured on for all leaderboard entries.

DEPENDENCIES
------------
  pip install huggingface_hub tqdm
  (already installed if you followed setup_env.sh)
"""

import argparse
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# HuggingFace repo with the official pre-tokenized shards
# This is the canonical source used by all parameter-golf participants
OFFICIAL_HF_REPO   = "kevclark/parameter-golf"

# Local paths (relative to the repo root — run this script from there)
DATA_ROOT          = Path("data")
DATASETS_DIR       = DATA_ROOT / "datasets"
TOKENIZERS_DIR     = DATA_ROOT / "tokenizers"

# Supported tokenizer variants
VARIANTS = {
    "sp1024": {
        "dataset_subdir":   "fineweb10B_sp1024",
        "tokenizer_file":   "fineweb_1024_bpe.model",
        "remote_prefix":    "sp1024",          # prefix inside the HF repo
        "vocab_size":       1024,
        "description":      "SentencePiece BPE, 1024 vocab (default challenge variant)",
    },
    "sp8192": {
        "dataset_subdir":   "fineweb10B_sp8192",
        "tokenizer_file":   "fineweb_8192_bpe.model",
        "remote_prefix":    "sp8192",
        "vocab_size":       8192,
        "description":      "SentencePiece BPE, 8192 vocab (used by SOTA submissions)",
    },
}

# Magic number and version in every shard header — must match train_llm_scratch.py
SHARD_MAGIC   = 20240520
SHARD_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hf_download_file(repo_id: str, remote_path: str, local_path: Path, quiet: bool = False):
    """Download a single file from a HuggingFace dataset repo."""
    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        if not quiet:
            print(f"  [skip] already exists: {local_path}")
        return True

    if not quiet:
        print(f"  Downloading {remote_path} → {local_path}")

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place the file nested; move it to the right place
        downloaded_path = Path(downloaded)
        if downloaded_path != local_path and downloaded_path.exists():
            downloaded_path.rename(local_path)
        return True
    except Exception as e:
        print(f"  ERROR downloading {remote_path}: {e}")
        return False


def _verify_shard(path: Path) -> bool:
    """Quick sanity check: verify the binary header of a downloaded shard."""
    import numpy as np
    try:
        header = np.fromfile(str(path), dtype="<i4", count=256)
        if len(header) < 3:
            return False
        if int(header[0]) != SHARD_MAGIC:
            print(f"  WARNING: bad magic in {path.name}: {header[0]} (expected {SHARD_MAGIC})")
            return False
        if int(header[1]) != SHARD_VERSION:
            print(f"  WARNING: bad version in {path.name}: {header[1]}")
            return False
        n_tokens = int(header[2])
        if n_tokens <= 0:
            print(f"  WARNING: zero tokens in {path.name}")
            return False
        return True
    except Exception as e:
        print(f"  WARNING: could not read {path.name}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main download function
# ─────────────────────────────────────────────────────────────────────────────

def download(variant: str, n_train_shards: int, repo_id: str):
    """
    Download pre-tokenized FineWeb shards + tokenizer from HuggingFace.

    Parameters
    ----------
    variant       : "sp1024" or "sp8192"
    n_train_shards: how many training shards to download (each ~190MB / 100M tokens)
    repo_id       : HuggingFace dataset repo ID (default: kevclark/parameter-golf)
    """
    if variant not in VARIANTS:
        print(f"ERROR: unknown variant '{variant}'. Choose from: {list(VARIANTS.keys())}")
        sys.exit(1)

    cfg         = VARIANTS[variant]
    dataset_dir = DATASETS_DIR / cfg["dataset_subdir"]
    token_dir   = TOKENIZERS_DIR
    prefix      = cfg["remote_prefix"]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Downloading FineWeb ({cfg['description']})")
    print(f"  Source repo: {repo_id}")
    print(f"  Train shards: {n_train_shards}  (~{n_train_shards * 190}MB)")
    print(f"  Val shards:   all (fixed challenge validation set)")
    print(f"  Local dir:    {dataset_dir}")
    print(f"{'='*60}\n")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("[1/3] Downloading tokenizer...")
    tok_remote = f"tokenizers/{cfg['tokenizer_file']}"
    tok_local  = token_dir / cfg["tokenizer_file"]
    ok = _hf_download_file(repo_id, tok_remote, tok_local)
    if not ok:
        print("\nERROR: Could not download tokenizer. Check your internet connection.")
        print(f"  Tried: {repo_id} / {tok_remote}")
        sys.exit(1)
    print(f"  Tokenizer ready: {tok_local}\n")

    # ── Training shards ────────────────────────────────────────────────────
    print(f"[2/3] Downloading {n_train_shards} training shard(s)...")
    train_ok = 0
    for i in range(n_train_shards):
        remote = f"datasets/{prefix}/fineweb_train_{i:06d}.bin"
        local  = dataset_dir / f"fineweb_train_{i:06d}.bin"
        if _hf_download_file(repo_id, remote, local, quiet=False):
            if _verify_shard(local):
                train_ok += 1
            else:
                print(f"  WARNING: shard {i} failed verification — deleting and will re-download next time")
                local.unlink(missing_ok=True)

    print(f"  Training shards ready: {train_ok}/{n_train_shards}\n")

    # ── Validation shards (download ALL — val set is fixed) ────────────────
    print("[3/3] Downloading validation shards (fixed challenge val set)...")
    print("  The val set is the first 50,000 FineWeb documents, identical")
    print("  for every participant — required for comparable val_bpb scores.")
    val_ok = 0
    for i in range(100):    # try up to 100 val shards; stop when 404
        remote = f"datasets/{prefix}/fineweb_val_{i:06d}.bin"
        local  = dataset_dir / f"fineweb_val_{i:06d}.bin"
        if local.exists():
            val_ok += 1
            continue
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=repo_id,
                filename=remote,
                repo_type="dataset",
                local_dir=str(dataset_dir),
                local_dir_use_symlinks=False,
            )
            if _verify_shard(local):
                val_ok += 1
                print(f"  Downloaded: {local.name}")
            else:
                local.unlink(missing_ok=True)
        except Exception:
            # File doesn't exist on the repo — we've downloaded all val shards
            break

    if val_ok == 0:
        print("\n  ERROR: Could not download any validation shards.")
        print("  Without the val set, val_bpb cannot be computed.")
        sys.exit(1)
    print(f"  Validation shards ready: {val_ok}\n")

    # ── Summary ────────────────────────────────────────────────────────────
    train_files = sorted(dataset_dir.glob("fineweb_train_*.bin"))
    val_files   = sorted(dataset_dir.glob("fineweb_val_*.bin"))
    total_mb    = sum(f.stat().st_size for f in train_files + val_files) / 1e6

    print(f"{'='*60}")
    print(f"  Download complete!")
    print(f"  Train shards: {len(train_files)} files")
    print(f"  Val shards:   {len(val_files)} files")
    print(f"  Total size:   {total_mb:.0f} MB on disk")
    print(f"  Dataset dir:  {dataset_dir}")
    print(f"  Tokenizer:    {tok_local}")
    print(f"{'='*60}")

    print(f"""
  NEXT STEP — run the training script:

    python train_llm_scratch.py \\
        --data_dir    ./data \\
        --num_layers  8 \\
        --model_dim   384 \\
        --num_heads   6 \\
        --num_kv_heads 3 \\
        --batch_tokens 8192 \\
        --max_train_seconds 1800

  Or submit to the cluster:

    sbatch train_saxa.slurm   (for saxa H200 node)
    sbatch train_damnii.slurm (for damnii RTX 2080 Ti, smaller batch)
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download official FineWeb data for the parameter-golf challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Teaching run — 4 train shards (~760MB) + full val set:
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4

  # Absolute minimum smoke test (1 train shard, ~190MB):
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

  # Full challenge dataset (80 train shards = 8B tokens, ~15GB):
  python3 data/cached_challenge_fineweb.py --variant sp1024

  # SOTA variant (8192 vocab, used by 1.081 BPB submission):
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 4
        """
    )
    parser.add_argument(
        "--variant", default="sp1024",
        choices=list(VARIANTS.keys()),
        help="Tokenizer variant: sp1024 (default) or sp8192 (SOTA uses this)",
    )
    parser.add_argument(
        "--train-shards", type=int, default=80,
        metavar="N",
        help=(
            "Number of training shards to download. "
            "Each shard = 100M tokens (~190MB). "
            "Default 80 (8B tokens). For teaching use 4. For smoke test use 1."
        ),
    )
    parser.add_argument(
        "--repo", default=OFFICIAL_HF_REPO,
        help=f"HuggingFace dataset repo to download from (default: {OFFICIAL_HF_REPO})",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only verify existing files, don't download anything new",
    )

    args = parser.parse_args()

    # Verify dependencies
    try:
        import huggingface_hub
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  Run: pip install huggingface_hub")
        sys.exit(1)

    try:
        import numpy
    except ImportError:
        print("ERROR: numpy not installed.")
        print("  Run: pip install numpy")
        sys.exit(1)

    # Check mode: just verify what's already downloaded
    if args.check:
        cfg         = VARIANTS[args.variant]
        dataset_dir = DATASETS_DIR / cfg["dataset_subdir"]
        train_files = sorted(dataset_dir.glob("fineweb_train_*.bin"))
        val_files   = sorted(dataset_dir.glob("fineweb_val_*.bin"))
        print(f"\nVerifying existing files in {dataset_dir}...")
        all_ok = True
        for f in train_files + val_files:
            ok = _verify_shard(f)
            status = "✓" if ok else "✗ CORRUPT"
            size_mb = f.stat().st_size / 1e6
            print(f"  {status}  {f.name}  ({size_mb:.0f} MB)")
            if not ok:
                all_ok = False
        print(f"\n  {'All files OK' if all_ok else 'Some files have issues — re-run without --check to re-download'}")
        return

    # Must run from repo root (so paths resolve correctly)
    if not Path("data").exists():
        print("ERROR: 'data/' directory not found.")
        print("  Run this script from the repo root:")
        print("    cd ~/parameter-golf")
        print("    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4")
        sys.exit(1)

    download(
        variant       = args.variant,
        n_train_shards = args.train_shards,
        repo_id       = args.repo,
    )


if __name__ == "__main__":
    main()
