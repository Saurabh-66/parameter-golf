import os
import numpy as np
import torch
from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm

# Constants to match your train_llm_scratch.py configuration
DATA_DIR = "data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "data/tokenizers/fineweb_1024_bpe.model"
MAGIC_NUMBER = 20240520
VERSION = 1

def write_shard(tokens, filename):
    """Writes tokens to a .bin file with the specific 256-int header."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC_NUMBER
    header[1] = VERSION
    header[2] = len(tokens)
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Load the tokenizer (Ensure this file exists in data/tokenizers/)
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}")
        return
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    
    # 2. Stream the FineWeb-Edu dataset
    print("Loading FineWeb-Edu (10BT subset)...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    all_tokens = []
    token_count = 0
    shard_idx = 1
    target_tokens_per_shard = 100_000_000 # 100M tokens per file
    
    print("Tokenizing and writing shards...")
    for example in tqdm(fw):
        text = example["text"]
        tokens = sp.encode_as_ids(text)
        all_tokens.extend(tokens)
        token_count += len(tokens)
        
        # Write a shard once we hit the target size
        if len(all_tokens) >= target_tokens_per_shard:
            shard_path = os.path.join(DATA_DIR, f"fineweb_train_{shard_idx:06d}.bin")
            write_shard(np.array(all_tokens), shard_path)
            print(f" Saved shard {shard_idx} to {shard_path}")
            all_tokens = []
            shard_idx += 1
            
        # Stop after 1 shard for a "smoke test" or continue for the full 10B
        if shard_idx > 5: # Adjust this for more data
            break

    # Write remaining tokens to a validation shard
    if all_tokens:
        val_path = os.path.join(DATA_DIR, "fineweb_val_000001.bin")
        write_shard(np.array(all_tokens), val_path)
        print(f"Saved validation shard to {val_path}")

if __name__ == "__main__":
    main()