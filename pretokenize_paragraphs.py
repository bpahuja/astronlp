import os
import json
import torch
import argparse
from tqdm import tqdm
import re
import multiprocessing as mp
import lmdb
import pickle
# We only need this import in the main process and the initializer
from transformers import AutoTokenizer 

# --- Best Practice: Prevent Deadlocks with Tokenizers and Forking ---
# This should be set before any other transformers imports if possible,
# especially before creating a multiprocessing Pool.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global tokenizer accessible to each worker process after initialization
global_tokenizer = None

def init_tokenizer(model_name, cache_path, max_length):
    """
    Initializes the tokenizer for each worker process.
    This function is called once per worker. It will load the tokenizer
    from the local cache path, which was already populated by the main process.
    """
    global global_tokenizer, GLOBAL_MAX_LEN
    global_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path, use_fast=True)
    GLOBAL_MAX_LEN = max_length

def clean_math_tags(paragraph):
    """Removes math tags from a paragraph."""
    return re.sub(r'\[MATH_tex=.*?\]', '[MATH]', paragraph)

def chunk_paragraph(text):
    """Tokenizes and chunks a paragraph into manageable lengths."""
    global global_tokenizer, GLOBAL_MAX_LEN
    tokens = global_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    # Chunk tokens into sequences of max_length, accounting for [CLS] and [SEP]
    for i in range(0, len(tokens), GLOBAL_MAX_LEN - 2):
        chunk = tokens[i:i + GLOBAL_MAX_LEN - 2]
        # Add special tokens
        chunk = [global_tokenizer.cls_token_id] + chunk + [global_tokenizer.sep_token_id]
        chunks.append(chunk)
    if not chunks:
        # Handle empty paragraphs
        chunks = [[global_tokenizer.cls_token_id, global_tokenizer.sep_token_id]]
    return chunks

def process_row(row):
    """Processes a single row from the input JSONL file."""
    try:
        paper_id = row.get("paper_id", "unknown")
        section_index = row.get("section_index", -1)
        para_index = row.get("paragraph_index", -1)
        paragraph = clean_math_tags(row.get("paragraph", ""))
        
        # Create a unique key. We serialize it to a string for consistency.
        key = f"{paper_id}|{section_index}|{para_index}"
        
        chunks = chunk_paragraph(paragraph)
        
        # We must serialize the value (the list of lists) to bytes. Pickle is good for this.
        value = pickle.dumps(chunks)
        
        return key.encode('utf-8'), value
    except Exception as e:
        # Return None for rows that fail, so we can filter them out
        print(f"Error processing row: {row}. Error: {e}")
        return None, None


def main(args):
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- KEY CHANGE: Pre-load and cache the tokenizer in the main process ---
    print("Pre-loading and caching tokenizer in main process...")
    _ = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path, use_fast=True)
    print("Tokenizer is cached.")

    # Initialize the LMDB environment.
    env = lmdb.open(args.output_file, map_size=1024**4, writemap=True)

    # Spawn safe multiprocessing. The initializer will now load from the cache.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=min(8, mp.cpu_count()),
        initializer=init_tokenizer,
        initargs=(args.model_name, args.cache_path, args.max_length)
    )

    with open(args.input_jsonl, "r") as f:
        batched_lines = []
        pbar = tqdm(desc="Processing and writing to LMDB")

        for line in f:
            batched_lines.append(json.loads(line))
            
            # Process in batches for efficiency
            if len(batched_lines) == 5000:
                results = pool.map(process_row, batched_lines)
                
                # Write the processed results to the database in a single transaction
                with env.begin(write=True) as txn:
                    for key, value in results:
                        if key is not None:
                            txn.put(key, value)
                
                pbar.update(len(batched_lines))
                batched_lines = []

        # Process any remaining lines
        if batched_lines:
            results = pool.map(process_row, batched_lines)
            with env.begin(write=True) as txn:
                for key, value in results:
                    if key is not None:
                        txn.put(key, value)
            pbar.update(len(batched_lines))

    pbar.close()
    pool.close()
    pool.join()
    env.close()

    print(f"✅ Successfully saved tokenized paragraphs to LMDB: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="data/unlabelled_methodology/pretokenized_db")
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--cache_path", type=str, default="/vol/bitbucket/bp824/hf_models/bert-base")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    main(args)
