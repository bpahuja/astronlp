#!/usr/bin/env python3
"""
Embed methodology paragraphs with your trained SPECTER2 adapter in memory-efficient batches.

Requires:
  pip install sentence-transformers adapters tqdm

Usage:
  python embed_paragraphs.py \
    --paras_dir /path/to/methodology_txts \
    --model_dir /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/specter2_astro_adapter \
    --base_model allenai/specter2_base \
    --out_dir   ./embeddings_out \
    --window_tokens 180 --stride 30 --max_seq_length 256 \
    --batch_size 32 --chunk_size 10000 --device auto
"""
import os, re, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc

from sentence_transformers import SentenceTransformer, models
import adapters
from transformers import AutoTokenizer

def clean_text(t: str) -> str:
    MATH = re.compile(r"(\$[^$]+\$)|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)")
    CIT  = re.compile(r"\(([^()]*\d{4}[^()]*)\)|\[[0-9,\s;]+\]")
    URL  = re.compile(r"https?://\S+")
    FIG  = re.compile(r"^\s*(Figure|Table)\s+\d+", re.I)
    t = re.sub(MATH, " ", t or "")
    t = re.sub(CIT, " <CIT> ", t)
    t = re.sub(URL, " ", t)
    lines = [ln for ln in (t or "").splitlines() if not re.match(FIG, ln or "")]
    t = " ".join(lines)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def read_paras_generator(dir_path: Path):
    """Generator to read paragraphs without loading all into memory"""
    for p in sorted(dir_path.glob("*.txt")):
        pid = p.stem
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            # paragraphs split on blank lines
            parts = re.split(r"\n\s*\n", raw.strip())
            if len(parts) == 1:
                parts = [ln for ln in raw.splitlines() if ln.strip()]
            parts = [clean_text(x) for x in parts]
            parts = [x for x in parts if len(x) >= 200]
            for i, x in enumerate(parts):
                yield {"paper_id": pid, "para_id": f"{pid}::p{i:04d}", "text": x}
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue

def build_model_with_adapter(model_dir: str, base_id: str, max_seq_len: int, device: str):
    word = models.Transformer(base_id, max_seq_length=max_seq_len)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    st = SentenceTransformer(modules=[word, pool], device=device)
    adapters.init(word.auto_model)
    adapter_dir = os.path.join(model_dir, "adapter")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter folder not found: {adapter_dir}")
    word.auto_model.load_adapter(adapter_dir, load_as="astro_methods")
    word.auto_model.set_active_adapters("astro_methods")
    return st

def parse_device(arg: str):
    import torch
    if arg and arg.lower() != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"

def encode_hier_single(st, tok, txt, window_tokens=180, stride=30):
    """Encode a single text with hierarchical windowing"""
    ids = tok.encode(txt, add_special_tokens=False)
    if len(ids) <= window_tokens:
        emb = st.encode([txt], convert_to_numpy=True, normalize_embeddings=True)[0]
        return emb
    
    # build chunks
    chunks = []
    step = max(1, window_tokens - stride)
    for s in range(0, max(1, len(ids) - window_tokens + 1), step):
        sub = tok.decode(ids[s:s+window_tokens], skip_special_tokens=True)
        chunks.append(sub)
    
    vecs = st.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, 
                    batch_size=min(32, len(chunks)), show_progress_bar=False)
    return vecs.mean(axis=0)

def process_chunk(st, tok, chunk_data, window_tokens, stride, batch_size):
    """Process a chunk of paragraphs efficiently"""
    embeddings = []
    texts = [item['text'] for item in chunk_data]
    
    # Group texts by similar length for more efficient batching
    text_lengths = [(i, len(tok.encode(txt, add_special_tokens=False))) for i, txt in enumerate(texts)]
    text_lengths.sort(key=lambda x: x[1])
    
    for i, txt_len in text_lengths:
        txt = texts[i]
        emb = encode_hier_single(st, tok, txt, window_tokens, stride)
        embeddings.append((i, emb))
    
    # Sort back to original order
    embeddings.sort(key=lambda x: x[0])
    return np.array([emb for _, emb in embeddings])

def save_chunk_embeddings(embeddings, chunk_data, chunk_idx, out_dir):
    """Save embeddings and metadata for a chunk"""
    chunk_dir = out_dir / f"chunk_{chunk_idx:04d}"
    chunk_dir.mkdir(exist_ok=True)
    
    # Save embeddings as numpy
    np.save(chunk_dir / "embeddings.npy", embeddings)
    
    # Save metadata
    metadata_df = pd.DataFrame(chunk_data)
    metadata_df.to_csv(chunk_dir / "metadata.csv", index=False)
    
    return len(chunk_data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paras_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--base_model", default="allenai/specter2_base")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--window_tokens", type=int, default=180)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--chunk_size", type=int, default=10000, help="Number of paragraphs per chunk")
    ap.add_argument("--device", default="auto")

    args = ap.parse_args()
    device = parse_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model on device: {device}")
    st = build_model_with_adapter(args.model_dir, args.base_model, args.max_seq_length, device=device)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Process in chunks
    chunk_idx = 0
    chunk_data = []
    total_processed = 0
    
    print("Processing paragraphs in chunks...")
    para_generator = read_paras_generator(Path(args.paras_dir))
    
    for para_data in tqdm(para_generator, desc="Reading paragraphs"):
        chunk_data.append(para_data)
        
        if len(chunk_data) >= args.chunk_size:
            # Process this chunk
            print(f"\nProcessing chunk {chunk_idx} ({len(chunk_data)} paragraphs)...")
            embeddings = process_chunk(st, tok, chunk_data, args.window_tokens, 
                                     args.stride, args.batch_size)
            
            # Save chunk
            chunk_size = save_chunk_embeddings(embeddings, chunk_data, chunk_idx, out_dir)
            total_processed += chunk_size
            
            print(f"Saved chunk {chunk_idx}: {chunk_size} embeddings")
            
            # Clear memory
            del embeddings, chunk_data
            gc.collect()
            
            # Reset for next chunk
            chunk_data = []
            chunk_idx += 1
    
    # Process remaining data
    if chunk_data:
        print(f"\nProcessing final chunk {chunk_idx} ({len(chunk_data)} paragraphs)...")
        embeddings = process_chunk(st, tok, chunk_data, args.window_tokens, 
                                 args.stride, args.batch_size)
        chunk_size = save_chunk_embeddings(embeddings, chunk_data, chunk_idx, out_dir)
        total_processed += chunk_size
        print(f"Saved final chunk {chunk_idx}: {chunk_size} embeddings")
    
    # Save processing info
    info = {
        "total_paragraphs": total_processed,
        "total_chunks": chunk_idx + (1 if chunk_data else 0),
        "chunk_size": args.chunk_size,
        "embedding_dim": embeddings.shape[1] if 'embeddings' in locals() else "unknown",
        "model_config": {
            "base_model": args.base_model,
            "window_tokens": args.window_tokens,
            "stride": args.stride,
            "max_seq_length": args.max_seq_length
        }
    }
    
    with (out_dir / "embedding_info.json").open("w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nEmbedding complete!")
    print(f"Total paragraphs processed: {total_processed}")
    print(f"Total chunks saved: {info['total_chunks']}")
    print(f"Output directory: {out_dir}")

if __name__ == "__main__":
    main()