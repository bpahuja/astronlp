#!/usr/bin/env python3
"""
Embed methodology paragraphs with your trained SPECTER2 adapter, then cluster.

Requires:
  pip install umap-learn hdbscan scikit-learn sentence-transformers adapters

Usage:
  python embed_and_cluster.py \
    --paras_dir /path/to/methodology_txts \
    --model_dir /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/specter2_astro_adapter \
    --base_model allenai/specter2_base \
    --out_dir   ./cluster_out \
    --umap_dim  30 --min_cluster_size 10 --min_samples 2 \
    --window_tokens 180 --stride 30 --max_seq_length 256 --device auto
"""
import os, re, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, models
import adapters
from transformers import AutoTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import umap
import hdbscan

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

def read_paras(dir_path: Path):
    rows = []
    for p in sorted(dir_path.glob("*.txt")):
        pid = p.stem
        raw = p.read_text(encoding="utf-8", errors="ignore")
        # paragraphs split on blank lines
        parts = re.split(r"\n\s*\n", raw.strip())
        if len(parts) == 1:
            parts = [ln for ln in raw.splitlines() if ln.strip()]
        parts = [clean_text(x) for x in parts]
        parts = [x for x in parts if len(x) >= 200]
        for i, x in enumerate(parts):
            rows.append({"paper_id": pid, "para_id": f"{pid}::p{i:04d}", "text": x})
    return pd.DataFrame(rows)

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

def encode_hier(st, tok, texts, window_tokens=180, stride=30, batch_size=64):
    # hierarchical: split per item into token windows, encode all windows, mean-pool per item
    embeddings = []
    for txt in tqdm(texts, desc="Encoding (hierarchical)"):
        ids = tok.encode(txt, add_special_tokens=False)
        if len(ids) <= window_tokens:
            emb = st.encode([txt], convert_to_numpy=True, normalize_embeddings=True)[0]
            embeddings.append(emb); continue
        # build chunks
        chunks = []
        step = max(1, window_tokens - stride)
        for s in range(0, max(1, len(ids) - window_tokens + 1), step):
            sub = tok.decode(ids[s:s+window_tokens], skip_special_tokens=True)
            chunks.append(sub)
        vecs = st.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        embeddings.append(vecs.mean(axis=0))
    return np.vstack(embeddings)

def tfidf_top_terms(texts, labels, n_top=10):
    mask = labels >= 0
    X_texts = [t for t, keep in zip(texts, mask) if keep]
    y = labels[mask]
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=3)
    X = vec.fit_transform(X_texts)
    X = normalize(X, norm="l2")
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if not len(idx): continue
        scores = np.asarray(X[idx].mean(axis=0)).ravel()
        top = vocab[scores.argsort()[-n_top:][::-1]]
        out[int(c)] = top.tolist()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paras_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--base_model", default="allenai/specter2_base")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--umap_dim", type=int, default=30)
    ap.add_argument("--min_cluster_size", type=int, default=10)
    ap.add_argument("--min_samples", type=int, default=2)

    ap.add_argument("--window_tokens", type=int, default=180)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")

    args = ap.parse_args()
    device = parse_device(args.device)

    df = read_paras(Path(args.paras_dir))
    if df.empty: 
        raise SystemExit("No methodology paragraphs found.")

    st = build_model_with_adapter(args.model_dir, args.base_model, args.max_seq_length, device=device)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    embs = encode_hier(st, tok, df["text"].tolist(), window_tokens=args.window_tokens, stride=args.stride, batch_size=args.batch_size)

    # UMAP → HDBSCAN
    reducer = umap.UMAP(n_components=args.umap_dim, metric="cosine", random_state=42)
    Z = reducer.fit_transform(embs)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, metric="euclidean")
    labels = clusterer.fit_predict(Z)

    # Stats
    n = len(labels)
    n_noise = int((labels < 0).sum())
    n_clusters = int((labels >= 0).sum() and labels.max() + 1 or 0)
    noise_pct = 100.0 * n_noise / n
    print(f"Paragraphs: {n} | clusters: {n_clusters} | noise: {n_noise} ({noise_pct:.1f}%)")

    # Top terms per cluster
    tops = tfidf_top_terms(df["text"].tolist(), labels, n_top=12)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # Save assignments
    out_df = df.copy()
    out_df["cluster"] = labels
    out_df.to_csv(out / "paragraph_clusters.csv", index=False)

    # Save summaries
    with (out / "cluster_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "paragraphs": n, "clusters": n_clusters, "noise": n_noise, "noise_pct": noise_pct,
            "top_terms": tops
        }, f, ensure_ascii=False, indent=2)

    # Paper-level bag-of-clusters
    bag = (out_df[out_df["cluster"] >= 0]
           .groupby(["paper_id", "cluster"]).size().unstack(fill_value=0).sort_index(axis=1))
    bag.to_csv(out / "paper_bag_of_methods.csv")

if __name__ == "__main__":
    main()
