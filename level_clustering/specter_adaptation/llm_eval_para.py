#!/usr/bin/env python3
import os
import json
import time
import backoff
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# If FAISS install is problematic, set USE_FAISS = False to use sklearn NearestNeighbors
USE_FAISS = False
if USE_FAISS:
    try:
        import faiss
    except ImportError:
        USE_FAISS = False
        from sklearn.neighbors import NearestNeighbors
else:
    from sklearn.neighbors import NearestNeighbors

# ---- OpenAI setup ----
from openai import OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
client = OpenAI(api_key=api_key)

# -------------------------
# Config
# -------------------------
# Input paths - adjusted for SPECTER2 clustering output
CLUSTER_OUTPUT_DIR     = "/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/cluster_out_v7_astrobert_1"  # Directory with clustering results
EMBEDDINGS_DIR         = "/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/embeddings_v1_astrobert"  # Directory with SPECTER2 embeddings

# Output paths
OUTPUT_CSV             = "llm_v7/cluster_llm_evaluation_specter2.csv"
OUTPUT_JSONL           = "llm_v7/cluster_llm_evaluation_specter2.jsonl"

# Retrieval
NEIGHBOURS_PER_CLUSTER = 5      # number of neighbours to show near the centroid
MAX_CLUSTERS           = 600    # set an int to cap clusters for a quick pass
MIN_CLUSTER_SIZE       = 1     # skip tiny clusters

# Prompt control
MAX_TEXT_CHARS = 1200           # per paragraph to stay within token limits
SYSTEM_PROMPT = """You are a careful scientific methods reviewer. 
Given a set of astrophysics paper paragraphs, judge whether their methodologies are close to each other. 
Be strict about methodology, not just topic. Short, precise answers."""

USER_PROMPT_TEMPLATE = """You are given paragraphs from papers that were clustered together by a vector method that is supposed to represent METHODOLOGICAL similarity.

Task:
1) Judge how methodologically close these paragraphs are overall on a 1–5 scale:
   1 = not close, 3 = somewhat close, 5 = very close.
2) Provide a short method-family label (2–6 words).
3) Give a brief rationale in 2–3 sentences focused on methods.

Guidance:
- Focus on methodology: data processing pipelines, simulation types, inference frameworks, instruments, or learning algorithms.
- Ignore surface topic overlap if the methods differ.

Cluster centroid description:
{centroid_note}

Paragraphs (truncated) of the 5 nearest members in this cluster:
{text_block}

Respond in strict JSON with keys: score, label, rationale.
"""

# -------------------------
# Data loading for SPECTER2 format
# -------------------------
def load_specter2_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load SPECTER2 embeddings and metadata from chunked format"""
    embeddings_dir = Path(embeddings_dir)
    
    # Load embedding info
    info_file = embeddings_dir / "embedding_info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"Embedding info not found: {info_file}")
    
    with info_file.open() as f:
        info = json.load(f)
    
    logger.info(f"Loading {info['total_paragraphs']} embeddings from {info['total_chunks']} chunks")
    
    # Get chunk directories
    chunk_dirs = sorted([d for d in embeddings_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("chunk_")])
    
    all_embeddings = []
    all_metadata = []
    
    for chunk_dir in chunk_dirs:
        emb_file = chunk_dir / "embeddings.npy"
        meta_file = chunk_dir / "metadata.csv"
        
        if emb_file.exists() and meta_file.exists():
            embeddings = np.load(emb_file).astype(np.float32)
            metadata = pd.read_csv(meta_file)
            
            all_embeddings.append(embeddings)
            all_metadata.append(metadata)
    
    # Combine
    X = np.vstack(all_embeddings)
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    
    logger.info(f"Loaded {X.shape[0]} embeddings with dimension {X.shape[1]}")
    
    return X, metadata_df

def load_cluster_results(cluster_dir: Path) -> Tuple[np.ndarray, Dict]:
    """Load clustering results from SPECTER2 clustering output"""
    cluster_dir = Path(cluster_dir)
    
    # Load cluster summary
    summary_file = cluster_dir / "cluster_summary.json"
    if summary_file.exists():
        with summary_file.open() as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Load cluster assignments from CSV parts
    cluster_files = sorted(cluster_dir.glob("paragraph_clusters_part_*.csv"))
    
    if not cluster_files:
        raise FileNotFoundError(f"No cluster result files found in {cluster_dir}")
    
    all_dfs = []
    for cf in cluster_files:
        df = pd.read_csv(cf)
        all_dfs.append(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Extract cluster labels
    labels = full_df['cluster'].values
    
    logger.info(f"Loaded {len(labels)} cluster assignments")
    logger.info(f"Found {summary.get('clusters', 'unknown')} clusters with {summary.get('noise_pct', 'unknown')}% noise")
    
    return labels, summary

# -------------------------
# Retrieval helpers
# -------------------------
def build_index(X_norm: np.ndarray):
    """Build search index for embeddings"""
    n, d = X_norm.shape
    if USE_FAISS:
        logger.info("Using FAISS")
        index = faiss.IndexFlatIP(d)  # dot product on L2-normalised equals cosine similarity
        index.add(X_norm)
        return index
    else:
        logger.info("Using NN")
        nn = NearestNeighbors(n_neighbors=NEIGHBOURS_PER_CLUSTER + 10, metric="cosine", algorithm="auto")
        nn.fit(X_norm)
        return nn

def cluster_centroid(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """Compute cluster centroid"""
    return X[idxs].mean(axis=0, keepdims=True)

def nearest_in_cluster(
    X_norm: np.ndarray,
    index,
    cluster_indices: np.ndarray,
    k: int = 5
) -> List[int]:
    """Find nearest neighbors to cluster centroid within the cluster"""
    # centroid in normalised space
    c = normalize(cluster_centroid(X_norm, cluster_indices))
    
    if USE_FAISS:
        # Retrieve many neighbours globally, then filter to cluster
        sims, inds = index.search(c, min(k + 50, X_norm.shape[0]))
        candidates = inds[0].tolist()
    else:
        # brute via cosine similarity among cluster members only
        sims = cosine_similarity(c, X_norm[cluster_indices])[0]
        order = np.argsort(-sims)
        return cluster_indices[order[:k]].tolist()

    # keep only members of the same cluster
    cluster_set = set(cluster_indices.tolist())
    filtered = [i for i in candidates if i in cluster_set]
    
    if len(filtered) < k:
        # fallback: compute within-cluster cosine
        sims = cosine_similarity(c, X_norm[cluster_indices])[0]
        order = np.argsort(-sims)
        filtered = cluster_indices[order[:k]].tolist()
    else:
        filtered = filtered[:k]
    
    return filtered

# -------------------------
# LLM call with retry
# -------------------------
@backoff.on_exception(backoff.expo, Exception, max_time=120, max_tries=5, jitter=None)
def call_llm(system_prompt: str, user_prompt: str) -> Dict:
    """Call OpenAI API with retry logic"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user",   "content": user_prompt.strip()},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)

# -------------------------
# Formatting utilities
# -------------------------
def truncate_text(s: str, limit: int) -> str:
    """Truncate text to character limit"""
    s = s.replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[:limit].rsplit(" ", 1)[0] + "..."

def build_prompt(centroid_note: str, texts: List[str]) -> str:
    """Build LLM prompt with paragraph texts"""
    lines = []
    for i, text in enumerate(texts, 1):
        lines.append(f"[{i}] {truncate_text(text, MAX_TEXT_CHARS)}")
    text_block = "\n\n".join(lines)
    return USER_PROMPT_TEMPLATE.format(
        centroid_note=centroid_note.strip(),
        text_block=text_block
    )

# -------------------------
# Main evaluation
# -------------------------
def evaluate_clusters_with_llm(
    X: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cluster_summary: Dict,
    neighbours_per_cluster: int = 5,
    max_clusters: Optional[int] = None,
    min_cluster_size: int = 10,
    centroid_note_text: str = "Centroid is computed in SPECTER2 embedding space; members shown are the nearest paragraphs in that space."
) -> pd.DataFrame:
    """Evaluate clusters using LLM"""
    
    assert X.shape[0] == len(labels) == len(texts), "Embeddings, labels, and texts must be aligned."
    
    # normalise embeddings for cosine retrieval
    print("normalizing")
    # X_norm = normalize(X.astype(np.float32))
    
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    print("norms calculated")
    np.maximum(norms, 1e-12, out=norms)  # avoid divide-by-zero
    X /= norms
    X_norm = X  # already normalized, no extra copy
    print("mornalization done")
    index = build_index(X_norm)
    
    # unique clusters excluding noise label
    uniq = np.unique(labels)
    uniq = [c for c in uniq if c >= 0]  # Exclude -1 (noise)
    
    if max_clusters is not None:
        uniq = uniq[:max_clusters]
    
    # Get top terms if available
    top_terms = cluster_summary.get('top_terms', {})
    
    rows = []
    for c in uniq:
        cluster_indices = np.where(labels == c)[0]
        if cluster_indices.size < min_cluster_size:
            continue
        
        # Get cluster's top terms if available
        cluster_terms = top_terms.get(str(c), [])
        terms_str = ', '.join(cluster_terms[:5]) if cluster_terms else "N/A"
        
        # representative docs near centroid
        picked = nearest_in_cluster(X_norm, index, cluster_indices, k=neighbours_per_cluster)
        texts_subset = [texts[i] for i in picked]
        
        # Add top terms to centroid note
        enhanced_centroid_note = f"{centroid_note_text} Top cluster terms: {terms_str}"
        
        prompt = build_prompt(enhanced_centroid_note, texts_subset)
        
        try:
            result = call_llm(SYSTEM_PROMPT, prompt)
            score = int(result.get("score", 0))
            label = result.get("label", "").strip()
            rationale = result.get("rationale", "").strip()
        except Exception as e:
            logger.warning(f"LLM error for cluster {c}: {e}")
            score, label, rationale = 0, "LLM_error", f"{type(e).__name__}: {e}"
        
        rows.append({
            "cluster_id": int(c),
            "cluster_size": int(cluster_indices.size),
            "picked_indices": json.dumps(picked),
            "top_terms": terms_str,
            "score": score,
            "label": label,
            "rationale": rationale
        })
        
        logger.info(f"Evaluated cluster {c} (size={cluster_indices.size}): score={score}, label={label}")
        
        # polite pacing to avoid hitting rate limits
        time.sleep(0.3)
    
    df = pd.DataFrame(rows).sort_values(["score", "cluster_size"], ascending=[False, False])
    return df

# -------------------------
# Run
# -------------------------
def main():
    logger.info("Starting LLM evaluation of SPECTER2 clusters...")
    
    # Check paths exist
    cluster_dir = Path(CLUSTER_OUTPUT_DIR)
    embeddings_dir = Path(EMBEDDINGS_DIR)
    
    if not cluster_dir.exists():
        raise FileNotFoundError(f"Cluster output directory not found: {cluster_dir}")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    # Load data
    logger.info("Loading embeddings...")
    X, metadata_df = load_specter2_embeddings(embeddings_dir)
    
    logger.info("Loading cluster results...")
    labels, cluster_summary = load_cluster_results(cluster_dir)
    
    # Verify alignment
    if len(labels) != X.shape[0]:
        raise ValueError(f"Mismatch: {len(labels)} labels vs {X.shape[0]} embeddings")
    if len(metadata_df) != X.shape[0]:
        raise ValueError(f"Mismatch: {len(metadata_df)} metadata rows vs {X.shape[0]} embeddings")
    
    # Extract texts from metadata
    texts = metadata_df['text'].tolist()
    
    logger.info(f"Evaluating clusters with LLM...")
    logger.info(f"Total paragraphs: {len(texts)}")
    logger.info(f"Total clusters: {len(np.unique(labels[labels >= 0]))}")
    
    # Run evaluation
    df = evaluate_clusters_with_llm(
        X=X,
        labels=labels,
        texts=texts,
        cluster_summary=cluster_summary,
        neighbours_per_cluster=NEIGHBOURS_PER_CLUSTER,
        max_clusters=MAX_CLUSTERS,
        min_cluster_size=MIN_CLUSTER_SIZE,
    )
    
    # Save results
    logger.info(f"Writing {OUTPUT_CSV} and {OUTPUT_JSONL}")
    df.to_csv(OUTPUT_CSV, index=False)
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    
    # Print summary
    if not df.empty:
        print("\n" + "="*60)
        print("TOP 10 CLUSTERS BY LLM SCORE:")
        print("="*60)
        print(df.head(10).to_string(index=False))
        
        print("\n" + "="*60)
        print("SCORE DISTRIBUTION:")
        print("="*60)
        score_dist = df["score"].value_counts().sort_index()
        for score, count in score_dist.items():
            print(f"Score {score}: {count} clusters")
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS:")
        print("="*60)
        print(f"Total clusters evaluated: {len(df)}")
        print(f"Mean score: {df['score'].mean():.2f}")
        print(f"Median score: {df['score'].median():.1f}")
        print(f"Mean cluster size: {df['cluster_size'].mean():.1f}")
        print(f"Total paragraphs in evaluated clusters: {df['cluster_size'].sum()}")
    else:
        logger.warning("No clusters evaluated. Check filters or inputs.")

if __name__ == "__main__":
    main()