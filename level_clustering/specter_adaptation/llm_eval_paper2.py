#!/usr/bin/env python3
"""
Paper Cluster LLM Evaluation Script
Evaluates paper-level clusters from HDBSCAN clustering using LLMs.

Input: 
  - CSV file with paper_id, hdbscan_label columns (cluster assignments)
  - CSV file with paper_id and bag-of-papers vectors (embeddings)
  - abstracts.jsonl containing paper abstracts

Output:
  - paper_cluster_llm_evaluation.csv with cluster quality scores
  - paper_cluster_llm_evaluation.jsonl with detailed results
"""

import os
import json
import time
import random
import argparse
import backoff
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ---- OpenAI setup ----
from openai import OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
client = OpenAI(api_key=api_key)

# -------------------------
# Config
# -------------------------
OUTPUT_CSV = "paper_cluster_llm_evaluation.csv"
OUTPUT_JSONL = "paper_cluster_llm_evaluation.jsonl"

# Retrieval
NEIGHBOURS_PER_CLUSTER = 5      # number of neighbours to show near the centroid
MAX_CLUSTERS = None             # set an int to cap clusters for a quick pass
MIN_CLUSTER_SIZE = 3            # skip tiny clusters

# Prompt control
MAX_ABSTRACT_CHARS = 1200      # per abstract to stay well within token limits
SYSTEM_PROMPT = """You are a careful scientific methods reviewer. 
Given a set of astrophysics papers, judge whether their methodologies are close to each other. 
Be strict about methodology, not just topic. Short, precise answers."""

USER_PROMPT_TEMPLATE = """You are given abstracts from papers that were clustered together by HDBSCAN clustering that is supposed to represent METHODOLOGICAL similarity.

Task:
1) Judge how methodologically close these papers are overall on a 1–5 scale:
   1 = not close, 3 = somewhat close, 5 = very close.
2) Provide a short method-family label (2–6 words).
3) Give a brief rationale in 2–3 sentences focused on methods.

Guidance:
- Focus on methodology: data processing pipelines, simulation types, inference frameworks, instruments, or learning algorithms.
- Ignore surface topic overlap if the methods differ.

Cluster centroid description:
{centroid_note}

Abstracts (truncated) of the {n_papers} nearest members in this cluster:
{abstract_block}

Respond in strict JSON with keys: score, label, rationale.
"""

# -------------------------
# Data loading
# -------------------------
def load_cluster_assignments(path: Path) -> pd.DataFrame:
    """Load cluster assignments from CSV with paper_id and hdbscan_label columns"""
    print(f"Loading cluster assignments from {path}")
    
    if not path.exists():
        raise FileNotFoundError(f"Cluster assignments file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Check for required columns
    if 'paper_id' not in df.columns:
        raise ValueError("Missing 'paper_id' column in cluster assignments file")
    if 'hdbscan_label' not in df.columns:
        raise ValueError("Missing 'hdbscan_label' column in cluster assignments file")
    
    print(f"Loaded {len(df):,} paper cluster assignments")
    print(f"Unique clusters: {df['hdbscan_label'].nunique():,}")
    
    # Count clusters (excluding noise)
    valid_clusters = df[df['hdbscan_label'] >= 0]['hdbscan_label'].nunique()
    noise_papers = (df['hdbscan_label'] == -1).sum()
    
    print(f"Valid clusters: {valid_clusters}")
    print(f"Noise papers: {noise_papers}")
    
    return df

def load_paper_vectors(path: Path) -> pd.DataFrame:
    """Load paper vectors from CSV with paper_id and vector columns"""
    print(f"Loading paper vectors from {path}")
    
    if not path.exists():
        raise FileNotFoundError(f"Paper vectors file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Check for required columns
    if 'paper_id' not in df.columns:
        raise ValueError("Missing 'paper_id' column in paper vectors file")
    
    # Get vector columns (assume all columns except paper_id are vector dimensions)
    vector_cols = [col for col in df.columns if col != 'paper_id']
    
    if not vector_cols:
        raise ValueError("No vector columns found in paper vectors file")
    
    print(f"Loaded {len(df):,} paper vectors")
    print(f"Vector dimensions: {len(vector_cols)}")
    
    return df

def merge_clusters_and_vectors(cluster_df: pd.DataFrame, vector_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Merge cluster assignments with vectors and return combined df and vector column names"""
    print("Merging cluster assignments with vectors...")
    
    # Merge on paper_id
    merged_df = pd.merge(cluster_df, vector_df, on='paper_id', how='inner')
    
    print(f"Successfully merged {len(merged_df):,} papers (from {len(cluster_df):,} cluster assignments and {len(vector_df):,} vectors)")
    
    # Get vector column names
    vector_cols = [col for col in vector_df.columns if col != 'paper_id']
    
    return merged_df, vector_cols

def extract_embeddings_and_labels(df: pd.DataFrame, vector_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract vectors as embeddings and cluster labels"""
    
    print(f"Extracting {len(vector_cols)} dimensional vectors")
    
    # Extract embeddings
    X = df[vector_cols].values.astype(np.float32)
    
    # Extract labels
    labels = df['hdbscan_label'].values
    
    # Extract paper IDs
    paper_ids = df['paper_id'].tolist()
    
    print(f"Embeddings shape: {X.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return X, labels, paper_ids

def load_abstracts_dict(jsonl_path: Path) -> Dict[str, str]:
    """Returns dict: paper_id -> abstract"""
    print(f"Loading abstracts from {jsonl_path}")
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Abstracts file not found: {jsonl_path}")
    
    d = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = str(obj.get("paper_id", "")).strip()
            if pid:
                d[pid] = obj.get("abstract", "") or ""
    
    print(f"Loaded {len(d):,} abstracts")
    return d

def align_abstracts_with_papers(paper_ids: List[str], abstracts_by_pid: Dict[str, str]) -> List[str]:
    """Align abstracts with paper order in embeddings"""
    abstracts = []
    missing_count = 0
    
    for pid in paper_ids:
        pid_str = str(pid)
        # Try different formats for paper ID matching
        if pid_str in abstracts_by_pid:
            abstracts.append(abstracts_by_pid[pid_str])
        elif pid_str.replace("_", "/") in abstracts_by_pid:
            abstracts.append(abstracts_by_pid[pid_str.replace("_", "/")])
        elif pid_str.replace("/", "_") in abstracts_by_pid:
            abstracts.append(abstracts_by_pid[pid_str.replace("/", "_")])
        else:
            abstracts.append("")
            missing_count += 1
    
    if missing_count > 0:
        print(f"Warning: {missing_count} papers missing abstracts")
    
    return abstracts

# -------------------------
# Retrieval helpers
# -------------------------
def cluster_centroid(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """Compute centroid of cluster members"""
    if len(idxs) == 0:
        return np.zeros((1, X.shape[1]))
    return X[idxs].mean(axis=0, keepdims=True)

def nearest_in_cluster(X_norm: np.ndarray, cluster_indices: np.ndarray, k: int = 5) -> List[int]:
    """Find k nearest members to cluster centroid"""
    if len(cluster_indices) <= k:
        return cluster_indices.tolist()
    
    # Compute centroid and normalize
    c = normalize(cluster_centroid(X_norm, cluster_indices))
    
    # Compute similarities to centroid
    sims = cosine_similarity(c, X_norm[cluster_indices])[0]
    
    # Get top k
    order = np.argsort(-sims)
    return cluster_indices[order[:k]].tolist()

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
            {"role": "user", "content": user_prompt.strip()},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)

# -------------------------
# Formatting utilities
# -------------------------
def truncate_text(s: str, limit: int) -> str:
    """Truncate text to limit characters"""
    s = s.replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[:limit].rsplit(" ", 1)[0] + "..."

def build_prompt(centroid_note: str, abstracts: List[str], n_papers: int) -> str:
    """Build prompt for LLM evaluation"""
    lines = []
    for i, abs_txt in enumerate(abstracts, 1):
        if abs_txt:  # Only include if abstract exists
            lines.append(f"[{i}] {truncate_text(abs_txt, MAX_ABSTRACT_CHARS)}")
    
    if not lines:
        lines.append("[No abstracts available for these papers]")
    
    abstract_block = "\n\n".join(lines)
    
    return USER_PROMPT_TEMPLATE.format(
        centroid_note=centroid_note.strip(),
        n_papers=n_papers,
        abstract_block=abstract_block
    )

# -------------------------
# Main evaluation
# -------------------------
def evaluate_clusters_with_llm(
    X: np.ndarray,
    labels: np.ndarray,
    abstracts: List[str],
    paper_ids: List[str],
    neighbours_per_cluster: int = 5,
    max_clusters: Optional[int] = None,
    min_cluster_size: int = 3,
    centroid_note_text: str = "Centroid is computed in bag-of-papers embedding space; members shown are the nearest in that space."
) -> pd.DataFrame:
    """Evaluate paper clusters using LLM"""
    
    assert X.shape[0] == len(labels) == len(abstracts) == len(paper_ids), \
        "Embeddings, labels, abstracts, and paper_ids must be aligned."
    
    # Normalize embeddings for cosine similarity
    X_norm = normalize(X)
    
    # Get unique clusters (excluding noise)
    uniq = np.unique(labels)
    uniq = [c for c in uniq if c >= 0]  # Only valid clusters
    
    if max_clusters is not None:
        uniq = uniq[:max_clusters]
    
    print(f"\nEvaluating {len(uniq)} clusters with LLM...")
    
    rows = []
    for cluster_id in uniq:
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if cluster_indices.size < min_cluster_size:
            print(f"Skipping cluster {cluster_id} (size {cluster_indices.size} < {min_cluster_size})")
            continue
        
        print(f"Evaluating cluster {cluster_id} (size: {cluster_indices.size})...")
        
        # Get representative papers near centroid
        picked = nearest_in_cluster(X_norm, cluster_indices, k=neighbours_per_cluster)
        
        # Get abstracts and paper IDs for picked papers
        abstracts_subset = [abstracts[i] for i in picked]
        paper_ids_subset = [paper_ids[i] for i in picked]
        
        # Count how many abstracts we actually have
        n_with_abstracts = sum(1 for a in abstracts_subset if a)
        
        if n_with_abstracts == 0:
            print(f"  Warning: No abstracts available for cluster {cluster_id}")
            rows.append({
                "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_indices.size),
                "picked_paper_ids": json.dumps(paper_ids_subset),
                "n_abstracts_available": 0,
                "score": 0,
                "label": "No abstracts",
                "rationale": "Cannot evaluate - no abstracts available for sampled papers"
            })
            continue
        
        # Build prompt
        prompt = build_prompt(centroid_note_text, abstracts_subset, len(picked))
        
        try:
            result = call_llm(SYSTEM_PROMPT, prompt)
            score = int(result.get("score", 0))
            label = result.get("label", "").strip()
            rationale = result.get("rationale", "").strip()
            print(f"  Score: {score}, Label: {label}")
        except Exception as e:
            print(f"  Error calling LLM: {e}")
            score, label, rationale = 0, "LLM_error", f"{type(e).__name__}: {e}"
        
        rows.append({
            "cluster_id": int(cluster_id),
            "cluster_size": int(cluster_indices.size),
            "picked_paper_ids": json.dumps(paper_ids_subset),
            "n_abstracts_available": n_with_abstracts,
            "score": score,
            "label": label,
            "rationale": rationale
        })
        
        # Polite pacing to avoid hitting rate limits
        time.sleep(0.5)
    
    df = pd.DataFrame(rows).sort_values(["score", "cluster_size"], ascending=[False, False])
    return df

# -------------------------
# Main function
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate paper clusters using LLM")
    
    parser.add_argument("--clusters", required=True, type=Path,
                       help="Path to CSV file with paper_id and hdbscan_label columns")
    parser.add_argument("--vectors", required=True, type=Path,
                       help="Path to CSV file with paper_id and bag-of-papers vector columns")
    parser.add_argument("--abstracts", required=True, type=Path,
                       help="Path to abstracts.jsonl file")
    parser.add_argument("--output_dir", type=Path, default=Path("."),
                       help="Directory for output files (default: current directory)")
    parser.add_argument("--neighbours", type=int, default=5,
                       help="Number of representative papers per cluster (default: 5)")
    parser.add_argument("--max_clusters", type=int, default=None,
                       help="Maximum number of clusters to evaluate (default: all)")
    parser.add_argument("--min_cluster_size", type=int, default=3,
                       help="Minimum cluster size to evaluate (default: 3)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("="*60)
    print("PAPER CLUSTER LLM EVALUATION")
    print("="*60)
    
    # Load cluster assignments and vectors
    cluster_df = load_cluster_assignments(args.clusters)
    vector_df = load_paper_vectors(args.vectors)
    
    # Merge cluster assignments with vectors
    merged_df, vector_cols = merge_clusters_and_vectors(cluster_df, vector_df)
    
    # Extract embeddings and labels
    X, labels, paper_ids = extract_embeddings_and_labels(merged_df, vector_cols)
    
    # Load abstracts
    abstracts_by_pid = load_abstracts_dict(args.abstracts)
    
    # Align abstracts with paper order
    abstracts = align_abstracts_with_papers(paper_ids, abstracts_by_pid)
    
    # Evaluate clusters
    eval_df = evaluate_clusters_with_llm(
        X=X,
        labels=labels,
        abstracts=abstracts,
        paper_ids=paper_ids,
        neighbours_per_cluster=args.neighbours,
        max_clusters=args.max_clusters,
        min_cluster_size=args.min_cluster_size,
    )
    
    # Save results
    output_csv = args.output_dir / OUTPUT_CSV
    output_jsonl = args.output_dir / OUTPUT_JSONL
    
    print(f"\nSaving results...")
    eval_df.to_csv(output_csv, index=False)
    print(f"  CSV saved to: {output_csv}")
    
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in eval_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"  JSONL saved to: {output_jsonl}")
    
    # Print summary
    if not eval_df.empty:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Clusters evaluated: {len(eval_df)}")
        print(f"Average score: {eval_df['score'].mean():.2f}")
        print(f"Median score: {eval_df['score'].median():.1f}")
        
        print("\nScore distribution:")
        print(eval_df["score"].value_counts().sort_index())
        
        print("\nTop 5 clusters by LLM score:")
        print(eval_df.head(5)[['cluster_id', 'cluster_size', 'score', 'label']].to_string(index=False))
        
        print("\nBottom 5 clusters by LLM score:")
        print(eval_df.tail(5)[['cluster_id', 'cluster_size', 'score', 'label']].to_string(index=False))
    else:
        print("No clusters evaluated. Check filters or inputs.")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()