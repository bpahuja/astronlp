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
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
# Input paths
INPUT_CSV              = "/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/cluster_out_v7_astrobert_1/labels_paragraph.csv"  # CSV with columns: para_id, paper_id, cluster
TEXT_DATA_CSV          = None  # Optional: CSV with para_id, text columns. Set to None if no text available

# Output paths
OUTPUT_CSV             = "llm_evaluation_results_para_v7.csv"
OUTPUT_JSONL           = "llm_evaluation_results_para_v7.jsonl"

# Evaluation settings
SAMPLES_PER_CLUSTER    = 5      # number of samples to show from each cluster
MAX_CLUSTERS           = 600    # set an int to cap clusters for a quick pass
MIN_CLUSTER_SIZE       = 1     # skip small clusters

# Prompt control
MAX_TEXT_CHARS = 1200           # per paragraph to stay within token limits
SYSTEM_PROMPT = """You are a careful scientific methods reviewer. 
Given a set of astrophysics paper paragraphs (or their identifiers), judge whether they likely represent similar methodologies. 
Be strict about methodology, not just topic. Short, precise answers."""

USER_PROMPT_TEMPLATE = """You are given paragraphs from papers that were clustered together by a method that is supposed to represent METHODOLOGICAL similarity.

Task:
1) Judge how methodologically close these items are likely to be overall on a 1–5 scale:
   1 = not close, 3 = somewhat close, 5 = very close.
2) Provide a short method-family label (2–6 words).
3) Give a brief rationale in 2–3 sentences focused on methods.

Guidance:
- Focus on methodology: data processing pipelines, simulation types, inference frameworks, instruments, or learning algorithms.
- Ignore surface topic overlap if the methods differ.
- If only IDs are provided, base judgment on clustering patterns and paper distribution.

Cluster information:
- Cluster ID: {cluster_id}
- Total size: {cluster_size}
- Papers represented: {num_papers}

Sample items from this cluster:
{content_block}

Respond in strict JSON with keys: score, label, rationale.
"""

# -------------------------
# Data loading
# -------------------------
def load_cluster_data(csv_path: str) -> pd.DataFrame:
    """Load cluster data from CSV with columns: para_id, paper_id, cluster"""
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['para_id', 'paper_id', 'cluster']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    logger.info(f"Found {df['cluster'].nunique()} unique clusters")
    logger.info(f"Found {df['paper_id'].nunique()} unique papers")
    
    return df

def load_text_data(csv_path: str) -> Optional[Dict[str, str]]:
    """Load text data from CSV with columns: para_id, text"""
    if csv_path is None or not Path(csv_path).exists():
        logger.info("No text data provided - will evaluate based on IDs only")
        return None
    
    df = pd.read_csv(csv_path)
    if 'para_id' not in df.columns or 'text' not in df.columns:
        raise ValueError("Text CSV must have columns: para_id, text")
    
    text_dict = dict(zip(df['para_id'], df['text']))
    logger.info(f"Loaded text for {len(text_dict)} paragraphs")
    return text_dict

# -------------------------
# Sampling helpers
# -------------------------
def sample_cluster_items(cluster_df: pd.DataFrame, k: int = 5) -> List[Dict]:
    """Sample k items from a cluster"""
    if len(cluster_df) <= k:
        sampled = cluster_df.copy()
    else:
        sampled = cluster_df.sample(n=k, random_state=42)
    
    items = []
    for _, row in sampled.iterrows():
        items.append({
            'para_id': row['para_id'],
            'paper_id': row['paper_id']
        })
    
    return items

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

def build_content_block(items: List[Dict], text_dict: Optional[Dict[str, str]]) -> str:
    """Build content block for LLM prompt"""
    lines = []
    
    for i, item in enumerate(items, 1):
        para_id = item['para_id']
        paper_id = item['paper_id']
        
        if text_dict and para_id in text_dict:
            # Include actual text if available
            text = truncate_text(text_dict[para_id], MAX_TEXT_CHARS)
            lines.append(f"[{i}] Paper {paper_id}, Para {para_id}: {text}")
        else:
            # Just show IDs if no text available
            lines.append(f"[{i}] Paper {paper_id}, Paragraph {para_id}")
    
    return "\n\n".join(lines)

def build_prompt(cluster_id: int, cluster_size: int, num_papers: int, items: List[Dict], text_dict: Optional[Dict[str, str]]) -> str:
    """Build LLM prompt"""
    content_block = build_content_block(items, text_dict)
    
    return USER_PROMPT_TEMPLATE.format(
        cluster_id=cluster_id,
        cluster_size=cluster_size,
        num_papers=num_papers,
        content_block=content_block
    )

# -------------------------
# Main evaluation
# -------------------------
def evaluate_clusters_with_llm(
    df: pd.DataFrame,
    text_dict: Optional[Dict[str, str]] = None,
    samples_per_cluster: int = 5,
    max_clusters: Optional[int] = None,
    min_cluster_size: int = 10,
) -> pd.DataFrame:
    """Evaluate clusters using LLM"""
    
    # Get unique clusters (excluding noise if using -1)
    clusters = df['cluster'].unique()
    clusters = [c for c in clusters if c >= 0]  # Exclude -1 (noise) if present
    clusters = sorted(clusters)
    
    if max_clusters is not None:
        clusters = clusters[:max_clusters]
    
    rows = []
    
    for cluster_id in clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        
        if len(cluster_df) < min_cluster_size:
            continue
        
        # Get cluster statistics
        cluster_size = len(cluster_df)
        num_papers = cluster_df['paper_id'].nunique()
        
        # Sample items from cluster
        sampled_items = sample_cluster_items(cluster_df, k=samples_per_cluster)
        
        # Build prompt
        prompt = build_prompt(cluster_id, cluster_size, num_papers, sampled_items, text_dict)
        
        try:
            result = call_llm(SYSTEM_PROMPT, prompt)
            score = int(result.get("score", 0))
            label = result.get("label", "").strip()
            rationale = result.get("rationale", "").strip()
        except Exception as e:
            logger.warning(f"LLM error for cluster {cluster_id}: {e}")
            score, label, rationale = 0, "LLM_error", f"{type(e).__name__}: {e}"
        
        # Store results
        rows.append({
            "cluster_id": int(cluster_id),
            "cluster_size": int(cluster_size),
            "num_papers": int(num_papers),
            "papers_per_para": round(cluster_size / num_papers, 2),
            "sampled_para_ids": json.dumps([item['para_id'] for item in sampled_items]),
            "sampled_paper_ids": json.dumps([item['paper_id'] for item in sampled_items]),
            "score": score,
            "label": label,
            "rationale": rationale
        })
        
        logger.info(f"Evaluated cluster {cluster_id} (size={cluster_size}, papers={num_papers}): score={score}, label={label}")
        
        # Rate limiting
        time.sleep(0.3)
    
    df_results = pd.DataFrame(rows).sort_values(["score", "cluster_size"], ascending=[False, False])
    return df_results

# -------------------------
# Main function
# -------------------------
def main():
    logger.info("Starting LLM evaluation of clusters from CSV...")
    
    # Load cluster data
    logger.info(f"Loading cluster data from {INPUT_CSV}...")
    df = load_cluster_data(INPUT_CSV)
    
    # Load text data if available
    text_dict = load_text_data(TEXT_DATA_CSV)
    
    logger.info(f"Evaluating clusters with LLM...")
    logger.info(f"Total paragraphs: {len(df)}")
    logger.info(f"Total clusters: {df['cluster'].nunique()}")
    logger.info(f"Total papers: {df['paper_id'].nunique()}")
    
    # Run evaluation
    results_df = evaluate_clusters_with_llm(
        df=df,
        text_dict=text_dict,
        samples_per_cluster=SAMPLES_PER_CLUSTER,
        max_clusters=MAX_CLUSTERS,
        min_cluster_size=MIN_CLUSTER_SIZE,
    )
    
    # Save results
    logger.info(f"Writing results to {OUTPUT_CSV} and {OUTPUT_JSONL}")
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in results_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    
    # Print summary
    if not results_df.empty:
        print("\n" + "="*60)
        print("TOP 10 CLUSTERS BY LLM SCORE:")
        print("="*60)
        print(results_df.head(10).to_string(index=False))
        
        print("\n" + "="*60)
        print("SCORE DISTRIBUTION:")
        print("="*60)
        score_dist = results_df["score"].value_counts().sort_index()
        for score, count in score_dist.items():
            print(f"Score {score}: {count} clusters")
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS:")
        print("="*60)
        print(f"Total clusters evaluated: {len(results_df)}")
        print(f"Mean score: {results_df['score'].mean():.2f}")
        print(f"Median score: {results_df['score'].median():.1f}")
        print(f"Mean cluster size: {results_df['cluster_size'].mean():.1f}")
        print(f"Mean papers per cluster: {results_df['num_papers'].mean():.1f}")
        print(f"Total paragraphs in evaluated clusters: {results_df['cluster_size'].sum()}")
    else:
        logger.warning("No clusters evaluated. Check filters or inputs.")

if __name__ == "__main__":
    main()