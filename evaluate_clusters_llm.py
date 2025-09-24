#!/usr/bin/env python3
import os
import json
import time
import backoff
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# If FAISS install is problematic, set USE_FAISS = False to use sklearn NearestNeighbors
USE_FAISS = True
if USE_FAISS:
    import faiss
else:
    from sklearn.neighbors import NearestNeighbors

# ---- OpenAI setup ----
# pip install openai==1.*  (modern SDK)
from openai import OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # cheap, strong reasoning
# Expect OPENAI_API_KEY in env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
client = OpenAI(api_key=api_key)

# -------------------------
# Config
# -------------------------
EMBEDDINGS_PATH        = "data/embeddings_facebook_v1/sentence-transformers*all-MiniLM-L6-v2/embeddings.npy"        # shape (N, D)
LABELS_PATH            = "skmeans_out/labels_B_best_k40.npy"            # shape (N,)
ABSTRACTS_JSONL        = "data/abstracts/abstracts.jsonl"       # contains {"paper_id", "abstract"}
INDEX_TO_PAPER_JSON    = "data/embeddings_facebook_v1/sentence-transformers*all-MiniLM-L6-v2/paperid_to_idx.json"  # maps row index -> paper_id (dict or list)

OUTPUT_CSV             = "cluster_llm_evaluation.csv"
OUTPUT_JSONL           = "cluster_llm_evaluation.jsonl"

# Retrieval
NEIGHBOURS_PER_CLUSTER = 5      # number of neighbours to show near the centroid
MAX_CLUSTERS           = 300      # set an int to cap clusters for a quick pass
MIN_CLUSTER_SIZE       = 1      # skip tiny clusters

# Prompt control
MAX_ABSTRACT_CHARS = 1200      # per abstract to stay well within token limits
SYSTEM_PROMPT = """You are a careful scientific methods reviewer. 
Given a set of astrophysics papers, judge whether their methodologies are close to each other. 
Be strict about methodology, not just topic. Short, precise answers."""
USER_PROMPT_TEMPLATE = """You are given abstracts from papers that were clustered together by a vector method that is supposed to represent METHODOLOGICAL similarity.

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

Abstracts (truncated) of the 5 nearest members in this cluster:
{abstract_block}

Respond in strict JSON with keys: score, label, rationale.
"""

# -------------------------
# Data loading
# -------------------------
def load_embeddings(path: str) -> np.ndarray:
    X = np.load(path)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    return X

def load_labels(path: str) -> np.ndarray:
    lab = np.load(path)
    return lab

def load_pid_to_index(path: str) -> Dict[str, int]:
    """
    Expects a JSON dict mapping: { "<paper_id>": <row_index_int>, ... }
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Mapping must be a JSON object of {paper_id: index}.")
    pid_to_index = {}
    for k, v in obj.items():
        try:
            pid_to_index[str(k)] = int(v)
        except Exception:
            raise ValueError(f"Bad mapping entry: {k} -> {v} (index must be int)")
    return pid_to_index


def load_abstracts_dict(jsonl_path: str) -> Dict[str, str]:
    """
    Returns dict: paper_id -> abstract
    """
    d = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = str(obj.get("paper_id", "")).strip()
            if pid:
                d[pid] = obj.get("abstract", "") or ""
    return d


def align_abstracts_by_pid_to_index(
    pid_to_index: Dict[str, int],
    abstracts_by_pid: Dict[str, str],
    n_rows: int
) -> List[str]:
    """
    Builds a list of length n_rows such that abstracts[idx] corresponds to
    the embedding/label row at 'idx'.
    """
    abstracts = [""] * n_rows
    placed = 0
    skipped_missing_map = 0
    skipped_oob = 0
    dup_overwrite = 0

    for pid, abs_txt in abstracts_by_pid.items():
        if pid not in pid_to_index:
            skipped_missing_map += 1
            continue
        idx = pid_to_index[pid]
        if idx < 0 or idx >= n_rows:
            skipped_oob += 1
            continue
        if abstracts[idx]:
            dup_overwrite += 1  # rare but possible if mapping duplicates
        abstracts[idx] = abs_txt
        placed += 1

    missing_slots = sum(1 for a in abstracts if not a)
    print(
        f"[info] abstracts placed: {placed}, "
        f"missing slots: {missing_slots}, "
        f"no-mapping: {skipped_missing_map}, "
        f"out-of-bounds: {skipped_oob}, "
        f"duplicates-overwritten: {dup_overwrite}"
    )
    return abstracts


# -------------------------
# Retrieval helpers
# -------------------------
def build_index(X_norm: np.ndarray):
    n, d = X_norm.shape
    if USE_FAISS:
        index = faiss.IndexFlatIP(d)  # dot product on L2-normalised equals cosine similarity
        index.add(X_norm)
        return index
    else:
        nn = NearestNeighbors(n_neighbors=NEIGHBOURS_PER_CLUSTER + 10, metric="cosine", algorithm="auto")
        nn.fit(X_norm)
        return nn

def cluster_centroid(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    return X[idxs].mean(axis=0, keepdims=True)

def nearest_in_cluster(
    X_norm: np.ndarray,
    index,
    cluster_indices: np.ndarray,
    k: int = 5
) -> List[int]:
    # centroid in normalised space
    c = normalize(cluster_centroid(X_norm, cluster_indices))
    if USE_FAISS:
        # Retrieve many neighbours globally, then filter to cluster
        sims, inds = index.search(c, k + 50)  # overshoot then filter
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
    s = s.replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[:limit].rsplit(" ", 1)[0] + "..."

def build_prompt(centroid_note: str, abstracts: List[str]) -> str:
    lines = []
    for i, abs_txt in enumerate(abstracts, 1):
        lines.append(f"[{i}] {truncate_text(abs_txt, MAX_ABSTRACT_CHARS)}")
    abstract_block = "\n\n".join(lines)
    return USER_PROMPT_TEMPLATE.format(
        centroid_note=centroid_note.strip(),
        abstract_block=abstract_block
    )

# -------------------------
# Main evaluation
# -------------------------
def evaluate_clusters_with_llm(
    X: np.ndarray,
    labels: np.ndarray,
    abstracts: List[str],
    neighbours_per_cluster: int = 5,
    max_clusters: Optional[int] = None,
    min_cluster_size: int = 10,
    centroid_note_text: str = "Centroid is computed in embedding space; members shown are the nearest in that space."
) -> pd.DataFrame:

    assert X.shape[0] == len(labels) == len(abstracts), "Embeddings, labels, and abstracts must be aligned."

    # normalise embeddings for cosine retrieval
    X_norm = normalize(X.astype(np.float32))
    index = build_index(X_norm)

    # unique clusters excluding noise label if present
    uniq = np.unique(labels)
    uniq = [c for c in uniq if c != -1]
    if max_clusters is not None:
        uniq = uniq[:max_clusters]

    rows = []
    for c in uniq:
        cluster_indices = np.where(labels == c)[0]
        if cluster_indices.size < min_cluster_size:
            continue

        # representative docs near centroid
        picked = nearest_in_cluster(X_norm, index, cluster_indices, k=neighbours_per_cluster)
        abstracts_subset = [abstracts[i] for i in picked]

        prompt = build_prompt(centroid_note_text, abstracts_subset)

        try:
            result = call_llm(SYSTEM_PROMPT, prompt)
            score = int(result.get("score", 0))
            label = result.get("label", "").strip()
            rationale = result.get("rationale", "").strip()
        except Exception as e:
            score, label, rationale = 0, "LLM_error", f"{type(e).__name__}: {e}"

        rows.append({
            "cluster_id": int(c),
            "cluster_size": int(cluster_indices.size),
            "picked_indices": json.dumps(picked),
            "score": score,
            "label": label,
            "rationale": rationale
        })

        # polite pacing to avoid hitting rate limits too fast
        time.sleep(0.3)

    df = pd.DataFrame(rows).sort_values(["score", "cluster_size"], ascending=[False, False])
    return df

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("Loading data...")
    X = load_embeddings(EMBEDDINGS_PATH)
    labels = load_labels(LABELS_PATH)

    pid_to_index     = load_pid_to_index(INDEX_TO_PAPER_JSON)  # paper_id -> index
    abstracts_by_pid = load_abstracts_dict(ABSTRACTS_JSONL)
    abstracts        = align_abstracts_by_pid_to_index(pid_to_index, abstracts_by_pid, n_rows=X.shape[0])

    if len(labels) != X.shape[0]:
        raise ValueError(f"labels length {len(labels)} != embeddings rows {X.shape[0]}")
    if len(abstracts) != X.shape[0]:
        raise ValueError(f"abstracts length {len(abstracts)} != embeddings rows {X.shape[0]}")

    print("Evaluating clusters with LLM...")
    df = evaluate_clusters_with_llm(
        X=X,
        labels=labels,
        abstracts=abstracts,
        neighbours_per_cluster=NEIGHBOURS_PER_CLUSTER,
        max_clusters=MAX_CLUSTERS,
        min_cluster_size=MIN_CLUSTER_SIZE,
    )

    print(f"Writing {OUTPUT_CSV} and {OUTPUT_JSONL}")
    df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    # Print summary
    if not df.empty:
        print("\nTop 10 clusters by LLM score:")
        print(df.head(10).to_string(index=False))

        print("\nScore distribution:")
        print(df["score"].value_counts().sort_index())
    else:
        print("No clusters evaluated. Check filters or inputs.")
