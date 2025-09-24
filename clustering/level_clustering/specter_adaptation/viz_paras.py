#!/usr/bin/env python3
"""
Visualize paragraph clusters + compute clustering effectiveness metrics.
Works with CSV files containing embeddings and cluster assignments.

Inputs
------
CSV files with pattern: paragraph_clusters_part_XXXX.csv
Expected columns: para_id, paper_id, cluster, and embedding columns
Located in a single directory.

What it does
------------
- Loads embeddings and cluster assignments from CSV files
- Computes internal metrics (on non-noise points):
    * Silhouette score (cosine)
    * Calinski–Harabasz
    * Davies–Bouldin
- Computes cluster-level stats:
    * size
    * mean intra-cluster cosine (sampled pairs)
    * mean cosine to centroid
    * nearest-other-centroid cosine
    * centroid margin = (mean_to_centroid - nearest_other_centroid)
- Visualizations:
    * 2D scatter (UMAP if available, else IPCA) colored by cluster
    * Heatmap of centroid cosine similarity (top-K largest clusters)

Outputs
-------
- metrics_global.json              (silhouette / CH / DB + counts)
- cluster_quality.csv              (per-cluster stats)
- cluster_sizes.csv                (size table)
- scatter_2d.png                   (2D colored scatter on sample)
- centroid_sim_heatmap.png         (top-K centroid similarity heatmap)
- data_summary.json                (loaded data statistics)

Usage
-----
python viz_clusters_csv.py \
  --csv_dir /path/to/csv/folder \
  --out_dir ./viz_metrics_out \
  --plot_sample 60000 --metric_sample 40000 --pairs_per_cluster 2000 \
  --topk_centroids 40 --noise_label -1
"""

import os
import re
import gc
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

# Optional UMAP
try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False
    print("Warning: umap-learn not available, will use IPCA for 2D projection")

# Optional matplotlib
try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

rng = np.random.default_rng(42)


# ----------------------- Load data from CSV files -----------------------

def scan_csv_files(csv_dir: Path, pattern: str = r"paragraph_clusters_part_\d{4}\.csv") -> List[Path]:
    """Scan for CSV files matching pattern"""
    rx = re.compile(pattern)
    files = [p for p in sorted(csv_dir.iterdir()) if p.is_file() and rx.fullmatch(p.name)]
    if not files:
        raise FileNotFoundError(f"No CSV files matching /{pattern}/ under {csv_dir}")
    return files


def detect_embedding_columns(df: pd.DataFrame) -> List[str]:
    """
    Automatically detect embedding columns.
    Assumes embedding columns are numeric and not in ['para_id', 'paper_id', 'cluster']
    """
    reserved_cols = {'para_id', 'paper_id', 'cluster'}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    embedding_cols = [col for col in numeric_cols if col not in reserved_cols]
    
    # If we find columns that look like embedding dimensions (e.g., dim_0, dim_1, etc.)
    if any(col.startswith('dim_') for col in embedding_cols):
        embedding_cols = [col for col in embedding_cols if col.startswith('dim_')]
    
    return sorted(embedding_cols)


def load_csv_data(csv_files: List[Path], 
                  sample_every: int = 1,
                  embedding_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict]:
    """
    Load embeddings and labels from CSV files.
    Returns: (X matrix, y labels, full dataframe, summary stats)
    """
    print(f"[load] Loading data from {len(csv_files)} CSV files...")
    
    dfs = []
    
    # Load and combine all CSV files
    for csv_path in tqdm(csv_files, desc="Loading CSV files", unit="file"):
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Error loading {csv_path}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No CSV files could be loaded successfully")
    
    # Combine all dataframes
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[load] Combined {len(dfs)} files into {len(full_df):,} rows")
    
    # Auto-detect embedding columns if not provided
    if embedding_cols is None:
        embedding_cols = detect_embedding_columns(full_df)
        print(f"[load] Auto-detected {len(embedding_cols)} embedding columns")
    
    if not embedding_cols:
        raise ValueError("No embedding columns found. Please check your data format.")
    
    # Validate required columns
    required_cols = ['para_id', 'cluster']
    missing_cols = [col for col in required_cols if col not in full_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sample data if requested
    if sample_every > 1:
        full_df = full_df.iloc[::sample_every].reset_index(drop=True)
        print(f"[load] Sampled every {sample_every}th row, kept {len(full_df):,} rows")
    
    # Extract embeddings and labels
    print("[load] Extracting embeddings and labels...")
    
    # Handle missing values in embedding columns
    embedding_data = full_df[embedding_cols].fillna(0.0)
    X = embedding_data.values.astype(np.float32)
    
    # Extract labels
    y = full_df['cluster'].values.astype(np.int32)
    
    # Create summary
    summary = {
        "total_files": len(csv_files),
        "files_loaded": len(dfs),
        "total_rows": int(len(full_df)),
        "embedding_dim": int(len(embedding_cols)),
        "unique_paragraphs": int(full_df['para_id'].nunique()),
        "unique_papers": int(full_df['paper_id'].nunique()) if 'paper_id' in full_df.columns else 0,
        "unique_clusters": int(full_df['cluster'].nunique()),
        "cluster_range": [int(full_df['cluster'].min()), int(full_df['cluster'].max())],
        "embedding_columns": embedding_cols[:10] + (['...'] if len(embedding_cols) > 10 else [])  # Show first 10
    }
    
    print(f"[load] Final data shape: {X.shape}")
    print(f"[load] Embedding dimension: {X.shape[1]}")
    print(f"[load] Unique clusters: {len(np.unique(y))}")
    
    return X, y, full_df, summary


# ----------------------- Utilities (unchanged from original) -----------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2 normalize rows of X"""
    return normalize(X, norm='l2', axis=1, copy=False)


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity (assumes rows are L2-normalized)"""
    return A @ B.T


def sample_indices_by_label(labels: np.ndarray, n_target: int, 
                           noise_label: int = -1, seed: int = 42) -> np.ndarray:
    """Stratified sample proportional to cluster sizes; drops noise."""
    rng = np.random.default_rng(seed)
    
    mask = labels != noise_label
    y = labels[mask]
    idxs = np.nonzero(mask)[0]
    n = len(y)
    
    if n <= n_target:
        return idxs
    
    # Proportional sampling by cluster
    unique_labels, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    per_cluster = (probs * n_target).astype(int)
    
    # Ensure at least 1 per cluster
    per_cluster = np.maximum(per_cluster, 1)
    
    # Adjust to sum exactly to n_target
    diff = n_target - per_cluster.sum()
    if diff > 0:
        # Add to largest clusters
        for _ in range(diff):
            idx = np.argmax(counts)
            per_cluster[idx] += 1
    elif diff < 0:
        # Remove from smallest clusters (but keep at least 1)
        for _ in range(-diff):
            candidates = np.where(per_cluster > 1)[0]
            if len(candidates) > 0:
                idx = candidates[np.argmin(counts[candidates])]
                per_cluster[idx] -= 1
    
    # Sample from each cluster
    chosen = []
    for label, n_samples in zip(unique_labels, per_cluster):
        cluster_idxs = idxs[y == label]
        if len(cluster_idxs) <= n_samples:
            chosen.append(cluster_idxs)
        else:
            chosen.append(rng.choice(cluster_idxs, size=n_samples, replace=False))
    
    return np.concatenate(chosen)


def sample_pairs_within(idx: np.ndarray, cap_pairs: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return two arrays (i, j) of indices to form pairs inside idx"""
    rng = np.random.default_rng(seed)
    
    m = len(idx)
    if m < 2 or cap_pairs <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Number of unique unordered pairs C(m,2)
    total_pairs = m * (m - 1) // 2
    n_pairs = min(cap_pairs, total_pairs, 2000000)  # Cap at 2M to avoid memory issues
    
    # Sample pairs
    pairs = set()
    attempts = 0
    max_attempts = n_pairs * 10
    
    while len(pairs) < n_pairs and attempts < max_attempts:
        i = rng.integers(0, m)
        j = rng.integers(0, m)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
        attempts += 1
    
    if len(pairs) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    pairs = list(pairs)
    i_sel = np.array([p[0] for p in pairs])
    j_sel = np.array([p[1] for p in pairs])
    
    # Map to global indices
    return idx[i_sel], idx[j_sel]


def ipca_2d(X: np.ndarray, batch: int = 10000) -> np.ndarray:
    """Reduce to 2D using IncrementalPCA"""
    print("[reduce] Computing 2D projection with IPCA...")
    
    n_components = min(2, min(X.shape))
    ip = IncrementalPCA(n_components=n_components, batch_size=batch)
    
    # Fit in batches
    for s in tqdm(range(0, X.shape[0], batch), desc="IPCA fit"):
        e = min(X.shape[0], s + batch)
        ip.partial_fit(X[s:e])
    
    # Transform in batches
    Z = []
    for s in tqdm(range(0, X.shape[0], batch), desc="IPCA transform"):
        e = min(X.shape[0], s + batch)
        Z.append(ip.transform(X[s:e]))
    
    return np.vstack(Z).astype(np.float32)


def umap_2d(X: np.ndarray, n_neighbors: int = 200, min_dist: float = 0.05) -> np.ndarray:
    """Reduce to 2D using UMAP"""
    print(f"[reduce] Computing 2D projection with UMAP (n_neighbors={n_neighbors})...")
    
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        verbose=True
    )
    return reducer.fit_transform(X)


# ----------------------- Metrics + Analysis (unchanged from original) -----------------------

def compute_internal_metrics(X: np.ndarray, y: np.ndarray, noise_label: int = -1) -> Dict:
    """Compute silhouette, CH, DB on non-noise sample."""
    mask = y != noise_label
    Xn, yn = X[mask], y[mask]
    res = {}
    
    unique_clusters = np.unique(yn)
    n_clusters = len(unique_clusters)
    n_samples = len(yn)
    
    res["n_samples_used"] = int(n_samples)
    res["n_clusters_used"] = int(n_clusters)
    
    # Need at least 2 clusters and sufficient samples
    if n_clusters >= 2 and n_samples >= 20:
        try:
            print("[metrics] Computing silhouette score...")
            res["silhouette_cosine"] = float(silhouette_score(Xn, yn, metric="cosine", sample_size=min(10000, n_samples)))
        except Exception as e:
            res["silhouette_cosine"] = None
            res["silhouette_error"] = str(e)
        
        try:
            print("[metrics] Computing Calinski-Harabasz score...")
            res["calinski_harabasz"] = float(calinski_harabasz_score(Xn, yn))
        except Exception as e:
            res["calinski_harabasz"] = None
            res["calinski_error"] = str(e)
        
        try:
            print("[metrics] Computing Davies-Bouldin score...")
            res["davies_bouldin"] = float(davies_bouldin_score(Xn, yn))
        except Exception as e:
            res["davies_bouldin"] = None
            res["davies_error"] = str(e)
    else:
        res["silhouette_cosine"] = None
        res["calinski_harabasz"] = None
        res["davies_bouldin"] = None
        res["note"] = "Not enough clusters/samples after removing noise."
    
    return res


def per_cluster_quality(X: np.ndarray, y: np.ndarray, noise_label: int = -1,
                       pairs_per_cluster: int = 2000, 
                       topk_for_heatmap: int = 40) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
    """
    Compute per-cluster quality metrics.
    Returns:
      - DataFrame with cluster statistics
      - Centroid cosine similarity matrix for top-K clusters
      - List of cluster IDs included in heatmap
    """
    print("[quality] Computing per-cluster quality metrics...")
    
    # Ensure normalized
    X = l2_normalize(X.copy())
    
    clusters = [c for c in np.unique(y) if c != noise_label]
    
    # Compute sizes
    sizes = {int(c): int((y == c).sum()) for c in clusters}
    
    # Compute centroids
    centroids = {}
    for c in tqdm(clusters, desc="Computing centroids"):
        idx = np.where(y == c)[0]
        cvec = X[idx].mean(axis=0)
        centroids[int(c)] = cvec / (np.linalg.norm(cvec) + 1e-12)
    
    # Get top-K clusters by size
    topK = sorted(clusters, key=lambda c: sizes[int(c)], reverse=True)[:topk_for_heatmap]
    
    # Compute centroid similarity matrix
    if topK:
        Cmat = np.vstack([centroids[int(c)] for c in topK])
        centroid_cos = Cmat @ Cmat.T
    else:
        centroid_cos = np.array([[]])
    
    # Compute per-cluster metrics
    rows = []
    for c in tqdm(clusters, desc="Computing cluster metrics"):
        idx = np.where(y == c)[0]
        m = len(idx)
        
        if m < 2:
            mean_pair = np.nan
            mean_to_centroid = np.nan
        else:
            # Sample pairs for intra-cluster similarity
            ii, jj = sample_pairs_within(idx, pairs_per_cluster)
            if len(ii) == 0:
                mean_pair = np.nan
            else:
                mean_pair = float((X[ii] * X[jj]).sum(axis=1).mean())
            
            # Mean cosine to centroid
            cvec = centroids[int(c)]
            mean_to_centroid = float((X[idx] @ cvec).mean())
        
        # Nearest other centroid cosine
        cvec = centroids[int(c)]
        others = [centroids[int(o)] for o in clusters if o != c]
        if others:
            sims = np.array([cvec @ other for other in others])
            near_other = float(sims.max())
        else:
            near_other = np.nan
        
        # Compute margin
        if not np.isnan(mean_to_centroid) and not np.isnan(near_other):
            margin = mean_to_centroid - near_other
        else:
            margin = np.nan
        
        rows.append({
            "cluster": int(c),
            "size": sizes[int(c)],
            "mean_intra_cosine_pairs": mean_pair,
            "mean_to_centroid": mean_to_centroid,
            "nearest_other_centroid": near_other,
            "centroid_margin": margin
        })
    
    df = pd.DataFrame(rows).sort_values("size", ascending=False, ignore_index=True)
    return df, centroid_cos, [int(c) for c in topK]


# ----------------------- Plotting (unchanged from original) -----------------------

def plot_scatter_2d(X2: np.ndarray, y: np.ndarray, out_png: Path, 
                   noise_label: int = -1, max_legends: int = 20):
    """Create 2D scatter plot colored by cluster"""
    if not HAVE_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping scatter plot")
        return
    
    print("[plot] Creating 2D scatter plot...")
    
    # Map labels to colors
    unique_labels = sorted(set(y.tolist()))
    colors = {}
    next_color = 0
    
    for label in unique_labels:
        if label == noise_label:
            continue
        colors[int(label)] = next_color
        next_color += 1
    
    # Noise gets gray color
    if noise_label in unique_labels:
        colors[noise_label] = -1
    
    # Create color array
    c_array = []
    for label in y:
        if label == noise_label:
            c_array.append('gray')
        else:
            c_array.append(colors.get(int(label), 0))
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot non-noise points
    mask = y != noise_label
    if mask.any():
        scatter = plt.scatter(
            X2[mask, 0], X2[mask, 1],
            c=[c for c, m in zip(c_array, mask) if m],
            s=3, alpha=0.6, linewidths=0, cmap='tab20'
        )
    
    # Plot noise points
    if (~mask).any():
        plt.scatter(
            X2[~mask, 0], X2[~mask, 1],
            c='lightgray', s=2, alpha=0.3, linewidths=0, label='noise'
        )
    
    plt.title("2D Projection of Paragraph Embeddings (colored by cluster)", fontsize=14)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    # Add legend if not too many clusters
    if len(unique_labels) <= max_legends:
        if mask.any():
            plt.colorbar(scatter, label="Cluster ID")
        if (~mask).any():
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved scatter plot to {out_png}")


def plot_centroid_heatmap(S: np.ndarray, cluster_ids: List[int], out_png: Path, 
                         noise_label: int = -1):
    """Create heatmap of centroid similarities"""
    if not HAVE_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping heatmap")
        return
    
    if S.size == 0:
        print("Warning: No clusters for heatmap")
        return
    
    print("[plot] Creating centroid similarity heatmap...")
    
    plt.figure(figsize=(10, 9))
    
    im = plt.imshow(S, vmin=-1.0, vmax=1.0, cmap='coolwarm', aspect='auto')
    plt.colorbar(im, label='Cosine Similarity')
    
    # Create labels
    labels = [("noise" if c == noise_label else f"C{c}") for c in cluster_ids]
    
    # Set ticks
    ticks = np.arange(len(cluster_ids))
    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=ticks, labels=labels)
    
    plt.title(f"Centroid Cosine Similarity (Top-{len(cluster_ids)} Clusters)", fontsize=14)
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved heatmap to {out_png}")


# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize and analyze paragraph clusters from CSV files")
    
    # Required arguments
    ap.add_argument("--csv_dir", required=True, help="Directory containing CSV files")
    ap.add_argument("--out_dir", required=True, help="Output directory for results")
    
    # Data loading options
    ap.add_argument("--csv_pattern", default=r"paragraph_clusters_part_\d{4}\.csv", 
                    help="Regex pattern for CSV files")
    ap.add_argument("--max_files", type=int, default=0, help="Limit to first N files (0=all)")
    ap.add_argument("--sample_every", type=int, default=1, help="Use every k-th row (for memory)")
    ap.add_argument("--embedding_cols", nargs='*', help="Explicit list of embedding column names (auto-detect if not provided)")
    
    # Sampling parameters
    ap.add_argument("--noise_label", type=int, default=-1, help="Label value for noise points")
    ap.add_argument("--metric_sample", type=int, default=40000, help="Max points for metric computation")
    ap.add_argument("--plot_sample", type=int, default=60000, help="Max points for 2D scatter")
    ap.add_argument("--pairs_per_cluster", type=int, default=2000, help="Pair samples per cluster")
    ap.add_argument("--topk_centroids", type=int, default=40, help="Top-K clusters for heatmap")
    
    # Visualization options
    ap.add_argument("--use_umap", action="store_true", help="Use UMAP for 2D (else IPCA)")
    ap.add_argument("--umap_neighbors", type=int, default=200, help="UMAP n_neighbors parameter")
    ap.add_argument("--umap_min_dist", type=float, default=0.05, help="UMAP min_dist parameter")
    
    args = ap.parse_args()
    
    # Create output directory
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CLUSTER VISUALIZATION AND METRICS (CSV Version)")
    print("=" * 60)
    
    # 1) Find CSV files
    print("\n[1/8] Scanning for CSV files...")
    csv_files = scan_csv_files(Path(args.csv_dir), args.csv_pattern)
    if args.max_files > 0:
        csv_files = csv_files[:args.max_files]
    print(f"  Found {len(csv_files)} CSV files")
    
    # 2) Load data from CSV files
    print("\n[2/8] Loading data from CSV files...")
    X_all, y_all, full_df, data_summary = load_csv_data(
        csv_files, 
        sample_every=args.sample_every,
        embedding_cols=args.embedding_cols
    )
    
    # Save data summary
    (out / "data_summary.json").write_text(json.dumps(data_summary, indent=2))
    print(f"  Data summary: {json.dumps(data_summary, indent=2)}")
    
    # 3) Save cluster size statistics
    print("\n[3/8] Computing cluster size statistics...")
    sizes = full_df["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
    sizes.sort_values("count", ascending=False, inplace=True, ignore_index=True)
    sizes.to_csv(out / "cluster_sizes.csv", index=False)
    print(f"  Saved cluster sizes to {out / 'cluster_sizes.csv'}")
    print(f"  Clusters: {len(sizes)}, Largest: {sizes['count'].iloc[0]:,}, Smallest: {sizes['count'].iloc[-1]:,}")
    
    # 4) Sample for metrics and normalize
    print("\n[4/8] Preparing sample for metrics computation...")
    idx_metric = sample_indices_by_label(y_all, n_target=args.metric_sample, noise_label=args.noise_label)
    X_metric = l2_normalize(X_all[idx_metric].copy())
    y_metric = y_all[idx_metric]
    print(f"  Sampled {len(idx_metric):,} points for metrics")
    
    # 5) Compute global metrics
    print("\n[5/8] Computing internal clustering metrics...")
    metrics = compute_internal_metrics(X_metric, y_metric, noise_label=args.noise_label)
    metrics.update({
        "total_points": int(len(y_all)),
        "points_for_metrics": int(len(y_metric)),
        "clusters_total": int(full_df["cluster"].nunique()),
        "embedding_dimension": int(X_all.shape[1])
    })
    
    (out / "metrics_global.json").write_text(json.dumps(metrics, indent=2))
    print("  Global metrics:")
    for key, value in metrics.items():
        if value is not None and not isinstance(value, str):
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    
    # 6) Compute per-cluster quality
    print("\n[6/8] Computing per-cluster quality metrics...")
    idx_quality = sample_indices_by_label(
        y_all, 
        n_target=min(len(y_all), max(args.metric_sample, 80000)),
        noise_label=args.noise_label
    )
    X_quality = l2_normalize(X_all[idx_quality].copy())
    y_quality = y_all[idx_quality]
    
    cq_df, C_sim, top_ids = per_cluster_quality(
        X_quality, y_quality,
        noise_label=args.noise_label,
        pairs_per_cluster=args.pairs_per_cluster,
        topk_for_heatmap=args.topk_centroids
    )
    cq_df.to_csv(out / "cluster_quality.csv", index=False)
    print(f"  Saved cluster quality metrics to {out / 'cluster_quality.csv'}")
    
    # Print top clusters by quality
    if len(cq_df) > 0:
        print("  Top 5 clusters by centroid margin:")
        top_margin = cq_df.nlargest(5, 'centroid_margin', keep='first')
        for _, row in top_margin.iterrows():
            print(f"    Cluster {row['cluster']}: size={row['size']:,}, margin={row['centroid_margin']:.3f}")
    
    # 7) Create 2D visualization
    print("\n[7/8] Creating 2D visualization...")
    idx_plot = sample_indices_by_label(y_all, n_target=args.plot_sample, noise_label=args.noise_label)
    X_plot = l2_normalize(X_all[idx_plot].copy())
    y_plot = y_all[idx_plot]
    print(f"  Sampled {len(idx_plot):,} points for visualization")
    
    # Choose dimensionality reduction method
    if args.use_umap and HAVE_UMAP:
        Z = umap_2d(X_plot, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
    else:
        if args.use_umap and not HAVE_UMAP:
            print("  Warning: UMAP requested but not available, using IPCA instead")
        Z = ipca_2d(X_plot)
    
    # Create scatter plot
    plot_scatter_2d(Z, y_plot, out / "scatter_2d.png", noise_label=args.noise_label)
    
    # 8) Create centroid heatmap
    print("\n[8/8] Creating centroid similarity heatmap...")
    plot_centroid_heatmap(C_sim, top_ids, out / "centroid_sim_heatmap.png", noise_label=args.noise_label)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {out.resolve()}")
    print("  - data_summary.json        : Data loading statistics")
    print("  - cluster_sizes.csv        : Cluster size distribution")
    print("  - cluster_quality.csv      : Per-cluster quality metrics")
    print("  - metrics_global.json      : Global clustering metrics")
    if HAVE_MATPLOTLIB:
        print("  - scatter_2d.png           : 2D visualization")
        print("  - centroid_sim_heatmap.png : Centroid similarity matrix")
    
    print("\nKey metrics summary:")
    if "silhouette_cosine" in metrics and metrics["silhouette_cosine"] is not None:
        print(f"  Silhouette Score: {metrics['silhouette_cosine']:.4f}")
    if "calinski_harabasz" in metrics and metrics["calinski_harabasz"] is not None:
        print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
    if "davies_bouldin" in metrics and metrics["davies_bouldin"] is not None:
        print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
    
    print(f"\n✓ Visualization and metrics computation completed successfully!")


if __name__ == "__main__":
    main()