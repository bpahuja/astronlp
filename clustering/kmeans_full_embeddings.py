#!/usr/bin/env python3
"""
Spherical K-means on full-size embeddings (from pickle).

Input schema (per file):
{
  "<paper_id>": {"embeddings": numpy_array, ...},
  ...
}

Outputs (per model):
- labels_{name}_k{K}.pkl           -> { "<paper_id>": <label> }
- labels_{name}_best_k{K}.pkl      -> { "<paper_id>": <label> }
- cluster_sizes_{name}_k{K}.csv    -> cluster size stats for best K
- summary_spherical_kmeans.csv     -> summary across models

Usage examples:
  python spherical_kmeans_full.py \
      --emb A=/path/to/embA.pkl B=/path/to/embB.pkl C=/path/to/embC.pkl \
      --k 25 --outdir results/

  python spherical_kmeans_full.py \
      --emb A=/path/to/embA.pkl B=/path/to/embB.pkl \
      --k 15 20 25 30 40 --sil-sample 60000 --outdir results/ --viz
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# Optional UMAP import only when requested
def try_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spherical K-means on full-dimension embeddings from pickle.")
    p.add_argument(
        "--emb",
        nargs="+",
        required=True,
        help="One or more name=path items, e.g., A=/path/embA.pkl B=/path/embB.pkl",
    )
    p.add_argument(
        "--k",
        nargs="+",
        type=int,
        required=True,
        help="One or more k values. If multiple, the best silhouette is kept.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="MiniBatchKMeans batch size (default: 10000).",
    )
    p.add_argument(
        "--sil-sample",
        type=int,
        default=50000,
        help="Max sample size for silhouette scoring (default: 50000).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="data/skmeans_out",
        help="Output directory for labels and summary CSV.",
    )
    p.add_argument(
        "--viz",
        action="store_true",
        help="If set, compute a 2D UMAP visualisation and save PNGs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return p.parse_args()


def parse_emb_arg(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--emb item must be name=path, got: {it}")
        name, path = it.split("=", 1)
        out[name] = path
    return out


def load_pickle_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Loads a pickle with schema:
      { "<paper_id>": {"embeddings": np.ndarray, ...}, ... }
    Returns:
      X (np.ndarray, float32) stacked by paper_id order,
      paper_ids (List[str]) in the same order as rows in X.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Deterministic order: sort by paper_id to guarantee stable mapping
    paper_ids = sorted(data.keys())

    # Extract embeddings
    emb_list = []
    for pid in paper_ids:
        emb = data[pid]["embeddings"]
        if not isinstance(emb, np.ndarray):
            raise TypeError(f"Embeddings for paper_id {pid} is not a numpy array.")
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32, copy=False)
        emb_list.append(emb)

    X = np.vstack(emb_list)
    return X, paper_ids


def spherical_kmeans_fit_predict(
    X_norm: np.ndarray, k: int, batch_size: int, seed: int
) -> np.ndarray:
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=seed,
        verbose=0,
    )
    return km.fit_predict(X_norm)


def silhouette_fast(X: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    n = X.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_samples, replace=False)
        return silhouette_score(X[idx], labels[idx])
    return silhouette_score(X, labels)


def plot_umap_2d(X_norm: np.ndarray, labels: np.ndarray, title: str, outpath: Path):
    umap = try_import_umap()
    if umap is None:
        print("UMAP not installed; skipping visualisation.")
        return
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, low_memory=True)
    X2 = reducer.fit_transform(X_norm)
    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=2, linewidths=0, cmap="tab20")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=140)
    plt.close()


def labels_to_pickle(labels: np.ndarray, paper_ids: List[str], outpath: Path) -> None:
    """
    Save labels as a pickle with schema:
      { "<paper_id>": "<label>" }
    Labels are cast to Python int for JSON/pickle friendliness.
    """
    mapping = {pid: int(lbl) for pid, lbl in zip(paper_ids, labels)}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    emb_map = parse_emb_arg(args.emb)
    ks = sorted(set(args.k))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for name, path in emb_map.items():
        print(f"\n=== Model {name} ===")
        print(f"Loading embeddings (pickle): {path}")
        X, paper_ids = load_pickle_embeddings(path)

        print(f"Normalising (L2) in full dimension …")
        X_norm = normalize(X)  # spherical space: cosine == dot

        best = {"k": None, "sil": -np.inf, "labels": None}

        for k in ks:
            print(f"  -> k={k}  (MiniBatchKMeans)")
            labels = spherical_kmeans_fit_predict(X_norm, k, args.batch_size, args.seed)
            sil = silhouette_fast(X_norm, labels, args.sil_sample, args.seed)
            print(f"     silhouette={sil:.4f}")

            # Save this k's labels as a pickle mapping {paper_id: label}
            labels_to_pickle(labels, paper_ids, outdir / f"labels_{name}_k{k}.pkl")

            if np.isnan(sil):
                continue
            if sil > best["sil"]:
                best = {"k": k, "sil": sil, "labels": labels}

        if best["k"] is None:
            print("No valid k produced a silhouette; check inputs.")
            continue

        # Save best labels and optional viz
        labels_to_pickle(best["labels"], paper_ids, outdir / f"labels_{name}_best_k{best['k']}.pkl")
        if args.viz:
            plot_umap_2d(
                X_norm,
                best["labels"],
                f"Spherical K-means (model {name}, k={best['k']})",
                outdir / f"umap_{name}_k{best['k']}.png",
            )

        # Cluster size distribution (quick stats) for the best k
        unique, counts = np.unique(best["labels"], return_counts=True)
        sizes = dict(zip(unique.tolist(), counts.tolist()))
        mean_sz = float(np.mean(counts))
        med_sz = float(np.median(counts))

        summary_rows.append(
            {
                "model": name,
                "best_k": best["k"],
                "silhouette": round(float(best["sil"]), 6),
                "n_clusters": int(len(unique)),
                "mean_cluster_size": round(mean_sz, 2),
                "median_cluster_size": round(med_sz, 2),
                "labels_pkl": str(outdir / f"labels_{name}_best_k{best['k']}.pkl"),
            }
        )

        # Also dump a CSV of cluster sizes for that model
        size_df = pd.DataFrame({"cluster": unique, "size": counts}).sort_values("size", ascending=False)
        size_df.to_csv(outdir / f"cluster_sizes_{name}_k{best['k']}.csv", index=False)

    # Write summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(outdir / "summary_spherical_kmeans.csv", index=False)
        print("\n=== Summary ===")
        print(summary_df.to_string(index=False))
        print(f"\nSaved labels and summaries to: {outdir.resolve()}")
    else:
        print("No results to summarise.")


if __name__ == "__main__":
    main()
