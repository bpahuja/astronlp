#!/usr/bin/env python3
"""
Spherical K-means + HDBSCAN on full-size embeddings (from pickle).

Input schema (per file):
{
  "<paper_id>": {"embeddings": np.ndarray, ...},
  ...
}

Outputs (per model):
KMeans:
- labels_{name}_k{K}.pkl
- labels_{name}_best_k{K}.pkl
- cluster_sizes_{name}_k{K}.csv

HDBSCAN:
- labels_{name}_hdbscan_mcs{M}_ms{S}.pkl
- labels_{name}_hdbscan_best.pkl
- cluster_sizes_{name}_hdbscan.csv

Shared:
- summary_spherical_kmeans.csv (KMeans rows)
- summary_hdbscan.csv (HDBSCAN rows)

Usage examples:
  python spherical_kmeans_full.py \
      --emb A=/path/embA.pkl B=/path/embB.pkl \
      --algo both --k 20 30 --hdb-min-cluster-size 50 100 \
      --outdir results/ --viz
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# ---- Optional imports ----
def try_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except Exception:
        return None

def try_import_hdbscan():
    try:
        import hdbscan  # type: ignore
        return hdbscan
    except Exception:
        return None


# ---- CLI ----
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spherical K-means + HDBSCAN on full-dimension embeddings from pickle.")
    p.add_argument("--emb", nargs="+", required=True,
                   help="name=path items, e.g., A=/path/embA.pkl B=/path/embB.pkl")
    p.add_argument("--algo", choices=["kmeans", "hdbscan", "both"], default="both",
                   help="Which algorithm(s) to run.")
    # KMeans params
    p.add_argument("--k", nargs="+", type=int, default=[],
                   help="One or more k values for KMeans. If multiple, best silhouette is kept.")
    p.add_argument("--batch-size", type=int, default=10000,
                   help="MiniBatchKMeans batch size.")
    # HDBSCAN params (allow a small sweep)
    p.add_argument("--hdb-min-cluster-size", nargs="+", type=int, default=[50],
                   help="HDBSCAN min_cluster_size values to try.")
    p.add_argument("--hdb-min-samples", type=int, default=None,
                   help="HDBSCAN min_samples. If not set, defaults to min_cluster_size for each trial.")
    p.add_argument("--hdb-metric", choices=["euclidean", "cosine"], default="euclidean",
                   help="HDBSCAN distance metric. With L2-normalised vectors, euclidean ~ cosine.")
    p.add_argument("--hdb-cluster-selection", choices=["eom", "leaf"], default="eom",
                   help="HDBSCAN cluster selection method.")
    # Shared
    p.add_argument("--sil-sample", type=int, default=50000,
                   help="Max sample size for silhouette scoring.")
    p.add_argument("--outdir", type=str, default="data/hdb_out",
                   help="Output directory.")
    p.add_argument("--viz", action="store_true",
                   help="If set, compute a 2D UMAP visualisation and save PNGs.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def parse_emb_arg(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--emb item must be name=path, got: {it}")
        name, path = it.split("=", 1)
        out[name] = path
    return out


# ---- IO ----
def load_pickle_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load {paper_id: {"embeddings": np.ndarray, ...}} -> (X, paper_ids)
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    paper_ids = sorted(data.keys())
    embs = []
    for pid in paper_ids:
        emb = data[pid]["embeddings"]
        if not isinstance(emb, np.ndarray):
            raise TypeError(f"Embeddings for paper_id {pid} is not a numpy array.")
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32, copy=False)
        embs.append(emb)
    X = np.vstack(embs)
    return X, paper_ids


def labels_to_pickle(labels: np.ndarray, paper_ids: List[str], outpath: Path) -> None:
    mapping = {pid: int(lbl) for pid, lbl in zip(paper_ids, labels)}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---- Metrics / Viz ----
def silhouette_fast(X: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> float:
    """Silhouette over all points, requires >=2 clusters."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    n = X.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_samples, replace=False)
        return silhouette_score(X[idx], labels[idx])
    return silhouette_score(X, labels)

def silhouette_ignore_noise(X: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> float:
    """Silhouette ignoring HDBSCAN noise (label == -1)."""
    mask = labels >= 0
    if mask.sum() < 2:
        return float("nan")
    uniq = np.unique(labels[mask])
    if len(uniq) < 2:
        return float("nan")
    n = mask.sum()
    if n > max_samples:
        rng = np.random.RandomState(seed)
        idx_local = rng.choice(np.where(mask)[0], max_samples, replace=False)
        return silhouette_score(X[idx_local], labels[idx_local])
    return silhouette_score(X[mask], labels[mask])

def plot_umap_2d(X_norm: np.ndarray, labels: np.ndarray, title: str, outpath: Path):
    umap = try_import_umap()
    if umap is None:
        print("UMAP not installed; skipping visualisation.")
        return
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, low_memory=True)
    X2 = reducer.fit_transform(X_norm)
    # Map noise (-1) to a distinct max label + 1 so it has its own colour
    lab = labels.copy()
    if (lab == -1).any():
        lab = lab.copy()
        lab[lab == -1] = lab.max() + 1
    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=lab, s=2, linewidths=0, cmap="tab20")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=140)
    plt.close()


# ---- Algorithms ----
def run_kmeans(X_norm: np.ndarray, ks: List[int], batch_size: int, seed: int):
    """Return dict {k: (labels, silhouette)} and the best (k, labels, sil)."""
    results = {}
    best = {"k": None, "sil": -np.inf, "labels": None}
    for k in sorted(set(ks)):
        print(f"  -> KMeans, k={k}")
        km = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=seed, verbose=0)
        labels = km.fit_predict(X_norm)
        sil = silhouette_fast(X_norm, labels, max_samples=50000, seed=seed)
        print(f"     silhouette={sil:.4f}")
        results[k] = (labels, sil)
        if not np.isnan(sil) and sil > best["sil"]:
            best = {"k": k, "sil": sil, "labels": labels}
    return results, best


def run_hdbscan(X_norm: np.ndarray,
                mcs_list: List[int],
                min_samples_opt: Optional[int],
                metric: str,
                cluster_selection_method: str,
                seed: int):
    """
    Try several (min_cluster_size, min_samples) pairs.
    Returns list of trials and best trial.
    """
    hdbscan = try_import_hdbscan()
    if hdbscan is None:
        raise ImportError("hdbscan is not installed. Try: pip install hdbscan")

    trials = []
    best = {"mcs": None, "ms": None, "sil": -np.inf, "labels": None}

    for mcs in sorted(set(mcs_list)):
        ms = mcs if min_samples_opt is None else min_samples_opt
        print(f"  -> HDBSCAN, min_cluster_size={mcs}, min_samples={ms}, metric={metric}, selection={cluster_selection_method}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            core_dist_n_jobs=1,  # be nice to shared boxes
            prediction_data=False
        )
        labels = clusterer.fit_predict(X_norm)
        # Silhouette ignoring noise
        sil = silhouette_ignore_noise(X_norm, labels, max_samples=50000, seed=seed)
        noise_frac = float((labels == -1).mean())
        n_clusters = int((labels >= 0).sum() > 0 and len(np.unique(labels[labels >= 0])) or 0)
        print(f"     silhouette(noise-ignored)={sil:.4f}, noise_fraction={noise_frac:.3f}, clusters={n_clusters}")
        trials.append({"mcs": mcs, "ms": ms, "sil": sil, "labels": labels, "noise_frac": noise_frac, "n_clusters": n_clusters})
        if not np.isnan(sil) and sil > best["sil"]:
            best = {"mcs": mcs, "ms": ms, "sil": sil, "labels": labels}

    return trials, best


# ---- Main ----
def main():
    args = parse_args()
    emb_map = parse_emb_arg(args.emb)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kmeans_summary_rows = []
    hdbscan_summary_rows = []

    for name, path in emb_map.items():
        print(f"\n=== Model {name} ===")
        print(f"Loading embeddings (pickle): {path}")
        X, paper_ids = load_pickle_embeddings(path)

        print("Normalising (L2) in full dimension …")
        X_norm = X.astype(np.float32, copy=False)  # spherical: cosine == dot

        # ---- KMeans ----
        if args.algo in ("kmeans", "both") and args.k:
            kmeans_results, kmeans_best = run_kmeans(X_norm, args.k, args.batch_size, args.seed)

            # Save all K results
            for k, (labels, sil) in kmeans_results.items():
                labels_to_pickle(labels, paper_ids, outdir / f"labels_{name}_k{k}.pkl")

            if kmeans_best["k"] is not None:
                labels_to_pickle(kmeans_best["labels"], paper_ids, outdir / f"labels_{name}_best_k{kmeans_best['k']}.pkl")

                if args.viz:
                    plot_umap_2d(
                        X_norm,
                        kmeans_best["labels"],
                        f"Spherical K-means (model {name}, k={kmeans_best['k']})",
                        outdir / f"umap_{name}_k{kmeans_best['k']}.png",
                    )

                # Cluster stats
                unique, counts = np.unique(kmeans_best["labels"], return_counts=True)
                pd.DataFrame({"cluster": unique, "size": counts}) \
                    .sort_values("size", ascending=False) \
                    .to_csv(outdir / f"cluster_sizes_{name}_k{kmeans_best['k']}.csv", index=False)

                kmeans_summary_rows.append(
                    {
                        "model": name,
                        "algorithm": "kmeans",
                        "best_k": int(kmeans_best["k"]),
                        "silhouette": round(float(kmeans_best["sil"]), 6),
                        "n_clusters": int(len(unique)),
                        "mean_cluster_size": round(float(np.mean(counts)), 2),
                        "median_cluster_size": round(float(np.median(counts)), 2),
                        "labels_pkl": str(outdir / f"labels_{name}_best_k{kmeans_best['k']}.pkl"),
                    }
                )
            else:
                print("No valid k produced a silhouette; check KMeans inputs.")

        # ---- HDBSCAN ----
        if args.algo in ("hdbscan", "both"):
            trials, best = run_hdbscan(
                X_norm,
                mcs_list=args.hdb_min_cluster_size,
                min_samples_opt=args.hdb_min_samples,
                metric=args.hdb_metric,
                cluster_selection_method=args.hdb_cluster_selection,
                seed=args.seed,
            )

            # Save each trial and pick best
            for t in trials:
                mcs, ms, labels = t["mcs"], t["ms"], t["labels"]
                labels_to_pickle(labels, paper_ids, outdir / f"labels_{name}_hdbscan_mcs{mcs}_ms{ms}.pkl")

            if best["labels"] is not None:
                labels_to_pickle(best["labels"], paper_ids, outdir / f"labels_{name}_hdbscan_best.pkl")

                if args.viz:
                    plot_umap_2d(
                        X_norm,
                        best["labels"],
                        f"HDBSCAN (model {name}, mcs={best['mcs']}, ms={best['ms']})",
                        outdir / f"umap_{name}_hdbscan.png",
                    )

                # Cluster size stats (include noise row as -1 for transparency)
                unique, counts = np.unique(best["labels"], return_counts=True)
                pd.DataFrame({"cluster": unique, "size": counts}) \
                    .sort_values("size", ascending=False) \
                    .to_csv(outdir / f"cluster_sizes_{name}_hdbscan.csv", index=False)

                # Summary row (exclude noise from "n_clusters")
                non_noise = best["labels"] >= 0
                n_clusters = int(len(np.unique(best["labels"][non_noise])) if non_noise.any() else 0)
                noise_frac = float((best["labels"] == -1).mean())

                hdbscan_summary_rows.append(
                    {
                        "model": name,
                        "algorithm": "hdbscan",
                        "min_cluster_size": int(best["mcs"]),
                        "min_samples": int(best["ms"]),
                        "silhouette_ignore_noise": round(float(best["sil"]), 6),
                        "n_clusters_ex_noise": n_clusters,
                        "noise_fraction": round(noise_frac, 4),
                        "labels_pkl": str(outdir / f"labels_{name}_hdbscan_best.pkl"),
                    }
                )
            else:
                print("HDBSCAN produced no valid clusters (silhouette was NaN).")

    # Write summaries
    if kmeans_summary_rows:
        pd.DataFrame(kmeans_summary_rows).to_csv(Path(args.outdir) / "summary_spherical_kmeans.csv", index=False)
    if hdbscan_summary_rows:
        pd.DataFrame(hdbscan_summary_rows).to_csv(Path(args.outdir) / "summary_hdbscan.csv", index=False)

    print(f"\nSaved outputs to: {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
