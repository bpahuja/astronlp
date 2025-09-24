#!/usr/bin/env python3
"""
Summarize LLM cluster ratings (JSONL) with optional cluster-size attachment.

Outputs (to --out_dir):
  - llm_summary_by_methodK.csv
  - llm_cluster_level.csv
  - llm_hist_<method>_K<k>.png
  - llm_size_scatter_<method>_K<k>.png  (if sizes are available)

JSONL schema is configurable via --schema.* flags or a YAML manifest.
Cluster label arrays (to compute sizes) are provided via a YAML manifest.

Example:
  python summarize_llm_ratings.py \
    --jsonl paper_cluster_llm_evaluation.jsonl \
    --out_dir results_llm \
    --manifest partitions.yaml
"""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy import stats


# ------------------------- IO helpers -------------------------

def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_labels_generic(path: Path, labels_format: Optional[str], label_column: Optional[str]) -> np.ndarray:
    """Load a 1D array of labels from npy/csv/json/pkl/etc."""
    if labels_format is None:
        # infer by extension
        ext = path.suffix.lower()
        if ext in [".npy", ".npz"]:
            labels_format = "npy"
        elif ext in [".csv", ".tsv"]:
            labels_format = "csv"
        elif ext in [".pkl", ".pickle"]:
            labels_format = "pkl"
        elif ext in [".json"]:
            labels_format = "json"
        else:
            raise ValueError(f"Cannot infer format from extension: {ext}. Please set labels_format.")

    if labels_format == "npy":
        arr = np.load(path)
        return np.asarray(arr).ravel()

    if labels_format == "csv":
        df = pd.read_csv(path)
        col = label_column or "label"
        if col not in df.columns:
            raise ValueError(f"CSV {path} missing label column '{col}'.")
        return df[col].to_numpy()

    if labels_format == "pkl":
        obj = pickle.load(open(path, "rb"))
        if isinstance(obj, (list, tuple)):
            return np.asarray(obj).ravel()
        if isinstance(obj, dict):
            # try some common keys
            for k in ["labels", "y", "cluster_labels"]:
                if k in obj:
                    return np.asarray(obj[k]).ravel()
            raise ValueError(f"PKL {path} is a dict without a known labels key.")
        return np.asarray(obj).ravel()

    if labels_format == "json":
        obj = json.load(open(path, "r"))
        if isinstance(obj, list):
            return np.asarray(obj).ravel()
        if isinstance(obj, dict):
            for k in ["labels", "y", "cluster_labels"]:
                if k in obj:
                    return np.asarray(obj[k]).ravel()
        raise ValueError(f"JSON {path} is not a list and lacks known label keys.")

    raise ValueError(f"Unsupported labels_format: {labels_format}")


# ------------------------- Core logic -------------------------

def apply_method_aliases(s: pd.Series, aliases: Dict[str, str], default: Optional[str]) -> pd.Series:
    s = s.astype(str).str.lower()
    if aliases:
        s = s.map(lambda x: aliases.get(x, x))
    if default is not None:
        s = s.fillna(default)
    return s


def coerce_llm_schema(df: pd.DataFrame,
                      method_field: str,
                      k_field: Optional[str],
                      cluster_field: str,
                      rating_field: str,
                      rater_field: Optional[str],
                      method_aliases: Optional[Dict[str, str]],
                      default_method: Optional[str],
                      default_k: Optional[int]) -> pd.DataFrame:
    """Standardize LLM JSONL columns → ['method','K','cluster_id','rating','rater']"""
    keep = [c for c in [method_field, k_field, cluster_field, rating_field, rater_field] if c and c in df.columns]
    df = df[keep].copy()

    rename_map = {method_field: "method", cluster_field: "cluster_id", rating_field: "rating"}
    if k_field and k_field in df.columns:
        rename_map[k_field] = "K"
    if rater_field and rater_field in df.columns:
        rename_map[rater_field] = "rater"
    df.rename(columns=rename_map, inplace=True)

    if "method" not in df.columns:
        df["method"] = default_method or "hdbscan"
    df["method"] = apply_method_aliases(df["method"], aliases=(method_aliases or {}), default=default_method)

    if "K" not in df.columns:
        df["K"] = default_k if default_k is not None else -1
    df["K"] = pd.to_numeric(df["K"], errors="coerce").fillna(default_k if default_k is not None else -1).astype("Int64")

    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "rater" not in df.columns:
        df["rater"] = "llm"

    df = df.dropna(subset=["method", "K", "cluster_id", "rating"]).reset_index(drop=True)
    return df


def build_size_series(labels: np.ndarray, ignore_noise: bool = True) -> pd.Series:
    labels = np.asarray(labels)
    if ignore_noise:
        labels = labels[labels >= 0]
    return pd.Series(labels).value_counts().sort_index()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.Series(values)
    weights = pd.Series(weights).fillna(0)
    wsum = (values * weights).sum()
    s = weights.sum()
    return float(wsum / s) if s > 0 else float("nan")


# ------------------------- CLI & execution -------------------------

def main():
    p = argparse.ArgumentParser(description="Summarize LLM cluster ratings from JSONL.")
    p.add_argument("--jsonl", required=True, help="Path to LLM ratings JSONL.")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--manifest", help="YAML manifest with partitions + method_aliases + schema (recommended).")

    # Optional inline schema overrides (if you don't supply a manifest)
    p.add_argument("--schema.method_field", default=None)
    p.add_argument("--schema.k_field", default=None)
    p.add_argument("--schema.cluster_field", default=None)
    p.add_argument("--schema.rating_field", default=None)
    p.add_argument("--schema.rater_field", default=None)
    p.add_argument("--default_method", default="hdbscan")
    p.add_argument("--default_k", type=int, default=-1)

    # Plot toggles
    p.add_argument("--no_plots", action="store_true", help="Disable plots.")

    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- read inputs ----------
    df_raw = read_jsonl(Path(args.jsonl))
    if df_raw.empty:
        raise SystemExit(f"No rows in {args.jsonl}")

    manifest = {}
    if args.manifest:
        manifest = yaml.safe_load(open(args.manifest, "r")) or {}

    schema = manifest.get("schema", {})
    # CLI overrides take precedence if provided
    mf = args.__dict__.get("schema.method_field") or schema.get("method_field", "method")
    kf = args.__dict__.get("schema.k_field") or schema.get("k_field", None)  # None → native K=-1
    cf = args.__dict__.get("schema.cluster_field") or schema.get("cluster_field", "cluster_id")
    rf = args.__dict__.get("schema.rating_field") or schema.get("rating_field", "rating")
    rtf = args.__dict__.get("schema.rater_field") or schema.get("rater_field", None)

    method_aliases = manifest.get("method_aliases", {})
    default_method = args.default_method or manifest.get("default_method", "hdbscan")
    default_k = args.default_k if args.default_k is not None else manifest.get("default_k", -1)

    df = coerce_llm_schema(
        df_raw, mf, kf, cf, rf, rtf,
        method_aliases=method_aliases,
        default_method=default_method,
        default_k=default_k,
    )

    # ---------- build size lookups from manifest partitions ----------
    partitions = manifest.get("partitions", [])
    size_lookup: Dict[Tuple[str, int], pd.Series] = {}

    for part in partitions:
        method = str(part.get("method")).lower()
        K = int(part.get("K"))
        labels_path = Path(part["labels_path"])
        labels_format = part.get("labels_format")
        label_col = part.get("label_column")
        ignore_noise = bool(part.get("ignore_noise", True))

        arr = load_labels_generic(labels_path, labels_format, label_col)
        sizes = build_size_series(arr, ignore_noise=ignore_noise)
        size_lookup[(method, K)] = sizes

    # ---------- aggregate per (method,K) ----------
    cluster_tables = []
    for (m, k), g in df.groupby(["method", "K"], dropna=False):
        g = g.copy()

        # attach sizes if available
        g["cluster_size"] = np.nan
        sizes = size_lookup.get((str(m).lower(), int(k)))
        if sizes is not None and not sizes.empty:
            g.loc[:, "cluster_size"] = g["cluster_id"].map(sizes).astype("float")

        # cluster-level table
        agg = g.groupby("cluster_id", dropna=True).agg(
            method=("method", "first"),
            K=("K", "first"),
            n_ratings=("rating", "count"),
            rating_mean=("rating", "mean"),
            rating_median=("rating", "median"),
            rating_std=("rating", "std"),
            cluster_size=("cluster_size", "first"),
        ).reset_index()

        cluster_tables.append(agg)

        # plots
        if not args.no_plots:
            # histogram
            fig = plt.figure(figsize=(6, 4))
            plt.hist(agg["rating_mean"].dropna().values, bins=10)
            plt.xlabel("LLM rating (cluster mean)")
            plt.ylabel("Count")
            plt.title(f"LLM rating histogram: {m}, K={int(k)}")
            fig.tight_layout()
            fig.savefig(out_dir / f"llm_hist_{m}_K{int(k)}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # rating vs size
            if agg["cluster_size"].notna().any():
                xs = np.log1p(agg["cluster_size"].values)
                ys = agg["rating_mean"].values
                rho, p = stats.spearmanr(xs, ys, nan_policy="omit") if len(agg) > 1 else (np.nan, np.nan)
                fig = plt.figure(figsize=(6, 4))
                plt.scatter(xs, ys, alpha=0.6)
                plt.xlabel("log1p(cluster size)")
                plt.ylabel("LLM rating (cluster mean)")
                plt.title(f"Rating vs size: {m}, K={int(k)} (Spearman ρ={rho:.2f}, p={p:.2g})")
                fig.tight_layout()
                fig.savefig(out_dir / f"llm_size_scatter_{m}_K{int(k)}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

    cluster_df = pd.concat(cluster_tables, ignore_index=True) if cluster_tables else pd.DataFrame()

    # ---------- summary by (method,K) ----------
    def frac(series: pd.Series, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
        s = series.dropna()
        if lo is not None:
            return float((s <= lo).mean()) if len(s) else math.nan
        if hi is not None:
            return float((s >= hi).mean()) if len(s) else math.nan
        return math.nan

    def coverage_rows(_m: str, _k: int, rated_clusters: pd.Index) -> Tuple[float, float, int, int]:
        # fraction of clusters & size covered by ratings
        sizes = size_lookup.get((str(_m).lower(), int(_k)))
        if sizes is None or sizes.empty:
            return math.nan, math.nan, len(rated_clusters), math.nan
        total_clusters = len(sizes)
        frac_clusters = len(set(rated_clusters).intersection(set(sizes.index))) / max(total_clusters, 1)
        rated_size = float(sizes.loc[sizes.index.intersection(rated_clusters)].sum())
        total_size = float(sizes.sum())
        frac_points = rated_size / total_size if total_size > 0 else math.nan
        return frac_clusters, frac_points, len(rated_clusters), total_clusters

    if not cluster_df.empty:
        summary_rows = []
        for (m, k), g in cluster_df.groupby(["method", "K"], dropna=False):
            size_w_mean = weighted_mean(g["rating_mean"], g["cluster_size"]) if "cluster_size" in g.columns else math.nan
            frac_low = frac(g["rating_mean"], lo=2.0)
            frac_high = frac(g["rating_mean"], hi=4.0)
            cov_clusters, cov_points, n_rated, n_total = coverage_rows(m, int(k), g["cluster_id"])
            summary_rows.append({
                "method": m,
                "K": int(k),
                "n_clusters_rated": int(n_rated),
                "mean_rating": float(g["rating_mean"].mean()),
                "median_rating": float(g["rating_mean"].median()),
                "std_rating": float(g["rating_mean"].std()),
                "iqr_rating": float(g["rating_mean"].quantile(0.75) - g["rating_mean"].quantile(0.25)),
                "frac_low_≤2": frac_low,
                "frac_high_≥4": frac_high,
                "median_cluster_size": float(g["cluster_size"].median()) if "cluster_size" in g.columns else math.nan,
                "size_weighted_mean_rating": size_w_mean,
                "coverage_rated_clusters": cov_clusters,
                "coverage_rated_points": cov_points,
                "total_clusters_available": n_total if not math.isnan(cov_clusters) else math.nan,
            })
        summary = pd.DataFrame(summary_rows).sort_values(["method", "K"])
    else:
        summary = pd.DataFrame(columns=[
            "method","K","n_clusters_rated","mean_rating","median_rating","std_rating","iqr_rating",
            "frac_low_≤2","frac_high_≥4","median_cluster_size","size_weighted_mean_rating",
            "coverage_rated_clusters","coverage_rated_points","total_clusters_available"
        ])

    # ---------- save ----------
    summary_path = out_dir / "llm_summary_by_methodK.csv"
    clusters_path = out_dir / "llm_cluster_level.csv"
    summary.to_csv(summary_path, index=False)
    cluster_df.to_csv(clusters_path, index=False)
    print(f"Saved {summary_path}")
    print(f"Saved {clusters_path}")


if __name__ == "__main__":
    main()
