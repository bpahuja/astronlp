#!/usr/bin/env python3
"""
Pipeline-2 (paper-level) gap-closer: matched-K comparison, keyword alignment, and plots.
- Merge HDBSCAN micro-clusters to target K and align to K-means(K)
- Report AMI/NMI/V, Sil/DB/CH, cluster size statistics, and bootstrap CIs
- Compute c-TF-IDF cluster keywords and F1@10 vs author keywords
- Plot confusion heatmap (Hungarian-aligned), size histograms, UMAP overlays
- Optional: noise vs metric scatter from experiments CSV

Usage:
  python p2_paper_level_close_gaps.py \
    --data reps/bag_of_papers.csv --vector-prefix cluster_ \
    --paper-id paper_id --title-col title --keywords-col keywords --abstract-col abstract \
    --hdbscan-labels hdbscan_labels.csv \
    --k-targets 50 100 \
    --work-dir out_p2_gaps

Dependencies: numpy, pandas, scikit-learn, matplotlib, seaborn, umap-learn (optional)
"""

import argparse, re, json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_mutual_info_score, normalized_mutual_info_score, v_measure_score,
    confusion_matrix
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

# Optional UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

plt.rcParams["figure.dpi"] = 140
sns.set_context("talk")

# ------------------------ IO & prep ------------------------

def load_bop(path: Path, vector_prefix: str, paper_id_col: str,
             title_col: Optional[str], keywords_col: Optional[str],
             abstract_col: Optional[str]) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    vec_cols = [c for c in df.columns if c.startswith(vector_prefix)]
    if not vec_cols:
        raise ValueError(f"No columns start with '{vector_prefix}'")
    X = df[vec_cols].to_numpy(dtype=np.float32)
    # Clean NaNs/Infs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    needed = [paper_id_col]
    for c in [title_col, keywords_col, abstract_col]:
        if c: needed.append(c)
    missing = [c for c in needed if c and c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in data: {missing} (continuing)")
    return X, df, vec_cols

def l2_norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

# ------------------------ K-means & metrics ------------------------

def run_kmeans_cosine(X: np.ndarray, K: int, random_state: int = 42,
                      n_init: int = 20, max_iter: int = 500) -> np.ndarray:
    Xn = l2_norm(X)
    km = KMeans(n_clusters=K, random_state=random_state, n_init=n_init, max_iter=max_iter, init="k-means++")
    labels = km.fit_predict(Xn)
    return labels

def internal_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    mask = labels != -1
    if mask.sum() < 3 or len(np.unique(labels[mask])) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan,
                "n_clusters": int(len(np.unique(labels[mask]))), "noise_ratio": float((~mask).mean()),
                "clustered_ratio": float(mask.mean())}
    Xm, ym = X[mask], labels[mask]
    return {
        "silhouette": float(silhouette_score(Xm, ym)),
        "calinski_harabasz": float(calinski_harabasz_score(Xm, ym)),
        "davies_bouldin": float(davies_bouldin_score(Xm, ym)),
        "n_clusters": int(len(np.unique(ym))),
        "noise_ratio": float((~mask).mean()),
        "clustered_ratio": float(mask.mean())
    }

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(values, size=len(values), replace=True)
        boots.append(np.nanmean(s))
    lo, hi = np.nanpercentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

# ------------------------ HDBSCAN merge to target K ------------------------

def merge_hdbscan_to_k(
    X: np.ndarray,
    labels_micro: np.ndarray,
    K_target: int,
    metric: str = "cosine",
    attach_noise: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Merge HDBSCAN micro-clusters to target K by agglomerating micro-centroids.
    metric: 'cosine' (average linkage) or 'euclidean' (ward)
    """
    mask_assigned = labels_micro != -1
    X_assigned = X[mask_assigned]
    labs = labels_micro[mask_assigned]
    uniq = np.unique(labs)
    # micro centroids
    centroids = []
    idx_map = {}
    for i, k in enumerate(uniq):
        c = X_assigned[labs == k].mean(axis=0)
        centroids.append(c)
        idx_map[k] = i
    C = np.vstack(centroids)

    if metric == "euclidean":
        agg = AgglomerativeClustering(n_clusters=K_target, linkage="ward")
    else:
        # cosine metric
        agg = AgglomerativeClustering(n_clusters=K_target, linkage="average", metric="cosine")
    micro_to_merged = agg.fit_predict(C)  # length: n_micro

    # merged centroids in data space
    merged_centroids = np.zeros((K_target, X.shape[1]), dtype=np.float32)
    for m in range(K_target):
        members = [i for i in range(len(C)) if micro_to_merged[i] == m]
        if members:
            merged_centroids[m] = C[members].mean(axis=0)

    # assign each paper a merged label
    merged_labels = np.full(shape=len(labels_micro), fill_value=-1, dtype=int)
    # assigned points inherit merged label of their micro-cluster
    for k in uniq:
        mi = idx_map[k]
        merged_k = micro_to_merged[mi]
        merged_labels[(labels_micro == k)] = merged_k

    # optionally attach noise to nearest merged centroid
    if attach_noise:
        if metric == "euclidean":
            d = ((X - merged_centroids[:, None, :])**2).sum(axis=2).T  # (n, K)
        else:
            Xn = l2_norm(X)
            Mn = l2_norm(merged_centroids)
            d = 1.0 - (Xn @ Mn.T)  # cosine distance
        nearest = d.argmin(axis=1)
        merged_labels[labels_micro == -1] = nearest[labels_micro == -1]

    details = {
        "n_micro": int(len(uniq)),
        "micro_to_merged": micro_to_merged.tolist(),
        "micro_ids": uniq.tolist(),
        "K_target": int(K_target),
        "attach_noise": bool(attach_noise)
    }
    return merged_labels, details

# ------------------------ Alignment & confusion ------------------------

def alignment_scores(a: np.ndarray, b: np.ndarray, exclude_noise: bool = True) -> Dict[str, float]:
    mask = np.ones_like(a, dtype=bool)
    if exclude_noise:
        mask = (a != -1) & (b != -1)
    aa, bb = a[mask], b[mask]
    return {
        "AMI": float(adjusted_mutual_info_score(aa, bb)),
        "NMI": float(normalized_mutual_info_score(aa, bb)),
        "V": float(v_measure_score(aa, bb)),
        "n_compared": int(len(aa))
    }

def plot_confusion_heatmap(a: np.ndarray, b: np.ndarray, out: Path, title: str):
    # align b to a via Hungarian using contingency (exclude noise)
    mask = (a != -1) & (b != -1)
    aa, bb = a[mask], b[mask]
    A = aa.max() + 1
    B = bb.max() + 1
    CM = contingency_matrix(aa, bb)  # shape (A,B)
    cost = CM.max() - CM
    r, c = linear_sum_assignment(cost)
    CM_aligned = CM[:, c]  # permute columns
    CM_norm = (CM_aligned.T / CM_aligned.sum(axis=1, keepdims=True).T).T  # row-normalized

    plt.figure(figsize=(8, 6))
    sns.heatmap(CM_norm, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "Row-normalized"})
    plt.xlabel("HDBSCAN→merge clusters (aligned)")
    plt.ylabel("K-means clusters")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion heatmap → {out}")

# ------------------------ Keyword alignment via c-TF-IDF ------------------------

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")

def normalize_kw(s: str) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(s or "")]
    toks = [t for t in toks if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    return toks

def parse_author_keywords(s: str) -> List[str]:
    if pd.isna(s):
        return []
    parts = re.split(r"[;,/|]", str(s))
    outs = []
    for p in parts:
        toks = normalize_kw(p)
        outs.extend(toks)
    return outs

def ctfidf_top_terms(df: pd.DataFrame, labels: np.ndarray,
                     text_cols: List[str], top_k: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    """
    Build class-based TF-IDF (aggregate one doc per cluster).
    Returns dict: cluster_id -> [(term, score), ...]
    """
    # aggregate text per cluster
    data = {}
    for cid in np.unique(labels[labels != -1]):
        rows = df.loc[labels == cid, text_cols].fillna("")
        agg = rows.apply(lambda r: " ".join(map(str, r.values)), axis=1)
        data[cid] = " ".join(agg.values.tolist())
    if not data:
        return {}

    classes = sorted(data.keys())
    docs = [data[c] for c in classes]
    # tokenize with CountVectorizer
    vect = CountVectorizer(
        token_pattern=r"[A-Za-z][A-Za-z0-9_\-]+",
        lowercase=True,
        stop_words="english",
        min_df=2
    )
    X = vect.fit_transform(docs)  # shape (n_classes, vocab)
    vocab = np.array(vect.get_feature_names_out())

    # class-based TF
    tf = X.astype(float)
    tf = tf / (tf.sum(axis=1) + 1e-9)

    # IDF across classes
    dfreq = (X > 0).sum(axis=0).A1  # in how many classes term appears
    nC = X.shape[0]
    idf = np.log((1 + nC) / (1 + dfreq)) + 1.0

    ctfidf = tf.multiply(idf)
    # top terms per class
    tops = {}
    for i, cid in enumerate(classes):
        row = np.asarray(ctfidf[i].todense()).ravel()
        idx = row.argsort()[::-1][:top_k]
        tops[cid] = [(vocab[j], float(row[j])) for j in idx]
    return tops

def keyword_alignment_f1(
    df: pd.DataFrame, labels: np.ndarray, top_terms: Dict[int, List[Tuple[str, float]]],
    author_kw_col: str, top_k: int = 10
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for cid, terms in top_terms.items():
        cluster_mask = labels == cid
        # set of author keywords across cluster members
        akws = set()
        for s in df.loc[cluster_mask, author_kw_col].fillna(""):
            akws.update(parse_author_keywords(s))
        # top-k terms
        toks = set([t for t, _ in terms[:top_k]])
        # compute precision/recall/F1
        inter = len(toks & akws)
        p = inter / max(len(toks), 1)
        r = inter / max(len(akws), 1)
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        rows.append({"cluster": cid, "precision@10": p, "recall@10": r, "f1@10": f1, "n_papers": int(cluster_mask.sum())})
    out = pd.DataFrame(rows).sort_values("f1@10", ascending=False)
    macro = out[["precision@10", "recall@10", "f1@10"]].mean().to_dict()
    lo, hi = bootstrap_ci(out["f1@10"].to_numpy())
    summary = {"macro_precision@10": macro["precision@10"], "macro_recall@10": macro["recall@10"],
               "macro_f1@10": macro["f1@10"], "f1@10_CI_lo": lo, "f1@10_CI_hi": hi,
               "n_clusters_eval": len(out)}
    return out, summary

# ------------------------ UMAP / PCA 2-D ------------------------

def embed_2d(X: np.ndarray, metric: str = "cosine", random_state: int = 42) -> np.ndarray:
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric=metric, random_state=random_state)
        return reducer.fit_transform(X)
    # fallback: PCA 2-D on L2-normalized vectors (approx cosine)
    Xn = l2_norm(X)
    return PCA(n_components=2, random_state=random_state).fit_transform(Xn)

def plot_2d(X2: np.ndarray, labels: np.ndarray, out: Path, title: str):
    mask = labels != -1
    plt.figure(figsize=(7, 6))
    plt.scatter(X2[mask, 0], X2[mask, 1], s=5, alpha=0.7, c=labels[mask], cmap="tab20")
    plt.scatter(X2[~mask, 0], X2[~mask, 1], s=5, alpha=0.15, c="lightgray", label="noise")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] 2-D projection → {out}")

def plot_size_hist(labels: np.ndarray, out: Path, title: str):
    labs = labels[labels != -1]
    sizes = pd.Series(labs).value_counts().values
    plt.figure(figsize=(6,4))
    plt.hist(sizes, bins=30, alpha=0.9)
    plt.xlabel("Cluster size")
    plt.ylabel("#clusters")
    plt.title(f"{title}\n(min={sizes.min()}, median={np.median(sizes):.0f}, IQR={np.percentile(sizes,75)-np.percentile(sizes,25):.0f}, max={sizes.max()})")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Size histogram → {out}")

# ------------------------ Noise vs metric scatter (optional) ------------------------

def plot_noise_tradeoff(experiments_csv: Path, out: Path):
    df = pd.read_csv(experiments_csv)
    # Expect columns: algorithm, preprocessing, noise_ratio, silhouette_score, davies_bouldin_score, calinski_harabasz_score
    plt.figure(figsize=(7,5))
    for m in ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"]:
        if m not in df.columns: continue
        ax = plt.gca()
        if m == "davies_bouldin_score":
            ylab = "Davies-Bouldin (↓)"
        elif m == "calinski_harabasz_score":
            ylab = "Calinski-Harabasz (↑)"
        else:
            ylab = "Silhouette (↑)"
        sns.scatterplot(data=df, x="noise_ratio", y=m, hue="algorithm", style="preprocessing", alpha=0.8)
        plt.xlabel("Noise ratio")
        plt.ylabel(ylab)
        plt.title("Noise vs Metric trade-off")
        break  # one panel is enough; swap above to plot another
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Noise trade-off → {out}")

# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--vector-prefix", type=str, default="cluster_")
    ap.add_argument("--paper-id", type=str, default="paper_id")
    ap.add_argument("--title-col", type=str, default="title")
    ap.add_argument("--keywords-col", type=str, default="keywords")
    ap.add_argument("--abstract-col", type=str, default=None)
    ap.add_argument("--hdbscan-labels", type=Path, required=True,
                    help="CSV with columns: paper_id, hdbscan_label (micro labels)")
    ap.add_argument("--k-targets", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--attach-noise", action="store_true", help="Attach HDBSCAN noise to nearest merged centroid")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--experiments-csv", type=Path, default=None,
                    help="(optional) CSV with experiment summary to plot noise-vs-metric")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Load BoP
    X, df, vec_cols = load_bop(args.data, args.vector_prefix, args.paper_id,
                               args.title_col, args.keywords_col, args.abstract_col)

    # Load HDBSCAN micro labels, join on paper_id
    labs_df = pd.read_csv(args.hdbscan_labels)
    if args.paper_id not in labs_df.columns or "hdbscan_label" not in labs_df.columns:
        raise ValueError("hdbscan-labels CSV must contain columns: paper_id, hdbscan_label")
    df = df.merge(labs_df[[args.paper_id, "hdbscan_label"]], on=args.paper_id, how="left")
    if df["hdbscan_label"].isna().any():
        print("[WARN] Some papers missing HDBSCAN labels; filling as -1.")
        df["hdbscan_label"] = df["hdbscan_label"].fillna(-1).astype(int)
    labels_micro = df["hdbscan_label"].to_numpy(dtype=int)

    # For each target K: run K-means(K), merge HDBSCAN→K, compare, metrics, plots, keywords
    text_cols = [c for c in [args.title_col, args.abstract_col] if c and c in df.columns]

    master_rows = []
    for K in args.k_targets:
        print(f"\n=== Matched granularity @ K={K} ===")
        # K-means (cosine + L2)
        km_labels = run_kmeans_cosine(X, K, random_state=42)
        # Merge HDBSCAN micro to K
        merged_labels, details = merge_hdbscan_to_k(
            X, labels_micro, K_target=K, metric="cosine",
            attach_noise=args.attach_noise
        )
        # Metrics
        km_int = internal_metrics(X, km_labels)
        hdb_int = internal_metrics(X, merged_labels)
        align = alignment_scores(km_labels, merged_labels, exclude_noise=True)

        print(f"KMeans(K={K}): Sil={km_int['silhouette']:.3f}, DB={km_int['davies_bouldin']:.3f}, CH={km_int['calinski_harabasz']:.1f}")
        print(f"HDB→merge(K={K}): Sil={hdb_int['silhouette']:.3f}, DB={hdb_int['davies_bouldin']:.3f}, CH={hdb_int['calinski_harabasz']:.1f}")
        print(f"Alignment: AMI={align['AMI']:.3f}, NMI={align['NMI']:.3f}, V={align['V']:.3f}, n={align['n_compared']}")

        # Save labels
        out_lab = args.work_dir / f"labels_matchedK{K}.csv"
        pd.DataFrame({
            args.paper_id: df[args.paper_id],
            f"kmeans_K{K}": km_labels,
            f"hdbmerge_K{K}": merged_labels
        }).to_csv(out_lab, index=False)
        print(f"[OK] Labels saved → {out_lab}")

        # Confusion heatmap (Hungarian-aligned)
        plot_confusion_heatmap(km_labels, merged_labels,
                               args.work_dir / f"confusion_K{K}.png",
                               title=f"Matched-K confusion (K={K})")

        # Size histograms
        plot_size_hist(km_labels, args.work_dir / f"size_kmeans_K{K}.png", f"K-means K={K}")
        plot_size_hist(merged_labels, args.work_dir / f"size_hdbmerge_K{K}.png", f"HDBSCAN→merge K={K}")

        # 2-D projection overlays
        X2 = embed_2d(X, metric="cosine", random_state=42)
        plot_2d(X2, km_labels, args.work_dir / f"umap_kmeans_K{K}.png", f"K-means K={K}")
        plot_2d(X2, merged_labels, args.work_dir / f"umap_hdbmerge_K{K}.png", f"HDBSCAN→merge K={K}")

        # Keyword alignment via c-TF-IDF
        if text_cols and args.keywords_col in df.columns:
            km_top = ctfidf_top_terms(df, km_labels, text_cols=text_cols, top_k=10)
            km_kw_df, km_summary = keyword_alignment_f1(df, km_labels, km_top, args.keywords_col, top_k=10)
            km_kw_df.to_csv(args.work_dir / f"keywords_kmeans_K{K}.csv", index=False)
            json.dump(km_summary, open(args.work_dir / f"keywords_kmeans_K{K}_summary.json","w"), indent=2)

            hb_top = ctfidf_top_terms(df, merged_labels, text_cols=text_cols, top_k=10)
            hb_kw_df, hb_summary = keyword_alignment_f1(df, merged_labels, hb_top, args.keywords_col, top_k=10)
            hb_kw_df.to_csv(args.work_dir / f"keywords_hdbmerge_K{K}.csv", index=False)
            json.dump(hb_summary, open(args.work_dir / f"keywords_hdbmerge_K{K}_summary.json","w"), indent=2)

            print(f"[OK] Keyword F1@10 — KM: {km_summary['macro_f1@10']:.3f} "
                  f"(CI {km_summary['f1@10_CI_lo']:.3f}–{km_summary['f1@10_CI_hi']:.3f}); "
                  f"HDB: {hb_summary['macro_f1@10']:.3f} "
                  f"(CI {hb_summary['f1@10_CI_lo']:.3f}–{hb_summary['f1@10_CI_hi']:.3f})")

        # Aggregate row
        master_rows.append({
            "K": K,
            "km_sil": km_int["silhouette"], "km_db": km_int["davies_bouldin"], "km_ch": km_int["calinski_harabasz"],
            "hb_sil": hdb_int["silhouette"], "hb_db": hdb_int["davies_bouldin"], "hb_ch": hdb_int["calinski_harabasz"],
            "AMI": align["AMI"], "NMI": align["NMI"], "V": align["V"], "n_compared": align["n_compared"]
        })

    pd.DataFrame(master_rows).to_csv(args.work_dir / "matchedK_summary.csv", index=False)
    print(f"[OK] Matched-K summary → {args.work_dir / 'matchedK_summary.csv'}")

    # Optional: noise vs metric scatter from experiments CSV
    if args.experiments_csv and args.experiments_csv.exists():
        plot_noise_tradeoff(args.experiments_csv, args.work_dir / "noise_tradeoff.png")

if __name__ == "__main__":
    main()
