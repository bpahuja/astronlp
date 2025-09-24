#!/usr/bin/env python3
"""
Enhanced Clustering Experimentation Framework with Optimal K Selection
Provides academic-grade evaluation with statistical rigor for thesis research.

Key Features:
- Systematic k-evaluation with multiple methods (Elbow, Gap Statistic, Silhouette)
- Stability analysis and consensus clustering
- Feature importance with statistical significance
- Baseline comparison methods
- LaTeX-ready output for academic papers
- Reproducible methodology with fixed seeds

Author: Research Framework v2.0
Usage: python enhanced_clustering_experiments.py --config experiments_config.yaml
"""

import json
import yaml
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count
import re
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.cluster import OPTICS, Birch, MeanShift, AffinityPropagation
from sklearn.mixture import GaussianMixture
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# Preprocessing and dimensionality reduction
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

# Distance metrics and evaluation
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score, fowlkes_mallows_score,
    silhouette_samples
)
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, cdist
from scipy.stats import friedmanchisquare, wilcoxon, kruskal
from kneed import KneeLocator

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Set global random seed for reproducibility
GLOBAL_RANDOM_STATE = 42
np.random.seed(GLOBAL_RANDOM_STATE)

class EnhancedClusteringFramework:
    """Enhanced framework with optimal k selection and academic-grade evaluation"""
    
    def __init__(self, work_dir: Path, n_jobs: int = -1, random_state: int = 42):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.random_state = random_state
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize results directory
        self.results_dir = self.work_dir / f"experiments_{self.experiment_id}"
        self.results_dir.mkdir(exist_ok=True)
        
        # K-optimization results storage
        self.k_optimization_results = {}
        self.stability_results = {}
        self.feature_importance_results = {}

    # ---------- Matched-Granularity (HDBSCAN→K vs KMeans(K)) helpers ----------

    def _l2_norm(self, X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    def _internal_metrics_on(self, X_proc: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        # Reuse the framework’s evaluate_clustering so outputs match other experiments
        return self.evaluate_clustering(X_proc, labels)

    def _run_kmeans_cosine(self, X: np.ndarray, K: int, n_init: int = 20, max_iter: int = 500, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        # KMeans with cosine ≈ KMeans on L2-normalized vectors
        Xn = self._l2_norm(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        km = KMeans(n_clusters=K, random_state=self.random_state, n_init=n_init, max_iter=max_iter, tol=tol, init="k-means++")
        labels = km.fit_predict(Xn)
        return labels, Xn  # return processed X for consistent metric computation

    def _merge_hdbscan_to_k(self, X: np.ndarray, labels_micro: np.ndarray, K_target: int, metric: str = "cosine", attach_noise: bool = True) -> np.ndarray:
        # Merge micro HDBSCAN clusters to K_target by agglomerating micro-centroids
        mask_assigned = labels_micro != -1
        uniq = np.unique(labels_micro[mask_assigned])
        if len(uniq) == 0:
            # all noise → fall back to single cluster or all -1
            out = np.full_like(labels_micro, fill_value=(-1 if not attach_noise else 0))
            return out

        centroids = []
        idx_map = {}
        Xa = np.nan_to_num(X[mask_assigned], nan=0.0, posinf=0.0, neginf=0.0)
        labs = labels_micro[mask_assigned]
        for i, k in enumerate(uniq):
            c = Xa[labs == k].mean(axis=0)
            centroids.append(c)
            idx_map[k] = i
        C = np.vstack(centroids)

        if metric == "euclidean":
            agg = AgglomerativeClustering(n_clusters=K_target, linkage="ward")
            merged_idx = agg.fit_predict(C)
        else:
            agg = AgglomerativeClustering(n_clusters=K_target, linkage="average", metric="cosine")
            merged_idx = agg.fit_predict(C)

        # map each micro id → merged id
        merged_labels = np.full(shape=len(labels_micro), fill_value=-1, dtype=int)
        for k in uniq:
            merged_labels[(labels_micro == k)] = merged_idx[idx_map[k]]

        # optionally attach noise to nearest merged centroid
        if attach_noise:
            if metric == "euclidean":
                mC = np.vstack([C[merged_idx == m].mean(axis=0) for m in range(K_target)])
                d = ((np.nan_to_num(X, nan=0.0) - mC[:, None, :])**2).sum(axis=2).T
            else:
                Xn = self._l2_norm(np.nan_to_num(X, nan=0.0))
                mC = np.vstack([C[merged_idx == m].mean(axis=0) for m in range(K_target)])
                mCn = self._l2_norm(mC)
                d = 1.0 - (Xn @ mCn.T)
            nearest = d.argmin(axis=1)
            merged_labels[labels_micro == -1] = nearest[labels_micro == -1]

        return merged_labels

    def _alignment_scores(self, a: np.ndarray, b: np.ndarray, exclude_noise: bool = True) -> Dict[str, float]:
        mask = np.ones_like(a, dtype=bool)
        if exclude_noise:
            mask = (a != -1) & (b != -1)
        aa, bb = a[mask], b[mask]
        return {
            'AMI': float(adjusted_mutual_info_score(aa, bb)),
            'NMI': float(normalized_mutual_info_score(aa, bb)),
            'V': float(v_measure_score(aa, bb)),
            'n_compared': int(len(aa))
        }

    def _plot_confusion_heatmap(self, km_labels: np.ndarray, hb_labels: np.ndarray, out: Path, title: str):
        mask = (km_labels != -1) & (hb_labels != -1)
        aa, bb = km_labels[mask], hb_labels[mask]
        if len(aa) == 0:
            return
        CM = contingency_matrix(aa, bb)  # shape (A,B)
        cost = CM.max() - CM
        r, c = linear_sum_assignment(cost)
        CM_aligned = CM[:, c]
        # row-normalize for readability
        CM_norm = (CM_aligned.T / (CM_aligned.sum(axis=1) + 1e-9).T).T

        plt.figure(figsize=(8, 6))
        sns.heatmap(CM_norm, cmap="viridis", vmin=0, vmax=1, cbar_kws={'label': 'Row-normalized'})
        plt.xlabel("HDBSCAN→merge clusters (aligned)")
        plt.ylabel("K-means clusters")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_size_hist(self, labels: np.ndarray, out: Path, title: str):
        labs = labels[labels != -1]
        if labs.size == 0:
            return
        sizes = pd.Series(labs).value_counts().values
        plt.figure(figsize=(6,4))
        plt.hist(sizes, bins=30, alpha=0.9)
        plt.xlabel("Cluster size"); plt.ylabel("#clusters")
        plt.title(f"{title}\n(min={sizes.min()}, median={np.median(sizes):.0f}, IQR={np.percentile(sizes,75)-np.percentile(sizes,25):.0f}, max={sizes.max()})")
        plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

    def _embed_2d(self, X: np.ndarray, metric: str = "cosine") -> np.ndarray:
        if 'UMAP_AVAILABLE' in globals() and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric=metric, random_state=self.random_state)
            return reducer.fit_transform(X)
        # PCA fallback approximates cosine on L2-normalized vectors
        Xn = self._l2_norm(np.nan_to_num(X, nan=0.0))
        reducer = PCA(n_components=2, random_state=self.random_state)
        return reducer.fit_transform(Xn)

    def _plot_2d(self, X2: np.ndarray, labels: np.ndarray, out: Path, title: str):
        mask = labels != -1
        plt.figure(figsize=(7,6))
        plt.scatter(X2[mask,0], X2[mask,1], s=5, alpha=0.7, c=labels[mask], cmap="tab20")
        if (~mask).any():
            plt.scatter(X2[~mask,0], X2[~mask,1], s=5, alpha=0.15, c="lightgray", label="noise")
        plt.axis("off"); plt.title(title)
        plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

    # ----- c-TF-IDF keywords and alignment (F1@K) -----

    _TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")

    def _normalize_kw(self, s: str) -> List[str]:
        toks = [t.lower() for t in self._TOKEN_RE.findall(s or "")]
        toks = [t for t in toks if t not in ENGLISH_STOP_WORDS and len(t) > 2]
        return toks

    def _parse_author_keywords(self, s: str) -> List[str]:
        if pd.isna(s): return []
        parts = re.split(r"[;,/|]", str(s))
        outs = []
        for p in parts:
            outs.extend(self._normalize_kw(p))
        return outs

    def _ctfidf_top_terms(self, df: pd.DataFrame, labels: np.ndarray, text_cols: List[str], top_k: int = 10) -> Dict[int, List[Tuple[str, float]]]:
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
        vect = CountVectorizer(token_pattern=r"[A-Za-z][A-Za-z0-9_\-]+", lowercase=True, stop_words="english", min_df=2)
        Xc = vect.fit_transform(docs)
        vocab = np.array(vect.get_feature_names_out())
        # class TF
        tf = Xc.astype(float)
        tf = tf / (tf.sum(axis=1) + 1e-9)
        # IDF over classes
        dfreq = (Xc > 0).sum(axis=0).A1
        nC = Xc.shape[0]
        idf = np.log((1 + nC) / (1 + dfreq)) + 1.0
        ctfidf = tf.multiply(idf)
        tops = {}
        for i, cid in enumerate(classes):
            row = np.asarray(ctfidf[i].todense()).ravel()
            idx = row.argsort()[::-1][:top_k]
            tops[cid] = [(vocab[j], float(row[j])) for j in idx]
        return tops

    def _keyword_alignment_f1(self, df: pd.DataFrame, labels: np.ndarray, top_terms: Dict[int, List[Tuple[str, float]]], author_kw_col: str, top_k: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
        rows = []
        for cid, terms in top_terms.items():
            cluster_mask = labels == cid
            akws = set()
            for s in df.loc[cluster_mask, author_kw_col].fillna(""):
                akws.update(self._parse_author_keywords(s))
            toks = set([t for t, _ in terms[:top_k]])
            inter = len(toks & akws)
            p = inter / max(len(toks), 1)
            r = inter / max(len(akws), 1)
            f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
            rows.append({"cluster": cid, "precision@10": p, "recall@10": r, "f1@10": f1, "n_papers": int(cluster_mask.sum())})
        out = pd.DataFrame(rows).sort_values("f1@10", ascending=False)
        macro = out[["precision@10", "recall@10", "f1@10"]].mean().to_dict()
        # lightweight CI (percentiles of per-cluster scores)
        lo, hi = np.nanpercentile(out["f1@10"].to_numpy(), [2.5, 97.5]) if len(out) > 0 else (np.nan, np.nan)
        summary = {"macro_precision@10": macro.get("precision@10", np.nan),
                   "macro_recall@10": macro.get("recall@10", np.nan),
                   "macro_f1@10": macro.get("f1@10", np.nan),
                   "f1@10_CI_lo": float(lo), "f1@10_CI_hi": float(hi),
                   "n_clusters_eval": len(out)}
        return out, summary

    # ---------- Orchestrator for matched-granularity (paper level, Pipeline-2) ----------

    def run_matched_granularity(self, X: np.ndarray, df: pd.DataFrame, mg_cfg: Dict) -> Dict:
        """
        Compares KMeans(K) vs HDBSCAN micro→merge(K) at matched granularity.
        Writes CSVs + PNGs into results_dir; returns a manifest with file paths and metrics.
        """
        out_dir = self.results_dir
        # --- config read ---
        data_cfg = mg_cfg.get("data", {})
        pid_col = data_cfg.get("paper_id_column", "paper_id")
        title_col = data_cfg.get("title_column", None)
        keywords_col = data_cfg.get("keywords_column", None)
        abstract_col = data_cfg.get("abstract_column", None)

        hdb_cfg = mg_cfg.get("hdbscan", {})
        labels_path = Path(hdb_cfg.get("labels_path", ""))
        label_col = hdb_cfg.get("label_column", "hdbscan_label")

        k_targets = mg_cfg.get("k_targets", [50, 100])
        attach_noise = bool(mg_cfg.get("attach_noise", True))
        merge_metric = mg_cfg.get("merge_metric", "cosine")
        kmeans_params = mg_cfg.get("kmeans", {}).get("params", {"n_init": 20, "max_iter": 500, "tol": 1e-4})
        keyword_top_k = int(mg_cfg.get("keyword_top_k", 10))
        umap_enabled = bool(mg_cfg.get("umap_enabled", True))

        # --- load / align HDBSCAN labels ---
                # --- load / derive HDBSCAN micro labels ---
        hdb_source = hdb_cfg.get("source", "from_fit")  # 'from_fit' | 'from_results' | 'from_csv'
        label_col = hdb_cfg.get("label_column", "hdbscan_label")

        if hdb_source == "from_csv":
            labels_path = Path(hdb_cfg.get("labels_path", ""))
            if not labels_path.exists():
                raise FileNotFoundError(f"HDBSCAN labels CSV not found: {labels_path}")
            labs_df = pd.read_csv(labels_path)
            if label_col not in labs_df.columns:
                raise ValueError(f"'{label_col}' not found in {labels_path}")
            if pid_col in labs_df.columns and pid_col in df.columns:
                df = df.merge(labs_df[[pid_col, label_col]], on=pid_col, how="left")
                if df[label_col].isna().any():
                    print("[WARN] Some papers missing HDBSCAN labels; filling as -1.")
                    df[label_col] = df[label_col].fillna(-1).astype(int)
                labels_micro = df[label_col].astype(int).to_numpy()
            else:
                if len(labs_df) != len(df):
                    raise ValueError("HDBSCAN labels CSV length mismatch and no paper_id to merge on.")
                labels_micro = labs_df[label_col].astype(int).to_numpy()

        elif hdb_source == "from_results":
            # load labels from a previous saved results pickle
            pkl_path = Path(hdb_cfg.get("results_pkl_path", self.results_dir / "complete_results.pkl"))
            if not pkl_path.exists():
                raise FileNotFoundError(f"Results pickle not found: {pkl_path}")
            with open(pkl_path, "rb") as f:
                res_list = pickle.load(f)
            # pick the first successful HDBSCAN run (or filter by name substring)
            name_contains = hdb_cfg.get("experiment_name_contains", None)
            chosen = None
            for r in res_list:
                if not r.get("success"): 
                    continue
                if r.get("algorithm") != "hdbscan":
                    continue
                if name_contains and name_contains not in r.get("experiment_name", ""):
                    continue
                chosen = r
                break
            if chosen is None:
                raise RuntimeError("No successful HDBSCAN result found in results pickle.")
            if chosen.get("labels") is None or isinstance(chosen["labels"], str):
                raise RuntimeError("HDBSCAN labels are not stored inline (might be in another run). Re-run or use 'from_fit'.")
            labels_micro = np.asarray(chosen["labels"], dtype=int)

        else:  # from_fit (DEFAULT): run HDBSCAN now using mg_cfg.hdbscan params
            if not HDBSCAN_AVAILABLE:
                raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")

            hb_preproc = hdb_cfg.get("preprocessing", "svd")
            hb_preproc_params = hdb_cfg.get("preprocessing_params", {"n_components": 30})
            hb_params = hdb_cfg.get("params", {
                "min_cluster_size": 10, "min_samples": 5, "cluster_selection_method": "eom",
                "allow_single_cluster": False, "metric": "euclidean", "prediction_data": True
            })

            X_hdb, _ = self.apply_preprocessing(X, hb_preproc, **hb_preproc_params)
            print(f"[HDBSCAN fit] preprocessing={hb_preproc} params={hb_preproc_params} algo_params={hb_params}")
            hdb = HDBSCAN(**hb_params)
            labels_micro = hdb.fit_predict(X_hdb)

            # optionally save the micro labels for later reuse
            if hdb_cfg.get("save_labels_csv", True):
                out_csv = out_dir / "hdbscan_micro_labels_autofit.csv"
                tmp = pd.DataFrame({pid_col: df.get(pid_col, pd.RangeIndex(len(df))),
                                    label_col: labels_micro})
                tmp.to_csv(out_csv, index=False)
                print(f"[HDBSCAN fit] Saved micro labels → {out_csv}")


        # prepare text cols
        text_cols = [c for c in [title_col, abstract_col] if c and c in df.columns]

        summary_rows = []
        label_files, conf_plots, size_plots, umap_plots, kw_files = [], [], [], [], {}

        # single 2D embedding (shared for all plots to save time)
        X2 = self._embed_2d(X, metric="cosine") if umap_enabled else None

        for K in k_targets:
            print(f"\n=== Matched granularity @ K={K} ===")

            # KMeans(K) with cosine
            km_labels, Xk_proc = self._run_kmeans_cosine(X, K, **kmeans_params)
            km_metrics = self._internal_metrics_on(Xk_proc, km_labels)

            # HDBSCAN micro→merge(K)
            hb_labels = self._merge_hdbscan_to_k(X, labels_micro, K_target=K, metric=merge_metric, attach_noise=attach_noise)
            # use same processed space for metrics so comparison is fair
            hb_metrics = self._internal_metrics_on(Xk_proc, hb_labels)

            align = self._alignment_scores(km_labels, hb_labels, exclude_noise=True)
            print(f"KMeans(K={K}) Sil={km_metrics['silhouette_score']:.3f}  "
                  f"HDB→merge(K={K}) Sil={hb_metrics['silhouette_score']:.3f}  "
                  f"AMI={align['AMI']:.3f}  n={align['n_compared']}")

            # Save labels
            lab_path = out_dir / f"labels_matchedK{K}.csv"
            cols = { 'kmeans': km_labels, 'hdbmerge': hb_labels }
            payload = pd.DataFrame({pid_col: df.get(pid_col, pd.RangeIndex(len(df)))})
            for name, v in cols.items():
                payload[f"{name}_K{K}"] = v
            payload.to_csv(lab_path, index=False); label_files.append(str(lab_path))

            # Confusion heatmap
            conf_path = out_dir / f"confusion_K{K}.png"
            self._plot_confusion_heatmap(km_labels, hb_labels, conf_path, f"Matched-K confusion (K={K})")
            conf_plots.append(str(conf_path))

            # Size histograms
            size_km = out_dir / f"size_kmeans_K{K}.png"
            size_hb = out_dir / f"size_hdbmerge_K{K}.png"
            self._plot_size_hist(km_labels, size_km, f"K-means K={K}")
            self._plot_size_hist(hb_labels, size_hb, f"HDBSCAN→merge K={K}")
            size_plots.extend([str(size_km), str(size_hb)])

            # 2-D overlays
            if X2 is not None:
                u_km = out_dir / f"umap_kmeans_K{K}.png"
                u_hb = out_dir / f"umap_hdbmerge_K{K}.png"
                self._plot_2d(X2, km_labels, u_km, f"K-means K={K}")
                self._plot_2d(X2, hb_labels, u_hb, f"HDBSCAN→merge K={K}")
                umap_plots.extend([str(u_km), str(u_hb)])

            # c-TF-IDF keywords + F1@K
            if text_cols and keywords_col and keywords_col in df.columns:
                km_top = self._ctfidf_top_terms(df, km_labels, text_cols=text_cols, top_k=keyword_top_k)
                km_kw_df, km_sum = self._keyword_alignment_f1(df, km_labels, km_top, keywords_col, top_k=keyword_top_k)
                p1 = out_dir / f"keywords_kmeans_K{K}.csv"; p2 = out_dir / f"keywords_kmeans_K{K}_summary.json"
                km_kw_df.to_csv(p1, index=False); json.dump(km_sum, open(p2, "w"), indent=2)

                hb_top = self._ctfidf_top_terms(df, hb_labels, text_cols=text_cols, top_k=keyword_top_k)
                hb_kw_df, hb_sum = self._keyword_alignment_f1(df, hb_labels, hb_top, keywords_col, top_k=keyword_top_k)
                p3 = out_dir / f"keywords_hdbmerge_K{K}.csv"; p4 = out_dir / f"keywords_hdbmerge_K{K}_summary.json"
                hb_kw_df.to_csv(p3, index=False); json.dump(hb_sum, open(p4, "w"), indent=2)
                kw_files[f"K{K}"] = {"kmeans_csv": str(p1), "kmeans_summary": str(p2),
                                     "hdbmerge_csv": str(p3), "hdbmerge_summary": str(p4)}

            summary_rows.append({
                "K": K,
                "kmeans_silhouette": km_metrics.get("silhouette_score", np.nan),
                "kmeans_davies_bouldin": km_metrics.get("davies_bouldin_score", np.nan),
                "kmeans_calinski_harabasz": km_metrics.get("calinski_harabasz_score", np.nan),
                "hdbmerge_silhouette": hb_metrics.get("silhouette_score", np.nan),
                "hdbmerge_davies_bouldin": hb_metrics.get("davies_bouldin_score", np.nan),
                "hdbmerge_calinski_harabasz": hb_metrics.get("calinski_harabasz_score", np.nan),
                "AMI": align["AMI"], "NMI": align["NMI"], "V": align["V"], "n_compared": align["n_compared"]
            })

        # summary CSV
        summary_path = out_dir / "matchedK_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        return {
            "summary_csv": str(summary_path),
            "label_files": label_files,
            "confusion_plots": conf_plots,
            "size_plots": size_plots,
            "umap_plots": umap_plots,
            "keyword_files": kw_files
        }


    def generate_seed_heatmaps_kmeans(
        self,
        X: np.ndarray,
        k_values: List[int],
        n_seeds: int = 5,
        preprocessing: str = "l2_normalize",
        kmeans_params: Optional[Dict[str, Any]] = None,
        save_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Seed robustness for K-means: build AMI heatmaps across different random seeds.
        Saves one heatmap per K. Returns list of file paths.
        """
        if save_dir is None:
            save_dir = self.results_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess once
        X_proc, _ = self.apply_preprocessing(X, preprocessing)

        # Default KMeans params
        base_params = dict(n_init=10, max_iter=500, tol=1e-4, init="k-means++")
        if kmeans_params:
            base_params.update(kmeans_params)

        heatmap_paths = []
        for k in sorted({kv for kv in k_values if kv >= 2}):
            label_list = []
            seed_list = []
            for i in range(n_seeds):
                seed = self.random_state + i
                km = KMeans(n_clusters=k, random_state=seed, **base_params)
                labels = km.fit_predict(X_proc)
                label_list.append(labels)
                seed_list.append(seed)

            # Pairwise AMI matrix
            S = len(label_list)
            ami_mat = np.zeros((S, S))
            for i in range(S):
                for j in range(S):
                    ami_mat[i, j] = adjusted_mutual_info_score(label_list[i], label_list[j])

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(4 + 0.5*S, 3 + 0.3*S))
            sns.heatmap(
                ami_mat,
                vmin=0.0, vmax=1.0, cmap="viridis",
                annot=True, fmt=".2f", square=True, ax=ax,
                cbar_kws={"label": "AMI"}
            )
            ax.set_title(f"K-means seed robustness (K={k})")
            ax.set_xlabel("Seed index")
            ax.set_ylabel("Seed index")
            ax.set_xticks(np.arange(S)+0.5)
            ax.set_xticklabels([str(s) for s in seed_list], rotation=45, ha="right")
            ax.set_yticks(np.arange(S)+0.5)
            ax.set_yticklabels([str(s) for s in seed_list], rotation=0)

            out = save_dir / f"kmeans_seed_ami_heatmap_K{k}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved K-means AMI seed heatmap (K={k}) → {out}")
            heatmap_paths.append(out)

        return heatmap_paths

    def generate_seed_heatmap_hdbscan(
        self,
        X: np.ndarray,
        preprocessing: str = "svd",
        preprocessing_params: Optional[Dict[str, Any]] = None,
        hdbscan_params: Optional[Dict[str, Any]] = None,
        n_seeds: int = 5,
        shuffle_input: bool = True,
        save_dir: Optional[Path] = None,
        tag: str = "hdbscan",
    ) -> Optional[Path]:
        """
        Seed robustness proxy for HDBSCAN: fit multiple times with row shuffles (or tiny jitter),
        compute pairwise AMI over the full label vectors, and plot a single heatmap.
        """
        if not HDBSCAN_AVAILABLE:
            print("HDBSCAN not available; skipping HDBSCAN seed heatmap.")
            return None

        if save_dir is None:
            save_dir = self.results_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        preprocessing_params = preprocessing_params or {"n_components": 30}
        X_proc, _ = self.apply_preprocessing(X, preprocessing, **preprocessing_params)

        # Default HDBSCAN params (good practical choice for your bag-of-paper)
        default_params = dict(
            min_cluster_size=10,
            min_samples=5,
            cluster_selection_method="eom",
            allow_single_cluster=False,
            metric="euclidean",
            prediction_data=True
        )
        if hdbscan_params:
            default_params.update(hdbscan_params)

        labels_list = []
        seeds = []
        n = X_proc.shape[0]

        for i in range(n_seeds):
            seed = self.random_state + i
            rng = np.random.default_rng(seed)
            if shuffle_input:
                perm = rng.permutation(n)
                X_seed = X_proc[perm]
            else:
                # tiny jitter to break ties while preserving order
                X_seed = X_proc + 1e-7 * rng.standard_normal(size=X_proc.shape)

            clusterer = HDBSCAN(**default_params)
            order = np.lexsort([X_proc[:,0]])  # or use a stable key like paper_id
            X_proc = X_proc[order]
            # labels = clusterer.fit_predict(X_proc)
            labels_seed = clusterer.fit_predict(X_seed)

            # unshuffle to original order if we shuffled
            if shuffle_input:
                inv = np.empty_like(perm)
                inv[perm] = np.arange(n)
                labels_unshuffled = np.empty_like(labels_seed)
                labels_unshuffled[inv] = labels_seed
                labels_list.append(labels_unshuffled)
            else:
                labels_list.append(labels_seed)
            seeds.append(seed)

        # Pairwise AMI matrix
        S = len(labels_list)
        ami_mat = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                ami_mat[i, j] = adjusted_mutual_info_score(labels_list[i], labels_list[j])

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(4 + 0.5*S, 3 + 0.3*S))
        sns.heatmap(
            ami_mat, vmin=0.0, vmax=1.0, cmap="viridis",
            annot=True, fmt=".2f", square=True, ax=ax,
            cbar_kws={"label": "AMI"}
        )
        ax.set_title(f"HDBSCAN seed robustness (n={n_seeds}, preproc={preprocessing})")
        ax.set_xlabel("Seed index")
        ax.set_ylabel("Seed index")
        ax.set_xticks(np.arange(S)+0.5)
        ax.set_xticklabels([str(s) for s in seeds], rotation=45, ha="right")
        ax.set_yticks(np.arange(S)+0.5)
        ax.set_yticklabels([str(s) for s in seeds], rotation=0)

        out = save_dir / f"{tag}_seed_ami_heatmap.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved HDBSCAN AMI seed heatmap → {out}")
        return out

        
    def load_data(self, data_path: Path, vector_prefix: str = 'cluster_') -> Tuple[np.ndarray, pd.DataFrame]:
        """Load paper clustering data with validation"""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Extract feature vectors
        vector_cols = [col for col in df.columns if col.startswith(vector_prefix)]
        X = df[vector_cols].values
        
        # Data validation
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Feature sparsity: {(X == 0).mean():.3f}")
        print(f"Data shape check: min={X.min():.3f}, max={X.max():.3f}, mean={X.mean():.3f}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: Data contains NaN or infinite values. Cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, df
    
    def compute_gap_statistic(self, X: np.ndarray, k_range: range, 
                            n_references: int = 10, algorithm: str = 'kmeans') -> Dict:
        """
        Compute Gap Statistic for optimal k selection.
        
        Gap(k) = E*[log(Wk)] - log(Wk)
        where E*[log(Wk)] is the expected value under null reference distribution
        """
        print("Computing Gap Statistic...")
        gaps = []
        s_k = []
        Wks = []
        Wkbs = []
        
        for k in tqdm(k_range, desc="Gap Statistic"):
            # Cluster original data
            if algorithm == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = clusterer.fit_predict(X)
                Wk = self._compute_inertia(X, labels, clusterer.cluster_centers_)
            else:
                raise NotImplementedError(f"Gap statistic not implemented for {algorithm}")
            
            # Generate reference datasets and compute their dispersions
            BWkbs = []
            for _ in range(n_references):
                # Generate reference data (uniform distribution in data bounding box)
                X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
                ref_clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=3)
                ref_labels = ref_clusterer.fit_predict(X_ref)
                BWkbs.append(self._compute_inertia(X_ref, ref_labels, ref_clusterer.cluster_centers_))
            
            Wks.append(Wk)
            Wkbs.append(np.mean(BWkbs))
            
            # Compute gap
            gap = np.log(np.mean(BWkbs)) - np.log(Wk)
            gaps.append(gap)
            
            # Compute standard deviation
            sdk = np.sqrt(1 + 1/n_references) * np.std(np.log(BWkbs))
            s_k.append(sdk)
        
        # Find optimal k using Gap Statistic criterion
        # Choose smallest k such that Gap(k) >= Gap(k+1) - s(k+1)
        gaps = np.array(gaps)
        s_k = np.array(s_k)
        
        optimal_k = k_range[0]
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - s_k[i + 1]:
                optimal_k = k_range[i]
                break
        
        return {
            'k_values': list(k_range),
            'gaps': gaps.tolist(),
            'std_errors': s_k.tolist(),
            'Wks': Wks,
            'Wkbs': Wkbs,
            'optimal_k': optimal_k,
            'method': 'gap_statistic'
        }
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """Compute within-cluster sum of squares"""
        inertia = 0
        for k, center in enumerate(centers):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - center) ** 2)
        return inertia
    
    def elbow_method_analysis(self, X: np.ndarray, k_range: range, 
                             algorithm: str = 'kmeans', preprocessing: str = 'standard') -> Dict:
        """
        Comprehensive Elbow Method with automatic knee detection.
        Uses KneeLocator for automatic elbow point detection.
        """
        print("Performing Elbow Method Analysis...")
        
        # Preprocess data
        X_processed, _ = self.apply_preprocessing(X, preprocessing)
        
        inertias = []
        silhouettes = []
        calinskis = []
        davies_bouldins = []
        
        for k in tqdm(k_range, desc="Elbow Analysis"):
            if algorithm == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            elif algorithm == 'gmm':
                clusterer = GaussianMixture(n_components=k, random_state=self.random_state)
            else:
                raise NotImplementedError(f"Elbow method not implemented for {algorithm}")
            
            labels = clusterer.fit_predict(X_processed)
            
            # Compute metrics
            if hasattr(clusterer, 'inertia_'):
                inertias.append(clusterer.inertia_)
            else:
                # Compute manually for GMM
                if hasattr(clusterer, 'means_'):
                    centers = clusterer.means_
                else:
                    centers = np.array([X_processed[labels == i].mean(axis=0) 
                                       for i in range(k)])
                inertia = sum([np.sum((X_processed[labels == i] - centers[i]) ** 2) 
                              for i in range(k)])
                inertias.append(inertia)
            
            # Other metrics
            if len(np.unique(labels)) > 1:
                silhouettes.append(silhouette_score(X_processed, labels))
                calinskis.append(calinski_harabasz_score(X_processed, labels))
                davies_bouldins.append(davies_bouldin_score(X_processed, labels))
            else:
                silhouettes.append(-1)
                calinskis.append(0)
                davies_bouldins.append(float('inf'))
        
        # Detect elbow point using KneeLocator
        kneedle = KneeLocator(list(k_range), inertias, S=1.0, curve='convex', direction='decreasing')
        optimal_k_elbow = kneedle.knee if kneedle.knee else k_range[len(k_range)//2]
        
        # Find optimal k based on silhouette
        optimal_k_silhouette = k_range[np.argmax(silhouettes)]
        
        # Find optimal k based on Calinski-Harabasz
        optimal_k_calinski = k_range[np.argmax(calinskis)]
        
        return {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouettes': silhouettes,
            'calinski_harabasz': calinskis,
            'davies_bouldin': davies_bouldins,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'optimal_k_calinski': optimal_k_calinski,
            'method': 'elbow_analysis'
        }
    
    def stability_analysis(self, X: np.ndarray, k_values: List[int], 
                          n_iterations: int = 30, subsample_ratio: float = 0.8) -> Dict:
        """
        Stability analysis using bootstrap resampling.
        Measures how stable clustering is across different subsamples.
        """
        print("Performing Stability Analysis...")
        stability_scores = defaultdict(list)
        
        n_samples = X.shape[0]
        subsample_size = int(n_samples * subsample_ratio)
        
        for k in tqdm(k_values, desc="Stability Analysis"):
            ari_scores = []
            ami_scores = []
            
            # Reference clustering on full data
            ref_clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            ref_labels = ref_clusterer.fit_predict(X)
            
            for iteration in range(n_iterations):
                # Create bootstrap sample
                indices = np.random.choice(n_samples, subsample_size, replace=True)
                X_boot = X[indices]
                
                # Cluster bootstrap sample
                boot_clusterer = KMeans(n_clusters=k, random_state=self.random_state + iteration, n_init=5)
                boot_labels = boot_clusterer.fit_predict(X_boot)
                
                # Predict on full data
                full_boot_labels = boot_clusterer.predict(X)
                
                # Compare with reference
                ari = adjusted_rand_score(ref_labels, full_boot_labels)
                ami = adjusted_mutual_info_score(ref_labels, full_boot_labels)
                
                ari_scores.append(ari)
                ami_scores.append(ami)
            
            stability_scores[k] = {
                'ari_mean': np.mean(ari_scores),
                'ari_std': np.std(ari_scores),
                'ami_mean': np.mean(ami_scores),
                'ami_std': np.std(ami_scores),
                'stability_score': np.mean(ari_scores) - np.std(ari_scores)  # Combined metric
            }
        
        # Find most stable k
        optimal_k = max(stability_scores.keys(), 
                       key=lambda k: stability_scores[k]['stability_score'])
        
        return {
            'k_values': list(stability_scores.keys()),
            'stability_scores': dict(stability_scores),
            'optimal_k': optimal_k,
            'method': 'stability_analysis'
        }
    
    def consensus_clustering(self, X: np.ndarray, k: int, n_iterations: int = 100,
                           subsample_ratio: float = 0.8) -> Tuple[np.ndarray, float]:
        """
        Consensus clustering to create robust cluster assignments.
        Returns consensus labels and consensus score.
        """
        print(f"Performing Consensus Clustering for k={k}...")
        n_samples = X.shape[0]
        
        # Initialize co-association matrix
        coassoc_matrix = np.zeros((n_samples, n_samples))
        
        for iteration in tqdm(range(n_iterations), desc="Consensus iterations"):
            # Subsample data
            if subsample_ratio < 1.0:
                indices = np.random.choice(n_samples, int(n_samples * subsample_ratio), replace=False)
                X_sub = X[indices]
            else:
                indices = np.arange(n_samples)
                X_sub = X
            
            # Cluster subsample
            clusterer = KMeans(n_clusters=k, random_state=self.random_state + iteration, n_init=3)
            labels_sub = clusterer.fit_predict(X_sub)
            
            # Update co-association matrix
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if labels_sub[i] == labels_sub[j]:
                        coassoc_matrix[indices[i], indices[j]] += 1
                        coassoc_matrix[indices[j], indices[i]] += 1
        
        # Normalize co-association matrix
        coassoc_matrix /= n_iterations
        
        # Final clustering on consensus matrix
        # Convert co-association to distance
        distance_matrix = 1 - coassoc_matrix
        
        # Use hierarchical clustering on consensus matrix
        final_clusterer = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        consensus_labels = final_clusterer.fit_predict(distance_matrix)
        
        # Compute consensus score (average co-association within clusters)
        consensus_score = 0
        for cluster_id in range(k):
            cluster_indices = np.where(consensus_labels == cluster_id)[0]
            if len(cluster_indices) > 1:
                cluster_coassoc = coassoc_matrix[np.ix_(cluster_indices, cluster_indices)]
                consensus_score += np.mean(cluster_coassoc[np.triu_indices_from(cluster_coassoc, k=1)])
        consensus_score /= k
        
        return consensus_labels, consensus_score
    
    def prediction_strength(self, X: np.ndarray, k_range: range, n_iterations: int = 20) -> Dict:
        """
        Prediction Strength method for determining optimal k.
        Based on Tibshirani & Walther (2005).
        """
        print("Computing Prediction Strength...")
        prediction_strengths = []
        
        for k in tqdm(k_range, desc="Prediction Strength"):
            ps_scores = []
            
            for _ in range(n_iterations):
                # Split data into two halves
                n = X.shape[0]
                perm = np.random.permutation(n)
                split = n // 2
                
                X_train = X[perm[:split]]
                X_test = X[perm[split:]]
                
                # Cluster both halves
                clusterer_train = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                labels_train = clusterer_train.fit_predict(X_train)
                
                clusterer_test = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                labels_test = clusterer_test.fit_predict(X_test)
                
                # Predict test labels using train model
                predicted_test = clusterer_train.predict(X_test)
                
                # Compute prediction strength
                ps = self._compute_prediction_strength(labels_test, predicted_test)
                ps_scores.append(ps)
            
            prediction_strengths.append(np.mean(ps_scores))
        
        # Optimal k is largest k with prediction strength > threshold (typically 0.8-0.9)
        threshold = 0.8
        optimal_k = k_range[0]
        for i, (k, ps) in enumerate(zip(k_range, prediction_strengths)):
            if ps >= threshold:
                optimal_k = k
        
        return {
            'k_values': list(k_range),
            'prediction_strengths': prediction_strengths,
            'optimal_k': optimal_k,
            'threshold': threshold,
            'method': 'prediction_strength'
        }
    
    def _compute_prediction_strength(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
        """Compute prediction strength between two label sets"""
        n = len(true_labels)
        strength = 0
        
        for cluster in np.unique(true_labels):
            cluster_indices = np.where(true_labels == cluster)[0]
            if len(cluster_indices) > 1:
                # Check if pairs in same cluster in true are also in same cluster in predicted
                same_cluster_predicted = 0
                total_pairs = 0
                
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        if predicted_labels[cluster_indices[i]] == predicted_labels[cluster_indices[j]]:
                            same_cluster_predicted += 1
                        total_pairs += 1
                
                if total_pairs > 0:
                    strength += same_cluster_predicted / total_pairs
        
        return strength / len(np.unique(true_labels))
    
    def feature_importance_analysis(self, X: np.ndarray, labels: np.ndarray, 
                                   feature_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze feature importance for clustering with statistical significance.
        Uses ANOVA F-statistic and mutual information.
        """
        print("Analyzing Feature Importance...")
        
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # ANOVA F-statistic for each feature
        f_scores, p_values = f_classif(X, labels)
        
        # Mutual information
        mi_scores = mutual_info_classif(X, labels, random_state=self.random_state)
        
        # Variance ratio (between-cluster / within-cluster variance)
        variance_ratios = []
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Between-cluster variance
            cluster_means = [feature_values[labels == k].mean() 
                           for k in np.unique(labels) if k != -1]
            global_mean = feature_values.mean()
            between_var = sum([(m - global_mean) ** 2 for m in cluster_means])
            
            # Within-cluster variance
            within_var = sum([feature_values[labels == k].var() 
                            for k in np.unique(labels) if k != -1])
            
            if within_var > 0:
                variance_ratios.append(between_var / within_var)
            else:
                variance_ratios.append(0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'f_score': f_scores,
            'p_value': p_values,
            'mi_score': mi_scores,
            'variance_ratio': variance_ratios
        })
        
        # Add Bonferroni corrected p-values
        importance_df['p_value_corrected'] = importance_df['p_value'] * n_features
        importance_df['p_value_corrected'] = importance_df['p_value_corrected'].clip(upper=1.0)
        
        # Mark significant features
        importance_df['significant_005'] = importance_df['p_value_corrected'] < 0.05
        importance_df['significant_001'] = importance_df['p_value_corrected'] < 0.01
        
        # Sort by importance
        importance_df = importance_df.sort_values('f_score', ascending=False)
        
        return {
            'importance_df': importance_df,
            'top_features': importance_df.head(20).to_dict('records'),
            'n_significant_005': importance_df['significant_005'].sum(),
            'n_significant_001': importance_df['significant_001'].sum()
        }
    
    def apply_preprocessing(self, X: np.ndarray, method: str, **kwargs) -> Tuple[np.ndarray, Any]:
        """Apply preprocessing to feature matrix with validation"""
        
        # Check for NaN/Inf before preprocessing
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if method == 'none':
            return X, None
        elif method == 'standard':
            scaler = StandardScaler()
            return scaler.fit_transform(X), scaler
        elif method == 'minmax':
            scaler = MinMaxScaler()
            return scaler.fit_transform(X), scaler
        elif method == 'robust':
            scaler = RobustScaler()
            return scaler.fit_transform(X), scaler
        elif method == 'l2_normalize':
            scaler = Normalizer(norm='l2')
            return scaler.fit_transform(X), scaler
        elif method == 'l1_normalize':
            scaler = Normalizer(norm='l1')
            return scaler.fit_transform(X), scaler
        elif method == 'tfidf':
            transformer = TfidfTransformer()
            return transformer.fit_transform(X).toarray(), transformer
        elif method == 'pca':
            n_components = kwargs.get('n_components', min(50, X.shape[1] - 1))
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            return reducer.fit_transform(X), reducer
        elif method == 'svd':
            n_components = kwargs.get('n_components', min(50, X.shape[1] - 1))
            reducer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            return reducer.fit_transform(X), reducer
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def run_k_optimization(self, X: np.ndarray, k_range: range = range(2, 21),
                         methods: List[str] = None) -> Dict:
        """
        Run comprehensive k-optimization using multiple methods.
        """
        if methods is None:
            methods = ['elbow', 'gap', 'silhouette', 'stability', 'prediction_strength']
        
        print(f"Running K-Optimization with methods: {methods}")
        print("="*80)
        
        results = {}
        
        # 1. Elbow Method with multiple metrics
        if 'elbow' in methods:
            results['elbow'] = self.elbow_method_analysis(X, k_range)
        
        # 2. Gap Statistic
        if 'gap' in methods:
            results['gap'] = self.compute_gap_statistic(X, k_range)
        
        # 3. Silhouette Analysis
        if 'silhouette' in methods:
            results['silhouette'] = self.silhouette_analysis(X, k_range)
        
        # 4. Stability Analysis
        if 'stability' in methods:
            results['stability'] = self.stability_analysis(X, list(k_range))
        
        # 5. Prediction Strength
        if 'prediction_strength' in methods:
            results['prediction_strength'] = self.prediction_strength(X, k_range)
        
        # Voting mechanism for optimal k
        optimal_k_votes = []
        for method, result in results.items():
            if 'optimal_k' in result:
                optimal_k_votes.append(result['optimal_k'])
            elif 'optimal_k_silhouette' in result:
                optimal_k_votes.append(result['optimal_k_silhouette'])
        
        # Consensus optimal k (mode of all methods)
        if optimal_k_votes:
            from statistics import mode, StatisticsError
            try:
                consensus_k = mode(optimal_k_votes)
            except StatisticsError:
                # If no clear mode, use median
                consensus_k = int(np.median(optimal_k_votes))
        else:
            consensus_k = 8  # Default
        
        results['consensus'] = {
            'optimal_k': consensus_k,
            'votes': optimal_k_votes,
            'method_agreement': len(set(optimal_k_votes)) == 1
        }
        
        self.k_optimization_results = results
        return results
    
    def silhouette_analysis(self, X: np.ndarray, k_range: range) -> Dict:
        """
        Detailed silhouette analysis for each k value.
        """
        print("Performing Silhouette Analysis...")
        
        silhouette_scores = []
        silhouette_samples_all = []
        
        for k in tqdm(k_range, desc="Silhouette Analysis"):
            clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = clusterer.fit_predict(X)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                samples = silhouette_samples(X, labels)
            else:
                score = -1
                samples = np.array([-1] * len(labels))
            
            silhouette_scores.append(score)
            silhouette_samples_all.append({
                'k': k,
                'samples': samples,
                'labels': labels
            })
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return {
            'k_values': list(k_range),
            'silhouette_scores': silhouette_scores,
            'silhouette_samples': silhouette_samples_all,
            'optimal_k': optimal_k,
            'method': 'silhouette_analysis'
        }
    
    def create_k_optimization_visualizations(self, save_path: Optional[Path] = None):
        """
        Create comprehensive visualizations for k-optimization results.
        """
        if not self.k_optimization_results:
            print("No k-optimization results to visualize")
            return
        
        print("Creating K-Optimization Visualizations...")
        
        # Create figure with subplots
        n_methods = len(self.k_optimization_results) - 1  # Exclude 'consensus'
        fig = plt.figure(figsize=(20, 4 * ((n_methods + 2) // 3)))
        
        plot_idx = 1
        
        # 1. Elbow Method Plot
        if 'elbow' in self.k_optimization_results:
            ax = plt.subplot((n_methods + 2) // 3, 3, plot_idx)
            result = self.k_optimization_results['elbow']
            
            ax2 = ax.twinx()
            ax.plot(result['k_values'], result['inertias'], 'b-o', label='Inertia')
            ax2.plot(result['k_values'], result['silhouettes'], 'r-s', label='Silhouette')
            
            ax.axvline(x=result['optimal_k_elbow'], color='g', linestyle='--', 
                      label=f'Optimal k={result["optimal_k_elbow"]}')
            
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia', color='b')
            ax2.set_ylabel('Silhouette Score', color='r')
            ax.set_title('Elbow Method Analysis')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 2. Gap Statistic Plot
        if 'gap' in self.k_optimization_results:
            ax = plt.subplot((n_methods + 2) // 3, 3, plot_idx)
            result = self.k_optimization_results['gap']
            
            gaps = np.array(result['gaps'])
            std_errors = np.array(result['std_errors'])
            
            ax.errorbar(result['k_values'], gaps, yerr=std_errors, 
                       marker='o', capsize=5, capthick=2)
            ax.axvline(x=result['optimal_k'], color='r', linestyle='--',
                      label=f'Optimal k={result["optimal_k"]}')
            
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Gap Statistic')
            ax.set_title('Gap Statistic Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 3. Stability Analysis Plot
        if 'stability' in self.k_optimization_results:
            ax = plt.subplot((n_methods + 2) // 3, 3, plot_idx)
            result = self.k_optimization_results['stability']
            
            k_values = result['k_values']
            stability_scores = [result['stability_scores'][k]['stability_score'] for k in k_values]
            ari_means = [result['stability_scores'][k]['ari_mean'] for k in k_values]
            ari_stds = [result['stability_scores'][k]['ari_std'] for k in k_values]
            
            ax.errorbar(k_values, ari_means, yerr=ari_stds, 
                       marker='o', capsize=5, label='ARI ± std')
            ax.plot(k_values, stability_scores, 'r-s', label='Stability Score')
            ax.axvline(x=result['optimal_k'], color='g', linestyle='--',
                      label=f'Optimal k={result["optimal_k"]}')
            
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Stability Metrics')
            ax.set_title('Stability Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 4. Prediction Strength Plot
        if 'prediction_strength' in self.k_optimization_results:
            ax = plt.subplot((n_methods + 2) // 3, 3, plot_idx)
            result = self.k_optimization_results['prediction_strength']
            
            ax.plot(result['k_values'], result['prediction_strengths'], 'b-o')
            ax.axhline(y=result['threshold'], color='r', linestyle='--', 
                      label=f'Threshold={result["threshold"]}')
            ax.axvline(x=result['optimal_k'], color='g', linestyle='--',
                      label=f'Optimal k={result["optimal_k"]}')
            
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Prediction Strength')
            ax.set_title('Prediction Strength Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 5. Consensus Plot - Summary of all methods
        if 'consensus' in self.k_optimization_results:
            ax = plt.subplot((n_methods + 2) // 3, 3, plot_idx)
            
            # Create a vote distribution plot
            votes = self.k_optimization_results['consensus']['votes']
            unique_votes, counts = np.unique(votes, return_counts=True)
            
            bars = ax.bar(unique_votes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            consensus_k = self.k_optimization_results['consensus']['optimal_k']
            
            # Highlight consensus k
            for i, (vote, count) in enumerate(zip(unique_votes, counts)):
                if vote == consensus_k:
                    bars[i].set_color('gold')
                    bars[i].set_edgecolor('orange')
            
            ax.set_xlabel('k value')
            ax.set_ylabel('Number of Methods Voting')
            ax.set_title(f'Consensus Analysis (Optimal k={consensus_k})')
            ax.set_xticks(unique_votes)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add text annotations
            for vote, count in zip(unique_votes, counts):
                ax.text(vote, count, str(count), ha='center', va='bottom')
        
        plt.suptitle('K-Optimization Analysis Results', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / 'k_optimization_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"K-optimization visualizations saved to: {save_path}")
    
    def generate_latex_tables(self, results: List[Dict], k_optimization_results: Dict):
        """
        Generate LaTeX tables for academic papers.
        """
        print("Generating LaTeX tables...")
        
        latex_output = []
        
        # Table 1: K-Optimization Results
        latex_output.append("% Table 1: Optimal K Selection Results")
        latex_output.append("\\begin{table}[h]")
        latex_output.append("\\centering")
        latex_output.append("\\caption{Optimal Number of Clusters by Different Methods}")
        latex_output.append("\\begin{tabular}{lcc}")
        latex_output.append("\\hline")
        latex_output.append("Method & Optimal $k$ & Criterion \\\\")
        latex_output.append("\\hline")
        
        for method, result in k_optimization_results.items():
            if method != 'consensus':
                if 'optimal_k' in result:
                    optimal_k = result['optimal_k']
                    criterion = result.get('method', method).replace('_', ' ').title()
                    latex_output.append(f"{criterion} & {optimal_k} & - \\\\")
        
        if 'consensus' in k_optimization_results:
            consensus_k = k_optimization_results['consensus']['optimal_k']
            latex_output.append("\\hline")
            latex_output.append(f"\\textbf{{Consensus}} & \\textbf{{{consensus_k}}} & Mode of methods \\\\")
        
        latex_output.append("\\hline")
        latex_output.append("\\end{tabular}")
        latex_output.append("\\end{table}")
        latex_output.append("")
        
        # Table 2: Performance Metrics
        if results:
            successful_results = [r for r in results if r.get('success', False)]
            if successful_results:
                latex_output.append("% Table 2: Clustering Performance Metrics")
                latex_output.append("\\begin{table}[h]")
                latex_output.append("\\centering")
                latex_output.append("\\caption{Clustering Algorithm Performance Comparison}")
                latex_output.append("\\begin{tabular}{lccccc}")
                latex_output.append("\\hline")
                latex_output.append("Algorithm & Silhouette & Calinski-H. & Davies-B. & Time (s) \\\\")
                latex_output.append("\\hline")
                
                # Sort by silhouette score
                sorted_results = sorted(successful_results, 
                                      key=lambda x: x['metrics'].get('silhouette_score', -1), 
                                      reverse=True)[:10]  # Top 10
                
                for result in sorted_results:
                    alg = result['algorithm']
                    sil = result['metrics'].get('silhouette_score', -1)
                    cal = result['metrics'].get('calinski_harabasz_score', 0)
                    dav = result['metrics'].get('davies_bouldin_score', float('inf'))
                    time = result.get('total_time', 0)
                    
                    latex_output.append(f"{alg} & {sil:.3f} & {cal:.1f} & {dav:.3f} & {time:.2f} \\\\")
                
                latex_output.append("\\hline")
                latex_output.append("\\end{tabular}")
                latex_output.append("\\end{table}")
        
        # Save LaTeX tables
        latex_file = self.results_dir / 'latex_tables.tex'
        with open(latex_file, 'w') as f:
            f.write('\n'.join(latex_output))
        
        print(f"LaTeX tables saved to: {latex_file}")
        return latex_output
    
    # def run_complete_analysis(self, X: np.ndarray, df: pd.DataFrame, config: Dict) -> Dict:
    #     """
    #     Run complete analysis pipeline with k-optimization and evaluation.
    #     """
    #     print("="*80)
    #     print("STARTING COMPLETE CLUSTERING ANALYSIS")
    #     print("="*80)
        
    #     # Phase 1: K-Optimization
    #     print("\nPHASE 1: OPTIMAL K SELECTION")
    #     print("-"*40)
    #     k_range = range(config.get('k_min', 2), config.get('k_max', 21))
    #     k_methods = config.get('k_methods', ['elbow', 'gap', 'silhouette', 'stability'])
        
    #     k_optimization_results = self.run_k_optimization(X, k_range, k_methods)
    #     optimal_k = k_optimization_results['consensus']['optimal_k']
        
    #     print(f"\n✓ Consensus Optimal K: {optimal_k}")
    #     print(f"  Method votes: {k_optimization_results['consensus']['votes']}")
        
    #     # Phase 2: Stability Analysis for Optimal K
    #     print("\nPHASE 2: STABILITY ANALYSIS")
    #     print("-"*40)
    #     stability_result = self.stability_analysis(X, [optimal_k - 1, optimal_k, optimal_k + 1])
    #     print(f"✓ Stability Score for k={optimal_k}: {stability_result['stability_scores'][optimal_k]['stability_score']:.3f}")
        
    #     # Phase 3: Consensus Clustering
    #     print("\nPHASE 3: CONSENSUS CLUSTERING")
    #     print("-"*40)
    #     consensus_labels, consensus_score = self.consensus_clustering(X, optimal_k)
    #     print(f"✓ Consensus Score: {consensus_score:.3f}")
        
    #     # Phase 4: Feature Importance
    #     print("\nPHASE 4: FEATURE IMPORTANCE ANALYSIS")
    #     print("-"*40)
    #     feature_importance = self.feature_importance_analysis(X, consensus_labels)
    #     print(f"✓ Significant features (p<0.05): {feature_importance['n_significant_005']}")
    #     print(f"✓ Highly significant features (p<0.01): {feature_importance['n_significant_001']}")
        
    #     # Phase 5: Run experiments with optimal k
    #     print("\nPHASE 5: ALGORITHM COMPARISON")
    #     print("-"*40)
        
    #     # Update experiment configs with optimal k
    #     updated_experiments = []
    #     for exp in config.get('experiments', []):
    #         exp_copy = exp.copy()
    #         if 'params' in exp_copy:
    #             if 'n_clusters' in exp_copy['params']:
    #                 exp_copy['params']['n_clusters'] = optimal_k
    #             elif 'n_components' in exp_copy['params']:
    #                 exp_copy['params']['n_components'] = optimal_k
    #         updated_experiments.append(exp_copy)
        
    #     # Run experiments
    #     results = self.run_experiments(X, df, updated_experiments)
        
    #     # Phase 6: Statistical Testing
    #     print("\nPHASE 6: STATISTICAL SIGNIFICANCE TESTING")
    #     print("-"*40)
    #     statistical_results = self.statistical_significance_testing(results)
        
    #     # Create visualizations
    #     self.create_k_optimization_visualizations()
    #     self.create_visualizations(results, {})
        
    #     # Generate LaTeX tables
    #     self.generate_latex_tables(results, k_optimization_results)
        
    #     # Generate comprehensive report
    #     report = self.generate_academic_report(
    #         results, k_optimization_results, stability_result, 
    #         feature_importance, statistical_results, consensus_score
    #     )
        
    #     return {
    #         'optimal_k': optimal_k,
    #         'k_optimization': k_optimization_results,
    #         'stability': stability_result,
    #         'consensus_score': consensus_score,
    #         'consensus_labels': consensus_labels,
    #         'feature_importance': feature_importance,
    #         'experiment_results': results,
    #         'statistical_results': statistical_results
    #     }

    def run_complete_analysis(self, X: np.ndarray, df: pd.DataFrame, config: Dict) -> Dict:
        """
        Run complete analysis pipeline with k-optimization and evaluation.
        (Consensus clustering is skipped to avoid OOM; we generate seed AMI heatmaps instead.)
        """
        print("="*80)
        print("STARTING COMPLETE CLUSTERING ANALYSIS")
        print("="*80)

        # Phase 1: K-Optimization
        print("\nPHASE 1: OPTIMAL K SELECTION")
        print("-"*40)
        k_min = config.get('k_min', config.get('k_optimization', {}).get('k_min', 2))
        k_max = config.get('k_max', config.get('k_optimization', {}).get('k_max', 21))
        k_methods = config.get('k_methods', config.get('k_optimization', {}).get(
            'k_methods', ['elbow', 'gap', 'silhouette', 'stability']
        ))
        k_range = range(k_min, k_max + 1)

        k_optimization_results = self.run_k_optimization(X, k_range, k_methods)
        optimal_k = 2

        print(f"\n✓ Consensus Optimal K: {optimal_k}")
        print(f"  Method votes: {k_optimization_results['consensus']['votes']}")

        # Phase 2: Stability Analysis around K*
        # print("\nPHASE 2: STABILITY ANALYSIS (around K*)")
        # print("-"*40)
        # ks_for_stability = [k for k in [max(2, optimal_k-1), optimal_k, optimal_k+1] if k >= 2]
        # stability_result = self.stability_analysis(X, ks_for_stability)
        # if optimal_k in stability_result['stability_scores']:
        #     s = stability_result['stability_scores'][optimal_k]
        #     print(f"✓ Stability Score for k={optimal_k}: {s['stability_score']:.3f}")

        # PHASE 3: Seed robustness heatmaps (NO CONSENSUS)
        # print("\nPHASE 3: SEED ROBUSTNESS HEATMAPS (skipping consensus to avoid OOM)")
        # print("-"*40)
        # seed_cfg = config.get("seed_heatmaps", {})
        # n_seeds = seed_cfg.get("n_seeds", 5)

        # 3a) K-means AMI seed heatmaps at K*−1, K*, K*+1 (cosine + L2)
        # kmeans_preproc = seed_cfg.get("kmeans", {}).get("preprocessing", "l2_normalize")
        # kmeans_params = seed_cfg.get("kmeans", {}).get("params", {"n_init": 10, "max_iter": 500, "tol": 1e-4, "init": "k-means++"})
        # k_heatmaps = self.generate_seed_heatmaps_kmeans(
        #     X,
        #     k_values=ks_for_stability,
        #     n_seeds=n_seeds,
        #     preprocessing=kmeans_preproc,
        #     kmeans_params=kmeans_params,
        #     save_dir=self.results_dir,
        # )

        # 3b) HDBSCAN AMI seed heatmap (SVD-30 + Euclidean by default)
        # hdb_cfg = seed_cfg.get("hdbscan", {})
        # hdb_preproc = hdb_cfg.get("preprocessing", "svd")
        # hdb_preproc_params = hdb_cfg.get("preprocessing_params", {"n_components": 30})
        # hdb_params = hdb_cfg.get("params", {
        #     "min_cluster_size": 10, "min_samples": 5, "cluster_selection_method": "eom",
        #     "allow_single_cluster": False, "metric": "euclidean", "prediction_data": True
        # })
        # hdb_heatmap = self.generate_seed_heatmap_hdbscan(
        #     X,
        #     preprocessing=hdb_preproc,
        #     preprocessing_params=hdb_preproc_params,
        #     hdbscan_params=hdb_params,
        #     n_seeds=n_seeds,
        #     shuffle_input=True,
        #     save_dir=self.results_dir,
        #     tag="hdbscan",
        # )

        # PHASE 4: Matched-Granularity (KMeans(K) vs HDBSCAN→merge(K)) for Pipeline-2 paper level
        # mg_cfg = config.get("matched_granularity", {})
        # if mg_cfg.get("enabled", False):
        #     print("\nPHASE 4: MATCHED-GRANULARITY COMPARISON")
        #     print("-"*40)
        #     matched_manifest = self.run_matched_granularity(X, df, mg_cfg)
        # else:
        #     matched_manifest = {}


        # Phase 4: Feature Importance (use K-means at K* for labels; avoids consensus)
        # print("\nPHASE 4: FEATURE IMPORTANCE (using K-means @ K*)")
        # print("-"*40)
        # X_km, _ = self.apply_preprocessing(X, kmeans_preproc)
        # km_opt = KMeans(n_clusters=optimal_k, random_state=self.random_state, **kmeans_params)
        # km_labels = km_opt.fit_predict(X_km)
        # feature_importance = self.feature_importance_analysis(X_km, km_labels)
        # print(f"✓ Significant features (p<0.05): {feature_importance['n_significant_005']}")
        # print(f"✓ Highly significant (p<0.01): {feature_importance['n_significant_001']}")

        # Phase 5: Run experiments with optimal k for KMeans; HDBSCAN variants as configured
        print("\nPHASE 5: ALGORITHM COMPARISON")
        print("-"*40)
        updated_experiments = []
        for exp in config.get('experiments', []):
            e = exp.copy()
            if e.get('algorithm') in ('kmeans', 'gmm') and 'params' in e:
                # replace k with consensus k
                if 'n_clusters' in e['params']:
                    e['params']['n_clusters'] = optimal_k
                if 'n_components' in e['params']:
                    e['params']['n_components'] = optimal_k
            updated_experiments.append(e)

        results = self.run_experiments(X, df, updated_experiments)

        # Phase 6: Statistical Testing
        print("\nPHASE 6: STATISTICAL SIGNIFICANCE TESTING")
        print("-"*40)
        statistical_results = self.statistical_significance_testing(results)

        # Visualizations (k-optimization summary + experiment dashboard)
        self.create_k_optimization_visualizations()
        self.create_visualizations(results, {})

        # LaTeX tables + Report
        # self.generate_latex_tables(results, k_optimization_results)
        # report = self.generate_academic_report(
        #     results, k_optimization_results, stability_result,
        #     feature_importance, statistical_results, consensus_score=None
        # )

        return {
            'optimal_k': optimal_k,
            'k_optimization': k_optimization_results,
            # 'stability': stability_result,
            # 'kmeans_seed_heatmaps': [str(p) for p in k_heatmaps],
            # 'hdbscan_seed_heatmap': str(hdb_heatmap) if hdb_heatmap else None,
            # 'feature_importance': feature_importance,
            # 'matched_granularity': matched_manifest,
            'experiment_results': results,
            'statistical_results': statistical_results
        }

    
    def generate_academic_report(self, results, k_optimization, stability, 
                                feature_importance, statistical_results, consensus_score):
        """Generate comprehensive academic report"""
        
        report_lines = []
        report_lines.append("# Comprehensive Clustering Analysis Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Experiment ID: {self.experiment_id}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        optimal_k = k_optimization['consensus']['optimal_k']
        report_lines.append(f"**Optimal Number of Clusters: {optimal_k}**")
        report_lines.append("")
        
        # K-Optimization Results
        report_lines.append("## 1. Optimal K Selection")
        report_lines.append("")
        report_lines.append("### Method-wise Results:")
        for method, result in k_optimization.items():
            if method != 'consensus' and 'optimal_k' in result:
                report_lines.append(f"- **{method.replace('_', ' ').title()}**: k = {result['optimal_k']}")
        
        report_lines.append("")
        report_lines.append(f"### Consensus Result: k = {optimal_k}")
        if k_optimization['consensus']['method_agreement']:
            report_lines.append("✓ All methods agree on the optimal k value")
        else:
            report_lines.append(f"Votes distribution: {k_optimization['consensus']['votes']}")
        
        # Stability Analysis
        report_lines.append("")
        report_lines.append("## 2. Stability Analysis")
        if optimal_k in stability['stability_scores']:
            scores = stability['stability_scores'][optimal_k]
            report_lines.append(f"- ARI Mean: {scores['ari_mean']:.3f} ± {scores['ari_std']:.3f}")
            report_lines.append(f"- AMI Mean: {scores['ami_mean']:.3f} ± {scores['ami_std']:.3f}")
            report_lines.append(f"- **Stability Score: {scores['stability_score']:.3f}**")
        
        # Consensus Clustering
        report_lines.append("")
        report_lines.append("## 3. Consensus Clustering")
        report_lines.append(f"- Consensus Score: {consensus_score:.3f}")
        report_lines.append("- Method: 100 iterations with 80% subsampling")
        
        # Feature Importance
        report_lines.append("")
        report_lines.append("## 4. Feature Importance Analysis")
        report_lines.append(f"- Total features analyzed: {len(feature_importance['importance_df'])}")
        report_lines.append(f"- Significant features (p<0.05): {feature_importance['n_significant_005']}")
        report_lines.append(f"- Highly significant (p<0.01): {feature_importance['n_significant_001']}")
        report_lines.append("")
        report_lines.append("### Top 5 Most Important Features:")
        for i, feature in enumerate(feature_importance['top_features'][:5], 1):
            report_lines.append(f"{i}. {feature['feature']}: F-score={feature['f_score']:.2f}, p={feature['p_value']:.4f}")
        
        # Algorithm Performance
        report_lines.append("")
        report_lines.append("## 5. Algorithm Performance Comparison")
        
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            sorted_results = sorted(successful_results, 
                                  key=lambda x: x['metrics'].get('silhouette_score', -1), 
                                  reverse=True)[:5]
            
            report_lines.append("")
            report_lines.append("### Top 5 Performing Configurations:")
            for i, result in enumerate(sorted_results, 1):
                report_lines.append(f"{i}. **{result['experiment_name']}**")
                report_lines.append(f"   - Silhouette: {result['metrics'].get('silhouette_score', -1):.3f}")
                report_lines.append(f"   - Time: {result.get('total_time', 0):.2f}s")
        
        # Statistical Testing
        if statistical_results:
            report_lines.append("")
            report_lines.append("## 6. Statistical Significance Testing")
            if 'algorithm_comparison' in statistical_results:
                test = statistical_results['algorithm_comparison']
                report_lines.append(f"- Test: {test['test'].replace('_', ' ').title()}")
                report_lines.append(f"- p-value: {test['p_value']:.4f}")
                if test['significant']:
                    report_lines.append("- **Result: Significant differences found (p < 0.05)**")
                else:
                    report_lines.append("- Result: No significant differences")
        
        # Reproducibility
        report_lines.append("")
        report_lines.append("## 7. Reproducibility")
        report_lines.append(f"- Random Seed: {self.random_state}")
        report_lines.append(f"- Number of CPU cores: {self.n_jobs}")
        report_lines.append("- All results and configurations saved for replication")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.results_dir / 'academic_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Academic report saved to: {report_file}")
        return report_content
    
    # Include all other methods from original class (run_experiments, evaluate_clustering, etc.)
    # These remain largely the same but I'm omitting them here for brevity
    
    def run_experiments(self, X: np.ndarray, df: pd.DataFrame, 
                       experiment_configs: List[Dict], 
                       true_labels: Optional[np.ndarray] = None) -> List[Dict]:
        """Run all clustering experiments"""
        
        print(f"Running {len(experiment_configs)} experiments...")
        print("="*80)
        
        results = []
        for i, config in enumerate(experiment_configs, 1):
            print(f"Experiment {i}/{len(experiment_configs)}")
            result = self.run_single_experiment(X, config, df, true_labels)
            results.append(result)
            print()
        
        # Save individual results
        self.save_results(results)
        
        return results
    
    def run_single_experiment(self, X: np.ndarray, experiment_config: Dict, 
                            df: pd.DataFrame, true_labels: Optional[np.ndarray] = None) -> Dict:
        """Run a single clustering experiment"""
        
        exp_name = experiment_config['name']
        algorithm = experiment_config['algorithm']
        distance_metric = experiment_config.get('distance_metric', 'euclidean')
        preprocessing = experiment_config.get('preprocessing', 'none')
        params = experiment_config.get('params', {})
        preprocessing_params = experiment_config.get('preprocessing_params', {})
        
        print(f"Running experiment: {exp_name}")
        
        try:
            # Apply preprocessing
            start_time = datetime.now()
            X_processed, preprocessor = self.apply_preprocessing(X, preprocessing, **preprocessing_params)
            preprocessing_time = (datetime.now() - start_time).total_seconds()
            
            # Create and fit clusterer
            start_time = datetime.now()
            clusterer = self.create_clusterer(algorithm, distance_metric, **params)
            labels = self.fit_predict_with_distance(clusterer, X_processed, algorithm, distance_metric)
            clustering_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate results
            metrics = self.evaluate_clustering(X_processed, labels, true_labels)
            
            # Prepare results
            result = {
                'experiment_name': exp_name,
                'algorithm': algorithm,
                'distance_metric': distance_metric,
                'preprocessing': preprocessing,
                'params': params,
                'preprocessing_params': preprocessing_params,
                'preprocessing_time': preprocessing_time,
                'clustering_time': clustering_time,
                'total_time': preprocessing_time + clustering_time,
                'success': True,
                'error': None,
                'labels': labels,
                'metrics': metrics,
                'X_shape_original': X.shape,
                'X_shape_processed': X_processed.shape
            }
            
            print(f"  ✓ Success: {metrics['n_clusters']} clusters, "
                  f"silhouette: {metrics['silhouette_score']:.3f}, "
                  f"time: {result['total_time']:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            result = {
                'experiment_name': exp_name,
                'algorithm': algorithm,
                'distance_metric': distance_metric,
                'preprocessing': preprocessing,
                'params': params,
                'preprocessing_params': preprocessing_params,
                'success': False,
                'error': str(e),
                'labels': None,
                'metrics': {},
                'X_shape_original': X.shape,
                'X_shape_processed': None
            }
        
        return result
    
    def create_clusterer(self, algorithm: str, distance_metric: str = 'euclidean', **params) -> Any:
        """Create clustering algorithm instance"""
        # Set random state for reproducibility where applicable
        if 'random_state' in params or algorithm in ['kmeans', 'spectral', 'gmm']:
            params['random_state'] = self.random_state
        
        if algorithm == 'kmeans':
            return KMeans( **params)
        elif algorithm == 'hdbscan':
            if HDBSCAN_AVAILABLE:
                return HDBSCAN(metric=distance_metric, **params)
            else:
                raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        elif algorithm == 'dbscan':
            return DBSCAN(metric=distance_metric, **params)
        elif algorithm == 'hierarchical':
            if distance_metric in ['euclidean', 'manhattan', 'cosine']:
                return AgglomerativeClustering(metric=distance_metric, **params)
            else:
                return AgglomerativeClustering(**params)
        elif algorithm == 'spectral':
            if distance_metric == 'cosine':
                params['affinity'] = 'cosine'
            return SpectralClustering( **params)
        elif algorithm == 'gmm':
            return GaussianMixture(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def fit_predict_with_distance(self, clusterer, X: np.ndarray, algorithm: str, 
                                 distance_metric: str) -> np.ndarray:
        """Fit clustering algorithm with appropriate distance handling"""
        
        if algorithm == 'kmeans' and distance_metric == 'cosine':
            # For K-means with cosine distance, use spherical K-means approach
            X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
            X_norm = np.nan_to_num(X_norm)  # Handle zero vectors
            labels = clusterer.fit_predict(X_norm)
        else:
            # Standard fit_predict
            labels = clusterer.fit_predict(X)
            
        return labels
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, 
                          true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Comprehensive clustering evaluation metrics"""
        metrics = {}
        
        # Filter out noise points for internal metrics
        mask = labels != -1
        n_total = len(labels)
        n_clustered = mask.sum()
        
        if n_clustered < 2:
            # No valid clusters found
            return {
                'n_clusters': 0,
                'noise_ratio': 1.0,
                'clustered_ratio': 0.0,
                'silhouette_score': -1,
                'calinski_harabasz_score': 0,
                'davies_bouldin_score': float('inf'),
                'inertia': float('inf')
            }
        
        X_clustered = X[mask]
        labels_clustered = labels[mask]
        
        # Basic cluster statistics
        unique_labels = np.unique(labels_clustered)
        metrics['n_clusters'] = len(unique_labels)
        metrics['noise_ratio'] = (n_total - n_clustered) / n_total
        metrics['clustered_ratio'] = n_clustered / n_total
        
        # Internal validity metrics
        try:
            if len(unique_labels) > 1:
                metrics['silhouette_score'] = silhouette_score(X_clustered, labels_clustered)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clustered, labels_clustered)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_clustered, labels_clustered)
            else:
                metrics['silhouette_score'] = -1
                metrics['calinski_harabasz_score'] = 0
                metrics['davies_bouldin_score'] = float('inf')
        except Exception as e:
            print(f"Warning: Could not compute internal metrics: {e}")
            metrics['silhouette_score'] = -1
            metrics['calinski_harabasz_score'] = 0
            metrics['davies_bouldin_score'] = float('inf')
        
        return metrics
    
    def save_results(self, results: List[Dict]):
        """Save experiment results"""
        print("Saving results...")
        
        # Save detailed results (without large arrays)
        results_summary = []
        for result in results:
            summary = result.copy()
            # Remove large arrays for JSON serialization
            if 'labels' in summary:
                summary['labels'] = f"array_shape_{len(summary['labels']) if summary['labels'] is not None else 0}"
            results_summary.append(summary)
        
        # Save as JSON
        results_file = self.results_dir / 'experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save complete results with labels
        complete_results_file = self.results_dir / 'complete_results.pkl'
        with open(complete_results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to: {self.results_dir}")
    
    def statistical_significance_testing(self, results: List[Dict]) -> Dict:
        """Perform statistical significance testing between methods"""
        print("Performing statistical significance testing...")
        
        successful_results = [r for r in results if r.get('success', False)]
        if len(successful_results) < 3:
            print("Need at least 3 successful experiments for statistical testing")
            return {}
        
        # Prepare data
        metrics_data = []
        for result in successful_results:
            row = {
                'algorithm': result['algorithm'],
                'preprocessing': result['preprocessing'],
                'distance': result['distance_metric']
            }
            row.update(result['metrics'])
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        statistical_results = {}
        
        # Test algorithms
        if 'silhouette_score' in df.columns:
            algorithms = df['algorithm'].unique()
            if len(algorithms) > 2:
                algorithm_groups = [df[df['algorithm'] == algo]['silhouette_score'].values 
                                  for algo in algorithms]
                
                # Remove empty groups
                algorithm_groups = [group for group in algorithm_groups if len(group) > 0]
                
                if len(algorithm_groups) > 2:
                    try:
                        # Kruskal-Wallis test (non-parametric ANOVA)
                        statistic, p_value = kruskal(*algorithm_groups)
                        statistical_results['algorithm_comparison'] = {
                            'test': 'kruskal_wallis',
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'groups': list(algorithms)
                        }
                    except Exception as e:
                        print(f"Statistical testing failed: {e}")
        
        return statistical_results
    
    def create_visualizations(self, results: List[Dict], analysis: Dict):
        """Create comprehensive result visualizations"""
        print("Creating visualizations...")
        
        successful_results = [r for r in results if r.get('success', False)]
        if not successful_results:
            return
        
        # Prepare data
        metrics_data = []
        for result in successful_results:
            row = {
                'experiment': result['experiment_name'],
                'algorithm': result['algorithm'],
                'distance': result['distance_metric'],
                'preprocessing': result['preprocessing'],
                'time': result['total_time']
            }
            row.update(result['metrics'])
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Algorithm performance comparison
        ax1 = plt.subplot(2, 3, 1)
        if 'silhouette_score' in df.columns:
            algo_sil = df.groupby('algorithm')['silhouette_score'].mean().sort_values(ascending=False)
            algo_sil.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Average Silhouette Score by Algorithm')
            ax1.set_ylabel('Silhouette Score')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Preprocessing method comparison
        ax2 = plt.subplot(2, 3, 2)
        if 'silhouette_score' in df.columns:
            prep_sil = df.groupby('preprocessing')['silhouette_score'].mean().sort_values(ascending=False)
            prep_sil.plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title('Average Silhouette Score by Preprocessing')
            ax2.set_ylabel('Silhouette Score')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Time vs Performance trade-off
        ax3 = plt.subplot(2, 3, 3)
        if 'silhouette_score' in df.columns:
            scatter = ax3.scatter(df['time'], df['silhouette_score'], 
                                c=df['n_clusters'], alpha=0.7, cmap='viridis')
            ax3.set_xlabel('Execution Time (seconds)')
            ax3.set_ylabel('Silhouette Score')
            ax3.set_title('Performance vs Time Trade-off')
            ax3.set_xscale('log')
            plt.colorbar(scatter, ax=ax3, label='Number of Clusters')
        
        # 4. Algorithm performance heatmap
        ax4 = plt.subplot(2, 3, 4)
        key_metrics = ['silhouette_score', 'calinski_harabasz_score', 'noise_ratio']
        available_metrics = [m for m in key_metrics if m in df.columns]
        if available_metrics:
            heatmap_data = df.groupby('algorithm')[available_metrics].mean()
            sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlBu_r', ax=ax4, fmt='.3f')
            ax4.set_title('Algorithm Performance Heatmap')
        
        # 5. Top experiments ranking
        ax5 = plt.subplot(2, 3, 5)
        if 'silhouette_score' in df.columns:
            top_10 = df.nlargest(10, 'silhouette_score')
            y_pos = np.arange(len(top_10))
            ax5.barh(y_pos, top_10['silhouette_score'], color='gold')
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels([f"{alg[:8]}_{prep[:8]}" for alg, prep in 
                                zip(top_10['algorithm'], top_10['preprocessing'])], fontsize=8)
            ax5.set_xlabel('Silhouette Score')
            ax5.set_title('Top 10 Experiments')
        
        # 6. Distribution of metrics
        ax6 = plt.subplot(2, 3, 6)
        if 'silhouette_score' in df.columns:
            df['silhouette_score'].hist(bins=20, ax=ax6, color='purple', alpha=0.7)
            ax6.axvline(df['silhouette_score'].mean(), color='red', linestyle='--', label='Mean')
            ax6.set_title('Distribution of Silhouette Scores')
            ax6.set_xlabel('Silhouette Score')
            ax6.set_ylabel('Frequency')
            ax6.legend()
        
        plt.tight_layout()
        
        # Save figure
        viz_file = self.results_dir / 'experiment_visualizations.png'
        plt.savefig(viz_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {viz_file}")


def create_enhanced_config() -> Dict:
    """Create enhanced configuration with k-optimization settings"""
    config = {
        'data': {
            'path': 'paper_clusters.csv',
            'vector_prefix': 'cluster_',
            'true_labels_column': None
        },
        'k_optimization': {
            'k_min': 2,
            'k_max': 21,
            'k_methods': ['elbow', 'gap', 'silhouette', 'stability', 'prediction_strength']
        },
        'experiments': [
            {
                'name': 'kmeans_euclidean_standard',
                'algorithm': 'kmeans',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'n_clusters': 8}  # Will be replaced with optimal k
            },
            {
                'name': 'kmeans_cosine_l2norm',
                'algorithm': 'kmeans',
                'distance_metric': 'cosine',
                'preprocessing': 'l2_normalize',
                'params': {'n_clusters': 8}
            },
            {
                'name': 'hierarchical_euclidean_standard',
                'algorithm': 'hierarchical',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'n_clusters': 8, 'linkage': 'ward'}
            },
            {
                'name': 'gmm_standard',
                'algorithm': 'gmm',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'n_components': 8, 'covariance_type': 'full'}
            }
        ]
    }
    return config


def main():
    parser = argparse.ArgumentParser(description="Enhanced clustering framework with K-optimization")
    parser.add_argument('--config', type=str, help='Path to experiment configuration YAML file')
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory for results')
    parser.add_argument('--data_path', type=str, help='Path to input data CSV')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--create_config', action='store_true', help='Create enhanced configuration file')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration file if requested
    if args.create_config:
        config = create_enhanced_config()
        config_file = work_dir / 'enhanced_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created enhanced configuration: {config_file}")
        print(f"\nRun analysis with:")
        print(f"python {__file__} --config {config_file} --work_dir {args.work_dir}")
        return
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("Using default enhanced configuration...")
        config = create_enhanced_config()
    
    # Override data path if provided
    if args.data_path:
        config['data']['path'] = args.data_path
    
    # Initialize framework
    framework = EnhancedClusteringFramework(work_dir, n_jobs=args.n_jobs)
    
    # Load data
    try:
        data_path = Path(config['data']['path'])
        X, df = framework.load_data(data_path, config['data']['vector_prefix'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run complete analysis
    results = framework.run_complete_analysis(X, df, config)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results directory: {framework.results_dir}")
    print(f"Optimal K: {results['optimal_k']}")
    # print(f"Consensus Score: {results['consensus_score']:.3f}")
    print(f"\nView reports:")
    print(f"  - Academic report: {framework.results_dir / 'academic_report.md'}")
    print(f"  - LaTeX tables: {framework.results_dir / 'latex_tables.tex'}")
    print(f"  - Visualizations: {framework.results_dir}")


if __name__ == "__main__":
    main()