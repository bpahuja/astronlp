#!/usr/bin/env python3
"""
End-to-end Paragraph-level Clustering (GPU/CPU, memory-optimized)
- HDBSCAN with IPCA/UMAP options, hyperparameter sweep, stability, and paper-join outputs.
- Produces per-paper paragraph-topic mixtures and LLM staging JSONL (5 nearest paragraphs per cluster).

Usage (minimal):
  python paragraph_clustering_end2end.py \
      --chunks_root /path/to/paragraph/chunks \
      --work_dir /path/to/outdir

Recommended (IPCA backend + sweep + paper alignment + LLM staging):
  python paragraph_clustering_end2end.py \
      --chunks_root /data/para_chunks --work_dir results/para_pipeline2 \
      --backend ipca_hdbscan --ipca_dim 128 --min_cluster_size 12 --min_samples 1 \
      --sweep_min_cluster_size 8,12,16 --sweep_min_samples 1,5 \
      --paper_labels_csv results/paper_pipeline2/final_hdbscan_labels.csv \
      --export_llm_staging --meta_text_columns paragraph_text \
      --topk 5 --topk_eval_cap 2000 \
      --seed_heatmap_runs 5 --seed_heatmap_shuffle

Expected chunk structure per shard:
  chunk_0000/
     embeddings.npy   (float32, [n_i, d])
     metadata.csv     (must contain: para_id, paper_id; optional: paragraph_text)
"""

import os
import re
import gc
import json
import math
import psutil
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Iterator, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, v_measure_score
from sklearn.cluster import KMeans

# GPU deps
try:
    import cupy as cp
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
    cp = None

try:
    import cuml
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    CUML_AVAILABLE = True
except Exception:
    CUML_AVAILABLE = False
    cuml = None

# CPU fallbacks
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False
    hdbscan = None

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False
    umap = None

warnings.filterwarnings("ignore")


# ---------------- Memory helpers ----------------
def print_mem(stage: str, debug: bool = False):
    if not debug: return
    rss = psutil.Process().memory_info().rss / (1024**3)
    if GPU_AVAILABLE:
        try:
            mp = cp.get_default_memory_pool()
            print(f"[{stage}] CPU={rss:.2f} GB | GPU used={mp.used_bytes()/1e9:.2f} GB")
        except Exception:
            print(f"[{stage}] CPU={rss:.2f} GB (GPU unknown)")
    else:
        print(f"[{stage}] CPU={rss:.2f} GB")

def free_gpu():
    if GPU_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


# ---------------- Disk-mapped embeddings ----------------
class MemmapEmb:
    def __init__(self, filepath: Path, shape: Tuple[int,int], dtype=np.float32):
        self.filepath = filepath; self.shape = shape; self.dtype = dtype; self._m = None
    def __enter__(self):
        self._m = np.memmap(self.filepath, dtype=self.dtype, mode="w+", shape=self.shape); return self._m
    def __exit__(self, *args):
        if self._m is not None: del self._m; self._m = None; gc.collect()
    def ro(self): return np.memmap(self.filepath, dtype=self.dtype, mode="r", shape=self.shape)

def scan_chunk_dirs(root: Path, patt=r"chunk_\d{4}") -> List[Path]:
    rx = re.compile(patt)
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and rx.fullmatch(p.name)]
    if not dirs: raise FileNotFoundError(f"No chunk dirs under {root} matching /{patt}/")
    return dirs

def chunks(arr: np.ndarray, sz: int):
    n = arr.shape[0]
    for s in range(0, n, sz):
        e = min(s+sz, n); yield s, e, arr[s:e]

def build_mmap(chunk_dirs: List[Path], meta_name: str, emb_name: str,
               sample_every: int, work_dir: Path, dbg=False):
    print_mem("Before consolidate", dbg)
    tot = 0; d = None
    for cd in tqdm(chunk_dirs, desc="Counting"):
        ep = cd/emb_name; mp = cd/meta_name
        if not ep.exists() or not mp.exists(): continue
        E = np.load(ep, mmap_mode="r"); m, dim = E.shape
        if d is None: d = dim
        elif d != dim: raise ValueError(f"Dim mismatch: {cd} has {dim} vs {d}")
        tot += (m + sample_every - 1)//sample_every
    print(f"[mmap] rows={tot:,} dim={d}")

    mmap_path = work_dir/"para_embeddings.dat"
    mmapE = MemmapEmb(mmap_path, (tot, d))

    para_ids, paper_ids, text_buf = [], [], []
    with mmapE as X:
        cur = 0
        for cd in tqdm(chunk_dirs, desc="Consolidating"):
            ep = cd/emb_name; mp = cd/meta_name
            E = np.load(ep, mmap_mode="r")
            base = 0
            for df in pd.read_csv(mp, chunksize=50_000):
                sz = len(df)
                for j in range(0, sz, sample_every):
                    if cur >= tot: break
                    X[cur] = E[base+j]
                    para_ids.append(df["para_id"].iat[j])
                    paper_ids.append(df["paper_id"].iat[j])
                    # Optional: carry paragraph text if exists
                    if "paragraph_text" in df.columns:
                        text_buf.append(df["paragraph_text"].iat[j])
                    else:
                        text_buf.append(None)
                    cur += 1
                base += sz
            del E; gc.collect()
    print_mem("After consolidate", dbg)
    return mmapE, para_ids, paper_ids, text_buf


# ---------------- Preprocessing reducers ----------------
def l2_normalize_inplace(X: np.memmap, chunk_size=100_000, dbg=False):
    print_mem("Before L2 norm", dbg); n = X.shape[0]
    for s,e,blk in tqdm(chunks(X, chunk_size), total=math.ceil(n/chunk_size), desc="L2 norm"):
        nrm = np.linalg.norm(blk, axis=1, keepdims=True); nrm = np.maximum(nrm, 1e-12); blk /= nrm; X[s:e]=blk
    print_mem("After L2 norm", dbg)

def ipca_reduce(X: np.ndarray, dim=128, batch=100_000, work_dir: Optional[Path]=None, dbg=False) -> np.ndarray:
    print_mem("Before IPCA", dbg); n,d0 = X.shape; dim = min(dim, d0-1)
    ip = IncrementalPCA(n_components=dim, batch_size=batch)
    for s,e,blk in tqdm(chunks(X, batch), total=math.ceil(n/batch), desc="IPCA fit"): ip.partial_fit(blk)
    if work_dir:
        outp = work_dir/"ipca.dat"; out = MemmapEmb(outp, (n,dim))
        with out as Z:
            for s,e,blk in tqdm(chunks(X, batch), total=math.ceil(n/batch), desc="IPCA transform"):
                Z[s:e] = ip.transform(blk).astype(np.float32)
        print_mem("After IPCA", dbg); return out.ro()
    Z = np.empty((n,dim), np.float32)
    for s,e,blk in tqdm(chunks(X, batch), total=math.ceil(n/batch), desc="IPCA transform"):
        Z[s:e] = ip.transform(blk).astype(np.float32)
    print_mem("After IPCA", dbg); return Z

def umap_reduce(X: np.ndarray, dim=50, n_neighbors=200, chunk_size=200_000, dbg=False) -> np.ndarray:
    if CUML_AVAILABLE:
        try:
            print_mem("Before cuUMAP", dbg); n=X.shape[0]
            reducer = cuUMAP(n_components=dim, n_neighbors=min(n_neighbors, max(2, n-1)),
                             min_dist=0.05, metric="cosine", random_state=42, verbose=True)
            if GPU_AVAILABLE:
                Xg = cp.asarray(X, dtype=cp.float32); Zg = reducer.fit_transform(Xg); Z = cp.asnumpy(Zg)
                del Xg, Zg; free_gpu()
            else:
                Z = reducer.fit_transform(X)
            print_mem("After cuUMAP", dbg); return Z
        except Exception as e:
            print(f"[cuUMAP] failed: {e} -> fallback CPU")
            free_gpu()
    if not UMAP_AVAILABLE:
        raise RuntimeError("umap-learn not installed.")
    print_mem("Before CPU UMAP", dbg)
    reducer = umap.UMAP(n_components=dim, n_neighbors=n_neighbors, min_dist=0.05,
                        metric="cosine", random_state=42, verbose=True, low_memory=True)
    Z = reducer.fit_transform(X)
    print_mem("After CPU UMAP", dbg); return Z


# ---------------- HDBSCAN (GPU/CPU) ----------------
def hdbscan_fit_predict(X: np.ndarray, min_cluster_size=12, min_samples=1, metric="euclidean", dbg=False) -> np.ndarray:
    if CUML_AVAILABLE:
        try:
            print_mem("Before cuHDBSCAN", dbg)
            cl = cuHDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           metric=metric, prediction_data=False, verbose=False)
            if GPU_AVAILABLE:
                Xg = cp.asarray(X, dtype=cp.float32); y = cl.fit_predict(Xg); y = cp.asnumpy(y)
                del Xg; free_gpu()
            else:
                y = cl.fit_predict(X)
            print_mem("After cuHDBSCAN", dbg); return y.astype(int)
        except Exception as e:
            print(f"[cuHDBSCAN] failed: {e} -> CPU fallback"); free_gpu()
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("hdbscan not installed.")
    print_mem("Before CPU HDBSCAN", dbg)
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                         metric=metric, core_dist_n_jobs=-1, prediction_data=False)
    y = cl.fit_predict(X)
    print_mem("After CPU HDBSCAN", dbg); return y.astype(int)


# ---------------- Metrics & Stability ----------------
def internal_metrics(X: np.ndarray, y: np.ndarray, sample_n: int = 100_000) -> Dict[str, float]:
    mask = y >= 0
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return {"silhouette": -1.0, "ch": 0.0, "db": float("inf")}
    Xc, yc = X[mask], y[mask]
    if sample_n and Xc.shape[0] > sample_n:
        idx = np.random.choice(Xc.shape[0], sample_n, replace=False)
        Xc = Xc[idx]; yc = yc[idx]
    try:
        sil = float(silhouette_score(Xc, yc))
    except Exception:
        sil = -1.0
    try:
        ch = float(calinski_harabasz_score(Xc, yc))
    except Exception:
        ch = 0.0
    try:
        db = float(davies_bouldin_score(Xc, yc))
    except Exception:
        db = float("inf")
    return {"silhouette": sil, "ch": ch, "db": db}

def seed_ami_heatmap(X: np.ndarray, runs=5, min_cluster_size=12, min_samples=1,
                     metric="euclidean", shuffle=True, save_path: Optional[Path]=None) -> np.ndarray:
    labels_list = []
    n = X.shape[0]
    for i in range(runs):
        if shuffle:
            perm = np.random.permutation(n); Xs = X[perm]
            y = hdbscan_fit_predict(Xs, min_cluster_size, min_samples, metric)
            inv = np.empty_like(perm); inv[perm] = np.arange(n); y = y[inv]
        else:
            y = hdbscan_fit_predict(X, min_cluster_size, min_samples, metric)
        labels_list.append(y)
    S = len(labels_list); AMI = np.zeros((S,S), dtype=float)
    for i in range(S):
        for j in range(S):
            AMI[i,j] = adjusted_mutual_info_score(labels_list[i], labels_list[j])
    if save_path is not None:
        import matplotlib.pyplot as plt, seaborn as sns
        fig, ax = plt.subplots(figsize=(1.2*S+4, 1.2*S+3))
        sns.heatmap(AMI, vmin=0.0, vmax=1.0, cmap="viridis", annot=True, fmt=".2f",
                    square=True, ax=ax, cbar_kws={"label":"AMI"})
        ax.set_title(f"HDBSCAN seed/jitter robustness (runs={runs})")
        fig.tight_layout(); fig.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)
    return AMI


# ---------------- LLM staging: top-k nearest paragraphs per cluster ----------------
def export_llm_staging(Z: np.ndarray, labels: np.ndarray, para_ids: List[Any], paper_ids: List[Any],
                       texts: List[Optional[str]], out_jsonl: Path, topk=5, eval_cap=2000):
    """
    For each cluster, compute centroid in Z and extract top-k nearest paragraphs.
    To bound cost, evaluate at most 'eval_cap' points per cluster (uniform sample if larger).
    """
    from math import ceil
    K = sorted([int(c) for c in np.unique(labels) if c >= 0])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for c in tqdm(K, desc="LLM staging"):
            idx = np.where(labels == c)[0]
            if idx.size == 0: continue
            # cap for efficiency
            if idx.size > eval_cap:
                sel = np.random.choice(idx, eval_cap, replace=False)
            else:
                sel = idx
            C = Z[sel]
            centroid = C.mean(axis=0, keepdims=True)
            # cosine w.r.t. Z (assume Z already in Euclidean space; use L2 dist)
            dists = np.linalg.norm(C - centroid, axis=1)
            order = np.argsort(dists)[:topk]
            items = []
            for r in order:
                g = sel[r]
                items.append({
                    "para_id": str(para_ids[g]),
                    "paper_id": str(paper_ids[g]),
                    "distance": float(dists[r]),
                    "text": texts[g] if texts[g] is not None else None
                })
            obj = {"cluster_id": int(c), "topk": items}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------- Paper-join: paragraph mixtures & alignment ----------------
def build_paper_mixtures(labels: np.ndarray, paper_ids: List[Any], include_noise=False) -> pd.DataFrame:
    """Return long-form mixture: columns [paper_id, cluster, count, prop]."""
    df = pd.DataFrame({"paper_id": paper_ids, "cluster": labels})
    if not include_noise:
        df = df[df["cluster"] >= 0]
    if df.empty:
        return pd.DataFrame(columns=["paper_id","cluster","count","prop"])
    grp = df.groupby(["paper_id","cluster"]).size().reset_index(name="count")
    tot = grp.groupby("paper_id")["count"].transform("sum")
    grp["prop"] = grp["count"] / tot
    return grp

def align_with_paper_labels(mix_long: pd.DataFrame, paper_labels_csv: Path) -> Dict[str, Any]:
    """
    1) Dominant paragraph topic per paper vs. paper-level label -> AMI/NMI/V.
    2) KMeans on mixture (K = num paper labels) vs. paper-level label -> AMI/NMI/V.
    """
    if mix_long.empty:
        return {"status":"no_mixture"}
    pdf = pd.read_csv(paper_labels_csv)  # expects columns: paper_id, cluster (paper-level)
    pdf = pdf.rename(columns={"cluster":"paper_label"})
    # dominant topic per paper
    dom = (mix_long.sort_values(["paper_id","prop"], ascending=[True, False])
                  .groupby("paper_id").head(1)[["paper_id","cluster"]]
                  .rename(columns={"cluster":"dominant_para_cluster"}))
    j = dom.merge(pdf, on="paper_id", how="inner")
    if j.empty:
        return {"status":"no_overlap"}
    out = {}
    out["dominant_AMI"] = float(adjusted_mutual_info_score(j["dominant_para_cluster"], j["paper_label"]))
    out["dominant_NMI"] = float(normalized_mutual_info_score(j["dominant_para_cluster"], j["paper_label"]))
    out["dominant_V"]   = float(v_measure_score(j["dominant_para_cluster"], j["paper_label"]))

    # KMeans on mixtures (wide)
    wide = mix_long.pivot(index="paper_id", columns="cluster", values="prop").fillna(0.0)
    jw = pdf.merge(wide.reset_index(), on="paper_id", how="inner")
    if jw.shape[0] >= 5:
        K = int(pdf["paper_label"].nunique())
        kk = KMeans(n_clusters=max(2, K), n_init=10, random_state=42)
        km_lbl = kk.fit_predict(jw.drop(columns=["paper_id","paper_label"]).values)
        out["mixture_kmeans_AMI"] = float(adjusted_mutual_info_score(km_lbl, jw["paper_label"]))
        out["mixture_kmeans_NMI"] = float(normalized_mutual_info_score(km_lbl, jw["paper_label"]))
        out["mixture_kmeans_V"]   = float(v_measure_score(km_lbl, jw["paper_label"]))
        out["mixture_k"] = int(max(2, K))
    else:
        out["mixture_status"] = "too_few_papers"
    return out


# ---------------- Sweep orchestration ----------------
def run_backend(X_mmap_ro: np.memmap, backend: str, args) -> Tuple[np.ndarray, str, str]:
    """
    Returns: (Z, metric_for_hdbscan, backend_tag)
    """
    if backend == "hdbscan_raw":
        # cosine-friendly: we l2-normalize in-place and use cosine
        with open(X_mmap_ro.filename, "r+b") as _:
            X_rw = np.memmap(X_mmap_ro.filename, dtype=X_mmap_ro.dtype, mode="r+", shape=X_mmap_ro.shape)
            l2_normalize_inplace(X_rw, args.chunk_size, args.debug_memory)
        Z = np.memmap(X_mmap_ro.filename, dtype=X_mmap_ro.dtype, mode="r", shape=X_mmap_ro.shape)
        return Z, "cosine", "raw"
    elif backend == "ipca_hdbscan":
        Z = ipca_reduce(X_mmap_ro, dim=args.ipca_dim, batch=args.chunk_size,
                        work_dir=(Path(args.work_dir) if args.use_mmap_output else None),
                        dbg=args.debug_memory)
        return Z, "euclidean", f"ipca{args.ipca_dim}"
    elif backend == "umap_hdbscan":
        Z = umap_reduce(X_mmap_ro, dim=args.umap_dim, n_neighbors=args.n_neighbors,
                        chunk_size=args.chunk_size, dbg=args.debug_memory)
        return Z, "euclidean", f"umap{args.umap_dim}"
    else:
        raise ValueError(f"Unknown backend: {backend}")

def evaluate_run(Z: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    k = int(np.sum(y >= 0 and True) if False else len(np.unique(y[y>=0])))
    noise = int(np.sum(y < 0))
    n = int(y.shape[0])
    cov = (n - noise)/n if n else 0.0
    met = internal_metrics(Z, y, sample_n=100_000)
    return {
        "n": n, "clusters": k, "noise": noise, "coverage": cov,
        "silhouette": met["silhouette"], "db": met["db"], "ch": met["ch"]
    }

def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def run_sweep(Z: np.ndarray, metric: str, args) -> pd.DataFrame:
    mcs = parse_list_int(args.sweep_min_cluster_size) if args.sweep_min_cluster_size else [args.min_cluster_size]
    msp = parse_list_int(args.sweep_min_samples) if args.sweep_min_samples else [args.min_samples]
    rows = []
    for a in mcs:
        for b in msp:
            y = hdbscan_fit_predict(Z, a, b, metric, args.debug_memory)
            res = evaluate_run(Z, y); res.update({"min_cluster_size":a, "min_samples":b})
            rows.append(res)
            # write labels for the "current best" by silhouette-coverage Pareto if requested
    df = pd.DataFrame(rows).sort_values(["silhouette","coverage"], ascending=[False, False])
    return df


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Paragraph-level clustering end-to-end (GPU/CPU)")
    # IO
    ap.add_argument("--chunks_root", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--chunk_pattern", default=r"chunk_\d{4}")
    ap.add_argument("--meta_name", default="metadata.csv")
    ap.add_argument("--emb_name", default="embeddings.npy")
    ap.add_argument("--max_chunks", type=int, default=0)
    ap.add_argument("--sample_every", type=int, default=1)
    # Backend
    ap.add_argument("--backend", choices=["hdbscan_raw","ipca_hdbscan","umap_hdbscan"], default="ipca_hdbscan")
    ap.add_argument("--ipca_dim", type=int, default=128)
    ap.add_argument("--umap_dim", type=int, default=50)
    ap.add_argument("--n_neighbors", type=int, default=200)
    # HDBSCAN core
    ap.add_argument("--min_cluster_size", type=int, default=12)
    ap.add_argument("--min_samples", type=int, default=1)
    # Sweep
    ap.add_argument("--sweep_min_cluster_size", type=str, default="")  # e.g., "8,12,16,24"
    ap.add_argument("--sweep_min_samples", type=str, default="")       # e.g., "1,3,5"
    # Stability
    ap.add_argument("--seed_heatmap_runs", type=int, default=0)
    ap.add_argument("--seed_heatmap_shuffle", action="store_true")
    # LLM staging
    ap.add_argument("--export_llm_staging", action="store_true")
    ap.add_argument("--meta_text_columns", type=str, default="paragraph_text")  # CSV column for paragraph text (optional)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--topk_eval_cap", type=int, default=2000)
    # Paper join
    ap.add_argument("--paper_labels_csv", type=str, default="")  # expects columns: paper_id, cluster
    # Perf/Memory
    ap.add_argument("--chunk_size", type=int, default=100_000)
    ap.add_argument("--use_mmap_output", action="store_true")
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--debug_memory", action="store_true")
    args = ap.parse_args()

    # GPU toggles
    if args.force_cpu:
        global GPU_AVAILABLE, CUML_AVAILABLE
        GPU_AVAILABLE = False; CUML_AVAILABLE = False

    outdir = Path(args.work_dir); outdir.mkdir(parents=True, exist_ok=True)
    print_mem("start", args.debug_memory)

    # Scan shards and consolidate
    shards = scan_chunk_dirs(Path(args.chunks_root), args.chunk_pattern)
    if args.max_chunks > 0: shards = shards[:args.max_chunks]
    mmapE, para_ids, paper_ids, texts = build_mmap(shards, args.meta_name, args.emb_name,
                                                   args.sample_every, outdir, args.debug_memory)
    try:
        X_ro = mmapE.ro()
        # Backend reduction
        Z, metric, tag = run_backend(X_ro, args.backend, args)

        # Sweep (optional)
        sweep_csv = outdir/"paragraph_hdbscan_sweep.csv"
        sweep_df = run_sweep(Z, metric, args)
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"[write] sweep -> {sweep_csv}")

        # Choose best by Silhouette then coverage
        best = sweep_df.iloc[0].to_dict()
        b_mcs = int(best["min_cluster_size"]); b_ms = int(best["min_samples"])
        print(f"[select] best params: min_cluster_size={b_mcs}, min_samples={b_ms} "
              f"(sil={best['silhouette']:.3f}, cov={best['coverage']:.3f})")

        # Final fit
        y = hdbscan_fit_predict(Z, b_mcs, b_ms, metric, args.debug_memory)
        summ = evaluate_run(Z, y)
        summ.update({"backend": args.backend, "metric": metric,
                     "min_cluster_size": b_mcs, "min_samples": b_ms})
        (outdir/"paragraph_clustering_summary.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps(summ, indent=2))

        # Labels
        lab_path = outdir/"labels_paragraph.csv"
        pd.DataFrame({"para_id": para_ids, "paper_id": paper_ids, "cluster": y}).to_csv(lab_path, index=False)
        print(f"[write] labels -> {lab_path}")

        # Stability heatmap (optional)
        if args.seed_heatmap_runs > 1:
            heat = seed_ami_heatmap(Z, runs=args.seed_heatmap_runs, min_cluster_size=b_mcs,
                                    min_samples=b_ms, metric=metric, shuffle=args.seed_heatmap_shuffle,
                                    save_path=outdir/"hdbscan_seed_ami_heatmap_paragraphs.png")
            np.save(outdir/"hdbscan_seed_ami_matrix.npy", heat)

        # Paper mixtures
        mix_long = build_paper_mixtures(y, paper_ids, include_noise=False)
        mix_long.to_csv(outdir/"paper_paragraph_mixture_long.csv", index=False)
        # wide
        wide = mix_long.pivot(index="paper_id", columns="cluster", values="prop").fillna(0.0)
        wide.to_csv(outdir/"paper_paragraph_mixture_wide.csv")
        print("[write] paper mixtures (long, wide)")

        # Align with paper-level labels (if provided)
        if args.paper_labels_csv and Path(args.paper_labels_csv).exists():
            align = align_with_paper_labels(mix_long, Path(args.paper_labels_csv))
            (outdir/"paragraph_to_paper_alignment.json").write_text(json.dumps(align, indent=2))
            print(f"[write] alignment -> paragraph_to_paper_alignment.json")

        # LLM staging
        if args.export_llm_staging:
            jsonl = outdir/"llm_paragraph_staging.jsonl"
            # if backend performed dimensionality reduction, Z is ready; for raw-cosine it is normalized embeddings
            export_llm_staging(Z, y, para_ids, paper_ids, texts, jsonl, topk=args.topk, eval_cap=args.topk_eval_cap)
            print(f"[write] LLM staging -> {jsonl}")

        # Save config echo
        cfg = {
            "backend": args.backend, "metric": metric, "ipca_dim": args.ipca_dim,
            "umap_dim": args.umap_dim, "n_neighbors": args.n_neighbors,
            "min_cluster_size": b_mcs, "min_samples": b_ms,
            "sweep_min_cluster_size": args.sweep_min_cluster_size,
            "sweep_min_samples": args.sweep_min_samples,
            "chunk_size": args.chunk_size, "sample_every": args.sample_every,
            "seed_heatmap_runs": args.seed_heatmap_runs,
            "seed_heatmap_shuffle": args.seed_heatmap_shuffle,
            "paper_labels_csv": args.paper_labels_csv,
            "export_llm_staging": args.export_llm_staging,
            "topk": args.topk, "topk_eval_cap": args.topk_eval_cap,
            "gpu_available": GPU_AVAILABLE, "cuml_available": CUML_AVAILABLE
        }
        (outdir/"paragraph_pipeline_config.json").write_text(json.dumps(cfg, indent=2))
        print("[done] paragraph pipeline complete.")

    finally:
        # cleanup mmap file
        if mmapE.filepath.exists():
            try: mmapE.filepath.unlink()
            except Exception: pass
        free_gpu()


if __name__ == "__main__":
    main()
