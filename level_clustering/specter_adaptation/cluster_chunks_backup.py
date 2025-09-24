# #!/usr/bin/env python3
# """
# Cluster paragraph-level embeddings stored as sharded chunk folders with GPU acceleration.
# Expected layout:
#   root/
#     chunk_0000/
#       embeddings.npy   # shape: (N0, d) float32
#       metadata.csv     # columns: paper_id, para_id, text (N0 rows, same order as embeddings.npy)
#     chunk_0001/
#       embeddings.npy
#       metadata.csv
#     ...

# This script:
#   1) Concatenates ALL paragraph embeddings (no averaging).
#   2) (optional) Reduces dimension with GPU UMAP or IncrementalPCA.
#   3) Clusters with GPU HDBSCAN.
#   4) Writes labels & summary.

# Dependencies:
#   pip install numpy pandas tqdm scikit-learn cuml cudf cupy-cuda12x psutil
#   # For RAPIDS CUML (GPU UMAP/HDBSCAN):
#   # conda install -c rapidsai -c conda-forge -c nvidia cuml cudf
# """

# import re
# import gc
# import json
# import argparse
# import psutil
# from pathlib import Path
# from typing import List, Tuple, Optional
# import warnings

# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.decomposition import IncrementalPCA

# # GPU libraries
# try:
#     import cupy as cp
#     GPU_AVAILABLE = True
# except ImportError:
#     print("Warning: CuPy not available, falling back to CPU")
#     GPU_AVAILABLE = False
#     cp = None

# try:
#     import cuml
#     from cuml.manifold import UMAP as cuUMAP
#     from cuml.cluster import HDBSCAN as cuHDBSCAN
#     CUML_AVAILABLE = True
# except ImportError:
#     print("Warning: CUML not available, falling back to CPU libraries")
#     CUML_AVAILABLE = False
#     cuml = None

# # Fallback CPU libraries
# try:
#     import hdbscan
#     HDBSCAN_AVAILABLE = True
# except ImportError:
#     print("Warning: hdbscan not available")
#     HDBSCAN_AVAILABLE = False
#     hdbscan = None

# try:
#     import umap
#     UMAP_AVAILABLE = True
# except ImportError:
#     print("Warning: umap-learn not available")
#     UMAP_AVAILABLE = False
#     umap = None


# # ---------------- Memory Management ----------------

# def print_memory_info(stage: str, debug: bool = False):
#     """Print memory usage information"""
#     if not debug:
#         return
    
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     rss_gb = mem_info.rss / (1024 ** 3)
    
#     if GPU_AVAILABLE:
#         try:
#             mempool = cp.get_default_memory_pool()
#             gpu_used_gb = mempool.used_bytes() / (1024 ** 3)
#             gpu_total_gb = mempool.total_bytes() / (1024 ** 3)
#             print(f"[{stage}] CPU: {rss_gb:.2f}GB | GPU: {gpu_used_gb:.2f}GB used / {gpu_total_gb:.2f}GB total")
#         except:
#             print(f"[{stage}] CPU: {rss_gb:.2f}GB")
#     else:
#         print(f"[{stage}] CPU: {rss_gb:.2f}GB")


# def estimate_memory_usage(n_samples: int, n_features: int, 
#                          backend: str, reduced_dim: int = 0) -> float:
#     """Estimate memory usage in GB"""
#     # Base memory for embeddings (float32)
#     base_memory = n_samples * n_features * 4 / (1024 ** 3)
    
#     # Additional memory for operations
#     if backend == "hdbscan_raw":
#         # HDBSCAN needs distance matrix in worst case
#         clustering_memory = (n_samples ** 2) * 4 / (1024 ** 3)
#     elif backend == "umap_hdbscan":
#         # UMAP + reduced embeddings + HDBSCAN
#         umap_memory = base_memory * 2  # Approximate
#         reduced_memory = n_samples * reduced_dim * 4 / (1024 ** 3)
#         clustering_memory = (n_samples ** 2) * 4 / (1024 ** 3) * 0.1  # Approximate
#         clustering_memory = min(clustering_memory, 10)  # Cap at 10GB
#         base_memory = umap_memory + reduced_memory + clustering_memory
#     else:  # ipca_hdbscan
#         # IPCA + reduced embeddings + HDBSCAN
#         ipca_memory = base_memory * 1.5
#         reduced_memory = n_samples * reduced_dim * 4 / (1024 ** 3)
#         clustering_memory = (n_samples ** 2) * 4 / (1024 ** 3) * 0.1
#         clustering_memory = min(clustering_memory, 10)
#         base_memory = ipca_memory + reduced_memory + clustering_memory
    
#     # Add overhead (20%)
#     return base_memory * 1.2


# def free_gpu_memory():
#     """Free GPU memory"""
#     if GPU_AVAILABLE:
#         try:
#             mempool = cp.get_default_memory_pool()
#             pinned_mempool = cp.get_default_pinned_memory_pool()
#             mempool.free_all_blocks()
#             pinned_mempool.free_all_blocks()
#         except:
#             pass


# # ---------------- Utils ----------------

# def scan_chunk_dirs(root: Path, pattern: str = r"chunk_\d{4}") -> List[Path]:
#     """Scan for chunk directories matching pattern"""
#     rx = re.compile(pattern)
#     dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and rx.fullmatch(p.name)]
#     if not dirs:
#         raise FileNotFoundError(f"No chunk dirs matching /{pattern}/ under {root}")
#     return dirs


# def l2norm_inplace_gpu(X: np.ndarray, block: int = 200_000, debug_memory: bool = False) -> np.ndarray:
#     """L2 normalize using GPU if available"""
#     print_memory_info("Before L2 norm", debug_memory)
    
#     if GPU_AVAILABLE and X.size > 1000000:  # Use GPU for large arrays
#         print("[norm] Using GPU for L2 normalization")
#         try:
#             # Convert to CuPy array
#             X_gpu = cp.asarray(X, dtype=cp.float32)
            
#             # Normalize in blocks to manage GPU memory
#             n = X_gpu.shape[0]
#             for s in tqdm(range(0, n, block), desc="L2 norm (GPU)", unit="block"):
#                 e = min(n, s + block)
#                 B = X_gpu[s:e]
#                 norms = cp.linalg.norm(B, axis=1, keepdims=True)
#                 norms = cp.maximum(norms, 1e-12)  # Avoid division by zero
#                 B /= norms
            
#             # Convert back to CPU
#             result = cp.asnumpy(X_gpu)
#             free_gpu_memory()
#             print_memory_info("After L2 norm", debug_memory)
#             return result
            
#         except Exception as e:
#             print(f"GPU normalization failed: {e}, falling back to CPU")
#             free_gpu_memory()
#             return l2norm_inplace_cpu(X, block, debug_memory)
#     else:
#         return l2norm_inplace_cpu(X, block, debug_memory)


# def l2norm_inplace_cpu(X: np.ndarray, block: int = 200_000, debug_memory: bool = False) -> np.ndarray:
#     """CPU L2 normalization"""
#     print("[norm] Using CPU for L2 normalization")
#     n = X.shape[0]
    
#     for s in tqdm(range(0, n, block), desc="L2 norm (CPU)", unit="block"):
#         e = min(n, s + block)
#         B = X[s:e]
#         norms = np.linalg.norm(B, axis=1, keepdims=True)
#         norms = np.maximum(norms, 1e-12)  # Avoid division by zero
#         B /= norms
    
#     print_memory_info("After L2 norm", debug_memory)
#     return X


# def summarize(labels: np.ndarray) -> dict:
#     """Generate clustering summary statistics"""
#     # Convert to numpy if it's a CuPy array
#     if GPU_AVAILABLE and hasattr(labels, 'get'):
#         labels = labels.get()
    
#     labels = labels.astype(int)
#     n = len(labels)
#     noise = int((labels < 0).sum())
#     unique_labels = np.unique(labels[labels >= 0])
#     k = len(unique_labels)
    
#     # Calculate cluster sizes
#     cluster_sizes = {}
#     if k > 0:
#         for label in unique_labels:
#             size = int((labels == label).sum())
#             cluster_sizes[int(label)] = size
    
#     return {
#         "paragraphs": n,
#         "clusters": k,
#         "noise": noise,
#         "noise_pct": round(100.0 * noise / n, 2) if n else 0.0,
#         "avg_cluster_size": round(np.mean(list(cluster_sizes.values())), 2) if cluster_sizes else 0.0,
#         "median_cluster_size": int(np.median(list(cluster_sizes.values()))) if cluster_sizes else 0,
#         "min_cluster_size": int(min(cluster_sizes.values())) if cluster_sizes else 0,
#         "max_cluster_size": int(max(cluster_sizes.values())) if cluster_sizes else 0
#     }


# # ---------------- Loading (two-pass, memory-efficient) ----------------

# def count_rows_and_dim(chunk_dirs: List[Path], meta_name: str, emb_name: str, 
#                        sample_every: int) -> Tuple[int, int]:
#     """Count total rows and check embedding dimensions"""
#     total = 0
#     d = None
    
#     for cdir in tqdm(chunk_dirs, desc="Counting rows", unit="chunk"):
#         emb_path = cdir / emb_name
#         meta_path = cdir / meta_name
        
#         if not emb_path.exists():
#             raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
#         if not meta_path.exists():
#             raise FileNotFoundError(f"Missing metadata file: {meta_path}")
        
#         # Load embeddings header only
#         emb = np.load(emb_path, mmap_mode="r")
#         m, dim = emb.shape
        
#         if d is None:
#             d = dim
#         elif d != dim:
#             raise ValueError(f"Dimension mismatch in {cdir}: {dim} vs {d}")
        
#         # Verify metadata row count
#         rows = 0
#         for chunk in pd.read_csv(meta_path, usecols=["paper_id", "para_id"], chunksize=200_000):
#             rows += len(chunk)
        
#         if rows != m:
#             raise ValueError(f"Row mismatch in {cdir}: embeddings={m} != metadata={rows}")
        
#         # Count sampled rows
#         total += (m + sample_every - 1) // sample_every
    
#     return total, d


# def load_embeddings_and_metadata(chunk_dirs: List[Path], meta_name: str, emb_name: str, 
#                                 sample_every: int, debug_memory: bool = False,
#                                 memory_limit_gb: float = 0) -> Tuple[np.ndarray, List, List]:
#     """Load embeddings and metadata directly into memory arrays"""
#     print_memory_info("Before loading data", debug_memory)
    
#     # Count total size
#     N, d = count_rows_and_dim(chunk_dirs, meta_name, emb_name, sample_every)
#     print(f"[load] Total paragraphs (after sampling): {N:,}, dim={d}")
    
#     # Check memory limit
#     estimated_gb = N * d * 4 / (1024 ** 3)
#     print(f"[load] Estimated memory for embeddings: {estimated_gb:.2f}GB")
    
#     if memory_limit_gb > 0 and estimated_gb > memory_limit_gb:
#         raise MemoryError(f"Estimated memory ({estimated_gb:.2f}GB) exceeds limit ({memory_limit_gb}GB)")
    
#     # Create arrays
#     X = np.empty((N, d), dtype=np.float32)
#     ids = []
#     papers = []
    
#     # Fill arrays
#     i = 0
#     for cdir in tqdm(chunk_dirs, desc="Loading shards", unit="chunk"):
#         emb = np.load(cdir / emb_name, mmap_mode="r")
        
#         # Process metadata in chunks
#         base = 0
#         for df in pd.read_csv(cdir / meta_name, usecols=["paper_id", "para_id"], chunksize=200_000):
#             M = len(df)
            
#             # Sample rows
#             for j in range(0, M, sample_every):
#                 if i >= N:
#                     break
                    
#                 X[i] = emb[base + j]
#                 ids.append(df["para_id"].iat[j])
#                 papers.append(df["paper_id"].iat[j])
#                 i += 1
            
#             base += M
        
#         # Garbage collect after each chunk
#         gc.collect()
    
#     assert i == N, f"Filled {i} rows, expected {N}"
    
#     # print_memory_info("After loading data", debug_memory)
#     return X, ids, papers


# # ---------------- GPU DR backends ----------------

# def reduce_umap_gpu(X: np.ndarray, dim: int, n_neighbors: int, 
#                    debug_memory: bool = False) -> np.ndarray:
#     """GPU-accelerated UMAP using RAPIDS cuML"""
#     if not CUML_AVAILABLE:
#         print("CUML not available, falling back to CPU UMAP")
#         return reduce_umap_cpu(X, dim, n_neighbors, debug_memory)
    
#     try:
#         print_memory_info("Before GPU UMAP", debug_memory)
#         print(f"[GPU-UMAP] Processing {X.shape[0]:,} points, reducing to {dim} dimensions")
        
#         # Create GPU UMAP reducer with optimized settings
#         reducer = cuUMAP(
#             n_components=dim,
#             n_neighbors=min(n_neighbors, X.shape[0] - 1),
#             min_dist=0.05,
#             metric="cosine",
#             random_state=42,
#             verbose=True,
#             low_memory=True  # Enable low memory mode if available
#         )
        
#         # Convert to GPU and transform
#         if GPU_AVAILABLE:
#             X_gpu = cp.asarray(X, dtype=cp.float32)
#             Z_gpu = reducer.fit_transform(X_gpu)
#             result = cp.asnumpy(Z_gpu)
            
#             # Clean up GPU memory
#             del X_gpu, Z_gpu
#             free_gpu_memory()
#         else:
#             result = reducer.fit_transform(X)
        
#         print_memory_info("After GPU UMAP", debug_memory)
#         return result
        
#     except Exception as e:
#         print(f"GPU UMAP failed: {e}, falling back to CPU")
#         free_gpu_memory()
#         return reduce_umap_cpu(X, dim, n_neighbors, debug_memory)


# def reduce_umap_cpu(X: np.ndarray, dim: int, n_neighbors: int, 
#                    debug_memory: bool = False) -> np.ndarray:
#     """Fallback CPU UMAP"""
#     if not UMAP_AVAILABLE:
#         raise RuntimeError("umap-learn is not installed. Install with: pip install umap-learn")
    
#     print_memory_info("Before CPU UMAP", debug_memory)
#     print(f"[CPU-UMAP] Processing {X.shape[0]:,} points, reducing to {dim} dimensions")
    
#     reducer = umap.UMAP(
#         n_components=dim,
#         n_neighbors=min(n_neighbors, X.shape[0] - 1),
#         min_dist=0.05,
#         metric="cosine",
#         random_state=42,
#         verbose=True,
#         low_memory=True
#     )
    
#     result = reducer.fit_transform(X)
#     print_memory_info("After CPU UMAP", debug_memory)
#     return result


# def reduce_ipca(X: np.ndarray, dim: int, batch: int = 200_000, 
#                debug_memory: bool = False) -> np.ndarray:
#     """IncrementalPCA (CPU only)"""
#     print_memory_info("Before IPCA", debug_memory)
#     print(f"[IPCA] Processing {X.shape[0]:,} points, reducing to {dim} dimensions")
    
#     # Ensure dim doesn't exceed data dimensions
#     dim = min(dim, min(X.shape))
    
#     ip = IncrementalPCA(n_components=dim, batch_size=batch)
    
#     # Fit in batches
#     n = X.shape[0]
#     for s in tqdm(range(0, n, batch), desc="IPCA fit", unit="batch"):
#         e = min(n, s + batch)
#         ip.partial_fit(X[s:e])
    
#     # Transform in batches
#     Z = np.empty((n, dim), dtype=np.float32)
#     for s in tqdm(range(0, n, batch), desc="IPCA transform", unit="batch"):
#         e = min(n, s + batch)
#         Z[s:e] = ip.transform(X[s:e]).astype(np.float32)
    
#     print_memory_info("After IPCA", debug_memory)
#     return Z


# # ---------------- GPU Clustering ----------------

# def cluster_hdbscan_gpu(X: np.ndarray, min_cluster_size: int, min_samples: int, 
#                        metric: str = "euclidean", debug_memory: bool = False) -> np.ndarray:
#     """GPU-accelerated HDBSCAN using RAPIDS cuML"""
#     if not CUML_AVAILABLE:
#         print("CUML not available, falling back to CPU HDBSCAN")
#         return cluster_hdbscan_cpu(X, min_cluster_size, min_samples, metric, debug_memory)
    
#     try:
#         print_memory_info("Before GPU HDBSCAN", debug_memory)
#         print(f"[GPU-HDBSCAN] Clustering {X.shape[0]:,} points with metric={metric}")
        
#         clusterer = cuHDBSCAN(
#             min_cluster_size=min_cluster_size,
#             min_samples=min_samples,
#             metric=metric,
#             verbose=True,
#             prediction_data=False  # Save memory by not storing prediction data
#         )
        
#         # Convert to GPU and cluster
#         if GPU_AVAILABLE:
#             X_gpu = cp.asarray(X, dtype=cp.float32)
#             labels_gpu = clusterer.fit_predict(X_gpu)
#             result = cp.asnumpy(labels_gpu)
            
#             # Clean up GPU memory
#             del X_gpu, labels_gpu
#             free_gpu_memory()
#         else:
#             result = clusterer.fit_predict(X)
        
#         print_memory_info("After GPU HDBSCAN", debug_memory)
#         return result
        
#     except Exception as e:
#         print(f"GPU HDBSCAN failed: {e}, falling back to CPU")
#         free_gpu_memory()
#         return cluster_hdbscan_cpu(X, min_cluster_size, min_samples, metric, debug_memory)


# def cluster_hdbscan_cpu(X: np.ndarray, min_cluster_size: int, min_samples: int,
#                        metric: str = "euclidean", debug_memory: bool = False) -> np.ndarray:
#     """Fallback CPU HDBSCAN"""
#     if not HDBSCAN_AVAILABLE:
#         raise RuntimeError("hdbscan is not installed. Install with: pip install hdbscan")
    
#     print_memory_info("Before CPU HDBSCAN", debug_memory)
#     print(f"[CPU-HDBSCAN] Clustering {X.shape[0]:,} points with metric={metric}")
    
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=min_cluster_size,
#         min_samples=min_samples,
#         metric=metric,
#         core_dist_n_jobs=-1,  # Use all CPU cores
#         prediction_data=False  # Save memory
#     )
    
#     result = clusterer.fit_predict(X)
#     print_memory_info("After CPU HDBSCAN", debug_memory)
#     return result


# # ---------------- Main ----------------

# def main():
#     ap = argparse.ArgumentParser(description="GPU-accelerated paragraph embedding clustering")
    
#     # Input/output arguments
#     ap.add_argument("--chunks_root", required=True, help="Root directory containing chunk folders")
#     ap.add_argument("--work_dir", required=True, help="Directory for temp files and outputs")
    
#     # Data loading arguments
#     ap.add_argument("--chunk_pattern", default=r"chunk_\d{4}", help="Regex pattern for chunk folders")
#     ap.add_argument("--meta_name", default="metadata.csv", help="Name of metadata file in each chunk")
#     ap.add_argument("--emb_name", default="embeddings.npy", help="Name of embeddings file in each chunk")
#     ap.add_argument("--max_chunks", type=int, default=0, help="Use first N shard folders (0=all)")
#     ap.add_argument("--sample_every", type=int, default=1, help="Use every k-th paragraph row (for speed/memory)")
    
#     # Algorithm arguments
#     ap.add_argument("--backend", choices=["hdbscan_raw", "umap_hdbscan", "ipca_hdbscan"], 
#                    default="hdbscan_raw", help="Clustering pipeline backend")
#     ap.add_argument("--min_cluster_size", type=int, default=12, help="HDBSCAN min cluster size")
#     ap.add_argument("--min_samples", type=int, default=1, help="HDBSCAN min samples")
    
#     # Dimensionality reduction arguments
#     ap.add_argument("--umap_dim", type=int, default=50, help="UMAP target dimensions")
#     ap.add_argument("--n_neighbors", type=int, default=200, help="UMAP number of neighbors")
#     ap.add_argument("--ipca_dim", type=int, default=128, help="IncrementalPCA target dimensions")
    
#     # System arguments
#     ap.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
#     ap.add_argument("--memory_limit_gb", type=float, default=0, 
#                    help="Skip processing if estimated memory > limit (0=no limit)")
#     ap.add_argument("--debug_memory", action="store_true", help="Print detailed memory usage info")
    
#     args = ap.parse_args()
    
#     # Setup
#     work = Path(args.work_dir)
#     work.mkdir(parents=True, exist_ok=True)
    
#     print_memory_info("Script start", args.debug_memory)
    
#     # Check GPU availability
#     if args.force_cpu:
#         print("[GPU] Forcing CPU usage")
#         global GPU_AVAILABLE, CUML_AVAILABLE
#         GPU_AVAILABLE = False
#         CUML_AVAILABLE = False
#     else:
#         print(f"[GPU] GPU Available: {GPU_AVAILABLE}")
#         print(f"[GPU] CUML Available: {CUML_AVAILABLE}")
#         if GPU_AVAILABLE:
#             try:
#                 # Print GPU info
#                 for i in range(cp.cuda.runtime.getDeviceCount()):
#                     props = cp.cuda.runtime.getDeviceProperties(i)
#                     print(f"[GPU] Device {i}: {props['name'].decode()} ({props['totalGlobalMem'] / (1024**3):.1f}GB)")
#             except:
#                 pass
    
#     # Find shard folders
#     chunk_dirs = scan_chunk_dirs(Path(args.chunks_root), args.chunk_pattern)
#     if args.max_chunks > 0:
#         chunk_dirs = chunk_dirs[:args.max_chunks]
#     print(f"[scan] Using {len(chunk_dirs)} shard folders")
    
#     # Estimate memory usage
#     if args.memory_limit_gb > 0:
#         N_est, d_est = count_rows_and_dim(chunk_dirs[:min(3, len(chunk_dirs))], 
#                                          args.meta_name, args.emb_name, args.sample_every)
#         N_est = N_est * len(chunk_dirs) // min(3, len(chunk_dirs))  # Extrapolate
        
#         reduced_dim = args.umap_dim if args.backend == "umap_hdbscan" else args.ipca_dim
#         est_memory = estimate_memory_usage(N_est, d_est, args.backend, reduced_dim)
        
#         print(f"[memory] Estimated total memory usage: {est_memory:.2f}GB")
#         if est_memory > args.memory_limit_gb:
#             raise MemoryError(f"Estimated memory ({est_memory:.2f}GB) exceeds limit ({args.memory_limit_gb}GB)")
    
#     # Load embeddings and metadata
#     X, para_ids, paper_ids = load_embeddings_and_metadata(
#         chunk_dirs, args.meta_name, args.emb_name, args.sample_every,
#         debug_memory=args.debug_memory, memory_limit_gb=args.memory_limit_gb
#     )
    
#     # Normalize for cosine metrics
#     # print("[norm] L2-normalizing embeddings")
#     # X = l2norm_inplace_gpu(X, debug_memory=args.debug_memory)
#     print(f"Skipping normalization, going to the backend : {args.backend}")
    
#     # Dimension reduction (optional backend)
#     if args.backend == "hdbscan_raw":
#         Z = X  # cosine on original space
#         metric = "cosine"
        
#     elif args.backend == "umap_hdbscan":
#         print(f"[umap] dim={args.umap_dim} n_neighbors={args.n_neighbors}")
#         Z = reduce_umap_gpu(X, dim=args.umap_dim, n_neighbors=args.n_neighbors,
#                            debug_memory=args.debug_memory)
#         metric = "euclidean"
        
#     else:  # ipca_hdbscan
#         print(f"[ipca] dim={args.ipca_dim}")
#         Z = reduce_ipca(X, dim=args.ipca_dim, debug_memory=args.debug_memory)
#         metric = "euclidean"
    
#     # Free original embeddings if we did dimensionality reduction
#     if args.backend != "hdbscan_raw":
#         print_memory_info("Before freeing original embeddings", args.debug_memory)
#         del X  # Free memory
#         gc.collect()
#         if GPU_AVAILABLE:
#             free_gpu_memory()
#         print_memory_info("After freeing original embeddings", args.debug_memory)
    
#     # Cluster with HDBSCAN
#     print(f"[hdbscan] metric={metric} min_cluster_size={args.min_cluster_size} min_samples={args.min_samples}")
#     labels = cluster_hdbscan_gpu(Z, args.min_cluster_size, args.min_samples, metric,
#                                  debug_memory=args.debug_memory)
    
#     print_memory_info("After clustering", args.debug_memory)
    
#     # Generate summary
#     summ = summarize(labels)
#     print("\n" + "=" * 60)
#     print("CLUSTERING RESULTS")
#     print("=" * 60)
#     print(json.dumps(summ, indent=2))
#     print("=" * 60)
    
#     # Save results
#     out_labels = work / "labels_paragraph.csv"
#     df = pd.DataFrame({
#         "para_id": para_ids,
#         "paper_id": paper_ids,
#         "cluster": labels.astype(int)
#     })
#     df.to_csv(out_labels, index=False)
#     print(f"\n[write] Labels saved to: {out_labels}")
    
#     # Save summary
#     summary_path = work / "cluster_summary.json"
#     summary_path.write_text(json.dumps(summ, indent=2))
#     print(f"[write] Summary saved to: {summary_path}")
    
#     # Save configuration for reproducibility
#     config = {
#         "backend": args.backend,
#         "min_cluster_size": args.min_cluster_size,
#         "min_samples": args.min_samples,
#         "sample_every": args.sample_every,
#         "n_chunks": len(chunk_dirs),
#         "gpu_used": GPU_AVAILABLE and not args.force_cpu,
#         "cuml_used": CUML_AVAILABLE and not args.force_cpu
#     }
    
#     if args.backend == "umap_hdbscan":
#         config.update({"umap_dim": args.umap_dim, "n_neighbors": args.n_neighbors})
#     elif args.backend == "ipca_hdbscan":
#         config["ipca_dim"] = args.ipca_dim
    
#     config_path = work / "cluster_config.json"
#     config_path.write_text(json.dumps(config, indent=2))
#     print(f"[write] Config saved to: {config_path}")
    
#     print_memory_info("Script end", args.debug_memory)
#     print("\n✓ Clustering completed successfully!")


# if __name__ == "__main__":
#     main()
