#!/usr/bin/env python3
"""
Improved memory-optimized clustering script with methodological feature enhancement.
Addresses embedding diversity issues and over-normalization problems.
"""

import re
import gc
import json
import argparse
import psutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

# GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("Warning: CuPy not available, falling back to CPU")
    GPU_AVAILABLE = False
    cp = None

try:
    import cuml
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    print("Warning: CUML not available, falling back to CPU libraries")
    CUML_AVAILABLE = False
    cuml = None

# Fallback CPU libraries
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    print("Warning: hdbscan not available")
    HDBSCAN_AVAILABLE = False
    hdbscan = None

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("Warning: umap-learn not available")
    UMAP_AVAILABLE = False
    umap = None


# ---------------- Memory Management ----------------

def print_memory_info(stage: str, debug: bool = False):
    """Print memory usage information"""
    if not debug:
        return
    
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024 ** 3)
    
    if GPU_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            gpu_used_gb = mempool.used_bytes() / (1024 ** 3)
            gpu_total_gb = mempool.total_bytes() / (1024 ** 3)
            print(f"[{stage}] CPU: {rss_gb:.2f}GB | GPU: {gpu_used_gb:.2f}GB used / {gpu_total_gb:.2f}GB total")
        except:
            print(f"[{stage}] CPU: {rss_gb:.2f}GB")
    else:
        print(f"[{stage}] CPU: {rss_gb:.2f}GB")


def free_gpu_memory():
    """Free GPU memory"""
    if GPU_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except:
            pass


# ---------------- Methodological Feature Extraction ----------------

def extract_methodological_features(texts: List[str], chunk_size: int = 10000) -> np.ndarray:
    """Extract methodological features from texts to enhance embeddings"""
    print("[method] Extracting methodological features...")
    
    # Define methodological keywords with weights
    method_keywords = {
        # Observational methods
        'observational': {
            'keywords': ['spectroscopy', 'spectroscopic', 'spectrum', 'spectra', 'photometry', 
                        'photometric', 'interferometry', 'polarimetry', 'astrometry', 'imaging',
                        'observation', 'observed', 'telescope', 'survey', 'detection'],
            'weight': 2.0
        },
        # Computational methods
        'computational': {
            'keywords': ['simulation', 'simulate', 'simulated', 'monte carlo', 'n-body', 
                        'numerical', 'computational', 'code', 'algorithm', 'modeling',
                        'hydro', 'magnetohydrodynamic', 'radiative transfer'],
            'weight': 2.0
        },
        # Theoretical methods
        'theoretical': {
            'keywords': ['theoretical', 'theory', 'analytical', 'analytic', 'model', 
                        'formalism', 'equation', 'derivation', 'calculation', 'prediction'],
            'weight': 1.5
        },
        # Statistical methods
        'statistical': {
            'keywords': ['bayesian', 'statistical', 'statistics', 'regression', 'correlation',
                        'fitting', 'likelihood', 'mcmc', 'bootstrap', 'chi-squared', 'p-value'],
            'weight': 1.5
        },
        # Instruments
        'instruments': {
            'keywords': ['hubble', 'spitzer', 'chandra', 'jwst', 'keck', 'vlt', 'alma',
                        'gaia', 'kepler', 'tess', 'wise', 'galex', 'herschel'],
            'weight': 1.0
        },
        # Data types
        'data_types': {
            'keywords': ['lightcurve', 'light curve', 'time series', 'catalog', 'survey',
                        'image', 'magnitude', 'flux', 'brightness', 'redshift'],
            'weight': 1.0
        }
    }
    
    # Process texts in chunks to save memory
    all_features = []
    n_features = len(method_keywords)
    
    for chunk_start in tqdm(range(0, len(texts), chunk_size), desc="Extracting features"):
        chunk_end = min(chunk_start + chunk_size, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]
        
        chunk_features = np.zeros((len(chunk_texts), n_features), dtype=np.float32)
        
        for i, text in enumerate(chunk_texts):
            text_lower = text.lower()
            
            for j, (category, info) in enumerate(method_keywords.items()):
                # Count keyword occurrences with weight
                score = sum(text_lower.count(keyword) for keyword in info['keywords'])
                # Apply category weight and normalize by text length
                weighted_score = score * info['weight'] / max(len(text.split()), 1)
                chunk_features[i, j] = weighted_score
        
        all_features.append(chunk_features)
        
        # Memory cleanup
        if chunk_start % (chunk_size * 10) == 0:
            gc.collect()
    
    features = np.vstack(all_features)
    print(f"[method] Extracted {features.shape[1]} methodological features")
    return features


# ---------------- Memory-Mapped Data Structures ----------------

class MemmapEmbeddings:
    """Memory-mapped embeddings with chunked access"""
    
    def __init__(self, filepath: Path, shape: Tuple[int, int], dtype=np.float32):
        self.filepath = filepath
        self.shape = shape
        self.dtype = dtype
        self._mmap = None
    
    def __enter__(self):
        self._mmap = np.memmap(self.filepath, dtype=self.dtype, mode='w+', shape=self.shape)
        return self._mmap
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mmap is not None:
            del self._mmap
            self._mmap = None
        gc.collect()
    
    def read_only(self):
        """Return read-only memory-mapped array"""
        return np.memmap(self.filepath, dtype=self.dtype, mode='r', shape=self.shape)


def create_enhanced_embeddings_mmap(chunk_dirs: List[Path], meta_name: str, emb_name: str, 
                                  sample_every: int, work_dir: Path, 
                                  enhance_with_methods: bool = True,
                                  debug_memory: bool = False) -> Tuple[MemmapEmbeddings, List, List]:
    """Create enhanced embeddings with methodological features"""
    
    print_memory_info("Before consolidation", debug_memory)
    
    # First pass: count total size and collect texts if needed
    print("[mmap] Counting total samples and loading texts...")
    total_samples = 0
    d = None
    all_texts = [] if enhance_with_methods else None
    
    for cdir in tqdm(chunk_dirs, desc="Counting & text loading", unit="chunk"):
        emb_path = cdir / emb_name
        meta_path = cdir / meta_name
        
        if not emb_path.exists() or not meta_path.exists():
            continue
            
        # Get embedding dimensions
        emb = np.load(emb_path, mmap_mode="r")
        m, dim = emb.shape
        
        if d is None:
            d = dim
        elif d != dim:
            raise ValueError(f"Dimension mismatch in {cdir}: {dim} vs {d}")
        
        # Load texts if needed for method features
        if enhance_with_methods:
            chunk_texts = []
            for df_chunk in pd.read_csv(meta_path, chunksize=50_000):
                chunk_texts.extend(df_chunk["text"].tolist())
                if len(chunk_texts) >= m:
                    break
            
            # Sample texts corresponding to sampled embeddings
            for j in range(0, m, sample_every):
                if j < len(chunk_texts):
                    all_texts.append(chunk_texts[j])
        
        # Count sampled rows
        total_samples += (m + sample_every - 1) // sample_every
    
    # Extract methodological features if requested
    method_features = None
    enhanced_dim = d
    
    if enhance_with_methods and all_texts:
        print("[enhance] Extracting methodological features...")
        method_features = extract_methodological_features(all_texts)
        enhanced_dim = d + method_features.shape[1]
        print(f"[enhance] Enhanced embedding dim: {d} -> {enhanced_dim}")
    
    print(f"[mmap] Total samples: {total_samples:,}, enhanced dimensions: {enhanced_dim}")
    
    # Create memory-mapped file for enhanced embeddings
    mmap_path = work_dir / "enhanced_embeddings.dat"
    mmap_embeddings = MemmapEmbeddings(mmap_path, (total_samples, enhanced_dim))
    
    # Metadata lists
    para_ids = []
    paper_ids = []
    
    # Second pass: fill the memory-mapped array with enhanced embeddings
    print("[mmap] Consolidating enhanced embeddings...")
    
    with mmap_embeddings as X_mmap:
        current_idx = 0
        
        for cdir in tqdm(chunk_dirs, desc="Consolidating", unit="chunk"):
            emb_path = cdir / emb_name
            meta_path = cdir / meta_name
            
            # Load embeddings with memory mapping
            emb = np.load(emb_path, mmap_mode="r")
            
            # Process metadata in chunks
            base_row = 0
            for df_chunk in pd.read_csv(meta_path, chunksize=50_000):
                chunk_size = len(df_chunk)
                
                # Sample rows from this chunk
                for j in range(0, chunk_size, sample_every):
                    if current_idx >= total_samples:
                        break
                    
                    # Copy original embedding
                    X_mmap[current_idx, :d] = emb[base_row + j]
                    
                    # Add methodological features if available
                    if method_features is not None:
                        X_mmap[current_idx, d:] = method_features[current_idx]
                    
                    # Store metadata
                    para_ids.append(df_chunk["para_id"].iat[j])
                    paper_ids.append(df_chunk["paper_id"].iat[j])
                    
                    current_idx += 1
                
                base_row += chunk_size
            
            # Force garbage collection after each chunk
            del emb
            gc.collect()
    
    print_memory_info("After consolidation", debug_memory)
    print(f"[mmap] Consolidated {current_idx:,} enhanced embeddings to {mmap_path}")
    
    return mmap_embeddings, para_ids, paper_ids


def chunked_array_iterator(X: np.ndarray, chunk_size: int) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Iterate over array in chunks, yielding (start_idx, end_idx, chunk)"""
    n = X.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, end, X[start:end]


# ---------------- Smart Normalization ----------------

def smart_normalize_chunked(X_mmap: np.ndarray, chunk_size: int = 100_000, 
                           norm_threshold: float = 0.05, debug_memory: bool = False) -> bool:
    """Smart normalization that checks if normalization is needed"""
    print_memory_info("Before smart normalization check", debug_memory)
    
    # Sample to check if normalization is needed
    n = X_mmap.shape[0]
    sample_size = min(5000, n)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    sample_embeddings = X_mmap[sample_indices]
    
    # Check embedding magnitude diversity
    norms = np.linalg.norm(sample_embeddings, axis=1)
    norm_std = np.std(norms)
    norm_mean = np.mean(norms)
    
    print(f"[norm] Embedding norms - Mean: {norm_mean:.4f}, Std: {norm_std:.4f}")
    
    if norm_std < norm_threshold:
        print(f"[norm] Skipping normalization - embeddings already have similar magnitudes (std={norm_std:.4f})")
        return False
    
    print(f"[norm] Proceeding with normalization - diverse magnitudes detected")
    
    if GPU_AVAILABLE and n > 1_000_000:
        print("[norm] Using GPU for normalization")
        try:
            for start, end, chunk in tqdm(chunked_array_iterator(X_mmap, chunk_size), 
                                        total=(n + chunk_size - 1) // chunk_size,
                                        desc="GPU L2 norm", unit="chunk"):
                chunk_gpu = cp.asarray(chunk, dtype=cp.float32)
                norms = cp.linalg.norm(chunk_gpu, axis=1, keepdims=True)
                norms = cp.maximum(norms, 1e-12)
                chunk_gpu /= norms
                
                X_mmap[start:end] = cp.asnumpy(chunk_gpu)
                
                del chunk_gpu, norms
                if start % (chunk_size * 10) == 0:
                    free_gpu_memory()
            
            free_gpu_memory()
            return True
        except Exception as e:
            print(f"GPU normalization failed: {e}, falling back to CPU")
            return smart_normalize_cpu(X_mmap, chunk_size, debug_memory)
    else:
        return smart_normalize_cpu(X_mmap, chunk_size, debug_memory)


def smart_normalize_cpu(X_mmap: np.ndarray, chunk_size: int, debug_memory: bool) -> bool:
    """CPU chunked normalization"""
    print("[norm] Using CPU for normalization")
    n = X_mmap.shape[0]
    
    for start, end, chunk in tqdm(chunked_array_iterator(X_mmap, chunk_size), 
                                total=(n + chunk_size - 1) // chunk_size,
                                desc="CPU L2 norm", unit="chunk"):
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        chunk /= norms
        X_mmap[start:end] = chunk
    
    return True


# ---------------- Improved Dimensionality Reduction ----------------

def reduce_umap_improved(X_mmap: np.ndarray, dim: int, n_neighbors: int,
                        chunk_size: int = 200_000, debug_memory: bool = False) -> np.ndarray:
    """Improved UMAP with better parameters for large-scale data"""
    if not CUML_AVAILABLE and not UMAP_AVAILABLE:
        raise RuntimeError("Neither CUML nor UMAP is available")
    
    print_memory_info("Before improved UMAP", debug_memory)
    n = X_mmap.shape[0]
    
    # Adjust parameters for better clustering
    adjusted_neighbors = min(n_neighbors, max(15, n // 1000))  # More conservative neighbors
    
    print(f"[UMAP] Processing {n:,} points, reducing to {dim} dimensions")
    print(f"[UMAP] Using n_neighbors={adjusted_neighbors} (adjusted from {n_neighbors})")
    
    if CUML_AVAILABLE and GPU_AVAILABLE:
        return reduce_umap_gpu_improved(X_mmap, dim, adjusted_neighbors, chunk_size, debug_memory)
    else:
        return reduce_umap_cpu_improved(X_mmap, dim, adjusted_neighbors, chunk_size, debug_memory)


def reduce_umap_gpu_improved(X_mmap: np.ndarray, dim: int, n_neighbors: int,
                           chunk_size: int, debug_memory: bool) -> np.ndarray:
    """GPU UMAP with improved parameters"""
    try:
        n = X_mmap.shape[0]
        
        # For large datasets, use subsampling for fitting
        if n > 300_000:
            print(f"[GPU-UMAP] Large dataset, using subsample fitting approach")
            subsample_size = min(150_000, n)
            indices = np.random.choice(n, subsample_size, replace=False)
            X_sample = X_mmap[indices]
            
            reducer = cuUMAP(
                n_components=dim,
                n_neighbors=min(n_neighbors, subsample_size - 1),
                min_dist=0.05,  # Slightly higher for better separation
                spread=1.5,     # Better spread
                metric="cosine",
                random_state=42,
                verbose=True
            )
            
            if GPU_AVAILABLE:
                X_sample_gpu = cp.asarray(X_sample, dtype=cp.float32)
                reducer.fit(X_sample_gpu)
                del X_sample_gpu
            else:
                reducer.fit(X_sample)
            
            del X_sample
            gc.collect()
            
            # Transform in chunks
            Z = np.empty((n, dim), dtype=np.float32)
            for start, end, chunk in tqdm(chunked_array_iterator(X_mmap, chunk_size), 
                                        total=(n + chunk_size - 1) // chunk_size,
                                        desc="UMAP transform", unit="chunk"):
                if GPU_AVAILABLE:
                    chunk_gpu = cp.asarray(chunk, dtype=cp.float32)
                    Z_chunk_gpu = reducer.transform(chunk_gpu)
                    Z[start:end] = cp.asnumpy(Z_chunk_gpu)
                    del chunk_gpu, Z_chunk_gpu
                else:
                    Z[start:end] = reducer.transform(chunk)
                
                if start % (chunk_size * 5) == 0:
                    free_gpu_memory()
                    gc.collect()
        else:
            # Direct processing for smaller datasets
            reducer = cuUMAP(
                n_components=dim,
                n_neighbors=min(n_neighbors, n - 1),
                min_dist=0.05,
                spread=1.5,
                metric="cosine",
                random_state=42,
                verbose=True
            )
            
            if GPU_AVAILABLE:
                X_gpu = cp.asarray(X_mmap[:], dtype=cp.float32)
                Z_gpu = reducer.fit_transform(X_gpu)
                Z = cp.asnumpy(Z_gpu)
                del X_gpu, Z_gpu
            else:
                Z = reducer.fit_transform(X_mmap[:])
        
        free_gpu_memory()
        print_memory_info("After improved GPU UMAP", debug_memory)
        return Z
        
    except Exception as e:
        print(f"GPU UMAP failed: {e}, falling back to CPU")
        free_gpu_memory()
        return reduce_umap_cpu_improved(X_mmap, dim, n_neighbors, chunk_size, debug_memory)


def reduce_umap_cpu_improved(X_mmap: np.ndarray, dim: int, n_neighbors: int,
                           chunk_size: int, debug_memory: bool = False) -> np.ndarray:
    """CPU UMAP with improved parameters"""
    if not UMAP_AVAILABLE:
        raise RuntimeError("umap-learn is not installed")
    
    print_memory_info("Before CPU UMAP", debug_memory)
    n = X_mmap.shape[0]
    
    try:
        # For CPU, try to load all data if possible
        if n <= 500_000:  # Conservative limit for CPU
            print(f"[CPU-UMAP] Loading {n:,} points into memory")
            X_full = X_mmap[:]
            
            reducer = umap.UMAP(
                n_components=dim,
                n_neighbors=min(n_neighbors, n - 1),
                min_dist=0.05,
                spread=1.5,
                metric="cosine",
                random_state=42,
                verbose=True,
                low_memory=True
            )
            
            Z = reducer.fit_transform(X_full)
            del X_full
        else:
            # Subsampling approach for very large datasets
            print(f"[CPU-UMAP] Using subsample approach for large dataset")
            subsample_size = min(100_000, n)
            indices = np.random.choice(n, subsample_size, replace=False)
            X_sample = X_mmap[indices]
            
            reducer = umap.UMAP(
                n_components=dim,
                n_neighbors=min(n_neighbors, subsample_size - 1),
                min_dist=0.05,
                spread=1.5,
                metric="cosine",
                random_state=42,
                verbose=True,
                low_memory=True
            )
            
            reducer.fit(X_sample)
            del X_sample
            
            # Transform in smaller chunks
            Z = np.empty((n, dim), dtype=np.float32)
            small_chunk_size = min(chunk_size // 4, 25_000)
            
            for start, end, chunk in tqdm(chunked_array_iterator(X_mmap, small_chunk_size), 
                                        total=(n + small_chunk_size - 1) // small_chunk_size,
                                        desc="CPU UMAP transform", unit="chunk"):
                Z[start:end] = reducer.transform(chunk)
                gc.collect()
        
        print_memory_info("After improved CPU UMAP", debug_memory)
        return Z
        
    except MemoryError as e:
        print(f"CPU UMAP memory error: {e}")
        raise RuntimeError("Not enough memory for UMAP processing. Try reducing sample size or using GPU.")


# ---------------- Clustering (unchanged but with better parameters) ----------------

def cluster_hdbscan_improved(X: np.ndarray, min_cluster_size: int, min_samples: int, 
                           metric: str = "euclidean", debug_memory: bool = False) -> np.ndarray:
    """Improved HDBSCAN with better parameters"""
    if not CUML_AVAILABLE and not HDBSCAN_AVAILABLE:
        raise RuntimeError("Neither CUML nor HDBSCAN is available")
    
    # Adjust parameters for large-scale data
    n = X.shape[0]
    adjusted_min_cluster_size = max(min_cluster_size, n // 10000)  # At least 0.01% of data
    adjusted_min_samples = max(min_samples, adjusted_min_cluster_size // 10)
    
    print(f"[HDBSCAN] Adjusted parameters - min_cluster_size: {adjusted_min_cluster_size}, min_samples: {adjusted_min_samples}")
    
    if CUML_AVAILABLE and GPU_AVAILABLE:
        return cluster_hdbscan_gpu(X, adjusted_min_cluster_size, adjusted_min_samples, metric, debug_memory)
    else:
        return cluster_hdbscan_cpu(X, adjusted_min_cluster_size, adjusted_min_samples, metric, debug_memory)


def cluster_hdbscan_gpu(X: np.ndarray, min_cluster_size: int, min_samples: int, 
                       metric: str = "euclidean", debug_memory: bool = False) -> np.ndarray:
    """GPU HDBSCAN clustering"""
    try:
        print_memory_info("Before GPU HDBSCAN", debug_memory)
        print(f"[GPU-HDBSCAN] Clustering {X.shape[0]:,} points with metric={metric}")
        
        clusterer = cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=0.05,  # Add epsilon for better merging
            verbose=True,
            prediction_data=False
        )
        
        if GPU_AVAILABLE:
            X_gpu = cp.asarray(X, dtype=cp.float32)
            labels_gpu = clusterer.fit_predict(X_gpu)
            result = cp.asnumpy(labels_gpu)
            del X_gpu, labels_gpu
            free_gpu_memory()
        else:
            result = clusterer.fit_predict(X)
        
        print_memory_info("After GPU HDBSCAN", debug_memory)
        return result
        
    except Exception as e:
        print(f"GPU HDBSCAN failed: {e}, falling back to CPU")
        free_gpu_memory()
        return cluster_hdbscan_cpu(X, min_cluster_size, min_samples, metric, debug_memory)


def cluster_hdbscan_cpu(X: np.ndarray, min_cluster_size: int, min_samples: int,
                       metric: str = "euclidean", debug_memory: bool = False) -> np.ndarray:
    """CPU HDBSCAN clustering"""
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("hdbscan is not installed")
    
    print_memory_info("Before CPU HDBSCAN", debug_memory)
    print(f"[CPU-HDBSCAN] Clustering {X.shape[0]:,} points with metric={metric}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=0.05,
        cluster_selection_method='leaf',  # Better for imbalanced data
        core_dist_n_jobs=-1,
        prediction_data=False
    )
    
    result = clusterer.fit_predict(X)
    print_memory_info("After CPU HDBSCAN", debug_memory)
    return result


# ---------------- Utils ----------------

def scan_chunk_dirs(root: Path, pattern: str = r"chunk_\d{4}") -> List[Path]:
    """Scan for chunk directories matching pattern"""
    rx = re.compile(pattern)
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and rx.fullmatch(p.name)]
    if not dirs:
        raise FileNotFoundError(f"No chunk dirs matching /{pattern}/ under {root}")
    return dirs


def summarize(labels: np.ndarray) -> dict:
    """Generate clustering summary statistics"""
    if GPU_AVAILABLE and hasattr(labels, 'get'):
        labels = labels.get()
    
    labels = labels.astype(int)
    n = len(labels)
    noise = int((labels < 0).sum())
    unique_labels = np.unique(labels[labels >= 0])
    k = len(unique_labels)
    
    cluster_sizes = {}
    if k > 0:
        for label in unique_labels:
            size = int((labels == label).sum())
            cluster_sizes[int(label)] = size
    
    return {
        "paragraphs": n,
        "clusters": k,
        "noise": noise,
        "noise_pct": round(100.0 * noise / n, 2) if n else 0.0,
        "avg_cluster_size": round(np.mean(list(cluster_sizes.values())), 2) if cluster_sizes else 0.0,
        "median_cluster_size": int(np.median(list(cluster_sizes.values()))) if cluster_sizes else 0,
        "min_cluster_size": int(min(cluster_sizes.values())) if cluster_sizes else 0,
        "max_cluster_size": int(max(cluster_sizes.values())) if cluster_sizes else 0
    }


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Improved memory-optimized clustering with methodological enhancement")
    
    # Input/output arguments
    ap.add_argument("--chunks_root", required=True, help="Root directory containing chunk folders")
    ap.add_argument("--work_dir", required=True, help="Directory for temp files and outputs")
    
    # Data loading arguments
    ap.add_argument("--chunk_pattern", default=r"chunk_\d{4}", help="Regex pattern for chunk folders")
    ap.add_argument("--meta_name", default="metadata.csv", help="Name of metadata file in each chunk")
    ap.add_argument("--emb_name", default="embeddings.npy", help="Name of embeddings file in each chunk")
    ap.add_argument("--max_chunks", type=int, default=0, help="Use first N shard folders (0=all)")
    ap.add_argument("--sample_every", type=int, default=1, help="Use every k-th paragraph row")
    
    # Enhancement options
    ap.add_argument("--enhance_with_methods", action="store_true", default=True, 
                   help="Add methodological features to embeddings")
    ap.add_argument("--skip_normalization", action="store_true", default=False,
                   help="Skip embedding normalization entirely")
    
    # Algorithm arguments
    ap.add_argument("--backend", choices=["hdbscan_raw", "umap_hdbscan", "ipca_hdbscan", "streaming_umap_hdbscan"], 
                   default="umap_hdbscan", help="Clustering pipeline backend")
    ap.add_argument("--min_cluster_size", type=int, default=500, help="HDBSCAN min cluster size (improved default)")
    ap.add_argument("--min_samples", type=int, default=50, help="HDBSCAN min samples (improved default)")
    
    # Dimensionality reduction arguments
    ap.add_argument("--umap_dim", type=int, default=50, help="UMAP target dimensions (improved default)")
    ap.add_argument("--n_neighbors", type=int, default=50, help="UMAP number of neighbors (improved default)")
    ap.add_argument("--ipca_dim", type=int, default=128, help="IncrementalPCA target dimensions")
    
    # Memory management arguments
    ap.add_argument("--chunk_size", type=int, default=100_000, help="Processing chunk size")
    ap.add_argument("--use_mmap_output", action="store_true", help="Use memory mapping for dimensionality reduction output")
    
    # System arguments
    ap.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    ap.add_argument("--debug_memory", action="store_true", help="Print detailed memory usage info")
    
    args = ap.parse_args()
    
    # Setup
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    
    print_memory_info("Script start", args.debug_memory)
    
    # Check GPU availability
    if args.force_cpu:
        print("[GPU] Forcing CPU usage")
        global GPU_AVAILABLE, CUML_AVAILABLE
        GPU_AVAILABLE = False
        CUML_AVAILABLE = False
    else:
        print(f"[GPU] GPU Available: {GPU_AVAILABLE}")
        print(f"[GPU] CUML Available: {CUML_AVAILABLE}")
    
    # Find shard folders
    chunk_dirs = scan_chunk_dirs(Path(args.chunks_root), args.chunk_pattern)
    if args.max_chunks > 0:
        chunk_dirs = chunk_dirs[:args.max_chunks]
    print(f"[scan] Using {len(chunk_dirs)} shard folders")
    
    # Create enhanced memory-mapped embeddings
    print(f"[enhance] Method enhancement: {args.enhance_with_methods}")
    mmap_embeddings, para_ids, paper_ids = create_enhanced_embeddings_mmap(
        chunk_dirs, args.meta_name, args.emb_name, args.sample_every, work, 
        args.enhance_with_methods, args.debug_memory
    )
    
    try:
        # Get read-only access to the memory-mapped embeddings
        X_mmap = mmap_embeddings.read_only()
        print(f"[mmap] Loaded enhanced embeddings shape: {X_mmap.shape}")
        
        # Smart normalization (or skip if requested)
        if args.skip_normalization:
            print("[norm] Skipping normalization as requested")
            Z = X_mmap
            metric = "euclidean"  # Use euclidean for non-normalized embeddings
        elif args.backend == "hdbscan_raw":
            print("[norm] Checking if normalization is needed...")
            # We need write access for normalization
            with mmap_embeddings as X_write:
                was_normalized = smart_normalize_chunked(X_write, args.chunk_size, debug_memory=args.debug_memory)
            
            # Get fresh read-only access
            X_mmap = mmap_embeddings.read_only()
            Z = X_mmap
            metric = "cosine" if was_normalized else "euclidean"
            print(f"[norm] Using metric: {metric}")
            
        elif args.backend == "umap_hdbscan":
            print(f"[umap] dim={args.umap_dim} n_neighbors={args.n_neighbors}")
            Z = reduce_umap_improved(X_mmap, dim=args.umap_dim, n_neighbors=args.n_neighbors,
                                   chunk_size=args.chunk_size, debug_memory=args.debug_memory)
            metric = "euclidean"
            
        else:  # ipca_hdbscan
            print(f"[ipca] dim={args.ipca_dim}")
            from sklearn.decomposition import IncrementalPCA
            
            # Use the existing reduce_ipca function with improved parameters
            output_dir = work if args.use_mmap_output else None
            Z = reduce_ipca_chunked(X_mmap, dim=args.ipca_dim, batch_size=args.chunk_size,
                                  work_dir=output_dir, debug_memory=args.debug_memory)
            metric = "euclidean"
        
        # Clean up original mmap if we did dimensionality reduction
        if args.backend != "hdbscan_raw":
            del X_mmap
            # Clean up the temporary file
            if mmap_embeddings.filepath.exists():
                mmap_embeddings.filepath.unlink()
                print(f"[cleanup] Removed intermediate file: {mmap_embeddings.filepath}")
        
        # Improved clustering with HDBSCAN
        print(f"[hdbscan] metric={metric} min_cluster_size={args.min_cluster_size}")
        labels = cluster_hdbscan_improved(Z, args.min_cluster_size, args.min_samples, 
                                        metric, args.debug_memory)
        
        # Generate and save results
        summ = summarize(labels)
        print("\n" + "=" * 60)
        print("IMPROVED CLUSTERING RESULTS")
        print("=" * 60)
        print(json.dumps(summ, indent=2))
        print("=" * 60)
        
        # Save results
        out_labels = work / "labels_paragraph.csv"
        df = pd.DataFrame({
            "para_id": para_ids,
            "paper_id": paper_ids,
            "cluster": labels.astype(int)
        })
        df.to_csv(out_labels, index=False)
        print(f"\n[write] Labels saved to: {out_labels}")
        
        # Save summary
        summary_path = work / "cluster_summary.json"
        summary_path.write_text(json.dumps(summ, indent=2))
        print(f"[write] Summary saved to: {summary_path}")
        
        # Save configuration for reproducibility
        config = {
            "backend": args.backend,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "sample_every": args.sample_every,
            "n_chunks": len(chunk_dirs),
            "chunk_size": args.chunk_size,
            "enhance_with_methods": args.enhance_with_methods,
            "skip_normalization": args.skip_normalization,
            "gpu_used": GPU_AVAILABLE and not args.force_cpu,
            "cuml_used": CUML_AVAILABLE and not args.force_cpu,
            "memory_optimized": True,
            "improvements_applied": [
                "methodological_feature_enhancement",
                "smart_normalization",
                "improved_parameters",
                "large_scale_optimizations"
            ]
        }
        
        if args.backend == "umap_hdbscan":
            config.update({"umap_dim": args.umap_dim, "n_neighbors": args.n_neighbors})
        elif args.backend == "ipca_hdbscan":
            config.update({"ipca_dim": args.ipca_dim, "use_mmap_output": args.use_mmap_output})
        
        config_path = work / "cluster_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        print(f"[write] Config saved to: {config_path}")
        
        # Print improvement summary
        print("\n" + "=" * 60)
        print("IMPROVEMENTS APPLIED")
        print("=" * 60)
        print("✓ Methodological feature enhancement")
        print("✓ Smart normalization (avoids over-normalization)")
        print("✓ Improved UMAP/HDBSCAN parameters")
        print("✓ Large-scale data optimizations")
        print("✓ Better memory management")
        print("=" * 60)
        
    finally:
        # Cleanup temporary files
        if mmap_embeddings.filepath.exists():
            try:
                mmap_embeddings.filepath.unlink()
                print(f"[cleanup] Removed temporary file: {mmap_embeddings.filepath}")
            except:
                pass
    
    print_memory_info("Script end", args.debug_memory)
    print("\n✓ Improved clustering completed successfully!")


def reduce_ipca_chunked(X_mmap: np.ndarray, dim: int, batch_size: int = 100_000,
                       work_dir: Path = None, debug_memory: bool = False) -> np.ndarray:
    """IncrementalPCA with chunked processing and memory mapping for output"""
    print_memory_info("Before chunked IPCA", debug_memory)
    
    n, original_dim = X_mmap.shape
    dim = min(dim, min(X_mmap.shape))
    
    print(f"[IPCA] Processing {n:,} points, reducing from {original_dim} to {dim} dimensions")
    
    # Create IncrementalPCA
    ipca = IncrementalPCA(n_components=dim, batch_size=batch_size)
    
    # Fit in batches
    print("[IPCA] Fitting model...")
    for start, end, chunk in tqdm(chunked_array_iterator(X_mmap, batch_size), 
                                total=(n + batch_size - 1) // batch_size,
                                desc="IPCA fit", unit="batch"):
        ipca.partial_fit(chunk)
        gc.collect()
    
    # Create output memory-mapped array if work_dir provided
    if work_dir:
        output_path = work_dir / "ipca_reduced.dat"
        Z_mmap = MemmapEmbeddings(output_path, (n, dim))
        
        with Z_mmap as Z:
            # Transform in batches
            print("[IPCA] Transforming data...")
            for start, end, chunk in tqdm(chunked_array_iterator(X_mmap, batch_size), 
                                        total=(n + batch_size - 1) // batch_size,
                                        desc="IPCA transform", unit="batch"):
                Z[start:end] = ipca.transform(chunk).astype(np.float32)
                gc.collect()
        
        result = Z_mmap.read_only()
    else:
        # Transform to regular array (memory permitting)
        Z = np.empty((n, dim), dtype=np.float32)
        for start, end, chunk in tqdm(chunked_array_iterator(X_mmap, batch_size), 
                                    total=(n + batch_size - 1) // batch_size,
                                    desc="IPCA transform", unit="batch"):
            Z[start:end] = ipca.transform(chunk).astype(np.float32)
            gc.collect()
        result = Z
    
    print_memory_info("After chunked IPCA", debug_memory)
    return result


if __name__ == "__main__":
    main()