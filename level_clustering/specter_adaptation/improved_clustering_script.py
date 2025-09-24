#!/usr/bin/env python3
"""
True streaming clustering script for large-scale embeddings.
Processes chunks individually without loading all data into memory.
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

# Libraries
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not available")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available")


def print_memory_info(stage: str, debug: bool = False):
    """Print memory usage information"""
    if not debug:
        return
    
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024 ** 3)
    print(f"[{stage}] CPU Memory: {rss_gb:.2f}GB")


def extract_methodological_features(texts: List[str]) -> np.ndarray:
    """Extract methodological features from texts"""
    method_keywords = {
        'observational': ['spectroscopy', 'spectroscopic', 'spectrum', 'spectra', 'photometry', 
                         'photometric', 'interferometry', 'polarimetry', 'astrometry', 'imaging',
                         'observation', 'observed', 'telescope', 'survey', 'detection'],
        'computational': ['simulation', 'simulate', 'simulated', 'monte carlo', 'n-body', 
                         'numerical', 'computational', 'code', 'algorithm', 'modeling',
                         'hydro', 'magnetohydrodynamic', 'radiative transfer'],
        'theoretical': ['theoretical', 'theory', 'analytical', 'analytic', 'model', 
                       'formalism', 'equation', 'derivation', 'calculation', 'prediction'],
        'statistical': ['bayesian', 'statistical', 'statistics', 'regression', 'correlation',
                       'fitting', 'likelihood', 'mcmc', 'bootstrap', 'chi-squared', 'p-value'],
        'instruments': ['hubble', 'spitzer', 'chandra', 'jwst', 'keck', 'vlt', 'alma',
                       'gaia', 'kepler', 'tess', 'wise', 'galex', 'herschel'],
        'data_types': ['lightcurve', 'light curve', 'time series', 'catalog', 'survey',
                      'image', 'magnitude', 'flux', 'brightness', 'redshift']
    }
    
    features = np.zeros((len(texts), len(method_keywords)), dtype=np.float32)
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        text_len = max(len(text.split()), 1)
        
        for j, (category, keywords) in enumerate(method_keywords.items()):
            # Count keyword occurrences, normalized by text length
            score = sum(text_lower.count(keyword) for keyword in keywords)
            features[i, j] = score / text_len
    
    return features


def scan_chunk_dirs(root: Path, pattern: str = r"chunk_\d{4}") -> List[Path]:
    """Scan for chunk directories matching pattern"""
    rx = re.compile(pattern)
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and rx.fullmatch(p.name)]
    if not dirs:
        raise FileNotFoundError(f"No chunk dirs matching /{pattern}/ under {root}")
    return dirs


def count_total_samples(chunk_dirs: List[Path], emb_name: str, sample_every: int) -> int:
    """Count total samples across all chunks"""
    total = 0
    for cdir in tqdm(chunk_dirs, desc="Counting samples"):
        emb_path = cdir / emb_name
        if emb_path.exists():
            emb = np.load(emb_path, mmap_mode="r")
            total += (emb.shape[0] + sample_every - 1) // sample_every
    return total


def create_umap_model_from_sample(chunk_dirs: List[Path], meta_name: str, emb_name: str,
                                 sample_every: int, enhance_with_methods: bool,
                                 umap_dim: int, n_neighbors: int, max_sample: int = 50000,
                                 debug_memory: bool = False):
    """Create and fit UMAP model on a representative sample"""
    print("[UMAP] Creating model from sample data...")
    print_memory_info("Before UMAP sampling", debug_memory)
    
    if not UMAP_AVAILABLE:
        raise RuntimeError("UMAP not available for streaming pipeline")
    
    # Collect sample data
    sample_embeddings = []
    samples_collected = 0
    target_per_chunk = max_sample // min(len(chunk_dirs), 20)  # Use up to 20 chunks
    
    for i, cdir in enumerate(tqdm(chunk_dirs[:20], desc="Sampling for UMAP")):
        if samples_collected >= max_sample:
            break
            
        emb_path = cdir / emb_name
        meta_path = cdir / meta_name
        
        if not emb_path.exists() or not meta_path.exists():
            continue
        
        # Load chunk
        embeddings = np.load(emb_path)
        
        # Sample from chunk
        chunk_size = embeddings.shape[0]
        if sample_every > 1:
            indices = np.arange(0, chunk_size, sample_every)
            embeddings = embeddings[indices]
        
        # Further subsample for UMAP fitting
        remaining_budget = max_sample - samples_collected
        chunk_sample_size = min(target_per_chunk, len(embeddings), remaining_budget)
        
        if chunk_sample_size > 0:
            sample_indices = np.random.choice(len(embeddings), chunk_sample_size, replace=False)
            chunk_embeddings = embeddings[sample_indices]
            
            # Add method features if requested
            if enhance_with_methods:
                # Load corresponding texts
                texts = []
                for df_chunk in pd.read_csv(meta_path, chunksize=10000):
                    texts.extend(df_chunk["text"].tolist())
                    if len(texts) >= chunk_size:
                        break
                
                if sample_every > 1:
                    sampled_indices = np.arange(0, len(texts), sample_every)
                    texts = [texts[i] for i in sampled_indices if i < len(texts)]
                
                chunk_texts = [texts[i] for i in sample_indices if i < len(texts)]
                method_features = extract_methodological_features(chunk_texts)
                chunk_embeddings = np.hstack([chunk_embeddings, method_features])
            
            sample_embeddings.append(chunk_embeddings)
            samples_collected += chunk_sample_size
        
        # Clean up
        del embeddings
        gc.collect()
    
    if not sample_embeddings:
        raise RuntimeError("No sample data collected for UMAP fitting")
    
    # Combine samples and fit UMAP
    X_sample = np.vstack(sample_embeddings)
    print(f"[UMAP] Fitting on {len(X_sample)} samples with {X_sample.shape[1]} dimensions")
    
    reducer = umap.UMAP(
        n_components=umap_dim,
        n_neighbors=min(n_neighbors, len(X_sample) - 1),
        min_dist=0.05,
        spread=1.5,
        metric="cosine",
        random_state=42,
        verbose=True
    )
    
    reducer.fit(X_sample)
    
    # Clean up
    del X_sample, sample_embeddings
    gc.collect()
    
    print_memory_info("After UMAP fitting", debug_memory)
    return reducer


def stream_transform_chunks(chunk_dirs: List[Path], meta_name: str, emb_name: str,
                           sample_every: int, enhance_with_methods: bool,
                           umap_reducer, work_dir: Path, debug_memory: bool = False):
    """Stream through chunks, transform each one, and save to temporary files"""
    print("[STREAM] Transforming chunks individually...")
    print_memory_info("Before streaming transform", debug_memory)
    
    temp_files = []
    all_para_ids = []
    all_paper_ids = []
    total_transformed = 0
    
    for i, cdir in enumerate(tqdm(chunk_dirs, desc="Streaming transform")):
        emb_path = cdir / emb_name
        meta_path = cdir / meta_name
        
        if not emb_path.exists() or not meta_path.exists():
            continue
        
        # Load embeddings
        embeddings = np.load(emb_path)
        
        # Load metadata
        metadata_chunks = []
        for df_chunk in pd.read_csv(meta_path, chunksize=50_000):
            metadata_chunks.append(df_chunk)
        metadata = pd.concat(metadata_chunks, ignore_index=True)
        del metadata_chunks
        
        # Ensure matching lengths
        min_len = min(len(embeddings), len(metadata))
        embeddings = embeddings[:min_len]
        metadata = metadata.iloc[:min_len].reset_index(drop=True)
        
        # Sample if needed
        if sample_every > 1:
            indices = np.arange(0, min_len, sample_every)
            embeddings = embeddings[indices]
            metadata = metadata.iloc[indices].reset_index(drop=True)
        
        # Add method features if requested
        if enhance_with_methods:
            method_features = extract_methodological_features(metadata["text"].tolist())
            embeddings = np.hstack([embeddings, method_features])
        
        # Transform with UMAP
        if umap_reducer is not None:
            reduced_embeddings = umap_reducer.transform(embeddings)
        else:
            # For raw clustering, keep original embeddings
            reduced_embeddings = embeddings
        
        # Save to temporary file
        temp_file = work_dir / f"chunk_{i:04d}_reduced.npy"
        np.save(temp_file, reduced_embeddings.astype(np.float32))
        temp_files.append(temp_file)
        
        # Store metadata
        all_para_ids.extend(metadata["para_id"].tolist())
        all_paper_ids.extend(metadata["paper_id"].tolist())
        
        total_transformed += len(reduced_embeddings)
        
        # Clean up chunk data
        del embeddings, reduced_embeddings, metadata
        gc.collect()
        
        if debug_memory and i % 10 == 0:
            print_memory_info(f"After chunk {i}", debug_memory)
    
    print(f"[STREAM] Transformed {total_transformed} samples across {len(temp_files)} chunks")
    print_memory_info("After streaming transform", debug_memory)
    
    return temp_files, all_para_ids, all_paper_ids


def load_and_cluster_transformed_data(temp_files: List[Path], min_cluster_size: int,
                                     min_samples: int, metric: str = "euclidean",
                                     debug_memory: bool = False) -> np.ndarray:
    """Load transformed data and perform clustering"""
    print("[CLUSTER] Loading transformed data for clustering...")
    print_memory_info("Before loading transformed data", debug_memory)
    
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("HDBSCAN not available")
    
    # Load all transformed embeddings
    embeddings_list = []
    total_samples = 0
    
    for temp_file in tqdm(temp_files, desc="Loading transformed chunks"):
        chunk_embeddings = np.load(temp_file)
        embeddings_list.append(chunk_embeddings)
        total_samples += len(chunk_embeddings)
    
    # Combine all transformed embeddings (should be much smaller now)
    all_embeddings = np.vstack(embeddings_list)
    del embeddings_list
    gc.collect()
    
    print(f"[CLUSTER] Loaded {total_samples} transformed samples with {all_embeddings.shape[1]} dimensions")
    print_memory_info("Before clustering", debug_memory)
    
    # Adjust parameters for dataset size
    adjusted_min_cluster_size = max(min_cluster_size, total_samples // 5000)
    adjusted_min_samples = max(min_samples, adjusted_min_cluster_size // 10)
    
    print(f"[CLUSTER] Using min_cluster_size={adjusted_min_cluster_size}, min_samples={adjusted_min_samples}")
    
    # Cluster
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=adjusted_min_cluster_size,
        min_samples=adjusted_min_samples,
        metric=metric,
        cluster_selection_epsilon=0.05,
        cluster_selection_method='leaf',
        core_dist_n_jobs=-1,
        prediction_data=False
    )
    
    labels = clusterer.fit_predict(all_embeddings)
    
    print_memory_info("After clustering", debug_memory)
    
    return labels


def cleanup_temp_files(temp_files: List[Path]):
    """Clean up temporary files"""
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except:
            pass


def streaming_clustering_pipeline(chunk_dirs: List[Path], meta_name: str, emb_name: str,
                                sample_every: int, work_dir: Path, args, debug_memory: bool = False):
    """True streaming clustering pipeline"""
    print("="*60)
    print("STREAMING CLUSTERING PIPELINE")
    print("="*60)
    
    print_memory_info("Pipeline start", debug_memory)
    
    # Count total samples for information
    total_samples = count_total_samples(chunk_dirs, emb_name, sample_every)
    print(f"[INFO] Total samples to process: {total_samples:,}")
    
    temp_files = []
    try:
        if args.backend == "streaming_umap_hdbscan":
            # Phase 1: Create UMAP model from sample
            umap_reducer = create_umap_model_from_sample(
                chunk_dirs, meta_name, emb_name, sample_every, args.enhance_with_methods,
                args.umap_dim, args.n_neighbors, debug_memory=debug_memory
            )
            metric = "euclidean"
            
        else:  # streaming_hdbscan_raw
            umap_reducer = None
            metric = "cosine"  # Use cosine for raw embeddings
        
        # Phase 2: Stream transform chunks
        temp_files, para_ids, paper_ids = stream_transform_chunks(
            chunk_dirs, meta_name, emb_name, sample_every, args.enhance_with_methods,
            umap_reducer, work_dir, debug_memory
        )
        
        # Phase 3: Load transformed data and cluster
        labels = load_and_cluster_transformed_data(
            temp_files, args.min_cluster_size, args.min_samples, metric, debug_memory
        )
        
        print_memory_info("Pipeline end", debug_memory)
        
        return labels, para_ids, paper_ids
        
    finally:
        # Always clean up temporary files
        if temp_files:
            print("[CLEANUP] Removing temporary files...")
            cleanup_temp_files(temp_files)


def summarize(labels: np.ndarray) -> dict:
    """Generate clustering summary statistics"""
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


def main():
    parser = argparse.ArgumentParser(description="True streaming clustering for large-scale embeddings")
    
    # Input/output arguments
    parser.add_argument("--chunks_root", required=True, help="Root directory containing chunk folders")
    parser.add_argument("--work_dir", required=True, help="Directory for temp files and outputs")
    
    # Data loading arguments
    parser.add_argument("--chunk_pattern", default=r"chunk_\d{4}", help="Regex pattern for chunk folders")
    parser.add_argument("--meta_name", default="metadata.csv", help="Name of metadata file in each chunk")
    parser.add_argument("--emb_name", default="embeddings.npy", help="Name of embeddings file in each chunk")
    parser.add_argument("--max_chunks", type=int, default=0, help="Use first N shard folders (0=all)")
    parser.add_argument("--sample_every", type=int, default=1, help="Use every k-th paragraph row")
    
    # Enhancement options
    parser.add_argument("--enhance_with_methods", action="store_true", default=True, 
                       help="Add methodological features to embeddings")
    
    # Algorithm arguments
    parser.add_argument("--backend", choices=["streaming_umap_hdbscan", "streaming_hdbscan_raw"], 
                       default="streaming_umap_hdbscan", help="Streaming clustering backend")
    parser.add_argument("--min_cluster_size", type=int, default=500, help="HDBSCAN min cluster size")
    parser.add_argument("--min_samples", type=int, default=50, help="HDBSCAN min samples")
    
    # UMAP arguments
    parser.add_argument("--umap_dim", type=int, default=50, help="UMAP target dimensions")
    parser.add_argument("--n_neighbors", type=int, default=50, help="UMAP number of neighbors")
    
    # System arguments
    parser.add_argument("--debug_memory", action="store_true", help="Print detailed memory usage info")
    
    args = parser.parse_args()
    
    # Setup
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    
    print_memory_info("Script start", args.debug_memory)
    
    # Find chunk directories
    chunk_dirs = scan_chunk_dirs(Path(args.chunks_root), args.chunk_pattern)
    if args.max_chunks > 0:
        chunk_dirs = chunk_dirs[:args.max_chunks]
    print(f"[SCAN] Using {len(chunk_dirs)} chunk folders")
    
    # Run streaming pipeline
    labels, para_ids, paper_ids = streaming_clustering_pipeline(
        chunk_dirs, args.meta_name, args.emb_name, args.sample_every, 
        work, args, args.debug_memory
    )
    
    # Generate and save results
    summ = summarize(labels)
    print("\n" + "=" * 60)
    print("STREAMING CLUSTERING RESULTS")
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
    print(f"\n[WRITE] Labels saved to: {out_labels}")
    
    # Save summary
    summary_path = work / "cluster_summary.json"
    summary_path.write_text(json.dumps(summ, indent=2))
    print(f"[WRITE] Summary saved to: {summary_path}")
    
    # Save configuration
    config = {
        "backend": args.backend,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "sample_every": args.sample_every,
        "n_chunks": len(chunk_dirs),
        "enhance_with_methods": args.enhance_with_methods,
        "streaming_pipeline": True,
        "memory_optimized": True
    }
    
    if args.backend == "streaming_umap_hdbscan":
        config.update({"umap_dim": args.umap_dim, "n_neighbors": args.n_neighbors})
    
    config_path = work / "cluster_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[WRITE] Config saved to: {config_path}")
    
    print_memory_info("Script end", args.debug_memory)
    print("\n✓ True streaming clustering completed successfully!")


if __name__ == "__main__":
    main()