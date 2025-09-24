#!/usr/bin/env python3
"""
Cluster pre-computed embeddings using memory-efficient UMAP and HDBSCAN.

Requires:
  pip install umap-learn hdbscan scikit-learn pandas numpy tqdm

Usage:
  python cluster_embeddings.py \
    --embeddings_dir ./embeddings_out \
    --out_dir ./cluster_out \
    --umap_dim 30 --min_cluster_size 10 --min_samples 2 \
    --subsample_for_umap 100000 --memory_map
"""
import os, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
import tempfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils import resample
import umap
import hdbscan

def load_embeddings_and_metadata(embeddings_dir: Path, memory_map=False):
    """Load all embeddings and metadata efficiently"""
    all_embeddings = []
    all_metadata = []
    
    # Get all chunk directories
    chunk_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk directories found in {embeddings_dir}")
    
    total_size = 0
    print(f"Found {len(chunk_dirs)} embedding chunks")
    
    # First pass: determine total size and embedding dimension
    embedding_dim = None
    for chunk_dir in chunk_dirs:
        emb_file = chunk_dir / "embeddings.npy"
        if emb_file.exists():
            if memory_map:
                emb = np.load(emb_file, mmap_mode='r')
            else:
                emb = np.load(emb_file)
            total_size += emb.shape[0]
            if embedding_dim is None:
                embedding_dim = emb.shape[1]
            if not memory_map:
                del emb
    
    print(f"Total embeddings: {total_size}, dimension: {embedding_dim}")
    
    # Pre-allocate arrays
    if not memory_map:
        all_embeddings_array = np.empty((total_size, embedding_dim), dtype=np.float32)
        current_idx = 0
    
    # Second pass: load data
    for chunk_dir in tqdm(chunk_dirs, desc="Loading chunks"):
        emb_file = chunk_dir / "embeddings.npy"
        meta_file = chunk_dir / "metadata.csv"
        
        if not (emb_file.exists() and meta_file.exists()):
            print(f"Warning: Missing files in {chunk_dir}")
            continue
        
        # Load embeddings
        if memory_map:
            emb = np.load(emb_file, mmap_mode='r')
            all_embeddings.append(emb)
        else:
            emb = np.load(emb_file).astype(np.float32)  # Convert to float32 to save memory
            all_embeddings_array[current_idx:current_idx + emb.shape[0]] = emb
            current_idx += emb.shape[0]
            del emb
        
        # Load metadata
        meta_df = pd.read_csv(meta_file)
        all_metadata.append(meta_df)
    
    # Combine metadata
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    del all_metadata
    gc.collect()
    
    if memory_map:
        return all_embeddings, combined_metadata
    else:
        return all_embeddings_array, combined_metadata

def create_umap_model(embeddings_list, n_components, subsample_size=None, random_state=42):
    """Create UMAP model, optionally using subsampling for large datasets"""
    
    if isinstance(embeddings_list, list):  # memory mapped
        # Count total embeddings
        total_size = sum(emb.shape[0] for emb in embeddings_list)
        embedding_dim = embeddings_list[0].shape[1]
        
        if subsample_size and subsample_size < total_size:
            print(f"Subsampling {subsample_size} from {total_size} embeddings for UMAP training")
            
            # Randomly sample indices
            indices = np.random.choice(total_size, subsample_size, replace=False)
            indices.sort()
            
            # Collect subsampled embeddings
            sample_embeddings = np.empty((subsample_size, embedding_dim), dtype=np.float32)
            current_global_idx = 0
            sample_idx = 0
            indices_set = set(indices)
            
            for emb in embeddings_list:
                chunk_size = emb.shape[0]
                chunk_indices = []
                for i in range(chunk_size):
                    if current_global_idx + i in indices_set:
                        chunk_indices.append(i)
                
                if chunk_indices:
                    chunk_sample = emb[chunk_indices]
                    end_idx = sample_idx + len(chunk_indices)
                    sample_embeddings[sample_idx:end_idx] = chunk_sample
                    sample_idx = end_idx
                
                current_global_idx += chunk_size
            
            train_embeddings = sample_embeddings
        else:
            # Load all embeddings
            print("Loading all embeddings for UMAP training")
            train_embeddings = np.vstack(embeddings_list)
    else:  # regular array
        if subsample_size and subsample_size < len(embeddings_list):
            print(f"Subsampling {subsample_size} from {len(embeddings_list)} embeddings for UMAP training")
            indices = np.random.choice(len(embeddings_list), subsample_size, replace=False)
            train_embeddings = embeddings_list[indices]
        else:
            train_embeddings = embeddings_list
    
    print(f"Training UMAP with {train_embeddings.shape[0]} embeddings")
    reducer = umap.UMAP(
        n_components=n_components, 
        metric="cosine", 
        random_state=random_state,
        low_memory=True,
        verbose=True
    )
    reducer.fit(train_embeddings)
    
    del train_embeddings
    gc.collect()
    
    return reducer

def transform_in_batches(reducer, embeddings, batch_size=10000):
    """Transform embeddings in batches to save memory"""
    if isinstance(embeddings, list):  # memory mapped
        all_transformed = []
        for emb_chunk in tqdm(embeddings, desc="Transforming chunks"):
            if emb_chunk.shape[0] <= batch_size:
                transformed = reducer.transform(emb_chunk)
                all_transformed.append(transformed)
            else:
                # Process in sub-batches
                chunk_transformed = []
                for i in range(0, emb_chunk.shape[0], batch_size):
                    batch = emb_chunk[i:i+batch_size]
                    batch_transformed = reducer.transform(batch)
                    chunk_transformed.append(batch_transformed)
                all_transformed.append(np.vstack(chunk_transformed))
        
        return np.vstack(all_transformed)
    
    else:  # regular array
        if embeddings.shape[0] <= batch_size:
            return reducer.transform(embeddings)
        
        results = []
        for i in tqdm(range(0, embeddings.shape[0], batch_size), desc="Transforming batches"):
            batch = embeddings[i:i+batch_size]
            batch_result = reducer.transform(batch)
            results.append(batch_result)
        
        return np.vstack(results)

def tfidf_top_terms(texts, labels, n_top=10, max_texts_per_cluster=1000):
    """Extract top terms for each cluster with memory efficiency"""
    mask = labels >= 0
    if not mask.any():
        return {}
    
    X_texts = [t for t, keep in zip(texts, mask) if keep]
    y = labels[mask]
    
    # Subsample if too many texts
    if len(X_texts) > max_texts_per_cluster * len(np.unique(y)):
        print(f"Subsampling texts for TF-IDF (from {len(X_texts)} texts)")
        indices = resample(range(len(X_texts)), n_samples=min(50000, len(X_texts)), 
                          random_state=42, replace=False)
        X_texts = [X_texts[i] for i in indices]
        y = y[indices]
    
    print("Computing TF-IDF...")
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=3, max_df=0.95)
    X = vec.fit_transform(X_texts)
    X = normalize(X, norm="l2")
    vocab = np.array(vec.get_feature_names_out())
    
    out = {}
    for c in tqdm(np.unique(y), desc="Extracting top terms"):
        idx = np.where(y == c)[0]
        if not len(idx): 
            continue
        scores = np.asarray(X[idx].mean(axis=0)).ravel()
        top = vocab[scores.argsort()[-n_top:][::-1]]
        out[int(c)] = top.tolist()
    
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True, help="Directory with embedding chunks")
    ap.add_argument("--out_dir", required=True, help="Output directory for clustering results")
    
    # Clustering parameters
    ap.add_argument("--umap_dim", type=int, default=30)
    ap.add_argument("--min_cluster_size", type=int, default=10)
    ap.add_argument("--min_samples", type=int, default=2)
    
    # Memory optimization
    ap.add_argument("--subsample_for_umap", type=int, default=100000, 
                    help="Subsample size for UMAP training (0 = use all)")
    ap.add_argument("--memory_map", action="store_true",
                    help="Use memory mapping for embeddings")
    ap.add_argument("--batch_size", type=int, default=10000,
                    help="Batch size for UMAP transform")
    
    args = ap.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embedding info
    info_file = embeddings_dir / "embedding_info.json"
    if info_file.exists():
        with info_file.open() as f:
            embedding_info = json.load(f)
        print(f"Embedding info: {embedding_info['total_paragraphs']} paragraphs in {embedding_info['total_chunks']} chunks")
    
    # Load embeddings and metadata
    print("Loading embeddings and metadata...")
    embeddings, metadata_df = load_embeddings_and_metadata(embeddings_dir, args.memory_map)
    
    print(f"Loaded {len(metadata_df)} paragraphs")
    
    # Create UMAP model
    print("Training UMAP...")
    subsample_size = args.subsample_for_umap if args.subsample_for_umap > 0 else None
    reducer = create_umap_model(embeddings, args.umap_dim, subsample_size)
    
    # Transform all embeddings
    print("Transforming embeddings with UMAP...")
    Z = transform_in_batches(reducer, embeddings, args.batch_size)
    
    # Clear embeddings from memory if possible
    if not args.memory_map:
        del embeddings
        gc.collect()
    
    print(f"UMAP output shape: {Z.shape}")
    
    # Cluster with HDBSCAN
    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size, 
        min_samples=args.min_samples, 
        metric="euclidean",
        core_dist_n_jobs=1  # Use single thread to save memory
    )
    labels = clusterer.fit_predict(Z)
    
    # Clear UMAP results
    del Z
    gc.collect()
    
    # Stats
    n = len(labels)
    n_noise = int((labels < 0).sum())
    n_clusters = int((labels >= 0).max() + 1 if (labels >= 0).any() else 0)
    noise_pct = 100.0 * n_noise / n
    print(f"Paragraphs: {n} | clusters: {n_clusters} | noise: {n_noise} ({noise_pct:.1f}%)")
    
    # Extract top terms
    print("Extracting top terms per cluster...")
    tops = tfidf_top_terms(metadata_df["text"].tolist(), labels, n_top=12)
    
    # Save results
    print("Saving results...")
    
    # Save assignments
    output_df = metadata_df.copy()
    output_df["cluster"] = labels
    
    # Save in chunks to avoid memory issues
    chunk_size = 100000
    for i in range(0, len(output_df), chunk_size):
        chunk_df = output_df.iloc[i:i+chunk_size]
        chunk_file = out_dir / f"paragraph_clusters_part_{i//chunk_size:04d}.csv"
        chunk_df.to_csv(chunk_file, index=False)
    
    # Save summary
    summary = {
        "paragraphs": n, 
        "clusters": n_clusters, 
        "noise": n_noise, 
        "noise_pct": noise_pct,
        "top_terms": tops,
        "clustering_params": {
            "umap_dim": args.umap_dim,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "subsample_for_umap": args.subsample_for_umap
        }
    }
    
    with (out_dir / "cluster_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Paper-level bag-of-clusters (process in chunks)
    print("Computing paper-level bag-of-clusters...")
    bag_parts = []
    for i in range(0, len(output_df), chunk_size):
        chunk_df = output_df.iloc[i:i+chunk_size]
        chunk_bag = (chunk_df[chunk_df["cluster"] >= 0]
                    .groupby(["paper_id", "cluster"]).size()
                    .unstack(fill_value=0))
        if not chunk_bag.empty:
            bag_parts.append(chunk_bag)
    
    if bag_parts:
        # Combine bag-of-clusters parts
        full_bag = pd.concat(bag_parts).groupby(level=0).sum()
        full_bag = full_bag.sort_index(axis=1)
        full_bag.to_csv(out_dir / "paper_bag_of_methods.csv")
    
    print(f"\nClustering complete!")
    print(f"Results saved to: {out_dir}")
    print(f"Total clusters found: {n_clusters}")

if __name__ == "__main__":
    main()