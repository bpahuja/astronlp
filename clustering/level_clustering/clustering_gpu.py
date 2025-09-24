import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated imports
try:
    import cuml
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.decomposition import PCA as cuPCA
    from cuml.decomposition import IncrementalPCA as cuIncrementalPCA
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    import cupy as cp
    import rmm
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with cuML/RAPIDS")
except ImportError as e:
    print(f"GPU libraries not available: {e}")
    print("Falling back to CPU versions...")
    import hdbscan
    from sklearn.decomposition import IncrementalPCA
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    GPU_AVAILABLE = False

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from collections import Counter
import pickle
import gc
import psutil
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='clustering.log')
logger = logging.getLogger(__name__)

def check_memory_usage():
    """Monitor memory usage (CPU and GPU)"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024**3)
    logger.info(f"Current CPU memory usage: {memory_gb:.2f} GB")
    
    if GPU_AVAILABLE:
        try:
            gpu_memory = cp.cuda.runtime.memGetInfo()
            gpu_free_gb = gpu_memory[0] / (1024**3)
            gpu_total_gb = gpu_memory[1] / (1024**3)
            gpu_used_gb = gpu_total_gb - gpu_free_gb
            logger.info(f"GPU memory usage: {gpu_used_gb:.2f}/{gpu_total_gb:.2f} GB")
            return memory_gb, gpu_used_gb
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
            return memory_gb, 0
    return memory_gb, 0

def setup_gpu_memory_pool():
    """Setup GPU memory pool for efficient memory management"""
    if GPU_AVAILABLE:
        try:
            # Initialize RMM memory pool
            rmm.reinitialize(
                pool_allocator=True,
                managed_memory=False,
                initial_pool_size=None  # Use all available GPU memory
            )
            logger.info("GPU memory pool initialized")
        except Exception as e:
            logger.warning(f"Could not initialize GPU memory pool: {e}")

class GPUAcceleratedParagraphClusterer:
    """
    GPU-accelerated paragraph clustering system for large datasets (4M+ points).
    Falls back to CPU if GPU is not available.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.embeddings = None
        self.df = None
        self.cluster_labels = None
        self.clusterer = None
        self.umap_model = None
        self.reduced_embeddings = None
        self.chunk_size = config.get('chunk_size', 50000)
        self.gpu_available = GPU_AVAILABLE and config.get('use_gpu', True)
        
        if self.gpu_available:
            setup_gpu_memory_pool()
            logger.info("GPU acceleration enabled")
        else:
            logger.info("Using CPU-only processing")
        
    def load_data_chunked(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load data in chunks to manage memory usage.
        """
        logger.info("Loading data in chunks...")
        check_memory_usage()
        
        try:
            # First, get the shape without loading full array
            embeddings_file = self.config['embeddings_file']
            temp_embeddings = np.load(embeddings_file, mmap_mode='r')  # Memory-mapped
            total_samples, embedding_dim = temp_embeddings.shape
            logger.info(f"Dataset: {total_samples:,} embeddings with {embedding_dim} dimensions")
            
            # Estimate memory requirements
            estimated_memory_gb = (total_samples * embedding_dim * 4) / (1024**3)  # 4 bytes per float32
            logger.info(f"Estimated memory for embeddings: {estimated_memory_gb:.2f} GB")
            
            # Load ID to index mapping
            with open(self.config['id_to_index_file'], 'r') as f:
                id_to_index = json.load(f)
            
            # Create reverse mapping
            index_to_id = {v: k for k, v in id_to_index.items()}
            
            # Load paragraphs metadata only (not full text initially)
            para_metadata = self._load_paragraphs_metadata()
            
            # Create DataFrame efficiently
            text_df = pd.DataFrame(para_metadata)
            map_df = pd.DataFrame(list(index_to_id.items()), columns=['embedding_index', 'para_id'])
            
            merged_df = pd.merge(map_df, text_df, on='para_id', how='inner')
            merged_df = merged_df.sort_values('embedding_index').reset_index(drop=True)
            
            logger.info(f"Metadata loaded for {len(merged_df):,} paragraphs.")
            
            # Load embeddings as float32 to save memory
            embeddings = np.load(embeddings_file).astype(np.float32)
            
            check_memory_usage()
            return merged_df, embeddings
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_paragraphs_metadata(self) -> List[Dict]:
        """
        Load only paragraph metadata initially to save memory.
        """
        para_metadata = []
        papers_dir = Path(self.config['papers_directory'])
        
        if not papers_dir.exists():
            raise FileNotFoundError(f"Papers directory not found: {papers_dir}")
        
        for txt_file in papers_dir.glob("*.txt"):
            paper_id = txt_file.stem
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                for i, para_text in enumerate(paragraphs):
                    para_id = f"{paper_id}_para_{i:04d}"
                    # Store minimal metadata, load full text later if needed
                    para_metadata.append({
                        'para_id': para_id,
                        'paper_id': paper_id,
                        'para_index': i,
                        'para_length': len(para_text),
                        'text_preview': para_text[:200] + "..." if len(para_text) > 200 else para_text
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {txt_file}: {e}")
                continue
        
        logger.info(f"Loaded metadata from {len(set(p['paper_id'] for p in para_metadata))} papers")
        return para_metadata
    
    def optimize_clustering_parameters_smart(self, embeddings: np.ndarray) -> Dict:
        """
        Smart parameter optimization using stratified sampling for large datasets with GPU acceleration.
        """
        logger.info("Optimizing clustering parameters with GPU-accelerated smart sampling...")
        
        # Use a smaller, stratified sample for parameter optimization
        sample_size = min(15000, embeddings.shape[0] // 8)  # Slightly larger for GPU
        logger.info(f"Using stratified sample of {sample_size:,} points for parameter optimization")
        
        # Stratified sampling
        total_samples = embeddings.shape[0]
        chunk_size = total_samples // sample_size
        indices = []
        
        for i in range(0, total_samples, max(1, chunk_size)):
            if len(indices) < sample_size:
                indices.append(i)
        
        # Add some random samples
        remaining = sample_size - len(indices)
        if remaining > 0:
            random_indices = np.random.choice(total_samples, remaining, replace=False)
            indices.extend(random_indices)
        
        sample_indices = np.array(indices[:sample_size])
        sample_embeddings = embeddings[sample_indices].copy()
        
        # Quick dimensionality reduction for parameter optimization
        if sample_embeddings.shape[1] > 100:
            logger.info("Applying PCA for parameter optimization")
            if self.gpu_available:
                # Use GPU PCA
                sample_gpu = cp.asarray(sample_embeddings)
                pca = cuPCA(n_components=50, random_state=42)
                sample_embeddings = cp.asnumpy(pca.fit_transform(sample_gpu))
                del sample_gpu
                cp.cuda.runtime.deviceSynchronize()
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=50, random_state=42)
                sample_embeddings = pca.fit_transform(sample_embeddings)
        
        # Test parameter combinations
        min_sizes = [20, 30, 50, 80]  # More options for GPU
        best_score = -1
        best_params = {}
        
        for min_size in min_sizes:
            try:
                if self.gpu_available:
                    # GPU HDBSCAN
                    sample_gpu = cp.asarray(sample_embeddings, dtype=cp.float32)
                    clusterer = cuHDBSCAN(
                        min_cluster_size=min_size,
                        min_samples=3,
                        cluster_selection_epsilon=0.0,
                        metric='euclidean'
                    )
                    labels = cp.asnumpy(clusterer.fit_predict(sample_gpu))
                    del sample_gpu
                    cp.cuda.runtime.deviceSynchronize()
                else:
                    # CPU HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_size,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        core_dist_n_jobs=1
                    )
                    labels = clusterer.fit_predict(sample_embeddings)
                
                if len(set(labels)) > 1 and -1 in labels:
                    mask = labels != -1
                    if mask.sum() > min_size * 2:
                        score = silhouette_score(sample_embeddings[mask], labels[mask])
                        
                        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        noise_ratio = (labels == -1).sum() / len(labels)
                        
                        logger.info(f"Min size {min_size}: {num_clusters} clusters, {noise_ratio:.2f} noise, silhouette: {score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'min_cluster_size': min_size,
                                'silhouette_score': score,
                                'num_clusters': num_clusters,
                                'noise_ratio': noise_ratio
                            }
                
                # Clean up
                del clusterer, labels
                gc.collect()
                if self.gpu_available:
                    cp.cuda.runtime.deviceSynchronize()
                            
            except Exception as e:
                logger.warning(f"Error with min_size {min_size}: {e}")
                continue
        
        # Clean up
        del sample_embeddings
        gc.collect()
        
        if best_params:
            logger.info(f"Best parameters: {best_params}")
            return best_params
        else:
            logger.warning("Could not find optimal parameters, using default")
            return {'min_cluster_size': max(50, self.config['min_cluster_size'])}
    
    def perform_gpu_dimensionality_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform GPU-accelerated dimensionality reduction.
        """
        target_dims = self.config.get('target_dimensions', 100)
        logger.info(f"Performing GPU dimensionality reduction: {embeddings.shape[1]} -> {target_dims}")
        
        if self.gpu_available:
            try:
                # Check if we can fit the data in GPU memory
                embedding_size_gb = embeddings.nbytes / (1024**3)
                gpu_memory = cp.cuda.runtime.memGetInfo()
                gpu_free_gb = gpu_memory[0] / (1024**3)
                
                if embedding_size_gb * 2 < gpu_free_gb:  # Need ~2x for processing
                    # Use GPU PCA
                    logger.info("Using GPU PCA for dimensionality reduction")
                    embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)
                    
                    pca = cuPCA(n_components=target_dims, random_state=42)
                    reduced_embeddings_gpu = pca.fit_transform(embeddings_gpu)
                    
                    reduced_embeddings = cp.asnumpy(reduced_embeddings_gpu).astype(np.float32)
                    
                    # Clean up GPU memory
                    del embeddings_gpu, reduced_embeddings_gpu
                    cp.cuda.runtime.deviceSynchronize()
                    
                    logger.info(f"GPU PCA complete. Explained variance ratio: {cp.asnumpy(pca.explained_variance_ratio_).sum():.3f}")
                    return reduced_embeddings
                else:
                    logger.warning("Not enough GPU memory for full PCA, using incremental GPU PCA")
                    return self._perform_incremental_gpu_pca(embeddings, target_dims)
                    
            except Exception as e:
                logger.warning(f"GPU PCA failed: {e}, falling back to CPU")
                return self.perform_incremental_dimensionality_reduction(embeddings)
        else:
            return self.perform_incremental_dimensionality_reduction(embeddings)
    
    def _perform_incremental_gpu_pca(self, embeddings: np.ndarray, target_dims: int) -> np.ndarray:
        """
        Perform incremental PCA using GPU in batches.
        """
        logger.info("Performing incremental GPU PCA")
        batch_size = self.config.get('pca_batch_size', 20000)  # Larger batches for GPU
        
        try:
            # Use cuML's IncrementalPCA if available
            ipca = cuIncrementalPCA(n_components=target_dims, batch_size=batch_size)
            
            n_samples = embeddings.shape[0]
            # Fit incrementally
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = embeddings[i:end_idx]
                batch_gpu = cp.asarray(batch, dtype=cp.float32)
                ipca.partial_fit(batch_gpu)
                del batch_gpu
                cp.cuda.runtime.deviceSynchronize()
                
                if i % (batch_size * 5) == 0:
                    logger.info(f"GPU PCA fitting progress: {i:,}/{n_samples:,}")
            
            # Transform incrementally
            logger.info("Transforming embeddings with GPU...")
            reduced_embeddings = np.zeros((n_samples, target_dims), dtype=np.float32)
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = embeddings[i:end_idx]
                batch_gpu = cp.asarray(batch, dtype=cp.float32)
                reduced_batch_gpu = ipca.transform(batch_gpu)
                reduced_embeddings[i:end_idx] = cp.asnumpy(reduced_batch_gpu)
                del batch_gpu, reduced_batch_gpu
                cp.cuda.runtime.deviceSynchronize()
                
                if i % (batch_size * 5) == 0:
                    logger.info(f"GPU PCA transform progress: {i:,}/{n_samples:,}")
            
            logger.info("Incremental GPU PCA complete")
            return reduced_embeddings
            
        except Exception as e:
            logger.warning(f"Incremental GPU PCA failed: {e}, falling back to CPU")
            return self.perform_incremental_dimensionality_reduction(embeddings)
    
    def perform_incremental_dimensionality_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fallback CPU incremental PCA for dimensionality reduction.
        """
        logger.info(f"Performing CPU incremental dimensionality reduction: {embeddings.shape[1]} -> {self.config['target_dimensions']}")
        
        target_dims = self.config.get('target_dimensions', 100)
        batch_size = self.config.get('pca_batch_size', 10000)
        
        ipca = IncrementalPCA(n_components=target_dims, batch_size=batch_size)
        
        # Fit incrementally
        n_samples = embeddings.shape[0]
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = embeddings[i:end_idx]
            ipca.partial_fit(batch)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"CPU PCA fitting progress: {i:,}/{n_samples:,}")
                check_memory_usage()
        
        # Transform incrementally
        logger.info("Transforming embeddings...")
        reduced_embeddings = np.zeros((n_samples, target_dims), dtype=np.float32)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = embeddings[i:end_idx]
            reduced_embeddings[i:end_idx] = ipca.transform(batch).astype(np.float32)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"CPU PCA transform progress: {i:,}/{n_samples:,}")
        
        logger.info(f"CPU dimensionality reduction complete. Explained variance ratio: {ipca.explained_variance_ratio_.sum():.3f}")
        return reduced_embeddings
    
    def perform_gpu_clustering(self, embeddings: np.ndarray, optimize_params: bool = True) -> np.ndarray:
        """
        GPU-accelerated clustering with fallback strategies.
        """
        logger.info("Starting GPU-accelerated clustering...")
        check_memory_usage()
        
        # Check if we need dimensionality reduction
        if embeddings.shape[1] > self.config.get('max_dimensions_direct', 200):
            logger.info("High dimensionality detected, applying dimensionality reduction")
            embeddings = self.perform_gpu_dimensionality_reduction(embeddings)
            gc.collect()
            if self.gpu_available:
                cp.cuda.runtime.deviceSynchronize()
            check_memory_usage()
        
        # Optimize parameters with smaller sample
        if optimize_params and not self.config.get('skip_optimization', False):
            optimal_params = self.optimize_clustering_parameters_smart(embeddings)
            min_cluster_size = optimal_params['min_cluster_size']
        else:
            min_cluster_size = max(50, self.config['min_cluster_size'])
        
        # Try GPU HDBSCAN first, with fallbacks
        try:
            if self.gpu_available:
                logger.info(f"Attempting GPU HDBSCAN clustering on {embeddings.shape[0]:,} samples...")
                
                # Check GPU memory
                embedding_size_gb = embeddings.nbytes / (1024**3)
                gpu_memory = cp.cuda.runtime.memGetInfo()
                gpu_free_gb = gpu_memory[0] / (1024**3)
                
                if embedding_size_gb * 3 < gpu_free_gb:  # Conservative estimate
                    # Use GPU HDBSCAN
                    embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)
                    
                    self.clusterer = cuHDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=self.config.get('min_samples', 3),
                        cluster_selection_epsilon=0.0,
                        metric='euclidean'
                    )
                    
                    cluster_labels_gpu = self.clusterer.fit_predict(embeddings_gpu)
                    cluster_labels = cp.asnumpy(cluster_labels_gpu)
                    
                    # Clean up GPU memory
                    del embeddings_gpu, cluster_labels_gpu
                    cp.cuda.runtime.deviceSynchronize()
                    
                    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    noise_points = (cluster_labels == -1).sum()
                    logger.info(f"GPU HDBSCAN complete: {num_clusters} clusters, {noise_points:,} noise points")
                    
                else:
                    logger.warning("Not enough GPU memory for HDBSCAN, falling back to GPU K-Means")
                    raise MemoryError("Insufficient GPU memory")
            else:
                raise ImportError("GPU not available")
                
        except (MemoryError, ImportError, Exception) as e:
            logger.warning(f"GPU HDBSCAN failed: {e}")
            
            if self.gpu_available:
                logger.info("Trying GPU K-Means as fallback...")
                try:
                    # Use GPU K-Means
                    n_clusters = min(1000, embeddings.shape[0] // (min_cluster_size * 2))
                    
                    embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)
                    kmeans = cuKMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=3,
                        max_iter=100
                    )
                    
                    cluster_labels_gpu = kmeans.fit_predict(embeddings_gpu)
                    cluster_labels = cp.asnumpy(cluster_labels_gpu)
                    
                    # Clean up
                    del embeddings_gpu, cluster_labels_gpu
                    cp.cuda.runtime.deviceSynchronize()
                    
                    self.clusterer = kmeans
                    logger.info(f"GPU K-Means complete: {n_clusters} clusters")
                    
                except Exception as e2:
                    logger.warning(f"GPU K-Means also failed: {e2}, falling back to CPU")
                    cluster_labels = self._fallback_cpu_clustering(embeddings, min_cluster_size)
            else:
                cluster_labels = self._fallback_cpu_clustering(embeddings, min_cluster_size)
        
        self.reduced_embeddings = embeddings
        return cluster_labels
    
    def _fallback_cpu_clustering(self, embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
        """
        Fallback to CPU clustering when GPU fails.
        """
        logger.info("Using CPU fallback clustering...")
        
        try:
            # Try CPU HDBSCAN
            clustering_params = {
                'min_cluster_size': min_cluster_size,
                'min_samples': self.config.get('min_samples', 3),
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'algorithm': 'boruvka_kdtree',
                'leaf_size': 50,
                'core_dist_n_jobs': 1,
                'cluster_selection_epsilon': 0.0
            }
            
            self.clusterer = hdbscan.HDBSCAN(**clustering_params)
            cluster_labels = self.clusterer.fit_predict(embeddings)
            
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            noise_points = (cluster_labels == -1).sum()
            logger.info(f"CPU HDBSCAN complete: {num_clusters} clusters, {noise_points:,} noise points")
            
        except MemoryError:
            logger.info("CPU HDBSCAN failed due to memory, using MiniBatch K-Means...")
            
            # Final fallback: MiniBatch K-Means
            n_clusters = min(1000, embeddings.shape[0] // (min_cluster_size * 2))
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=self.config.get('kmeans_batch_size', 10000),
                random_state=42,
                n_init=3,
                max_iter=100
            )
            
            cluster_labels = kmeans.fit_predict(embeddings)
            self.clusterer = kmeans
            
            logger.info(f"CPU MiniBatch K-Means complete: {n_clusters} clusters")
        
        return cluster_labels
    
    def generate_labels_efficiently(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, str]:
        """
        Generate cluster labels efficiently for large datasets.
        """
        logger.info("Generating cluster labels efficiently...")
        
        df_copy = df.copy()
        df_copy['cluster_id'] = cluster_labels
        
        labels = {}
        cluster_counts = pd.Series(cluster_labels).value_counts()
        
        # Process only top clusters to save time
        top_clusters = cluster_counts.head(self.config.get('max_clusters_to_label', 200))
        
        for cluster_id, count in top_clusters.items():
            if cluster_id == -1:
                labels[cluster_id] = "-1_Noise"
                continue
            
            # Simple label generation for large datasets
            labels[cluster_id] = f"cluster_{cluster_id}_n{count}"
        
        # Add generic labels for remaining clusters
        for cluster_id in cluster_counts.index:
            if cluster_id not in labels:
                count = cluster_counts[cluster_id]
                labels[cluster_id] = f"cluster_{cluster_id}_n{count}"
        
        return labels
    
    def save_results_chunked(self, df: pd.DataFrame, cluster_labels: np.ndarray, cluster_label_map: Dict):
        """
        Save results in chunks to manage memory.
        """
        logger.info("Saving results in chunks...")
        
        # Prepare final DataFrame
        result_df = df.copy()
        result_df['cluster_id'] = cluster_labels
        result_df['cluster_label'] = result_df['cluster_id'].map(cluster_label_map)
        
        # Add clustering metadata if available
        if hasattr(self.clusterer, 'probabilities_') and self.clusterer.probabilities_ is not None:
            if self.gpu_available and hasattr(self.clusterer.probabilities_, 'get'):
                result_df['cluster_probability'] = cp.asnumpy(self.clusterer.probabilities_)
            else:
                result_df['cluster_probability'] = self.clusterer.probabilities_
                
        if hasattr(self.clusterer, 'outlier_scores_') and self.clusterer.outlier_scores_ is not None:
            if self.gpu_available and hasattr(self.clusterer.outlier_scores_, 'get'):
                result_df['outlier_score'] = cp.asnumpy(self.clusterer.outlier_scores_)
            else:
                result_df['outlier_score'] = self.clusterer.outlier_scores_
        
        # Save in chunks
        output_file = self.config['output_file']
        chunk_size = self.config.get('save_chunk_size', 100000)
        
        for i, chunk_start in enumerate(range(0, len(result_df), chunk_size)):
            chunk_end = min(chunk_start + chunk_size, len(result_df))
            chunk = result_df.iloc[chunk_start:chunk_end]
            
            if i == 0:
                chunk.to_csv(output_file, index=False, mode='w')
            else:
                chunk.to_csv(output_file, index=False, mode='a', header=False)
            
            logger.info(f"Saved chunk {i+1}: rows {chunk_start:,} to {chunk_end:,}")
        
        logger.info(f"Results saved to {output_file}")
        
        # Save cluster summary
        self._save_cluster_summary_efficient(result_df)
    
    def _save_cluster_summary_efficient(self, df: pd.DataFrame):
        """
        Save cluster summary efficiently for large datasets.
        """
        summary_file = self.config['output_file'].replace('.csv', '_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("PARAGRAPH CLUSTERING SUMMARY (GPU-Accelerated)\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            num_clusters = df['cluster_id'].nunique()
            if -1 in df['cluster_id'].values:
                num_clusters -= 1
                noise_points = (df['cluster_id'] == -1).sum()
                noise_percentage = noise_points / len(df) * 100
            else:
                noise_points = 0
                noise_percentage = 0
            
            f.write(f"Total paragraphs: {len(df):,}\n")
            f.write(f"Number of clusters: {num_clusters:,}\n")
            f.write(f"Noise points: {noise_points:,} ({noise_percentage:.2f}%)\n")
            f.write(f"GPU acceleration: {'Enabled' if self.gpu_available else 'Disabled'}\n\n")
            
            # Top cluster sizes
            f.write("TOP 50 CLUSTER SIZES\n")
            f.write("-" * 30 + "\n")
            cluster_sizes = df[df['cluster_id'] != -1]['cluster_id'].value_counts().head(50)
            for cluster_id, size in cluster_sizes.items():
                f.write(f"Cluster {cluster_id}: {size:,} paragraphs\n")
        
        logger.info(f"Cluster summary saved to {summary_file}")
    
    def run_gpu_pipeline(self):
        """
        Run the complete GPU-accelerated clustering pipeline.
        """
        logger.info("Starting GPU-accelerated clustering pipeline...")
        check_memory_usage()
        
        # Load data
        self.df, self.embeddings = self.load_data_chunked()
        check_memory_usage()
        
        # Perform clustering
        self.cluster_labels = self.perform_gpu_clustering(self.embeddings)
        check_memory_usage()
        
        # Generate labels
        cluster_label_map = self.generate_labels_efficiently(self.df, self.cluster_labels)
        
        # Save results
        self.save_results_chunked(self.df, self.cluster_labels, cluster_label_map)
        
        # Clean up
        del self.embeddings, self.reduced_embeddings
        gc.collect()
        if self.gpu_available:
            cp.cuda.runtime.deviceSynchronize()
        
        logger.info("GPU-accelerated clustering pipeline completed successfully!")
        final_memory = check_memory_usage()
        
        return self.df, self.cluster_labels, cluster_label_map


def create_gpu_optimized_config():
    """
    Create GPU-optimized configuration for large datasets.
    """
    return {
        # Input files
        'embeddings_file': 'embeddings.npy',
        'id_to_index_file': 'id_to_index.json',
        'papers_directory': 'papers/',
        
        # Output files
        'output_file': 'clustered_paragraphs.csv',
        
        # GPU optimization parameters
        'use_gpu': True,  # Enable/disable GPU acceleration
        'chunk_size': 75000,  # Larger chunks for GPU processing
        'save_chunk_size': 100000,  # Save results in chunks
        'pca_batch_size': 20000,  # Larger batch size for GPU PCA
        'kmeans_batch_size': 20000,  # Larger batch size for GPU K-Means
        
        # Clustering parameters (optimized for GPU + large datasets)
        'min_cluster_size': 50,  # Larger clusters for big datasets
        'min_samples': 3,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        
        # Dimensionality reduction
        'target_dimensions': 100,  # Reduce to this many dimensions
        'max_dimensions_direct': 200,  # Apply reduction if dims > this
        
        # Pipeline optimizations
        'skip_optimization': False,
        'max_clusters_to_label': 300,  # More clusters to label with GPU speed
        'save_models': False,  # Disable to save memory
        'generate_visualizations': False,  # Disable for large datasets
        
        # GPU-specific settings
        'gpu_memory_fraction': 0.8,  # Use 80% of GPU memory max
        'prefer_gpu_pca': True,  # Prefer GPU PCA when possible
        'fallback_to_cpu': True,  # Fallback to CPU if GPU fails
    }


def check_gpu_requirements():
    """
    Check GPU requirements and provide installation instructions if needed.
    """
    if not GPU_AVAILABLE:
        print("\n" + "="*60)
        print("GPU ACCELERATION NOT AVAILABLE")
        print("="*60)
        print("To enable GPU acceleration, install RAPIDS cuML:")
        print("\n# For CUDA 11.x:")
        print("conda install -c rapidsai -c nvidia -c conda-forge cuml cudatoolkit=11.x")
        print("\n# For CUDA 12.x:")
        print("conda install -c rapidsai -c nvidia -c conda-forge cuml cudatoolkit=12.x")
        print("\n# Or with pip:")
        print("pip install cuml-cu11  # for CUDA 11.x")
        print("pip install cuml-cu12  # for CUDA 12.x")
        print("\nThe script will run on CPU only for now.")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("GPU ACCELERATION ENABLED")
        print("="*60)
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            for i in range(gpu_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode('utf-8')
                memory_gb = props['totalGlobalMem'] / (1024**3)
                print(f"GPU {i}: {name} ({memory_gb:.1f} GB)")
        except Exception as e:
            print(f"GPU info unavailable: {e}")
        print("="*60 + "\n")


def main():
    """
    Main function with GPU acceleration support.
    """
    parser = argparse.ArgumentParser(description='GPU-Accelerated Memory-Optimized Paragraph Clustering')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--embeddings', type=str, default='embeddings.npy', help='Path to embeddings file')
    parser.add_argument('--id-map', type=str, default='id_to_index.json', help='Path to ID mapping file')
    parser.add_argument('--papers-dir', type=str, default='papers/', help='Directory containing paper text files')
    parser.add_argument('--output', type=str, default='clustered_paragraphs.csv', help='Output CSV file')
    parser.add_argument('--min-cluster-size', type=int, default=50, help='Minimum cluster size')
    parser.add_argument('--target-dims', type=int, default=100, help='Target dimensions after reduction')
    parser.add_argument('--chunk-size', type=int, default=75000, help='Chunk size for processing')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip parameter optimization')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.8, help='Fraction of GPU memory to use')
    parser.add_argument('--pca-batch-size', type=int, default=20000, help='Batch size for PCA operations')
    
    args = parser.parse_args()
    
    # Check GPU availability and requirements
    check_gpu_requirements()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_gpu_optimized_config()
        
        # Override with command line arguments
        config.update({
            'embeddings_file': args.embeddings,
            'id_to_index_file': args.id_map,
            'papers_directory': args.papers_dir,
            'output_file': args.output,
            'min_cluster_size': args.min_cluster_size,
            'target_dimensions': args.target_dims,
            'chunk_size': args.chunk_size,
            'skip_optimization': args.skip_optimization,
            'use_gpu': not args.no_gpu,
            'gpu_memory_fraction': args.gpu_memory_fraction,
            'pca_batch_size': args.pca_batch_size,
        })
    
    # Log system info
    logger.info(f"System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    if GPU_AVAILABLE and config.get('use_gpu', True):
        try:
            gpu_memory = cp.cuda.runtime.memGetInfo()
            gpu_total_gb = gpu_memory[1] / (1024**3)
            gpu_free_gb = gpu_memory[0] / (1024**3)
            logger.info(f"GPU Memory: {gpu_free_gb:.1f}/{gpu_total_gb:.1f} GB available")
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
    
    # Run clustering
    clusterer = GPUAcceleratedParagraphClusterer(config)
    clusterer.run_gpu_pipeline()


def benchmark_gpu_vs_cpu():
    """
    Benchmark function to compare GPU vs CPU performance on a sample dataset.
    """
    print("\n" + "="*60)
    print("GPU vs CPU CLUSTERING BENCHMARK")
    print("="*60)
    
    if not GPU_AVAILABLE:
        print("GPU not available, cannot run benchmark.")
        return
    
    # Create sample data
    sample_size = 10000
    dimensions = 384
    print(f"Creating sample dataset: {sample_size} points, {dimensions} dimensions")
    
    np.random.seed(42)
    sample_embeddings = np.random.randn(sample_size, dimensions).astype(np.float32)
    
    # Test GPU clustering
    print("\nTesting GPU HDBSCAN...")
    import time
    
    start_time = time.time()
    try:
        sample_gpu = cp.asarray(sample_embeddings, dtype=cp.float32)
        gpu_clusterer = cuHDBSCAN(min_cluster_size=50)
        gpu_labels = cp.asnumpy(gpu_clusterer.fit_predict(sample_gpu))
        gpu_time = time.time() - start_time
        gpu_clusters = len(set(gpu_labels)) - (1 if -1 in gpu_labels else 0)
        print(f"GPU HDBSCAN: {gpu_time:.2f}s, {gpu_clusters} clusters")
        del sample_gpu
        cp.cuda.runtime.deviceSynchronize()
    except Exception as e:
        print(f"GPU HDBSCAN failed: {e}")
        gpu_time = float('inf')
        gpu_clusters = 0
    
    # Test CPU clustering
    print("Testing CPU HDBSCAN...")
    start_time = time.time()
    try:
        cpu_clusterer = hdbscan.HDBSCAN(min_cluster_size=50, core_dist_n_jobs=1)
        cpu_labels = cpu_clusterer.fit_predict(sample_embeddings)
        cpu_time = time.time() - start_time
        cpu_clusters = len(set(cpu_labels)) - (1 if -1 in cpu_labels else 0)
        print(f"CPU HDBSCAN: {cpu_time:.2f}s, {cpu_clusters} clusters")
    except Exception as e:
        print(f"CPU HDBSCAN failed: {e}")
        cpu_time = float('inf')
        cpu_clusters = 0
    
    # Compare results
    if gpu_time != float('inf') and cpu_time != float('inf'):
        speedup = cpu_time / gpu_time
        print(f"\nSpeedup: {speedup:.1f}x faster with GPU")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Optionally run benchmark
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--benchmark":
        benchmark_gpu_vs_cpu()
    else:
        main()