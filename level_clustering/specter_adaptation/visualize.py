#!/usr/bin/env python3
"""
Cluster Visualization and Quality Analysis for SPECTER2 Embeddings

This script visualizes clustering results from cluster_specter_embeddings.py
and calculates comprehensive clustering quality metrics.

Usage:
  python visualize_clusters.py \
    --embeddings_dir ./embeddings_out \
    --cluster_dir ./cluster_out \
    --output_dir ./cluster_analysis \
    --use_gpu --max_samples 50000
"""

import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

# GPU-accelerated imports
try:
    import cuml
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import TSNE as cuTSNE
    from cuml.manifold import UMAP as cuUMAP
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with cuML/RAPIDS")
except ImportError as e:
    print(f"GPU libraries not available: {e}")
    print("Falling back to CPU versions...")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("UMAP not available, will skip UMAP visualizations")
    GPU_AVAILABLE = False

# Standard imports
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import argparse
from collections import Counter, defaultdict
import pickle
import gc
import psutil
import os
from tqdm import tqdm
import networkx as nx

# Optional wordcloud import
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud not available - will skip word cloud generation")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClusterVisualizer:
    """
    Comprehensive visualization and analysis of clustering results
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.embeddings_dir = Path(config['embeddings_dir'])
        self.cluster_dir = Path(config['cluster_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = None
        self.metadata_df = None
        self.cluster_labels = None
        self.cluster_summary = None
        self.gpu_available = GPU_AVAILABLE and config.get('use_gpu', True)
        
        # Visualization data
        self.embeddings_2d = None
        self.embeddings_3d = None
        self.metrics = {}
        
        if self.gpu_available:
            logger.info("GPU acceleration enabled for visualizations")
        else:
            logger.info("Using CPU-only processing")
    
    def load_data(self):
        """Load embeddings and clustering results"""
        logger.info("Loading embeddings and clustering results...")
        
        # Load embeddings (sample if too large)
        max_samples = self.config.get('max_samples', 50000)
        self.embeddings, self.metadata_df = self._load_embeddings_sample(max_samples)
        
        # Load clustering results
        self._load_clustering_results()
        
        logger.info(f"Loaded {len(self.embeddings):,} samples for visualization")
        logger.info(f"Clusters: {len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)}")
    
    def _load_embeddings_sample(self, max_samples: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load a sample of embeddings for visualization"""
        # First check if we have reduced embeddings from clustering
        reduced_emb_file = self.cluster_dir / "reduced_embeddings.npy"
        if reduced_emb_file.exists():
            logger.info("Loading reduced embeddings from clustering results")
            embeddings = np.load(reduced_emb_file)
            # Load corresponding metadata
            metadata_files = sorted(self.cluster_dir.glob("paragraph_clusters_part_*.csv"))
            metadata_list = []
            for file in metadata_files:
                df_chunk = pd.read_csv(file)
                metadata_list.append(df_chunk)
            metadata_df = pd.concat(metadata_list, ignore_index=True)
        else:
            # Load from original embeddings
            from cluster_specter_embeddings import load_embeddings_from_chunks
            embeddings, metadata_df = load_embeddings_from_chunks(self.embeddings_dir)
        
        # Sample if too large
        if len(embeddings) > max_samples:
            logger.info(f"Sampling {max_samples:,} from {len(embeddings):,} total embeddings")
            sample_indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[sample_indices]
            metadata_df = metadata_df.iloc[sample_indices].reset_index(drop=True)
        
        return embeddings.astype(np.float32), metadata_df
    
    def _load_clustering_results(self):
        """Load clustering results"""
        # Load cluster summary
        summary_file = self.cluster_dir / "cluster_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Cluster summary not found: {summary_file}")
        
        with summary_file.open() as f:
            self.cluster_summary = json.load(f)
        
        # Get cluster labels for our sample
        if 'cluster' in self.metadata_df.columns:
            self.cluster_labels = self.metadata_df['cluster'].values
            logger.info("Found cluster labels in metadata")
        else:
            # Load from clustering results files and match with our sample
            logger.info("Loading cluster labels from result files...")
            cluster_files = sorted(self.cluster_dir.glob("paragraph_clusters_part_*.csv"))
            if not cluster_files:
                raise FileNotFoundError("No clustering result files found")
            
            # Load all clustering results
            all_cluster_results = []
            logger.info(f"Loading {len(cluster_files)} clustering result files...")
            
            for file in tqdm(cluster_files, desc="Loading cluster files"):
                try:
                    df_chunk = pd.read_csv(file)
                    all_cluster_results.append(df_chunk)
                except Exception as e:
                    logger.warning(f"Could not load {file}: {e}")
                    continue
            
            if not all_cluster_results:
                raise ValueError("Could not load any clustering result files")
            
            # Combine all results
            full_cluster_df = pd.concat(all_cluster_results, ignore_index=True)
            logger.info(f"Loaded {len(full_cluster_df):,} total clustering results")
            
            # Match with our sample using a combination of identifiers
            # We'll use paper_id + paragraph index or text hash for matching
            self.cluster_labels = self._match_sample_with_results(full_cluster_df)
            
            # Clean up memory
            del all_cluster_results, full_cluster_df
            gc.collect()
    
    def _match_sample_with_results(self, full_cluster_df: pd.DataFrame) -> np.ndarray:
        """Match sample embeddings with full clustering results"""
        logger.info("Matching sample with clustering results...")
        
        # Initialize cluster labels array with -1 (noise/unmatched)
        cluster_labels = np.full(len(self.metadata_df), -1, dtype=int)
        matched_count = 0
        
        # Strategy 1: Try exact matching using multiple columns
        matching_columns = ['paper_id', 'paragraph_idx', 'text']
        available_columns = [col for col in matching_columns if col in self.metadata_df.columns and col in full_cluster_df.columns]
        
        if not available_columns:
            logger.error("No common columns found for matching")
            raise ValueError("Cannot match sample with clustering results - no common identifier columns")
        
        logger.info(f"Using columns for matching: {available_columns}")
        
        # Create composite keys for matching
        if len(available_columns) >= 2:
            # Use multiple columns for more robust matching
            sample_keys = self.metadata_df[available_columns].apply(
                lambda x: tuple(x), axis=1
            )
            result_keys = full_cluster_df[available_columns].apply(
                lambda x: tuple(x), axis=1
            )
            
            # Create lookup dictionary
            result_lookup = dict(zip(result_keys, full_cluster_df['cluster']))
            
            # Match samples
            for i, key in enumerate(sample_keys):
                if key in result_lookup:
                    cluster_labels[i] = result_lookup[key]
                    matched_count += 1
        
        # Strategy 2: If multi-column matching didn't work well, try text-based matching
        if matched_count < len(self.metadata_df) * 0.8:  # Less than 80% matched
            logger.info("Multi-column matching incomplete, trying text-based matching...")
            
            if 'text' in available_columns:
                # Create text hash lookup for remaining unmatched samples
                unmatched_mask = cluster_labels == -1
                
                if unmatched_mask.any():
                    # Create hash lookup for faster matching
                    result_text_lookup = {}
                    for idx, row in full_cluster_df.iterrows():
                        text_hash = hash(row['text'][:200])  # Use first 200 chars to avoid memory issues
                        result_text_lookup[text_hash] = row['cluster']
                    
                    # Match unmatched samples using text hash
                    for i in np.where(unmatched_mask)[0]:
                        text_hash = hash(self.metadata_df.iloc[i]['text'][:200])
                        if text_hash in result_text_lookup:
                            cluster_labels[i] = result_text_lookup[text_hash]
                            matched_count += 1
        
        # Strategy 3: Fuzzy matching for remaining unmatched (if still low match rate)
        if matched_count < len(self.metadata_df) * 0.5:  # Less than 50% matched
            logger.warning("Low match rate, attempting fuzzy matching...")
            self._fuzzy_match_remaining(cluster_labels, full_cluster_df)
            matched_count = (cluster_labels != -1).sum()
        
        match_rate = matched_count / len(self.metadata_df)
        logger.info(f"Successfully matched {matched_count:,}/{len(self.metadata_df):,} samples ({match_rate:.1%})")
        
        if match_rate < 0.3:
            logger.error(f"Very low match rate ({match_rate:.1%}). This may indicate incompatible data.")
            logger.error("Please ensure the clustering results correspond to the same embeddings.")
        elif match_rate < 0.7:
            logger.warning(f"Moderate match rate ({match_rate:.1%}). Some samples will be treated as noise.")
        
        return cluster_labels
    
    def _fuzzy_match_remaining(self, cluster_labels: np.ndarray, full_cluster_df: pd.DataFrame):
        """Attempt fuzzy matching for remaining unmatched samples"""
        try:
            from difflib import SequenceMatcher
        except ImportError:
            logger.warning("difflib not available for fuzzy matching")
            return
        
        unmatched_indices = np.where(cluster_labels == -1)[0]
        if len(unmatched_indices) == 0:
            return
        
        logger.info(f"Attempting fuzzy matching for {len(unmatched_indices)} unmatched samples...")
        
        # Limit fuzzy matching to avoid excessive computation
        max_fuzzy_attempts = min(1000, len(unmatched_indices))
        fuzzy_indices = np.random.choice(unmatched_indices, max_fuzzy_attempts, replace=False)
        
        # Create a sample of result texts for comparison (limit to avoid memory issues)
        max_result_sample = min(5000, len(full_cluster_df))
        result_sample = full_cluster_df.sample(n=max_result_sample, random_state=42)
        
        matched_fuzzy = 0
        
        for sample_idx in tqdm(fuzzy_indices, desc="Fuzzy matching", leave=False):
            sample_text = self.metadata_df.iloc[sample_idx]['text'][:100]  # First 100 chars
            
            best_match_score = 0
            best_match_cluster = -1
            
            for _, result_row in result_sample.iterrows():
                result_text = str(result_row['text'])[:100]
                
                # Calculate similarity
                similarity = SequenceMatcher(None, sample_text, result_text).ratio()
                
                if similarity > best_match_score and similarity > 0.8:  # High threshold for fuzzy matching
                    best_match_score = similarity
                    best_match_cluster = result_row['cluster']
            
            if best_match_cluster != -1:
                cluster_labels[sample_idx] = best_match_cluster
                matched_fuzzy += 1
        
        if matched_fuzzy > 0:
            logger.info(f"Fuzzy matching found {matched_fuzzy} additional matches")
    
    def calculate_clustering_metrics(self) -> Dict:
        """Calculate comprehensive clustering quality metrics"""
        logger.info("Calculating clustering quality metrics...")
        
        # Filter out noise points for metrics that require it
        mask = self.cluster_labels >= 0
        if not mask.any():
            logger.warning("No valid clusters found (all noise)")
            return {}
        
        embeddings_clean = self.embeddings[mask]
        labels_clean = self.cluster_labels[mask]
        
        metrics = {}
        
        try:
            # Internal clustering metrics
            metrics['silhouette_score'] = silhouette_score(embeddings_clean, labels_clean)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings_clean, labels_clean)
            metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings_clean, labels_clean)
            
            # Cluster statistics
            unique_labels = np.unique(self.cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = (self.cluster_labels == -1).sum()
            
            metrics['n_clusters'] = n_clusters
            metrics['n_noise_points'] = int(n_noise)
            metrics['noise_ratio'] = float(n_noise / len(self.cluster_labels))
            metrics['total_points'] = len(self.cluster_labels)
            
            # Cluster size statistics
            cluster_sizes = Counter(labels_clean)
            sizes = list(cluster_sizes.values())
            metrics['cluster_size_mean'] = float(np.mean(sizes))
            metrics['cluster_size_std'] = float(np.std(sizes))
            metrics['cluster_size_min'] = int(np.min(sizes))
            metrics['cluster_size_max'] = int(np.max(sizes))
            metrics['cluster_size_median'] = float(np.median(sizes))
            
            # Density metrics
            metrics.update(self._calculate_density_metrics(embeddings_clean, labels_clean))
            
            # Separation metrics
            metrics.update(self._calculate_separation_metrics(embeddings_clean, labels_clean))
            
            # Stability metrics (if possible)
            if len(embeddings_clean) > 1000:
                metrics.update(self._calculate_stability_metrics(embeddings_clean, labels_clean))
            
            logger.info(f"Calculated {len(metrics)} clustering metrics")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)
        
        self.metrics = metrics
        return metrics
    
    def _calculate_density_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate cluster density metrics"""
        metrics = {}
        
        try:
            cluster_densities = []
            cluster_compactness = []
            
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_points = embeddings[cluster_mask]
                
                if len(cluster_points) < 2:
                    continue
                
                # Calculate average intra-cluster distance
                distances = pdist(cluster_points)
                avg_distance = np.mean(distances)
                cluster_densities.append(1.0 / (1.0 + avg_distance))  # Inverse distance as density
                
                # Calculate compactness (distance to centroid)
                centroid = np.mean(cluster_points, axis=0)
                centroid_distances = np.linalg.norm(cluster_points - centroid, axis=1)
                cluster_compactness.append(np.mean(centroid_distances))
            
            metrics['avg_cluster_density'] = float(np.mean(cluster_densities))
            metrics['avg_cluster_compactness'] = float(np.mean(cluster_compactness))
            metrics['density_std'] = float(np.std(cluster_densities))
            metrics['compactness_std'] = float(np.std(cluster_compactness))
            
        except Exception as e:
            logger.warning(f"Could not calculate density metrics: {e}")
        
        return metrics
    
    def _calculate_separation_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate cluster separation metrics"""
        metrics = {}
        
        try:
            # Calculate centroids
            unique_labels = np.unique(labels)
            centroids = []
            
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                cluster_points = embeddings[cluster_mask]
                centroids.append(np.mean(cluster_points, axis=0))
            
            centroids = np.array(centroids)
            
            # Inter-cluster distances
            if len(centroids) > 1:
                inter_distances = pdist(centroids)
                metrics['avg_inter_cluster_distance'] = float(np.mean(inter_distances))
                metrics['min_inter_cluster_distance'] = float(np.min(inter_distances))
                metrics['max_inter_cluster_distance'] = float(np.max(inter_distances))
                metrics['inter_cluster_distance_std'] = float(np.std(inter_distances))
        
        except Exception as e:
            logger.warning(f"Could not calculate separation metrics: {e}")
        
        return metrics
    
    def _calculate_stability_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate clustering stability using bootstrap sampling"""
        metrics = {}
        
        try:
            n_bootstrap = 5
            bootstrap_scores = []
            
            for i in range(n_bootstrap):
                # Bootstrap sample
                sample_size = min(5000, len(embeddings) // 2)
                indices = np.random.choice(len(embeddings), sample_size, replace=True)
                
                boot_embeddings = embeddings[indices]
                boot_labels = labels[indices]
                
                # Calculate silhouette score for bootstrap sample
                if len(np.unique(boot_labels)) > 1:
                    score = silhouette_score(boot_embeddings, boot_labels)
                    bootstrap_scores.append(score)
            
            if bootstrap_scores:
                metrics['stability_mean'] = float(np.mean(bootstrap_scores))
                metrics['stability_std'] = float(np.std(bootstrap_scores))
                metrics['stability_min'] = float(np.min(bootstrap_scores))
                metrics['stability_max'] = float(np.max(bootstrap_scores))
        
        except Exception as e:
            logger.warning(f"Could not calculate stability metrics: {e}")
        
        return metrics
    
    def create_2d_embeddings(self, method: str = 'umap') -> np.ndarray:
        """Create 2D embeddings for visualization"""
        logger.info(f"Creating 2D embeddings using {method.upper()}...")
        
        if method.lower() == 'umap' and UMAP_AVAILABLE:
            return self._create_umap_2d()
        elif method.lower() == 'tsne':
            return self._create_tsne_2d()
        elif method.lower() == 'pca':
            return self._create_pca_2d()
        else:
            logger.warning(f"Method {method} not available, falling back to PCA")
            return self._create_pca_2d()
    
    def _create_umap_2d(self) -> np.ndarray:
        """Create 2D UMAP embeddings"""
        if self.gpu_available:
            try:
                embeddings_gpu = cp.asarray(self.embeddings, dtype=cp.float32)
                reducer = cuUMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                result = cp.asnumpy(reducer.fit_transform(embeddings_gpu))
                del embeddings_gpu
                cp.cuda.runtime.deviceSynchronize()
                return result
            except Exception as e:
                logger.warning(f"GPU UMAP failed: {e}, using CPU")
        
        # CPU UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(self.embeddings)
    
    def _create_tsne_2d(self) -> np.ndarray:
        """Create 2D t-SNE embeddings"""
        # Sample for t-SNE if too large
        sample_size = min(10000, len(self.embeddings))
        if len(self.embeddings) > sample_size:
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            sample_embeddings = self.embeddings[indices]
            sample_labels = self.cluster_labels[indices]
        else:
            sample_embeddings = self.embeddings
            sample_labels = self.cluster_labels
            indices = np.arange(len(self.embeddings))
        
        if self.gpu_available:
            try:
                sample_gpu = cp.asarray(sample_embeddings, dtype=cp.float32)
                reducer = cuTSNE(n_components=2, random_state=42, perplexity=30)
                result_sample = cp.asnumpy(reducer.fit_transform(sample_gpu))
                del sample_gpu
                cp.cuda.runtime.deviceSynchronize()
            except Exception as e:
                logger.warning(f"GPU t-SNE failed: {e}, using CPU")
                reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=1)
                result_sample = reducer.fit_transform(sample_embeddings)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=1)
            result_sample = reducer.fit_transform(sample_embeddings)
        
        # If we sampled, we need to handle the full dataset
        if len(self.embeddings) > sample_size:
            # Create full result array and fill with NaN
            result_full = np.full((len(self.embeddings), 2), np.nan)
            result_full[indices] = result_sample
            return result_full, indices  # Return indices to track which points were embedded
        
        return result_sample
    
    def _create_pca_2d(self) -> np.ndarray:
        """Create 2D PCA embeddings"""
        if self.gpu_available:
            try:
                embeddings_gpu = cp.asarray(self.embeddings, dtype=cp.float32)
                pca = cuPCA(n_components=2, random_state=42)
                result = cp.asnumpy(pca.fit_transform(embeddings_gpu))
                del embeddings_gpu
                cp.cuda.runtime.deviceSynchronize()
                return result
            except Exception as e:
                logger.warning(f"GPU PCA failed: {e}, using CPU")
        
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(self.embeddings)
    
    def create_3d_embeddings(self, method: str = 'umap') -> np.ndarray:
        """Create 3D embeddings for visualization"""
        logger.info(f"Creating 3D embeddings using {method.upper()}...")
        
        if method.lower() == 'umap' and UMAP_AVAILABLE:
            return self._create_umap_3d()
        elif method.lower() == 'pca':
            return self._create_pca_3d()
        else:
            logger.warning(f"Method {method} not available for 3D, using PCA")
            return self._create_pca_3d()
    
    def _create_umap_3d(self) -> np.ndarray:
        """Create 3D UMAP embeddings"""
        if self.gpu_available:
            try:
                embeddings_gpu = cp.asarray(self.embeddings, dtype=cp.float32)
                reducer = cuUMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
                result = cp.asnumpy(reducer.fit_transform(embeddings_gpu))
                del embeddings_gpu
                cp.cuda.runtime.deviceSynchronize()
                return result
            except Exception as e:
                logger.warning(f"GPU UMAP 3D failed: {e}, using CPU")
        
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(self.embeddings)
    
    def _create_pca_3d(self) -> np.ndarray:
        """Create 3D PCA embeddings"""
        if self.gpu_available:
            try:
                embeddings_gpu = cp.asarray(self.embeddings, dtype=cp.float32)
                pca = cuPCA(n_components=3, random_state=42)
                result = cp.asnumpy(pca.fit_transform(embeddings_gpu))
                del embeddings_gpu
                cp.cuda.runtime.deviceSynchronize()
                return result
            except Exception as e:
                logger.warning(f"GPU PCA 3D failed: {e}, using CPU")
        
        pca = PCA(n_components=3, random_state=42)
        return pca.fit_transform(self.embeddings)
    
    def plot_2d_clusters_matplotlib(self, embeddings_2d: np.ndarray, save_path: str):
        """Create 2D cluster plot using matplotlib"""
        logger.info("Creating 2D matplotlib visualization...")
        
        # Filter out noise for color mapping
        mask = self.cluster_labels >= 0
        noise_mask = self.cluster_labels == -1
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: All points including noise
        if noise_mask.any():
            ax1.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1], 
                       c='lightgray', alpha=0.3, s=1, label='Noise')
        
        if mask.any():
            scatter = ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                                c=self.cluster_labels[mask], cmap='tab20', alpha=0.7, s=2)
            plt.colorbar(scatter, ax=ax1)
        
        ax1.set_title(f'2D Cluster Visualization\n{self.metrics.get("n_clusters", 0)} clusters, '
                     f'{self.metrics.get("noise_ratio", 0):.1%} noise')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.legend()
        
        # Plot 2: Cluster centers and boundaries
        if mask.any():
            # Calculate cluster centers
            unique_clusters = np.unique(self.cluster_labels[mask])
            centers = []
            
            for cluster_id in unique_clusters:
                cluster_mask = (self.cluster_labels == cluster_id) & mask
                if cluster_mask.any():
                    center = np.mean(embeddings_2d[cluster_mask], axis=0)
                    centers.append(center)
                    
                    # Plot cluster points
                    ax2.scatter(embeddings_2d[cluster_mask, 0], embeddings_2d[cluster_mask, 1], 
                              alpha=0.6, s=3, label=f'Cluster {cluster_id}' if len(unique_clusters) <= 20 else None)
            
            # Plot centers
            if centers:
                centers = np.array(centers)
                ax2.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, 
                          linewidths=3, label='Centroids')
        
        ax2.set_title('Cluster Centers and Distribution')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        if len(unique_clusters) <= 20:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved 2D matplotlib plot to {save_path}")
    
    def plot_3d_clusters_plotly(self, embeddings_3d: np.ndarray, save_path: str):
        """Create interactive 3D cluster plot using Plotly"""
        logger.info("Creating 3D Plotly visualization...")
        
        # Prepare data
        mask = self.cluster_labels >= 0
        noise_mask = self.cluster_labels == -1
        
        fig = go.Figure()
        
        # Add noise points
        if noise_mask.any():
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[noise_mask, 0],
                y=embeddings_3d[noise_mask, 1],
                z=embeddings_3d[noise_mask, 2],
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.3),
                name='Noise',
                text=[f'Paper: {pid}' for pid in self.metadata_df.loc[noise_mask, 'paper_id']],
                hovertemplate='<b>Noise Point</b><br>%{text}<extra></extra>'
            ))
        
        # Add clustered points
        if mask.any():
            # Use a color scale for clusters
            cluster_ids = self.cluster_labels[mask]
            unique_clusters = np.unique(cluster_ids)
            
            # Sample clusters if too many for visualization
            if len(unique_clusters) > 20:
                top_clusters = Counter(cluster_ids).most_common(20)
                display_clusters = [c[0] for c in top_clusters]
                cluster_mask = np.isin(cluster_ids, display_clusters)
                
                embeddings_subset = embeddings_3d[mask][cluster_mask]
                labels_subset = cluster_ids[cluster_mask]
                metadata_subset = self.metadata_df.loc[mask].iloc[cluster_mask]
            else:
                embeddings_subset = embeddings_3d[mask]
                labels_subset = cluster_ids
                metadata_subset = self.metadata_df.loc[mask]
            
            fig.add_trace(go.Scatter3d(
                x=embeddings_subset[:, 0],
                y=embeddings_subset[:, 1],
                z=embeddings_subset[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=labels_subset,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Cluster ID")
                ),
                name='Clusters',
                text=[f'Cluster: {label}<br>Paper: {pid}' 
                     for label, pid in zip(labels_subset, metadata_subset['paper_id'])],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        fig.update_layout(
            title=f'3D Cluster Visualization<br>{self.metrics.get("n_clusters", 0)} clusters, '
                  f'{len(self.embeddings):,} points',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=1000,
            height=800
        )
        
        # Save interactive HTML
        pyo.plot(fig, filename=save_path, auto_open=False)
        logger.info(f"Saved 3D Plotly plot to {save_path}")
    
    def plot_cluster_metrics_dashboard(self, save_path: str):
        """Create a comprehensive metrics dashboard"""
        logger.info("Creating metrics dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clustering Quality Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Cluster size distribution
        cluster_sizes = Counter(self.cluster_labels[self.cluster_labels >= 0])
        sizes = list(cluster_sizes.values())
        
        axes[0, 0].hist(sizes, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster Size')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(sizes), color='red', linestyle='--', label=f'Mean: {np.mean(sizes):.0f}')
        axes[0, 0].legend()
        
        # 2. Quality metrics bar plot
        quality_metrics = {
            'Silhouette': self.metrics.get('silhouette_score', 0),
            'Calinski-Harabasz': self.metrics.get('calinski_harabasz_score', 0) / 1000,  # Scale down
            'Davies-Bouldin': 1 / (1 + self.metrics.get('davies_bouldin_score', 1))  # Invert (lower is better)
        }
        
        bars = axes[0, 1].bar(quality_metrics.keys(), quality_metrics.values(), 
                             color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 1].set_title('Quality Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, quality_metrics.values()):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Noise ratio pie chart
        noise_count = self.metrics.get('n_noise_points', 0)
        clustered_count = self.metrics.get('total_points', 1) - noise_count
        
        axes[0, 2].pie([clustered_count, noise_count], 
                      labels=['Clustered', 'Noise'], 
                      autopct='%1.1f%%',
                      colors=['lightblue', 'lightcoral'])
        axes[0, 2].set_title('Points Distribution')
        
        # 4. Top clusters by size
        top_clusters = Counter(self.cluster_labels[self.cluster_labels >= 0]).most_common(15)
        if top_clusters:
            cluster_ids, cluster_counts = zip(*top_clusters)
            axes[1, 0].bar(range(len(cluster_ids)), cluster_counts, color='steelblue')
            axes[1, 0].set_title('Top 15 Clusters by Size')
            axes[1, 0].set_xlabel('Cluster Rank')
            axes[1, 0].set_ylabel('Number of Points')
            axes[1, 0].set_xticks(range(len(cluster_ids)))
            axes[1, 0].set_xticklabels([f'C{cid}' for cid in cluster_ids], rotation=45)
        
        # 5. Distance distribution
        if len(self.embeddings) <= 5000:  # Only for smaller datasets
            try:
                sample_indices = np.random.choice(len(self.embeddings), 
                                                min(2000, len(self.embeddings)), replace=False)
                sample_embeddings = self.embeddings[sample_indices]
                distances = pdist(sample_embeddings)
                
                axes[1, 1].hist(distances, bins=50, alpha=0.7, color='orange', edgecolor='black')
                axes[1, 1].set_title('Pairwise Distance Distribution')
                axes[1, 1].set_xlabel('Distance')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].axvline(np.mean(distances), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(distances):.3f}')
                axes[1, 1].legend()
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Distance plot unavailable\n{str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Distance Distribution (Error)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Distance plot skipped\n(dataset too large)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Distance Distribution (Skipped)')
        
        # 6. Metrics summary table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        metrics_table = [
            ['Metric', 'Value'],
            ['Total Points', f"{self.metrics.get('total_points', 0):,}"],
            ['Number of Clusters', f"{self.metrics.get('n_clusters', 0):,}"],
            ['Noise Points', f"{self.metrics.get('n_noise_points', 0):,}"],
            ['Noise Ratio', f"{self.metrics.get('noise_ratio', 0):.1%}"],
            ['Silhouette Score', f"{self.metrics.get('silhouette_score', 0):.3f}"],
            ['Calinski-Harabasz', f"{self.metrics.get('calinski_harabasz_index', 0):.1f}"],
            ['Davies-Bouldin', f"{self.metrics.get('davies_bouldin_score', 0):.3f}"],
            ['Avg Cluster Size', f"{self.metrics.get('cluster_size_mean', 0):.1f}"],
            ['Cluster Size Std', f"{self.metrics.get('cluster_size_std', 0):.1f}"]
        ]
        
        table = axes[1, 2].table(cellText=metrics_table[1:], colLabels=metrics_table[0],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 2].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics dashboard to {save_path}")
    
    def create_interactive_plotly_dashboard(self, embeddings_2d: np.ndarray, save_path: str):
        """Create interactive Plotly dashboard"""
        logger.info("Creating interactive Plotly dashboard...")
        
        # Ensure save_path is a string
        save_path = str(save_path)
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('2D Cluster Visualization', 'Cluster Size Distribution', 
                          'Quality Metrics', 'Top Clusters'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. 2D scatter plot
        mask = self.cluster_labels >= 0
        noise_mask = self.cluster_labels == -1
        
        # Add noise points
        if noise_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[noise_mask, 0],
                    y=embeddings_2d[noise_mask, 1],
                    mode='markers',
                    marker=dict(size=3, color='lightgray', opacity=0.4),
                    name='Noise',
                    text=[f'Paper: {pid}' for pid in self.metadata_df.loc[noise_mask, 'paper_id']],
                    hovertemplate='<b>Noise Point</b><br>%{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add clustered points
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=self.cluster_labels[mask],
                        colorscale='Viridis',
                        opacity=0.7,
                        colorbar=dict(title="Cluster ID", x=0.48)
                    ),
                    name='Clusters',
                    text=[f'Cluster: {label}<br>Paper: {pid}' 
                         for label, pid in zip(self.cluster_labels[mask], 
                                             self.metadata_df.loc[mask, 'paper_id'])],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Cluster size distribution
        cluster_sizes = Counter(self.cluster_labels[self.cluster_labels >= 0])
        sizes = list(cluster_sizes.values())
        
        fig.add_trace(
            go.Histogram(x=sizes, nbinsx=30, name='Cluster Sizes', marker_color='skyblue'),
            row=1, col=2
        )
        
        # 3. Quality metrics
        quality_metrics = {
            'Silhouette': self.metrics.get('silhouette_score', 0),
            'Calinski-Harabasz<br>(scaled)': self.metrics.get('calinski_harabasz_index', 0) / 1000,
            'Davies-Bouldin<br>(inverted)': 1 / (1 + self.metrics.get('davies_bouldin_score', 1))
        }
        
        fig.add_trace(
            go.Bar(
                x=list(quality_metrics.keys()),
                y=list(quality_metrics.values()),
                name='Quality Metrics',
                marker_color=['lightblue', 'lightgreen', 'lightsalmon']
            ),
            row=2, col=1
        )
        
        # 4. Top clusters by size
        top_clusters = Counter(self.cluster_labels[self.cluster_labels >= 0]).most_common(15)
        if top_clusters:
            cluster_ids, cluster_counts = zip(*top_clusters)
            
            fig.add_trace(
                go.Bar(
                    x=[f'C{cid}' for cid in cluster_ids],
                    y=list(cluster_counts),  # Ensure it's a list
                    name='Top Clusters',
                    marker_color='steelblue'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"Clustering Analysis Dashboard - {self.metrics.get('n_clusters', 0)} Clusters",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Dimension 1", row=1, col=1)
        fig.update_yaxes(title_text="Dimension 2", row=1, col=1)
        fig.update_xaxes(title_text="Cluster Size", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Metric", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_xaxes(title_text="Cluster ID", row=2, col=2)
        fig.update_yaxes(title_text="Size", row=2, col=2)
        
        # Save
        pyo.plot(fig, filename=save_path, auto_open=False)
        logger.info(f"Saved interactive dashboard to {save_path}")
    
    def generate_cluster_wordclouds(self, save_dir: str, n_top_clusters: int = 10):
        """Generate word clouds for top clusters"""
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud library not available, skipping word cloud generation")
            return
            
        logger.info(f"Generating word clouds for top {n_top_clusters} clusters...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if 'top_terms' not in self.cluster_summary:
            logger.warning("No top terms found in cluster summary")
            return
        
        top_terms = self.cluster_summary['top_terms']
        
        # Get cluster sizes to determine top clusters
        cluster_sizes = Counter(self.cluster_labels[self.cluster_labels >= 0])
        top_clusters = cluster_sizes.most_common(n_top_clusters)
        
        for rank, (cluster_id, size) in enumerate(top_clusters):
            cluster_id_str = str(cluster_id)
            if cluster_id_str not in top_terms:
                continue
            
            # Create word cloud from top terms
            terms = top_terms[cluster_id_str]
            text = ' '.join(terms)
            
            try:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=50,
                    colormap='viridis'
                ).generate(text)
                
                # Plot
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Cluster {cluster_id} - Size: {size:,} paragraphs\nTop Terms', 
                         fontsize=14, fontweight='bold')
                
                # Save
                save_path = save_dir / f'cluster_{cluster_id:03d}_wordcloud.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.warning(f"Could not create word cloud for cluster {cluster_id}: {e}")
        
        logger.info(f"Generated word clouds in {save_dir}")
    
    def analyze_cluster_characteristics(self) -> Dict:
        """Analyze characteristics of each cluster"""
        logger.info("Analyzing cluster characteristics...")
        
        characteristics = {}
        
        # Get clustered data only
        mask = self.cluster_labels >= 0
        if not mask.any():
            return characteristics
        
        clustered_metadata = self.metadata_df[mask].copy()
        clustered_metadata['cluster'] = self.cluster_labels[mask]
        
        unique_clusters = np.unique(self.cluster_labels[mask])
        
        for cluster_id in unique_clusters:
            cluster_mask = clustered_metadata['cluster'] == cluster_id
            cluster_data = clustered_metadata[cluster_mask]
            
            char = {}
            
            # Basic statistics
            char['size'] = int(len(cluster_data))
            char['papers'] = int(cluster_data['paper_id'].nunique())
            char['avg_paragraph_length'] = float(cluster_data['text'].str.len().mean())
            char['std_paragraph_length'] = float(cluster_data['text'].str.len().std())
            
            # Paper diversity
            paper_counts = cluster_data['paper_id'].value_counts()
            char['max_paragraphs_per_paper'] = int(paper_counts.max())
            char['min_paragraphs_per_paper'] = int(paper_counts.min())
            char['avg_paragraphs_per_paper'] = float(paper_counts.mean())
            
            # Content analysis
            if 'section' in cluster_data.columns:
                section_counts = cluster_data['section'].value_counts().head()
                char['top_sections'] = {str(k): int(v) for k, v in section_counts.to_dict().items()}
            
            # Top terms from summary
            if 'top_terms' in self.cluster_summary:
                cluster_id_str = str(cluster_id)
                if cluster_id_str in self.cluster_summary['top_terms']:
                    char['top_terms'] = self.cluster_summary['top_terms'][cluster_id_str][:8]
            
            characteristics[int(cluster_id)] = char
        
        return characteristics
    
    def create_cluster_network_graph(self, save_path: str, threshold: float = 0.1):
        """Create network graph of cluster relationships"""
        logger.info("Creating cluster network graph...")
        
        # Calculate cluster centroids
        mask = self.cluster_labels >= 0
        if not mask.any():
            logger.warning("No clusters to create network graph")
            return
        
        unique_clusters = np.unique(self.cluster_labels[mask])
        centroids = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = (self.cluster_labels == cluster_id) & mask
            if cluster_mask.any():
                centroids[cluster_id] = np.mean(self.embeddings[cluster_mask], axis=0)
        
        # Calculate pairwise similarities between centroids
        cluster_ids = list(centroids.keys())
        similarities = np.zeros((len(cluster_ids), len(cluster_ids)))
        
        for i, id1 in enumerate(cluster_ids):
            for j, id2 in enumerate(cluster_ids):
                if i != j:
                    # Cosine similarity
                    sim = np.dot(centroids[id1], centroids[id2]) / (
                        np.linalg.norm(centroids[id1]) * np.linalg.norm(centroids[id2]))
                    similarities[i, j] = sim
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (clusters)
        cluster_sizes = Counter(self.cluster_labels[mask])
        for cluster_id in cluster_ids:
            size = cluster_sizes[cluster_id]
            G.add_node(cluster_id, size=size)
        
        # Add edges based on similarity threshold
        for i, id1 in enumerate(cluster_ids):
            for j, id2 in enumerate(cluster_ids):
                if i < j and similarities[i, j] > threshold:
                    G.add_edge(id1, id2, weight=similarities[i, j])
        
        # Plot network
        plt.figure(figsize=(15, 12))
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes with size proportional to cluster size
        node_sizes = [cluster_sizes[node] * 10 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                              alpha=0.7, edgecolors='black')
        
        # Draw edges with thickness proportional to similarity
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f'Cluster Network Graph\nSimilarity Threshold: {threshold:.2f}', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cluster network graph to {save_path}")
        
        return G
    
    def generate_comprehensive_report(self, save_path: str):
        """Generate a comprehensive HTML report"""
        logger.info("Generating comprehensive HTML report...")
        
        # Calculate cluster characteristics
        characteristics = self.analyze_cluster_characteristics()
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clustering Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 15px; 
                             background-color: #e8f4fd; border-radius: 5px; min-width: 150px; }}
                .cluster-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .cluster-table th, .cluster-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .cluster-table th {{ background-color: #f2f2f2; }}
                .section {{ margin: 30px 0; }}
                .top-terms {{ font-style: italic; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SPECTER2 Clustering Analysis Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric-box">
                    <strong>Total Paragraphs</strong><br>
                    {self.metrics.get('total_points', 0):,}
                </div>
                <div class="metric-box">
                    <strong>Clusters Found</strong><br>
                    {self.metrics.get('n_clusters', 0):,}
                </div>
                <div class="metric-box">
                    <strong>Noise Ratio</strong><br>
                    {self.metrics.get('noise_ratio', 0):.1%}
                </div>
                <div class="metric-box">
                    <strong>Silhouette Score</strong><br>
                    {self.metrics.get('silhouette_score', 0):.3f}
                </div>
                <div class="metric-box">
                    <strong>Avg Cluster Size</strong><br>
                    {self.metrics.get('cluster_size_mean', 0):.0f}
                </div>
            </div>
            
            <div class="section">
                <h2>Quality Metrics</h2>
                <table class="cluster-table">
                    <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                    <tr>
                        <td>Silhouette Score</td>
                        <td>{self.metrics.get('silhouette_score', 0):.3f}</td>
                        <td>{'Excellent' if self.metrics.get('silhouette_score', 0) > 0.7 else 
                            'Good' if self.metrics.get('silhouette_score', 0) > 0.5 else 
                            'Fair' if self.metrics.get('silhouette_score', 0) > 0.25 else 'Poor'}</td>
                    </tr>
                    <tr>
                        <td>Calinski-Harabasz Score</td>
                        <td>{self.metrics.get('calinski_harabasz_score', 0):.1f}</td>
                        <td>Higher is better (cluster separation)</td>
                    </tr>
                    <tr>
                        <td>Davies-Bouldin Score</td>
                        <td>{self.metrics.get('davies_bouldin_score', 0):.3f}</td>
                        <td>Lower is better (cluster compactness)</td>
                    </tr>
                    <tr>
                        <td>Average Density</td>
                        <td>{self.metrics.get('avg_cluster_density', 0):.3f}</td>
                        <td>Higher indicates more compact clusters</td>
                    </tr>
                    <tr>
                        <td>Average Inter-cluster Distance</td>
                        <td>{self.metrics.get('avg_inter_cluster_distance', 0):.3f}</td>
                        <td>Higher indicates better separation</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Top Clusters by Size</h2>
                <table class="cluster-table">
                    <tr><th>Cluster ID</th><th>Size</th><th>Papers</th><th>Avg Para Length</th><th>Top Terms</th></tr>
        """
        
        # Add top clusters to table
        cluster_sizes = Counter(self.cluster_labels[self.cluster_labels >= 0])
        top_clusters = cluster_sizes.most_common(15)
        
        for cluster_id, size in top_clusters:
            if cluster_id in characteristics:
                char = characteristics[cluster_id]
                top_terms = ', '.join(char.get('top_terms', [])[:6])
                html_content += f"""
                    <tr>
                        <td>{cluster_id}</td>
                        <td>{size:,}</td>
                        <td>{char.get('papers', 0):,}</td>
                        <td>{char.get('avg_paragraph_length', 0):.0f}</td>
                        <td class="top-terms">{top_terms}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Clustering Configuration</h2>
                <ul>
        """
        
        # Add configuration details
        if 'clustering_params' in self.cluster_summary:
            params = self.cluster_summary['clustering_params']
            for key, value in params.items():
                html_content += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Files Generated</h2>
                <ul>
                    <li><strong>2D Visualization:</strong> cluster_2d_matplotlib.png</li>
                    <li><strong>3D Visualization:</strong> cluster_3d_plotly.html</li>
                    <li><strong>Interactive Dashboard:</strong> interactive_dashboard.html</li>
                    <li><strong>Metrics Dashboard:</strong> metrics_dashboard.png</li>
                    <li><strong>Network Graph:</strong> cluster_network.png</li>
                    <li><strong>Word Clouds:</strong> wordclouds/ directory</li>
                    <li><strong>Detailed Metrics:</strong> detailed_metrics.json</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved comprehensive report to {save_path}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        else:
            return obj
    
    def save_detailed_metrics(self, save_path: str):
        """Save detailed metrics to JSON"""
        logger.info("Saving detailed metrics...")
        
        # Include cluster characteristics
        characteristics = self.analyze_cluster_characteristics()
        
        detailed_metrics = {
            'clustering_quality': self.metrics,
            'cluster_characteristics': characteristics,
            'summary_statistics': {
                'total_embeddings_analyzed': len(self.embeddings),
                'embedding_dimension': self.embeddings.shape[1],
                'visualization_methods_used': self.config.get('methods', ['pca', 'umap']),
                'gpu_acceleration': self.gpu_available
            },
            'cluster_summary': self.cluster_summary
        }
        
        # Convert to JSON serializable format
        detailed_metrics = self._convert_to_json_serializable(detailed_metrics)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_metrics, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved detailed metrics to {save_path}")
    
    def run_visualization_pipeline(self):
        """Run the complete visualization and analysis pipeline"""
        logger.info("Starting clustering visualization and analysis pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Calculate metrics
            self.calculate_clustering_metrics()
            
            # Create 2D embeddings and visualizations
            methods_2d = ['pca', 'umap'] if UMAP_AVAILABLE else ['pca']
            if 'tsne' in self.config.get('methods', []):
                methods_2d.append('tsne')
            
            for method in methods_2d:
                try:
                    logger.info(f"Creating 2D visualization with {method.upper()}...")
                    embeddings_2d = self.create_2d_embeddings(method)
                    
                    # Handle t-SNE special case (may return indices)
                    if isinstance(embeddings_2d, tuple):
                        embeddings_2d, valid_indices = embeddings_2d
                        # Use only valid points for visualization
                        valid_mask = ~np.isnan(embeddings_2d[:, 0])
                        embeddings_2d = embeddings_2d[valid_mask]
                        temp_labels = self.cluster_labels[valid_mask]
                        temp_metadata = self.metadata_df.iloc[valid_mask]
                    else:
                        temp_labels = self.cluster_labels
                        temp_metadata = self.metadata_df
                    
                    # Store for other uses
                    if method == 'umap' or (method == 'pca' and self.embeddings_2d is None):
                        self.embeddings_2d = embeddings_2d
                    
                    # Save 2D matplotlib plot
                    save_path = self.output_dir / f'cluster_2d_{method}.png'
                    
                    # Temporarily swap data if needed
                    original_labels = self.cluster_labels
                    original_metadata = self.metadata_df
                    if isinstance(embeddings_2d, tuple) or 'temp_labels' in locals():
                        self.cluster_labels = temp_labels
                        self.metadata_df = temp_metadata
                    
                    self.plot_2d_clusters_matplotlib(embeddings_2d, save_path)
                    
                    # Restore original data
                    self.cluster_labels = original_labels
                    self.metadata_df = original_metadata
                    
                except Exception as e:
                    logger.error(f"Error creating 2D visualization with {method}: {e}")
            
            # Use the best 2D embeddings for dashboard
            if self.embeddings_2d is None:
                logger.warning("No 2D embeddings created, using PCA for dashboard")
                self.embeddings_2d = self.create_2d_embeddings('pca')
            
            # Create 3D embeddings and visualization
            try:
                self.embeddings_3d = self.create_3d_embeddings('umap' if UMAP_AVAILABLE else 'pca')
                save_path_3d = self.output_dir / 'cluster_3d_plotly.html'
                self.plot_3d_clusters_plotly(self.embeddings_3d, str(save_path_3d))
            except Exception as e:
                logger.error(f"Error creating 3D visualization: {e}")
            
            # Create metrics dashboard
            try:
                dashboard_path = self.output_dir / 'metrics_dashboard.png'
                self.plot_cluster_metrics_dashboard(dashboard_path)
            except Exception as e:
                logger.error(f"Error creating metrics dashboard: {e}")
        except Exception as e:
            logger.error(f"First part: {e}")


        try:
            # Create interactive dashboard
            if not self.config.get('skip_interactive', False):
                interactive_path = self.output_dir / 'interactive_dashboard.html'
                self.create_interactive_plotly_dashboard(self.embeddings_2d, str(interactive_path))
            else:
                logger.info("Skipping interactive dashboard (disabled)")
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            
            # Generate word clouds
            try:
                if not self.config.get('skip_wordclouds', False):
                    wordcloud_dir = self.output_dir / 'wordclouds'
                    self.generate_cluster_wordclouds(wordcloud_dir)
                else:
                    logger.info("Skipping word cloud generation (disabled)")
            except Exception as e:
                logger.error(f"Error creating word clouds: {e}")
            
            # Create network graph
            try:
                if not self.config.get('skip_network', False):
                    network_path = self.output_dir / 'cluster_network.png'
                    self.create_cluster_network_graph(str(network_path))
                else:
                    logger.info("Skipping network graph (disabled)")
            except Exception as e:
                logger.error(f"Error creating network graph: {e}")
            
            # Save detailed metrics
            try:
                metrics_path = self.output_dir / 'detailed_metrics.json'
                self.save_detailed_metrics(str(metrics_path))
            except Exception as e:
                logger.error(f"Error saving detailed metrics: {e}")
            
            # Generate comprehensive report
            try:
                report_path = self.output_dir / 'clustering_report.html'
                self.generate_comprehensive_report(str(report_path))
            except Exception as e:
                logger.error(f"Error generating comprehensive report: {e}")
            
            logger.info("Visualization pipeline completed successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            
            return {
                'metrics': self.metrics,
                'output_dir': str(self.output_dir),
                'files_generated': [
                    'cluster_2d_*.png',
                    'cluster_3d_plotly.html',
                    'metrics_dashboard.png',
                    'interactive_dashboard.html',
                    'cluster_network.png',
                    'wordclouds/',
                    'detailed_metrics.json',
                    'clustering_report.html'
                ]
            }
            
        except Exception as e:
            logger.error(f"Visualization pipeline failed: {e}")
            raise


def print_quick_metrics(metrics: Dict):
    """Print a quick summary of clustering metrics"""
    print("\n" + "="*50)
    print("CLUSTERING QUALITY SUMMARY")
    print("="*50)
    print(f"Total Points: {metrics.get('total_points', 0):,}")
    print(f"Clusters Found: {metrics.get('n_clusters', 0):,}")
    print(f"Noise Points: {metrics.get('n_noise_points', 0):,} ({metrics.get('noise_ratio', 0):.1%})")
    print(f"Silhouette Score: {metrics.get('silhouette_score', 0):.3f}")
    print(f"Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 0):.1f}")
    print(f"Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 0):.3f}")
    print(f"Average Cluster Size: {metrics.get('cluster_size_mean', 0):.1f} ± {metrics.get('cluster_size_std', 0):.1f}")
    
    # Quality assessment
    sil_score = metrics.get('silhouette_score', 0)
    if sil_score > 0.7:
        quality = "EXCELLENT"
    elif sil_score > 0.5:
        quality = "GOOD" 
    elif sil_score > 0.25:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    print(f"\nOverall Quality Assessment: {quality}")
    print("="*50 + "\n")


def main():
    """Main function for clustering visualization and analysis"""
    parser = argparse.ArgumentParser(description='Clustering Visualization and Quality Analysis')
    parser.add_argument('--embeddings_dir', required=True, 
                       help='Directory with original embeddings from embed_paragraphs.py')
    parser.add_argument('--cluster_dir', required=True, 
                       help='Directory with clustering results from cluster_specter_embeddings.py')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for visualizations and analysis')
    
    # Visualization parameters
    parser.add_argument('--max_samples', type=int, default=50000, 
                       help='Maximum number of samples to use for visualization')
    parser.add_argument('--methods', nargs='+', default=['pca', 'umap'], 
                       choices=['pca', 'umap', 'tsne'],
                       help='Dimensionality reduction methods to use')
    parser.add_argument('--use_gpu', action='store_true', 
                       help='Enable GPU acceleration')
    
    # Analysis parameters
    parser.add_argument('--network_threshold', type=float, default=0.1,
                       help='Similarity threshold for cluster network graph')
    parser.add_argument('--n_wordclouds', type=int, default=10,
                       help='Number of word clouds to generate for top clusters')
    
    # Output options
    parser.add_argument('--skip_3d', action='store_true',
                       help='Skip 3D visualizations (faster)')
    parser.add_argument('--skip_wordclouds', action='store_true',
                       help='Skip word cloud generation')
    parser.add_argument('--skip_network', action='store_true',
                       help='Skip network graph generation')
    
    args = parser.parse_args()
    
    # Verify input directories exist
    embeddings_dir = Path(args.embeddings_dir)
    cluster_dir = Path(args.cluster_dir)
    
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return
    
    if not cluster_dir.exists():
        logger.error(f"Cluster directory not found: {cluster_dir}")
        return
    
    # Check for required files
    cluster_summary = cluster_dir / "cluster_summary.json"
    if not cluster_summary.exists():
        logger.error(f"Cluster summary not found: {cluster_summary}")
        logger.error("Please run cluster_specter_embeddings.py first")
        return
    
    # Create config
    config = {
        'embeddings_dir': str(embeddings_dir),
        'cluster_dir': str(cluster_dir),
        'output_dir': args.output_dir,
        'max_samples': args.max_samples,
        'methods': args.methods,
        'use_gpu': args.use_gpu,
        'network_threshold': args.network_threshold,
        'n_wordclouds': args.n_wordclouds,
        'skip_3d': args.skip_3d,
        'skip_wordclouds': args.skip_wordclouds,
        'skip_network': args.skip_network
    }
    
    # Log configuration
    logger.info("Visualization Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Check system resources
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    logger.info(f"Available memory: {available_gb:.1f} GB")
    
    if available_gb < 4:
        logger.warning("Low memory available. Consider reducing max_samples.")
    
    # Run visualization pipeline
    try:
        visualizer = ClusterVisualizer(config)
        results = visualizer.run_visualization_pipeline()
        
        # Print quick summary
        print_quick_metrics(visualizer.metrics)
        
        logger.info("Visualization and analysis completed successfully!")
        logger.info(f"Output directory: {results['output_dir']}")
        logger.info("Generated files:")
        for file_pattern in results['files_generated']:
            logger.info(f"  - {file_pattern}")
        
        # Print recommendations
        sil_score = visualizer.metrics.get('silhouette_score', 0)
        n_clusters = visualizer.metrics.get('n_clusters', 0)
        noise_ratio = visualizer.metrics.get('noise_ratio', 0)
        
        print("\nRECOMMENDAIONS:")
        print("-" * 30)
        
        if sil_score < 0.25:
            print("• Consider adjusting clustering parameters (min_cluster_size, target_dimensions)")
        if noise_ratio > 0.3:
            print("• High noise ratio - consider lowering min_cluster_size or improving data preprocessing")
        if n_clusters > 500:
            print("• Very large number of clusters - consider increasing min_cluster_size")
        if n_clusters < 10:
            print("• Few clusters found - consider decreasing min_cluster_size")
        
        if sil_score > 0.5 and 0.1 < noise_ratio < 0.2:
            print("• Clustering quality looks good! ✓")
        
    except Exception as e:
        logger.error(f"Visualization pipeline failed: {e}")
        raise


def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        missing_deps.append("matplotlib, seaborn")
    
    try:
        import plotly
    except ImportError:
        missing_deps.append("plotly")
    
    try:
        import networkx
    except ImportError:
        missing_deps.append("networkx")
    
    try:
        import wordcloud
        global WORDCLOUD_AVAILABLE
        WORDCLOUD_AVAILABLE = True
    except ImportError:
        logger.warning("WordCloud not available - will skip word cloud visualizations")
        missing_deps.append("wordcloud")
        WORDCLOUD_AVAILABLE = False
    
    try:
        import umap
        global UMAP_AVAILABLE
        UMAP_AVAILABLE = True
    except ImportError:
        logger.warning("UMAP not available - will skip UMAP visualizations")
        UMAP_AVAILABLE = False
    
    if missing_deps:
        logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        logger.error("Please install with: pip install matplotlib seaborn plotly networkx")
        if 'wordcloud' in missing_deps:
            logger.info("Note: wordcloud is optional - install with: pip install wordcloud")
        return False
    
    logger.info("All dependencies available ✓")
    return True


if __name__ == "__main__":
    if not check_dependencies():
        exit(1)
    main()