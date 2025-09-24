#!/usr/bin/env python3
"""
Thesis Clustering Framework - Complete End-to-End Pipeline
Handles both paragraph-level and methodological summary pipelines with checkpointing

Author: Thesis Research Framework
"""

import os
import json
import pickle
import yaml
import argparse
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, pairwise_distances
)
from sklearn.decomposition import PCA
import hdbscan
import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

class CheckpointManager:
    """Manages checkpointing for robust recovery from failures"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / 'pipeline_state.json'
        self.load_state()
    
    def load_state(self):
        """Load existing pipeline state or initialize new"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            print(f"Loaded checkpoint state from {self.state_file}")
        else:
            self.state = {
                'completed_steps': [],
                'current_step': None,
                'pipeline_config': {},
                'timestamps': {}
            }
    
    def save_state(self):
        """Save current pipeline state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def save_checkpoint(self, step_name: str, data: Any, metadata: Dict = None):
        """Save checkpoint for a specific step"""
        checkpoint_file = self.checkpoint_dir / f'{step_name}.pkl'
        checkpoint_data = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Update state
        if step_name not in self.state['completed_steps']:
            self.state['completed_steps'].append(step_name)
        self.state['timestamps'][step_name] = datetime.now().isoformat()
        self.save_state()
        
        print(f"✓ Checkpoint saved: {step_name}")
    
    def load_checkpoint(self, step_name: str) -> Tuple[Any, Dict]:
        """Load checkpoint for a specific step"""
        checkpoint_file = self.checkpoint_dir / f'{step_name}.pkl'
        
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"✓ Checkpoint loaded: {step_name}")
            return checkpoint_data['data'], checkpoint_data.get('metadata', {})
        return None, None
    
    def is_completed(self, step_name: str) -> bool:
        """Check if a step has been completed"""
        return step_name in self.state['completed_steps']
    
    def clear_downstream(self, step_name: str):
        """Clear all checkpoints downstream of a given step"""
        # Define step dependencies
        step_order = [
            'data_loading',
            'preprocessing',
            'dimensionality_reduction',
            'clustering',
            'noise_reassignment',
            'visualization',
            'evaluation',
            'report_generation'
        ]
        
        if step_name in step_order:
            step_idx = step_order.index(step_name)
            for downstream_step in step_order[step_idx + 1:]:
                if downstream_step in self.state['completed_steps']:
                    self.state['completed_steps'].remove(downstream_step)
                checkpoint_file = self.checkpoint_dir / f'{downstream_step}.pkl'
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
            self.save_state()


class ThesisClusteringPipeline:
    """Main pipeline for thesis clustering experiments"""
    
    def __init__(self, config: Dict, pipeline_type: str, checkpoint_dir: Path = None):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
            pipeline_type: Either 'paragraph' or 'methodological'
            checkpoint_dir: Directory for checkpoints
        """
        self.config = config
        self.pipeline_type = pipeline_type
        self.random_state = config.get('random_state', 42)
        np.random.seed(self.random_state)
        
        # Set up directories
        self.work_dir = Path(config['work_dir']) / pipeline_type
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        if checkpoint_dir is None:
            checkpoint_dir = self.work_dir / 'checkpoints'
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Store results
        self.results = {}
        self.metrics = {}
        
        print(f"Initialized {pipeline_type} pipeline")
        print(f"Work directory: {self.work_dir}")
    
    def run_or_load_step(self, step_name: str, step_function, *args, **kwargs):
        """Run a step or load from checkpoint if completed"""
        if self.checkpoint_manager.is_completed(step_name):
            print(f"\n[{step_name}] Loading from checkpoint...")
            data, metadata = self.checkpoint_manager.load_checkpoint(step_name)
            if metadata:
                self.metrics[step_name] = metadata
            return data
        else:
            print(f"\n[{step_name}] Running step...")
            result = step_function(*args, **kwargs)
            
            # Extract data and metadata if step returns tuple
            if isinstance(result, tuple) and len(result) == 2:
                data, metadata = result
            else:
                data = result
                metadata = {}
            
            self.checkpoint_manager.save_checkpoint(step_name, data, metadata)
            if metadata:
                self.metrics[step_name] = metadata
            return data
    
    def load_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load and validate input data"""
        def _load():
            print(f"Loading data from {self.config['data_path']}")
            df = pd.read_csv(self.config['data_path'])
            
            # Extract features based on pipeline type
            if self.pipeline_type == 'paragraph':
                # For paragraph pipeline: load paragraph embeddings
                embedding_cols = [col for col in df.columns if col.startswith(self.config['paragraph_embedding_prefix'])]
            else:
                # For methodological pipeline: load summary embeddings
                embedding_cols = [col for col in df.columns if col.startswith(self.config['method_embedding_prefix'])]
            
            X = df[embedding_cols].values
            
            # Validate data
            print(f"Data shape: {X.shape}")
            print(f"Sparsity: {(X == 0).mean():.3f}")
            
            # Clean NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            metadata = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'sparsity': float((X == 0).mean()),
                'pipeline_type': self.pipeline_type
            }
            
            return (X, df), metadata
        
        return self.run_or_load_step('data_loading', _load)
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing based on configuration"""
        def _preprocess():
            method = self.config['preprocessing']['method']
            print(f"Applying {method} preprocessing")
            
            if method == 'none':
                X_processed = X
            elif method == 'standard':
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X)
            elif method == 'l2_normalize':
                from sklearn.preprocessing import Normalizer
                normalizer = Normalizer(norm='l2')
                X_processed = normalizer.fit_transform(X)
            else:
                raise ValueError(f"Unknown preprocessing: {method}")
            
            metadata = {
                'method': method,
                'shape_before': X.shape,
                'shape_after': X_processed.shape
            }
            
            return X_processed, metadata
        
        return self.run_or_load_step('preprocessing', _preprocess, X)
    
    def apply_dimensionality_reduction(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dimensionality reduction for clustering and visualization
        
        Reasoning for choice:
        - UMAP for clustering: Preserves global and local structure
        - UMAP/t-SNE for 2D visualization: Better separation for visual inspection
        - No DR option: When features are already meaningful and low-dimensional
        """
        def _reduce():
            dr_config = self.config['dimensionality_reduction']
            
            # For clustering
            if dr_config['for_clustering']['method'] == 'none':
                print("No dimensionality reduction for clustering")
                X_clustering = X
            elif dr_config['for_clustering']['method'] == 'umap':
                print("Applying UMAP for clustering")
                n_components = dr_config['for_clustering']['n_components']
                
                reducer_clustering = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=dr_config['for_clustering'].get('n_neighbors', 15),
                    min_dist=dr_config['for_clustering'].get('min_dist', 0.1),
                    metric='cosine',
                    random_state=self.random_state
                )
                X_clustering = reducer_clustering.fit_transform(X)
            elif dr_config['for_clustering']['method'] == 'pca':
                print("Applying PCA for clustering")
                reducer_clustering = PCA(
                    n_components=dr_config['for_clustering']['n_components'],
                    random_state=self.random_state
                )
                X_clustering = reducer_clustering.fit_transform(X)
            else:
                raise ValueError(f"Unknown DR method: {dr_config['for_clustering']['method']}")
            
            # For visualization (always 2D)
            if dr_config['for_visualization']['method'] == 'umap':
                print("Applying UMAP for 2D visualization")
                reducer_viz = umap.UMAP(
                    n_components=2,
                    n_neighbors=dr_config['for_visualization'].get('n_neighbors', 30),
                    min_dist=dr_config['for_visualization'].get('min_dist', 0.3),
                    metric='cosine',
                    random_state=self.random_state
                )
                X_viz = reducer_viz.fit_transform(X)
            elif dr_config['for_visualization']['method'] == 'tsne':
                print("Applying t-SNE for 2D visualization")
                reducer_viz = TSNE(
                    n_components=2,
                    perplexity=dr_config['for_visualization'].get('perplexity', 30),
                    random_state=self.random_state
                )
                X_viz = reducer_viz.fit_transform(X)
            else:
                # Fallback to first 2 PCs
                reducer_viz = PCA(n_components=2, random_state=self.random_state)
                X_viz = reducer_viz.fit_transform(X)
            
            metadata = {
                'clustering_method': dr_config['for_clustering']['method'],
                'clustering_components': X_clustering.shape[1],
                'viz_method': dr_config['for_visualization']['method'],
                'original_dim': X.shape[1],
                'clustering_dim': X_clustering.shape[1]
            }
            
            return (X_clustering, X_viz), metadata
        
        return self.run_or_load_step('dimensionality_reduction', _reduce, X)
    
    def perform_clustering(self, X_clustering: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform clustering with K-means and HDBSCAN
        
        Reasoning for algorithm choices:
        - K-means: Baseline, works well for spherical clusters, interpretable
        - HDBSCAN: Handles noise, finds clusters of varying densities, no k needed
        """
        def _cluster():
            results = {}
            
            # K-means clustering
            print("Running K-means clustering")
            k = self.config['clustering']['kmeans']['n_clusters']
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            kmeans_labels = kmeans.fit_predict(X_clustering)
            
            # Compute K-means metrics
            kmeans_silhouette = silhouette_score(X_clustering, kmeans_labels)
            kmeans_calinski = calinski_harabasz_score(X_clustering, kmeans_labels)
            kmeans_davies = davies_bouldin_score(X_clustering, kmeans_labels)
            
            results['kmeans'] = {
                'labels': kmeans_labels,
                'centers': kmeans.cluster_centers_,
                'metrics': {
                    'silhouette': kmeans_silhouette,
                    'calinski_harabasz': kmeans_calinski,
                    'davies_bouldin': kmeans_davies,
                    'n_clusters': k
                }
            }
            
            # HDBSCAN clustering
            print("Running HDBSCAN clustering")
            hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config['clustering']['hdbscan']['min_cluster_size'],
                min_samples=self.config['clustering']['hdbscan'].get('min_samples', 5),
                metric='euclidean',
                cluster_selection_epsilon=self.config['clustering']['hdbscan'].get('epsilon', 0.0)
            )
            hdbscan_labels = hdbscan_clusterer.fit_predict(X_clustering)
            
            # Compute HDBSCAN metrics (excluding noise points)
            mask = hdbscan_labels >= 0
            if mask.sum() > 0 and len(np.unique(hdbscan_labels[mask])) > 1:
                hdbscan_silhouette = silhouette_score(X_clustering[mask], hdbscan_labels[mask])
                hdbscan_calinski = calinski_harabasz_score(X_clustering[mask], hdbscan_labels[mask])
                hdbscan_davies = davies_bouldin_score(X_clustering[mask], hdbscan_labels[mask])
            else:
                hdbscan_silhouette = -1
                hdbscan_calinski = 0
                hdbscan_davies = float('inf')
            
            results['hdbscan'] = {
                'labels': hdbscan_labels,
                'probabilities': hdbscan_clusterer.probabilities_,
                'metrics': {
                    'silhouette': hdbscan_silhouette,
                    'calinski_harabasz': hdbscan_calinski,
                    'davies_bouldin': hdbscan_davies,
                    'n_clusters': len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0),
                    'noise_ratio': (hdbscan_labels == -1).mean()
                }
            }
            
            metadata = {
                'kmeans_metrics': results['kmeans']['metrics'],
                'hdbscan_metrics': results['hdbscan']['metrics']
            }
            
            print(f"K-means: {k} clusters, silhouette={kmeans_silhouette:.3f}")
            print(f"HDBSCAN: {results['hdbscan']['metrics']['n_clusters']} clusters, "
                  f"noise={results['hdbscan']['metrics']['noise_ratio']:.2%}")
            
            return results, metadata
        
        return self.run_or_load_step('clustering', _cluster, X_clustering)
    
    def reassign_noise_points(self, X_clustering: np.ndarray, clustering_results: Dict) -> Dict:
        """
        Reassign noise points from HDBSCAN to nearest clusters
        
        Strategy: Assign to nearest cluster center based on distance
        """
        def _reassign():
            results = {}
            
            # Only reassign for HDBSCAN
            hdbscan_labels = clustering_results['hdbscan']['labels'].copy()
            noise_mask = hdbscan_labels == -1
            n_noise = noise_mask.sum()
            
            if n_noise > 0:
                print(f"Reassigning {n_noise} noise points")
                
                # Compute cluster centers
                cluster_ids = np.unique(hdbscan_labels[~noise_mask])
                centers = np.array([
                    X_clustering[hdbscan_labels == cid].mean(axis=0)
                    for cid in cluster_ids
                ])
                
                # Assign noise points to nearest center
                noise_points = X_clustering[noise_mask]
                distances = pairwise_distances(noise_points, centers)
                nearest_clusters = cluster_ids[distances.argmin(axis=1)]
                
                # Update labels
                hdbscan_labels_reassigned = hdbscan_labels.copy()
                hdbscan_labels_reassigned[noise_mask] = nearest_clusters
                
                results['hdbscan_reassigned'] = {
                    'labels': hdbscan_labels_reassigned,
                    'n_reassigned': n_noise,
                    'reassignment_distances': distances.min(axis=1)
                }
            else:
                print("No noise points to reassign")
                results['hdbscan_reassigned'] = {
                    'labels': hdbscan_labels,
                    'n_reassigned': 0,
                    'reassignment_distances': np.array([])
                }
            
            # Keep K-means as is
            results['kmeans'] = clustering_results['kmeans']
            
            metadata = {
                'n_noise_points': int(n_noise),
                'reassignment_performed': n_noise > 0
            }
            
            return results, metadata
        
        return self.run_or_load_step('noise_reassignment', _reassign, X_clustering, clustering_results)
    
    def create_visualizations(self, X_viz: np.ndarray, df: pd.DataFrame, 
                            clustering_results: Dict, save_dir: Path = None):
        """Create comprehensive 2D visualizations with proper bounds"""
        def _visualize():
            if save_dir is None:
                save_dir = self.work_dir / 'visualizations'
            save_dir.mkdir(exist_ok=True)
            
            # Create figure with subplots for each algorithm
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Ensure X_viz is within reasonable bounds for visualization
            x_min, x_max = X_viz[:, 0].min(), X_viz[:, 0].max()
            y_min, y_max = X_viz[:, 1].min(), X_viz[:, 1].max()
            
            # Add padding to prevent cutoff
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05
            
            # Plot K-means
            ax = axes[0, 0]
            scatter = ax.scatter(X_viz[:, 0], X_viz[:, 1], 
                               c=clustering_results['kmeans']['labels'],
                               cmap='tab20', s=20, alpha=0.6)
            ax.set_title('K-means Clustering', fontsize=12, fontweight='bold')
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            
            # Plot HDBSCAN (original)
            ax = axes[0, 1]
            scatter = ax.scatter(X_viz[:, 0], X_viz[:, 1],
                               c=clustering_results['hdbscan']['labels'],
                               cmap='tab20', s=20, alpha=0.6)
            # Highlight noise points
            noise_mask = clustering_results['hdbscan']['labels'] == -1
            if noise_mask.any():
                ax.scatter(X_viz[noise_mask, 0], X_viz[noise_mask, 1],
                         c='red', s=10, alpha=0.8, marker='x', label='Noise')
            ax.set_title('HDBSCAN (Original)', fontsize=12, fontweight='bold')
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            if noise_mask.any():
                ax.legend()
            
            # Plot HDBSCAN (reassigned)
            if 'hdbscan_reassigned' in clustering_results:
                ax = axes[0, 2]
                scatter = ax.scatter(X_viz[:, 0], X_viz[:, 1],
                                   c=clustering_results['hdbscan_reassigned']['labels'],
                                   cmap='tab20', s=20, alpha=0.6)
                ax.set_title('HDBSCAN (Noise Reassigned)', fontsize=12, fontweight='bold')
                ax.set_xlim(x_min - x_padding, x_max + x_padding)
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                plt.colorbar(scatter, ax=ax, label='Cluster')
            
            # Plot cluster sizes comparison
            ax = axes[1, 0]
            kmeans_sizes = pd.Series(clustering_results['kmeans']['labels']).value_counts().sort_index()
            ax.bar(kmeans_sizes.index, kmeans_sizes.values, color='skyblue', edgecolor='navy')
            ax.set_title('K-means Cluster Sizes', fontsize=12, fontweight='bold')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Number of Samples')
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 1]
            hdbscan_labels = clustering_results['hdbscan']['labels']
            hdbscan_sizes = pd.Series(hdbscan_labels[hdbscan_labels >= 0]).value_counts().sort_index()
            bars = ax.bar(hdbscan_sizes.index, hdbscan_sizes.values, color='lightgreen', edgecolor='darkgreen')
            if (hdbscan_labels == -1).any():
                noise_count = (hdbscan_labels == -1).sum()
                ax.bar([-1], [noise_count], color='red', edgecolor='darkred', label='Noise')
                ax.legend()
            ax.set_title('HDBSCAN Cluster Sizes', fontsize=12, fontweight='bold')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Number of Samples')
            ax.grid(True, alpha=0.3)
            
            # Metrics comparison
            ax = axes[1, 2]
            metrics_data = {
                'K-means': [
                    clustering_results['kmeans']['metrics']['silhouette'],
                    clustering_results['kmeans']['metrics']['calinski_harabasz'] / 1000,  # Scale for visualization
                    1 / clustering_results['kmeans']['metrics']['davies_bouldin']  # Invert so higher is better
                ],
                'HDBSCAN': [
                    clustering_results['hdbscan']['metrics']['silhouette'],
                    clustering_results['hdbscan']['metrics']['calinski_harabasz'] / 1000,
                    1 / max(clustering_results['hdbscan']['metrics']['davies_bouldin'], 0.01)
                ]
            }
            
            x = np.arange(3)
            width = 0.35
            metrics_df = pd.DataFrame(metrics_data, index=['Silhouette', 'Calinski-H/1000', '1/Davies-B'])
            metrics_df.plot(kind='bar', ax=ax, width=0.7)
            ax.set_title('Clustering Metrics Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            plt.suptitle(f'{self.pipeline_type.title()} Pipeline - Clustering Results', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Save figure
            fig_path = save_dir / f'{self.pipeline_type}_clustering_visualization.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualizations saved to {fig_path}")
            
            metadata = {
                'figure_path': str(fig_path),
                'n_subplots': 6,
                'x_range': [float(x_min), float(x_max)],
                'y_range': [float(y_min), float(y_max)]
            }
            
            return fig_path, metadata
        
        return self.run_or_load_step('visualization', _visualize, X_viz, df, clustering_results)
    
    def evaluate_clustering_coherence(self, df: pd.DataFrame, clustering_results: Dict):
        """Evaluate clustering coherence using keyword alignment"""
        def _evaluate():
            print("Evaluating clustering coherence with keywords")
            
            coherence_results = {}
            
            # Assuming df has a 'keywords' column
            if 'keywords' not in df.columns:
                print("Warning: No keywords column found for coherence evaluation")
                return coherence_results, {}
            
            for method in ['kmeans', 'hdbscan_reassigned']:
                if method not in clustering_results:
                    continue
                
                labels = clustering_results[method]['labels']
                unique_labels = np.unique(labels[labels >= 0])
                
                cluster_coherence = []
                cluster_keywords = {}
                
                for cluster_id in unique_labels:
                    cluster_mask = labels == cluster_id
                    cluster_papers = df[cluster_mask]
                    
                    # Get all keywords for this cluster
                    all_keywords = []
                    for keywords in cluster_papers['keywords'].dropna():
                        if isinstance(keywords, str):
                            all_keywords.extend(keywords.lower().split(','))
                    
                    if all_keywords:
                        # Count keyword frequencies
                        from collections import Counter
                        keyword_counts = Counter(all_keywords)
                        top_keywords = keyword_counts.most_common(10)
                        
                        # Compute coherence as ratio of shared keywords
                        if len(keyword_counts) > 0:
                            coherence = sum([count for _, count in top_keywords]) / len(all_keywords)
                        else:
                            coherence = 0
                        
                        cluster_coherence.append(coherence)
                        cluster_keywords[int(cluster_id)] = top_keywords
                
                coherence_results[method] = {
                    'mean_coherence': np.mean(cluster_coherence) if cluster_coherence else 0,
                    'std_coherence': np.std(cluster_coherence) if cluster_coherence else 0,
                    'cluster_keywords': cluster_keywords
                }
                
                print(f"{method}: Mean coherence = {coherence_results[method]['mean_coherence']:.3f}")
            
            metadata = coherence_results
            return coherence_results, metadata
        
        return self.run_or_load_step('evaluation', _evaluate, df, clustering_results)
    
    def find_cluster_representatives(self, X_clustering: np.ndarray, clustering_results: Dict, df: pd.DataFrame):
        """Find representative papers (closest to centroid) for manual evaluation"""
        def _find_representatives():
            representatives = {}
            
            for method in ['kmeans', 'hdbscan_reassigned']:
                if method not in clustering_results:
                    continue
                
                labels = clustering_results[method]['labels']
                unique_labels = np.unique(labels[labels >= 0])
                
                method_representatives = {}
                
                for cluster_id in unique_labels:
                    cluster_mask = labels == cluster_id
                    cluster_points = X_clustering[cluster_mask]
                    
                    if len(cluster_points) > 0:
                        # Find centroid
                        centroid = cluster_points.mean(axis=0)
                        
                        # Find closest point to centroid
                        distances = np.linalg.norm(cluster_points - centroid, axis=1)
                        closest_idx = distances.argmin()
                        
                        # Get global index
                        global_indices = np.where(cluster_mask)[0]
                        representative_idx = global_indices[closest_idx]
                        
                        method_representatives[int(cluster_id)] = {
                            'index': int(representative_idx),
                            'distance_to_centroid': float(distances[closest_idx])
                        }
                        
                        # Add paper info if available
                        if 'paper_id' in df.columns:
                            method_representatives[int(cluster_id)]['paper_id'] = df.iloc[representative_idx]['paper_id']
                        if 'title' in df.columns:
                            method_representatives[int(cluster_id)]['title'] = df.iloc[representative_idx]['title']
                
                representatives[method] = method_representatives
            
            # Save for manual evaluation
            eval_file = self.work_dir / f'{self.pipeline_type}_representatives_for_evaluation.json'
            with open(eval_file, 'w') as f:
                json.dump(representatives, f, indent=2)
            
            print(f"Cluster representatives saved to {eval_file}")
            
            return representatives, {'eval_file': str(eval_file)}
        
        return self.run_or_load_step('representatives', _find_representatives, 
                                    X_clustering, clustering_results, df)
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print(f"CLUSTERING REPORT - {self.pipeline_type.upper()} PIPELINE")
        print("="*80)
        
        report_lines = [
            f"# Clustering Analysis Report - {self.pipeline_type.title()} Pipeline",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Pipeline Configuration",
            f"- Type: {self.pipeline_type}",
            f"- Random State: {self.random_state}",
            ""
        ]
        
        # Add metrics from each step
        for step, metrics in self.metrics.items():
            report_lines.append(f"## {step.replace('_', ' ').title()}")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    report_lines.append(f"### {key}")
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            report_lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
                elif isinstance(value, (int, float)):
                    report_lines.append(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
                else:
                    report_lines.append(f"- {key}: {value}")
            report_lines.append("")
        
        # Algorithm comparison
        if 'clustering' in self.metrics:
            report_lines.append("## Algorithm Comparison")
            report_lines.append("")
            report_lines.append("| Metric | K-means | HDBSCAN |")
            report_lines.append("|--------|---------|---------|")
            
            kmeans_m = self.metrics['clustering']['kmeans_metrics']
            hdbscan_m = self.metrics['clustering']['hdbscan_metrics']
            
            report_lines.append(f"| Silhouette | {kmeans_m['silhouette']:.3f} | {hdbscan_m['silhouette']:.3f} |")
            report_lines.append(f"| Calinski-Harabasz | {kmeans_m['calinski_harabasz']:.1f} | {hdbscan_m['calinski_harabasz']:.1f} |")
            report_lines.append(f"| Davies-Bouldin | {kmeans_m['davies_bouldin']:.3f} | {hdbscan_m['davies_bouldin']:.3f} |")
            report_lines.append(f"| N Clusters | {kmeans_m['n_clusters']} | {hdbscan_m['n_clusters']} |")
            report_lines.append("")
        
        # Decision reasoning
        report_lines.extend([
            "## Algorithm Selection Reasoning",
            "",
            "### Why K-means and HDBSCAN?",
            "- **K-means**: Provides baseline performance, works well for spherical clusters,",
            "  interpretable results, and allows direct comparison with expected number of clusters",
            "- **HDBSCAN**: Handles noise and outliers, identifies clusters of varying densities,",
            "  no need to specify k, robust to parameter choices",
            "",
            "### Why UMAP for Dimensionality Reduction?",
            "- Preserves both local and global structure better than t-SNE",
            "- Scales well with high-dimensional data",
            "- Provides meaningful distances between clusters",
            "- Reproducible with fixed random state",
            "",
            "### Preprocessing Strategy",
            f"- Method used: {self.config['preprocessing']['method']}",
            "- Reasoning: Ensures features are on same scale for distance-based clustering",
            ""
        ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.work_dir / f'{self.pipeline_type}_clustering_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"\nReport saved to {report_file}")
        print("\n" + "="*80)
        
        return report_file
    
    def run_complete_pipeline(self):
        """Run the complete pipeline with checkpointing"""
        print(f"\n{'='*80}")
        print(f"STARTING {self.pipeline_type.upper()} PIPELINE")
        print(f"{'='*80}")
        
        try:
            # Step 1: Load data
            (X, df), _ = self.load_data()
            
            # Step 2: Preprocess
            X_preprocessed = self.preprocess_data(X)
            
            # Step 3: Dimensionality reduction
            (X_clustering, X_viz) = self.apply_dimensionality_reduction(X_preprocessed)
            
            # Step 4: Clustering
            clustering_results = self.perform_clustering(X_clustering)
            
            # Step 5: Noise reassignment
            final_results = self.reassign_noise_points(X_clustering, clustering_results)
            
            # Step 6: Visualization
            self.create_visualizations(X_viz, df, final_results)
            
            # Step 7: Evaluation
            coherence_results = self.evaluate_clustering_coherence(df, final_results)
            
            # Step 8: Find representatives
            representatives = self.find_cluster_representatives(X_clustering, final_results, df)
            
            # Step 9: Generate report
            report_path = self.generate_report()
            
            print(f"\n{'='*80}")
            print(f"PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            
            return {
                'clustering_results': final_results,
                'coherence_results': coherence_results,
                'representatives': representatives,
                'report_path': report_path
            }
            
        except Exception as e:
            print(f"\n✗ Pipeline failed at step: {e}")
            print(f"You can restart from the last checkpoint")
            raise


def create_default_config():
    """Create default configuration for both pipelines"""
    return {
        'work_dir': 'thesis_clustering_results',
        'data_path': 'paper_data.csv',
        'paragraph_embedding_prefix': 'paragraph_emb_',
        'method_embedding_prefix': 'method_emb_',
        'random_state': 42,
        
        'preprocessing': {
            'method': 'standard'  # Options: 'none', 'standard', 'l2_normalize'
        },
        
        'dimensionality_reduction': {
            'for_clustering': {
                'method': 'umap',  # Options: 'none', 'umap', 'pca'
                'n_components': 50,
                'n_neighbors': 15,
                'min_dist': 0.1
            },
            'for_visualization': {
                'method': 'umap',  # Options: 'umap', 'tsne', 'pca'
                'n_neighbors': 30,
                'min_dist': 0.3,
                'perplexity': 30  # For t-SNE
            }
        },
        
        'clustering': {
            'kmeans': {
                'n_clusters': 8
            },
            'hdbscan': {
                'min_cluster_size': 10,
                'min_samples': 5,
                'epsilon': 0.0
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Thesis Clustering Framework")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--pipeline', type=str, required=True, 
                       choices=['paragraph', 'methodological', 'both'],
                       help='Pipeline type to run')
    parser.add_argument('--data', type=str, help='Path to input data')
    parser.add_argument('--work-dir', type=str, help='Working directory')
    parser.add_argument('--clear-checkpoints', action='store_true',
                       help='Clear existing checkpoints and start fresh')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.data:
        config['data_path'] = args.data
    if args.work_dir:
        config['work_dir'] = args.work_dir
    
    # Run pipelines
    if args.pipeline == 'both':
        pipelines = ['paragraph', 'methodological']
    else:
        pipelines = [args.pipeline]
    
    results = {}
    for pipeline_type in pipelines:
        print(f"\n{'='*80}")
        print(f"Running {pipeline_type} pipeline")
        print(f"{'='*80}")
        
        pipeline = ThesisClusteringPipeline(config, pipeline_type)
        
        if args.clear_checkpoints:
            print("Clearing existing checkpoints...")
            for step in ['data_loading', 'preprocessing', 'dimensionality_reduction',
                        'clustering', 'noise_reassignment', 'visualization', 
                        'evaluation', 'representatives']:
                pipeline.checkpoint_manager.clear_downstream(step)
        
        results[pipeline_type] = pipeline.run_complete_pipeline()
    
    # Compare pipelines if both were run
    if len(results) == 2:
        print(f"\n{'='*80}")
        print("PIPELINE COMPARISON")
        print(f"{'='*80}")
        
        # Add comparison logic here
        comparison_file = Path(config['work_dir']) / 'pipeline_comparison.md'
        with open(comparison_file, 'w') as f:
            f.write("# Pipeline Comparison Report\n\n")
            f.write("## Summary\n")
            # Add comparison metrics
        
        print(f"Comparison report saved to {comparison_file}")
    
    print("\n✓ All pipelines completed successfully!")


if __name__ == "__main__":
    main()
