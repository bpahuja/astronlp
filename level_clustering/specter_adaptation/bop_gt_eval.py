#!/usr/bin/env python3
"""
Complete standalone script for comprehensive evaluation of paper clustering results
including ground truth comparison using manual classifications.

Input: 
  - paper_clusters.csv (output from paper clustering script)
  - list.csv (manual classifications with obs. type and arXiv columns)
Output: Comprehensive evaluation metrics including ground truth comparison
"""

import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score, 
                            adjusted_rand_score, normalized_mutual_info_score, 
                            homogeneity_score, completeness_score, v_measure_score,
                            fowlkes_mallows_score, confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from collections import Counter
import multiprocessing

warnings.filterwarnings('ignore')
plt.style.use('default')


class ComprehensiveClusteringEvaluator:
    """Complete evaluation including both basic clustering metrics and ground truth comparison"""
    
    def __init__(self, paper_df: pd.DataFrame, ground_truth_df: Optional[pd.DataFrame] = None, 
                 work_dir: Path = None, n_jobs: Optional[int] = None):
        self.work_dir = Path(work_dir) if work_dir else Path('.')
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up parallel processing
        if n_jobs is None or n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif n_jobs > 0:
            self.n_jobs = n_jobs
        else:
            self.n_jobs = 1
        
        # Store original data
        self.original_paper_df = paper_df
        self.ground_truth_df = ground_truth_df
        
        # Prepare data (merge with ground truth if available)
        if ground_truth_df is not None:
            self.paper_df = self._prepare_ground_truth_data()
            self.has_ground_truth = True
        else:
            self.paper_df = paper_df
            self.has_ground_truth = False
        
        # Identify vector columns and cluster labels
        self.vector_cols = [col for col in self.paper_df.columns if col.startswith('cluster_')]
        if 'paper_cluster' not in self.paper_df.columns:
            raise ValueError("No 'paper_cluster' column found in the data")
        
        self.cluster_labels = self.paper_df['paper_cluster'].values
        
        if self.has_ground_truth:
            self.ground_truth_labels = self.paper_df['obs_type'].values
        
        if not self.vector_cols:
            raise ValueError("No cluster vector columns found (columns starting with 'cluster_')")
        
        print(f"Initialized evaluator with:")
        print(f"  - {len(self.paper_df)} papers")
        print(f"  - {len(self.vector_cols)} vector dimensions")
        print(f"  - {len(np.unique(self.cluster_labels))} predicted cluster labels")
        if self.has_ground_truth:
            print(f"  - {len(np.unique(self.ground_truth_labels))} ground truth categories")
        print(f"  - Using {self.n_jobs} parallel jobs")
    
    def _prepare_ground_truth_data(self) -> pd.DataFrame:
        """Merge clustering results with ground truth data"""
        print("Preparing ground truth data...")
        
        # Clean ground truth data
        gt_df = self.ground_truth_df.copy()
        
        # Remove rows with empty arXiv values
        gt_df = gt_df[gt_df['arXiv'].notna() & (gt_df['arXiv'] != '')]
        print(f"Ground truth: {len(gt_df)} papers with valid arXiv IDs")
        
        # Clean arXiv IDs
        gt_df['arXiv_clean'] = gt_df['arXiv'].astype(str).str.replace('arXiv:', '', regex=False).str.strip()
        
        # Clean paper IDs in clustering results
        paper_df_clean = self.original_paper_df.copy()
        paper_df_clean['paper_id_clean'] = paper_df_clean['paper_id'].astype(str).str.strip()
        
        # Try multiple matching strategies
        merged = None
        strategies = [
            # Strategy 1: Direct match
            ('paper_id_clean', 'arXiv_clean'),
            # Strategy 2: Try with arXiv prefix in paper_id
            ('paper_id_clean', 'arXiv'),
            # Strategy 3: Add arXiv prefix to paper_id
            (None, None)  # Special case handled below
        ]
        
        for i, (paper_col, gt_col) in enumerate(strategies):
            if i == 2:  # Strategy 3: Add prefix
                paper_df_clean['paper_id_prefixed'] = 'arXiv:' + paper_df_clean['paper_id_clean']
                merged = pd.merge(paper_df_clean, gt_df, 
                                 left_on='paper_id_prefixed', right_on='arXiv', 
                                 how='inner')
            else:
                merged = pd.merge(paper_df_clean, gt_df, 
                                 left_on=paper_col, right_on=gt_col, 
                                 how='inner')
            
            if len(merged) > 0:
                print(f"Successfully matched using strategy {i+1}: {len(merged)} papers")
                break
        
        if merged is None or len(merged) == 0:
            # Try partial matching as last resort
            gt_df['arXiv_short'] = gt_df['arXiv_clean'].str.split('/').str[-1]
            paper_df_clean['paper_id_short'] = paper_df_clean['paper_id_clean'].str.split('/').str[-1]
            merged = pd.merge(paper_df_clean, gt_df, 
                             left_on='paper_id_short', right_on='arXiv_short', 
                             how='inner')
            
            if len(merged) > 0:
                print(f"Successfully matched using partial matching: {len(merged)} papers")
        
        if merged is None or len(merged) == 0:
            raise ValueError("No matching papers found between clustering results and ground truth")
        
        # Rename ground truth column for consistency
        merged['obs_type'] = merged['obs. type']
        
        print(f"Final dataset: {len(merged)} papers with both clustering and ground truth")
        print(f"Ground truth categories: {sorted(merged['obs_type'].unique())}")
        
        return merged
    
    # ===============================================================================
    # BASIC CLUSTERING EVALUATION METHODS
    # ===============================================================================
    
    def validate_data(self) -> Dict:
        """Validate input data and provide basic statistics"""
        print("Validating clustering data...")
        
        validation = {}
        
        # Basic data validation
        validation['total_papers'] = len(self.paper_df)
        validation['vector_dimensions'] = len(self.vector_cols)
        
        # Cluster label analysis
        unique_labels = np.unique(self.cluster_labels)
        validation['unique_clusters'] = len(unique_labels)
        validation['cluster_labels'] = unique_labels.tolist()
        
        # Count different types of assignments
        n_clustered = np.sum(self.cluster_labels >= 0)  # Regular clusters (>=0)
        n_noise = np.sum(self.cluster_labels == -1)     # Noise
        n_excluded = np.sum(self.cluster_labels == -2)  # Excluded from clustering
        
        validation['clustered_papers'] = n_clustered
        validation['noise_papers'] = n_noise
        validation['excluded_papers'] = n_excluded
        validation['clustering_coverage'] = n_clustered / len(self.paper_df)
        
        # Vector validation
        X = self.paper_df[self.vector_cols].values
        validation['zero_vectors'] = np.sum(np.all(X == 0, axis=1))
        validation['vector_range'] = {'min': float(X.min()), 'max': float(X.max())}
        validation['vector_sparsity'] = np.mean(X == 0)
        
        print(f"Data validation results:")
        print(f"  - Total papers: {validation['total_papers']:,}")
        print(f"  - Clustered papers: {validation['clustered_papers']:,}")
        print(f"  - Noise papers: {validation['noise_papers']:,}")
        print(f"  - Excluded papers: {validation['excluded_papers']:,}")
        print(f"  - Clustering coverage: {validation['clustering_coverage']:.3f}")
        
        return validation
    
    def evaluate_clustering_quality(self) -> Dict:
        """Comprehensive clustering quality assessment"""
        print("Evaluating clustering quality...")
        
        # Get vector data and standardize
        X = self.paper_df[self.vector_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Only evaluate papers that were actually clustered
        clustered_mask = self.cluster_labels >= -1  # Include noise (-1) but exclude excluded (-2)
        X_clustered = X_scaled[clustered_mask]
        labels_clustered = self.cluster_labels[clustered_mask]
        
        metrics = {}
        
        # Basic cluster statistics
        unique_labels = np.unique(labels_clustered)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels_clustered == -1)
        n_clustered_points = len(labels_clustered)
        
        # Cluster sizes
        cluster_sizes = []
        for label in unique_labels:
            if label >= 0:  # Regular clusters only
                size = np.sum(labels_clustered == label)
                cluster_sizes.append(size)
        
        metrics['basic_stats'] = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_clustered_points': n_clustered_points,
            'noise_ratio': n_noise / n_clustered_points if n_clustered_points > 0 else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': np.min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': np.max(cluster_sizes) if cluster_sizes else 0,
            'cluster_sizes': cluster_sizes
        }
        
        # Internal clustering metrics
        if n_clusters > 1 and len(np.unique(labels_clustered[labels_clustered >= 0])) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(X_clustered, labels_clustered)
                
                # CH and DB only for non-noise points
                non_noise_mask = labels_clustered >= 0
                if np.sum(non_noise_mask) > 1 and len(np.unique(labels_clustered[non_noise_mask])) > 1:
                    X_non_noise = X_clustered[non_noise_mask]
                    labels_non_noise = labels_clustered[non_noise_mask]
                    
                    metrics['calinski_harabasz'] = calinski_harabasz_score(X_non_noise, labels_non_noise)
                    metrics['davies_bouldin'] = davies_bouldin_score(X_non_noise, labels_non_noise)
                else:
                    metrics['calinski_harabasz'] = None
                    metrics['davies_bouldin'] = None
                    
            except Exception as e:
                print(f"Warning: Could not calculate some quality metrics: {e}")
                metrics['silhouette_score'] = None
                metrics['calinski_harabasz'] = None
                metrics['davies_bouldin'] = None
        else:
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz'] = None
            metrics['davies_bouldin'] = None
        
        return metrics
    
    def evaluate_cluster_coherence(self) -> Dict:
        """Evaluate methodological coherence within clusters"""
        print("Evaluating cluster coherence...")
        
        unique_clusters = np.unique(self.cluster_labels)
        valid_clusters = [c for c in unique_clusters if c >= 0]
        
        if not valid_clusters:
            return {'error': 'no_valid_clusters'}
        
        cluster_coherences = {}
        
        for cluster_id in valid_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_papers = self.paper_df[cluster_mask]
            
            if len(cluster_papers) < 2:
                continue
            
            # Vector coherence
            cluster_vectors = cluster_papers[self.vector_cols].values
            
            # Handle zero vectors
            norms = np.linalg.norm(cluster_vectors, axis=1)
            valid_vectors_mask = norms > 1e-10
            
            if np.sum(valid_vectors_mask) < 2:
                continue
                
            valid_vectors = cluster_vectors[valid_vectors_mask]
            
            # Calculate pairwise cosine similarities
            similarities = cosine_similarity(valid_vectors)
            
            # Get upper triangular part (excluding diagonal)
            upper_tri_mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
            pairwise_sims = similarities[upper_tri_mask]
            
            if len(pairwise_sims) > 0:
                cluster_coherences[cluster_id] = {
                    'size': len(cluster_papers),
                    'valid_vectors': np.sum(valid_vectors_mask),
                    'mean_pairwise_similarity': float(np.mean(pairwise_sims)),
                    'std_pairwise_similarity': float(np.std(pairwise_sims)),
                    'min_similarity': float(np.min(pairwise_sims)),
                    'max_similarity': float(np.max(pairwise_sims))
                }
        
        # Overall coherence statistics
        if cluster_coherences:
            all_mean_sims = [metrics['mean_pairwise_similarity'] for metrics in cluster_coherences.values()]
            
            coherence_metrics = {
                'cluster_coherences': cluster_coherences,
                'overall_mean_coherence': np.mean(all_mean_sims),
                'coherence_std': np.std(all_mean_sims),
                'min_cluster_coherence': np.min(all_mean_sims),
                'max_cluster_coherence': np.max(all_mean_sims),
                'n_evaluated_clusters': len(cluster_coherences)
            }
        else:
            coherence_metrics = {'error': 'no_evaluable_clusters'}
        
        return coherence_metrics
    
    def evaluate_cluster_separation(self) -> Dict:
        """Evaluate separation between different clusters"""
        print("Evaluating cluster separation...")
        
        X = self.paper_df[self.vector_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        unique_clusters = np.unique(self.cluster_labels)
        valid_clusters = [c for c in unique_clusters if c >= 0]
        
        if len(valid_clusters) < 2:
            return {'error': 'insufficient_clusters', 'n_clusters': len(valid_clusters)}
        
        # Calculate centroids for each cluster
        cluster_centroids = {}
        cluster_sizes = {}
        
        for cluster_id in valid_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_vectors = X_scaled[cluster_mask]
            
            if len(cluster_vectors) > 0:
                cluster_centroids[cluster_id] = np.mean(cluster_vectors, axis=0)
                cluster_sizes[cluster_id] = len(cluster_vectors)
        
        if len(cluster_centroids) < 2:
            return {'error': 'insufficient_valid_centroids'}
        
        # Calculate pairwise distances between centroids
        centroid_matrix = np.array(list(cluster_centroids.values()))
        inter_cluster_distances = pdist(centroid_matrix, metric='euclidean')
        
        # Calculate average intra-cluster distances
        intra_cluster_distances = []
        
        for cluster_id in cluster_centroids.keys():
            cluster_mask = self.cluster_labels == cluster_id
            cluster_vectors = X_scaled[cluster_mask]
            
            if len(cluster_vectors) > 1:
                intra_distances = pdist(cluster_vectors, metric='euclidean')
                intra_cluster_distances.extend(intra_distances)
        
        # Calculate separation metrics
        mean_inter = np.mean(inter_cluster_distances)
        mean_intra = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
        
        separation_metrics = {
            'mean_inter_cluster_distance': float(mean_inter),
            'std_inter_cluster_distance': float(np.std(inter_cluster_distances)),
            'min_inter_cluster_distance': float(np.min(inter_cluster_distances)),
            'max_inter_cluster_distance': float(np.max(inter_cluster_distances)),
            'mean_intra_cluster_distance': float(mean_intra),
            'separation_ratio': float(mean_inter / mean_intra) if mean_intra > 0 else float('inf'),
            'n_clusters_evaluated': len(cluster_centroids)
        }
        
        return separation_metrics
    
    def evaluate_cluster_balance(self) -> Dict:
        """Evaluate cluster size balance and distribution"""
        print("Evaluating cluster balance...")
        
        unique_clusters = np.unique(self.cluster_labels)
        valid_clusters = [c for c in unique_clusters if c >= 0]
        
        if not valid_clusters:
            return {'error': 'no_valid_clusters'}
        
        # Get cluster sizes
        cluster_sizes = []
        cluster_size_dict = {}
        
        for cluster_id in valid_clusters:
            size = np.sum(self.cluster_labels == cluster_id)
            cluster_sizes.append(size)
            cluster_size_dict[cluster_id] = size
        
        cluster_sizes = np.array(cluster_sizes)
        
        # Calculate balance metrics
        total_clustered = np.sum(cluster_sizes)
        
        balance_metrics = {
            'n_clusters': len(cluster_sizes),
            'total_clustered_points': int(total_clustered),
            'mean_cluster_size': float(np.mean(cluster_sizes)),
            'median_cluster_size': float(np.median(cluster_sizes)),
            'std_cluster_size': float(np.std(cluster_sizes)),
            'min_cluster_size': int(np.min(cluster_sizes)),
            'max_cluster_size': int(np.max(cluster_sizes)),
            'coefficient_of_variation': float(np.std(cluster_sizes) / np.mean(cluster_sizes)) if np.mean(cluster_sizes) > 0 else 0,
            'size_entropy': float(entropy(cluster_sizes / total_clustered)) if total_clustered > 0 else 0,
            'largest_cluster_ratio': float(np.max(cluster_sizes) / total_clustered) if total_clustered > 0 else 0,
            'cluster_size_distribution': cluster_size_dict
        }
        
        return {
            'cluster_sizes': cluster_sizes.tolist(),
            'balance_metrics': balance_metrics
        }
    
    def evaluate_stability(self, n_trials: int = 5, noise_level: float = 0.01) -> Dict:
        """Evaluate clustering stability through perturbation"""
        print(f"Evaluating clustering stability ({n_trials} trials)...")
        
        X = self.paper_df[self.vector_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Only evaluate papers that were actually clustered
        clustered_mask = self.cluster_labels >= -1
        X_clustered = X_scaled[clustered_mask]
        original_labels = self.cluster_labels[clustered_mask]
        
        if len(X_clustered) < 10:
            return {'error': 'insufficient_data'}
        
        stability_scores = []
        
        # Estimate reasonable clustering parameters
        unique_original = np.unique(original_labels[original_labels >= 0])
        n_original_clusters = len(unique_original)
        min_cluster_size = max(5, len(X_clustered) // (n_original_clusters * 3) if n_original_clusters > 0 else 10)
        
        for trial in range(n_trials):
            try:
                # Add small random noise
                noise = np.random.normal(0, noise_level, X_clustered.shape)
                X_perturbed = X_clustered + noise
                
                # Re-cluster
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(3, min_cluster_size // 3),
                    n_jobs=1
                )
                trial_labels = clusterer.fit_predict(X_perturbed)
                
                # Calculate ARI
                ari = adjusted_rand_score(original_labels, trial_labels)
                stability_scores.append(ari)
                
            except Exception as e:
                print(f"Warning: Trial {trial} failed: {e}")
                continue
        
        if not stability_scores:
            return {'error': 'all_trials_failed'}
        
        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'min_stability': np.min(stability_scores),
            'max_stability': np.max(stability_scores),
            'n_successful_trials': len(stability_scores),
            'stability_scores': stability_scores
        }
    
    # ===============================================================================
    # GROUND TRUTH EVALUATION METHODS
    # ===============================================================================
    
    def evaluate_ground_truth_alignment(self) -> Dict:
        """Comprehensive ground truth comparison"""
        if not self.has_ground_truth:
            return {'error': 'no_ground_truth_available'}
        
        print("Evaluating alignment with ground truth...")
        
        # Get valid clustered papers only
        valid_mask = self.cluster_labels >= -1  # Include noise (-1)
        clustered_mask = self.cluster_labels >= 0  # Only proper clusters
        
        predicted_labels = self.cluster_labels[valid_mask]
        true_labels = self.ground_truth_labels[valid_mask]
        
        predicted_clustered = self.cluster_labels[clustered_mask]
        true_clustered = self.ground_truth_labels[clustered_mask]
        
        alignment_metrics = {}
        
        # Basic alignment statistics
        alignment_metrics['basic_stats'] = {
            'total_papers_with_gt': len(self.paper_df),
            'papers_with_valid_clusters': np.sum(valid_mask),
            'papers_with_proper_clusters': np.sum(clustered_mask),
            'coverage_ratio': np.sum(clustered_mask) / len(self.paper_df)
        }
        
        # Standard clustering comparison metrics
        try:
            alignment_metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, predicted_labels)
            alignment_metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, predicted_labels)
            alignment_metrics['fowlkes_mallows'] = fowlkes_mallows_score(true_labels, predicted_labels)
            
            if len(np.unique(predicted_clustered)) > 1 and len(np.unique(true_clustered)) > 1:
                alignment_metrics['homogeneity'] = homogeneity_score(true_clustered, predicted_clustered)
                alignment_metrics['completeness'] = completeness_score(true_clustered, predicted_clustered)
                alignment_metrics['v_measure'] = v_measure_score(true_clustered, predicted_clustered)
            else:
                alignment_metrics['homogeneity'] = None
                alignment_metrics['completeness'] = None
                alignment_metrics['v_measure'] = None
                
        except Exception as e:
            print(f"Warning: Could not calculate some alignment metrics: {e}")
            alignment_metrics.update({
                'adjusted_rand_index': None,
                'normalized_mutual_info': None,
                'fowlkes_mallows': None,
                'homogeneity': None,
                'completeness': None,
                'v_measure': None
            })
        
        return alignment_metrics
    
    def analyze_confusion_matrix(self) -> Dict:
        """Detailed confusion matrix analysis"""
        if not self.has_ground_truth:
            return {'error': 'no_ground_truth_available'}
        
        print("Analyzing confusion matrix...")
        
        # Only use properly clustered papers
        clustered_mask = self.cluster_labels >= 0
        predicted_clustered = self.cluster_labels[clustered_mask]
        true_clustered = self.ground_truth_labels[clustered_mask]
        
        if len(predicted_clustered) == 0:
            return {'error': 'no_clustered_papers'}
        
        # Convert labels to ensure compatibility
        # Keep original for interpretation, but create compatible versions for sklearn
        true_categories = sorted(np.unique(true_clustered))
        predicted_clusters = sorted(np.unique(predicted_clustered))
        
        # Create label encoders
        true_label_encoder = LabelEncoder()
        pred_label_encoder = LabelEncoder()
        
        # Fit and transform labels to integers
        true_encoded = true_label_encoder.fit_transform(true_clustered)
        pred_encoded = pred_label_encoder.fit_transform(predicted_clustered)
        
        # Create confusion matrix with encoded labels
        conf_matrix = confusion_matrix(true_encoded, pred_encoded)
        
        # Calculate per-category metrics using original labels
        category_analysis = {}
        
        for category in true_categories:
            category_mask = true_clustered == category
            category_predicted = predicted_clustered[category_mask]
            
            # Find dominant cluster
            cluster_counts = Counter(category_predicted)
            if cluster_counts:
                dominant_cluster = cluster_counts.most_common(1)[0][0]
                dominant_count = cluster_counts.most_common(1)[0][1]
                
                # Calculate purity and coverage
                purity = dominant_count / len(category_predicted)
                total_in_dominant = np.sum(predicted_clustered == dominant_cluster)
                coverage = dominant_count / total_in_dominant if total_in_dominant > 0 else 0
                
                category_analysis[category] = {
                    'total_papers': len(category_predicted),
                    'dominant_cluster': int(dominant_cluster),
                    'dominant_count': int(dominant_count),
                    'purity': float(purity),
                    'coverage': float(coverage),
                    'num_clusters_spanned': len(cluster_counts),
                    'cluster_distribution': {int(k): int(v) for k, v in cluster_counts.items()}
                }
        
        # Calculate per-cluster reverse analysis
        cluster_analysis = {}
        
        for cluster_id in predicted_clusters:
            cluster_mask = predicted_clustered == cluster_id
            cluster_true = true_clustered[cluster_mask]
            
            if len(cluster_true) > 0:
                # Find dominant category
                category_counts = Counter(cluster_true)
                dominant_category = category_counts.most_common(1)[0][0]
                dominant_count = category_counts.most_common(1)[0][1]
                
                # Calculate cluster purity
                cluster_purity = dominant_count / len(cluster_true)
                
                cluster_analysis[int(cluster_id)] = {
                    'total_papers': len(cluster_true),
                    'dominant_category': str(dominant_category),
                    'dominant_count': int(dominant_count),
                    'purity': float(cluster_purity),
                    'num_categories_mixed': len(category_counts),
                    'category_distribution': {str(k): int(v) for k, v in category_counts.items()}
                }
        
        return {
            'confusion_matrix': conf_matrix.tolist(),
            'true_categories': [str(cat) for cat in true_categories],
            'predicted_clusters': [int(x) for x in predicted_clusters],
            'category_analysis': category_analysis,
            'cluster_analysis': cluster_analysis,
            'label_mappings': {
                'true_categories_to_indices': {cat: int(idx) for idx, cat in enumerate(true_categories)},
                'predicted_clusters_to_indices': {int(cluster): int(idx) for idx, cluster in enumerate(predicted_clusters)}
            }
        }
    
    # ===============================================================================
    # VISUALIZATION METHODS
    # ===============================================================================
    
    def create_basic_visualizations(self, basic_results: Dict) -> List[plt.Figure]:
        """Create basic clustering evaluation visualizations"""
        print("Creating basic evaluation visualizations...")
        
        figures = []
        
        # Figure 1: Quality and Balance Overview
        fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Cluster sizes
        if 'balance' in basic_results and 'cluster_sizes' in basic_results['balance']:
            cluster_sizes = basic_results['balance']['cluster_sizes']
            if cluster_sizes:
                axes[0, 0].hist(cluster_sizes, bins=min(20, len(cluster_sizes)), alpha=0.7, 
                               color='skyblue', edgecolor='black')
                axes[0, 0].set_xlabel('Cluster Size')
                axes[0, 0].set_ylabel('Number of Clusters')
                axes[0, 0].set_title('Cluster Size Distribution')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add statistics
                balance_metrics = basic_results['balance']['balance_metrics']
                stats_text = f"Mean: {balance_metrics['mean_cluster_size']:.1f}\n"
                stats_text += f"CV: {balance_metrics['coefficient_of_variation']:.3f}"
                axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 2: Quality metrics
        quality_metrics = basic_results.get('quality', {})
        metric_names = []
        metric_values = []
        
        if quality_metrics.get('silhouette_score') is not None:
            metric_names.append('Silhouette')
            metric_values.append(quality_metrics['silhouette_score'])
        
        if quality_metrics.get('calinski_harabasz') is not None:
            metric_names.append('CH/1000')
            metric_values.append(quality_metrics['calinski_harabasz'] / 1000)
        
        if quality_metrics.get('davies_bouldin') is not None:
            metric_names.append('1/DB')
            metric_values.append(1 / quality_metrics['davies_bouldin'])
        
        if metric_names:
            bars = axes[0, 1].bar(metric_names, metric_values, alpha=0.7, 
                                 color=['lightgreen', 'orange', 'lightcoral'][:len(metric_names)], 
                                 edgecolor='black')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Clustering Quality Metrics')
            axes[0, 1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Coherence distribution
        if ('coherence' in basic_results and 
            'cluster_coherences' in basic_results['coherence']):
            coherences = [metrics['mean_pairwise_similarity'] 
                         for metrics in basic_results['coherence']['cluster_coherences'].values()]
            if coherences:
                axes[1, 0].hist(coherences, bins=min(15, len(coherences)), alpha=0.7, 
                               color='lightgreen', edgecolor='black')
                axes[1, 0].set_xlabel('Mean Pairwise Similarity')
                axes[1, 0].set_ylabel('Number of Clusters')
                axes[1, 0].set_title('Cluster Coherence Distribution')
                axes[1, 0].grid(True, alpha=0.3)
                
                mean_coherence = basic_results['coherence']['overall_mean_coherence']
                axes[1, 0].axvline(mean_coherence, color='red', linestyle='--', 
                                  label=f'Mean: {mean_coherence:.3f}')
                axes[1, 0].legend()
        
        # Plot 4: Separation analysis
        if ('separation' in basic_results and 
            'separation_ratio' in basic_results['separation'] and
            basic_results['separation']['separation_ratio'] != float('inf')):
            sep_metrics = basic_results['separation']
            
            labels = ['Intra-cluster\nDistance', 'Inter-cluster\nDistance']
            values = [sep_metrics['mean_intra_cluster_distance'], 
                     sep_metrics['mean_inter_cluster_distance']]
            
            bars = axes[1, 1].bar(labels, values, alpha=0.7, 
                                 color=['lightcoral', 'lightblue'], edgecolor='black')
            axes[1, 1].set_ylabel('Distance')
            axes[1, 1].set_title('Cluster Separation Analysis')
            axes[1, 1].grid(True, alpha=0.3)
            
            sep_ratio = sep_metrics['separation_ratio']
            axes[1, 1].text(0.5, 0.95, f'Separation Ratio: {sep_ratio:.2f}', 
                           transform=axes[1, 1].transAxes, ha='center', va='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        figures.append(fig1)
        
        return figures
    
    def create_ground_truth_visualizations(self, gt_results: Dict) -> List[plt.Figure]:
        """Create ground truth comparison visualizations"""
        if not self.has_ground_truth:
            return []
        
        print("Creating ground truth visualizations...")
        
        figures = []
        
        # Figure: Ground Truth Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Alignment metrics
        if 'alignment' in gt_results:
            alignment = gt_results['alignment']
            metric_names = []
            metric_values = []
            
            metrics_to_plot = [
                ('ARI', 'adjusted_rand_index'),
                ('NMI', 'normalized_mutual_info'),
                ('Homogeneity', 'homogeneity'),
                ('Completeness', 'completeness'),
                ('V-measure', 'v_measure')
            ]
            
            for name, key in metrics_to_plot:
                if key in alignment and alignment[key] is not None:
                    metric_names.append(name)
                    metric_values.append(alignment[key])
            
            if metric_names:
                bars = axes[0, 0].bar(metric_names, metric_values, alpha=0.7, 
                                     color='lightgreen', edgecolor='black')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_title('Ground Truth Alignment Metrics')
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].grid(True, alpha=0.3)
                
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Category fragmentation
        if 'confusion_matrix_analysis' in gt_results:
            conf_analysis = gt_results['confusion_matrix_analysis']
            if 'category_analysis' in conf_analysis:
                categories = list(conf_analysis['category_analysis'].keys())
                fragmentations = [conf_analysis['category_analysis'][cat]['num_clusters_spanned'] 
                                for cat in categories]
                
                # Sort by fragmentation for better visualization
                sorted_data = sorted(zip(categories, fragmentations), key=lambda x: x[1], reverse=True)
                
                # Take top 15 for readability
                if len(sorted_data) > 15:
                    sorted_data = sorted_data[:15]
                    title_suffix = " (Top 15 Most Fragmented)"
                else:
                    title_suffix = ""
                
                if sorted_data:
                    categories, fragmentations = zip(*sorted_data)
                    
                    bars = axes[0, 1].bar(range(len(categories)), fragmentations, alpha=0.7, 
                                         color='orange', edgecolor='black')
                    axes[0, 1].set_xticks(range(len(categories)))
                    axes[0, 1].set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
                    axes[0, 1].set_ylabel('Number of Clusters')
                    axes[0, 1].set_title(f'Category Fragmentation{title_suffix}')
                    axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cluster purity
        if 'confusion_matrix_analysis' in gt_results:
            conf_analysis = gt_results['confusion_matrix_analysis']
            if 'cluster_analysis' in conf_analysis:
                cluster_analysis = conf_analysis['cluster_analysis']
                
                # Filter clusters by size and sort by purity
                cluster_data = [(cid, cluster_analysis[cid]['purity'], cluster_analysis[cid]['total_papers']) 
                               for cid in cluster_analysis.keys()
                               if cluster_analysis[cid]['total_papers'] >= 3]
                
                if cluster_data:
                    cluster_data.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top 20
                    if len(cluster_data) > 20:
                        cluster_data = cluster_data[:20]
                        title_suffix = " (Top 20 Purest)"
                    else:
                        title_suffix = ""
                    
                    cluster_ids, purities, sizes = zip(*cluster_data)
                    
                    bars = axes[1, 0].bar(range(len(cluster_ids)), purities, alpha=0.7, 
                                         color='lightcoral', edgecolor='black')
                    axes[1, 0].set_xticks(range(len(cluster_ids)))
                    axes[1, 0].set_xticklabels([f'C{cid}' for cid in cluster_ids], 
                                              rotation=45, fontsize=8)
                    axes[1, 0].set_ylabel('Purity')
                    axes[1, 0].set_title(f'Cluster Purity{title_suffix}')
                    axes[1, 0].set_ylim(0, 1)
                    axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        stats_text = "ALIGNMENT SUMMARY\n"
        stats_text += "=" * 20 + "\n"
        
        if 'alignment' in gt_results:
            alignment = gt_results['alignment']
            if alignment.get('adjusted_rand_index') is not None:
                stats_text += f"ARI: {alignment['adjusted_rand_index']:.3f}\n"
            if alignment.get('homogeneity') is not None:
                stats_text += f"Homogeneity: {alignment['homogeneity']:.3f}\n"
            if alignment.get('completeness') is not None:
                stats_text += f"Completeness: {alignment['completeness']:.3f}\n"
            
            basic_stats = alignment.get('basic_stats', {})
            if 'coverage_ratio' in basic_stats:
                stats_text += f"Coverage: {basic_stats['coverage_ratio']:.3f}\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Alignment Summary')
        
        plt.tight_layout()
        figures.append(fig)
        
        return figures
    
    # ===============================================================================
    # MAIN EVALUATION METHODS
    # ===============================================================================
    
    def run_complete_evaluation(self) -> Dict:
        """Run comprehensive evaluation including both basic metrics and ground truth comparison"""
        print("\n" + "="*70)
        print("COMPREHENSIVE CLUSTERING EVALUATION")
        print("="*70)
        
        all_results = {}
        
        # 1. Basic clustering evaluation
        print("\n" + "-"*50)
        print("BASIC CLUSTERING EVALUATION")
        print("-"*50)
        
        basic_results = {}
        basic_results['validation'] = self.validate_data()
        basic_results['quality'] = self.evaluate_clustering_quality()
        basic_results['coherence'] = self.evaluate_cluster_coherence()
        basic_results['separation'] = self.evaluate_cluster_separation()
        basic_results['balance'] = self.evaluate_cluster_balance()
        basic_results['stability'] = self.evaluate_stability()
        
        all_results['basic_evaluation'] = basic_results
        
        # 2. Ground truth evaluation (if available)
        if self.has_ground_truth:
            print("\n" + "-"*50)
            print("GROUND TRUTH EVALUATION")
            print("-"*50)
            
            gt_results = {}
            gt_results['alignment'] = self.evaluate_ground_truth_alignment()
            gt_results['confusion_matrix_analysis'] = self.analyze_confusion_matrix()
            
            all_results['ground_truth_evaluation'] = gt_results
        
        # 3. Generate interpretations
        all_results['interpretation'] = self.interpret_results(all_results)
        
        return all_results
    
    def interpret_results(self, all_results: Dict) -> Dict:
        """Provide comprehensive interpretation of all results"""
        interpretation = {
            'overall_assessment': 'unknown',
            'clustering_quality': 'unknown',
            'ground_truth_alignment': 'not_available',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Analyze basic clustering quality
        basic_results = all_results.get('basic_evaluation', {})
        quality = basic_results.get('quality', {})
        
        silhouette = quality.get('silhouette_score')
        if silhouette is not None:
            if silhouette > 0.5:
                interpretation['clustering_quality'] = 'excellent'
                interpretation['strengths'].append(f"High silhouette score ({silhouette:.3f})")
            elif silhouette > 0.3:
                interpretation['clustering_quality'] = 'good'
                interpretation['strengths'].append(f"Good silhouette score ({silhouette:.3f})")
            elif silhouette > 0.1:
                interpretation['clustering_quality'] = 'fair'
                interpretation['weaknesses'].append(f"Low silhouette score ({silhouette:.3f})")
            else:
                interpretation['clustering_quality'] = 'poor'
                interpretation['weaknesses'].append(f"Very low silhouette score ({silhouette:.3f})")
        
        # Analyze ground truth alignment (if available)
        if self.has_ground_truth and 'ground_truth_evaluation' in all_results:
            gt_results = all_results['ground_truth_evaluation']
            alignment = gt_results.get('alignment', {})
            ari = alignment.get('adjusted_rand_index')
            
            if ari is not None:
                if ari > 0.7:
                    interpretation['ground_truth_alignment'] = 'excellent'
                    interpretation['strengths'].append(f"Excellent ground truth alignment (ARI={ari:.3f})")
                elif ari > 0.5:
                    interpretation['ground_truth_alignment'] = 'good'
                    interpretation['strengths'].append(f"Good ground truth alignment (ARI={ari:.3f})")
                elif ari > 0.3:
                    interpretation['ground_truth_alignment'] = 'fair'
                    interpretation['weaknesses'].append(f"Fair ground truth alignment (ARI={ari:.3f})")
                else:
                    interpretation['ground_truth_alignment'] = 'poor'
                    interpretation['weaknesses'].append(f"Poor ground truth alignment (ARI={ari:.3f})")
        
        # Overall assessment
        if self.has_ground_truth:
            # Weight both clustering quality and ground truth alignment
            cluster_score = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 0}[interpretation['clustering_quality']]
            gt_score = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'not_available': 0}[interpretation['ground_truth_alignment']]
            
            if gt_score > 0:  # Ground truth available
                avg_score = (cluster_score + gt_score) / 2
            else:
                avg_score = cluster_score
        else:
            cluster_score = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 0}[interpretation['clustering_quality']]
            avg_score = cluster_score
        
        if avg_score >= 3.5:
            interpretation['overall_assessment'] = 'excellent'
        elif avg_score >= 2.5:
            interpretation['overall_assessment'] = 'good'
        elif avg_score >= 1.5:
            interpretation['overall_assessment'] = 'fair'
        elif avg_score >= 0.5:
            interpretation['overall_assessment'] = 'poor'
        
        # Add recommendations
        if interpretation['overall_assessment'] in ['poor', 'fair']:
            interpretation['recommendations'].extend([
                "Consider adjusting clustering parameters",
                "Evaluate if paragraph-level clustering captured meaningful distinctions",
                "Consider alternative aggregation methods for paper vectors"
            ])
        
        return interpretation
    
    def print_comprehensive_summary(self, all_results: Dict):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*70)
        
        # Basic clustering results
        basic_results = all_results.get('basic_evaluation', {})
        
        # Validation
        validation = basic_results.get('validation', {})
        print(f"DATA OVERVIEW:")
        print(f"  Total papers: {validation.get('total_papers', 'N/A'):,}")
        print(f"  Clustered papers: {validation.get('clustered_papers', 'N/A'):,}")
        print(f"  Coverage ratio: {validation.get('clustering_coverage', 0):.3f}")
        
        # Quality
        quality = basic_results.get('quality', {})
        basic_stats = quality.get('basic_stats', {})
        print(f"\nCLUSTERING QUALITY:")
        print(f"  Number of clusters: {basic_stats.get('n_clusters', 'N/A')}")
        print(f"  Noise ratio: {basic_stats.get('noise_ratio', 0):.3f}")
        if quality.get('silhouette_score') is not None:
            print(f"  Silhouette score: {quality['silhouette_score']:.3f}")
        
        # Ground truth results (if available)
        if self.has_ground_truth and 'ground_truth_evaluation' in all_results:
            gt_results = all_results['ground_truth_evaluation']
            alignment = gt_results.get('alignment', {})
            
            print(f"\nGROUND TRUTH ALIGNMENT:")
            if alignment.get('adjusted_rand_index') is not None:
                print(f"  Adjusted Rand Index: {alignment['adjusted_rand_index']:.3f}")
            if alignment.get('homogeneity') is not None:
                print(f"  Homogeneity: {alignment['homogeneity']:.3f}")
            if alignment.get('completeness') is not None:
                print(f"  Completeness: {alignment['completeness']:.3f}")
        
        # Overall interpretation
        interpretation = all_results.get('interpretation', {})
        if interpretation:
            print(f"\nOVERALL ASSESSMENT: {interpretation.get('overall_assessment', 'unknown').upper()}")
            
            if interpretation.get('strengths'):
                print(f"\nStrengths:")
                for strength in interpretation['strengths']:
                    print(f"  • {strength}")
            
            if interpretation.get('weaknesses'):
                print(f"\nWeaknesses:")
                for weakness in interpretation['weaknesses']:
                    print(f"  • {weakness}")
            
            if interpretation.get('recommendations'):
                print(f"\nRecommendations:")
                for rec in interpretation['recommendations']:
                    print(f"  • {rec}")
    
    def save_results(self, all_results: Dict, figures: List[plt.Figure]):
        """Save all evaluation results and visualizations"""
        print("\nSaving evaluation results...")
        
        # Save complete results
        results_file = self.work_dir / "comprehensive_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Saved comprehensive results to: {results_file}")
        
        # Save interpretation
        interpretation_file = self.work_dir / "evaluation_interpretation.json"
        interpretation = all_results.get('interpretation', {})
        with open(interpretation_file, 'w') as f:
            json.dump(interpretation, f, indent=2, default=str)
        print(f"Saved interpretation to: {interpretation_file}")
        
        # Save ground truth specific results (if available)
        if self.has_ground_truth and 'ground_truth_evaluation' in all_results:
            gt_file = self.work_dir / "ground_truth_evaluation.json"
            with open(gt_file, 'w') as f:
                json.dump(all_results['ground_truth_evaluation'], f, indent=2, default=str)
            print(f"Saved ground truth evaluation to: {gt_file}")
        
        # Save figures
        for i, fig in enumerate(figures):
            fig_path = self.work_dir / f"evaluation_visualization_{i+1}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved figure: {fig_path}")
        
        # Create summary text report
        summary_file = self.work_dir / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE CLUSTERING EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            self.print_comprehensive_summary(all_results)
            sys.stdout = original_stdout
        
        print(f"Saved summary report to: {summary_file}")


def load_paper_clustering_results(csv_file: Path) -> pd.DataFrame:
    """Load paper clustering results from CSV file"""
    print(f"Loading paper clustering results from: {csv_file}")
    
    if not csv_file.exists():
        raise FileNotFoundError(f"File not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Validate required columns
    required_cols = ['paper_id', 'paper_cluster']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for cluster vector columns
    vector_cols = [col for col in df.columns if col.startswith('cluster_')]
    if not vector_cols:
        raise ValueError("No cluster vector columns found (columns starting with 'cluster_')")
    
    print(f"Loaded {len(df):,} papers with {len(vector_cols)} vector dimensions")
    print(f"Cluster labels: {sorted(df['paper_cluster'].unique())}")
    
    return df


def load_ground_truth_data(gt_file: Path) -> pd.DataFrame:
    """Load ground truth data from CSV file"""
    print(f"Loading ground truth data from: {gt_file}")
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    df = pd.read_csv(gt_file)
    
    # Validate required columns
    required_cols = ['obs. type', 'arXiv']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in ground truth: {missing_cols}")
    
    # Clean the data - remove rows with missing obs. type or arXiv
    original_len = len(df)
    df = df.dropna(subset=['obs. type', 'arXiv'])
    
    # Also remove empty string values
    df = df[(df['obs. type'].str.strip() != '') & (df['arXiv'].str.strip() != '')]
    
    print(f"Loaded {len(df)} ground truth entries (removed {original_len - len(df)} invalid entries)")
    
    # Get unique categories safely (no NaN values now)
    unique_categories = sorted(df['obs. type'].unique())
    print(f"Unique categories: {unique_categories}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive clustering evaluation with optional ground truth comparison"
    )
    
    parser.add_argument("--paper_clusters_file", required=True, 
                       help="Path to paper_clusters.csv file from clustering analysis")
    parser.add_argument("--ground_truth_file", required=False,
                       help="Path to list.csv file with manual classifications (optional)")
    parser.add_argument("--work_dir", required=True, 
                       help="Directory for evaluation outputs")
    parser.add_argument("--n_jobs", type=int, default=None,
                       help="Number of parallel jobs (default: all CPU cores)")
    parser.add_argument("--stability_trials", type=int, default=5,
                       help="Number of trials for stability analysis (default: 5)")
    
    args = parser.parse_args()
    
    try:
        # Load data
        paper_df = load_paper_clustering_results(Path(args.paper_clusters_file))
        
        ground_truth_df = None
        if args.ground_truth_file:
            ground_truth_df = load_ground_truth_data(Path(args.ground_truth_file))
        
        # Setup evaluator
        evaluator = ComprehensiveClusteringEvaluator(
            paper_df, 
            ground_truth_df,
            Path(args.work_dir), 
            n_jobs=args.n_jobs
        )
        
        # Run comprehensive evaluation
        all_results = evaluator.run_complete_evaluation()
        
        # Create visualizations
        basic_figures = evaluator.create_basic_visualizations(all_results['basic_evaluation'])
        figures = basic_figures
        
        if evaluator.has_ground_truth:
            gt_figures = evaluator.create_ground_truth_visualizations(all_results['ground_truth_evaluation'])
            figures.extend(gt_figures)
        
        # Save results
        evaluator.save_results(all_results, figures)
        
        # Print comprehensive summary
        evaluator.print_comprehensive_summary(all_results)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Results saved to: {args.work_dir}")
        
        # Print final assessment
        interpretation = all_results.get('interpretation', {})
        overall_assessment = interpretation.get('overall_assessment', 'unknown')
        print(f"\nOVERALL CLUSTERING ASSESSMENT: {overall_assessment.upper()}")
        
        if evaluator.has_ground_truth:
            gt_alignment = interpretation.get('ground_truth_alignment', 'unknown')
            print(f"GROUND TRUTH ALIGNMENT: {gt_alignment.upper()}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())