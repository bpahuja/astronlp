#!/usr/bin/env python3
"""
Standalone script for comprehensive evaluation of paper clustering results.
Takes the output CSV from paper clustering analysis and provides detailed evaluation metrics.

Input: paper_clusters.csv (output from paper clustering script)
Output: Comprehensive evaluation metrics and visualizations
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from collections import Counter
import multiprocessing

warnings.filterwarnings('ignore')
plt.style.use('default')


class StandalonePaperClusteringEvaluator:
    """Comprehensive evaluation of paper clustering results from CSV file"""
    
    def __init__(self, paper_df: pd.DataFrame, work_dir: Path, n_jobs: Optional[int] = None):
        self.paper_df = paper_df
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up parallel processing
        if n_jobs is None or n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif n_jobs > 0:
            self.n_jobs = n_jobs
        else:
            self.n_jobs = 1
        
        # Identify vector columns and cluster labels
        self.vector_cols = [col for col in paper_df.columns if col.startswith('cluster_')]
        self.cluster_labels = paper_df['paper_cluster'].values if 'paper_cluster' in paper_df.columns else None
        
        if self.cluster_labels is None:
            raise ValueError("No 'paper_cluster' column found in the data")
        
        if not self.vector_cols:
            raise ValueError("No cluster vector columns found (columns starting with 'cluster_')")
        
        print(f"Initialized evaluator with:")
        print(f"  - {len(paper_df)} papers")
        print(f"  - {len(self.vector_cols)} vector dimensions")
        print(f"  - {len(np.unique(self.cluster_labels))} unique cluster labels")
        print(f"  - Using {self.n_jobs} parallel jobs")
    
    def validate_data(self) -> Dict:
        """Validate input data and provide basic statistics"""
        print("Validating input data...")
        
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
        print(f"  - Zero vectors: {validation['zero_vectors']:,}")
        print(f"  - Vector sparsity: {validation['vector_sparsity']:.3f}")
        
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
        
        # Internal clustering metrics (only if we have valid clusters)
        if n_clusters > 1 and len(np.unique(labels_clustered[labels_clustered >= 0])) > 1:
            try:
                # Silhouette Score
                metrics['silhouette_score'] = silhouette_score(X_clustered, labels_clustered)
                
                # Only calculate CH and DB for non-noise points
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
    
    def evaluate_cluster_stability(self, n_trials: int = 10, noise_level: float = 0.01) -> Dict:
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
            print("Warning: Too few points for stability analysis")
            return {'error': 'insufficient_data'}
        
        stability_scores = []
        
        # Estimate reasonable clustering parameters from original results
        unique_original = np.unique(original_labels[original_labels >= 0])
        n_original_clusters = len(unique_original)
        
        # Use adaptive min_cluster_size based on data size
        min_cluster_size = max(5, len(X_clustered) // (n_original_clusters * 3) if n_original_clusters > 0 else 10)
        
        for trial in range(n_trials):
            try:
                # Add small random noise
                noise = np.random.normal(0, noise_level, X_clustered.shape)
                X_perturbed = X_clustered + noise
                
                # Re-cluster with adaptive parameters
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(3, min_cluster_size // 3),
                    n_jobs=1  # Use single job to avoid nested parallelism
                )
                trial_labels = clusterer.fit_predict(X_perturbed)
                
                # Calculate ARI with original clustering
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
    
    def evaluate_cluster_coherence(self) -> Dict:
        """Evaluate methodological coherence within clusters"""
        print("Evaluating cluster coherence...")
        
        # Get unique clusters (excluding noise and excluded)
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
            
            # Vector coherence: cosine similarity within cluster
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
                    'max_similarity': float(np.max(pairwise_sims)),
                    'q25_similarity': float(np.percentile(pairwise_sims, 25)),
                    'q75_similarity': float(np.percentile(pairwise_sims, 75))
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
                'median_cluster_coherence': np.median(all_mean_sims),
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
        intra_cluster_stats = {}
        
        for cluster_id in cluster_centroids.keys():
            cluster_mask = self.cluster_labels == cluster_id
            cluster_vectors = X_scaled[cluster_mask]
            
            if len(cluster_vectors) > 1:
                intra_distances = pdist(cluster_vectors, metric='euclidean')
                intra_cluster_distances.extend(intra_distances)
                intra_cluster_stats[cluster_id] = {
                    'mean_intra_distance': float(np.mean(intra_distances)),
                    'std_intra_distance': float(np.std(intra_distances)),
                    'size': len(cluster_vectors)
                }
        
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
            'n_clusters_evaluated': len(cluster_centroids),
            'intra_cluster_stats': intra_cluster_stats,
            'cluster_sizes': cluster_sizes
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
            'size_quartiles': [float(q) for q in np.percentile(cluster_sizes, [25, 50, 75])],
            'cluster_size_distribution': cluster_size_dict
        }
        
        return {
            'cluster_sizes': cluster_sizes.tolist(),
            'balance_metrics': balance_metrics
        }
    
    def evaluate_dimensionality_impact(self) -> Dict:
        """Evaluate impact of vector dimensionality on clustering"""
        print("Evaluating dimensionality impact...")
        
        X = self.paper_df[self.vector_cols].values
        
        # Remove zero vectors for PCA analysis
        non_zero_mask = ~np.all(X == 0, axis=1)
        X_non_zero = X[non_zero_mask]
        
        if len(X_non_zero) < 2:
            return {'error': 'insufficient_non_zero_vectors'}
        
        # Standardize for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_non_zero)
        
        # PCA analysis
        n_components = min(len(X_non_zero), X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Find components for variance thresholds
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Find number of components for different variance levels
        variance_thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
        components_for_variance = {}
        
        for threshold in variance_thresholds:
            idx = np.argmax(cumsum_var >= threshold)
            if cumsum_var[idx] >= threshold:
                components_for_variance[f'{int(threshold*100)}%'] = int(idx + 1)
            else:
                components_for_variance[f'{int(threshold*100)}%'] = n_components
        
        # Calculate participation ratio (intrinsic dimensionality)
        eigenvalues = pca.explained_variance_
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        
        return {
            'original_dimensions': X.shape[1],
            'non_zero_samples': len(X_non_zero),
            'zero_vector_ratio': (len(X) - len(X_non_zero)) / len(X),
            'components_for_variance': components_for_variance,
            'participation_ratio': float(participation_ratio),
            'explained_variance_ratio': [float(x) for x in pca.explained_variance_ratio_[:20]],  # Top 20
            'cumulative_variance': [float(x) for x in cumsum_var[:20]],
            'total_variance_in_top_10': float(np.sum(pca.explained_variance_ratio_[:10])),
            'effective_rank': float(np.sum(pca.explained_variance_ratio_ > 0.01))  # Components with >1% variance
        }
    
    def create_evaluation_visualizations(self, evaluation_results: Dict) -> List[plt.Figure]:
        """Create comprehensive evaluation visualizations"""
        print("Creating evaluation visualizations...")
        
        figures = []
        
        # Figure 1: Quality and Balance Overview
        fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Cluster sizes
        if 'balance' in evaluation_results and 'cluster_sizes' in evaluation_results['balance']:
            cluster_sizes = evaluation_results['balance']['cluster_sizes']
            axes[0, 0].hist(cluster_sizes, bins=min(20, len(cluster_sizes)), alpha=0.7, 
                           color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Cluster Size')
            axes[0, 0].set_ylabel('Number of Clusters')
            axes[0, 0].set_title('Cluster Size Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add statistics
            balance_metrics = evaluation_results['balance']['balance_metrics']
            stats_text = f"Mean: {balance_metrics['mean_cluster_size']:.1f}\n"
            stats_text += f"CV: {balance_metrics['coefficient_of_variation']:.3f}\n"
            stats_text += f"Entropy: {balance_metrics['size_entropy']:.3f}"
            axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 2: Quality metrics
        quality_metrics = evaluation_results.get('quality', {})
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
        if ('coherence' in evaluation_results and 
            'cluster_coherences' in evaluation_results['coherence']):
            coherences = [metrics['mean_pairwise_similarity'] 
                         for metrics in evaluation_results['coherence']['cluster_coherences'].values()]
            axes[1, 0].hist(coherences, bins=min(15, len(coherences)), alpha=0.7, 
                           color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('Mean Pairwise Similarity')
            axes[1, 0].set_ylabel('Number of Clusters')
            axes[1, 0].set_title('Cluster Coherence Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            mean_coherence = evaluation_results['coherence']['overall_mean_coherence']
            axes[1, 0].axvline(mean_coherence, color='red', linestyle='--', 
                              label=f'Mean: {mean_coherence:.3f}')
            axes[1, 0].legend()
        
        # Plot 4: Separation analysis
        if ('separation' in evaluation_results and 
            'separation_ratio' in evaluation_results['separation'] and
            evaluation_results['separation']['separation_ratio'] != float('inf')):
            sep_metrics = evaluation_results['separation']
            
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
        
        # Figure 2: Dimensionality and Stability Analysis
        fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: PCA Explained Variance
        if ('dimensionality' in evaluation_results and 
            'explained_variance_ratio' in evaluation_results['dimensionality']):
            var_ratios = evaluation_results['dimensionality']['explained_variance_ratio']
            cumsum_var = evaluation_results['dimensionality']['cumulative_variance']
            
            x_range = range(1, min(len(var_ratios) + 1, 21))  # Limit to 20 components
            var_ratios_plot = var_ratios[:20]
            cumsum_var_plot = cumsum_var[:20]
            
            ax1 = axes[0]
            ax2 = ax1.twinx()
            
            bars = ax1.bar(x_range, var_ratios_plot, alpha=0.6, color='skyblue', 
                          label='Individual Variance')
            line = ax2.plot(x_range, cumsum_var_plot, 'ro-', color='red', 
                           label='Cumulative Variance')
            
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio', color='blue')
            ax2.set_ylabel('Cumulative Explained Variance', color='red')
            ax1.set_title('PCA Explained Variance Analysis')
            ax1.grid(True, alpha=0.3)
            
            # Add threshold lines
            ax2.axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
            ax2.axhline(0.90, color='green', linestyle='--', alpha=0.7, label='90%')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        # Plot 2: Stability analysis
        if ('stability' in evaluation_results and 
            'stability_scores' in evaluation_results['stability']):
            stability_scores = evaluation_results['stability']['stability_scores']
            axes[1].plot(range(1, len(stability_scores) + 1), stability_scores, 
                        'o-', alpha=0.7, color='purple', linewidth=2, markersize=6)
            
            mean_stability = evaluation_results['stability']['mean_stability']
            axes[1].axhline(mean_stability, color='red', linestyle='--', 
                           label=f"Mean: {mean_stability:.3f}")
            
            axes[1].set_xlabel('Trial')
            axes[1].set_ylabel('ARI Score')
            axes[1].set_title('Clustering Stability Across Trials')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        figures.append(fig2)
        
        return figures
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run all evaluation metrics and generate comprehensive report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE PAPER CLUSTERING EVALUATION")
        print("="*70)
        
        evaluation_results = {}
        
        # 1. Data validation
        evaluation_results['validation'] = self.validate_data()
        
        # 2. Basic clustering quality
        evaluation_results['quality'] = self.evaluate_clustering_quality()
        
        # 3. Cluster coherence
        evaluation_results['coherence'] = self.evaluate_cluster_coherence()
        
        # 4. Cluster separation
        evaluation_results['separation'] = self.evaluate_cluster_separation()
        
        # 5. Cluster balance
        evaluation_results['balance'] = self.evaluate_cluster_balance()
        
        # 6. Dimensionality impact
        evaluation_results['dimensionality'] = self.evaluate_dimensionality_impact()
        
        # 7. Stability analysis (fewer trials to avoid long runtime)
        evaluation_results['stability'] = self.evaluate_cluster_stability(n_trials=5)
        
        # 8. Generate summary interpretation
        evaluation_results['interpretation'] = self.interpret_results(evaluation_results)
        
        # Print comprehensive summary
        self.print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def interpret_results(self, evaluation_results: Dict) -> Dict:
        """Provide interpretation and recommendations based on evaluation results"""
        interpretation = {
            'overall_quality': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Analyze silhouette score
        silhouette = evaluation_results.get('quality', {}).get('silhouette_score')
        if silhouette is not None:
            if silhouette > 0.5:
                interpretation['strengths'].append(f"High silhouette score ({silhouette:.3f}) indicates well-separated clusters")
                interpretation['overall_quality'] = 'excellent'
            elif silhouette > 0.3:
                interpretation['strengths'].append(f"Good silhouette score ({silhouette:.3f}) indicates reasonable cluster separation")
                interpretation['overall_quality'] = 'good'
            elif silhouette > 0.1:
                interpretation['weaknesses'].append(f"Low silhouette score ({silhouette:.3f}) suggests overlapping clusters")
                interpretation['overall_quality'] = 'fair'
            else:
                interpretation['weaknesses'].append(f"Very low silhouette score ({silhouette:.3f}) indicates poor clustering")
                interpretation['overall_quality'] = 'poor'
        
        # Analyze cluster balance
        balance_metrics = evaluation_results.get('balance', {}).get('balance_metrics', {})
        cv = balance_metrics.get('coefficient_of_variation', 0)
        entropy = balance_metrics.get('size_entropy', 0)
        largest_ratio = balance_metrics.get('largest_cluster_ratio', 0)
        
        if cv < 1.0 and entropy > 2.0:
            interpretation['strengths'].append(f"Well-balanced cluster sizes (CV={cv:.2f}, entropy={entropy:.2f})")
        elif largest_ratio > 0.8:
            interpretation['weaknesses'].append(f"One cluster dominates ({largest_ratio:.1%} of points)")
            interpretation['recommendations'].append("Consider increasing clustering granularity")
        elif cv > 2.0:
            interpretation['weaknesses'].append(f"Highly imbalanced cluster sizes (CV={cv:.2f})")
        
        # Analyze coherence
        coherence_metrics = evaluation_results.get('coherence', {})
        if 'overall_mean_coherence' in coherence_metrics:
            mean_coherence = coherence_metrics['overall_mean_coherence']
            if mean_coherence > 0.7:
                interpretation['strengths'].append(f"High intra-cluster coherence ({mean_coherence:.3f})")
            elif mean_coherence < 0.3:
                interpretation['weaknesses'].append(f"Low intra-cluster coherence ({mean_coherence:.3f})")
                interpretation['recommendations'].append("Papers within clusters are not very similar")
        
        # Analyze separation
        separation_metrics = evaluation_results.get('separation', {})
        separation_ratio = separation_metrics.get('separation_ratio')
        if separation_ratio is not None and separation_ratio != float('inf'):
            if separation_ratio > 2.0:
                interpretation['strengths'].append(f"Good cluster separation (ratio={separation_ratio:.2f})")
            elif separation_ratio < 1.2:
                interpretation['weaknesses'].append(f"Poor cluster separation (ratio={separation_ratio:.2f})")
                interpretation['recommendations'].append("Consider adjusting clustering parameters for better separation")
        
        # Analyze stability
        stability_metrics = evaluation_results.get('stability', {})
        if 'mean_stability' in stability_metrics:
            mean_stability = stability_metrics['mean_stability']
            if mean_stability > 0.7:
                interpretation['strengths'].append(f"High clustering stability ({mean_stability:.3f})")
            elif mean_stability < 0.3:
                interpretation['weaknesses'].append(f"Low clustering stability ({mean_stability:.3f})")
                interpretation['recommendations'].append("Clustering is sensitive to small changes in data")
        
        # Analyze noise levels
        quality_stats = evaluation_results.get('quality', {}).get('basic_stats', {})
        noise_ratio = quality_stats.get('noise_ratio', 0)
        if noise_ratio > 0.3:
            interpretation['weaknesses'].append(f"High noise ratio ({noise_ratio:.1%})")
            interpretation['recommendations'].append("Consider preprocessing to reduce noise")
        elif noise_ratio < 0.1:
            interpretation['strengths'].append(f"Low noise ratio ({noise_ratio:.1%})")
        
        # Overall recommendations
        if interpretation['overall_quality'] in ['poor', 'fair']:
            interpretation['recommendations'].extend([
                "Consider using different clustering parameters",
                "Evaluate if the paragraph-level clustering captured meaningful distinctions",
                "Consider alternative aggregation methods for paper vectors"
            ])
        
        return interpretation
    
    def print_evaluation_summary(self, evaluation_results: Dict):
        """Print comprehensive evaluation summary"""
        print("\nEVALUATION SUMMARY")
        print("-" * 50)
        
        # Basic statistics
        validation = evaluation_results.get('validation', {})
        print(f"DATA OVERVIEW:")
        print(f"  Total papers: {validation.get('total_papers', 'N/A'):,}")
        print(f"  Clustered papers: {validation.get('clustered_papers', 'N/A'):,}")
        print(f"  Clustering coverage: {validation.get('clustering_coverage', 0):.3f}")
        print(f"  Vector dimensions: {validation.get('vector_dimensions', 'N/A')}")
        
        # Quality metrics
        quality = evaluation_results.get('quality', {})
        basic_stats = quality.get('basic_stats', {})
        print(f"\nCLUSTERING QUALITY:")
        print(f"  Number of clusters: {basic_stats.get('n_clusters', 'N/A')}")
        print(f"  Noise ratio: {basic_stats.get('noise_ratio', 0):.3f}")
        print(f"  Average cluster size: {basic_stats.get('avg_cluster_size', 0):.1f}")
        print(f"  Size range: {basic_stats.get('min_cluster_size', 'N/A')} - {basic_stats.get('max_cluster_size', 'N/A')}")
        
        if quality.get('silhouette_score') is not None:
            print(f"  Silhouette score: {quality['silhouette_score']:.3f}")
        if quality.get('calinski_harabasz') is not None:
            print(f"  Calinski-Harabasz: {quality['calinski_harabasz']:.1f}")
        if quality.get('davies_bouldin') is not None:
            print(f"  Davies-Bouldin: {quality['davies_bouldin']:.3f}")
        
        # Coherence
        coherence = evaluation_results.get('coherence', {})
        if 'overall_mean_coherence' in coherence:
            print(f"\nCLUSTER COHERENCE:")
            print(f"  Mean coherence: {coherence['overall_mean_coherence']:.3f}")
            print(f"  Coherence range: {coherence.get('min_cluster_coherence', 'N/A'):.3f} - {coherence.get('max_cluster_coherence', 'N/A'):.3f}")
            print(f"  Clusters evaluated: {coherence.get('n_evaluated_clusters', 'N/A')}")
        
        # Separation
        separation = evaluation_results.get('separation', {})
        if 'separation_ratio' in separation and separation['separation_ratio'] != float('inf'):
            print(f"\nCLUSTER SEPARATION:")
            print(f"  Separation ratio: {separation['separation_ratio']:.2f}")
            print(f"  Inter-cluster distance: {separation['mean_inter_cluster_distance']:.3f}")
            print(f"  Intra-cluster distance: {separation['mean_intra_cluster_distance']:.3f}")
        
        # Balance
        balance = evaluation_results.get('balance', {}).get('balance_metrics', {})
        if balance:
            print(f"\nCLUSTER BALANCE:")
            print(f"  Size coefficient of variation: {balance.get('coefficient_of_variation', 'N/A'):.3f}")
            print(f"  Size entropy: {balance.get('size_entropy', 'N/A'):.3f}")
            print(f"  Largest cluster ratio: {balance.get('largest_cluster_ratio', 'N/A'):.3f}")
        
        # Stability
        stability = evaluation_results.get('stability', {})
        if 'mean_stability' in stability:
            print(f"\nCLUSTER STABILITY:")
            print(f"  Mean stability (ARI): {stability['mean_stability']:.3f}")
            print(f"  Stability std: {stability['std_stability']:.3f}")
            print(f"  Successful trials: {stability.get('n_successful_trials', 'N/A')}")
        
        # Dimensionality
        dimensionality = evaluation_results.get('dimensionality', {})
        if 'participation_ratio' in dimensionality:
            print(f"\nDIMENSIONALITY ANALYSIS:")
            print(f"  Original dimensions: {dimensionality.get('original_dimensions', 'N/A')}")
            components_95 = dimensionality.get('components_for_variance', {}).get('95%', 'N/A')
            print(f"  Effective dims (95% var): {components_95}")
            print(f"  Participation ratio: {dimensionality['participation_ratio']:.1f}")
            print(f"  Effective rank: {dimensionality.get('effective_rank', 'N/A'):.1f}")
        
        # Interpretation
        interpretation = evaluation_results.get('interpretation', {})
        if interpretation:
            print(f"\nINTERPRETATION:")
            print(f"  Overall quality: {interpretation.get('overall_quality', 'unknown').upper()}")
            
            if interpretation.get('strengths'):
                print(f"  Strengths:")
                for strength in interpretation['strengths']:
                    print(f"    • {strength}")
            
            if interpretation.get('weaknesses'):
                print(f"  Weaknesses:")
                for weakness in interpretation['weaknesses']:
                    print(f"    • {weakness}")
            
            if interpretation.get('recommendations'):
                print(f"  Recommendations:")
                for rec in interpretation['recommendations']:
                    print(f"    • {rec}")
    
    def save_evaluation_results(self, evaluation_results: Dict, figures: List[plt.Figure]):
        """Save all evaluation results and visualizations"""
        print("\nSaving evaluation results...")
        
        # Save comprehensive evaluation JSON
        results_file = self.work_dir / "comprehensive_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"Saved evaluation results to: {results_file}")
        
        # Save interpretation summary
        interpretation_file = self.work_dir / "evaluation_interpretation.json"
        interpretation = evaluation_results.get('interpretation', {})
        with open(interpretation_file, 'w') as f:
            json.dump(interpretation, f, indent=2, default=str)
        print(f"Saved interpretation to: {interpretation_file}")
        
        # Save cluster-specific details
        if 'coherence' in evaluation_results and 'cluster_coherences' in evaluation_results['coherence']:
            cluster_details_file = self.work_dir / "cluster_coherence_details.json"
            cluster_details = evaluation_results['coherence']['cluster_coherences']
            with open(cluster_details_file, 'w') as f:
                json.dump(cluster_details, f, indent=2, default=str)
            print(f"Saved cluster details to: {cluster_details_file}")
        
        # Save figures
        figure_names = ['evaluation_overview.png', 'dimensionality_stability.png']
        for fig, name in zip(figures, figure_names):
            fig_path = self.work_dir / name
            fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved figure: {fig_path}")
        
        # Create summary text report
        summary_file = self.work_dir / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PAPER CLUSTERING EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            self.print_evaluation_summary(evaluation_results)
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


def main():
    parser = argparse.ArgumentParser(
        description="Standalone comprehensive evaluation of paper clustering results"
    )
    
    parser.add_argument("--paper_clusters_file", required=True, 
                       help="Path to paper_clusters.csv file from clustering analysis")
    parser.add_argument("--work_dir", required=True, 
                       help="Directory for evaluation outputs")
    parser.add_argument("--n_jobs", type=int, default=None,
                       help="Number of parallel jobs (default: all CPU cores)")
    parser.add_argument("--stability_trials", type=int, default=10,
                       help="Number of trials for stability analysis (default: 10)")
    parser.add_argument("--noise_level", type=float, default=0.01,
                       help="Noise level for stability testing (default: 0.01)")
    
    args = parser.parse_args()
    
    try:
        # Load data
        paper_df = load_paper_clustering_results(Path(args.paper_clusters_file))
        
        # Setup evaluator
        evaluator = StandalonePaperClusteringEvaluator(
            paper_df, 
            Path(args.work_dir), 
            n_jobs=args.n_jobs
        )
        
        # Run comprehensive evaluation
        evaluation_results = evaluator.run_comprehensive_evaluation()
        
        # Update stability analysis parameters if specified
        if args.stability_trials != 10 or args.noise_level != 0.01:
            print(f"\nRunning custom stability analysis...")
            evaluation_results['stability'] = evaluator.evaluate_cluster_stability(
                n_trials=args.stability_trials, 
                noise_level=args.noise_level
            )
        
        # Create visualizations
        figures = evaluator.create_evaluation_visualizations(evaluation_results)
        
        # Save results
        evaluator.save_evaluation_results(evaluation_results, figures)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Results saved to: {args.work_dir}")
        print("\nFiles created:")
        print("  - comprehensive_evaluation_results.json: Complete evaluation metrics")
        print("  - evaluation_interpretation.json: Summary interpretation")
        print("  - cluster_coherence_details.json: Per-cluster coherence analysis")
        print("  - evaluation_summary.txt: Human-readable summary report")
        print("  - evaluation_overview.png: Quality and balance visualization")
        print("  - dimensionality_stability.png: Advanced analysis visualization")
        
        # Print final recommendation
        interpretation = evaluation_results.get('interpretation', {})
        overall_quality = interpretation.get('overall_quality', 'unknown')
        print(f"\nOVERALL CLUSTERING QUALITY: {overall_quality.upper()}")
        
        if overall_quality in ['excellent', 'good']:
            print("✓ Your paper clustering shows good quality metrics")
        elif overall_quality == 'fair':
            print("⚠ Your paper clustering shows mixed results - see recommendations")
        else:
            print("⚠ Your paper clustering may need improvement - see detailed analysis")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())