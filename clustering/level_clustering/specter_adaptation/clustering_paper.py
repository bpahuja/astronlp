#!/usr/bin/env python3
"""
Comprehensive clustering experimentation framework for thesis research.
Allows systematic comparison of different algorithms, preprocessing methods, 
distance metrics, and parameters on the same dataset.

Author: Research Framework
Usage: python clustering_experiments.py --config experiments_config.yaml
"""

import json
import yaml
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.cluster import OPTICS, Birch, MeanShift, AffinityPropagation
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN

# Preprocessing and dimensionality reduction
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.manifold import TSNE

# Distance metrics and evaluation
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score, fowlkes_mallows_score
)
from sklearn.metrics.pairwise import pairwise_distances
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Statistics
import scipy.stats as stats
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class ClusteringExperimentFramework:
    """Comprehensive framework for clustering algorithm comparison"""
    
    def __init__(self, work_dir: Path, n_jobs: int = -1, random_state: int = 42):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize results directory
        self.results_dir = self.work_dir / f"experiments_{self.experiment_id}"
        self.results_dir.mkdir(exist_ok=True)
        
    def load_data(self, data_path: Path, vector_prefix: str = 'cluster_') -> Tuple[np.ndarray, pd.DataFrame]:
        """Load paper clustering data"""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Extract feature vectors
        vector_cols = [col for col in df.columns if col.startswith(vector_prefix)]
        X = df[vector_cols].values
        
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Feature sparsity: {(X == 0).mean():.3f}")
        
        return X, df
    
    def apply_preprocessing(self, X: np.ndarray, method: str, **kwargs) -> Tuple[np.ndarray, Any]:
        """Apply preprocessing to feature matrix"""
        if method == 'none':
            return X, None
        elif method == 'standard':
            scaler = StandardScaler()
            return scaler.fit_transform(X), scaler
        elif method == 'minmax':
            scaler = MinMaxScaler()
            return scaler.fit_transform(X), scaler
        elif method == 'robust':
            scaler = RobustScaler()
            return scaler.fit_transform(X), scaler
        elif method == 'l2_normalize':
            scaler = Normalizer(norm='l2')
            return scaler.fit_transform(X), scaler
        elif method == 'l1_normalize':
            scaler = Normalizer(norm='l1')
            return scaler.fit_transform(X), scaler
        elif method == 'tfidf':
            # Apply TF-IDF transformation to count vectors
            transformer = TfidfTransformer()
            return transformer.fit_transform(X).toarray(), transformer
        elif method == 'pca':
            n_components = kwargs.get('n_components', min(50, X.shape[1] - 1))
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            return reducer.fit_transform(X), reducer
        elif method == 'svd':
            n_components = kwargs.get('n_components', min(50, X.shape[1] - 1))
            reducer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            return reducer.fit_transform(X), reducer
        elif method == 'nmf':
            n_components = kwargs.get('n_components', min(20, X.shape[1] - 1))
            reducer = NMF(n_components=n_components, random_state=self.random_state, max_iter=1000)
            return reducer.fit_transform(X), reducer
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def create_clusterer(self, algorithm: str, distance_metric: str = 'euclidean', **params) -> Any:
        """Create clustering algorithm instance"""
        # Set random state for reproducibility where applicable
        if 'random_state' in params or algorithm in ['kmeans', 'spectral', 'gmm']:
            params['random_state'] = self.random_state
        
        if algorithm == 'kmeans':
            # Handle distance metric for K-means
            if distance_metric == 'cosine':
                # For cosine distance, we'll precompute distances later
                return KMeans( **params)
            else:
                return KMeans( **params)
                
        elif algorithm == 'hdbscan':
            return HDBSCAN(metric=distance_metric,  **params)
            
        elif algorithm == 'dbscan':
            return DBSCAN(metric=distance_metric,  **params)
            
        elif algorithm == 'hierarchical':
            # Note: sklearn AgglomerativeClustering doesn't support all metrics
            if distance_metric in ['euclidean', 'manhattan', 'cosine']:
                return AgglomerativeClustering(metric=distance_metric, **params)
            else:
                return AgglomerativeClustering(**params)
                
        elif algorithm == 'spectral':
            if distance_metric == 'cosine':
                params['affinity'] = 'cosine'
            return SpectralClustering( **params)
            
        elif algorithm == 'optics':
            return OPTICS(metric=distance_metric,  **params)
            
        elif algorithm == 'birch':
            return Birch(**params)
            
        elif algorithm == 'meanshift':
            return MeanShift( **params)
            
        elif algorithm == 'affinity':
            return AffinityPropagation(random_state=self.random_state, **params)
            
        elif algorithm == 'gmm':
            return GaussianMixture(**params)
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def fit_predict_with_distance(self, clusterer, X: np.ndarray, algorithm: str, 
                                 distance_metric: str) -> np.ndarray:
        """Fit clustering algorithm with appropriate distance handling"""
        
        if algorithm == 'kmeans' and distance_metric == 'cosine':
            # For K-means with cosine distance, use spherical K-means approach
            from sklearn.cluster import KMeans
            # Normalize vectors for cosine similarity
            X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
            X_norm = np.nan_to_num(X_norm)  # Handle zero vectors
            labels = clusterer.fit_predict(X_norm)
            
        elif algorithm == 'hierarchical' and distance_metric not in ['euclidean', 'manhattan', 'cosine']:
            # For hierarchical clustering with custom distances
            from scipy.cluster.hierarchy import linkage, fcluster
            distances = pdist(X, metric=distance_metric)
            linkage_matrix = linkage(distances, method='average')
            n_clusters = getattr(clusterer, 'n_clusters', 8)
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            
        else:
            # Standard fit_predict
            labels = clusterer.fit_predict(X)
            
        return labels
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, 
                          true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Comprehensive clustering evaluation metrics"""
        metrics = {}
        
        # Filter out noise points for internal metrics
        mask = labels != -1
        n_total = len(labels)
        n_clustered = mask.sum()
        
        if n_clustered < 2:
            # No valid clusters found
            return {
                'n_clusters': 0,
                'noise_ratio': 1.0,
                'clustered_ratio': 0.0,
                'silhouette_score': -1,
                'calinski_harabasz_score': 0,
                'davies_bouldin_score': float('inf'),
                'inertia': float('inf')
            }
        
        X_clustered = X[mask]
        labels_clustered = labels[mask]
        
        # Basic cluster statistics
        unique_labels = np.unique(labels_clustered)
        metrics['n_clusters'] = len(unique_labels)
        metrics['noise_ratio'] = (n_total - n_clustered) / n_total
        metrics['clustered_ratio'] = n_clustered / n_total
        
        # Cluster size statistics
        cluster_sizes = [np.sum(labels_clustered == label) for label in unique_labels]
        metrics['min_cluster_size'] = min(cluster_sizes) if cluster_sizes else 0
        metrics['max_cluster_size'] = max(cluster_sizes) if cluster_sizes else 0
        metrics['avg_cluster_size'] = np.mean(cluster_sizes) if cluster_sizes else 0
        metrics['cluster_size_std'] = np.std(cluster_sizes) if cluster_sizes else 0
        
        # Internal validity metrics
        try:
            if len(unique_labels) > 1:
                metrics['silhouette_score'] = silhouette_score(X_clustered, labels_clustered)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clustered, labels_clustered)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_clustered, labels_clustered)
            else:
                metrics['silhouette_score'] = -1
                metrics['calinski_harabasz_score'] = 0
                metrics['davies_bouldin_score'] = float('inf')
        except Exception as e:
            print(f"Warning: Could not compute internal metrics: {e}")
            metrics['silhouette_score'] = -1
            metrics['calinski_harabasz_score'] = 0
            metrics['davies_bouldin_score'] = float('inf')
        
        # Inertia (sum of squared distances to centroids)
        try:
            centroids = []
            total_inertia = 0
            for label in unique_labels:
                cluster_points = X_clustered[labels_clustered == label]
                centroid = np.mean(cluster_points, axis=0)
                inertia = np.sum((cluster_points - centroid) ** 2)
                total_inertia += inertia
                centroids.append(centroid)
            metrics['inertia'] = total_inertia
        except:
            metrics['inertia'] = float('inf')
        
        # External validity metrics (if true labels provided)
        if true_labels is not None and len(true_labels) == len(labels):
            try:
                # Filter true labels for clustered points only
                true_labels_clustered = true_labels[mask]
                
                metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels_clustered, labels_clustered)
                metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(true_labels_clustered, labels_clustered)
                metrics['homogeneity_score'] = homogeneity_score(true_labels_clustered, labels_clustered)
                metrics['completeness_score'] = completeness_score(true_labels_clustered, labels_clustered)
                metrics['v_measure_score'] = v_measure_score(true_labels_clustered, labels_clustered)
                metrics['fowlkes_mallows_score'] = fowlkes_mallows_score(true_labels_clustered, labels_clustered)
            except Exception as e:
                print(f"Warning: Could not compute external metrics: {e}")
        
        return metrics
    
    def run_single_experiment(self, X: np.ndarray, experiment_config: Dict, 
                            df: pd.DataFrame, true_labels: Optional[np.ndarray] = None) -> Dict:
        """Run a single clustering experiment"""
        
        exp_name = experiment_config['name']
        algorithm = experiment_config['algorithm']
        distance_metric = experiment_config.get('distance_metric', 'euclidean')
        preprocessing = experiment_config.get('preprocessing', 'none')
        params = experiment_config.get('params', {})
        preprocessing_params = experiment_config.get('preprocessing_params', {})
        
        print(f"Running experiment: {exp_name}")
        
        try:
            # Apply preprocessing
            start_time = datetime.now()
            X_processed, preprocessor = self.apply_preprocessing(X, preprocessing, **preprocessing_params)
            preprocessing_time = (datetime.now() - start_time).total_seconds()
            
            # Create and fit clusterer
            start_time = datetime.now()
            clusterer = self.create_clusterer(algorithm, distance_metric, **params)
            labels = self.fit_predict_with_distance(clusterer, X_processed, algorithm, distance_metric)
            clustering_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate results
            metrics = self.evaluate_clustering(X_processed, labels, true_labels)
            
            # Prepare results
            result = {
                'experiment_name': exp_name,
                'algorithm': algorithm,
                'distance_metric': distance_metric,
                'preprocessing': preprocessing,
                'params': params,
                'preprocessing_params': preprocessing_params,
                'preprocessing_time': preprocessing_time,
                'clustering_time': clustering_time,
                'total_time': preprocessing_time + clustering_time,
                'success': True,
                'error': None,
                'labels': labels,
                'metrics': metrics,
                'X_shape_original': X.shape,
                'X_shape_processed': X_processed.shape
            }
            
            print(f"  ✓ Success: {metrics['n_clusters']} clusters, "
                  f"silhouette: {metrics['silhouette_score']:.3f}, "
                  f"time: {result['total_time']:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            result = {
                'experiment_name': exp_name,
                'algorithm': algorithm,
                'distance_metric': distance_metric,
                'preprocessing': preprocessing,
                'params': params,
                'preprocessing_params': preprocessing_params,
                'success': False,
                'error': str(e),
                'labels': None,
                'metrics': {},
                'X_shape_original': X.shape,
                'X_shape_processed': None
            }
        
        return result
    
    def run_experiments(self, X: np.ndarray, df: pd.DataFrame, 
                       experiment_configs: List[Dict], 
                       true_labels: Optional[np.ndarray] = None) -> List[Dict]:
        """Run all clustering experiments"""
        
        print(f"Running {len(experiment_configs)} experiments...")
        print("="*80)
        
        results = []
        for i, config in enumerate(experiment_configs, 1):
            print(f"Experiment {i}/{len(experiment_configs)}")
            result = self.run_single_experiment(X, config, df, true_labels)
            results.append(result)
            print()
        
        # Save individual results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save experiment results"""
        print("Saving results...")
        
        # Save detailed results (without large arrays)
        results_summary = []
        for result in results:
            summary = result.copy()
            # Remove large arrays for JSON serialization
            if 'labels' in summary:
                summary['labels'] = f"array_shape_{len(summary['labels']) if summary['labels'] is not None else 0}"
            results_summary.append(summary)
        
        # Save as JSON
        results_file = self.results_dir / 'experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        metrics_data = []
        for result in results:
            if result['success']:
                row = {
                    'experiment_name': result['experiment_name'],
                    'algorithm': result['algorithm'],
                    'distance_metric': result['distance_metric'],
                    'preprocessing': result['preprocessing'],
                    'total_time': result['total_time'],
                }
                row.update(result['metrics'])
                metrics_data.append(row)
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_file = self.results_dir / 'experiment_metrics.csv'
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Saved metrics to: {metrics_file}")
        
        # Save complete results with labels
        complete_results_file = self.results_dir / 'complete_results.pkl'
        pd.to_pickle(results, complete_results_file)
        print(f"Saved complete results to: {complete_results_file}")
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze and compare experiment results"""
        print("Analyzing results...")
        
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            print("No successful experiments to analyze!")
            return {}
        
        # Create metrics comparison DataFrame
        metrics_data = []
        for result in successful_results:
            row = {
                'experiment': result['experiment_name'],
                'algorithm': result['algorithm'],
                'distance': result['distance_metric'],
                'preprocessing': result['preprocessing'],
                'time': result['total_time']
            }
            row.update(result['metrics'])
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Statistical analysis
        analysis = {
            'summary_stats': metrics_df.describe(),
            'best_experiments': {},
            'algorithm_comparison': {},
            'preprocessing_comparison': {},
            'distance_comparison': {}
        }
        
        # Find best experiments for each metric
        key_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        for metric in key_metrics:
            if metric in metrics_df.columns:
                if metric == 'davies_bouldin_score':
                    # Lower is better for Davies-Bouldin
                    best_idx = metrics_df[metric].idxmin()
                else:
                    # Higher is better for others
                    best_idx = metrics_df[metric].idxmax()
                
                best_exp = metrics_df.loc[best_idx]
                analysis['best_experiments'][metric] = {
                    'experiment': best_exp['experiment'],
                    'algorithm': best_exp['algorithm'],
                    'value': best_exp[metric]
                }
        
        # Algorithm comparison
        algo_stats = metrics_df.groupby('algorithm')[key_metrics].agg(['mean', 'std', 'count'])
        analysis['algorithm_comparison'] = algo_stats.to_dict()
        
        # Preprocessing comparison
        prep_stats = metrics_df.groupby('preprocessing')[key_metrics].agg(['mean', 'std', 'count'])
        analysis['preprocessing_comparison'] = prep_stats.to_dict()
        
        # Distance metric comparison
        dist_stats = metrics_df.groupby('distance')[key_metrics].agg(['mean', 'std', 'count'])
        analysis['distance_comparison'] = dist_stats.to_dict()
        
        # Save analysis
        analysis_file = self.results_dir / 'results_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save comparison tables
        comparison_file = self.results_dir / 'comparison_tables.xlsx'
        with pd.ExcelWriter(comparison_file, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='All_Results', index=False)
            algo_stats.to_excel(writer, sheet_name='Algorithm_Comparison')
            prep_stats.to_excel(writer, sheet_name='Preprocessing_Comparison')
            dist_stats.to_excel(writer, sheet_name='Distance_Comparison')
        
        print(f"Analysis saved to: {analysis_file}")
        print(f"Comparison tables saved to: {comparison_file}")
        
        return analysis
    
    def create_visualizations(self, results: List[Dict], analysis: Dict):
        """Create comprehensive result visualizations"""
        print("Creating visualizations...")
        
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            return
        
        # Prepare data
        metrics_data = []
        for result in successful_results:
            row = {
                'experiment': result['experiment_name'],
                'algorithm': result['algorithm'],
                'distance': result['distance_metric'],
                'preprocessing': result['preprocessing'],
                'time': result['total_time']
            }
            row.update(result['metrics'])
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Algorithm performance comparison
        ax1 = plt.subplot(3, 3, 1)
        if 'silhouette_score' in df.columns:
            algo_sil = df.groupby('algorithm')['silhouette_score'].mean().sort_values(ascending=False)
            algo_sil.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Average Silhouette Score by Algorithm')
            ax1.set_ylabel('Silhouette Score')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Preprocessing method comparison
        ax2 = plt.subplot(3, 3, 2)
        if 'silhouette_score' in df.columns:
            prep_sil = df.groupby('preprocessing')['silhouette_score'].mean().sort_values(ascending=False)
            prep_sil.plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title('Average Silhouette Score by Preprocessing')
            ax2.set_ylabel('Silhouette Score')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Distance metric comparison
        ax3 = plt.subplot(3, 3, 3)
        if 'silhouette_score' in df.columns:
            dist_sil = df.groupby('distance')['silhouette_score'].mean().sort_values(ascending=False)
            dist_sil.plot(kind='bar', ax=ax3, color='salmon')
            ax3.set_title('Average Silhouette Score by Distance Metric')
            ax3.set_ylabel('Silhouette Score')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Execution time comparison
        ax4 = plt.subplot(3, 3, 4)
        time_by_algo = df.groupby('algorithm')['time'].mean().sort_values()
        time_by_algo.plot(kind='bar', ax=ax4, color='orange')
        ax4.set_title('Average Execution Time by Algorithm')
        ax4.set_ylabel('Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Number of clusters distribution
        ax5 = plt.subplot(3, 3, 5)
        if 'n_clusters' in df.columns:
            df['n_clusters'].hist(bins=20, ax=ax5, color='purple', alpha=0.7)
            ax5.set_title('Distribution of Number of Clusters')
            ax5.set_xlabel('Number of Clusters')
            ax5.set_ylabel('Frequency')
        
        # 6. Silhouette vs Calinski-Harabasz scatter
        ax6 = plt.subplot(3, 3, 6)
        if 'silhouette_score' in df.columns and 'calinski_harabasz_score' in df.columns:
            scatter = ax6.scatter(df['silhouette_score'], df['calinski_harabasz_score'], 
                                c=df['n_clusters'], cmap='viridis', alpha=0.7)
            ax6.set_xlabel('Silhouette Score')
            ax6.set_ylabel('Calinski-Harabasz Score')
            ax6.set_title('Silhouette vs Calinski-Harabasz')
            plt.colorbar(scatter, ax=ax6, label='Number of Clusters')
        
        # 7. Noise ratio by algorithm
        ax7 = plt.subplot(3, 3, 7)
        if 'noise_ratio' in df.columns:
            noise_by_algo = df.groupby('algorithm')['noise_ratio'].mean().sort_values()
            noise_by_algo.plot(kind='bar', ax=ax7, color='red', alpha=0.7)
            ax7.set_title('Average Noise Ratio by Algorithm')
            ax7.set_ylabel('Noise Ratio')
            ax7.tick_params(axis='x', rotation=45)
        
        # 8. Algorithm performance heatmap
        ax8 = plt.subplot(3, 3, 8)
        key_metrics = ['silhouette_score', 'calinski_harabasz_score', 'noise_ratio']
        available_metrics = [m for m in key_metrics if m in df.columns]
        if available_metrics:
            heatmap_data = df.groupby('algorithm')[available_metrics].mean()
            sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlBu_r', ax=ax8, fmt='.3f')
            ax8.set_title('Algorithm Performance Heatmap')
        
        # 9. Top experiments ranking
        ax9 = plt.subplot(3, 3, 9)
        if 'silhouette_score' in df.columns:
            top_10 = df.nlargest(10, 'silhouette_score')
            y_pos = np.arange(len(top_10))
            ax9.barh(y_pos, top_10['silhouette_score'], color='gold')
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels([f"{alg}_{prep}" for alg, prep in 
                                zip(top_10['algorithm'], top_10['preprocessing'])], fontsize=8)
            ax9.set_xlabel('Silhouette Score')
            ax9.set_title('Top 10 Experiments')
        
        plt.tight_layout()
        
        # Save figure
        viz_file = self.results_dir / 'experiment_visualizations.png'
        plt.savefig(viz_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        # Create detailed algorithm comparison plot
        self._create_detailed_comparison(df)
        
        print(f"Visualizations saved to: {viz_file}")
    
    def _create_detailed_comparison(self, df: pd.DataFrame):
        """Create detailed comparison plots for top performing methods"""
        
        # Select top performing experiments based on silhouette score
        if 'silhouette_score' not in df.columns:
            return
        
        top_experiments = df.nlargest(15, 'silhouette_score')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance radar chart for top methods
        ax1 = axes[0, 0]
        metrics = ['silhouette_score', 'calinski_harabasz_score', 'clustered_ratio']
        available_metrics = [m for m in metrics if m in top_experiments.columns]
        
        if len(available_metrics) >= 2:
            # Normalize metrics for radar chart
            normalized_data = top_experiments[available_metrics].copy()
            for col in available_metrics:
                if col == 'davies_bouldin_score':
                    # Invert Davies-Bouldin (lower is better)
                    normalized_data[col] = 1 / (1 + normalized_data[col])
                
                # Min-max normalization
                min_val, max_val = normalized_data[col].min(), normalized_data[col].max()
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
            
            # Plot top 5 as lines
            for i, (idx, row) in enumerate(top_experiments.head(5).iterrows()):
                values = normalized_data.loc[idx, available_metrics].values
                ax1.plot(available_metrics, values, 'o-', label=f"{row['algorithm']}_{row['preprocessing']}")
            
            ax1.set_title('Top 5 Methods Performance Profile')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Time vs Performance trade-off
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['time'], df['silhouette_score'], 
                            c=df['n_clusters'], alpha=0.7, cmap='viridis')
        ax2.set_xlabel('Execution Time (seconds)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Performance vs Time Trade-off')
        ax2.set_xscale('log')
        plt.colorbar(scatter, ax=ax2, label='Number of Clusters')
        
        # 3. Statistical significance test results
        ax3 = axes[1, 0]
        # Group by algorithm and compare silhouette scores
        algorithms = df['algorithm'].unique()
        if len(algorithms) > 1:
            algorithm_scores = []
            algorithm_names = []
            for algo in algorithms:
                scores = df[df['algorithm'] == algo]['silhouette_score'].values
                if len(scores) > 0:
                    algorithm_scores.append(scores)
                    algorithm_names.append(algo)
            
            # Create box plot
            ax3.boxplot(algorithm_scores, labels=algorithm_names)
            ax3.set_title('Silhouette Score Distribution by Algorithm')
            ax3.set_ylabel('Silhouette Score')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Preprocessing impact analysis
        ax4 = axes[1, 1]
        preprocessing_methods = df['preprocessing'].unique()
        if len(preprocessing_methods) > 1:
            prep_scores = []
            prep_names = []
            for prep in preprocessing_methods:
                scores = df[df['preprocessing'] == prep]['silhouette_score'].values
                if len(scores) > 0:
                    prep_scores.append(scores)
                    prep_names.append(prep)
            
            ax4.boxplot(prep_scores, labels=prep_names)
            ax4.set_title('Silhouette Score Distribution by Preprocessing')
            ax4.set_ylabel('Silhouette Score')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        detailed_viz_file = self.results_dir / 'detailed_comparison.png'
        plt.savefig(detailed_viz_file, dpi=200, bbox_inches='tight')
        plt.close()
    
    def statistical_significance_testing(self, results: List[Dict]) -> Dict:
        """Perform statistical significance testing between methods"""
        print("Performing statistical significance testing...")
        
        successful_results = [r for r in results if r['success']]
        if len(successful_results) < 3:
            print("Need at least 3 successful experiments for statistical testing")
            return {}
        
        # Prepare data
        metrics_data = []
        for result in successful_results:
            row = {
                'algorithm': result['algorithm'],
                'preprocessing': result['preprocessing'],
                'distance': result['distance_metric']
            }
            row.update(result['metrics'])
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        statistical_results = {}
        
        # Test algorithms
        if 'silhouette_score' in df.columns:
            algorithms = df['algorithm'].unique()
            if len(algorithms) > 2:
                algorithm_groups = [df[df['algorithm'] == algo]['silhouette_score'].values 
                                  for algo in algorithms]
                
                # Remove empty groups
                algorithm_groups = [group for group in algorithm_groups if len(group) > 0]
                
                if len(algorithm_groups) > 2:
                    try:
                        # Kruskal-Wallis test (non-parametric ANOVA)
                        statistic, p_value = stats.kruskal(*algorithm_groups)
                        statistical_results['algorithm_comparison'] = {
                            'test': 'kruskal_wallis',
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'groups': list(algorithms)
                        }
                        
                        # Pairwise comparisons if significant
                        if p_value < 0.05:
                            pairwise_results = []
                            for i, algo1 in enumerate(algorithms):
                                for j, algo2 in enumerate(algorithms):
                                    if i < j:
                                        group1 = df[df['algorithm'] == algo1]['silhouette_score'].values
                                        group2 = df[df['algorithm'] == algo2]['silhouette_score'].values
                                        
                                        if len(group1) > 0 and len(group2) > 0:
                                            try:
                                                stat, pval = stats.mannwhitneyu(group1, group2)
                                                pairwise_results.append({
                                                    'group1': algo1,
                                                    'group2': algo2,
                                                    'statistic': stat,
                                                    'p_value': pval,
                                                    'significant': pval < 0.05
                                                })
                                            except:
                                                pass
                            
                            statistical_results['algorithm_pairwise'] = pairwise_results
                    except Exception as e:
                        print(f"Statistical testing failed: {e}")
        
        # Save statistical results
        if statistical_results:
            stats_file = self.results_dir / 'statistical_analysis.json'
            with open(stats_file, 'w') as f:
                json.dump(statistical_results, f, indent=2, default=str)
            print(f"Statistical analysis saved to: {stats_file}")
        
        return statistical_results
    
    def generate_report(self, results: List[Dict], analysis: Dict, 
                       statistical_results: Dict) -> str:
        """Generate comprehensive experiment report"""
        print("Generating comprehensive report...")
        
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        report_lines = []
        report_lines.append("# Clustering Algorithm Comparison Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Experiment ID: {self.experiment_id}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append(f"- Total experiments: {len(results)}")
        report_lines.append(f"- Successful experiments: {len(successful_results)}")
        report_lines.append(f"- Failed experiments: {len(failed_results)}")
        
        if successful_results:
            # Find best performing experiment
            best_result = max(successful_results, 
                            key=lambda x: x['metrics'].get('silhouette_score', -1))
            report_lines.append(f"- Best performing method: {best_result['experiment_name']}")
            report_lines.append(f"  - Algorithm: {best_result['algorithm']}")
            report_lines.append(f"  - Distance metric: {best_result['distance_metric']}")
            report_lines.append(f"  - Preprocessing: {best_result['preprocessing']}")
            report_lines.append(f"  - Silhouette score: {best_result['metrics'].get('silhouette_score', 'N/A'):.3f}")
        
        report_lines.append("")
        
        # Detailed Results
        if 'best_experiments' in analysis:
            report_lines.append("## Best Performing Methods by Metric")
            for metric, info in analysis['best_experiments'].items():
                report_lines.append(f"- **{metric}**: {info['experiment']} ({info['algorithm']}) - {info['value']:.3f}")
            report_lines.append("")
        
        # Algorithm Comparison
        if successful_results:
            report_lines.append("## Algorithm Performance Summary")
            algorithms = {}
            for result in successful_results:
                algo = result['algorithm']
                if algo not in algorithms:
                    algorithms[algo] = {'scores': [], 'times': [], 'experiments': []}
                
                algorithms[algo]['scores'].append(result['metrics'].get('silhouette_score', -1))
                algorithms[algo]['times'].append(result['total_time'])
                algorithms[algo]['experiments'].append(result['experiment_name'])
            
            for algo, data in algorithms.items():
                valid_scores = [s for s in data['scores'] if s > -1]
                if valid_scores:
                    avg_score = np.mean(valid_scores)
                    avg_time = np.mean(data['times'])
                    report_lines.append(f"- **{algo}**:")
                    report_lines.append(f"  - Average silhouette score: {avg_score:.3f}")
                    report_lines.append(f"  - Average execution time: {avg_time:.2f}s")
                    report_lines.append(f"  - Number of experiments: {len(data['experiments'])}")
        
        report_lines.append("")
        
        # Statistical Significance
        if statistical_results:
            report_lines.append("## Statistical Significance Testing")
            if 'algorithm_comparison' in statistical_results:
                test_result = statistical_results['algorithm_comparison']
                report_lines.append(f"- Kruskal-Wallis test p-value: {test_result['p_value']:.4f}")
                if test_result['significant']:
                    report_lines.append("- **Significant differences found between algorithms**")
                else:
                    report_lines.append("- No significant differences found between algorithms")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        if successful_results:
            # Top 3 methods
            sorted_results = sorted(successful_results, 
                                  key=lambda x: x['metrics'].get('silhouette_score', -1), 
                                  reverse=True)[:3]
            
            report_lines.append("### Top 3 Recommended Methods:")
            for i, result in enumerate(sorted_results, 1):
                score = result['metrics'].get('silhouette_score', -1)
                time = result['total_time']
                report_lines.append(f"{i}. **{result['experiment_name']}**")
                report_lines.append(f"   - Silhouette score: {score:.3f}")
                report_lines.append(f"   - Execution time: {time:.2f}s")
                report_lines.append(f"   - Configuration: {result['algorithm']} + {result['preprocessing']} + {result['distance_metric']}")
        
        # Failed Experiments
        if failed_results:
            report_lines.append("## Failed Experiments")
            for result in failed_results:
                report_lines.append(f"- **{result['experiment_name']}**: {result['error']}")
        
        report_lines.append("")
        
        # Files Generated
        report_lines.append("## Generated Files")
        report_lines.append("- `experiment_results.json`: Detailed results summary")
        report_lines.append("- `experiment_metrics.csv`: Metrics comparison table")
        report_lines.append("- `complete_results.pkl`: Complete results with clustering labels")
        report_lines.append("- `results_analysis.json`: Statistical analysis")
        report_lines.append("- `comparison_tables.xlsx`: Comparison tables for algorithms, preprocessing, distances")
        report_lines.append("- `experiment_visualizations.png`: Comprehensive visualization dashboard")
        report_lines.append("- `detailed_comparison.png`: Detailed method comparison plots")
        if statistical_results:
            report_lines.append("- `statistical_analysis.json`: Statistical significance testing results")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.results_dir / 'experiment_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to: {report_file}")
        return report_content


def load_experiment_config(config_path: Path) -> Dict:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config() -> Dict:
    """Create a comprehensive default configuration for experiments"""
    config = {
        'data': {
            'path': 'paper_clusters.csv',
            'vector_prefix': 'cluster_',
            'true_labels_column': None  # Optional column with ground truth labels
        },
        'experiments': [
            # K-means variations
            {
                'name': 'kmeans_euclidean_standard',
                'algorithm': 'kmeans',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'n_clusters': 8}
            },
            {
                'name': 'kmeans_cosine_l2norm',
                'algorithm': 'kmeans',
                'distance_metric': 'cosine',
                'preprocessing': 'l2_normalize',
                'params': {'n_clusters': 8}
            },
            {
                'name': 'kmeans_euclidean_tfidf',
                'algorithm': 'kmeans',
                'distance_metric': 'euclidean',
                'preprocessing': 'tfidf',
                'params': {'n_clusters': 8}
            },
            
            # HDBSCAN variations
            {
                'name': 'hdbscan_euclidean_standard',
                'algorithm': 'hdbscan',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'min_cluster_size': 10}
            },
            {
                'name': 'hdbscan_cosine_l2norm',
                'algorithm': 'hdbscan',
                'distance_metric': 'cosine',
                'preprocessing': 'l2_normalize',
                'params': {'min_cluster_size': 10}
            },
            {
                'name': 'hdbscan_manhattan_robust',
                'algorithm': 'hdbscan',
                'distance_metric': 'manhattan',
                'preprocessing': 'robust',
                'params': {'min_cluster_size': 15}
            },
            
            # Hierarchical clustering
            {
                'name': 'hierarchical_euclidean_standard',
                'algorithm': 'hierarchical',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'n_clusters': 8, 'linkage': 'ward'}
            },
            {
                'name': 'hierarchical_cosine_l2norm',
                'algorithm': 'hierarchical',
                'distance_metric': 'cosine',
                'preprocessing': 'l2_normalize',
                'params': {'n_clusters': 8, 'linkage': 'average'}
            },
            
            # Spectral clustering
            {
                'name': 'spectral_rbf_standard',
                'algorithm': 'spectral',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'n_clusters': 8, 'affinity': 'rbf'}
            },
            {
                'name': 'spectral_cosine_l2norm',
                'algorithm': 'spectral',
                'distance_metric': 'cosine',
                'preprocessing': 'l2_normalize',
                'params': {'n_clusters': 8}
            },
            
            # With dimensionality reduction
            {
                'name': 'kmeans_euclidean_pca',
                'algorithm': 'kmeans',
                'distance_metric': 'euclidean',
                'preprocessing': 'pca',
                'preprocessing_params': {'n_components': 20},
                'params': {'n_clusters': 8}
            },
            {
                'name': 'hdbscan_euclidean_svd',
                'algorithm': 'hdbscan',
                'distance_metric': 'euclidean',
                'preprocessing': 'svd',
                'preprocessing_params': {'n_components': 30},
                'params': {'min_cluster_size': 10}
            },
            {
                'name': 'kmeans_euclidean_nmf',
                'algorithm': 'kmeans',
                'distance_metric': 'euclidean',
                'preprocessing': 'nmf',
                'preprocessing_params': {'n_components': 15},
                'params': {'n_clusters': 8}
            },
            
            # DBSCAN
            {
                'name': 'dbscan_euclidean_standard',
                'algorithm': 'dbscan',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'eps': 0.5, 'min_samples': 5}
            },
            {
                'name': 'dbscan_cosine_l2norm',
                'algorithm': 'dbscan',
                'distance_metric': 'cosine',
                'preprocessing': 'l2_normalize',
                'params': {'eps': 0.3, 'min_samples': 5}
            },
            
            # Gaussian Mixture Model
            {
                'name': 'gmm_standard',
                'algorithm': 'gmm',
                'distance_metric': 'euclidean',
                'preprocessing': 'standard',
                'params': {'n_components': 8, 'covariance_type': 'full'}
            },
            {
                'name': 'gmm_pca',
                'algorithm': 'gmm',
                'distance_metric': 'euclidean',
                'preprocessing': 'pca',
                'preprocessing_params': {'n_components': 20},
                'params': {'n_components': 8, 'covariance_type': 'tied'}
            }
        ]
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Comprehensive clustering experimentation framework")
    parser.add_argument('--config', type=str, help='Path to experiment configuration YAML file')
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory for results')
    parser.add_argument('--data_path', type=str, help='Path to input data CSV (overrides config)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (use 1 to avoid compatibility issues)')
    parser.add_argument('--create_default_config', action='store_true', 
                       help='Create default configuration file and exit')
    parser.add_argument('--create_safe_config', action='store_true',
                       help='Create safe configuration (works with older package versions)')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    
    # Create configuration files if requested
    if args.create_default_config or args.create_safe_config:
        work_dir.mkdir(parents=True, exist_ok=True)
        
        if args.create_safe_config:
            config = create_safe_config()
            config_file = work_dir / 'safe_experiment_config.yaml'
            config_type = "safe"
        else:
            config = create_default_config()
            config_file = work_dir / 'default_experiment_config.yaml'
            config_type = "default"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Created {config_type} configuration: {config_file}")
        
        if args.create_safe_config:
            print("\nThis safe configuration avoids algorithms that may have version compatibility issues.")
            print("It includes reliable methods that work across different package versions.")
        else:
            print("\nThis default configuration includes comprehensive experiments.")
            print("If you encounter version errors, try --create_safe_config instead.")
            
        print(f"\nEdit this file to customize your experiments, then run:")
        print(f"python clustering_experiments.py --config {config_file} --work_dir {args.work_dir}")
        return
    
    # Load configuration
    if args.config:
        config = load_experiment_config(Path(args.config))
    else:
        print("Using default configuration...")
        config = create_default_config()
    
    # Override data path if provided
    if args.data_path:
        config['data']['path'] = args.data_path
    
    # Initialize framework
    framework = ClusteringExperimentFramework(work_dir, n_jobs=args.n_jobs)
    
    # Load data
    try:
        data_path = Path(config['data']['path'])
        X, df = framework.load_data(data_path, config['data']['vector_prefix'])
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the data file exists and contains the expected columns.")
        return
    
    # Load true labels if specified
    true_labels = None
    if config['data'].get('true_labels_column') and config['data']['true_labels_column'] in df.columns:
        true_labels = df[config['data']['true_labels_column']].values
        print(f"Loaded ground truth labels from column: {config['data']['true_labels_column']}")
    
    # Run experiments
    print(f"\nStarting {len(config['experiments'])} experiments...")
    print("="*80)
    
    results = framework.run_experiments(X, df, config['experiments'], true_labels)
    
    # Analyze results
    analysis = framework.analyze_results(results)
    
    # Statistical testing
    statistical_results = framework.statistical_significance_testing(results)
    
    # Create visualizations
    framework.create_visualizations(results, analysis)
    
    # Generate report
    report = framework.generate_report(results, analysis, statistical_results)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results directory: {framework.results_dir}")
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['metrics'].get('silhouette_score', -1))
        print(f"\nBest performing method: {best_result['experiment_name']}")
        print(f"  - Algorithm: {best_result['algorithm']}")
        print(f"  - Distance: {best_result['distance_metric']}")
        print(f"  - Preprocessing: {best_result['preprocessing']}")
        print(f"  - Silhouette score: {best_result['metrics'].get('silhouette_score', 'N/A'):.3f}")
        print(f"  - Execution time: {best_result['total_time']:.2f}s")
    
    print(f"\nView the complete report: {framework.results_dir / 'experiment_report.md'}")


if __name__ == "__main__":
    main()