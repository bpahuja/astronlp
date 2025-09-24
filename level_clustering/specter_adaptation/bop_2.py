#!/usr/bin/env python3
"""
End-to-end paper clustering analysis script.
Takes paragraph clustering results and creates paper-level analysis with visualizations.

Input: labels_paragraph.csv from clustering script
Output: Paper clusters, noise analysis, and 2D/3D visualizations

FIXED VERSION: Handles large number of clusters in visualizations
"""

import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from collections import Counter

# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')
plt.style.use('default')


class PaperClusteringAnalyzer:
    """Complete paper-level clustering analysis from paragraph results"""
    
    def __init__(self, work_dir: Path, n_jobs: Optional[int] = None):
        self.work_dir = Path(work_dir)
        self.results = {}
        
        # Set n_jobs: None means use all available cores
        if n_jobs is None or n_jobs == -1:
            import multiprocessing
            self.n_jobs = multiprocessing.cpu_count()
        elif n_jobs > 0:
            self.n_jobs = n_jobs
        else:
            self.n_jobs = 1
    
    def load_paragraph_results(self, labels_file: Path) -> pd.DataFrame:
        """Load paragraph clustering results"""
        print(f"Loading paragraph results from {labels_file}")
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        df = pd.read_csv(labels_file)
        
        # Validate required columns
        required_cols = ['para_id', 'paper_id', 'cluster']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"Loaded {len(df):,} paragraph labels")
        print(f"Unique papers: {df['paper_id'].nunique():,}")
        print(f"Unique clusters: {df['cluster'].nunique():,}")
        print(f"Noise paragraphs (cluster=-1): {(df['cluster'] == -1).sum():,}")
        
        return df
    
    def _process_single_paper(self, paper_data: Tuple[str, pd.DataFrame], 
                             clusters_for_vectors: List[int], 
                             exclude_noise: bool) -> Tuple[np.ndarray, Dict]:
        """Process a single paper to create its vector and metadata
        
        This is a helper function for parallel processing.
        """
        paper_id, paper_paragraphs = paper_data
        
        # Count paragraphs in each cluster
        cluster_counts = Counter(paper_paragraphs['cluster'])
        total_paragraphs = len(paper_paragraphs)
        noise_count = cluster_counts.get(-1, 0)
        
        # Calculate denominator for normalization
        if exclude_noise:
            # Normalize by non-noise paragraphs only
            non_noise_paragraphs = total_paragraphs - noise_count
            normalization_factor = max(non_noise_paragraphs, 1)  # Avoid division by zero
        else:
            # Normalize by all paragraphs
            normalization_factor = total_paragraphs
        
        # Create normalized vector (frequencies)
        vector = np.zeros(len(clusters_for_vectors))
        for i, cluster_id in enumerate(clusters_for_vectors):
            vector[i] = cluster_counts.get(cluster_id, 0) / normalization_factor
        
        # Store metadata
        metadata = {
            'paper_id': paper_id,
            'total_paragraphs': total_paragraphs,
            'noise_paragraphs': noise_count,
            'noise_ratio': noise_count / total_paragraphs,
            'unique_clusters': len([c for c in cluster_counts.keys() if c != -1]),
            'non_noise_paragraphs': total_paragraphs - noise_count
        }
        
        return vector, metadata
    
    def create_paper_vectors(self, df: pd.DataFrame, exclude_noise: bool = False) -> pd.DataFrame:
        """Create bag-of-clusters vectors for each paper using parallel processing
        
        Args:
            df: DataFrame with paragraph clustering results
            exclude_noise: If True, exclude noise cluster (-1) from paper vectors
        """
        print("Creating paper vectors from paragraph clusters...")
        print(f"Using {self.n_jobs} parallel jobs for processing")
        
        # Get all unique clusters
        all_clusters = sorted(df['cluster'].unique())
        
        # Optionally exclude noise cluster
        if exclude_noise and -1 in all_clusters:
            clusters_for_vectors = [c for c in all_clusters if c != -1]
            print(f"Excluding noise cluster (-1) from paper vectors")
        else:
            clusters_for_vectors = all_clusters
            
        n_clusters = len(clusters_for_vectors)
        
        print(f"Creating vectors with {n_clusters} dimensions")
        if exclude_noise:
            print(f"Note: Noise paragraphs will be ignored in vector creation")
        
        # OPTIMIZED: Use groupby for efficient grouping (O(n) instead of O(n*m))
        print("Grouping paragraphs by paper...")
        paper_groups = list(df.groupby('paper_id'))
        
        # Process papers in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_single_paper)(
                paper_data, clusters_for_vectors, exclude_noise
            ) 
            for paper_data in tqdm(paper_groups, desc="Processing papers")
        )
        
        # Unpack results
        paper_vectors = [r[0] for r in results]
        paper_metadata = [r[1] for r in results]
        
        # Create DataFrame
        paper_vectors = np.array(paper_vectors)
        metadata_df = pd.DataFrame(paper_metadata)
        
        # Add vector columns
        vector_cols = [f'cluster_{c}' for c in clusters_for_vectors]
        vector_df = pd.DataFrame(paper_vectors, columns=vector_cols)
        
        result_df = pd.concat([metadata_df, vector_df], axis=1)
        
        print(f"Created vectors for {len(result_df):,} papers")
        print(f"Vector dimensionality: {paper_vectors.shape[1]}")
        
        if exclude_noise:
            # Show impact of excluding noise
            papers_with_noise = (result_df['noise_paragraphs'] > 0).sum()
            print(f"Papers with noise paragraphs: {papers_with_noise:,}")
            avg_noise_ratio = result_df['noise_ratio'].mean()
            print(f"Average noise ratio: {avg_noise_ratio:.3f}")
        
        return result_df
    
    def analyze_noise_papers(self, paper_df: pd.DataFrame) -> Dict:
        """Analyze papers with high noise content"""
        print("Analyzing noise in papers...")
        
        # Define noise thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9, 1.0]
        noise_analysis = {}
        
        for threshold in thresholds:
            noisy_papers = paper_df[paper_df['noise_ratio'] >= threshold]
            noise_analysis[f'noise_ratio_{threshold}'] = {
                'count': len(noisy_papers),
                'percentage': len(noisy_papers) / len(paper_df) * 100,
                'paper_ids': noisy_papers['paper_id'].tolist()
            }
        
        # Fully noised papers (100% noise)
        fully_noised = paper_df[paper_df['noise_ratio'] == 1.0]
        
        # Statistics
        stats = {
            'total_papers': len(paper_df),
            'fully_noised_papers': len(fully_noised),
            'fully_noised_percentage': len(fully_noised) / len(paper_df) * 100,
            'avg_noise_ratio': paper_df['noise_ratio'].mean(),
            'median_noise_ratio': paper_df['noise_ratio'].median(),
            'noise_ratio_std': paper_df['noise_ratio'].std(),
            'threshold_analysis': noise_analysis
        }
        
        print(f"Noise Analysis Results:")
        print(f"  Total papers: {stats['total_papers']:,}")
        print(f"  Fully noised papers: {stats['fully_noised_papers']:,} ({stats['fully_noised_percentage']:.1f}%)")
        print(f"  Average noise ratio: {stats['avg_noise_ratio']:.3f}")
        print(f"  Papers with >50% noise: {noise_analysis['noise_ratio_0.5']['count']:,} ({noise_analysis['noise_ratio_0.5']['percentage']:.1f}%)")
        print(f"  Papers with >80% noise: {noise_analysis['noise_ratio_0.8']['count']:,} ({noise_analysis['noise_ratio_0.8']['percentage']:.1f}%)")
        
        return stats
    
    def cluster_papers(self, paper_df: pd.DataFrame, min_cluster_size: int = 10, 
                      exclude_noise: bool = False) -> Tuple[np.ndarray, Dict]:
        """Cluster papers based on their cluster vectors"""
        print("Clustering papers...")
        
        # Get vector columns (exclude metadata)
        vector_cols = [col for col in paper_df.columns if col.startswith('cluster_')]
        X = paper_df[vector_cols].values
        
        print(f"Clustering {len(X):,} papers with {len(vector_cols)} features")
        if exclude_noise:
            print("Note: Noise cluster was excluded from paper vectors")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Remove fully noised papers for clustering (they would dominate)
        # When exclude_noise=True, fully noised papers will have zero vectors
        if exclude_noise:
            # Papers with all paragraphs being noise will have zero vectors
            zero_vector_mask = np.all(X == 0, axis=1)  # Check original X, not scaled
            non_zero_mask = ~zero_vector_mask
            print(f"Excluding {zero_vector_mask.sum()} papers with zero vectors (fully noised)")
        else:
            non_zero_mask = paper_df['noise_ratio'] < 1.0
            print(f"Excluding {(~non_zero_mask).sum()} fully noised papers")
        
        X_clustering = X_scaled[non_zero_mask]
        
        print(f"Clustering {len(X_clustering):,} papers")
        
        # Cluster with HDBSCAN (uses parallel processing internally with n_jobs)
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(3, min_cluster_size // 3),
            metric='euclidean',
            cluster_selection_epsilon=0.1,
            n_jobs=self.n_jobs  # Use parallel processing for HDBSCAN
        )
        
        cluster_labels_subset = clusterer.fit_predict(X_clustering)
        
        # Map back to full dataset
        cluster_labels = np.full(len(paper_df), -2, dtype=int)  # -2 = excluded from clustering
        cluster_labels[non_zero_mask] = cluster_labels_subset
        
        # Calculate metrics on clustered papers only
        n_clusters = len(set(cluster_labels_subset)) - (1 if -1 in cluster_labels_subset else 0)
        n_noise = list(cluster_labels_subset).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_excluded': (~non_zero_mask).sum(),
            'noise_pct': n_noise / len(cluster_labels_subset) * 100 if len(cluster_labels_subset) > 0 else 0,
            'excluded_pct': (~non_zero_mask).sum() / len(paper_df) * 100,
            'exclude_noise_used': exclude_noise
        }
        
        if n_clusters > 1 and len(X_clustering) > 1:
            # Only calculate silhouette score if we have valid clusters
            valid_labels = cluster_labels_subset[cluster_labels_subset != -1]
            if len(np.unique(valid_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(X_clustering, cluster_labels_subset)
            else:
                metrics['silhouette_score'] = -1
        else:
            metrics['silhouette_score'] = -1
        
        print(f"Paper Clustering Results:")
        print(f"  Clusters: {n_clusters}")
        print(f"  Noise papers: {n_noise} ({metrics['noise_pct']:.1f}%)")
        print(f"  Excluded papers: {metrics['n_excluded']} ({metrics['excluded_pct']:.1f}%)")
        if metrics['silhouette_score'] > -1:
            print(f"  Silhouette score: {metrics['silhouette_score']:.3f}")
        
        return cluster_labels, metrics
    
    def create_2d_visualization(self, paper_df: pd.DataFrame, cluster_labels: np.ndarray, 
                               method: str = 'tsne') -> Tuple[np.ndarray, plt.Figure]:
        """Create 2D visualization of paper clusters with smart legend handling"""
        print(f"Creating 2D visualization using {method.upper()}...")
        
        # Get vector data
        vector_cols = [col for col in paper_df.columns if col.startswith('cluster_')]
        X = paper_df[vector_cols].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality reduction
        if method.lower() == 'tsne':
            # Use parallel processing for t-SNE
            reducer = TSNE(n_components=2, random_state=42, 
                          perplexity=min(30, len(X)//4), n_jobs=self.n_jobs)
            X_2d = reducer.fit_transform(X_scaled)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X_scaled)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        # Create plot with constrained figure size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Colored by cluster
        unique_labels = np.unique(cluster_labels)
        n_unique = len(unique_labels)
        
        # Use a colormap that can handle many clusters
        if n_unique <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_unique, 20)))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, n_unique))
        
        # Handle legend intelligently based on number of clusters
        show_legend = n_unique <= 25  # Only show legend if manageable number of clusters
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points
                mask = cluster_labels == label
                ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', alpha=0.5, s=20, 
                           label='Noise' if show_legend else '')
            elif label == -2:
                # Excluded papers
                mask = cluster_labels == label
                ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c='gray', alpha=0.3, s=15, 
                           label='Excluded' if show_legend else '')
            else:
                # Regular clusters
                mask = cluster_labels == label
                color = colors[i % len(colors)]
                ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], alpha=0.7, s=30, 
                           label=f'Cluster {label}' if show_legend else '')
        
        ax1.set_title(f'Paper Clusters ({method.upper()})')
        ax1.set_xlabel(f'{method.upper()}-1')
        ax1.set_ylabel(f'{method.upper()}-2')
        ax1.grid(True, alpha=0.3)
        
        # Only add legend if we have a manageable number of clusters
        if show_legend:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        else:
            # Add text annotation about number of clusters
            ax1.text(0.02, 0.98, f'{n_unique} clusters\n(legend omitted)', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Colored by noise ratio
        noise_ratios = paper_df['noise_ratio'].values
        scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=noise_ratios, cmap='Reds', alpha=0.7, s=30)
        ax2.set_title('Papers by Noise Ratio')
        ax2.set_xlabel(f'{method.upper()}-1')
        ax2.set_ylabel(f'{method.upper()}-2')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Noise Ratio')
        
        plt.tight_layout()
        
        return X_2d, fig
    
    def create_3d_visualization(self, paper_df: pd.DataFrame, cluster_labels: np.ndarray,
                               method: str = 'pca') -> Tuple[np.ndarray, plt.Figure]:
        """Create 3D visualization of paper clusters with smart legend handling"""
        print(f"Creating 3D visualization using {method.upper()}...")
        
        # Get vector data
        vector_cols = [col for col in paper_df.columns if col.startswith('cluster_')]
        X = paper_df[vector_cols].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=3, random_state=42)
            X_3d = reducer.fit_transform(X_scaled)
            explained_var = reducer.explained_variance_ratio_
            print(f"PCA explained variance: {explained_var.sum():.3f} ({explained_var})")
        else:
            # For other methods, use PCA as fallback since t-SNE 3D is computationally expensive
            print("Using PCA for 3D visualization (recommended)")
            reducer = PCA(n_components=3, random_state=42)
            X_3d = reducer.fit_transform(X_scaled)
            explained_var = reducer.explained_variance_ratio_
        
        # Create 3D plot with constrained size
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by cluster
        unique_labels = np.unique(cluster_labels)
        n_unique = len(unique_labels)
        
        # Use appropriate colormap
        if n_unique <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_unique, 20)))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, n_unique))
        
        # Handle legend intelligently
        show_legend = n_unique <= 15  # Even more restrictive for 3D plots
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            color = colors[i % len(colors)]
            
            if label == -1:
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
                          c='black', alpha=0.4, s=15, label='Noise' if show_legend else '')
            elif label == -2:
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
                          c='gray', alpha=0.2, s=10, label='Excluded' if show_legend else '')
            else:
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
                          c=[color], alpha=0.7, s=20, label=f'Cluster {label}' if show_legend else '')
        
        ax.set_title('3D Paper Clusters (PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        if show_legend:
            ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize='small')
        else:
            # Add text annotation
            ax.text2D(0.02, 0.98, f'{n_unique} clusters\n(legend omitted)', 
                     transform=ax.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return X_3d, fig
    
    def create_summary_plots(self, paper_df: pd.DataFrame, noise_stats: Dict) -> plt.Figure:
        """Create summary statistics plots"""
        print("Creating summary plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Noise ratio distribution
        axes[0, 0].hist(paper_df['noise_ratio'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(paper_df['noise_ratio'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {paper_df["noise_ratio"].mean():.3f}')
        axes[0, 0].set_xlabel('Noise Ratio')
        axes[0, 0].set_ylabel('Number of Papers')
        axes[0, 0].set_title('Distribution of Noise Ratios')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Papers per paragraph count
        axes[0, 1].hist(paper_df['total_paragraphs'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Paragraphs per Paper')
        axes[0, 1].set_ylabel('Number of Papers')
        axes[0, 1].set_title('Distribution of Paper Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Unique clusters per paper
        axes[1, 0].hist(paper_df['unique_clusters'], bins=range(0, paper_df['unique_clusters'].max()+2), 
                       alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Unique Clusters per Paper')
        axes[1, 0].set_ylabel('Number of Papers')
        axes[1, 0].set_title('Distribution of Cluster Diversity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Noise threshold analysis
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        available_thresholds = [t for t in thresholds if f'noise_ratio_{t}' in noise_stats['threshold_analysis']]
        percentages = [noise_stats['threshold_analysis'][f'noise_ratio_{t}']['percentage'] for t in available_thresholds]
        
        axes[1, 1].bar(available_thresholds, percentages, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 1].set_xlabel('Noise Ratio Threshold')
        axes[1, 1].set_ylabel('Percentage of Papers')
        axes[1, 1].set_title('Papers Above Noise Thresholds')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def run_complete_analysis(self, labels_file: Path, min_cluster_size: int = 10, 
                            exclude_noise: bool = False) -> Dict:
        """Run complete end-to-end analysis"""
        print("="*60)
        print("PAPER CLUSTERING ANALYSIS")
        print("="*60)
        print(f"Using {self.n_jobs} parallel jobs for processing")
        
        # Load paragraph results
        paragraph_df = self.load_paragraph_results(labels_file)
        
        # Create paper vectors (uses parallelization)
        paper_df = self.create_paper_vectors(paragraph_df, exclude_noise=exclude_noise)
        
        # Analyze noise
        noise_stats = self.analyze_noise_papers(paper_df)
        
        # Cluster papers
        cluster_labels, cluster_metrics = self.cluster_papers(paper_df, min_cluster_size, exclude_noise=exclude_noise)
        
        # Add cluster labels to dataframe
        paper_df['paper_cluster'] = cluster_labels
        
        # Create visualizations with error handling
        try:
            X_2d_tsne, fig_2d_tsne = self.create_2d_visualization(paper_df, cluster_labels, 'tsne')
        except Exception as e:
            print(f"Warning: t-SNE 2D visualization failed: {e}")
            X_2d_tsne, fig_2d_tsne = None, None
        
        try:
            X_2d_pca, fig_2d_pca = self.create_2d_visualization(paper_df, cluster_labels, 'pca')
        except Exception as e:
            print(f"Warning: PCA 2D visualization failed: {e}")
            X_2d_pca, fig_2d_pca = None, None
        
        try:
            X_3d, fig_3d = self.create_3d_visualization(paper_df, cluster_labels, 'pca')
        except Exception as e:
            print(f"Warning: 3D visualization failed: {e}")
            X_3d, fig_3d = None, None
        
        try:
            fig_summary = self.create_summary_plots(paper_df, noise_stats)
        except Exception as e:
            print(f"Warning: Summary plots failed: {e}")
            fig_summary = None
        
        # Save results
        figures = [fig for fig in [fig_2d_tsne, fig_2d_pca, fig_3d, fig_summary] if fig is not None]
        self.save_results(paper_df, noise_stats, cluster_metrics, figures, exclude_noise)
        
        # Compile final results
        results = {
            'paper_df': paper_df,
            'noise_analysis': noise_stats,
            'cluster_metrics': cluster_metrics,
            'visualizations': {
                '2d_tsne': X_2d_tsne,
                '2d_pca': X_2d_pca,
                '3d_pca': X_3d
            }
        }
        
        return results
    
    def save_results(self, paper_df: pd.DataFrame, noise_stats: Dict, 
                    cluster_metrics: Dict, figures: List[plt.Figure], exclude_noise: bool = False):
        """Save all results to files"""
        print("Saving results...")
        
        # Add suffix to filenames if noise was excluded
        suffix = "_no_noise" if exclude_noise else ""
        
        # Save paper dataframe
        paper_file = self.work_dir / f"paper_clusters{suffix}.csv"
        paper_df.to_csv(paper_file, index=False)
        print(f"Saved paper clusters to: {paper_file}")
        
        # Save noise analysis
        noise_file = self.work_dir / f"noise_analysis{suffix}.json"
        with open(noise_file, 'w') as f:
            json.dump(noise_stats, f, indent=2, default=str)  # default=str handles numpy types
        print(f"Saved noise analysis to: {noise_file}")
        
        # Save cluster metrics
        metrics_file = self.work_dir / f"paper_cluster_metrics{suffix}.json"
        with open(metrics_file, 'w') as f:
            json.dump(cluster_metrics, f, indent=2, default=str)
        print(f"Saved cluster metrics to: {metrics_file}")
        
        # Save figures with error handling
        figure_names = [f'2d_tsne_clusters{suffix}.png', f'2d_pca_clusters{suffix}.png', 
                       f'3d_pca_clusters{suffix}.png', f'summary_stats{suffix}.png']
        
        for fig, name in zip(figures, figure_names[:len(figures)]):
            if fig is not None:
                try:
                    fig_path = self.work_dir / name
                    fig.savefig(fig_path, dpi=200, bbox_inches='tight')  # Reduced DPI to avoid memory issues
                    print(f"Saved figure: {fig_path}")
                    plt.close(fig)  # Close figure to free memory
                except Exception as e:
                    print(f"Warning: Could not save {name}: {e}")
        
        # Save fully noised papers list
        if exclude_noise:
            # When excluding noise, save papers with zero vectors
            zero_vector_papers = paper_df[paper_df['noise_ratio'] == 1.0]['paper_id'].tolist()
            zero_vector_file = self.work_dir / f"zero_vector_papers{suffix}.txt"
            with open(zero_vector_file, 'w') as f:
                for paper_id in zero_vector_papers:
                    f.write(f"{paper_id}\n")
            print(f"Saved zero vector papers list to: {zero_vector_file}")
        else:
            fully_noised = paper_df[paper_df['noise_ratio'] == 1.0]['paper_id'].tolist()
            fully_noised_file = self.work_dir / f"fully_noised_papers{suffix}.txt"
            with open(fully_noised_file, 'w') as f:
                for paper_id in fully_noised:
                    f.write(f"{paper_id}\n")
            print(f"Saved fully noised papers list to: {fully_noised_file}")


def resume_from_existing_results(work_dir: Path, exclude_noise: bool = False):
    """Resume plotting from existing results files"""
    print("Resuming analysis from existing results...")
    
    suffix = "_no_noise" if exclude_noise else ""
    
    # Load existing results
    paper_file = work_dir / f"paper_clusters{suffix}.csv"
    noise_file = work_dir / f"noise_analysis{suffix}.json"
    
    if not paper_file.exists():
        raise FileNotFoundError(f"Paper clusters file not found: {paper_file}")
    if not noise_file.exists():
        raise FileNotFoundError(f"Noise analysis file not found: {noise_file}")
    
    print(f"Loading paper data from: {paper_file}")
    paper_df = pd.read_csv(paper_file)
    
    print(f"Loading noise analysis from: {noise_file}")
    with open(noise_file, 'r') as f:
        noise_stats = json.load(f)
    
    cluster_labels = paper_df['paper_cluster'].values
    
    print(f"Loaded {len(paper_df)} papers with {len(np.unique(cluster_labels))} unique cluster labels")
    
    # Create analyzer instance
    analyzer = PaperClusteringAnalyzer(work_dir)
    
    # Create visualizations with error handling
    figures_created = []
    
    try:
        print("Creating t-SNE 2D visualization...")
        X_2d_tsne, fig_2d_tsne = analyzer.create_2d_visualization(paper_df, cluster_labels, 'tsne')
        figures_created.append((fig_2d_tsne, f'2d_tsne_clusters{suffix}.png'))
    except Exception as e:
        print(f"Warning: t-SNE 2D visualization failed: {e}")
    
    try:
        print("Creating PCA 2D visualization...")
        X_2d_pca, fig_2d_pca = analyzer.create_2d_visualization(paper_df, cluster_labels, 'pca')
        figures_created.append((fig_2d_pca, f'2d_pca_clusters{suffix}.png'))
    except Exception as e:
        print(f"Warning: PCA 2D visualization failed: {e}")
    
    try:
        print("Creating 3D visualization...")
        X_3d, fig_3d = analyzer.create_3d_visualization(paper_df, cluster_labels, 'pca')
        figures_created.append((fig_3d, f'3d_pca_clusters{suffix}.png'))
    except Exception as e:
        print(f"Warning: 3D visualization failed: {e}")
    
    try:
        print("Creating summary plots...")
        fig_summary = analyzer.create_summary_plots(paper_df, noise_stats)
        figures_created.append((fig_summary, f'summary_stats{suffix}.png'))
    except Exception as e:
        print(f"Warning: Summary plots failed: {e}")
    
    # Save figures
    print("Saving figures...")
    for fig, name in figures_created:
        if fig is not None:
            try:
                fig_path = work_dir / name
                fig.savefig(fig_path, dpi=200, bbox_inches='tight')
                print(f"Saved figure: {fig_path}")
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not save {name}: {e}")
    
    print("Resume completed!")


def main():
    parser = argparse.ArgumentParser(description="End-to-end paper clustering analysis")
    
    parser.add_argument("--labels_file", help="Path to labels_paragraph.csv file")
    parser.add_argument("--work_dir", required=True, help="Directory for outputs")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Minimum cluster size for paper clustering")
    parser.add_argument("--exclude_noise", action="store_true", default=False,
                       help="Exclude noise cluster from paper vectors (recommended for better clustering)")
    parser.add_argument("--n_jobs", type=int, default=None,
                       help="Number of parallel jobs for paper processing (default: all CPU cores, -1: all cores, 1: no parallelization)")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume from existing results (only create plots)")
    
    args = parser.parse_args()
    
    # Setup
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    if args.resume:
        # Resume mode - only create plots from existing results
        resume_from_existing_results(work_dir, exclude_noise=args.exclude_noise)
    else:
        # Full analysis mode
        if not args.labels_file:
            raise ValueError("--labels_file is required for full analysis (not in resume mode)")
        
        # Run analysis
        analyzer = PaperClusteringAnalyzer(work_dir, n_jobs=args.n_jobs)
        results = analyzer.run_complete_analysis(Path(args.labels_file), args.min_cluster_size, 
                                               exclude_noise=args.exclude_noise)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED")
        print("="*60)
        print(f"Results saved to: {work_dir}")
        print("Configuration:")
        print(f"  - Noise excluded from vectors: {args.exclude_noise}")
        print(f"  - Minimum cluster size: {args.min_cluster_size}")
        print(f"  - Parallel jobs used: {analyzer.n_jobs}")
        print("\nFiles created:")
        suffix = "_no_noise" if args.exclude_noise else ""
        print(f"  - paper_clusters{suffix}.csv: Paper-level data with cluster assignments")
        print(f"  - noise_analysis{suffix}.json: Detailed noise statistics")
        print(f"  - paper_cluster_metrics{suffix}.json: Clustering metrics")
        if args.exclude_noise:
            print(f"  - zero_vector_papers{suffix}.txt: List of papers with zero vectors (100% noise)")
        else:
            print(f"  - fully_noised_papers{suffix}.txt: List of 100% noise papers")
        print(f"  - *{suffix}.png: Visualization plots")


if __name__ == "__main__":
    main()