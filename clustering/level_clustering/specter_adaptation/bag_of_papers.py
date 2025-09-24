#!/usr/bin/env python3
"""
End-to-end paper clustering analysis script.
Takes paragraph clustering results and creates paper-level analysis with visualizations.

Input: labels_paragraph.csv from clustering script
Output: Paper clusters, noise analysis, and 2D/3D visualizations
"""

import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

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

# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')
plt.style.use('default')


class PaperClusteringAnalyzer:
    """Complete paper-level clustering analysis from paragraph results"""
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.results = {}
        
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
    
    def create_paper_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create bag-of-clusters vectors for each paper"""
        print("Creating paper vectors from paragraph clusters...")
        
        # Get all unique clusters (including noise)
        all_clusters = sorted(df['cluster'].unique())
        n_clusters = len(all_clusters)
        
        print(f"Creating vectors with {n_clusters} dimensions (including noise)")
        
        # Create paper vectors
        paper_vectors = []
        paper_metadata = []
        
        for paper_id in tqdm(df['paper_id'].unique(), desc="Processing papers"):
            paper_paragraphs = df[df['paper_id'] == paper_id]
            
            # Count paragraphs in each cluster
            cluster_counts = Counter(paper_paragraphs['cluster'])
            total_paragraphs = len(paper_paragraphs)
            
            # Create normalized vector (frequencies)
            vector = np.zeros(n_clusters)
            for i, cluster_id in enumerate(all_clusters):
                vector[i] = cluster_counts.get(cluster_id, 0) / total_paragraphs
            
            paper_vectors.append(vector)
            
            # Store metadata
            noise_count = cluster_counts.get(-1, 0)
            paper_metadata.append({
                'paper_id': paper_id,
                'total_paragraphs': total_paragraphs,
                'noise_paragraphs': noise_count,
                'noise_ratio': noise_count / total_paragraphs,
                'unique_clusters': len([c for c in cluster_counts.keys() if c != -1])
            })
        
        # Create DataFrame
        paper_vectors = np.array(paper_vectors)
        metadata_df = pd.DataFrame(paper_metadata)
        
        # Add vector columns
        vector_cols = [f'cluster_{c}' for c in all_clusters]
        vector_df = pd.DataFrame(paper_vectors, columns=vector_cols)
        
        result_df = pd.concat([metadata_df, vector_df], axis=1)
        
        print(f"Created vectors for {len(result_df):,} papers")
        print(f"Vector dimensionality: {paper_vectors.shape[1]}")
        
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
    
    def cluster_papers(self, paper_df: pd.DataFrame, min_cluster_size: int = 10) -> Tuple[np.ndarray, Dict]:
        """Cluster papers based on their cluster vectors"""
        print("Clustering papers...")
        
        # Get vector columns (exclude metadata)
        vector_cols = [col for col in paper_df.columns if col.startswith('cluster_')]
        X = paper_df[vector_cols].values
        
        print(f"Clustering {len(X):,} papers with {len(vector_cols)} features")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Remove fully noised papers for clustering (they would dominate)
        non_noised_mask = paper_df['noise_ratio'] < 1.0
        X_clustering = X_scaled[non_noised_mask]
        
        print(f"Clustering {len(X_clustering):,} non-fully-noised papers")
        
        # Cluster with HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(3, min_cluster_size // 3),
            metric='euclidean',
            cluster_selection_epsilon=0.1
        )
        
        cluster_labels_subset = clusterer.fit_predict(X_clustering)
        
        # Map back to full dataset
        cluster_labels = np.full(len(paper_df), -2, dtype=int)  # -2 = fully noised
        cluster_labels[non_noised_mask] = cluster_labels_subset
        
        # Calculate metrics on non-noised papers only
        n_clusters = len(set(cluster_labels_subset)) - (1 if -1 in cluster_labels_subset else 0)
        n_noise = list(cluster_labels_subset).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_fully_noised': (~non_noised_mask).sum(),
            'noise_pct': n_noise / len(cluster_labels_subset) * 100 if len(cluster_labels_subset) > 0 else 0,
            'fully_noised_pct': (~non_noised_mask).sum() / len(paper_df) * 100
        }
        
        if n_clusters > 1:
            metrics['silhouette_score'] = silhouette_score(X_clustering, cluster_labels_subset)
        else:
            metrics['silhouette_score'] = -1
        
        print(f"Paper Clustering Results:")
        print(f"  Clusters: {n_clusters}")
        print(f"  Noise papers: {n_noise} ({metrics['noise_pct']:.1f}%)")
        print(f"  Fully noised papers: {metrics['n_fully_noised']} ({metrics['fully_noised_pct']:.1f}%)")
        if metrics['silhouette_score'] > -1:
            print(f"  Silhouette score: {metrics['silhouette_score']:.3f}")
        
        return cluster_labels, metrics
    
    def create_2d_visualization(self, paper_df: pd.DataFrame, cluster_labels: np.ndarray, 
                               method: str = 'tsne') -> Tuple[np.ndarray, plt.Figure]:
        """Create 2D visualization of paper clusters"""
        print(f"Creating 2D visualization using {method.upper()}...")
        
        # Get vector data
        vector_cols = [col for col in paper_df.columns if col.startswith('cluster_')]
        X = paper_df[vector_cols].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
            X_2d = reducer.fit_transform(X_scaled)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X_scaled)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Colored by cluster
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                mask = cluster_labels == label
                ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', alpha=0.5, s=20, label='Noise')
            elif label == -2:
                # Fully noised papers
                mask = cluster_labels == label
                ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c='gray', alpha=0.3, s=15, label='Fully Noised')
            else:
                # Regular clusters
                mask = cluster_labels == label
                ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], alpha=0.7, s=30, label=f'Cluster {label}')
        
        ax1.set_title(f'Paper Clusters ({method.upper()})')
        ax1.set_xlabel(f'{method.upper()}-1')
        ax1.set_ylabel(f'{method.upper()}-2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
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
        """Create 3D visualization of paper clusters"""
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
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by cluster
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = cluster_labels == label
            if label == -1:
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
                          c='black', alpha=0.4, s=15, label='Noise')
            elif label == -2:
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
                          c='gray', alpha=0.2, s=10, label='Fully Noised')
            else:
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
                          c=[color], alpha=0.7, s=20, label=f'Cluster {label}')
        
        ax.set_title('3D Paper Clusters (PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        
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
        percentages = [noise_stats['threshold_analysis'][f'noise_ratio_{t}']['percentage'] for t in thresholds]
        
        axes[1, 1].bar(thresholds, percentages, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 1].set_xlabel('Noise Ratio Threshold')
        axes[1, 1].set_ylabel('Percentage of Papers')
        axes[1, 1].set_title('Papers Above Noise Thresholds')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def run_complete_analysis(self, labels_file: Path, min_cluster_size: int = 10) -> Dict:
        """Run complete end-to-end analysis"""
        print("="*60)
        print("PAPER CLUSTERING ANALYSIS")
        print("="*60)
        
        # Load paragraph results
        paragraph_df = self.load_paragraph_results(labels_file)
        
        # Create paper vectors
        paper_df = self.create_paper_vectors(paragraph_df)
        
        # Analyze noise
        noise_stats = self.analyze_noise_papers(paper_df)
        
        # Cluster papers
        cluster_labels, cluster_metrics = self.cluster_papers(paper_df, min_cluster_size)
        
        # Add cluster labels to dataframe
        paper_df['paper_cluster'] = cluster_labels
        
        # Create visualizations
        X_2d_tsne, fig_2d_tsne = self.create_2d_visualization(paper_df, cluster_labels, 'tsne')
        X_2d_pca, fig_2d_pca = self.create_2d_visualization(paper_df, cluster_labels, 'pca')
        X_3d, fig_3d = self.create_3d_visualization(paper_df, cluster_labels, 'pca')
        fig_summary = self.create_summary_plots(paper_df, noise_stats)
        
        # Save results
        self.save_results(paper_df, noise_stats, cluster_metrics, 
                         [fig_2d_tsne, fig_2d_pca, fig_3d, fig_summary])
        
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
                    cluster_metrics: Dict, figures: List[plt.Figure]):
        """Save all results to files"""
        print("Saving results...")
        
        # Save paper dataframe
        paper_file = self.work_dir / "paper_clusters.csv"
        paper_df.to_csv(paper_file, index=False)
        print(f"Saved paper clusters to: {paper_file}")
        
        # Save noise analysis
        noise_file = self.work_dir / "noise_analysis.json"
        with open(noise_file, 'w') as f:
            json.dump(noise_stats, f, indent=2)
        print(f"Saved noise analysis to: {noise_file}")
        
        # Save cluster metrics
        metrics_file = self.work_dir / "paper_cluster_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(cluster_metrics, f, indent=2)
        print(f"Saved cluster metrics to: {metrics_file}")
        
        # Save figures
        figure_names = ['2d_tsne_clusters.png', '2d_pca_clusters.png', '3d_pca_clusters.png', 'summary_stats.png']
        for fig, name in zip(figures, figure_names):
            fig_path = self.work_dir / name
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {fig_path}")
        
        # Save fully noised papers list
        fully_noised = paper_df[paper_df['noise_ratio'] == 1.0]['paper_id'].tolist()
        fully_noised_file = self.work_dir / "fully_noised_papers.txt"
        with open(fully_noised_file, 'w') as f:
            for paper_id in fully_noised:
                f.write(f"{paper_id}\n")
        print(f"Saved fully noised papers list to: {fully_noised_file}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end paper clustering analysis")
    
    parser.add_argument("--labels_file", required=True, help="Path to labels_paragraph.csv file")
    parser.add_argument("--work_dir", required=True, help="Directory for outputs")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Minimum cluster size for paper clustering")
    
    args = parser.parse_args()
    
    # Setup
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    analyzer = PaperClusteringAnalyzer(work_dir)
    results = analyzer.run_complete_analysis(Path(args.labels_file), args.min_cluster_size)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    print(f"Results saved to: {work_dir}")
    print("Files created:")
    print("  - paper_clusters.csv: Paper-level data with cluster assignments")
    print("  - noise_analysis.json: Detailed noise statistics")
    print("  - paper_cluster_metrics.json: Clustering metrics")
    print("  - fully_noised_papers.txt: List of 100% noise papers")
    print("  - *.png: Visualization plots")


if __name__ == "__main__":
    main()