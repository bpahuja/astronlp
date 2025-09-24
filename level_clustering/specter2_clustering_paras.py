import json
import numpy as np
import pandas as pd
import hdbscan
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from collections import Counter
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParagraphClusterer:
    """
    Enhanced paragraph clustering system with improved functionality and error handling.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.embeddings = None
        self.df = None
        self.cluster_labels = None
        self.clusterer = None
        self.umap_model = None
        self.reduced_embeddings = None
        
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load embeddings, ID mappings, and original texts with improved error handling.
        """
        logger.info("Loading data...")
        
        try:
            # Load embeddings
            embeddings = np.load(self.config['embeddings_file'])
            logger.info(f"Loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
            
            # Load ID to index mapping
            with open(self.config['id_to_index_file'], 'r') as f:
                id_to_index = json.load(f)
            
            # Create reverse mapping
            index_to_id = {v: k for k, v in id_to_index.items()}
            
            # Load original texts from directory structure
            para_data = self._load_paragraphs_from_directory()
            
            # Create DataFrame and merge
            text_df = pd.DataFrame(para_data)
            map_df = pd.DataFrame(list(index_to_id.items()), columns=['embedding_index', 'para_id'])
            
            merged_df = pd.merge(map_df, text_df, on='para_id', how='inner')
            
            if len(merged_df) != len(embeddings):
                logger.warning(f"Mismatch: {len(embeddings)} embeddings vs {len(merged_df)} texts")
            
            merged_df = merged_df.sort_values('embedding_index').reset_index(drop=True)
            
            # Validate data integrity
            self._validate_data(merged_df, embeddings)
            
            logger.info(f"Data loaded successfully. Found {len(merged_df)} paragraphs.")
            return merged_df, embeddings
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_paragraphs_from_directory(self) -> List[Dict]:
        """
        Load paragraphs from directory structure where each paper is a .txt file.
        """
        para_data = []
        papers_dir = Path(self.config['papers_directory'])
        
        if not papers_dir.exists():
            raise FileNotFoundError(f"Papers directory not found: {papers_dir}")
        
        for txt_file in papers_dir.glob("*.txt"):
            paper_id = txt_file.stem
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into paragraphs (assuming double newlines separate paragraphs)
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                for i, para_text in enumerate(paragraphs):
                    para_id = f"{paper_id}_para_{i:04d}"
                    para_data.append({
                        'para_id': para_id,
                        'text': para_text,
                        'paper_id': paper_id,
                        'para_index': i,
                        'para_length': len(para_text)
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {txt_file}: {e}")
                continue
        
        logger.info(f"Loaded paragraphs from {len(set(p['paper_id'] for p in para_data))} papers")
        return para_data
    
    def _validate_data(self, df: pd.DataFrame, embeddings: np.ndarray):
        """
        Validate data integrity and log statistics.
        """
        # Check for missing texts
        empty_texts = df['text'].isna().sum() + (df['text'] == '').sum()
        if empty_texts > 0:
            logger.warning(f"Found {empty_texts} empty or missing texts")
        
        # Log text length statistics
        text_lengths = df['text'].str.len()
        logger.info(f"Text lengths - Min: {text_lengths.min()}, Max: {text_lengths.max()}, Mean: {text_lengths.mean():.1f}")
        
        # Check for duplicate paragraph IDs
        duplicates = df['para_id'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate paragraph IDs")
    
    def optimize_clustering_parameters(self, embeddings: np.ndarray) -> Dict:
        """
        Find optimal clustering parameters using silhouette analysis.
        """
        logger.info("Optimizing clustering parameters...")
        
        # For very large datasets, use a smaller sample for parameter optimization
        if embeddings.shape[0] > 50000:
            sample_size = min(50000, embeddings.shape[0])
            logger.info(f"Using random sample of {sample_size:,} points for parameter optimization")
            indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
        
        min_sizes = [10, 15, 20, 25, 30, 50]
        best_score = -1
        best_params = {}
        
        # Use dimensionality reduction for parameter optimization if enabled and needed
        if (self.config.get('use_dimensionality_reduction', True) and 
            sample_embeddings.shape[1] > 50):
            try:
                umap_model = umap.UMAP(
                    n_components=50, 
                    random_state=42,
                    low_memory=True,
                    n_jobs=1
                )
                reduced_embeddings = umap_model.fit_transform(sample_embeddings)
            except Exception as e:
                logger.warning(f"UMAP failed during optimization: {e}")
                reduced_embeddings = sample_embeddings
        else:
            reduced_embeddings = sample_embeddings
        
        for min_size in min_sizes:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_size,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                
                labels = clusterer.fit_predict(reduced_embeddings)
                
                if len(set(labels)) > 1 and -1 in labels:  # Has clusters and noise
                    # Only calculate silhouette for non-noise points
                    mask = labels != -1
                    if mask.sum() > min_size * 2:  # Ensure enough points for meaningful score
                        score = silhouette_score(reduced_embeddings[mask], labels[mask])
                        
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
                            
            except Exception as e:
                logger.warning(f"Error with min_size {min_size}: {e}")
                continue
        
        if best_params:
            logger.info(f"Best parameters: {best_params}")
            return best_params
        else:
            logger.warning("Could not find optimal parameters, using default")
            return {'min_cluster_size': self.config['min_cluster_size']}
    
    def perform_clustering(self, embeddings: np.ndarray, optimize_params: bool = True) -> np.ndarray:
        """
        Perform HDBSCAN clustering with optional parameter optimization.
        """
        logger.info("Starting clustering...")
        
        # Check memory requirements and warn user
        memory_gb = (embeddings.nbytes) / (1024**3)
        logger.info(f"Embeddings require {memory_gb:.2f} GB of memory")
        
        # For very large datasets, recommend disabling dimensionality reduction
        if embeddings.shape[0] > 1000000:  # > 1M samples
            logger.warning(f"Large dataset detected ({embeddings.shape[0]:,} samples)")
            logger.warning("Consider using --no-dim-reduction for better memory efficiency")
        
        # Optionally optimize parameters (use subset for large datasets)
        if optimize_params and not self.config.get('skip_optimization', False):
            optimal_params = self.optimize_clustering_parameters(embeddings)
            min_cluster_size = optimal_params['min_cluster_size']
        else:
            min_cluster_size = self.config['min_cluster_size']
        
        # Perform dimensionality reduction if enabled and embeddings are high-dimensional
        if (self.config.get('use_dimensionality_reduction', True) and 
            embeddings.shape[1] > self.config.get('dim_reduction_threshold', 100)):
            
            # Check if dataset is too large for UMAP
            if embeddings.shape[0] > self.config.get('max_samples_for_umap', 500000):
                logger.warning(f"Dataset too large for UMAP ({embeddings.shape[0]:,} samples > {self.config.get('max_samples_for_umap', 500000):,})")
                logger.warning("Disabling dimensionality reduction to avoid memory issues")
                logger.warning("Use --max-samples-umap to increase limit or --no-dim-reduction to disable")
                self.reduced_embeddings = embeddings
            else:
                logger.info(f"Reducing dimensionality from {embeddings.shape[1]} to {self.config['umap_n_components']}")
                try:
                    # Use more memory-efficient UMAP settings for large datasets
                    umap_params = {
                        'n_components': self.config['umap_n_components'],
                        'random_state': 42,
                        'metric': 'cosine',
                        'low_memory': True,  # Enable low memory mode
                        'n_jobs': 1  # Disable parallel processing to save memory
                    }
                    
                    # Additional memory-saving parameters for large datasets
                    if embeddings.shape[0] > 100000:
                        umap_params.update({
                            'n_neighbors': min(15, max(5, embeddings.shape[0] // 10000)),
                            'min_dist': 0.1,
                            'spread': 1.0
                        })
                        logger.info("Using memory-optimized UMAP parameters for large dataset")
                    
                    self.umap_model = umap.UMAP(**umap_params)
                    self.reduced_embeddings = self.umap_model.fit_transform(embeddings)
                    
                except Exception as e:
                    logger.error(f"UMAP failed: {e}")
                    logger.info("Falling back to original embeddings")
                    self.reduced_embeddings = embeddings
        else:
            if not self.config.get('use_dimensionality_reduction', True):
                logger.info("Dimensionality reduction disabled - using original embeddings")
            else:
                logger.info(f"Embeddings dimensionality ({embeddings.shape[1]}) below threshold - using original embeddings")
            self.reduced_embeddings = embeddings
        
        # Perform clustering
        logger.info(f"Performing HDBSCAN clustering on {self.reduced_embeddings.shape[0]:,} samples with {self.reduced_embeddings.shape[1]} dimensions...")
        
        # Use memory-efficient clustering settings for large datasets
        clustering_params = {
            'min_cluster_size': min_cluster_size,
            'min_samples': self.config.get('min_samples', 1),
            'metric': self.config.get('metric', 'euclidean'),
            'cluster_selection_method': self.config.get('cluster_selection_method', 'eom'),
            'cluster_selection_epsilon': self.config.get('cluster_selection_epsilon', 0.0),
            'core_dist_n_jobs': 1  # Disable parallel processing to save memory
        }
        
        # For very large datasets, use additional memory-saving options
        if self.reduced_embeddings.shape[0] > 500000:
            clustering_params['algorithm'] = 'boruvka_kdtree'
            clustering_params['leaf_size'] = 40
            logger.info("Using memory-optimized clustering parameters for large dataset")
        
        self.clusterer = hdbscan.HDBSCAN(**clustering_params)
        
        cluster_labels = self.clusterer.fit_predict(self.reduced_embeddings)
        
        # Log clustering results
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_points = (cluster_labels == -1).sum()
        logger.info(f"Clustering complete: {num_clusters} clusters, {noise_points} noise points")
        
        return cluster_labels
    
    def generate_enhanced_labels(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, str]:
        """
        Generate enhanced cluster labels using multiple approaches.
        """
        logger.info("Generating enhanced cluster labels...")
        
        df_copy = df.copy()
        df_copy['cluster_id'] = cluster_labels
        
        # Group by cluster
        cluster_groups = df_copy.groupby('cluster_id')
        labels = {}
        
        for cluster_id, group in cluster_groups:
            if cluster_id == -1:
                labels[cluster_id] = "-1_Noise"
                continue
            
            # Method 1: TF-IDF keywords
            tfidf_keywords = self._get_tfidf_keywords(group['text'].tolist())
            
            # Method 2: Most common paper topics (if available)
            common_papers = group['paper_id'].value_counts().head(2).index.tolist()
            
            # Method 3: Statistical features
            avg_length = group['para_length'].mean()
            
            # Combine into label
            size_info = f"n={len(group)}"
            length_info = f"avg_len={int(avg_length)}"
            
            if tfidf_keywords:
                keyword_part = "_".join(tfidf_keywords[:3])
                labels[cluster_id] = f"{cluster_id}_{keyword_part}_{size_info}"
            else:
                labels[cluster_id] = f"{cluster_id}_cluster_{size_info}_{length_info}"
        
        return labels
    
    def _get_tfidf_keywords(self, texts: List[str], top_n: int = 3) -> List[str]:
        """
        Extract top TF-IDF keywords from a list of texts.
        """
        if len(texts) == 0:
            return []
        
        try:
            # Combine all texts in the cluster
            combined_text = ' '.join(texts)
            
            # Use TF-IDF with all cluster texts as background
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8,
                max_features=1000
            )
            
            # Create a small corpus for this cluster vs background
            corpus = [combined_text]
            
            # If we have multiple texts, add them individually for better TF-IDF
            if len(texts) > 1:
                corpus.extend(texts[:10])  # Limit to avoid memory issues
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for the combined text (first document)
            scores = tfidf_matrix[0].toarray().flatten()
            top_indices = scores.argsort()[-top_n:][::-1]
            
            keywords = [feature_names[idx].replace(' ', '_') for idx in top_indices if scores[idx] > 0]
            return keywords
            
        except Exception as e:
            logger.warning(f"Error generating TF-IDF keywords: {e}")
            return []
    
    def save_results(self, df: pd.DataFrame, cluster_labels: np.ndarray, cluster_label_map: Dict):
        """
        Save clustering results with multiple output formats.
        """
        logger.info("Saving results...")
        
        # Prepare final DataFrame
        result_df = df.copy()
        result_df['cluster_id'] = cluster_labels
        result_df['cluster_label'] = result_df['cluster_id'].map(cluster_label_map)
        
        # Add clustering metadata
        if self.clusterer is not None:
            result_df['cluster_probability'] = self.clusterer.probabilities_
            result_df['outlier_score'] = self.clusterer.outlier_scores_
        
        # Save main results
        output_file = self.config['output_file']
        result_df.to_csv(output_file, index=False)
        logger.info(f"Main results saved to {output_file}")
        
        # Save cluster summary
        self._save_cluster_summary(result_df)
        
        # Save model artifacts if requested
        if self.config.get('save_models', False):
            self._save_models()
    
    def _save_cluster_summary(self, df: pd.DataFrame):
        """
        Save a summary of clustering results.
        """
        summary_file = self.config['output_file'].replace('.csv', '_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("PARAGRAPH CLUSTERING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            num_clusters = len(df['cluster_id'].unique()) - (1 if -1 in df['cluster_id'].values else 0)
            noise_points = (df['cluster_id'] == -1).sum()
            noise_percentage = noise_points / len(df) * 100
            
            f.write(f"Total paragraphs: {len(df)}\n")
            f.write(f"Number of clusters: {num_clusters}\n")
            f.write(f"Noise points: {noise_points} ({noise_percentage:.2f}%)\n\n")
            
            # Cluster size distribution
            f.write("CLUSTER SIZE DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            cluster_sizes = df[df['cluster_id'] != -1]['cluster_id'].value_counts().sort_index()
            for cluster_id, size in cluster_sizes.head(20).items():
                label = df[df['cluster_id'] == cluster_id]['cluster_label'].iloc[0]
                f.write(f"Cluster {cluster_id}: {size} paragraphs - {label}\n")
            
            if len(cluster_sizes) > 20:
                f.write(f"... and {len(cluster_sizes) - 20} more clusters\n")
        
        logger.info(f"Cluster summary saved to {summary_file}")
    
    def _save_models(self):
        """
        Save trained models for later use.
        """
        models_dir = Path(self.config['output_file']).parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Save HDBSCAN model
        if self.clusterer is not None:
            with open(models_dir / 'hdbscan_model.pkl', 'wb') as f:
                pickle.dump(self.clusterer, f)
        
        # Save UMAP model
        if self.umap_model is not None:
            with open(models_dir / 'umap_model.pkl', 'wb') as f:
                pickle.dump(self.umap_model, f)
        
        logger.info(f"Models saved to {models_dir}")
    
    def generate_visualizations(self, df: pd.DataFrame):
        """
        Generate visualization plots for clustering results.
        """
        if self.reduced_embeddings is None:
            logger.warning("No reduced embeddings available for visualization")
            return
        
        logger.info("Generating visualizations...")
        
        # Create visualization directory
        viz_dir = Path(self.config['output_file']).parent / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 2D visualization of clusters
        if self.reduced_embeddings.shape[1] >= 2:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                self.reduced_embeddings[:, 0], 
                self.reduced_embeddings[:, 1],
                c=df['cluster_id'], 
                cmap='tab20', 
                alpha=0.6,
                s=30
            )
            plt.colorbar(scatter)
            plt.title('Paragraph Clusters (2D Projection)')
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.savefig(viz_dir / 'cluster_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Cluster size distribution
        plt.figure(figsize=(12, 6))
        cluster_sizes = df[df['cluster_id'] != -1]['cluster_id'].value_counts().sort_values(ascending=False)
        cluster_sizes.head(20).plot(kind='bar')
        plt.title('Top 20 Cluster Sizes')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Paragraphs')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / 'cluster_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def run_full_pipeline(self):
        """
        Run the complete clustering pipeline.
        """
        logger.info("Starting full clustering pipeline...")
        
        # Load data
        self.df, self.embeddings = self.load_data()
        
        # Perform clustering
        self.cluster_labels = self.perform_clustering(self.embeddings)
        
        # Generate labels
        cluster_label_map = self.generate_enhanced_labels(self.df, self.cluster_labels)
        
        # Save results
        self.save_results(self.df, self.cluster_labels, cluster_label_map)
        
        # Generate visualizations
        if self.config.get('generate_visualizations', True):
            self.generate_visualizations(self.df.copy().assign(cluster_id=self.cluster_labels))
        
        logger.info("Clustering pipeline completed successfully!")
        
        return self.df, self.cluster_labels, cluster_label_map


def create_default_config():
    """
    Create default configuration dictionary.
    """
    return {
        # Input files
        'embeddings_file': 'embeddings.npy',
        'id_to_index_file': 'id_to_index.json',
        'papers_directory': 'papers/',  # Directory containing paper .txt files
        
        # Output files
        'output_file': 'clustered_paragraphs.csv',
        
        # Clustering parameters
        'min_cluster_size': 15,
        'min_samples': 1,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'cluster_selection_epsilon': 0.0,
        
        # Dimensionality reduction parameters
        'use_dimensionality_reduction': True,  # Set to False to disable UMAP
        'dim_reduction_threshold': 100,  # Only apply UMAP if dims > this threshold
        'max_samples_for_umap': 500000,  # Max samples for UMAP (memory limit)
        'umap_n_components': 50,
        
        # Label generation parameters
        'top_n_keywords': 4,
        
        # Pipeline options
        'skip_optimization': False,
        'save_models': True,
        'generate_visualizations': True
    }


def main():
    """
    Main function with command line interface.
    """
    parser = argparse.ArgumentParser(description='Enhanced Paragraph Clustering')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--embeddings', type=str, default='embeddings.npy', help='Path to embeddings file')
    parser.add_argument('--id-map', type=str, default='id_to_index.json', help='Path to ID mapping file')
    parser.add_argument('--papers-dir', type=str, default='papers/', help='Directory containing paper text files')
    parser.add_argument('--output', type=str, default='clustered_paragraphs.csv', help='Output CSV file')
    parser.add_argument('--min-cluster-size', type=int, default=15, help='Minimum cluster size')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip parameter optimization')
    parser.add_argument('--no-dim-reduction', action='store_true', help='Disable dimensionality reduction')
    parser.add_argument('--dim-threshold', type=int, default=100, help='Dimensionality threshold for UMAP')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # Override with command line arguments
        config.update({
            'embeddings_file': args.embeddings,
            'id_to_index_file': args.id_map,
            'papers_directory': args.papers_dir,
            'output_file': args.output,
            'min_cluster_size': args.min_cluster_size,
            'skip_optimization': args.skip_optimization,
            'use_dimensionality_reduction': not args.no_dim_reduction,
            'dim_reduction_threshold': args.dim_threshold
        })
    
    # Run clustering
    clusterer = ParagraphClusterer(config)
    clusterer.run_full_pipeline()


if __name__ == "__main__":
    main()