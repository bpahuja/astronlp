#!/usr/bin/env python3
"""
Comprehensive diagnostic script for embedding clustering issues.
Analyzes embedding quality, diversity, and clustering parameters.
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class EmbeddingDiagnostics:
    """Comprehensive embedding and clustering diagnostics"""
    
    def __init__(self, work_dir: Path, sample_size: int = 10000):
        self.work_dir = Path(work_dir)
        self.sample_size = sample_size
        self.results = {}
        
    def load_sample_data(self, chunks_root: Path, chunk_pattern: str = r"chunk_\d{4}",
                        meta_name: str = "metadata.csv", emb_name: str = "embeddings.npy") -> Tuple[np.ndarray, List[str]]:
        """Load sample embeddings and texts for analysis"""
        print(f"Loading sample of {self.sample_size} embeddings...")
        
        chunk_dirs = self._scan_chunks(chunks_root, chunk_pattern)
        
        embeddings = []
        texts = []
        total_loaded = 0
        
        for cdir in tqdm(chunk_dirs, desc="Loading chunks"):
            if total_loaded >= self.sample_size:
                break
                
            emb_path = cdir / emb_name
            meta_path = cdir / meta_name
            
            if not emb_path.exists() or not meta_path.exists():
                continue
            
            # Load embeddings
            chunk_embeddings = np.load(emb_path, mmap_mode="r")
            
            # Load metadata in batches
            chunk_texts = []
            for df in pd.read_csv(meta_path, chunksize=10000):
                chunk_texts.extend(df["text"].tolist())
                if len(chunk_texts) >= chunk_embeddings.shape[0]:
                    break
            
            # Sample from this chunk
            chunk_size = min(chunk_embeddings.shape[0], self.sample_size - total_loaded)
            if chunk_size <= 0:
                break
                
            indices = np.random.choice(chunk_embeddings.shape[0], chunk_size, replace=False)
            
            embeddings.append(chunk_embeddings[indices])
            texts.extend([chunk_texts[i] for i in indices])
            total_loaded += chunk_size
        
        if not embeddings:
            raise ValueError("No embeddings loaded")
            
        embeddings = np.vstack(embeddings)
        print(f"Loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
        
        return embeddings, texts
    
    def _scan_chunks(self, root: Path, pattern: str) -> List[Path]:
        """Scan for chunk directories"""
        rx = re.compile(pattern)
        dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and rx.fullmatch(p.name)]
        if not dirs:
            raise FileNotFoundError(f"No chunk dirs matching /{pattern}/ under {root}")
        return dirs
    
    def diagnose_embedding_diversity(self, embeddings: np.ndarray) -> Dict:
        """Analyze embedding diversity and potential collapse issues"""
        print("Analyzing embedding diversity...")
        
        # Sample for computational efficiency
        n_sample = min(5000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), n_sample, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        # Calculate pairwise distances
        distances = pdist(sample_embeddings, metric='cosine')
        euclidean_distances = pdist(sample_embeddings, metric='euclidean')
        
        # Embedding magnitudes
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Dimensionality analysis
        pca = PCA()
        pca.fit(sample_embeddings)
        explained_variance = pca.explained_variance_ratio_
        
        # Calculate effective dimensionality (95% variance)
        cumvar = np.cumsum(explained_variance)
        effective_dim = np.argmax(cumvar >= 0.95) + 1
        
        diversity_metrics = {
            "cosine_distances": {
                "mean": float(np.mean(distances)),
                "std": float(np.std(distances)),
                "min": float(np.min(distances)),
                "max": float(np.max(distances)),
                "q25": float(np.percentile(distances, 25)),
                "q75": float(np.percentile(distances, 75))
            },
            "euclidean_distances": {
                "mean": float(np.mean(euclidean_distances)),
                "std": float(np.std(euclidean_distances))
            },
            "embedding_norms": {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "min": float(np.min(norms)),
                "max": float(np.max(norms))
            },
            "dimensionality": {
                "original_dim": int(embeddings.shape[1]),
                "effective_dim": int(effective_dim),
                "variance_concentration": float(explained_variance[0]),  # First PC variance
                "top_10_pc_variance": float(np.sum(explained_variance[:10]))
            }
        }
        
        # Identify potential issues
        issues = []
        if np.std(distances) < 0.05:
            issues.append("CRITICAL: Very low cosine distance diversity - possible embedding collapse")
        if np.std(norms) < 0.01:
            issues.append("WARNING: Very similar embedding magnitudes - possible over-normalization")
        if explained_variance[0] > 0.5:
            issues.append("WARNING: High variance concentration in first PC - possible underdiversified embeddings")
        if effective_dim < embeddings.shape[1] * 0.1:
            issues.append("CRITICAL: Very low effective dimensionality - severe information loss")
        
        diversity_metrics["issues"] = issues
        
        # Visualization
        self._plot_diversity_analysis(distances, norms, explained_variance)
        
        return diversity_metrics
    
    def test_methodological_separation(self, embeddings: np.ndarray, texts: List[str]) -> Dict:
        """Test if embeddings can distinguish methodological differences"""
        print("Testing methodological separation...")
        
        # Define methodological keywords
        method_categories = {
            "spectroscopy": ["spectroscopy", "spectroscopic", "spectrum", "spectra"],
            "simulation": ["simulation", "simulate", "simulated", "monte carlo", "n-body"],
            "photometry": ["photometry", "photometric", "magnitude", "brightness"],
            "theoretical": ["theoretical", "theory", "model", "analytical"],
            "observational": ["observed", "observation", "telescope", "survey"]
        }
        
        # Find paragraphs for each method
        method_indices = {}
        for method, keywords in method_categories.items():
            indices = []
            for i, text in enumerate(texts):
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in keywords):
                    indices.append(i)
            
            if len(indices) >= 10:  # Need minimum samples
                method_indices[method] = indices[:50]  # Limit to 50 samples
        
        if len(method_indices) < 2:
            return {"error": "Insufficient methodological categories found in text"}
        
        separation_results = {}
        
        # Calculate within vs between method similarities
        for method1, indices1 in method_indices.items():
            emb1 = embeddings[indices1]
            
            # Within-method similarity
            if len(indices1) > 1:
                within_sim = np.mean([
                    np.dot(emb1[i], emb1[j]) / (np.linalg.norm(emb1[i]) * np.linalg.norm(emb1[j]))
                    for i in range(len(emb1)) for j in range(i+1, len(emb1))
                ])
            else:
                within_sim = 1.0
            
            # Between-method similarities
            between_sims = []
            for method2, indices2 in method_indices.items():
                if method1 != method2:
                    emb2 = embeddings[indices2]
                    between_sim = np.mean([
                        np.dot(emb1[i], emb2[j]) / (np.linalg.norm(emb1[i]) * np.linalg.norm(emb2[j]))
                        for i in range(len(emb1)) for j in range(len(emb2))
                    ])
                    between_sims.append(between_sim)
            
            separation_results[method1] = {
                "within_similarity": float(within_sim),
                "avg_between_similarity": float(np.mean(between_sims)) if between_sims else 0.0,
                "separation_ratio": float(within_sim / np.mean(between_sims)) if between_sims else 1.0,
                "n_samples": len(indices1)
            }
        
        # Overall separation quality
        all_within = [r["within_similarity"] for r in separation_results.values()]
        all_between = [r["avg_between_similarity"] for r in separation_results.values()]
        
        overall_metrics = {
            "avg_within_similarity": float(np.mean(all_within)),
            "avg_between_similarity": float(np.mean(all_between)),
            "separation_quality": float(np.mean(all_within) - np.mean(all_between)),
            "methods_found": list(method_indices.keys())
        }
        
        return {
            "method_separation": separation_results,
            "overall": overall_metrics,
            "interpretation": self._interpret_separation(overall_metrics)
        }
    
    def analyze_clustering_parameters(self, embeddings: np.ndarray, 
                                    current_params: Dict) -> Dict:
        """Analyze and suggest optimal clustering parameters"""
        print("Analyzing clustering parameters...")
        
        # Test different UMAP parameters
        umap_results = self._test_umap_parameters(embeddings)
        
        # Test different HDBSCAN parameters  
        hdbscan_results = self._test_hdbscan_parameters(embeddings)
        
        # Current configuration analysis
        current_analysis = self._analyze_current_params(embeddings, current_params)
        
        return {
            "umap_parameter_analysis": umap_results,
            "hdbscan_parameter_analysis": hdbscan_results,
            "current_configuration": current_analysis,
            "recommendations": self._generate_recommendations(umap_results, hdbscan_results, current_analysis)
        }
    
    def _test_umap_parameters(self, embeddings: np.ndarray) -> Dict:
        """Test different UMAP parameter combinations"""
        if not UMAP_AVAILABLE:
            return {"error": "UMAP not available"}
        
        # Sample for speed
        n_sample = min(2000, len(embeddings))
        sample_embeddings = embeddings[np.random.choice(len(embeddings), n_sample, replace=False)]
        
        param_combinations = [
            {"n_neighbors": 15, "min_dist": 0.1, "n_components": 20},
            {"n_neighbors": 30, "min_dist": 0.1, "n_components": 50}, 
            {"n_neighbors": 50, "min_dist": 0.05, "n_components": 100},
            {"n_neighbors": 100, "min_dist": 0.0, "n_components": 50},
        ]
        
        results = []
        for params in param_combinations:
            try:
                reducer = umap.UMAP(**params, metric='cosine', random_state=42)
                reduced_emb = reducer.fit_transform(sample_embeddings)
                
                # Quick clustering for evaluation
                if HDBSCAN_AVAILABLE:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
                    labels = clusterer.fit_predict(reduced_emb)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    if n_clusters > 1:
                        silhouette = silhouette_score(reduced_emb, labels)
                    else:
                        silhouette = -1
                    
                    results.append({
                        "params": params,
                        "n_clusters": n_clusters,
                        "noise_pct": n_noise / len(labels) * 100,
                        "silhouette_score": silhouette,
                        "reduced_shape": reduced_emb.shape
                    })
            except Exception as e:
                results.append({
                    "params": params,
                    "error": str(e)
                })
        
        return results
    
    def _test_hdbscan_parameters(self, embeddings: np.ndarray) -> Dict:
        """Test different HDBSCAN parameters on sample data"""
        if not HDBSCAN_AVAILABLE:
            return {"error": "HDBSCAN not available"}
        
        # Use UMAP-reduced embeddings for speed
        if UMAP_AVAILABLE:
            n_sample = min(3000, len(embeddings))
            sample_embeddings = embeddings[np.random.choice(len(embeddings), n_sample, replace=False)]
            
            reducer = umap.UMAP(n_components=20, n_neighbors=30, metric='cosine', random_state=42)
            reduced_emb = reducer.fit_transform(sample_embeddings)
        else:
            # Use PCA if UMAP not available
            n_sample = min(3000, len(embeddings))
            sample_embeddings = embeddings[np.random.choice(len(embeddings), n_sample, replace=False)]
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            reduced_emb = pca.fit_transform(sample_embeddings)
        
        param_combinations = [
            {"min_cluster_size": 20, "min_samples": 5},
            {"min_cluster_size": 50, "min_samples": 10},
            {"min_cluster_size": 100, "min_samples": 20},
            {"min_cluster_size": 200, "min_samples": 50},
            {"min_cluster_size": 500, "min_samples": 100},
        ]
        
        results = []
        for params in param_combinations:
            try:
                clusterer = hdbscan.HDBSCAN(**params, metric='euclidean')
                labels = clusterer.fit_predict(reduced_emb)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 1:
                    silhouette = silhouette_score(reduced_emb, labels)
                    calinski = calinski_harabasz_score(reduced_emb, labels)
                    davies_bouldin = davies_bouldin_score(reduced_emb, labels)
                else:
                    silhouette = calinski = davies_bouldin = -1
                
                # Calculate cluster size distribution
                if n_clusters > 0:
                    cluster_sizes = [list(labels).count(i) for i in range(n_clusters)]
                    size_distribution = {
                        "min": min(cluster_sizes),
                        "max": max(cluster_sizes), 
                        "median": np.median(cluster_sizes),
                        "std": np.std(cluster_sizes)
                    }
                else:
                    size_distribution = {}
                
                results.append({
                    "params": params,
                    "n_clusters": n_clusters,
                    "noise_pct": n_noise / len(labels) * 100,
                    "silhouette_score": silhouette,
                    "calinski_harabasz": calinski,
                    "davies_bouldin": davies_bouldin,
                    "cluster_size_distribution": size_distribution
                })
                
            except Exception as e:
                results.append({
                    "params": params,
                    "error": str(e)
                })
        
        return results
    
    def _analyze_current_params(self, embeddings: np.ndarray, params: Dict) -> Dict:
        """Analyze current parameter configuration"""
        analysis = {
            "current_params": params,
            "issues": [],
            "observations": []
        }
        
        # Parameter-specific analysis
        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            n_samples = len(embeddings)
            
            if n_neighbors > n_samples * 0.1:
                analysis["issues"].append(f"n_neighbors ({n_neighbors}) is very high relative to sample size ({n_samples})")
            elif n_neighbors < 15:
                analysis["observations"].append(f"n_neighbors ({n_neighbors}) is quite low - may miss global structure")
        
        if "min_cluster_size" in params:
            min_cluster_size = params["min_cluster_size"]
            n_samples = len(embeddings)
            
            if min_cluster_size < n_samples * 0.001:  # Less than 0.1% of data
                analysis["issues"].append(f"min_cluster_size ({min_cluster_size}) may be too small for {n_samples} samples")
            elif min_cluster_size > n_samples * 0.05:  # More than 5% of data
                analysis["issues"].append(f"min_cluster_size ({min_cluster_size}) may be too large, could create very few clusters")
        
        return analysis
    
    def _generate_recommendations(self, umap_results: Dict, hdbscan_results: Dict, 
                                current_analysis: Dict) -> List[str]:
        """Generate parameter recommendations based on analysis"""
        recommendations = []
        
        # UMAP recommendations
        if "error" not in umap_results and umap_results:
            best_umap = max(umap_results, key=lambda x: x.get("silhouette_score", -1))
            if best_umap.get("silhouette_score", -1) > 0.3:
                params = best_umap["params"]
                recommendations.append(f"Consider UMAP params: n_neighbors={params['n_neighbors']}, "
                                     f"min_dist={params['min_dist']}, n_components={params['n_components']}")
        
        # HDBSCAN recommendations  
        if "error" not in hdbscan_results and hdbscan_results:
            # Find configuration with good balance of clusters and silhouette score
            valid_results = [r for r in hdbscan_results if "error" not in r and r["n_clusters"] > 1]
            if valid_results:
                best_hdbscan = max(valid_results, key=lambda x: x.get("silhouette_score", -1))
                params = best_hdbscan["params"]
                recommendations.append(f"Consider HDBSCAN params: min_cluster_size={params['min_cluster_size']}, "
                                     f"min_samples={params['min_samples']}")
        
        # General recommendations based on issues
        issues = current_analysis.get("issues", [])
        for issue in issues:
            if "n_neighbors" in issue and "very high" in issue:
                recommendations.append("Reduce n_neighbors to 15-50 for better local structure preservation")
            elif "min_cluster_size" in issue and "too small" in issue:
                recommendations.append("Increase min_cluster_size to at least 100-500 for large datasets")
            elif "min_cluster_size" in issue and "too large" in issue:
                recommendations.append("Reduce min_cluster_size to allow more granular clustering")
        
        return recommendations
    
    def _plot_diversity_analysis(self, distances: np.ndarray, norms: np.ndarray, 
                               explained_variance: np.ndarray):
        """Create visualization plots for diversity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distance distribution
        axes[0, 0].hist(distances, bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Cosine Distance Distribution')
        axes[0, 0].set_xlabel('Cosine Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.3f}')
        axes[0, 0].legend()
        
        # Embedding norms
        axes[0, 1].hist(norms, bins=50, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Embedding Norm Distribution')
        axes[0, 1].set_xlabel('L2 Norm')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(norms), color='red', linestyle='--', label=f'Mean: {np.mean(norms):.3f}')
        axes[0, 1].legend()
        
        # PCA variance explained
        axes[1, 0].plot(range(1, min(51, len(explained_variance) + 1)), 
                       explained_variance[:50], 'bo-')
        axes[1, 0].set_title('PCA Explained Variance Ratio')
        axes[1, 0].set_xlabel('Principal Component')
        axes[1, 0].set_ylabel('Variance Explained')
        axes[1, 0].grid(True)
        
        # Cumulative variance
        cumvar = np.cumsum(explained_variance)
        axes[1, 1].plot(range(1, min(101, len(cumvar) + 1)), cumvar[:100], 'ro-')
        axes[1, 1].axhline(0.95, color='green', linestyle='--', label='95% threshold')
        axes[1, 1].set_title('Cumulative Variance Explained')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Variance')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.work_dir / 'diversity_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _interpret_separation(self, metrics: Dict) -> str:
        """Interpret methodological separation results"""
        separation_quality = metrics["separation_quality"]
        
        if separation_quality > 0.1:
            return "GOOD: Embeddings show clear methodological separation"
        elif separation_quality > 0.05:
            return "MODERATE: Some methodological separation detected"
        elif separation_quality > 0:
            return "WEAK: Minimal methodological separation"
        else:
            return "POOR: No clear methodological separation - embeddings may not capture method differences"
    
    def run_full_diagnostics(self, chunks_root: Path, current_params: Dict) -> Dict:
        """Run complete diagnostic analysis"""
        print("="*60)
        print("EMBEDDING CLUSTERING DIAGNOSTICS")
        print("="*60)
        
        # Load sample data
        embeddings, texts = self.load_sample_data(chunks_root)
        
        # Run all analyses
        diversity_analysis = self.diagnose_embedding_diversity(embeddings)
        separation_analysis = self.test_methodological_separation(embeddings, texts)
        parameter_analysis = self.analyze_clustering_parameters(embeddings, current_params)
        
        # Compile results
        full_results = {
            "sample_info": {
                "n_embeddings": len(embeddings),
                "embedding_dim": embeddings.shape[1],
                "n_texts": len(texts)
            },
            "diversity_analysis": diversity_analysis,
            "methodological_separation": separation_analysis,
            "parameter_analysis": parameter_analysis
        }
        # Print summary
        self._print_summary(full_results)
        
        # Save results
        results_path = self.work_dir / "diagnostic_results.json"
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\nDiagnostic results saved to: {results_path}")
        
        return full_results
    
    def _print_summary(self, results: Dict):
        """Print diagnostic summary"""
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        
        # Diversity issues
        diversity = results["diversity_analysis"]
        if diversity.get("issues"):
            print("\n🚨 EMBEDDING DIVERSITY ISSUES:")
            for issue in diversity["issues"]:
                print(f"  - {issue}")
        
        # Separation quality
        separation = results["methodological_separation"]
        if "overall" in separation:
            print(f"\n📊 METHODOLOGICAL SEPARATION:")
            print(f"  - {separation.get('interpretation', 'Unknown')}")
            print(f"  - Separation quality: {separation['overall']['separation_quality']:.4f}")
        
        # Parameter recommendations
        param_analysis = results["parameter_analysis"]
        if param_analysis.get("recommendations"):
            print(f"\n💡 PARAMETER RECOMMENDATIONS:")
            for rec in param_analysis["recommendations"]:
                print(f"  - {rec}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic script for embedding clustering issues")
    
    parser.add_argument("--chunks_root", required=True, help="Root directory containing chunk folders")
    parser.add_argument("--work_dir", required=True, help="Directory for diagnostic outputs")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of embeddings to sample for analysis")
    
    # Current configuration
    parser.add_argument("--umap_dim", type=int, default=100, help="Current UMAP dimensions")
    parser.add_argument("--n_neighbors", type=int, default=280, help="Current UMAP n_neighbors")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Current HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=5, help="Current HDBSCAN min_samples")
    
    args = parser.parse_args()
    
    # Setup
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Current parameters
    current_params = {
        "umap_dim": args.umap_dim,
        "n_neighbors": args.n_neighbors,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples
    }
    
    # Run diagnostics
    diagnostics = EmbeddingDiagnostics(work_dir, args.sample_size)
    results = diagnostics.run_full_diagnostics(Path(args.chunks_root), current_params)
    
    print(f"\n✅ Diagnostics completed! Check {work_dir} for detailed results.")


if __name__ == "__main__":
    main()