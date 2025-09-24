# Methodology Clustering Pipeline

This directory contains the comprehensive clustering framework for grouping astrophysics papers by their methodological approaches using UMAP dimensionality reduction and advanced clustering algorithms.

## Files

### Main Clustering Scripts
- **`hdb_full_embeddings.py`**: HDBSCAN clustering with full embedding processing
- **`kmeans_full_embeddings.py`**: K-means clustering with full embedding processing

### Clustering Framework (`level_clustering/`)
- **`clustering_framework/thesis_clustering_framework.py`**: Main clustering pipeline class
- **`clustering_framework/thesis_master_runner.py`**: Master runner for distributed processing
- **`clustering_framework/thesis_results_analyzer.py`**: Results analysis and visualization
- **`clustering_framework/pipeline_comparison_evaluator.py`**: Pipeline comparison and evaluation

### GPU-Accelerated Clustering
- **`clustering_gpu.py`**: GPU-accelerated clustering implementation
- **`clustering_gpu2.py`**: Enhanced GPU clustering with memory optimization
- **`clustering_gpu3.py`**: Advanced GPU clustering with multi-GPU support

### Specialized Clustering
- **`specter2_clustering_paras.py`**: SPECTER2-based paragraph clustering
- **`specter2_embeddings.py`**: SPECTER2 embedding generation
- **`vf_kmeans.py`**: Vectorized K-means implementation

### Visualization and Analysis
- **`visualize_families.py`**: 2D/3D visualization of clustering results
- **`cluster_profiler.py`**: Cluster profiling and analysis
- **`memmap_to_npy.py`**: Memory-mapped array conversion utilities

## Usage

### 1. Basic Clustering

```bash
# HDBSCAN clustering
python hdb_full_embeddings.py \
    --embeddings_file results/embeddings/bge-large-en-v1.5.npy \
    --output_dir results/clusters/hdbscan \
    --min_cluster_size 10 \
    --min_samples 5

# K-means clustering
python kmeans_full_embeddings.py \
    --embeddings_file results/embeddings/bge-large-en-v1.5.npy \
    --output_dir results/clusters/kmeans \
    --n_clusters 50
```

### 2. GPU-Accelerated Clustering

```bash
# Single GPU
python clustering_gpu.py \
    --embeddings_file results/embeddings/bge-large-en-v1.5.npy \
    --output_dir results/clusters/gpu \
    --device cuda:0 \
    --batch_size 1000

# Multi-GPU
python clustering_gpu3.py \
    --embeddings_file results/embeddings/bge-large-en-v1.5.npy \
    --output_dir results/clusters/multi_gpu \
    --devices cuda:0,cuda:1,cuda:2,cuda:3
```

### 3. Comprehensive Pipeline

```bash
python clustering_framework/thesis_master_runner.py \
    --config configs/thesis_config.yaml \
    --input_dir results/embeddings \
    --output_dir results/clusters/full_pipeline
```

### 4. Visualization

```bash
python visualize_families.py \
    --clusters_file results/clusters/hdbscan/clusters.csv \
    --embeddings_file results/embeddings/bge-large-en-v1.5.npy \
    --output_dir results/visualizations
```

## Architecture

### Clustering Pipeline

The clustering system follows a multi-stage approach:

1. **Preprocessing**: Embedding normalization and dimensionality reduction
2. **Clustering**: Application of clustering algorithms (K-means, HDBSCAN)
3. **Post-processing**: Cluster refinement and validation
4. **Visualization**: 2D/3D visualization using UMAP/t-SNE
5. **Analysis**: Cluster profiling and quality assessment

### Key Components

#### UMAP Dimensionality Reduction
```python
umap_reducer = umap.UMAP(
    n_components=2,  # For visualization
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine'
)
```

#### HDBSCAN Clustering
```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean',
    cluster_selection_epsilon=0.0
)
```

#### K-means Clustering
```python
kmeans = KMeans(
    n_clusters=50,
    random_state=42,
    n_init=10,
    max_iter=300
)
```

## Configuration

### Clustering Parameters

```yaml
# configs/thesis_config.yaml
clustering:
  algorithms:
    - name: "hdbscan"
      params:
        min_cluster_size: 10
        min_samples: 5
        metric: "euclidean"
    - name: "kmeans"
      params:
        n_clusters: 50
        random_state: 42
  
  preprocessing:
    umap:
      n_components: 2
      n_neighbors: 15
      min_dist: 0.1
      metric: "cosine"
    
    normalization:
      method: "l2"
      per_feature: false
```

### GPU Configuration

```python
gpu_config = {
    'device': 'cuda:0',
    'batch_size': 1000,
    'memory_fraction': 0.8,
    'mixed_precision': True
}
```

## Supported Algorithms

### Clustering Algorithms
- **HDBSCAN**: Hierarchical density-based clustering
- **K-means**: Centroid-based clustering
- **Mini-batch K-means**: Memory-efficient K-means
- **Gaussian Mixture**: Probabilistic clustering

### Dimensionality Reduction
- **UMAP**: Uniform Manifold Approximation and Projection
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding
- **PCA**: Principal Component Analysis

## Dependencies

- `umap-learn`: UMAP dimensionality reduction
- `hdbscan`: HDBSCAN clustering
- `scikit-learn`: K-means and other algorithms
- `cupy`: GPU-accelerated computations
- `plotly`: Interactive visualizations
- `matplotlib`: Static visualizations
- `pandas`: Data manipulation
- `numpy`: Numerical operations

## Performance

### Processing Speed

- **CPU (HDBSCAN)**: ~1000 samples/second
- **GPU (K-means)**: ~10000 samples/second
- **Multi-GPU**: ~40000 samples/second (4 GPUs)

### Memory Usage

- **CPU**: ~2GB for 100K samples
- **GPU**: ~4GB for 100K samples
- **Multi-GPU**: ~8GB total for 100K samples

## Output Format

### Cluster Results

```csv
paper_id,cluster_id,confidence,methodology_type
astro-ph/1234567,0,0.95,observational
astro-ph/1234568,1,0.87,theoretical
astro-ph/1234569,0,0.92,observational
```

### Visualization Files

- **2D Plot**: `families_visualization_2d.png`
- **3D Interactive**: `families_visualization_3d.html`
- **HDBSCAN 3D**: `families_visualization_3d_hdbscan.html`

### Cluster Profiles

```json
{
  "cluster_id": 0,
  "size": 150,
  "methodology_type": "observational",
  "representative_papers": ["astro-ph/1234567", "astro-ph/1234568"],
  "keywords": ["photometry", "spectroscopy", "imaging"]
}
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Use mini-batch algorithms or reduce batch size
2. **GPU Memory**: Reduce batch size or use CPU fallback
3. **Slow Clustering**: Enable GPU acceleration or reduce data size

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Related Documentation

- [Embedding Generation](../embeddings/README.md)
- [Evaluation Pipeline](../evaluation/README.md)
- [Main Documentation](../docs/)
