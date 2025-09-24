# Configuration Files

This directory contains configuration files for the entire methodology classification and clustering pipeline.

## Files

- **`thesis_config.yaml`**: Main configuration file for the complete pipeline
- **`algorithm_rationale.json`**: Detailed rationale for algorithm choices
- **`algorithm_rationale.md`**: Human-readable algorithm rationale documentation

## Usage

### Main Configuration

```bash
# Use the main configuration
python clustering_framework/thesis_master_runner.py \
    --config configs/thesis_config.yaml \
    --input_dir results/embeddings \
    --output_dir results/clusters
```

### Custom Configuration

```yaml
# Create custom configuration
clustering:
  algorithms:
    - name: "hdbscan"
      params:
        min_cluster_size: 15
        min_samples: 8
    - name: "kmeans"
      params:
        n_clusters: 75

preprocessing:
  umap:
    n_components: 2
    n_neighbors: 20
    min_dist: 0.05
```

## Configuration Structure

### Clustering Configuration

```yaml
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

### Model Configuration

```yaml
models:
  embedding:
    - name: "bge-large-en-v1.5"
      model_name: "BAAI/bge-large-en-v1.5"
      max_length: 512
    - name: "e5-large-v2"
      model_name: "intfloat/e5-large-v2"
      max_length: 512
  
  classification:
    model_name: "allenai/scibert_scivocab_uncased"
    max_length: 512
    batch_size: 4
```

### Processing Configuration

```yaml
processing:
  batch_size: 32
  max_workers: 4
  device: "cuda:0"
  checkpoint_interval: 1000
  output_format: "numpy"
```

## Algorithm Rationale

The `algorithm_rationale.json` file provides detailed justification for algorithm choices:

### HDBSCAN Rationale

- **Hierarchical**: Captures nested cluster structure
- **Density-based**: Handles clusters of varying shapes and sizes
- **Noise handling**: Automatically identifies outliers
- **Parameter-free**: Minimal hyperparameter tuning required

### K-means Rationale

- **Scalability**: Efficient for large datasets
- **Interpretability**: Clear cluster centroids
- **Stability**: Deterministic results with fixed random seed
- **Speed**: Fast convergence for well-separated clusters

### UMAP Rationale

- **Preservation**: Maintains both local and global structure
- **Flexibility**: Works with various distance metrics
- **Quality**: Produces high-quality 2D/3D visualizations
- **Efficiency**: Faster than t-SNE for large datasets

## Dependencies

- `pyyaml`: YAML configuration parsing
- `json`: JSON configuration parsing
- `pandas`: Data manipulation
- `numpy`: Numerical operations

## Related Documentation

- [Clustering Pipeline](../04_clustering/README.md)
- [Main Documentation](../docs/)
