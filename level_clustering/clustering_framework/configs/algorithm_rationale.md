# Algorithm Selection Rationale

## Why Only K-means and HDBSCAN?

### K-means
- Baseline algorithm that all reviewers understand
- Works well when number of clusters is known
- Fast and scalable to large datasets
- Produces convex, spherical clusters suitable for topic modeling
- Easy to interpret cluster centers as topic centroids

### HDBSCAN
- Handles noise and outliers automatically
- No need to specify number of clusters a priori
- Finds clusters of varying densities
- More robust to parameter choices than DBSCAN
- Produces hierarchical clustering structure for analysis

## Why UMAP for Dimensionality Reduction?

- Preserves both local and global structure
- Faster than t-SNE for large datasets
- Provides meaningful inter-cluster distances
- More stable and reproducible with fixed seed
- Better at preserving continuity of data
