# Clustering Algorithm Comparison Report
Generated: 2025-09-10 13:59:48
Experiment ID: 20250910_135947

## Executive Summary
- Total experiments: 12
- Successful experiments: 0
- Failed experiments: 12


## Recommendations
## Failed Experiments
- **kmeans_euclidean_raw**: KMeans.__init__() got an unexpected keyword argument 'n_jobs'
- **kmeans_euclidean_standard**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by StandardScaler.
- **kmeans_cosine_l2norm**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by Normalizer.
- **kmeans_euclidean_tfidf**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by TfidfTransformer.
- **kmeans_euclidean_pca20**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by PCA.
- **kmeans_euclidean_svd30**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 2 is required by TruncatedSVD.
- **kmeans_euclidean_nmf15**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by NMF.
- **hdbscan_euclidean_standard_min10**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by StandardScaler.
- **hdbscan_cosine_l2norm_min10**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by Normalizer.
- **hdbscan_euclidean_standard_min15**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by StandardScaler.
- **hdbscan_manhattan_robust_min10**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 1 is required by RobustScaler.
- **hdbscan_euclidean_svd30_min10**: Found array with 0 feature(s) (shape=(1375099, 0)) while a minimum of 2 is required by TruncatedSVD.

## Generated Files
- `experiment_results.json`: Detailed results summary
- `experiment_metrics.csv`: Metrics comparison table
- `complete_results.pkl`: Complete results with clustering labels
- `results_analysis.json`: Statistical analysis
- `comparison_tables.xlsx`: Comparison tables for algorithms, preprocessing, distances
- `experiment_visualizations.png`: Comprehensive visualization dashboard
- `detailed_comparison.png`: Detailed method comparison plots