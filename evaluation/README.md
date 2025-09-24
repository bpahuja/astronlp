# Evaluation and Analysis Pipeline

This directory contains comprehensive evaluation tools for assessing the quality and performance of the methodology classification and clustering system.

## Files

- **`evaluate_clusters_llm.py`**: LLM-based cluster quality assessment using GPT-4
- **`evaluate_multi.py`**: Multi-metric evaluation system for classification and clustering
- **`example_selector.py`**: Representative example selection for cluster analysis
- **`Clustering-pipeline1.ipynb`**: Jupyter notebook for interactive clustering analysis

## Usage

### 1. LLM-based Cluster Evaluation

```bash
python evaluate_clusters_llm.py \
    --clusters_file results/clusters/hdbscan/clusters.csv \
    --embeddings_file results/embeddings/bge-large-en-v1.5.npy \
    --output_dir results/evaluations/llm \
    --model gpt-4 \
    --api_key YOUR_API_KEY \
    --max_clusters 10
```

### 2. Multi-metric Evaluation

```bash
python evaluate_multi.py \
    --predictions_file results/classification/predictions.jsonl \
    --ground_truth_file data/methodology_labels/test_labels.jsonl \
    --clusters_file results/clusters/hdbscan/clusters.csv \
    --output_dir results/evaluations/multi_metric
```

### 3. Representative Example Selection

```bash
python example_selector.py \
    --clusters_file results/clusters/hdbscan/clusters.csv \
    --embeddings_file results/embeddings/bge-large-en-v1.5.npy \
    --output_dir results/evaluations/examples \
    --num_examples_per_cluster 5
```

### 4. Interactive Analysis

```bash
jupyter notebook Clustering-pipeline1.ipynb
```

## Architecture

### Evaluation Framework

The evaluation system provides multiple assessment approaches:

1. **LLM-based Evaluation**: GPT-4 powered cluster quality assessment
2. **Traditional Metrics**: Standard clustering and classification metrics
3. **Representative Examples**: Selection of typical cluster examples
4. **Interactive Analysis**: Jupyter notebook for exploratory analysis

### Key Components

#### LLM Cluster Evaluator
```python
class LLMClusterEvaluator:
    def __init__(self, model_name, api_key):
        self.model = model_name
        self.api_key = api_key
    
    def evaluate_cluster(self, cluster_data):
        # Generate cluster description
        # Assess cluster quality
        # Provide recommendations
```

#### Multi-metric Evaluator
```python
class MultiMetricEvaluator:
    def __init__(self):
        self.metrics = {
            'classification': ['accuracy', 'f1', 'precision', 'recall'],
            'clustering': ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        }
```

## Configuration

### LLM Evaluation Parameters

```python
llm_config = {
    'model': 'gpt-4',
    'api_key': 'your-api-key',
    'max_tokens': 1000,
    'temperature': 0.7,
    'max_clusters': 10,
    'timeout': 30
}
```

### Multi-metric Parameters

```python
evaluation_config = {
    'classification_metrics': [
        'accuracy', 'f1_macro', 'f1_micro',
        'precision_macro', 'recall_macro'
    ],
    'clustering_metrics': [
        'silhouette_score', 'calinski_harabasz_score',
        'davies_bouldin_score', 'inertia'
    ],
    'rouge_metrics': ['rouge1', 'rouge2', 'rougeL'],
    'bertscore_metrics': ['precision', 'recall', 'f1']
}
```

## Evaluation Metrics

### Classification Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Confusion Matrix**: Detailed error analysis

### Clustering Metrics

- **Silhouette Score**: Measure of cluster cohesion and separation
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Index**: Average similarity between clusters
- **Inertia**: Sum of squared distances to centroids

### Text Quality Metrics

- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **BERTScore**: BERT-based semantic similarity
- **SummaC**: Summary consistency evaluation

## Dependencies

- `openai`: GPT-4 API access
- `scikit-learn`: Clustering and classification metrics
- `rouge-score`: ROUGE evaluation metrics
- `bert-score`: BERT-based evaluation
- `summac`: Summary consistency evaluation
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `seaborn`: Statistical visualization
- `jupyter`: Interactive notebooks

## Performance

### Evaluation Speed

- **LLM Evaluation**: ~10 clusters/minute (GPT-4)
- **Traditional Metrics**: ~1000 samples/second
- **Representative Examples**: ~100 samples/second

### Cost Analysis

- **GPT-4 API**: ~$0.10 per cluster evaluation
- **Local Evaluation**: Free (computational cost only)

## Output Format

### LLM Evaluation Results

```json
{
  "cluster_id": 0,
  "size": 150,
  "description": "Observational astronomy methods using photometry and spectroscopy",
  "quality_score": 8.5,
  "coherence": 0.9,
  "distinctiveness": 0.85,
  "representative_papers": ["astro-ph/1234567", "astro-ph/1234568"],
  "recommendations": ["Consider splitting into sub-clusters", "Add more diverse examples"]
}
```

### Multi-metric Results

```json
{
  "classification": {
    "accuracy": 0.92,
    "f1_macro": 0.89,
    "precision_macro": 0.91,
    "recall_macro": 0.87
  },
  "clustering": {
    "silhouette_score": 0.65,
    "calinski_harabasz_score": 1250.5,
    "davies_bouldin_score": 0.45
  }
}
```

### Representative Examples

```json
{
  "cluster_id": 0,
  "examples": [
    {
      "paper_id": "astro-ph/1234567",
      "paragraph": "We used photometric observations...",
      "similarity_score": 0.95,
      "representativeness": 0.92
    }
  ]
}
```

## Use Cases

### 1. Cluster Quality Assessment

- Evaluate cluster coherence and distinctiveness
- Identify problematic clusters
- Provide improvement recommendations

### 2. Model Performance Analysis

- Compare different embedding models
- Assess clustering algorithm performance
- Evaluate classification accuracy

### 3. Representative Example Selection

- Select typical examples for each cluster
- Create human-readable cluster descriptions
- Support manual cluster validation

### 4. Interactive Analysis

- Explore clustering results interactively
- Visualize cluster distributions
- Perform ad-hoc analysis

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Implement exponential backoff
2. **Memory Issues**: Process clusters in batches
3. **Slow Evaluation**: Use parallel processing

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Related Documentation

- [Classification Pipeline](../classification/README.md)
- [Clustering Pipeline](../clustering/README.md)
- [Main Documentation](../docs/)
