# Usage Examples

This directory contains practical examples demonstrating how to use the methodology classification and clustering system.

## Files

- **`basic_pipeline.py`**: Complete end-to-end pipeline example
- **`classification_example.py`**: BERT classification example
- **`clustering_example.py`**: Clustering pipeline example
- **`evaluation_example.py`**: Evaluation and analysis example

## Quick Start

### Basic Pipeline

```bash
python examples/basic_pipeline.py
```

This example demonstrates:
- Data preprocessing
- Model training
- Clustering
- Evaluation
- Visualization

### Classification Example

```bash
python examples/classification_example.py
```

This example shows:
- BERT model training
- Semi-supervised learning
- Model evaluation
- Prediction generation

### Clustering Example

```bash
python examples/clustering_example.py
```

This example covers:
- Embedding generation
- UMAP dimensionality reduction
- HDBSCAN clustering
- Visualization

### Evaluation Example

```bash
python examples/evaluation_example.py
```

This example demonstrates:
- LLM-based evaluation
- Multi-metric assessment
- Representative example selection
- Results analysis

## Example Structure

### Basic Pipeline Example

```python
# examples/basic_pipeline.py
from data_processing.create_dataset_methodolodgy import DataProcessor
from classification.bert_classifier_trainer import BERTMetaClassifierTrainer
from clustering.level_clustering.clustering_framework.thesis_clustering_framework import ThesisClusteringPipeline

def main():
    # 1. Data processing
    processor = DataProcessor()
    processor.process_papers('data/arxiv_papers/')
    
    # 2. Classification training
    trainer = BERTMetaClassifierTrainer()
    trainer.train('data/methodology_labels/labeled_paragraphs.jsonl')
    
    # 3. Clustering
    clustering_pipeline = ThesisClusteringPipeline()
    clustering_pipeline.run('results/embeddings/', 'results/clusters/')
    
    # 4. Evaluation
    evaluator = MultiMetricEvaluator()
    evaluator.evaluate('results/clusters/', 'results/evaluations/')

if __name__ == "__main__":
    main()
```

### Classification Example

```python
# examples/classification_example.py
from classification.bert_classifier_trainer import BERTMetaClassifierTrainer

def main():
    # Initialize trainer
    trainer = BERTMetaClassifierTrainer(
        model_name='allenai/scibert_scivocab_uncased',
        cache_path='./models',
        output_dir='./results'
    )
    
    # Train model
    trainer.train('data/methodology_labels/labeled_paragraphs.jsonl')
    
    # Evaluate model
    trainer.evaluate('data/methodology_labels/test_labels.jsonl')
    
    # Generate predictions
    trainer.predict('data/unlabelled_methodology/unlabelled_paragraphs.jsonl')

if __name__ == "__main__":
    main()
```

### Clustering Example

```python
# examples/clustering_example.py
from clustering.level_clustering.clustering_framework.thesis_clustering_framework import ThesisClusteringPipeline

def main():
    # Initialize clustering pipeline
    pipeline = ThesisClusteringPipeline(
        config_path='configs/thesis_config.yaml',
        input_dir='results/embeddings/',
        output_dir='results/clusters/'
    )
    
    # Run clustering
    pipeline.run()
    
    # Visualize results
    pipeline.visualize('results/visualizations/')

if __name__ == "__main__":
    main()
```

### Evaluation Example

```python
# examples/evaluation_example.py
from evaluation.evaluate_clusters_llm import LLMClusterEvaluator
from evaluation.evaluate_multi import MultiMetricEvaluator

def main():
    # LLM-based evaluation
    llm_evaluator = LLMClusterEvaluator(
        model='gpt-4',
        api_key='your-api-key'
    )
    llm_evaluator.evaluate('results/clusters/', 'results/evaluations/llm/')
    
    # Multi-metric evaluation
    multi_evaluator = MultiMetricEvaluator()
    multi_evaluator.evaluate('results/clusters/', 'results/evaluations/multi/')

if __name__ == "__main__":
    main()
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `seaborn`: Statistical visualization
- `tqdm`: Progress bars

## Related Documentation

- [Main README](../README.md)
- [Pipeline Documentation](../)
- [API Reference](../docs/api_reference.md)
