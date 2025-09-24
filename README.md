# Astrophysics Methodology Classification and Clustering

A comprehensive pipeline for automatically identifying and clustering methodological approaches in astrophysics research papers using state-of-the-art NLP techniques.

## Project Overview

This repository contains the complete implementation of a methodology classification and clustering system designed for astrophysics literature analysis. The system combines BERT-based classification, multi-model embedding generation, and advanced clustering techniques to automatically identify and group papers by their methodological approaches.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd astronlp

# Install dependencies
pip install -r requirements.txt

# Run basic pipeline
python examples/basic_pipeline.py
```

### Basic Usage

```python
from classification.bert_classifier_trainer import BERTMetaClassifierTrainer
from clustering.level_clustering.clustering_framework.thesis_clustering_framework import ThesisClusteringPipeline

# Initialize classifier
trainer = BERTMetaClassifierTrainer(
    model_name='allenai/scibert_scivocab_uncased',
    cache_path='./models',
    output_dir='./results'
)

# Train on methodology data
trainer.train('data/methodology_labels/labeled_paragraphs.jsonl')
```

## Repository Structure

```
astronlp/
├── data_processing/            # Data extraction and preprocessing
├── classification/             # BERT-based methodology classification
├── embeddings/                 # Multi-model embedding generation
├── clustering/                 # UMAP + K-means/HDBSCAN clustering
├── summarization/              # Text summarization pipeline
├── evaluation/                 # Evaluation and analysis tools
├── utilities/                  # Helper scripts and utilities
├── configs/                    # Configuration files
├── docs/                       # Documentation and thesis
├── examples/                   # Usage examples
├── tests/                      # Unit tests
└── results/                    # Output directory
```

## Research Contributions

- **Novel Methodology Classification**: BERT-based approach for identifying methodology vs non-methodology content in scientific papers
- **Multi-scale Clustering**: Hierarchical clustering at both paragraph and paper levels
- **Semi-supervised Learning**: Bootstrap approach with confidence-based pseudo-labeling
- **LLM-integrated Evaluation**: GPT-4 based cluster quality assessment and interpretation
- **Distributed Processing**: Scalable pipeline for large-scale scientific literature analysis

## Pipeline Components

### Data Processing (`data_processing/`)
- Extract methodology paragraphs from arXiv papers
- Apply heuristic rules for methodology identification
- Preprocess and tokenize text for BERT models

### Classification (`classification/`)
- BERT-based methodology classification
- Semi-supervised learning with bootstrap training
- Multi-GPU distributed training support

### Embedding Generation (`embeddings/`)
- Multi-model embedding extraction (BGE, E5, etc.)
- Fault-tolerant processing with checkpointing
- GPU-optimized for large-scale processing

### Clustering (`clustering/`)
- UMAP dimensionality reduction
- K-means and HDBSCAN clustering algorithms
- Comprehensive evaluation and visualization

### Summarization (`summarization/`)
- Multi-model text summarization (BART, Pegasus)
- Parallel processing across multiple models
- Attention weight analysis

### Evaluation (`evaluation/`)
- LLM-based cluster quality assessment
- Multi-metric evaluation (ROUGE, BERTScore, SummaC)
- Representative example selection

## Key Features

- **End-to-end Pipeline**: From raw papers to clustered methodologies
- **Multiple Perspectives**: Both paragraph-level and paper-level analysis
- **Robust Evaluation**: Multiple evaluation methods including LLM assessment
- **Scalability**: Designed for large-scale processing on HPC systems
- **Reproducibility**: Extensive configuration management and checkpointing

## Results

The system achieves:
- High accuracy in methodology classification (>90% F1-score)
- Meaningful clustering of papers by methodological approach
- Interpretable cluster representations through LLM evaluation
- Scalable processing of large astrophysics literature datasets

## Getting Started

1. **Data Preparation**: See `data_processing/README.md`
2. **Model Training**: See `classification/README.md`
3. **Clustering**: See `clustering/README.md`
4. **Evaluation**: See `evaluation/README.md`

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Examples](docs/usage_examples.md)
- [API Reference](docs/api_reference.md)
- [Methodology](docs/methodology.md)

## Contributing

This is a research project. For questions or collaboration, please contact the authors.

## License

[Add your license information here]

## Contact

[Add your contact information here]

---

**Note**: This repository contains the complete implementation of the methodology classification and clustering system described in the accompanying thesis. All scripts are designed for reproducibility and can be run on both single machines and HPC clusters.
