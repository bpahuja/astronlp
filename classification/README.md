# Methodology Classification

This directory contains the BERT-based methodology classification system for distinguishing methodology vs non-methodology content in astrophysics papers.

## Files

- **`bert_classifier_trainer.py`**: Main BERT classifier with LSTM layer for paragraph-level understanding
- **`bootstrap_bert.py`**: Semi-supervised learning with confidence-based pseudo-labeling
- **`bootstrap_multi.py`**: Distributed version with multi-GPU support and model weight averaging
- **`classifier_inference_multi.py`**: Distributed inference system for large-scale processing

## Usage

### 1. Basic Training

```bash
python bert_classifier_trainer.py \
    --model_name allenai/scibert_scivocab_uncased \
    --data_path data/methodology_labels/labeled_paragraphs.jsonl \
    --output_dir results/classification \
    --epochs 5 \
    --batch_size 4
```

### 2. Semi-supervised Bootstrap Training

```bash
python bootstrap_bert.py
```

This script:
- Uses pre-trained model for initial classification
- Iteratively adds high-confidence pseudo-labels
- Implements checkpointing for fault tolerance

### 3. Distributed Training

```bash
# Controller job
python bootstrap_multi.py --job_id 0 --num_jobs 4

# Worker jobs
python bootstrap_multi.py --job_id 1 --num_jobs 4
python bootstrap_multi.py --job_id 2 --num_jobs 4
python bootstrap_multi.py --job_id 3 --num_jobs 4
```

### 4. Distributed Inference

```bash
# Controller job
python classifier_inference_multi.py \
    --job_id 0 --num_jobs 4 \
    --model_pt_path results/classification/model.pt \
    --input_file data/unlabelled_methodology/unlabelled_paragraphs.jsonl \
    --output_file results/predictions.jsonl

# Worker jobs
python classifier_inference_multi.py --job_id 1 --num_jobs 4 [same args]
```

## Architecture

### BERTMetaClassifier

The core model combines:
- **BERT Encoder**: For contextual understanding
- **LSTM Layer**: For paragraph-level sequence modeling
- **Classification Head**: Binary classification (methodology/non-methodology)

```python
class BERTMetaClassifier(nn.Module):
    def __init__(self, model_name, cache_path, hidden_dim=768, 
                 meta_hidden=256, num_classes=2, dropout=0.2):
        self.bert = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(hidden_dim, meta_hidden, bidirectional=True)
        self.classifier = nn.Linear(meta_hidden * 2, num_classes)
```

### Key Features

- **Chunking**: Handles long paragraphs by splitting into chunks
- **Attention Masking**: Proper handling of variable-length sequences
- **Gradient Scaling**: Mixed precision training for efficiency
- **Checkpointing**: Robust recovery from failures

## Configuration

### Training Parameters

```python
config = {
    'learning_rate': 1e-5,
    'batch_size': 4,
    'meta_hidden': 256,
    'dropout': 0.2,
    'epochs': 5,
    'optimizer': 'adamw',
    'max_length': 512,
    'num_classes': 2
}
```

### Bootstrap Parameters

```python
bootstrap_config = {
    'confidence_threshold': 0.95,
    'num_iterations': 5,
    'max_new_labels': 30000,
    'min_new_labels': 1000,
    'patience': 2
}
```

## Performance

### Classification Accuracy

- **F1-Score**: >90% on held-out test set
- **Precision**: >92% for methodology detection
- **Recall**: >88% for methodology detection

### Training Performance

- **Single GPU**: ~2 hours for 10K samples
- **Multi-GPU**: ~30 minutes for 10K samples (4 GPUs)
- **Memory Usage**: ~8GB per GPU

## Dependencies

- `torch`: Deep learning framework
- `transformers`: BERT models and tokenizers
- `sklearn`: Metrics and data splitting
- `pandas`: Data manipulation
- `tqdm`: Progress bars
- `lmdb`: Efficient data storage

## Evaluation Metrics

The system evaluates using:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis
- **Confidence Scores**: Model uncertainty quantification

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Enable mixed precision training
3. **Poor Performance**: Check data quality and model configuration

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Related Documentation

- [Data Processing](../data_processing/README.md)
- [Embedding Generation](../embeddings/README.md)
- [Main Documentation](../docs/)
