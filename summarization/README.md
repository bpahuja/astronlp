# Text Summarization Pipeline

This directory contains the multi-model text summarization system for generating concise summaries of methodology paragraphs and clustering results.

## Files

- **`gen_multi.py`**: Multi-model text summarization with parallel processing and attention weight analysis

## Usage

### Basic Usage

```bash
python gen_multi.py \
    --input_file data/methodology_labels/labeled_paragraphs.jsonl \
    --output_dir results/summaries \
    --models bart pegasus \
    --max_length 512 \
    --batch_size 4
```

### Advanced Usage

```bash
python gen_multi.py \
    --input_file data/methodology_labels/labeled_paragraphs.jsonl \
    --output_dir results/summaries \
    --models bart pegasus t5 \
    --max_length 512 \
    --batch_size 4 \
    --max_workers 2 \
    --attention_analysis \
    --device cuda:0
```

## Architecture

### Multi-Model Pipeline

The summarization system supports multiple state-of-the-art models:

1. **BART**: Facebook's denoising autoencoder for sequence-to-sequence tasks
2. **Pegasus**: Google's pre-training with extracted gap-sentences
3. **T5**: Google's Text-to-Text Transfer Transformer
4. **Custom Models**: Support for any HuggingFace-compatible model

### Key Features

- **Parallel Processing**: Multi-model inference with concurrent execution
- **Attention Analysis**: Visualization of attention weights for interpretability
- **Batch Processing**: Efficient processing of large datasets
- **Memory Management**: Optimized memory usage for large models
- **Progress Tracking**: Real-time progress monitoring

## Configuration

### Model Configuration

```python
models = {
    'bart': {
        'model_name': 'facebook/bart-large-cnn',
        'max_length': 512,
        'min_length': 50,
        'num_beams': 4,
        'early_stopping': True
    },
    'pegasus': {
        'model_name': 'google/pegasus-cnn_dailymail',
        'max_length': 512,
        'min_length': 50,
        'num_beams': 4,
        'early_stopping': True
    },
    't5': {
        'model_name': 't5-large',
        'max_length': 512,
        'min_length': 50,
        'num_beams': 4,
        'early_stopping': True
    }
}
```

### Processing Parameters

```python
config = {
    'batch_size': 4,
    'max_workers': 2,
    'device': 'cuda:0',
    'attention_analysis': True,
    'output_format': 'jsonl'
}
```

## Supported Models

### BART Models
- `facebook/bart-large-cnn`: CNN/DailyMail fine-tuned
- `facebook/bart-large-xsum`: XSum fine-tuned
- `facebook/bart-base`: Base model

### Pegasus Models
- `google/pegasus-cnn_dailymail`: CNN/DailyMail fine-tuned
- `google/pegasus-xsum`: XSum fine-tuned
- `google/pegasus-large`: Large model

### T5 Models
- `t5-large`: Large T5 model
- `t5-base`: Base T5 model
- `t5-small`: Small T5 model

## Dependencies

- `torch`: PyTorch for model inference
- `transformers`: HuggingFace transformers library
- `sentencepiece`: T5 tokenization
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `tqdm`: Progress bars
- `concurrent.futures`: Parallel processing
- `matplotlib`: Attention visualization

## Performance

### Processing Speed

- **BART Large**: ~50 paragraphs/minute (RTX 3090)
- **Pegasus Large**: ~45 paragraphs/minute (RTX 3090)
- **T5 Large**: ~40 paragraphs/minute (RTX 3090)

### Memory Usage

- **BART Large**: ~8GB GPU memory
- **Pegasus Large**: ~7GB GPU memory
- **T5 Large**: ~6GB GPU memory

## Output Format

### Summary Results

```json
{
  "paper_id": "astro-ph/1234567",
  "paragraph_index": 5,
  "original_text": "We used the following methodology...",
  "summary": "The methodology involves...",
  "model": "bart",
  "confidence": 0.95,
  "attention_weights": [[0.1, 0.2, ...], ...]
}
```

### Attention Analysis

```json
{
  "model": "bart",
  "attention_analysis": {
    "layer_0": {
      "head_0": [0.1, 0.2, 0.3, ...],
      "head_1": [0.2, 0.1, 0.4, ...]
    }
  }
}
```

## Use Cases

### 1. Methodology Summarization

Generate concise summaries of methodology paragraphs for:
- Quick understanding of research approaches
- Literature review preparation
- Methodology comparison

### 2. Cluster Summarization

Create summaries for clustering results:
- Cluster characterization
- Methodology type identification
- Research trend analysis

### 3. Paper Summarization

Summarize entire papers or sections:
- Abstract generation
- Section summarization
- Key findings extraction

## Attention Analysis

The system provides detailed attention weight analysis:

### Attention Visualization

```python
# Generate attention heatmap
attention_heatmap = visualize_attention(
    attention_weights=attention_weights,
    tokens=tokenized_text,
    output_file='attention_heatmap.png'
)
```

### Attention Metrics

- **Attention Entropy**: Measure of attention distribution
- **Attention Focus**: Concentration of attention weights
- **Cross-Head Attention**: Attention patterns across heads

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **Slow Processing**: Enable mixed precision or use more workers
3. **Poor Quality**: Try different models or adjust parameters

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Related Documentation

- [Classification Pipeline](../classification/README.md)
- [Clustering Pipeline](../clustering/README.md)
- [Evaluation Pipeline](../evaluation/README.md)
- [Main Documentation](../docs/)
