# Multi-Model Embedding Generation

This directory contains the embedding generation pipeline that creates high-quality vector representations of methodology paragraphs using multiple state-of-the-art embedding models.

## Files

- **`embed_multi.py`**: Multi-model embedding generation with fault tolerance and checkpointing

## Usage

### Basic Usage

```bash
python embed_multi.py \
    --input_file data/methodology_labels/labeled_paragraphs.jsonl \
    --output_dir results/embeddings \
    --models bge-large-en-v1.5 e5-large-v2 \
    --batch_size 32
```

### Advanced Usage

```bash
python embed_multi.py \
    --input_file data/methodology_labels/labeled_paragraphs.jsonl \
    --output_dir results/embeddings \
    --models bge-large-en-v1.5 e5-large-v2 sentence-transformers/all-MiniLM-L6-v2 \
    --batch_size 32 \
    --max_workers 4 \
    --checkpoint_interval 1000 \
    --device cuda:0
```

## Architecture

### Multi-Model Pipeline

The embedding generation system supports multiple embedding models:

1. **BGE (BAAI General Embedding)**: High-quality general-purpose embeddings
2. **E5 (Embeddings from End-to-End)**: Microsoft's state-of-the-art embeddings
3. **Sentence Transformers**: Various pre-trained models
4. **Custom Models**: Support for any HuggingFace-compatible model

### Key Features

- **Fault Tolerance**: Automatic recovery from failures with checkpointing
- **Parallel Processing**: Multi-worker support for faster processing
- **Memory Efficiency**: Batch processing with configurable batch sizes
- **Model Caching**: Automatic model downloading and caching
- **Progress Tracking**: Real-time progress monitoring

## Configuration

### Model Configuration

```python
models = {
    'bge-large-en-v1.5': {
        'model_name': 'BAAI/bge-large-en-v1.5',
        'max_length': 512,
        'normalize_embeddings': True
    },
    'e5-large-v2': {
        'model_name': 'intfloat/e5-large-v2',
        'max_length': 512,
        'prefix': 'passage:'
    }
}
```

### Processing Parameters

```python
config = {
    'batch_size': 32,
    'max_workers': 4,
    'checkpoint_interval': 1000,
    'device': 'cuda:0',
    'output_format': 'numpy'
}
```

## Supported Models

### BGE Models
- `BAAI/bge-large-en-v1.5`: 1024-dimensional embeddings
- `BAAI/bge-base-en-v1.5`: 768-dimensional embeddings
- `BAAI/bge-small-en-v1.5`: 384-dimensional embeddings

### E5 Models
- `intfloat/e5-large-v2`: 1024-dimensional embeddings
- `intfloat/e5-base-v2`: 768-dimensional embeddings
- `intfloat/e5-small-v2`: 384-dimensional embeddings

### Sentence Transformers
- `sentence-transformers/all-MiniLM-L6-v2`: 384-dimensional embeddings
- `sentence-transformers/all-mpnet-base-v2`: 768-dimensional embeddings
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: 384-dimensional embeddings

## Dependencies

- `torch`: PyTorch for model inference
- `transformers`: HuggingFace transformers library
- `sentence-transformers`: Sentence transformer models
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `tqdm`: Progress bars
- `concurrent.futures`: Parallel processing

## Performance

### Processing Speed

- **BGE Large**: ~1000 paragraphs/minute (RTX 3090)
- **E5 Large**: ~800 paragraphs/minute (RTX 3090)
- **Sentence Transformers**: ~1500 paragraphs/minute (RTX 3090)

### Memory Usage

- **BGE Large**: ~6GB GPU memory
- **E5 Large**: ~5GB GPU memory
- **Sentence Transformers**: ~2GB GPU memory

## Output Format

Embeddings are saved in multiple formats:

### NumPy Format
```python
# Shape: (num_paragraphs, embedding_dim)
embeddings = np.load('results/embeddings/bge-large-en-v1.5.npy')
```

### JSONL Format
```json
{
  "paper_id": "astro-ph/1234567",
  "paragraph_index": 5,
  "embedding": [0.1, 0.2, ...],
  "model": "bge-large-en-v1.5"
}
```

### Metadata
```json
{
  "model_name": "BAAI/bge-large-en-v1.5",
  "embedding_dim": 1024,
  "num_paragraphs": 10000,
  "processing_time": 120.5,
  "batch_size": 32
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Download Errors**: Check internet connection and disk space
3. **Slow Processing**: Increase batch size or use more workers

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
