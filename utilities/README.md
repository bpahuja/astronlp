# Utilities and Helper Scripts

This directory contains utility scripts and helper functions that support the main pipeline components.

## Files

- **`merge_tokens.py`**: Utility for merging and processing tokenized data

## Usage

### Token Merging

```bash
python merge_tokens.py \
    --input_dir data/tokens \
    --output_file data/merged_tokens.jsonl \
    --chunk_size 1000
```

## Architecture

### Utility Functions

The utilities directory provides:

1. **Data Processing**: Token merging and data consolidation
2. **File Management**: Batch processing and file operations
3. **Format Conversion**: Data format transformations
4. **Validation**: Data quality checks and validation

### Key Features

- **Batch Processing**: Efficient processing of large datasets
- **Memory Management**: Optimized memory usage for large files
- **Error Handling**: Robust error handling and recovery
- **Progress Tracking**: Real-time progress monitoring

## Configuration

### Token Merging Parameters

```python
config = {
    'input_dir': 'data/tokens',
    'output_file': 'data/merged_tokens.jsonl',
    'chunk_size': 1000,
    'max_workers': 4,
    'compression': 'gzip'
}
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `concurrent.futures`: Parallel processing
- `json`: JSON processing
- `gzip`: Compression utilities

## Performance

### Processing Speed

- **Token Merging**: ~1000 files/minute
- **Memory Usage**: ~1GB for 100K files
- **Output Size**: ~50MB for 100K files (compressed)

## Related Documentation

- [Data Processing](../data_processing/README.md)
- [Classification Pipeline](../classification/README.md)
- [Main Documentation](../docs/)
