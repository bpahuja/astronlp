# Data Processing Pipeline

This directory contains scripts for extracting, preprocessing, and preparing astrophysics papers for methodology classification and clustering.

## Files

- **`create_dataset_methodolodgy.py`**: Main script for extracting methodology paragraphs from arXiv papers using heuristic rules
- **`create_methodology_dataset.py`**: Processes labeled data to create clean datasets for training
- **`pretokenize_paragraphs.py`**: Pre-tokenizes text using BERT tokenizers and stores in LMDB for efficient access
- **`merge_and_sort_jsonl.py`**: Combines labeled and unlabeled data for training

## Usage

### 1. Extract Methodology Paragraphs

```bash
python create_dataset_methodolodgy.py
```

This script:
- Reads arXiv papers from XML format
- Applies heuristic rules to identify methodology sections
- Extracts paragraphs with methodology labels
- Outputs JSONL format for further processing

### 2. Create Training Dataset

```bash
python create_methodology_dataset.py
```

This script:
- Processes labeled methodology data
- Creates clean training datasets
- Handles data balancing and preprocessing

### 3. Pre-tokenize Text

```bash
python pretokenize_paragraphs.py \
    --input_jsonl data/methodology_labels/labeled_paragraphs.jsonl \
    --output_file data/tokens_astrobert \
    --model_name adsabs/astroBERT \
    --max_length 512
```

### 4. Merge Data Sources

```bash
python merge_and_sort_jsonl.py
```

## Configuration

The data processing pipeline uses several configuration parameters:

- **MIN_PARAGRAPH_WORDS**: Minimum word count for paragraph inclusion (default: 25)
- **METHOD_HEADINGS**: Keywords for identifying methodology sections
- **METHOD_REGEXES**: Regex patterns for methodology detection
- **N_WORKERS**: Number of parallel processing workers

## Output Format

All scripts output data in JSONL format with the following structure:

```json
{
  "paper_id": "astro-ph/1234567",
  "section": "Data Analysis",
  "paragraph": "We used the following methodology...",
  "label": "methodology",
  "section_index": 2,
  "paragraph_index": 5
}
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `concurrent.futures`: Parallel processing
- `xml.etree.ElementTree`: XML parsing
- `transformers`: BERT tokenization

## Performance

- **Processing Speed**: ~1000 papers per hour (single machine)
- **Memory Usage**: ~2GB for 10K papers
- **Output Size**: ~50MB per 10K papers (compressed)

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `N_WORKERS` or process in smaller batches
2. **XML Parsing Errors**: Check paper format and encoding
3. **Tokenization Errors**: Verify model name and cache path

### Debug Mode

Enable verbose logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Related Documentation

- [Classification Pipeline](../classification/README.md)
- [Embedding Generation](../embeddings/README.md)
- [Main Documentation](../docs/)
