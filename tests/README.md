# Test Suite

This directory contains comprehensive unit tests and integration tests for the methodology classification and clustering system.

## Files

- **`test_data_processing.py`**: Tests for data processing pipeline
- **`test_classification.py`**: Tests for classification models
- **`test_clustering.py`**: Tests for clustering algorithms
- **`test_evaluation.py`**: Tests for evaluation metrics
- **`test_utilities.py`**: Tests for utility functions

## Running Tests

### Run All Tests

```bash
python -m pytest tests/
```

### Run Specific Test Categories

```bash
# Data processing tests
python -m pytest tests/test_data_processing.py

# Classification tests
python -m pytest tests/test_classification.py

# Clustering tests
python -m pytest tests/test_clustering.py

# Evaluation tests
python -m pytest tests/test_evaluation.py
```

### Run with Coverage

```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Structure

### Unit Tests

Each component has comprehensive unit tests:

```python
# tests/test_classification.py
import pytest
from 02_classification.bert_classifier_trainer import BERTMetaClassifierTrainer

class TestBERTMetaClassifier:
    def test_model_initialization(self):
        trainer = BERTMetaClassifierTrainer()
        assert trainer.model is not None
        assert trainer.tokenizer is not None
    
    def test_training(self):
        trainer = BERTMetaClassifierTrainer()
        # Test training with sample data
        pass
    
    def test_prediction(self):
        trainer = BERTMetaClassifierTrainer()
        # Test prediction with sample data
        pass
```

### Integration Tests

End-to-end pipeline tests:

```python
# tests/test_integration.py
import pytest
from examples.basic_pipeline import main

class TestIntegration:
    def test_full_pipeline(self):
        # Test complete pipeline execution
        main()
        # Verify outputs exist
        assert os.path.exists('results/classification/')
        assert os.path.exists('results/clusters/')
        assert os.path.exists('results/evaluations/')
```

### Performance Tests

Benchmark tests for performance:

```python
# tests/test_performance.py
import pytest
import time
from 04_clustering.level_clustering.clustering_framework.thesis_clustering_framework import ThesisClusteringPipeline

class TestPerformance:
    def test_clustering_speed(self):
        pipeline = ThesisClusteringPipeline()
        start_time = time.time()
        pipeline.run()
        end_time = time.time()
        assert (end_time - start_time) < 300  # Should complete in 5 minutes
```

## Test Configuration

### Test Data

Test data is stored in `tests/data/`:
- `sample_papers.jsonl`: Sample arXiv papers
- `sample_labels.jsonl`: Sample methodology labels
- `sample_embeddings.npy`: Sample embedding data

### Test Parameters

```python
# tests/conftest.py
import pytest

@pytest.fixture
def sample_data():
    return {
        'papers': 'tests/data/sample_papers.jsonl',
        'labels': 'tests/data/sample_labels.jsonl',
        'embeddings': 'tests/data/sample_embeddings.npy'
    }

@pytest.fixture
def test_config():
    return {
        'batch_size': 2,
        'max_length': 128,
        'num_epochs': 1
    }
```

## Dependencies

- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities
- `pytest-benchmark`: Performance testing
- `pandas`: Data manipulation
- `numpy`: Numerical operations

## Test Coverage

### Coverage Targets

- **Overall Coverage**: >90%
- **Critical Components**: >95%
- **Utility Functions**: >85%

### Coverage Report

```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Debugging

### Verbose Output

```bash
python -m pytest tests/ -v
```

### Debug Mode

```bash
python -m pytest tests/ --pdb
```

### Specific Test

```bash
python -m pytest tests/test_classification.py::TestBERTMetaClassifier::test_training -v
```

## Related Documentation

- [Main README](../README.md)
- [Pipeline Documentation](../)
- [API Reference](../docs/api_reference.md)
