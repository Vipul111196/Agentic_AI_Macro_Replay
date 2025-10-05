# Configuration File Guide

## Overview

All input and output file paths are now centrally managed in `config.yaml`. This makes it easy to customize paths without editing code.

## Complete Configuration Reference

### 1. Embedding Configuration

```yaml
embedding:
  provider: "voyage"              # Options: "voyage" or "selfmade"
  model_name: "voyage-multimodal-3"
  embedding_dim: auto             # Auto-detected (1024 or 256)
  validate_embeddings: true
  timeout: 30
```

**What it controls:**
- Which embedding provider to use (Voyage AI or SelfMade)
- Model settings for Voyage AI
- API timeout settings

### 2. Similarity Matching

```yaml
matching:
  similarity_threshold: 0.90      # Cosine similarity threshold
  top_k_candidates: 5             # Number of similar nodes to consider
  min_visit_count: 1              # Minimum visits before trusting
```

**What it controls:**
- How similar screenshots need to be to match
- Number of candidates to search
- Trust threshold for replay decisions

### 3. Graph Configuration

```yaml
graph:
  max_nodes: 100000               # Maximum nodes before pruning
  enable_pruning: false           # Enable/disable pruning
  min_success_rate: 0.7           # Minimum success rate to keep
```

**What it controls:**
- Graph size limits
- Pruning behavior
- Node quality thresholds

### 4. FAISS Vector Index

```yaml
vector_index:
  index_type: "IndexFlatIP"       # Inner product for cosine similarity
  use_gpu: false                  # GPU acceleration
  rebuild_interval: 1000          # Rebuild frequency
```

**What it controls:**
- FAISS index type
- GPU usage
- Index optimization

### 5. Data Processing & Input Files ⭐

```yaml
data:
  # Input data files
  training_files:
    - "data_validation_split/maf_train.json"
    - "data_validation_split/sona_train.json"
  validation_files:
    - "data_validation_split/maf_validate.json"
    - "data_validation_split/sona_validate.json"
  
  # Processing limits
  max_conversations_train: null   # null = process all
  max_conversations_test: null    # null = process all
  batch_size: 10
  max_image_size: [1024, 1024]
```

**What it controls:**
- **Input training files** - Which JSON files to use for training
- **Input validation files** - Which JSON files to use for testing
- **Processing limits** - How many conversations to process (null = all)
- **Batch size** - Processing batch size
- **Image size limits** - Maximum image dimensions

### 6. Output Paths & File Naming ⭐

```yaml
output:
  # Training outputs
  trained_graph_path: "models/trained_graph"
  trained_embeddings_path: "embeddings/trained_graph_embeddings.npz"
  training_report_dir: "reports"
  training_report_latest: "reports/report_latest.json"
  
  # Testing outputs
  test_results_dir: "test_results"
  test_report_latest: "test_results/test_report_latest.json"
  test_embeddings_path: "embeddings/test_embeddings.npz"
  
  # Cache directories
  embedding_cache_dir: "cache"
  voyage_cache: "cache/embedding_cache.pkl"
  selfmade_cache: "cache/selfmade_embedding_cache.pkl"
```

**What it controls:**
- **Graph output location** - Where to save trained graph (.pkl and .faiss)
- **Embeddings output** - Where to save embedding visualizations
- **Report directories** - Where to save training/test reports
- **Cache locations** - Where to store embedding caches for each provider

### 7. Action Extraction

```yaml
actions:
  coordinate_variance: 10         # Pixel tolerance for matching
```

**What it controls:**
- Coordinate matching tolerance

### 8. Logging & Monitoring

```yaml
logging:
  level: "INFO"
  log_file: "macro_replay.log"
  save_embeddings: true
  save_graph_snapshots: true
  snapshot_interval: 500
```

**What it controls:**
- Log level and output file
- Whether to save embeddings
- Graph snapshot frequency

### 9. Validation Settings

```yaml
validation:
  enable_fallback_logging: true
  save_failure_cases: true
  metrics_output: "validation_metrics.json"
```

**What it controls:**
- Logging behavior during validation
- Failure case tracking
- Metrics output location

## Common Customizations

### Example 1: Change Training Data Location

```yaml
data:
  training_files:
    - "my_custom_data/training_set_1.json"
    - "my_custom_data/training_set_2.json"
```

### Example 2: Change Output Directory

```yaml
output:
  trained_graph_path: "my_models/graph_v2"
  training_report_dir: "my_reports"
```

### Example 3: Limit Conversations for Quick Testing

```yaml
data:
  max_conversations_train: 5      # Only process 5 conversations
  max_conversations_test: 3       # Only test 3 conversations
```

### Example 4: Use All Data

```yaml
data:
  max_conversations_train: null   # Process all training data
  max_conversations_test: null    # Process all test data
```

### Example 5: Switch to SelfMade Embeddings

```yaml
embedding:
  provider: "selfmade"            # Switch to pixel-based embeddings
```

The system will automatically use `selfmade_cache` instead of `voyage_cache`.

## File Path Variables Used in Code

The code now reads all paths from config:

### train.py reads:
- `data.training_files` → Input training data
- `data.max_conversations_train` → Training limit
- `matching.similarity_threshold` → Graph/replay threshold
- `output.trained_graph_path` → Where to save graph
- `output.trained_embeddings_path` → Where to save embeddings
- `output.training_report_dir` → Report directory
- `output.training_report_latest` → Latest report file
- `output.voyage_cache` or `output.selfmade_cache` → Cache file

### test.py reads:
- `data.validation_files` → Input test data
- `data.max_conversations_test` → Testing limit
- `matching.similarity_threshold` → Replay threshold
- `output.trained_graph_path` → Where to load graph from
- `output.test_results_dir` → Test results directory
- `output.test_report_latest` → Latest test report
- `output.test_embeddings_path` → Test embeddings output
- `output.voyage_cache` or `output.selfmade_cache` → Cache file

## Benefits

✅ **No code editing needed** - Change paths in one place
✅ **Easy experimentation** - Test different data sources
✅ **Multiple configurations** - Maintain different config files
✅ **Clear organization** - All settings in one place
✅ **Type safety** - Default values prevent errors

## Tips

1. **Relative paths** are relative to the project root
2. **Directories are created automatically** if they don't exist
3. **null values** mean "all" or "unlimited"
4. **Keep backups** of config.yaml when experimenting
5. **Provider-specific caches** are automatically selected

## Validation

After editing `config.yaml`, validate it:

```bash
python -c "import yaml; yaml.safe_load(open('config.yaml')); print('✅ Valid YAML')"
```

## Quick Reference Table

| Configuration | Controls | Default |
|---------------|----------|---------|
| `embedding.provider` | Voyage AI or SelfMade | `"voyage"` |
| `data.training_files` | Input training data | `maf_train.json`, `sona_train.json` |
| `data.validation_files` | Input test data | `maf_validate.json`, `sona_validate.json` |
| `data.max_conversations_train` | Training limit | `null` (all) |
| `data.max_conversations_test` | Testing limit | `null` (all) |
| `output.trained_graph_path` | Graph save location | `models/trained_graph` |
| `output.trained_embeddings_path` | Training embeddings | `embeddings/trained_graph_embeddings.npz` |
| `output.training_report_dir` | Report directory | `reports` |
| `output.test_results_dir` | Test results dir | `test_results` |
| `output.voyage_cache` | Voyage cache | `cache/embedding_cache.pkl` |
| `output.selfmade_cache` | SelfMade cache | `cache/selfmade_embedding_cache.pkl` |
| `matching.similarity_threshold` | Match threshold | `0.90` |

## Summary

All file paths are now in `config.yaml`:

✅ **Input files** - Training and validation data
✅ **Output files** - Graphs, embeddings, reports
✅ **Cache files** - Embedding caches for both providers
✅ **Directories** - Where to save results
✅ **Processing limits** - How many conversations to process

Simply edit `config.yaml` and run `train.py` or `test.py` - no code changes needed!


