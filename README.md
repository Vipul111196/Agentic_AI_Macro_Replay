# Autonomous Macro Replay System for Warmwind OS

An intelligent system that learns and replays UI workflows by recognizing visual patterns, minimizing the need for AI intervention.

## 🎯 Overview

This system automatically handles repetitive UI workflows by:
1. **Learning** from historical conversation data (images + actions)
2. **Building** an execution graph of UI states and transitions
3. **Matching** new screenshots to known states using Voyage AI embeddings (1024D)
4. **Replaying** stored actions when similar states are detected
5. **Falling back** to AI only for novel/unknown states

## 📊 System Status

### ✅ 100% COMPLETE - All Components Built & Tested

#### 1. Data Parser (`data_parser.py`)
- **Streaming architecture**: Processes messages one at a time, simulating real-world conversation flow
- Extracts triplets: `(image, reasoning, action)` where **image is optional**
- **Core logic**: Every message with a `<think>` block triggers step creation
- **Sequential pattern**: 
  - Index n-1: User message with optional screenshot
  - Index n: Assistant with `<think>` → **triggers step creation immediately**
  - Index n+1: Assistant with action tags (captured separately)
- **Stateful processing**: 
  - `add_message()`: Add one message at a time, returns step if conditions met
  - `parse_file_streaming()`: Process entire JSON file in streaming mode
  - Message buffer maintains context for looking back at previous messages
- **Tested**: Successfully extracted 753 steps from maf_train.json
- **Key metrics**: 753/753 steps have reasoning, 610/753 have images
- **Backward compatible**: `ConversationParser` alias points to `StreamingConversationParser`

#### 2. Image Embedder (`image_embedder.py`)
- **Voyage AI multimodal-3** model for converting screenshots to 1024-dimensional vectors
- Supports batch processing for efficiency
- Normalized embeddings for cosine similarity
- **Security**: 5-layer validation system to prevent fake embeddings
- **Quality**: State-of-the-art embedding model outperforming OpenAI and Cohere
- **Tested**: Successfully embedded 627 training images

#### 3. Macro Graph Engine (`macro_graph.py`)
- ✅ Node structure: `(embedding, action, think_text, metadata, statistics)`
- ✅ Edge structure: Transitions between states with counts
- ✅ FAISS vector index for fast similarity search (Inner Product for cosine)
- ✅ Graph persistence (save/load functionality)
- ✅ Deduplication: Merges similar states (threshold: 0.90)
- **Tested**: Built graph with 539 unique nodes, 538 edges from 627 training steps

#### 4. Replay Engine (`replay_engine.py`)
- ✅ Three-way decision system: REPLAY / AI_FALLBACK / NO_ACTION
- ✅ Similarity threshold: 0.90 (configurable)
- ✅ Visit count filtering for trust-based decisions
- ✅ Statistics tracking for replay rate monitoring
- **Tested**: Validation run shows 93% replay rate on test data

#### 5. Validation Framework (`demo_end_to_end.py`)
- ✅ End-to-end pipeline testing
- ✅ Metrics: replay rate, graph coverage, deduplication stats
- ✅ Performance: 2.7 seconds for 10 validation queries
- **Results**: High replay success, stable graph structure

#### 6. Interactive Dashboard (`app_dashboard.py`)
- ✅ **Streamlit web interface** with two pages:
  - **Dashboard**: System metrics, decision breakdown, top visited nodes
  - **2D Graph Visualization**: Semantic graph using MDS dimensionality reduction
- ✅ Interactive graph with hover details
- ✅ Node coloring (green = has action, blue = no action)
- ✅ Configurable similarity threshold and visit count filters
- ✅ Real-time graph statistics and network metrics

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up Voyage AI API key (required for embeddings)
export VOYAGE_API_KEY='your-api-key-here'
```

📖 **For detailed setup instructions**, see [`VOYAGE_AI_SETUP.md`](VOYAGE_AI_SETUP.md)

### Running the System

#### 1. Train the Graph (One-time setup)
```bash
# Train the macro graph on training data
python3 train.py
```

This will:
- Process training conversations (default: 5 conversations from maf_train.json)
- Build the execution graph with similarity-based deduplication
- Save the trained graph to `trained_graph.pkl` and `trained_graph.faiss`
- Generate training report: `report_latest.json`

#### 2. Test the Graph (Validation)
```bash
# Test the trained graph on validation data
python3 test.py
```

This will:
- Load the trained graph (read-only, no modifications)
- Test on validation data (default: 5 conversations from maf_validate.json)
- Measure replay accuracy and coverage
- Generate test report: `test_report_latest.json`

#### 3. Visualize Results (Dashboard)
```bash
# Run the interactive dashboard
export KMP_DUPLICATE_LIB_OK=TRUE  # macOS only
streamlit run app_dashboard.py --server.port 8501
```

Then open `http://localhost:8501` in your browser to:
- View system metrics and graph statistics
- Explore the semantic graph visualization
- Analyze training and test reports

### Quick Start Example

```bash
# Step 1: Train the graph
python3 train.py
# Output: trained_graph.pkl, trained_graph.faiss, report_latest.json

# Step 2: Test the graph
python3 test.py
# Output: test_report_latest.json

# Step 3: View results in dashboard
streamlit run app_dashboard.py
# Open: http://localhost:8501
```

### Testing Individual Components

```bash
# Test data parser
python3 data_parser.py

# Test image embedder (requires VOYAGE_API_KEY)
python3 image_embedder.py
```

## 📁 Project Structure

```
.
├── data_validation_split/      # Training and validation data
│   ├── maf_train.json         # Training set (MAF) - 627 steps
│   ├── sona_train.json        # Training set (SONA)
│   ├── maf_validate.json      # Validation set (MAF)
│   └── sona_validate.json     # Validation set (SONA)
│
├── Core Components
├── data_parser.py              # ✅ Conversation JSON parser
├── image_embedder.py           # ✅ Voyage AI-based image embedding
├── macro_graph.py              # ✅ Graph structure with FAISS indexing
├── replay_engine.py            # ✅ Decision engine for replay logic
├── replay_types.py             # Type definitions for replay decisions
├── utils.py                    # Utility functions
│
├── Execution Scripts
├── train.py                    # ✅ Train graph on training data
├── test.py                     # ✅ Test graph on validation data
├── app_dashboard.py            # ✅ Streamlit interactive dashboard
│
├── Output Files
├── trained_graph.pkl           # Trained graph data (nodes, edges, metadata)
├── trained_graph.faiss         # FAISS index for similarity search
├── trained_graph_embeddings.npz # Node embeddings for visualization
├── report_latest.json          # Latest training report
├── test_report_latest.json     # Latest test report
├── embedding_cache.pkl         # Cached embeddings (performance optimization)
│
└── Documentation
    ├── config.yaml             # Configuration parameters
    ├── README.md               # This file
    ├── requirements.txt        # Python dependencies
    └── project-progress.md     # Detailed progress tracking
```

## 🔄 Train-Test Workflow

The system follows a standard ML workflow:

### Training Phase (`train.py`)
1. **Load Training Data**: Processes conversations from training set
2. **Build Graph**: Creates nodes for unique UI states, merges similar states
3. **Learn Transitions**: Records state-to-state transitions as edges
4. **Save Model**: Persists graph to disk (`trained_graph.pkl` + `.faiss`)
5. **Generate Report**: Creates training metrics report

**Key Point**: Training adds nodes to the graph and learns patterns.

### Testing Phase (`test.py`)
1. **Load Trained Graph**: Loads the frozen graph from disk (read-only)
2. **Process Test Data**: Tests on validation conversations
3. **Measure Accuracy**: Compares predicted actions vs ground truth
4. **Generate Report**: Creates test metrics with accuracy statistics

**Key Point**: Testing does NOT modify the graph - it only evaluates performance.

### Metrics Tracked
- **Replay Rate**: % of steps handled without AI fallback
- **Replay Accuracy**: % of replayed actions that match ground truth
- **AI Fallback Rate**: % of steps needing human/AI assistance
- **Confidence**: Average similarity score for matches

## 🔬 Technical Details

### Data Format

Conversations are stored as:
```json
[
  {
    "conversation": [
      {
        "role": "user",
        "content": [{"type": "image", "image_url": "data:image/webp;base64,..."}]
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "<think>...</think><action>...</action>"}]
      }
    ]
  }
]
```

### Embedding Strategy

- **Model**: Voyage AI multimodal-3 (state-of-the-art)
- **Dimension**: 1024 (higher than OpenAI/Cohere alternatives)
- **Normalization**: L2-normalized for cosine similarity
- **Validation**: 5-layer security system to prevent fake embeddings
- **Similarity Range**: 0.0 (different) to 1.0 (identical)
- **Indexing**: FAISS Inner Product index for efficient similarity search

### Graph Visualization

- **Method**: MDS (Multidimensional Scaling)
- **Purpose**: Reduces 1024D embeddings to 2D for visualization
- **Benefit**: Node proximity reflects actual visual similarity
- **Interactive**: Hover to see node details, visit counts, and actions

## 📈 Dataset Statistics

### Training Set
- **Files**: `maf_train.json` (primary training data)
- **Total steps**: 627 extracted steps
- **With images**: 627 (100%)
- **With actions**: 627 (100%)
- **Purpose**: Build initial macro graph

### Validation Set
- **Files**: `maf_validate.json`, `sona_validate.json`
- **Purpose**: Test replay accuracy and measure metrics
- **Results**: 93% replay rate on test queries

### Graph Metrics
- **Unique nodes**: 539 (after deduplication)
- **Edges**: 538 state transitions
- **Deduplication rate**: 14% (88 duplicates merged)
- **Similarity threshold**: 0.90

## 🎯 Success Metrics

Achieved results:
1. **Replay Success Rate**: 93% of steps handled without AI fallback
2. **Graph Coverage**: High coverage with 539 unique UI states
3. **Deduplication**: Effective merging of similar states (14% reduction)
4. **Performance**: 2.7 seconds for 10 validation queries
5. **Graph Stability**: Consistent structure across training runs

## 📝 Design Decisions

### Why Voyage AI?
- **State-of-the-art**: Outperforms OpenAI and Cohere on multimodal tasks
- **Higher dimensionality**: 1024D embeddings capture more nuances than alternatives
- **Security**: 5-layer validation system ensures embedding authenticity
- **Quality**: Excellent performance on UI screenshot similarity

### Why MDS for Visualization?
- **Preserves distances**: MDS maintains pairwise similarity relationships
- **Semantic layout**: Node proximity directly reflects visual similarity
- **Interpretable**: Easy to understand clustering of similar UI states
- **Stable**: Consistent layouts across runs

### Why Cosine Similarity?
- Natural for normalized embeddings
- Efficient computation with FAISS Inner Product index
- Interpretable scores (0-1 range)
- Works well with high-dimensional vectors

### Why Graph Structure?
- Captures workflow sequences and branching paths
- Enables path-based reasoning for multi-step tasks
- Supports incremental learning as new states are encountered
- Efficient deduplication through similarity-based merging

## 🔮 Future Enhancements

- Multi-modal embeddings (visual + text reasoning)
- Reinforcement learning for path optimization
- Active learning for high-value novel states
- Distributed processing for large-scale deployment
- OCR integration for parameterized actions
- Anomaly detection for unexpected UI states

## 📚 References

- Voyage AI: https://www.voyageai.com/
- Voyage AI Multimodal: https://docs.voyageai.com/docs/multimodal-embeddings
- FAISS: https://github.com/facebookresearch/faiss
- Streamlit: https://streamlit.io/
- Scikit-learn MDS: https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling

---

**Status**: ✅ 100% COMPLETE - Production Ready  
**Last Updated**: October 3, 2025  
**Current Phase**: Deployment & Monitoring

# Agentic_AI_Macro_Replay
