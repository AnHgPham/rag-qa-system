# RAG Document Q&A System

An end-to-end document question answering system using **Retrieval-Augmented Generation (RAG)** with baseline and advanced models.

## 🎯 Project Overview

This project implements a complete document Q&A pipeline with two approaches:

1. **Baseline Model**: Traditional TF-IDF retrieval (simple, fast)
2. **Advanced Model**: Dense embeddings + RAG + LLM (accurate, semantic)

### Inspired by Deep Learning Best Practices

| Aspect | Waste Classification | This Project |
|--------|---------------------|--------------|
| **Baseline** | Custom CNN (85% acc) | TF-IDF retrieval |
| **Advanced** | MobileNetV2 Transfer Learning (95% acc) | Dense Embeddings + RAG |
| **Improvement** | +10% accuracy | Significant precision improvement |
| **Technique** | Pre-trained CNN | Pre-trained transformers |

## 🌟 Features

### Technical Features
- ✅ **Baseline Model**: TF-IDF + BM25 retrieval
- ✅ **Advanced Model**: sentence-transformers + ChromaDB + LLM
- ✅ **Comprehensive Evaluation**: Precision@k, Recall, Query time
- ✅ **Model Comparison**: Side-by-side performance analysis
- ✅ **Modular Architecture**: Clean separation of components

### Dataset
- **SQuAD v1.1** (Stanford Question Answering Dataset)
- 100,000+ question-answer pairs
- 500+ Wikipedia articles
- Perfect for RAG evaluation

## 📊 Results Preview

| Metric | Baseline (TF-IDF) | Advanced (Embeddings) | Improvement |
|--------|-------------------|----------------------|-------------|
| Precision@5 | ~0.60 | ~0.80 | +33% |
| Query Time | 20-30ms | 100-200ms | Acceptable |
| Semantic Understanding | Limited | Excellent | ✓ |

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Git
git --version
```

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd rag_document_qa

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```
### Quick Run

```bash
# Run full pipeline (download + train + compare)
python main.py --all
```

This will:
1. Download SQuAD dataset (~30MB)
2. Train baseline model (~10 seconds)
3. Train advanced model (~10 minutes)
4. Generate comparison report

## 📖 Usage

### Command Line Interface

```bash
# Download dataset only
python main.py --download

# Train baseline model
python main.py --train-baseline

# Train advanced model
python main.py --train-advanced

# Compare models
python main.py --compare

# Query the system
python main.py --query "What is the capital of France?"
python main.py --query "Who invented the telephone?" --model baseline
```

### Individual Scripts

```bash
# Download SQuAD dataset
python scripts/01_download_squad_dataset.py

# Train baseline (TF-IDF)
python scripts/baseline/02_train_baseline.py

# Train advanced (Embeddings + RAG)
python scripts/advanced/03_train_advanced.py

# Compare models
python scripts/04_compare_models.py
```

### Python API

```python
# Baseline model
from src.baseline.tfidf_retriever import TFIDFRetriever

retriever = TFIDFRetriever()
retriever.load("outputs/models/baseline/tfidf_retriever.pkl")

results = retriever.retrieve("What is photosynthesis?", top_k=5)
answer = retriever.extract_answer("What is photosynthesis?", results)
print(answer['answer'])

# Advanced model
from src.embedding.dense_retriever import DenseEmbeddingRetriever, RAGAnswerGenerator

retriever = DenseEmbeddingRetriever(
    collection_name='squad_train',
    persist_directory='data/vector_db'
)

results = retriever.retrieve("What is photosynthesis?", top_k=5)

rag = RAGAnswerGenerator(model="gpt-4.1-nano")
answer = rag.generate_answer("What is photosynthesis?", results)
print(answer['answer'])
```

## 📁 Project Structure

```
rag_document_qa/
│
├── src/                          # Source code
│   ├── baseline/                 # Baseline model (TF-IDF)
│   │   ├── __init__.py
│   │   └── tfidf_retriever.py   # TF-IDF retrieval implementation
│   │
│   ├── embedding/                # Advanced model (Embeddings)
│   │   ├── __init__.py
│   │   └── dense_retriever.py   # Dense embedding + RAG
│   │
│   ├── evaluation/               # Evaluation metrics
│   └── utils/                    # Utilities
│
├── scripts/                      # Training scripts
│   ├── 01_download_squad_dataset.py
│   ├── baseline/
│   │   └── 02_train_baseline.py
│   ├── advanced/
│   │   └── 03_train_advanced.py
│   └── 04_compare_models.py
│
├── data/                         # Data directory
│   ├── raw/squad/               # SQuAD dataset
│   ├── vector_db/               # ChromaDB storage
│   └── uploads/                 # User uploads
│
├── outputs/                      # Output directory
│   ├── models/                  # Saved models
│   ├── results/                 # Evaluation results
│   │   ├── baseline/
│   │   ├── advanced/
│   │   └── comparison/
│   └── logs/                    # Log files
│
├── docs/                         # Documentation
├── tests/                        # Unit tests
│
├── main.py                       # Main CLI
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🏗️ Architecture

### Baseline Model (TF-IDF)

```
Document → TF-IDF Vectorizer → Sparse Matrix → Cosine Similarity → Top-K Results
```

**Characteristics:**
- Fast (20-30ms per query)
- Simple, no training required
- Good for keyword matching
- Limited semantic understanding

### Advanced Model (Dense Embeddings + RAG)

```
Document → Sentence-Transformers → Dense Embeddings (384-dim) → ChromaDB
                                                                      ↓
Query → Embedding → Semantic Search → Top-K Contexts → LLM → Answer
```

**Characteristics:**
- Semantic understanding
- Higher accuracy
- Slower but acceptable (100-200ms)
- Uses pre-trained models (transfer learning)

## 📈 Evaluation Metrics

### Retrieval Metrics
- **Precision@k**: Proportion of relevant docs in top-k
- **Recall@k**: Coverage of relevant docs
- **MRR**: Mean Reciprocal Rank
- **Query Time**: Average retrieval time

### Answer Quality (RAG)
- **Exact Match**: Exact string match with ground truth
- **F1 Score**: Token overlap with ground truth
- **Citation Accuracy**: Correctness of source citations

## 🔧 Configuration

### Environment Variables

```bash
# Required for RAG (LLM generation)
export OPENAI_API_KEY="your-api-key"
```

The system supports these models via OpenAI-compatible API:
- `gpt-4.1-nano` (fast, cheap)
- `gpt-4.1-mini` (balanced)
- `gemini-2.5-flash` (Google's model)

### Model Parameters

**Baseline (TF-IDF):**
```python
TFIDFRetriever(
    max_features=10000,      # Vocabulary size
    ngram_range=(1, 2),      # Unigrams + bigrams
    min_df=2,                # Minimum document frequency
    max_df=0.8               # Maximum document frequency
)
```

**Advanced (Embeddings):**
```python
DenseEmbeddingRetriever(
    model_name='all-MiniLM-L6-v2',  # Pre-trained model
    collection_name='squad_train',   # ChromaDB collection
    persist_directory='data/vector_db'
)
```

## 📚 Learning Resources

### Understanding the Approach

1. **Baseline (TF-IDF)**
   - [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
   - [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

2. **Advanced (RAG)**
   - [Sentence Transformers](https://www.sbert.net/)
   - [RAG Paper](https://arxiv.org/abs/2005.11401)
   - [ChromaDB Documentation](https://docs.trychroma.com/)

3. **Transfer Learning**
   - Similar to MobileNetV2 in waste classification
   - Pre-trained on large text corpora
   - Fine-tuned for specific tasks

## 🎓 Educational Value

This project demonstrates:

1. **Progressive Learning**: Start simple (baseline) → advance (RAG)
2. **Transfer Learning**: Leverage pre-trained models
3. **Evaluation**: Comprehensive metrics and comparison
4. **Best Practices**: Clean code, modular architecture
5. **Real-world Application**: Production-ready pipeline

## 🔬 Experiments & Extensions

### Possible Improvements

1. **Hybrid Approach**
   ```python
   # Fast baseline filter → Advanced reranking
   baseline_results = tfidf.retrieve(query, top_k=100)
   advanced_results = embeddings.rerank(baseline_results, top_k=5)
   ```

2. **Fine-tuning**
   - Fine-tune embedding model on domain-specific data
   - Train custom reranker

3. **Optimization**
   - Caching for common queries
   - Batch processing
   - GPU acceleration

4. **Multi-modal**
   - Add image understanding
   - Table extraction
   - Chart analysis

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more baseline models (BM25+, etc.)
- [ ] Implement hybrid retrieval
- [ ] Add web interface (Streamlit/Gradio)
- [ ] Support more datasets
- [ ] Add Vietnamese language support
- [ ] Implement query expansion
- [ ] Add conversational memory

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **SQuAD Dataset**: Stanford NLP Group
- **Sentence Transformers**: UKP Lab
- **ChromaDB**: Chroma team
- **Inspiration**: [Waste Classification Project](https://github.com/AnHgPham/waste_DEEPLEARNING)

## 📧 Contact

For questions or feedback, please open an issue.

---

**Note**: This project is designed for learning and demonstration. For production use, consider:
- Adding authentication and rate limiting
- Implementing proper error handling
- Setting up monitoring and logging
- Optimizing for scale
- Adding security measures
