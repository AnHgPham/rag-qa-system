#!/usr/bin/env python3
"""
Train Baseline Model (TF-IDF + BM25)

This script trains the baseline retrieval model using traditional IR techniques.
Similar to training the baseline CNN in the waste classification project.

Training steps:
1. Load SQuAD documents
2. Train TF-IDF vectorizer
3. Build document index
4. Save trained model
5. Evaluate on dev set
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.tfidf_retriever import TFIDFRetriever, BM25Retriever

# Directories
DATA_DIR = PROJECT_ROOT / "data"
SQUAD_DIR = DATA_DIR / "raw" / "squad"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "baseline"

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_squad_documents(split='train'):
    """Load SQuAD documents."""
    doc_file = SQUAD_DIR / f"{split}_documents.json"
    
    print(f"Loading {split} documents from {doc_file}...")
    with open(doc_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"✓ Loaded {len(documents)} documents")
    
    return documents


def prepare_training_data(documents):
    """Prepare training data from documents."""
    print("Preparing training data...")
    
    texts = []
    metadata = []
    
    for doc in documents:
        texts.append(doc['context'])
        metadata.append({
            'doc_id': doc['doc_id'],
            'title': doc['title'],
            'num_questions': doc['num_questions']
        })
    
    print(f"✓ Prepared {len(texts)} document texts")
    
    return texts, metadata


def train_tfidf_model(texts, metadata):
    """Train TF-IDF baseline model."""
    print("\n" + "=" * 60)
    print("Training TF-IDF Baseline Model")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize model
    model = TFIDFRetriever(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Train (fit)
    model.fit(texts, metadata)
    
    train_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    
    # Save model
    model_path = MODEL_DIR / "tfidf_retriever.pkl"
    model.save(model_path)
    
    return model, train_time


def train_bm25_model(texts, metadata):
    """Train BM25 baseline model."""
    print("\n" + "=" * 60)
    print("Training BM25 Baseline Model")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize model
    model = BM25Retriever(k1=1.5, b=0.75)
    
    # Train (fit)
    model.fit(texts, metadata)
    
    train_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    
    return model, train_time


def quick_test(model, model_name="TF-IDF"):
    """Quick test of the trained model."""
    print(f"\n" + "=" * 60)
    print(f"Quick Test - {model_name} Model")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?",
        "What is photosynthesis?",
        "How does gravity work?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        start_time = time.time()
        results = model.retrieve(query, top_k=3)
        query_time = (time.time() - start_time) * 1000  # ms
        
        print(f"Retrieval time: {query_time:.2f}ms")
        print(f"Top result (score={results[0]['score']:.4f}):")
        print(f"  {results[0]['text'][:200]}...")
        
        # === PHẦN ĐÃ SỬA ===
        # Chỉ gọi extract_answer nếu phương thức đó tồn tại
        if hasattr(model, 'extract_answer'):
            answer = model.extract_answer(query, results, strategy='top_chunk')
            print(f"Answer confidence: {answer['confidence']:.4f}")
        # === KẾT THÚC PHẦN SỬA ===


def evaluate_on_dev(model, model_name="TF-IDF"):
    """Evaluate model on dev set."""
    print(f"\n" + "=" * 60)
    print(f"Evaluating {model_name} on Dev Set")
    print("=" * 60)
    
    # Load dev Q&A pairs
    qa_file = SQUAD_DIR / "dev_qa_pairs.json"
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Sample for quick evaluation (use subset)
    sample_size = min(1000, len(qa_pairs))
    qa_sample = np.random.choice(qa_pairs, sample_size, replace=False)
    
    print(f"Evaluating on {sample_size} samples...")
    
    # Metrics
    total_queries = 0
    total_time = 0
    retrieval_hits = 0  # How many times the correct context was retrieved
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    
    for qa in qa_sample:
        question = qa['question']
        correct_context = qa['context']
        
        # Retrieve
        start_time = time.time()
        results = model.retrieve(question, top_k=5)
        query_time = time.time() - start_time
        
        total_queries += 1
        total_time += query_time
        
        # Check if correct context is in results
        retrieved_texts = [r['text'] for r in results]
        
        if correct_context in retrieved_texts:
            retrieval_hits += 1
            
            # Check rank
            rank = retrieved_texts.index(correct_context) + 1
            if rank == 1:
                top1_hits += 1
            if rank <= 3:
                top3_hits += 1
            if rank <= 5:
                top5_hits += 1
    
    # Calculate metrics
    avg_time = (total_time / total_queries) * 1000  # ms
    recall_at_5 = retrieval_hits / total_queries
    precision_at_1 = top1_hits / total_queries
    precision_at_3 = top3_hits / total_queries
    precision_at_5 = top5_hits / total_queries
    
    # Print results
    print(f"\n{model_name} Baseline Results:")
    print(f"  Queries evaluated: {total_queries}")
    print(f"  Average query time: {avg_time:.2f}ms")
    print(f"  Recall@5: {recall_at_5:.4f}")
    print(f"  Precision@1: {precision_at_1:.4f}")
    print(f"  Precision@3: {precision_at_3:.4f}")
    print(f"  Precision@5: {precision_at_5:.4f}")
    
    # Save results
    results_data = {
        'model': model_name,
        'num_queries': total_queries,
        'avg_query_time_ms': avg_time,
        'recall_at_5': recall_at_5,
        'precision_at_1': precision_at_1,
        'precision_at_3': precision_at_3,
        'precision_at_5': precision_at_5
    }
    
    results_file = RESULTS_DIR / f"{model_name.lower()}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results_data


def main():
    """Main training function."""
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    print("\nThis trains the baseline retrieval model using:")
    print("  - TF-IDF vectorization")
    print("  - Cosine similarity ranking")
    print("  - Simple heuristic answer extraction")
    print("\nSimilar to the baseline CNN in waste classification,")
    print("this establishes a performance baseline before applying")
    print("advanced techniques (RAG + LLM).")
    print("=" * 60)
    
    # Load data
    train_docs = load_squad_documents('train')
    
    # Prepare training data
    texts, metadata = prepare_training_data(train_docs)
    
    # Train TF-IDF model
    tfidf_model, tfidf_train_time = train_tfidf_model(texts, metadata)
    
    # Quick test
    quick_test(tfidf_model, "TF-IDF")
    
    # Evaluate on dev set
    tfidf_results = evaluate_on_dev(tfidf_model, "TF-IDF")
    
    # Train BM25 model (optional - improved version)
    print("\n" + "=" * 60)
    print("Training BM25 model (improved baseline)...")
    print("=" * 60)
    bm25_model, bm25_train_time = train_bm25_model(texts, metadata)
    
    # Quick test BM25
    quick_test(bm25_model, "BM25")
    
    # Evaluate BM25
    bm25_results = evaluate_on_dev(bm25_model, "BM25")
    
    # Summary
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)
    print("\nTF-IDF Model:")
    print(f"  Training time: {tfidf_train_time:.2f}s")
    print(f"  Precision@5: {tfidf_results['precision_at_5']:.4f}")
    print(f"  Avg query time: {tfidf_results['avg_query_time_ms']:.2f}ms")
    
    print("\nBM25 Model:")
    print(f"  Training time: {bm25_train_time:.2f}s")
    print(f"  Precision@5: {bm25_results['precision_at_5']:.4f}")
    print(f"  Avg query time: {bm25_results['avg_query_time_ms']:.2f}ms")
    
    print("\nNext steps:")
    print("  1. Train advanced model (RAG + LLM)")
    print("  2. Compare baseline vs advanced performance")
    print("  3. Analyze improvement and trade-offs")
    print("=" * 60)


if __name__ == "__main__":
    main()