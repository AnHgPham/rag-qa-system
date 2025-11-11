#!/usr/bin/env python3
"""
Train Advanced Model (Dense Embeddings + RAG + LLM)

This script trains the advanced retrieval model using:
- Pre-trained sentence transformers (transfer learning)
- Dense embeddings (384-dim)
- Vector database (ChromaDB)
- LLM-based answer generation

Similar to the transfer learning approach in waste classification (MobileNetV2),
this leverages pre-trained models for better performance.

Training steps:
1. Load SQuAD documents
2. Encode with pre-trained sentence-transformers
3. Build vector database
4. Test retrieval
5. Integrate LLM for answer generation
6. Evaluate on dev set
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.embedding.dense_retriever import DenseEmbeddingRetriever, RAGAnswerGenerator

# Directories
DATA_DIR = PROJECT_ROOT / "data"
SQUAD_DIR = DATA_DIR / "raw" / "squad"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "advanced"

# Create directories
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
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


def train_dense_retriever(texts, metadata):
    """Train dense embedding retriever."""
    print("\n" + "=" * 60)
    print("Training Advanced Dense Embedding Model")
    print("=" * 60)
    print("\nUsing transfer learning approach:")
    print("  - Pre-trained model: all-MiniLM-L6-v2")
    print("  - Embedding dimension: 384")
    print("  - Vector database: ChromaDB")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize model
    retriever = DenseEmbeddingRetriever(
        model_name='all-MiniLM-L6-v2',
        collection_name='squad_train',
        persist_directory=VECTOR_DB_DIR
    )
    
    # Train (encode and index)
    retriever.fit(texts, metadata, batch_size=32)
    
    train_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    
    # Print stats
    stats = retriever.get_collection_stats()
    print(f"\nModel Statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    return retriever, train_time


def quick_test_retrieval(retriever):
    """Quick test of retrieval."""
    print(f"\n" + "=" * 60)
    print("Quick Test - Dense Embedding Retrieval")
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
        results = retriever.retrieve(query, top_k=3)
        query_time = (time.time() - start_time) * 1000  # ms
        
        print(f"Retrieval time: {query_time:.2f}ms")
        print(f"Top result (score={results[0]['score']:.4f}):")
        print(f"  Title: {results[0]['metadata']['title']}")
        print(f"  Text: {results[0]['text'][:200]}...")


def test_rag_generation(retriever):
    """Test RAG answer generation."""
    print(f"\n" + "=" * 60)
    print("Testing RAG Answer Generation")
    print("=" * 60)
    
    # Initialize RAG generator
    rag = RAGAnswerGenerator(model="gemini-2.0-flash")
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")

        # Retrieve
        start_time = time.time()
        retrieved = retriever.retrieve(query, top_k=3)
        retrieval_time = (time.time() - start_time) * 1000

        # Generate answer
        start_time = time.time()
        answer = rag.generate_answer(query, retrieved)
        generation_time = (time.time() - start_time) * 1000

        total_time = retrieval_time + generation_time

        print(f"Timing:")
        print(f"  - Retrieval: {retrieval_time:.2f}ms")
        print(f"  - Generation: {generation_time:.2f}ms")
        print(f"  - Total: {total_time:.2f}ms")

        print(f"\nAnswer:")
        print(f"  {answer['answer']}")
        print(f"\nSources: {answer['num_sources']}")
        print(f"Confidence: {answer['confidence']:.4f}")

        # Add delay to avoid rate limiting
        time.sleep(1.5)


def evaluate_on_dev(retriever, use_rag=False):
    """Evaluate model on dev set."""
    model_name = "RAG" if use_rag else "Dense Embedding"
    
    print(f"\n" + "=" * 60)
    print(f"Evaluating {model_name} on Dev Set")
    print("=" * 60)
    
    # Load dev Q&A pairs
    qa_file = SQUAD_DIR / "dev_qa_pairs.json"
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Sample for evaluation
    sample_size = min(500, len(qa_pairs))
    qa_sample = np.random.choice(qa_pairs, sample_size, replace=False)
    
    print(f"Evaluating on {sample_size} samples...")
    
    # Initialize RAG if needed
    rag = RAGAnswerGenerator(model="gemini-2.0-flash") if use_rag else None
    
    # Metrics
    total_queries = 0
    total_retrieval_time = 0
    total_generation_time = 0
    
    # For retrieval evaluation
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    
    # For answer evaluation (if using RAG)
    answers_generated = 0
    
    print("\nProcessing queries...")
    for i, qa in enumerate(qa_sample):
        if i % 100 == 0:
            print(f"  Progress: {i}/{sample_size}")
        
        question = qa['question']
        correct_context = qa['context']
        
        # Retrieve
        start_time = time.time()
        results = retriever.retrieve(question, top_k=5)
        retrieval_time = time.time() - start_time
        
        total_queries += 1
        total_retrieval_time += retrieval_time
        
        # Check if correct context is in results
        retrieved_texts = [r['text'] for r in results]
        
        # Use fuzzy matching (check if contexts overlap significantly)
        for rank, retrieved_text in enumerate(retrieved_texts):
            # Simple overlap check
            overlap = len(set(correct_context.split()) & set(retrieved_text.split()))
            overlap_ratio = overlap / len(set(correct_context.split()))
            
            if overlap_ratio > 0.8:  # 80% word overlap
                if rank == 0:
                    top1_hits += 1
                if rank < 3:
                    top3_hits += 1
                if rank < 5:
                    top5_hits += 1
                break
        
        # Generate answer if using RAG (only for first 50 to save API calls)
        if use_rag and i < 50:
            start_time = time.time()
            answer = rag.generate_answer(question, results)
            generation_time = time.time() - start_time

            total_generation_time += generation_time
            answers_generated += 1

            # Add delay to avoid rate limiting
            time.sleep(1.5)
    
    # Calculate metrics
    avg_retrieval_time = (total_retrieval_time / total_queries) * 1000  # ms
    precision_at_1 = top1_hits / total_queries
    precision_at_3 = top3_hits / total_queries
    precision_at_5 = top5_hits / total_queries
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Queries evaluated: {total_queries}")
    print(f"  Average retrieval time: {avg_retrieval_time:.2f}ms")
    print(f"  Precision@1: {precision_at_1:.4f}")
    print(f"  Precision@3: {precision_at_3:.4f}")
    print(f"  Precision@5: {precision_at_5:.4f}")
    
    if use_rag and answers_generated > 0:
        avg_generation_time = (total_generation_time / answers_generated) * 1000
        avg_total_time = avg_retrieval_time + avg_generation_time
        
        print(f"\nRAG Generation Metrics:")
        print(f"  Answers generated: {answers_generated}")
        print(f"  Average generation time: {avg_generation_time:.2f}ms")
        print(f"  Average total time: {avg_total_time:.2f}ms")
    
    # Save results
    results_data = {
        'model': model_name,
        'num_queries': total_queries,
        'avg_retrieval_time_ms': avg_retrieval_time,
        'precision_at_1': precision_at_1,
        'precision_at_3': precision_at_3,
        'precision_at_5': precision_at_5
    }
    
    if use_rag and answers_generated > 0:
        results_data['avg_generation_time_ms'] = avg_generation_time
        results_data['avg_total_time_ms'] = avg_total_time
        results_data['answers_generated'] = answers_generated
    
    results_file = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results_data


def main():
    """Main training function."""
    print("=" * 60)
    print("ADVANCED MODEL TRAINING (Transfer Learning)")
    print("=" * 60)
    print("\nThis trains the advanced retrieval model using:")
    print("  - Pre-trained sentence-transformers (all-MiniLM-L6-v2)")
    print("  - Dense embeddings (384 dimensions)")
    print("  - Vector database (ChromaDB)")
    print("  - LLM-based answer generation (RAG)")
    print("\nSimilar to MobileNetV2 transfer learning in waste")
    print("classification, this leverages pre-trained models for")
    print("better performance than the baseline.")
    print("=" * 60)

    # Load data from BOTH train and dev sets
    print("\nLoading documents from train and dev sets...")
    train_docs = load_squad_documents('train')
    dev_docs = load_squad_documents('dev')

    # Combine all documents
    all_docs = train_docs + dev_docs
    print(f"✓ Total documents: {len(all_docs)} (train: {len(train_docs)}, dev: {len(dev_docs)})")

    # Prepare training data
    texts, metadata = prepare_training_data(all_docs)
    
    # Train dense retriever
    retriever, train_time = train_dense_retriever(texts, metadata)
    
    # Quick test retrieval
    quick_test_retrieval(retriever)
    
    # Test RAG generation
    print("\n" + "=" * 60)
    print("Note: RAG generation requires API calls")
    print("Testing with a few examples...")
    print("=" * 60)
    test_rag_generation(retriever)
    
    # Evaluate retrieval only (no RAG to save API calls)
    print("\n" + "=" * 60)
    print("Full evaluation (retrieval only)")
    print("=" * 60)
    retrieval_results = evaluate_on_dev(retriever, use_rag=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("ADVANCED TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Precision@5: {retrieval_results['precision_at_5']:.4f}")
    print(f"Avg retrieval time: {retrieval_results['avg_retrieval_time_ms']:.2f}ms")
    
    print("\nNext steps:")
    print("  1. Compare with baseline results")
    print("  2. Analyze improvement")
    print("  3. Create comparison report")
    print("=" * 60)


if __name__ == "__main__":
    main()
