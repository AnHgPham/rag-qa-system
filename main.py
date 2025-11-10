#!/usr/bin/env python3
"""
RAG Document Q&A System - Main CLI

This is the main entry point for the RAG Document Q&A system.
Similar to main.py in the waste classification project.

Usage:
    python main.py --help                    # Show help
    python main.py --download                # Download SQuAD dataset
    python main.py --train-baseline          # Train baseline model
    python main.py --train-advanced          # Train advanced model
    python main.py --compare                 # Compare models
    python main.py --all                     # Run full pipeline
    python main.py --query "Your question"   # Query the system
"""

import argparse
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_script(script_path, description):
    """Run a Python script."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}\n")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        print(f"\n⚠ Error running {script_path}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="RAG Document Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset
  python main.py --download
  
  # Train baseline model
  python main.py --train-baseline
  
  # Train advanced model
  python main.py --train-advanced
  
  # Compare models
  python main.py --compare
  
  # Run full pipeline
  python main.py --all
  
  # Query the system
  python main.py --query "What is the capital of France?"
        """
    )
    
    # Actions
    parser.add_argument('--download', action='store_true',
                       help='Download SQuAD dataset')
    parser.add_argument('--train-baseline', action='store_true',
                       help='Train baseline TF-IDF model')
    parser.add_argument('--train-advanced', action='store_true',
                       help='Train advanced embedding model')
    parser.add_argument('--compare', action='store_true',
                       help='Compare baseline vs advanced models')
    parser.add_argument('--all', action='store_true',
                       help='Run full pipeline (download + train + compare)')
    parser.add_argument('--query', type=str,
                       help='Query the system with a question')
    
    # Options
    parser.add_argument('--model', choices=['baseline', 'advanced'], default='advanced',
                       help='Model to use for queries (default: advanced)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Run full pipeline
    if args.all:
        print("=" * 60)
        print("RUNNING FULL PIPELINE")
        print("=" * 60)
        
        steps = [
            (PROJECT_ROOT / "scripts" / "01_download_squad_dataset.py", "Step 1: Download SQuAD Dataset"),
            (PROJECT_ROOT / "scripts" / "baseline" / "02_train_baseline.py", "Step 2: Train Baseline Model"),
            (PROJECT_ROOT / "scripts" / "advanced" / "03_train_advanced.py", "Step 3: Train Advanced Model"),
            (PROJECT_ROOT / "scripts" / "04_compare_models.py", "Step 4: Compare Models")
        ]
        
        for script, desc in steps:
            if not run_script(script, desc):
                print(f"\n⚠ Pipeline stopped due to error")
                return
        
        print("\n" + "=" * 60)
        print("FULL PIPELINE COMPLETE!")
        print("=" * 60)
        return
    
    # Download dataset
    if args.download:
        run_script(
            PROJECT_ROOT / "scripts" / "01_download_squad_dataset.py",
            "Downloading SQuAD Dataset"
        )
    
    # Train baseline
    if args.train_baseline:
        run_script(
            PROJECT_ROOT / "scripts" / "baseline" / "02_train_baseline.py",
            "Training Baseline Model"
        )
    
    # Train advanced
    if args.train_advanced:
        run_script(
            PROJECT_ROOT / "scripts" / "advanced" / "03_train_advanced.py",
            "Training Advanced Model"
        )
    
    # Compare models
    if args.compare:
        run_script(
            PROJECT_ROOT / "scripts" / "04_compare_models.py",
            "Comparing Models"
        )
    
    # Query
    if args.query:
        print(f"\nQuerying with {args.model} model...")
        print(f"Question: {args.query}")
        
        # Import and use the appropriate model
        if args.model == 'baseline':
            from src.baseline.tfidf_retriever import TFIDFRetriever
            
            model_path = PROJECT_ROOT / "outputs" / "models" / "baseline" / "tfidf_retriever.pkl"
            if not model_path.exists():
                print("⚠ Baseline model not found. Please train first:")
                print("  python main.py --train-baseline")
                return
            
            retriever = TFIDFRetriever()
            retriever.load(model_path)
            
            results = retriever.retrieve(args.query, top_k=3)
            answer = retriever.extract_answer(args.query, results)
            
            print(f"\nAnswer: {answer['answer'][:500]}...")
            print(f"Confidence: {answer['confidence']:.4f}")
            
        else:  # advanced
            from src.embedding.dense_retriever import DenseEmbeddingRetriever, RAGAnswerGenerator
            
            vector_db_dir = PROJECT_ROOT / "data" / "vector_db"
            if not vector_db_dir.exists():
                print("⚠ Advanced model not found. Please train first:")
                print("  python main.py --train-advanced")
                return
            
            retriever = DenseEmbeddingRetriever(
                collection_name='squad_train',
                persist_directory=vector_db_dir
            )
            
            results = retriever.retrieve(args.query, top_k=3)
            
            # Use RAG for answer generation
            rag = RAGAnswerGenerator()
            answer = rag.generate_answer(args.query, results)
            
            print(f"\nAnswer: {answer['answer']}")
            print(f"Confidence: {answer['confidence']:.4f}")
            print(f"Sources: {answer['num_sources']}")


if __name__ == "__main__":
    main()
