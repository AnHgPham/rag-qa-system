#!/usr/bin/env python3
"""
Script to download Stanford Question Answering Dataset (SQuAD) from Kaggle.

This script downloads the SQuAD v1.1 dataset which contains:
- 100,000+ question-answer pairs
- 500+ Wikipedia articles
- Training and development sets
"""

import os
import sys
import json
import requests
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SQUAD_DIR = RAW_DIR / "squad"

# Create directories
SQUAD_DIR.mkdir(parents=True, exist_ok=True)

# SQuAD dataset URLs (direct from Stanford)
SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
}


def download_file(url, output_path):
    """Download file with progress bar."""
    print(f"Downloading {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    print(f"✓ Downloaded to {output_path}")


def load_and_inspect_squad(file_path):
    """Load and inspect SQuAD dataset."""
    print(f"\nInspecting {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count statistics
    num_articles = len(data['data'])
    num_paragraphs = sum(len(article['paragraphs']) for article in data['data'])
    num_qas = sum(
        len(para['qas']) 
        for article in data['data'] 
        for para in article['paragraphs']
    )
    
    print(f"  Articles: {num_articles}")
    print(f"  Paragraphs: {num_paragraphs}")
    print(f"  Question-Answer pairs: {num_qas}")
    
    # Show sample
    sample_article = data['data'][0]
    sample_para = sample_article['paragraphs'][0]
    sample_qa = sample_para['qas'][0]
    
    print(f"\nSample from {file_path.name}:")
    print(f"  Article: {sample_article['title']}")
    print(f"  Context: {sample_para['context'][:200]}...")
    print(f"  Question: {sample_qa['question']}")
    print(f"  Answer: {sample_qa['answers'][0]['text']}")
    
    return data


def extract_documents(squad_data, output_file):
    """Extract all unique documents (contexts) from SQuAD."""
    print(f"\nExtracting documents...")
    
    documents = []
    doc_id = 0
    
    for article in squad_data['data']:
        title = article['title']
        
        for para in article['paragraphs']:
            context = para['context']
            
            # Extract questions for this context
            questions = [qa['question'] for qa in para['qas']]
            answers = [qa['answers'][0]['text'] if qa['answers'] else "" 
                      for qa in para['qas']]
            
            documents.append({
                'doc_id': doc_id,
                'title': title,
                'context': context,
                'questions': questions,
                'answers': answers,
                'num_questions': len(questions)
            })
            
            doc_id += 1
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Extracted {len(documents)} documents to {output_file}")
    
    return documents


def create_qa_pairs(squad_data, output_file):
    """Create question-answer pairs dataset."""
    print(f"\nCreating Q&A pairs...")
    
    qa_pairs = []
    qa_id = 0
    
    for article in squad_data['data']:
        title = article['title']
        
        for para in article['paragraphs']:
            context = para['context']
            
            for qa in para['qas']:
                question = qa['question']
                answer_text = qa['answers'][0]['text'] if qa['answers'] else ""
                answer_start = qa['answers'][0]['answer_start'] if qa['answers'] else -1
                
                qa_pairs.append({
                    'qa_id': qa_id,
                    'question_id': qa['id'],
                    'title': title,
                    'context': context,
                    'question': question,
                    'answer': answer_text,
                    'answer_start': answer_start
                })
                
                qa_id += 1
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created {len(qa_pairs)} Q&A pairs to {output_file}")
    
    return qa_pairs


def main():
    """Main function."""
    print("=" * 60)
    print("SQuAD Dataset Download and Preparation")
    print("=" * 60)
    
    # Download datasets
    for split, url in SQUAD_URLS.items():
        output_path = SQUAD_DIR / f"{split}-v1.1.json"
        
        if output_path.exists():
            print(f"✓ {output_path.name} already exists, skipping download")
        else:
            download_file(url, output_path)
    
    # Load and inspect
    train_data = load_and_inspect_squad(SQUAD_DIR / "train-v1.1.json")
    dev_data = load_and_inspect_squad(SQUAD_DIR / "dev-v1.1.json")
    
    # Extract documents
    train_docs = extract_documents(
        train_data, 
        SQUAD_DIR / "train_documents.json"
    )
    dev_docs = extract_documents(
        dev_data, 
        SQUAD_DIR / "dev_documents.json"
    )
    
    # Create Q&A pairs
    train_qas = create_qa_pairs(
        train_data,
        SQUAD_DIR / "train_qa_pairs.json"
    )
    dev_qas = create_qa_pairs(
        dev_data,
        SQUAD_DIR / "dev_qa_pairs.json"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset Download Complete!")
    print("=" * 60)
    print(f"Training set:")
    print(f"  - Documents: {len(train_docs)}")
    print(f"  - Q&A pairs: {len(train_qas)}")
    print(f"Dev set:")
    print(f"  - Documents: {len(dev_docs)}")
    print(f"  - Q&A pairs: {len(dev_qas)}")
    print(f"\nFiles saved to: {SQUAD_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
