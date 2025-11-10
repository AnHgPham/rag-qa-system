"""
Baseline Retrieval Model using TF-IDF + BM25

This module implements a traditional information retrieval system using:
- TF-IDF vectorization for document representation
- Cosine similarity for ranking
- Simple heuristic-based answer extraction

Similar to the baseline CNN in the waste classification project,
this serves as a performance baseline before applying advanced techniques.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class TFIDFRetriever:
    """
    Baseline retrieval system using TF-IDF vectorization.
    
    This is the baseline model (similar to baseline CNN in waste classification).
    It uses traditional IR techniques without deep learning.
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.8
    ):
        """
        Initialize TF-IDF retriever.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency (to filter common words)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.tfidf_matrix = None
        self.documents = None
        self.metadata = None
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better retrieval.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def fit(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Fit the TF-IDF vectorizer on documents.
        
        This is the "training" phase for baseline model.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        print(f"Training baseline TF-IDF model on {len(documents)} documents...")
        
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        self.documents = documents
        self.metadata = metadata if metadata else [{} for _ in documents]
        
        # Print statistics
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"✓ Baseline model trained")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Document count: {len(documents)}")
        print(f"  - TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"  - Matrix sparsity: {1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4f}")
        
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of retrieved documents with scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Preprocess and vectorize query
        processed_query = self.preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by minimum score
        results = []
        for rank, idx in enumerate(top_indices):
            score = similarities[idx]
            
            if score >= min_score:
                results.append({
                    'rank': rank + 1,
                    'doc_id': idx,
                    'text': self.documents[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def extract_answer(
        self, 
        query: str, 
        retrieved_docs: List[Dict],
        strategy: str = 'top_chunk'
    ) -> Dict:
        """
        Extract answer from retrieved documents using simple heuristics.
        
        This is a simple baseline approach (no LLM).
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents
            strategy: Answer extraction strategy
                - 'top_chunk': Return highest scoring chunk
                - 'combine_top3': Combine top 3 chunks
                - 'sentence_match': Find best matching sentence
                
        Returns:
            Answer dictionary with text and metadata
        """
        if not retrieved_docs:
            return {
                'answer': "No relevant information found.",
                'confidence': 0.0,
                'source': None
            }
        
        if strategy == 'top_chunk':
            # Simply return the top chunk
            top_doc = retrieved_docs[0]
            return {
                'answer': top_doc['text'],
                'confidence': top_doc['score'],
                'source': top_doc['metadata'],
                'rank': 1
            }
        
        elif strategy == 'combine_top3':
            # Combine top 3 chunks
            top_3 = retrieved_docs[:3]
            combined_text = "\n\n".join([doc['text'] for doc in top_3])
            avg_score = np.mean([doc['score'] for doc in top_3])
            
            return {
                'answer': combined_text,
                'confidence': float(avg_score),
                'source': [doc['metadata'] for doc in top_3],
                'num_sources': len(top_3)
            }
        
        elif strategy == 'sentence_match':
            # Find sentence with highest word overlap
            query_words = set(query.lower().split())
            best_sentence = None
            best_score = 0
            best_source = None
            
            for doc in retrieved_docs[:5]:
                sentences = re.split(r'[.!?]+', doc['text'])
                
                for sent in sentences:
                    sent_words = set(sent.lower().split())
                    overlap = len(query_words & sent_words)
                    
                    if overlap > best_score:
                        best_score = overlap
                        best_sentence = sent.strip()
                        best_source = doc['metadata']
            
            return {
                'answer': best_sentence if best_sentence else retrieved_docs[0]['text'],
                'confidence': best_score / len(query_words) if query_words else 0,
                'source': best_source
            }
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def save(self, save_path: Path):
        """Save the trained model."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'documents': self.documents,
            'metadata': self.metadata
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Baseline model saved to {save_path}")
    
    def load(self, load_path: Path):
        """Load a trained model."""
        load_path = Path(load_path)
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.documents = model_data['documents']
        self.metadata = model_data['metadata']
        
        print(f"✓ Baseline model loaded from {load_path}")
        print(f"  - Documents: {len(self.documents)}")
        print(f"  - Vocabulary: {len(self.vectorizer.vocabulary_)}")


class BM25Retriever:
    """
    BM25 retrieval system (improved version of TF-IDF).
    
    BM25 is a ranking function used by search engines to estimate
    the relevance of documents to a given search query.
    It's an improvement over TF-IDF.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.vectorizer = TfidfVectorizer(
            use_idf=True,
            norm=None,
            stop_words='english'
        )
        self.documents = None
        self.doc_lengths = None
        self.avg_doc_length = None
        self.idf = None
        
    def fit(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Fit BM25 on documents."""
        print(f"Training BM25 model on {len(documents)} documents...")
        
        self.documents = documents
        self.metadata = metadata if metadata else [{} for _ in documents]
        
        # Fit vectorizer
        tf_matrix = self.vectorizer.fit_transform(documents)
        
        # Calculate document lengths
        self.doc_lengths = np.array(tf_matrix.sum(axis=1)).flatten()
        self.avg_doc_length = self.doc_lengths.mean()
        
        # Get IDF values
        self.idf = self.vectorizer.idf_
        
        print(f"✓ BM25 model trained")
        print(f"  - Average document length: {self.avg_doc_length:.2f}")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents using BM25 scoring."""
        if self.documents is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate BM25 scores
        scores = []
        for idx, doc in enumerate(self.documents):
            doc_vec = self.vectorizer.transform([doc])
            
            # BM25 formula
            score = 0
            for term_idx in query_vec.nonzero()[1]:
                tf = doc_vec[0, term_idx]
                idf = self.idf[term_idx]
                doc_len = self.doc_lengths[idx]
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
            
            scores.append(score)
        
        # Get top-k
        scores = np.array(scores)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'rank': rank + 1,
                'doc_id': idx,
                'text': self.documents[idx],
                'score': float(scores[idx]),
                'metadata': self.metadata[idx]
            })
        
        return results
