"""
Advanced Retrieval Model using Dense Embeddings + RAG

This module implements an advanced retrieval system using:
- Dense embeddings (sentence-transformers)
- Vector database (ChromaDB)
- Semantic search
- LLM-based answer generation (RAG)

Similar to the transfer learning approach in the waste classification project,
this uses pre-trained models for better performance.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm


class DenseEmbeddingRetriever:
    """
    Advanced retrieval system using dense embeddings.
    
    This is the advanced model (similar to MobileNetV2 transfer learning
    in waste classification). It uses pre-trained embedding models.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        collection_name: str = 'squad_documents',
        persist_directory: Optional[Path] = None
    ):
        """
        Initialize dense embedding retriever.
        
        Args:
            model_name: Pre-trained sentence transformer model
            collection_name: Name for the vector database collection
            persist_directory: Directory to persist the vector database
        """
        print(f"Initializing advanced model with {model_name}...")
        
        # Load pre-trained embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"✓ Loaded pre-trained model: {model_name}")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        
        # Initialize ChromaDB
        if persist_directory:
            persist_directory = Path(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(persist_directory)
            )
        else:
            self.client = chromadb.Client()

        self.collection_name = collection_name
        self.collection = None

        # Try to load existing collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
            print(f"  - Documents in collection: {self.collection.count()}")
        except:
            print(f"ℹ No existing collection found. Will create on fit().")
        
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts into dense embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def fit(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32
    ):
        """
        Fit the retriever by encoding documents and building vector index.
        
        This is the "training" phase for advanced model.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            batch_size: Batch size for encoding
        """
        print(f"\nTraining advanced embedding model on {len(documents)} documents...")
        print("This uses pre-trained sentence-transformers (transfer learning)")
        
        # Create or get collection
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Encode documents in batches
        print("Encoding documents with pre-trained model...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Encoding batches"):
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = self.encode_texts(
                batch_docs, 
                batch_size=batch_size,
                show_progress=False
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        print(f"✓ Encoded {len(documents)} documents")
        print(f"  - Embedding shape: {embeddings.shape}")
        
        # Add to vector database
        print("Building vector index...")
        
        # Prepare metadata
        if metadata is None:
            metadata = [{'doc_id': i} for i in range(len(documents))]
        
        # Add in batches to ChromaDB
        batch_size_db = 1000
        for i in tqdm(range(0, len(documents), batch_size_db), desc="Adding to DB"):
            end_idx = min(i + batch_size_db, len(documents))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadata[i:end_idx],
                ids=[f"doc_{j}" for j in range(i, end_idx)]
            )
        
        print(f"✓ Advanced model training complete")
        print(f"  - Vector database: {self.collection.count()} documents")
        
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve top-k most semantically similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of retrieved documents with scores
        """
        if self.collection is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Encode query
        query_embedding = self.encode_texts([query], show_progress=False)[0]
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            # ChromaDB returns distances, convert to similarity
            distance = results['distances'][0][i]
            similarity = 1 - distance  # Cosine distance to similarity
            
            if similarity >= min_score:
                formatted_results.append({
                    'rank': i + 1,
                    'doc_id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'score': float(similarity),
                    'metadata': results['metadatas'][0][i]
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector database."""
        if self.collection is None:
            return {}
        
        return {
            'collection_name': self.collection_name,
            'num_documents': self.collection.count(),
            'embedding_dimension': self.embedding_dim,
            'model_name': self.embedding_model._model_card_vars.get('model_name', 'unknown')
        }


class RAGAnswerGenerator:
    """
    RAG (Retrieval-Augmented Generation) answer generator.

    Combines retrieval with LLM generation for high-quality answers.
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        """
        Initialize RAG answer generator.

        Args:
            model: LLM model to use
                - OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo
                - Gemini: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
        """
        import os

        self.model = model

        # Detect provider
        if model.startswith('gemini'):
            import google.generativeai as genai
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            self.provider = 'gemini'
        else:
            from openai import OpenAI
            self.client = OpenAI()  # Uses OPENAI_API_KEY from env
            self.provider = 'openai'

        print(f"✓ Initialized RAG with LLM: {model} ({self.provider})")
    
    def build_context(self, retrieved_docs: List[Dict]) -> str:
        """Build context from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.get('metadata', {})
            title = metadata.get('title', 'Unknown')
            doc_id = metadata.get('doc_id', 'N/A')
            text = doc['text']
            score = doc['score']
            
            context_parts.append(
                f"[Source {i+1}: {title} (ID: {doc_id}, Relevance: {score:.3f})]\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ) -> Dict:
        """
        Generate answer using RAG approach.
        
        Args:
            query: User question
            retrieved_docs: Retrieved documents from retriever
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            Answer dictionary with text, citations, and metadata
        """
        if not retrieved_docs:
            return {
                'answer': "I don't have enough information to answer this question.",
                'confidence': 0.0,
                'sources': [],
                'method': 'rag'
            }
        
        # Build context
        context = self.build_context(retrieved_docs)
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Always cite your sources using [Source X] notation.
If the context doesn't contain enough information, say so clearly.
Be concise and accurate."""
        
        user_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer (include citations):"""
        
        # Call LLM based on provider with retry logic
        import time

        try:
            answer_text = None
            for attempt in range(retry_attempts):
                try:
                    if self.provider == 'gemini':
                        # Gemini API call
                        full_prompt = f"{system_prompt}\n\n{user_prompt}"
                        response = self.client.generate_content(
                            full_prompt,
                            generation_config={
                                'temperature': temperature,
                                'max_output_tokens': max_tokens,
                            }
                        )
                        answer_text = response.text
                    else:
                        # OpenAI API call
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        answer_text = response.choices[0].message.content

                    # Success - break out of retry loop
                    break

                except Exception as e:
                    error_msg = str(e)
                    if '429' in error_msg or 'rate limit' in error_msg.lower():
                        if attempt < retry_attempts - 1:
                            wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                            print(f"⚠ Rate limit hit, waiting {wait_time}s before retry {attempt + 2}/{retry_attempts}...")
                            time.sleep(wait_time)
                            continue
                    # Re-raise if not rate limit or out of retries
                    raise

            # Extract sources
            sources = [
                {
                    'title': doc['metadata'].get('title', 'Unknown'),
                    'doc_id': doc['metadata'].get('doc_id', 'N/A'),
                    'relevance': doc['score']
                }
                for doc in retrieved_docs
            ]

            return {
                'answer': answer_text,
                'confidence': retrieved_docs[0]['score'],  # Use top result score
                'sources': sources,
                'method': 'rag',
                'num_sources': len(retrieved_docs)
            }

        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'method': 'rag',
                'num_sources': 0,
                'error': str(e)
            }
