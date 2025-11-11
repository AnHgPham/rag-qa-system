"""
Simple Streamlit Web Interface for RAG Document Q&A

This is an optional web interface. The core functionality is in CLI (main.py).

Run:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.tfidf_retriever import TFIDFRetriever
from src.embedding.dense_retriever import DenseEmbeddingRetriever, RAGAnswerGenerator


# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 RAG Document Q&A System")
st.markdown("Ask questions and get answers from documents using AI")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    model_type = st.selectbox(
        "Select Model",
        ["Advanced (Embeddings + RAG)", "Baseline (TF-IDF)"],
        help="Advanced model is more accurate but slower"
    )
    
    top_k = st.slider("Number of results", 1, 10, 5)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system uses:
    - **Baseline**: TF-IDF retrieval
    - **Advanced**: Dense embeddings + RAG
    
    Trained on SQuAD dataset.
    """)

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
    st.session_state.rag = None
    st.session_state.model_loaded = False

# Load model
@st.cache_resource
def load_baseline_model():
    """Load baseline TF-IDF model"""
    model_path = PROJECT_ROOT / "outputs" / "models" / "baseline" / "tfidf_retriever.pkl"
    
    if not model_path.exists():
        return None
    
    retriever = TFIDFRetriever()
    retriever.load(model_path)
    return retriever

@st.cache_resource
def load_advanced_model():
    """Load advanced embedding model"""
    from dotenv import load_dotenv
    load_dotenv()

    vector_db_dir = PROJECT_ROOT / "data" / "vector_db"

    if not vector_db_dir.exists():
        return None, None

    retriever = DenseEmbeddingRetriever(
        collection_name='squad_train',
        persist_directory=vector_db_dir
    )

    rag = RAGAnswerGenerator(model="gemini-2.0-flash")

    return retriever, rag

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Ask a Question")
    
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What is the capital of France?",
        help="Enter your question here"
    )
    
    search_button = st.button("🔍 Search", type="primary")

with col2:
    st.header("Quick Examples")
    
    examples = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?",
        "What is photosynthesis?"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            query = example
            search_button = True

# Process query
if search_button and query:
    with st.spinner("Processing..."):
        try:
            if "Advanced" in model_type:
                # Load advanced model
                retriever, rag = load_advanced_model()
                
                if retriever is None:
                    st.error("⚠️ Advanced model not found. Please train first: `python main.py --train-advanced`")
                else:
                    # Retrieve
                    results = retriever.retrieve(query, top_k=top_k)
                    
                    # Generate answer
                    answer = rag.generate_answer(query, results)
                    
                    # Display answer
                    st.success("✅ Answer Generated")
                    
                    st.markdown("### 📝 Answer")
                    st.markdown(answer['answer'])
                    
                    st.markdown(f"**Confidence**: {answer['confidence']:.4f}")
                    st.markdown(f"**Sources**: {answer['num_sources']}")
                    
                    # Display sources
                    st.markdown("### 📚 Sources")
                    for i, doc in enumerate(results):
                        with st.expander(f"Source {i+1} (Score: {doc['score']:.4f})"):
                            st.markdown(f"**Title**: {doc['metadata'].get('title', 'N/A')}")
                            st.markdown(f"**Text**: {doc['text']}")
            
            else:
                # Load baseline model
                retriever = load_baseline_model()
                
                if retriever is None:
                    st.error("⚠️ Baseline model not found. Please train first: `python main.py --train-baseline`")
                else:
                    # Retrieve
                    results = retriever.retrieve(query, top_k=top_k)
                    
                    # Extract answer
                    answer = retriever.extract_answer(query, results)
                    
                    # Display answer
                    st.success("✅ Results Retrieved")
                    
                    st.markdown("### 📝 Answer")
                    st.markdown(answer['answer'][:500] + "...")
                    
                    st.markdown(f"**Confidence**: {answer['confidence']:.4f}")
                    
                    # Display results
                    st.markdown("### 📚 Retrieved Documents")
                    for i, doc in enumerate(results):
                        with st.expander(f"Result {i+1} (Score: {doc['score']:.4f})"):
                            st.markdown(f"**Title**: {doc['metadata'].get('title', 'N/A')}")
                            st.markdown(f"**Text**: {doc['text']}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using RAG (Retrieval-Augmented Generation)</p>
    <p>Baseline: TF-IDF | Advanced: sentence-transformers + ChromaDB + LLM</p>
</div>
""", unsafe_allow_html=True)
