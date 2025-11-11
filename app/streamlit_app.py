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

# Don't import here - lazy load to avoid DLL errors
# from src.baseline.tfidf_retriever import TFIDFRetriever
# from src.embedding.dense_retriever import DenseEmbeddingRetriever, RAGAnswerGenerator


# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Load sample questions
@st.cache_data
def load_sample_questions(n=50):
    """Load sample questions from dev set"""
    import json
    import random

    qa_file = PROJECT_ROOT / "data" / "raw" / "squad" / "dev_qa_pairs.json"

    if not qa_file.exists():
        return []

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    # Get random sample
    sample = random.sample(qa_pairs, min(n, len(qa_pairs)))
    return [qa['question'] for qa in sample]

# Title
st.title("📚 RAG Document Q&A System")
st.markdown("Ask questions and get answers from documents using AI")

# Stats
@st.cache_data
def get_dataset_stats():
    """Get dataset statistics"""
    import json
    qa_file = PROJECT_ROOT / "data" / "raw" / "squad" / "dev_qa_pairs.json"

    if not qa_file.exists():
        return None

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    return {
        'total_questions': len(qa_pairs),
        'sample_questions': qa_pairs[:5]
    }

stats = get_dataset_stats()
if stats:
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        st.metric("📊 Total Questions", f"{stats['total_questions']:,}")
    with col_b:
        st.metric("📁 Dataset", "SQuAD v2.0")
    with col_c:
        st.info("💡 Select questions from sidebar or type your own!")
    st.markdown("---")

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

    st.markdown("---")
    st.markdown("### 📋 Sample Questions")

    sample_questions = load_sample_questions(100)

    if sample_questions:
        st.info(f"💡 {len(sample_questions)} random questions loaded from dataset")

        num_to_show = st.slider("Show questions", 5, 50, 10, key="num_questions")

        with st.expander("📝 Click to see and select questions", expanded=False):
            st.markdown("*Click any question to use it*")
            for i, q in enumerate(sample_questions[:num_to_show], 1):
                if st.button(f"❓ {q}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.selected_question = q
                    st.rerun()
    else:
        st.warning("No questions found. Make sure dataset is downloaded.")

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
    st.session_state.rag = None
    st.session_state.model_loaded = False
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""

# Load model
@st.cache_resource
def load_baseline_model():
    """Load baseline TF-IDF model"""
    # Import here to avoid DLL errors on startup
    from src.baseline.tfidf_retriever import TFIDFRetriever

    model_path = PROJECT_ROOT / "outputs" / "models" / "baseline" / "tfidf_retriever.pkl"

    if not model_path.exists():
        return None

    retriever = TFIDFRetriever()
    retriever.load(model_path)
    return retriever

@st.cache_resource
def load_advanced_model():
    """Load advanced embedding model"""
    # Import here to avoid DLL errors on startup
    from dotenv import load_dotenv
    from src.embedding.dense_retriever import DenseEmbeddingRetriever, RAGAnswerGenerator

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

    # Use selected question from sidebar if available
    default_query = st.session_state.selected_question if st.session_state.selected_question else ""

    query = st.text_input(
        "Your question:",
        value=default_query,
        placeholder="e.g., What is the capital of France?",
        help="Enter your question here or select from sample questions in sidebar"
    )

    search_button = st.button("🔍 Search", type="primary")

    # Clear selected question after displaying
    if st.session_state.selected_question:
        st.session_state.selected_question = ""

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
