"""
RAG Fallacy Masterclass - Interactive Demo

This Streamlit app demonstrates the difference between naive and advanced RAG architectures.
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.agents.naive_rag import NaiveRAGAgent
from src.agents.advanced_rag import AdvancedRAGAgent
from src.config import Config


# Page configuration
st.set_page_config(
    page_title="RAG Fallacy Masterclass",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin-bottom: 1rem;
    }
    .naive-card {
        background-color: #ffebee;
        border-color: #f44336;
    }
    .advanced-card {
        background-color: #e8f5e9;
        border-color: #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_agents():
    """Initialize both agents and cache them."""
    if "naive_agent" not in st.session_state:
        with st.spinner("Initializing Naive RAG Agent..."):
            naive_agent = NaiveRAGAgent(langfuse_enabled=False)
            naive_agent.index_documents()
            st.session_state.naive_agent = naive_agent

    if "advanced_agent" not in st.session_state:
        with st.spinner("Initializing Advanced RAG Agent..."):
            advanced_agent = AdvancedRAGAgent(langfuse_enabled=False)
            advanced_agent.index_documents()
            st.session_state.advanced_agent = advanced_agent


def display_header():
    """Display the main header."""
    st.markdown('<div class="main-header">🧠 The RAG Fallacy: An Architectural Clinic</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demonstrating why "just having data" isn\'t enough for trustworthy agents</div>', unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with information and controls."""
    with st.sidebar:
        st.header("About This Demo")

        st.markdown("""
        This demo shows the critical difference between **naive** and **advanced** RAG architectures.

        ### 🔴 Naive RAG
        - Basic chunking (1000 chars)
        - Simple vector search
        - No reranking
        - Often fails on complex queries

        ### 🟢 Advanced RAG
        - Sentence-window retrieval
        - Cohere reranking
        - Better context handling
        - Reliable on complex queries
        """)

        st.divider()

        st.header("Try This Query")
        st.info("""
        **Complex Question:**
        "Compare the Q1 revenue growth mentioned in the 10-Q report with the CEO's statements in the earnings call."

        This query requires:
        1. Finding info from both documents
        2. Understanding numerical differences
        3. Explaining any discrepancies
        """)

        st.divider()

        st.header("Configuration")
        st.caption(f"Embedding Model: {Config.EMBEDDING_MODEL}")
        st.caption(f"LLM Model: {Config.LLM_MODEL}")
        st.caption(f"Naive Top-K: {Config.NAIVE_TOP_K}")
        st.caption(f"Advanced Top-K: {Config.ADVANCED_TOP_K} → {Config.ADVANCED_RERANK_TOP_K}")


def format_retrieved_context(result: dict, agent_type: str):
    """Format and display retrieved context."""
    with st.expander("🔍 View Retrieved Context", expanded=False):
        if agent_type == "naive":
            chunks = result.get("retrieved_chunks", [])
            st.write(f"**Retrieved {len(chunks)} chunks:**")

            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}** (Source: {chunk['metadata']['source']})")
                st.text_area(
                    f"chunk_{i}",
                    chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                    height=150,
                    key=f"naive_chunk_{i}",
                    label_visibility="collapsed"
                )

        else:  # advanced
            items = result.get("retrieved_items", [])
            st.write(f"**Retrieved and reranked {len(items)} items:**")

            for i, item in enumerate(items):
                rerank_score = item.get('rerank_score', 'N/A')
                st.markdown(f"**Item {i+1}** (Source: {item['metadata']['source']}, Rerank Score: {rerank_score:.3f})")

                col1, col2 = st.columns(2)

                with col1:
                    st.caption("Matched Sentence:")
                    st.text_area(
                        f"sentence_{i}",
                        item['sentence'],
                        height=100,
                        key=f"adv_sent_{i}",
                        label_visibility="collapsed"
                    )

                with col2:
                    st.caption("Context Window:")
                    st.text_area(
                        f"window_{i}",
                        item['window'][:300] + "..." if len(item['window']) > 300 else item['window'],
                        height=100,
                        key=f"adv_win_{i}",
                        label_visibility="collapsed"
                    )


def main():
    """Main application."""
    # Display header
    display_header()

    # Display sidebar
    display_sidebar()

    # Initialize agents
    try:
        Config.validate()
        initialize_agents()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.info("Please create a `.env` file with your API keys. See `.env.example` for reference.")
        return

    # Main content
    st.divider()

    # Agent selection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="agent-card naive-card">', unsafe_allow_html=True)
        st.markdown("### 🔴 Naive RAG Agent")
        st.caption("Basic vector search with fixed chunking")
        naive_selected = st.checkbox("Use Naive Agent", value=True, key="naive_check")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="agent-card advanced-card">', unsafe_allow_html=True)
        st.markdown("### 🟢 Advanced RAG Agent")
        st.caption("Sentence-window retrieval + reranking")
        advanced_selected = st.checkbox("Use Advanced Agent", value=True, key="advanced_check")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Query input
    st.subheader("Ask a Question")

    default_question = "Compare the Q1 revenue growth mentioned in the 10-Q report with the CEO's statements in the earnings call."

    question = st.text_area(
        "Enter your question:",
        value=default_question,
        height=100,
        placeholder="Ask a complex question that requires information from multiple sources..."
    )

    if st.button("🚀 Ask Both Agents", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        st.divider()
        st.subheader("Results")

        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)

        # Query Naive Agent
        if naive_selected:
            with col1:
                st.markdown("### 🔴 Naive RAG Response")
                with st.spinner("Querying Naive RAG Agent..."):
                    try:
                        naive_result = st.session_state.naive_agent.query(question)
                        st.markdown(naive_result["response"])

                        st.divider()
                        format_retrieved_context(naive_result, "naive")

                    except Exception as e:
                        st.error(f"Error: {e}")

        # Query Advanced Agent
        if advanced_selected:
            with col2:
                st.markdown("### 🟢 Advanced RAG Response")
                with st.spinner("Querying Advanced RAG Agent..."):
                    try:
                        advanced_result = st.session_state.advanced_agent.query(question)
                        st.markdown(advanced_result["response"])

                        st.divider()
                        format_retrieved_context(advanced_result, "advanced")

                    except Exception as e:
                        st.error(f"Error: {e}")

    # Footer
    st.divider()
    st.markdown("""
    ---
    **The RAG Fallacy Masterclass** by Pranjal Joshi

    📧 Contact: pranjal.alias@gmail.com | 💼 LinkedIn: [linkedin.com/in/jpranjal](https://linkedin.com/in/jpranjal)
    """)


if __name__ == "__main__":
    main()
