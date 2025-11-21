"""
Test script to compare naive and advanced RAG agents.

Usage:
    python scripts/test_agents.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.naive_rag import NaiveRAGAgent
from src.agents.advanced_rag import AdvancedRAGAgent
from src.config import Config


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def format_result(result: dict, agent_name: str):
    """Format and print query results."""
    print(f"\n{'='*80}")
    print(f"  {agent_name.upper()} RESULTS")
    print(f"{'='*80}\n")

    print("RESPONSE:")
    print("-" * 80)
    print(result["response"])
    print()

    if agent_name == "naive":
        print(f"\nRETRIEVED CHUNKS: {result['num_chunks']}")
        print("-" * 80)
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"\nChunk {i} (Source: {chunk['metadata']['source']}):")
            print(f"Distance: {chunk.get('distance', 'N/A'):.4f}")
            print(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
    else:
        print(f"\nRETRIEVED & RERANKED ITEMS: {result['num_items']}")
        print("-" * 80)
        for i, item in enumerate(result["retrieved_items"], 1):
            print(f"\nItem {i} (Source: {item['metadata']['source']}):")
            print(f"Rerank Score: {item.get('rerank_score', 'N/A'):.4f}")
            print(f"Sentence: {item['sentence']}")
            print(f"Window: {item['window'][:200]}...")

    print("\n" + "=" * 80 + "\n")


def main():
    """Main test function."""
    print_section("RAG FALLACY MASTERCLASS - AGENT COMPARISON TEST")

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease create a .env file with your API keys.")
        print("See .env.example for reference.")
        return

    # Test query
    test_query = "Compare the Q1 revenue growth mentioned in the 10-Q report with the CEO's statements in the earnings call."

    print("TEST QUERY:")
    print(f"  \"{test_query}\"\n")

    # Initialize and test Naive RAG Agent
    print_section("TESTING NAIVE RAG AGENT")
    print("Initializing Naive RAG Agent...")

    naive_agent = NaiveRAGAgent()
    naive_agent.index_documents()

    print(f"✓ Indexed documents")
    print(f"✓ Configuration: Top-K = {Config.NAIVE_TOP_K}, Chunk Size = {Config.NAIVE_CHUNK_SIZE}\n")

    print("Querying Naive RAG Agent...")
    naive_result = naive_agent.query(test_query)
    format_result(naive_result, "naive")

    # Initialize and test Advanced RAG Agent
    print_section("TESTING ADVANCED RAG AGENT")
    print("Initializing Advanced RAG Agent...")

    advanced_agent = AdvancedRAGAgent()
    advanced_agent.index_documents()

    print(f"✓ Indexed documents")
    print(f"✓ Configuration: Top-K = {Config.ADVANCED_TOP_K} → Rerank Top-K = {Config.ADVANCED_RERANK_TOP_K}\n")

    print("Querying Advanced RAG Agent...")
    advanced_result = advanced_agent.query(test_query)
    format_result(advanced_result, "advanced")

    # Summary
    print_section("SUMMARY")
    print("NAIVE RAG AGENT:")
    print(f"  - Retrieved {naive_result['num_chunks']} chunks using basic vector search")
    print(f"  - No reranking applied")
    print(f"  - Response quality: Check above for accuracy\n")

    print("ADVANCED RAG AGENT:")
    print(f"  - Retrieved {len(advanced_result['retrieved_items'])} items with sentence-window retrieval")
    print(f"  - Applied Cohere reranking")
    print(f"  - Response quality: Check above for accuracy\n")

    print("KEY DIFFERENCES:")
    print("  1. Naive uses fixed-size chunks; Advanced uses sentences with context windows")
    print("  2. Naive relies only on vector similarity; Advanced adds reranking")
    print("  3. Advanced provides more precise, relevant context to the LLM")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
