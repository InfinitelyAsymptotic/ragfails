"""
Test script with LangFuse observability enabled.

This script demonstrates how to use LangFuse to trace and debug RAG queries.

Usage:
    python scripts/test_with_langfuse.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.naive_rag import NaiveRAGAgent
from src.agents.advanced_rag import AdvancedRAGAgent
from src.config import Config


def main():
    """Main test function with LangFuse tracing."""
    print("=" * 80)
    print("  RAG FALLACY MASTERCLASS - LANGFUSE OBSERVABILITY DEMO")
    print("=" * 80)
    print()

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return

    # Check LangFuse credentials
    if not Config.LANGFUSE_PUBLIC_KEY or not Config.LANGFUSE_SECRET_KEY:
        print("⚠️  Warning: LangFuse credentials not set in .env file")
        print("   Tracing will be disabled.")
        print("\n   To enable LangFuse observability:")
        print("   1. Sign up at https://cloud.langfuse.com")
        print("   2. Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env")
        print()
        langfuse_enabled = False
    else:
        print(f"✓ LangFuse configured: {Config.LANGFUSE_HOST}")
        langfuse_enabled = True
        print()

    # Test query
    test_query = "Compare the Q1 revenue growth mentioned in the 10-Q report with the CEO's statements in the earnings call."

    print("TEST QUERY:")
    print(f'  "{test_query}"')
    print()

    # Test Naive RAG Agent
    print("-" * 80)
    print("TESTING NAIVE RAG AGENT (with LangFuse tracing)")
    print("-" * 80)
    print()

    naive_agent = NaiveRAGAgent(langfuse_enabled=langfuse_enabled)
    print("Indexing documents...")
    naive_agent.index_documents()

    print("Querying naive agent...")
    naive_result = naive_agent.query(test_query)

    print("\nNAIVE RAG RESPONSE:")
    print("-" * 80)
    print(naive_result["response"])
    print()

    # Test Advanced RAG Agent
    print("-" * 80)
    print("TESTING ADVANCED RAG AGENT (with LangFuse tracing)")
    print("-" * 80)
    print()

    advanced_agent = AdvancedRAGAgent(langfuse_enabled=langfuse_enabled)
    print("Indexing documents...")
    advanced_agent.index_documents()

    print("Querying advanced agent...")
    advanced_result = advanced_agent.query(test_query)

    print("\nADVANCED RAG RESPONSE:")
    print("-" * 80)
    print(advanced_result["response"])
    print()

    # Instructions
    print("=" * 80)
    if langfuse_enabled:
        print("✓ TRACES SENT TO LANGFUSE")
        print()
        print("Next steps:")
        print("  1. Visit your LangFuse dashboard: https://cloud.langfuse.com")
        print("  2. Navigate to 'Traces' to see the query traces")
        print("  3. Compare the 'naive_rag_query' and 'advanced_rag_query' traces")
        print("  4. Examine the retrieved context in each trace")
        print()
        print("Key things to observe:")
        print("  - Naive: Large, potentially irrelevant chunks")
        print("  - Advanced: Precise sentences with reranking scores")
        print("  - Context quality difference between the two approaches")
    else:
        print("ℹ️  TRACING DISABLED")
        print()
        print("To enable observability:")
        print("  1. Sign up for LangFuse: https://cloud.langfuse.com")
        print("  2. Add credentials to .env file")
        print("  3. Run this script again")

    print("=" * 80)


if __name__ == "__main__":
    main()
