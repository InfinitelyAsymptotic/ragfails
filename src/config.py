"""
Configuration for RAG agents.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for RAG agents."""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    # LangFuse Configuration
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Model Configuration
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.0

    # Retrieval Configuration
    NAIVE_CHUNK_SIZE = 1000
    NAIVE_CHUNK_OVERLAP = 200
    NAIVE_TOP_K = 3

    ADVANCED_WINDOW_SIZE = 3  # sentences
    ADVANCED_TOP_K = 10
    ADVANCED_RERANK_TOP_K = 3

    # Data paths
    DATA_DIR = "data"
    CHROMA_DIR = ".chroma_db"

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in .env file.")

        if not cls.COHERE_API_KEY:
            print("Warning: COHERE_API_KEY not set. Advanced RAG reranking will not work.")

        return True
