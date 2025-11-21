"""
Naive RAG Agent - demonstrates the "RAG Fallacy"

This agent uses basic chunking and vector search, which often fails on complex queries.
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import hashlib

from src.config import Config
from src.utils.document_loader import load_documents, basic_text_splitter


class NaiveRAGAgent:
    """
    A naive RAG implementation that demonstrates common failure patterns.

    Issues with this approach:
    1. Fixed chunk size loses semantic boundaries
    2. No reranking - relies only on vector similarity
    3. Context windows are limited to chunk boundaries
    4. No handling of information spread across multiple chunks
    """

    def __init__(self, data_dir: str = None, langfuse_enabled: bool = False):
        """
        Initialize the naive RAG agent.

        Args:
            data_dir: Directory containing documents to index
            langfuse_enabled: Whether to enable LangFuse tracing
        """
        self.config = Config()
        self.data_dir = data_dir or Config.DATA_DIR
        self.langfuse_enabled = langfuse_enabled

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)

        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # Collection name
        self.collection_name = "naive_rag_collection"
        self.collection = None

        # LangFuse integration
        self.langfuse = None
        if self.langfuse_enabled:
            try:
                from langfuse import Langfuse
                self.langfuse = Langfuse(
                    public_key=Config.LANGFUSE_PUBLIC_KEY,
                    secret_key=Config.LANGFUSE_SECRET_KEY,
                    host=Config.LANGFUSE_HOST
                )
            except Exception as e:
                print(f"Warning: Could not initialize LangFuse: {e}")
                self.langfuse_enabled = False

    def index_documents(self, force_reindex: bool = False):
        """
        Load and index documents using naive chunking strategy.

        Args:
            force_reindex: If True, delete existing collection and reindex
        """
        # Check if collection exists
        try:
            if force_reindex:
                self.chroma_client.delete_collection(self.collection_name)
        except:
            pass

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Naive RAG collection with basic chunking"}
        )

        # Check if already indexed
        if self.collection.count() > 0 and not force_reindex:
            print(f"Collection already indexed with {self.collection.count()} chunks")
            return

        # Load documents
        print("Loading documents...")
        documents = load_documents(self.data_dir)

        # Chunk and embed documents
        print("Chunking and indexing documents...")
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            chunks = basic_text_splitter(
                doc["content"],
                chunk_size=Config.NAIVE_CHUNK_SIZE,
                chunk_overlap=Config.NAIVE_CHUNK_OVERLAP
            )

            for i, chunk in enumerate(chunks):
                # Generate unique ID
                chunk_id = hashlib.md5(f"{doc['source']}_{i}".encode()).hexdigest()

                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": doc["source"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                all_ids.append(chunk_id)

        # Get embeddings from OpenAI
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self._get_embeddings(all_chunks)

        # Add to collection
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )

        print(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI."""
        response = self.client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks using basic vector similarity.

        Args:
            query: User query
            top_k: Number of chunks to retrieve

        Returns:
            List of retrieved chunks with metadata
        """
        if self.collection is None:
            raise ValueError("Documents not indexed. Call index_documents() first.")

        top_k = top_k or Config.NAIVE_TOP_K

        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        retrieved_chunks = []
        for i in range(len(results["documents"][0])):
            retrieved_chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })

        return retrieved_chunks

    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate response using retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks

        Returns:
            Generated response
        """
        # Build context
        context = "\n\n---\n\n".join([
            f"Source: {chunk['metadata']['source']}\n{chunk['text']}"
            for chunk in context_chunks
        ])

        # Create prompt
        prompt = f"""You are a financial analyst assistant. Answer the user's question based on the provided context.

Context:
{context}

User Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite specific sources when referencing information
- If the context doesn't contain enough information to fully answer, say so
- Be precise with numbers and facts

Answer:"""

        # Generate response
        response = self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=Config.LLM_TEMPERATURE
        )

        return response.choices[0].message.content

    def query(self, question: str) -> Dict:
        """
        Execute full RAG pipeline: retrieve and generate.

        Args:
            question: User question

        Returns:
            Dictionary with response and metadata
        """
        trace = None

        # Start LangFuse trace if enabled
        if self.langfuse_enabled and self.langfuse:
            trace = self.langfuse.trace(
                name="naive_rag_query",
                metadata={"agent_type": "naive"}
            )

        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question)

        if trace:
            trace.span(
                name="retrieval",
                metadata={
                    "num_chunks": len(retrieved_chunks),
                    "top_k": Config.NAIVE_TOP_K
                }
            )

        # Generate response
        response = self.generate_response(question, retrieved_chunks)

        if trace:
            trace.span(
                name="generation",
                input=question,
                output=response
            )

        return {
            "response": response,
            "retrieved_chunks": retrieved_chunks,
            "num_chunks": len(retrieved_chunks),
            "agent_type": "naive"
        }
