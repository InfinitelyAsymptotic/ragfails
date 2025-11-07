"""
Advanced RAG Agent - demonstrates superior architecture

This agent uses sentence-window retrieval and reranking to provide accurate results.
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import hashlib
import cohere

from src.config import Config
from src.utils.document_loader import load_documents, sentence_splitter


class AdvancedRAGAgent:
    """
    An advanced RAG implementation with sentence-window retrieval and reranking.

    Key improvements:
    1. Sentence-level retrieval with context windows
    2. Reranking using Cohere for better relevance
    3. Better handling of information spread across chunks
    4. More precise context delivery to LLM
    """

    def __init__(self, data_dir: str = None, langfuse_enabled: bool = False):
        """
        Initialize the advanced RAG agent.

        Args:
            data_dir: Directory containing documents to index
            langfuse_enabled: Whether to enable LangFuse tracing
        """
        self.config = Config()
        self.data_dir = data_dir or Config.DATA_DIR
        self.langfuse_enabled = langfuse_enabled

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)

        # Initialize Cohere client for reranking
        self.cohere_client = None
        if self.config.COHERE_API_KEY:
            self.cohere_client = cohere.Client(self.config.COHERE_API_KEY)

        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # Collection name
        self.collection_name = "advanced_rag_collection"
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
        Load and index documents using sentence-window strategy.

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
            metadata={"description": "Advanced RAG collection with sentence-window retrieval"}
        )

        # Check if already indexed
        if self.collection.count() > 0 and not force_reindex:
            print(f"Collection already indexed with {self.collection.count()} sentences")
            return

        # Load documents
        print("Loading documents...")
        documents = load_documents(self.data_dir)

        # Process documents with sentence-window approach
        print("Processing documents with sentence-window strategy...")
        all_sentences = []
        all_windows = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            sentence_windows = sentence_splitter(doc["content"])

            for sw in sentence_windows:
                # Generate unique ID
                sentence_id = hashlib.md5(
                    f"{doc['source']}_{sw['position']}".encode()
                ).hexdigest()

                # We embed the sentence but store both sentence and window
                all_sentences.append(sw["sentence"])
                all_windows.append(sw["window"])
                all_metadatas.append({
                    "source": doc["source"],
                    "position": sw["position"],
                    "total_sentences": sw["total_sentences"],
                    "window": sw["window"]  # Store window in metadata
                })
                all_ids.append(sentence_id)

        # Get embeddings for sentences (not windows)
        print(f"Generating embeddings for {len(all_sentences)} sentences...")
        embeddings = self._get_embeddings(all_sentences)

        # Add to collection - store sentences for embedding, windows in metadata
        self.collection.add(
            documents=all_sentences,  # Small, precise sentences for retrieval
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )

        print(f"Indexed {len(all_sentences)} sentences from {len(documents)} documents")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI."""
        response = self.client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant sentences using vector similarity.

        Args:
            query: User query
            top_k: Number of sentences to retrieve (before reranking)

        Returns:
            List of retrieved sentences with windows
        """
        if self.collection is None:
            raise ValueError("Documents not indexed. Call index_documents() first.")

        top_k = top_k or Config.ADVANCED_TOP_K

        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        retrieved_items = []
        for i in range(len(results["documents"][0])):
            retrieved_items.append({
                "sentence": results["documents"][0][i],
                "window": results["metadatas"][0][i]["window"],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })

        return retrieved_items

    def rerank(self, query: str, retrieved_items: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Rerank retrieved items using Cohere's reranking model.

        Args:
            query: User query
            retrieved_items: Items to rerank
            top_k: Number of top items to keep after reranking

        Returns:
            Reranked and filtered items
        """
        if not self.cohere_client:
            print("Warning: Cohere client not initialized. Skipping reranking.")
            return retrieved_items[:top_k or Config.ADVANCED_RERANK_TOP_K]

        top_k = top_k or Config.ADVANCED_RERANK_TOP_K

        # Prepare documents for reranking (use windows for reranking)
        documents = [item["window"] for item in retrieved_items]

        try:
            # Rerank using Cohere
            rerank_results = self.cohere_client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-english-v3.0"
            )

            # Reorder items based on reranking scores
            reranked_items = []
            for result in rerank_results.results:
                original_item = retrieved_items[result.index].copy()
                original_item["rerank_score"] = result.relevance_score
                reranked_items.append(original_item)

            return reranked_items

        except Exception as e:
            print(f"Warning: Reranking failed: {e}. Falling back to original ranking.")
            return retrieved_items[:top_k]

    def generate_response(self, query: str, context_items: List[Dict]) -> str:
        """
        Generate response using reranked context windows.

        Args:
            query: User query
            context_items: Reranked context items

        Returns:
            Generated response
        """
        # Build context using windows (which contain surrounding context)
        context = "\n\n---\n\n".join([
            f"Source: {item['metadata']['source']}\nRelevance Score: {item.get('rerank_score', 'N/A'):.3f}\n\n{item['window']}"
            for item in context_items
        ])

        # Create prompt
        prompt = f"""You are a financial analyst assistant. Answer the user's question based on the provided context.

Context:
{context}

User Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite specific sources when referencing information
- If you notice any discrepancies or nuances between sources, explain them
- Be precise with numbers and facts
- If the context doesn't contain enough information, say so

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
        Execute full advanced RAG pipeline: retrieve, rerank, and generate.

        Args:
            question: User question

        Returns:
            Dictionary with response and metadata
        """
        trace = None

        # Start LangFuse trace if enabled
        if self.langfuse_enabled and self.langfuse:
            trace = self.langfuse.trace(
                name="advanced_rag_query",
                metadata={"agent_type": "advanced"}
            )

        # Retrieve relevant sentences
        retrieved_items = self.retrieve(question)

        if trace:
            trace.span(
                name="retrieval",
                metadata={
                    "num_items": len(retrieved_items),
                    "top_k": Config.ADVANCED_TOP_K
                }
            )

        # Rerank for better relevance
        reranked_items = self.rerank(question, retrieved_items)

        if trace:
            trace.span(
                name="reranking",
                metadata={
                    "num_items": len(reranked_items),
                    "rerank_top_k": Config.ADVANCED_RERANK_TOP_K
                }
            )

        # Generate response
        response = self.generate_response(question, reranked_items)

        if trace:
            trace.span(
                name="generation",
                input=question,
                output=response
            )

        return {
            "response": response,
            "retrieved_items": reranked_items,
            "num_items": len(reranked_items),
            "agent_type": "advanced"
        }
