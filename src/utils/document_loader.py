"""
Document loading and processing utilities.
"""
import os
from typing import List, Dict
from pathlib import Path


def load_documents(data_dir: str = "data") -> List[Dict[str, str]]:
    """
    Load all text documents from the data directory.

    Args:
        data_dir: Directory containing the documents

    Returns:
        List of dictionaries with 'content' and 'source' keys
    """
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_path in data_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append({
                "content": content,
                "source": file_path.name,
                "path": str(file_path)
            })

    return documents


def basic_text_splitter(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Simple text splitter that splits on paragraph boundaries.
    This is the 'naive' approach that will cause issues.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    # Split on double newlines (paragraphs)
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def sentence_splitter(text: str) -> List[Dict[str, any]]:
    """
    Split text into sentences while preserving context windows.
    This is the advanced approach for sentence-window retrieval.

    Args:
        text: Text to split

    Returns:
        List of dictionaries with sentence, context window, and metadata
    """
    # Simple sentence splitting (in production, use spaCy or NLTK)
    sentences = []
    current_sentence = ""

    for char in text:
        current_sentence += char
        if char in ".!?" and len(current_sentence) > 20:
            sentences.append(current_sentence.strip())
            current_sentence = ""

    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # Create sentence windows
    window_size = 3  # sentences before and after
    sentence_windows = []

    for i, sentence in enumerate(sentences):
        start_idx = max(0, i - window_size)
        end_idx = min(len(sentences), i + window_size + 1)

        window = " ".join(sentences[start_idx:end_idx])

        sentence_windows.append({
            "sentence": sentence,
            "window": window,
            "position": i,
            "total_sentences": len(sentences)
        })

    return sentence_windows
