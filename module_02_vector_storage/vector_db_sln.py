"""Reference implementations for the vector storage tutorial module."""

import os
import re
from datetime import datetime
from typing import Any, TypedDict

import chromadb
import numpy as np
import ollama

# Ensure telemetry is disabled via env var if not already set by setup_env.sh
os.environ["ANONYMIZED_TELEMETRY"] = "False"

EMBED_MODEL = "embeddinggemma"
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", -1))


class DocumentMetadata(TypedDict):
    """Metadata attached to a source astronomy document."""

    date: str
    category: str


class ChunkMetadata(DocumentMetadata):
    """Metadata attached to a stored text chunk."""

    date_timestamp: float
    source: str


class DocumentRecord(TypedDict):
    """Raw document content plus document-level metadata."""

    source: str
    text: str
    metadata: DocumentMetadata


class ChunkRecord(TypedDict):
    """Chunk content paired with the metadata stored in ChromaDB."""

    id: str
    text: str
    metadata: ChunkMetadata


def semantic_chunk_text(text: str, threshold: float = 0.6) -> list[str]:
    """Split text when adjacent sentence embeddings become dissimilar."""
    # Split by whitespace right after sentence-ending punctuation, keeping punctuation attached
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()
    ]
    if not sentences:
        return []

    # Get embeddings for all sentences
    raw_embeddings = np.array(
        ollama.embed(
            model=EMBED_MODEL,
            input=sentences,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )["embeddings"]
    )

    # Normalize vectors upfront so dot product directly equals cosine similarity
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    embeddings = raw_embeddings / np.maximum(norms, 1e-12)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity_score = np.dot(embeddings[i], embeddings[i - 1])

        if similarity_score < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_document_records() -> list[DocumentRecord]:
    """Load the source documents and attach tutorial metadata."""
    data_dir = "dataset/astronomy_docs"
    document_records: list[DocumentRecord] = []
    metadata_map: dict[str, DocumentMetadata] = {
        "black_holes.md": {"date": "2023-01-01", "category": "Astrophysics"},
        "dark_energy.md": {"date": "2023-06-15", "category": "Cosmology"},
        "exoplanets.md": {"date": "2024-02-10", "category": "Planetary Science"},
        "galaxy_clusters.md": {"date": "2024-05-20", "category": "Cosmology"},
        "nebulae.md": {"date": "2024-11-05", "category": "Astrophysics"},
    }

    for file in os.listdir(data_dir):
        if file.endswith(".md"):
            with open(os.path.join(data_dir, file), "r") as f:
                document_records.append(
                    {
                        "source": file,
                        "text": f.read(),
                        "metadata": metadata_map.get(
                            file, {"date": "2000-01-01", "category": "Unknown"}
                        ),
                    }
                )
    return document_records


def date_to_timestamp(date_str: str) -> float:
    """Converts an ISO date string ('YYYY-MM-DD') to a Unix timestamp."""
    if not date_str:
        return 0.0
    return datetime.fromisoformat(date_str).timestamp()


def build_chunk_records(document_records):
    """
    Turn each document record into chunk records while preserving metadata
    and attaching a numeric date_timestamp for range filtering.
    """
    chunks = []
    for document in document_records:
        semantic_chunks = semantic_chunk_text(document["text"])

        # Parse ISO date string into a Unix timestamp
        raw_date = document["metadata"].get("date", "2000-01-01")
        dt_timestamp = date_to_timestamp(raw_date)

        for index, chunk_text in enumerate(semantic_chunks):
            chunks.append(
                {
                    "id": f"{document['source']}_{index}",
                    "text": chunk_text,
                    "metadata": {
                        **document["metadata"],
                        "date_timestamp": dt_timestamp,  # Numeric timestamp for ChromaDB $gte/$lte
                        "source": document["source"],
                    },
                }
            )
    return chunks


def get_chunks() -> list[ChunkRecord]:
    """Build all chunk records for the tutorial dataset."""
    return build_chunk_records(get_document_records())


def setup_vector_db() -> Any:
    """Create the ChromaDB collection and populate it with chunk embeddings."""
    # Persistent client stores data in a folder
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    # Reset collection for testing
    try:
        client.delete_collection("astronomy")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="astronomy",
        metadata={"hnsw:space": "cosine"},
    )

    chunks_info = get_chunks()
    texts = [c["text"] for c in chunks_info]
    metadatas = [c["metadata"] for c in chunks_info]
    ids = [c["id"] for c in chunks_info]

    # Generate embeddings locally via Ollama
    embeddings = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )["embeddings"]

    collection.upsert(
        documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
    )
    stored = collection.get()
    print(f"Stored {len(stored['ids'])} chunks in the 'astronomy' collection.")
    print("Sample stored IDs:", stored["ids"][:3])
    return collection


def query_db(
    collection: Any, query_text: str
) -> list[tuple[str, float, ChunkMetadata]]:
    """Query the collection and return document, similarity, and metadata tuples."""
    query_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=[query_text],
        keep_alive=OLLAMA_KEEP_ALIVE,
    )["embeddings"]
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2,
        include=["documents", "distances", "metadatas"],
    )
    matches: list[tuple[str, float, ChunkMetadata]] = []
    for document, distance, metadata in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        similarity = 1 - distance
        matches.append((document, similarity, metadata))
    return matches


if __name__ == "__main__":
    col = setup_vector_db()
    q = "What are supermassive black holes?"
    print(f"Query: {q}")
    res = query_db(col, q)
    for i, (doc, similarity, metadata) in enumerate(res):
        print(
            f"Match {i+1} | similarity={similarity:.4f} | source={metadata['source']}: {doc}\n"
        )
