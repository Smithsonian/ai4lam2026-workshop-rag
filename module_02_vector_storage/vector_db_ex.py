"""Exercise starter code for the vector storage tutorial module."""

import os
import chromadb
import ollama

EMBED_MODEL = "embeddinggemma"


def semantic_chunk_text(text: str, threshold: float = 0.6) -> list[str]:
    """
    Split text into chunks by comparing the embedding similarity of adjacent sentences.

    Args:
        text (str): The input document text.
        threshold (float): If similarity drops below this value, start a new chunk.

    Returns:
        list[str]: A list of semantically grouped text chunks.
    """
    # YOUR CODE HERE
    pass


def get_chunks() -> list[str]:
    """
    Helper to load semantically chunked text from the astronomy dataset directory.

    Returns:
        list[str]: A list of strings representing different parts of the documents.

    Hint:
        Start with a flat list of chunk strings so you can focus on chunking and
        vector storage first.

        If you want to mirror `vector_db_sln.py` more closely, the next step is
        to stop returning bare strings and instead return records such as:

        {
            "id": "black_holes.md_0",
            "text": "...chunk text...",
            "metadata": {"source": "black_holes.md", "date": "...", "category": "..."},
        }

        That makes it obvious how chunk text stays paired with document metadata
        when you later call `collection.upsert(..., metadatas=...)`.
    """
    data_dir = "dataset/astronomy_docs"
    chunks = []
    for file in os.listdir(data_dir):
        if file.endswith(".md"):
            with open(os.path.join(data_dir, file), "r") as f:
                text = f.read()
                chunks.extend(semantic_chunk_text(text))
    return chunks


def setup_vector_db() -> None:
    """
    Initialize the vector database and populate it with embeddings from local text chunks.

    This function should perform the following steps:
    1. Initialize a ChromaDB client (ephemeral or persistent).
    2. Create or get a collection named 'astronomy'.
    3. Transform the semantically chunked texts into vectors using `ollama.embed()`.
    4. Insert the documents and their corresponding embeddings into the database.
    5. Inspect the collection so you can confirm what was stored.

    Hint:
        For the base exercise, `get_chunks()` returns only strings, so a minimal
        solution can upsert just `ids`, `documents`, and `embeddings`.

        As an optional stretch step, promote each chunk into a richer record with
        `text`, `id`, and `metadata` fields. Then you can also pass
        `metadatas=[record["metadata"] for record in records]` to ChromaDB.

    Returns:
        None
    """
    # Hint:
    # client = chromadb.PersistentClient(path="./chroma_db", settings=chromadb.config.Settings(anonymized_telemetry=False))
    # collection = client.get_or_create_collection(
    #     name="astronomy",
    #     metadata={"hnsw:space": "cosine"},
    # )
    # stored = collection.get()
    # print(stored["ids"])
    # print(stored["documents"])
    # YOUR CODE HERE
    pass


def query_db(query_text: str) -> list[str]:
    """
    Search the vector database for the most semantically similar chunks to a given query.

    Args:
        query_text (str): The search query string.

    Returns:
        list[str]: A list of the top matching document segments found in the DB.
    """
    # Hint:
    # Ask ChromaDB to include distances, then convert cosine distance to
    # similarity with `1 - distance` before printing each match.
    #
    # Optional stretch:
    # If you stored metadata in `setup_vector_db()`, also ask for
    # `include=["documents", "distances", "metadatas"]` so you can print the
    # source file, date, or category next to each retrieved chunk.
    # YOUR CODE HERE
    pass


if __name__ == "__main__":
    setup_vector_db()
    query_db("Tell me about black holes")
