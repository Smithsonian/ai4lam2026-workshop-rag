"""Reference implementations for the RAG tutorial module."""

import os
from typing import Any

import ollama
import chromadb

EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "gemma4:e4b"
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", -1))
client = chromadb.PersistentClient(
    path="./chroma_db", settings=chromadb.config.Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name="astronomy")


def format_chunk(index: int, document: str, metadata: dict[str, Any]) -> str:
    """Format one retrieved chunk and its metadata for prompt injection."""
    return "\n".join(
        [
            f"--- Retrieved Chunk {index} ---",
            f"Source: {metadata['source']}",
            f"Category: {metadata['category']}",
            f"Date: {metadata['date']}",
            "Content:",
            document,
        ]
    )


def retrieve_context(query: str, n: int = 2) -> str:
    """Retrieve the most relevant chunks and join them into a prompt context."""
    query_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=[query],
        keep_alive=OLLAMA_KEEP_ALIVE,
    )["embeddings"]
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        include=["documents", "distances", "metadatas"],
    )
    formatted_chunks = []
    print("\nRetrieved chunks that will be sent to the model:\n")
    for i, (document, distance, metadata) in enumerate(
        zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ),
        start=1,
    ):
        similarity = 1 - distance
        formatted_chunk = format_chunk(i, document, metadata)
        print(f"Chunk {i} | similarity={similarity:.4f} | source={metadata['source']}")
        print(formatted_chunk)
        print()
        formatted_chunks.append(formatted_chunk)
    return "\n\n".join(formatted_chunks)


def generate_answer(query: str, context: str) -> str:
    """Generate an answer using only the retrieved context."""
    prompt = f"""You are a helpful astronomy assistant. Use ONLY the following pieces of retrieved context to answer the question. Pay attention to both the chunk content and the source metadata. If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {query}
Answer:"""

    response_stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    chunks = []
    for chunk in response_stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        chunks.append(content)
    print()
    return "".join(chunks)


if __name__ == "__main__":
    user_query = "What is the difference between dark energy and dark matter?"
    print(f"Question: {user_query}")
    print("\nRetrieving context...")
    ctx = retrieve_context(user_query)
    print("Generating answer via Ollama...\n")
    print("Answer:")
    generate_answer(user_query, ctx)
