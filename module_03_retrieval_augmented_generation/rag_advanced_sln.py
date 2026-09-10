"""Advanced reference implementations for the RAG tutorial module."""

import os
from datetime import datetime
from typing import Any, Callable

import chromadb
import ollama

EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "gemma4:e4b"
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", -1))

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)
collection = client.get_collection(name="astronomy")


def date_to_timestamp(date_str: str, end_of_day: bool = False) -> int:
    """Parses ISO date string ('YYYY-MM-DD') into a Unix timestamp integer."""
    if not date_str:
        return 0

    dt = datetime.fromisoformat(date_str)

    # If end_date is given as '2024-12-31' without time, ensure it covers the entire day
    if end_of_day and len(date_str) == 10:
        dt = dt.replace(hour=23, minute=59, second=59)

    return int(dt.timestamp())


def retrieve_context_with_filter(
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    n: int = 2,
) -> dict[str, Any]:
    """
    Retrieve context with an optional date range filter using ChromaDB's metadata filtering.

    NOTE:
    ChromaDB does not support string-based comparison operators like $gte or $lte.
    Passing "2024-01-01" as a string to $gte will trigger a ValueError.
    Therefore, metadata in ChromaDB should store years as integers (e.g., {"year": 2024})
    or dates as Unix timestamps (e.g., 1704067200). Here, we filter using unix timestamps.
    """
    query_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=[query],
        keep_alive=OLLAMA_KEEP_ALIVE,
    )["embeddings"]

    # Convert YYYY-MM-DD dates to Unix timestamps
    start_ts = date_to_timestamp(start_date) if start_date else None
    end_ts = date_to_timestamp(end_date, end_of_day=True) if end_date else None

    # Construct ChromaDB metadata filter using numeric comparison on the 'date_timestamp' field
    where_conditions = []
    if start_ts is not None:
        where_conditions.append({"date_timestamp": {"$gte": start_ts}})
    if end_ts is not None:
        where_conditions.append({"date_timestamp": {"$lte": end_ts}})

    # Format 'where' filter syntax according to ChromaDB specifications
    if len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}
    elif len(where_conditions) == 1:
        where_filter = where_conditions[0]
    else:
        where_filter = None  # None indicates no filter applied

    return collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        where=where_filter,
        include=["documents", "distances", "metadatas"],
    )


def retrieve_full_documents(query: str, n: int = 2) -> str:
    """
    Retrieves top snippet matches to identify relevant source files,
    then loads the full text of those files from disk to prevent context truncation.
    """
    query_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=[query],
        keep_alive=OLLAMA_KEEP_ALIVE,
    )["embeddings"]
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        include=["metadatas", "distances"],
    )

    # Collect unique source filenames from metadata
    sources = {meta["source"] for meta in results["metadatas"][0]}

    full_context: list[str] = []
    data_dir = "dataset/astronomy_docs"
    for source in sources:
        file_path = os.path.join(data_dir, source)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                full_context.append(f"--- Source: {source} ---\n{f.read()}")

    return "\n\n".join(full_context)


def generate_answer(query: str, context: str) -> str:
    """
    Generates an answer using Ollama streaming based strictly on retrieved context.
    """
    prompt = f"""You are a helpful astronomy assistant. Use ONLY the following pieces of retrieved context to answer the question. If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {query}
Answer:"""

    response_stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
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


def print_matches(
    label: str,
    distances: list[float],
    metadatas: list[dict[str, Any]],
    documents: list[str] | None = None,
) -> None:
    """Print retrieval matches with similarity scores and optional content."""
    for i, (distance, metadata) in enumerate(zip(distances, metadatas), start=1):
        print(
            f"{label} {i} | similarity={1 - distance:.4f} | source={metadata['source']}"
        )
        if documents is not None:
            print(documents[i - 1])
            print()


def run_scenario(
    title: str,
    query: str,
    context: str,
    preamble: str | None = None,
    matches: Callable[[], None] | None = None,
) -> None:
    """Print a scenario header, optional retrieval details, and a generated answer."""
    print(title)
    print(query)
    if preamble:
        print(preamble)
    if matches:
        matches()
    print("Answer:")
    generate_answer(query, context)
    print()


def main() -> None:
    """Run the advanced retrieval examples used in the tutorial."""
    # Scenario 1: Metadata Filtering (e.g., only docs from 2024)
    query_1 = "What can you tell me about exoplanets?"
    res_filtered = retrieve_context_with_filter(
        query_1, start_date="2024-01-01", end_date="2024-12-31"
    )

    if res_filtered["documents"] and res_filtered["documents"][0]:
        filtered_documents = res_filtered["documents"][0]
        ctx_filtered = "\n\n".join(filtered_documents)
        run_scenario(
            "--- Scenario 1: Filtered Retrieval (Docs from 2024) ---",
            query_1,
            ctx_filtered,
            f"Context retrieved from {len(filtered_documents)} snippets.",
            matches=lambda: print_matches(
                "Snippet",
                res_filtered["distances"][0],
                res_filtered["metadatas"][0],
                filtered_documents,
            ),
        )
    else:
        print("--- Scenario 1: Filtered Retrieval (Docs from 2024) ---")
        print(query_1)
        print("No documents found matching the specified filters.\n")

    query_1 = "What can you tell me about exoplanets?"
    res_filtered = retrieve_context_with_filter(
        query_1, start_date="2023-01-01", end_date="2023-12-31"
    )

    if res_filtered["documents"] and res_filtered["documents"][0]:
        filtered_documents = res_filtered["documents"][0]
        ctx_filtered = "\n\n".join(filtered_documents)
        run_scenario(
            "--- Scenario 1: Filtered Retrieval (Docs from 2023) ---",
            query_1,
            ctx_filtered,
            f"Context retrieved from {len(filtered_documents)} snippets.",
            matches=lambda: print_matches(
                "Snippet",
                res_filtered["distances"][0],
                res_filtered["metadatas"][0],
                filtered_documents,
            ),
        )
    else:
        print("--- Scenario 1: Filtered Retrieval (Docs from 2023) ---")
        print(query_1)
        print("No documents found matching the specified filters.\n")

    # Scenario 2: Full Document Retrieval
    query_2 = "Where are galaxy clusters found?"
    ctx_full = retrieve_full_documents(query_2)
    top_matches = collection.query(
        query_embeddings=ollama.embed(
            model=EMBED_MODEL,
            input=[query_2],
            keep_alive=OLLAMA_KEEP_ALIVE,
        )["embeddings"],
        n_results=2,
        include=["metadatas", "distances"],
    )

    run_scenario(
        "--- Scenario 2: Full Document Retrieval ---",
        query_2,
        ctx_full,
        "Retrieved full documents for the most relevant snippets.",
        matches=lambda: print_matches(
            "Full-doc seed match",
            top_matches["distances"][0],
            top_matches["metadatas"][0],
        ),
    )

    # NOTE FOR STUDENTS:
    # A further improvement would be to implement a 'Reranker'.
    # Instead of just taking the top-N from ChromaDB, you could retrieve top-20,
    # and use a Cross-Encoder model to rerank them based on actual relevance
    # before passing only the top-5 to the LLM.


if __name__ == "__main__":
    main()
