"""Exercise starter code for the RAG tutorial module."""

import ollama
import chromadb

EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "gemma4:e4b"
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)
collection = client.get_collection(name="astronomy")


def retrieve_context(query: str, n: int = 2) -> str:
    """
    Retrieve the top N most relevant chunks from the vector database based on a user query.

    Args:
        query (str): The natural language question asked by the user.
        n (int): Number of document segments to retrieve. Defaults to 2.

    Returns:
        str: A single string containing all retrieved context, with each
        chunk clearly delimited and annotated with document metadata.
    """
    # Hint:
    # Request `distances` from ChromaDB and print each retrieved chunk's
    # similarity score with `1 - distance` before you build the context string.
    # YOUR CODE HERE
    return ""


def generate_answer(query: str, context: str) -> str:
    """
    Generate a response from the LLM using provided context to avoid hallucinations.

    Args:
        query (str): The original user question.
        context (str): The relevant text retrieved from the vector database.

    Returns:
        str: The generated answer produced by the local LLM.
    """
    # YOUR CODE HERE
    pass


if __name__ == "__main__":
    user_query = "What is the difference between dark energy and dark matter?"
    ctx = retrieve_context(user_query)
    ans = generate_answer(user_query, ctx)
    print(f"Q: {user_query}\nA: {ans}")
