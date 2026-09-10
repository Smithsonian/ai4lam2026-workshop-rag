"""Streamlit interface for the astronomy RAG tutorial application."""

from collections.abc import Generator
import os
from pathlib import Path
from typing import Any

import chromadb
import ollama
import streamlit as st

EMBED_MODEL = "embeddinggemma"
DEFAULT_CHAT_MODEL = "gemma4:e4b"
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", -1))
REPO_ROOT = Path(__file__).resolve().parent.parent
STREAMLIT_CONFIG_PATH = REPO_ROOT / ".streamlit" / "config.toml"


def ensure_local_privacy_defaults() -> None:
    """Create the local Streamlit config if it does not already exist."""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    if STREAMLIT_CONFIG_PATH.exists():
        return

    STREAMLIT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    STREAMLIT_CONFIG_PATH.write_text("[browser]\ngatherUsageStats = false\n")


ensure_local_privacy_defaults()

# ==========================================
# BACKEND: RAG PIPELINE
# ==========================================


@st.cache_resource
def load_resources() -> Any:
    """Load the ChromaDB client once per session."""
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    # Note: In a real scenario, ensure this path matches where module_02 saved the DB
    try:
        collection = client.get_collection(name="astronomy")
    except Exception as e:
        st.error(
            "Database not found! Please run the previous modules to populate './chroma_db'."
        )
        st.stop()
    return collection


def retrieve_context(
    query: str, collection: Any, n: int = 3
) -> list[dict[str, Any]]:
    """Retrieve top N relevant documents and metadata from the vector store."""
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

    matches = []
    for document, distance, metadata in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        matches.append(
            {
                "document": document,
                "distance": distance,
                "similarity": 1 - distance,
                "metadata": metadata or {},
            }
        )

    return matches


def render_retrieved_sources(context_matches: list[dict[str, Any]]) -> None:
    """Render retrieved source snippets with source metadata delimiters."""
    for i, match in enumerate(context_matches, start=1):
        metadata = match["metadata"]
        source_name = metadata.get("source", "Unknown source")
        st.markdown(
            "\n".join(
                [
                    f"**Match {i}**",
                    f"`source: {source_name}`",
                    f"`distance: {match['distance']:.4f}`",
                    f"`similarity: {match['similarity']:.4f}`",
                ]
            )
        )
        st.write(match["document"])
        st.divider()


def generate_answer(
    query: str,
    context_list: list[dict[str, Any]],
    model_name: str = DEFAULT_CHAT_MODEL,
) -> Generator[str, None, None]:
    """Stream an answer using a LLM and the retrieved context."""
    context_text = "\n\n".join(
        [
            "\n".join(
                [
                    f"Source: {match['metadata'].get('source', 'Unknown source')}",
                    f"Distance: {match['distance']:.4f}",
                    f"Similarity: {match['similarity']:.4f}",
                    match["document"],
                ]
            )
            for match in context_list
        ]
    )
    prompt = f"""You are a professional Astronomy AI Assistant. 
Use ONLY the following pieces of retrieved context to answer the question. 
If you don't know the answer based on the context, just say that you don't know.

Context:
{context_text}

Question: {query}
Answer:"""

    response_stream = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=True,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    for chunk in response_stream:
        content = chunk["message"]["content"]
        if content:
            yield content


# ==========================================
# FRONTEND: STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Astronomy RAG Bot", page_icon="🔭")

st.title("🔭 Astronomy Knowledge Assistant")
st.markdown("Ask any question about space based on the provided astronomy documents.")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    llm_model = st.text_input("LLM Model", value=DEFAULT_CHAT_MODEL)
    num_docs = st.slider(
        "Number of documents to retrieve", min_value=1, max_value=5, value=3
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Load resources (DB)
collection = load_resources()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:  # Only display context for assistant responses
            with st.expander("View retrieved sources"):
                render_retrieved_sources(message["context"])

# Chat input
if prompt := st.chat_input(
    "What is the difference between dark energy and dark matter?"
):
    # 1. Add user message to history & display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            try:
                # Retrieval phase
                context_docs = retrieve_context(prompt, collection, n=num_docs)

                # Generation phase
                answer = st.write_stream(
                    generate_answer(prompt, context_docs, model_name=llm_model)
                )

                # Display sources in an expander
                with st.expander("View retrieved sources"):
                    render_retrieved_sources(context_docs)

                # Store response and context in history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "context": context_docs}
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
