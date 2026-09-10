# Module 3: Retrieval Augmented Generation (RAG)

In this module, we tie everything together. Instead of asking a local LLM to answer from its general knowledge (which can lead to hallucinations), we will provide it with the most relevant chunks from our vector database as "context."

## Prerequisites

- You must already have a local ChromaDB collection named `astronomy` in `./chroma_db/`.
- Ollama must be running locally.
- The Ollama models `gemma4:e4b` and `embeddinggemma` must already be pulled.

Typical preparation:

```bash
ollama pull gemma4:e4b
ollama pull embeddinggemma
python3 module_02_vector_storage/vector_db_sln.py
```

## Key Concepts
- **Context Injection**: Inserting retrieved documents into the prompt before sending it to the LLM.
- **System Prompt**: Instructions given to the LLM on how to behave (e.g., "Only use the provided context to answer").
- **Local LLMs**: Using Ollama allows us to run Gemma models completely offline once they are available locally.

## Talking To Ollama Directly

Before adding retrieval, it helps to see the basic chat call on its own. Using streaming makes the program feel more responsive because tokens appear as they are generated:

```python
import ollama

response = ollama.chat(
    model="gemma4:e4b",
    messages=[
        {
            "role": "user",
            "content": "What is the difference between dark energy and dark matter?",
        }
    ],
    stream=True,
)

for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)
print()
```

That is the core interaction: send a message, get a response.

## Why RAG Is Just Context Concatenation

RAG is not magic. At its simplest, you:

1. Retrieve relevant text.
2. Paste that text into the prompt.
3. Ask the model to answer using that text.

This example shows the pattern with a normal question, a nonsense question the model should not know, and then the same nonsense question after we inject fictional background:

```python
import ollama

MODEL = "gemma4:e4b"

def ask_ollama(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    chunks = []
    for chunk in response:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        chunks.append(content)
    print()
    return "".join(chunks)

sample_question = "What is the difference between dark energy and dark matter?"
print("Sample question:")
ask_ollama(sample_question)
print()

nonsense_question = "Why did Captain Marzipan lose the Moonkey race on Planet Crumb?"
print("Nonsense question without context:")
ask_ollama(nonsense_question)
print()

fictional_background = """
Planet Crumb holds a yearly Moonkey race across the Glass Dunes.
Captain Marzipan was the fastest rider in practice, but lost the race
because he stopped to help a fallen competitor repair a broken saddle.
The official winner was Admiral Toffee.
"""

rag_prompt = f"""
Answer the question using only the background below.
If the answer is not in the background, say you do not know.

Background:
{fictional_background}

Question: {nonsense_question}
"""

print("Nonsense question with injected background:")
ask_ollama(rag_prompt)
```

Expected behavior:
- The sample question should get a reasonable astronomy answer.
- The nonsense question without context should produce uncertainty, a guess, or a refusal.
- The nonsense question with injected background should now be answerable because the needed facts were pasted into the prompt.

That is the essential RAG idea. In a real pipeline, your vector database simply chooses the background text for you.

## Exercises
1. `rag_ex.py`: Create a function that takes a user query, retrieves chunks from ChromaDB, and constructs a prompt for the Ollama API.

Suggested command:

```bash
python3 module_03_retrieval_augmented_generation/rag_ex.py
```

## Solutions
- See `rag_sln.py` for a full implementation of a local knowledge assistant.
- See `rag_advanced_sln.py` for advanced techniques including **Metadata filtering** (date ranges) and **Full Document Retrieval**.

Suggested commands:

```bash
python3 module_03_retrieval_augmented_generation/rag_sln.py
python3 module_03_retrieval_augmented_generation/rag_advanced_sln.py
```

Expected outcome:
- `rag_sln.py` should print a short answer grounded in retrieved astronomy snippets.
- `rag_advanced_sln.py` should demonstrate both filtered retrieval and full-document retrieval.
- In both scripts, the answer should stream to the terminal incrementally instead of appearing only at the end.

## Production Challenges To Keep In Mind

Even when a basic RAG pipeline is working, some question types still remain difficult in production:

- **Date reasoning**: retrieving documents from the right time window is not the same as correctly understanding phrases like "before 2020," "between March and June," or "the most recent report."
- **Counting and aggregation**: models often struggle when the task requires counting events, summing values, or comparing totals across multiple retrieved chunks.
- **Cross-document synthesis**: the answer may require combining evidence from several documents rather than quoting a single chunk.
- **Conflicting sources**: production corpora often contain duplicates, outdated facts, or documents that disagree with one another.
- **Entity resolution**: the system may need to recognize that two names, abbreviations, or references point to the same person, organization, or event.
- **Metadata quality**: filtering only works well when dates, authors, document types, and other metadata are consistently captured.
- **Chunking tradeoffs**: chunks that are too small can lose context, while chunks that are too large can dilute retrieval quality.
- **Citation and traceability**: users may need to see exactly which source supported each claim, not just a fluent answer.

These are common reasons production RAG systems add extra steps such as metadata normalization, reranking, structured filtering, tool use, or post-retrieval reasoning instead of relying on similarity search alone.

If this module fails:
- Check that Ollama is running on `http://localhost:11434`.
- Check that `gemma4:e4b` and `embeddinggemma` exist locally with `ollama list`.
- Check that Module 2 already created `./chroma_db/`.
