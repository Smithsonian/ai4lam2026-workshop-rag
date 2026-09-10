# Module 2: Local Embeddings and Vector Storage

In this module, we move from raw text chunks to lists of numbers (vectors) called "embeddings." Embeddings allow us to perform similarity searches using vector mathematics (e.g., cosine similarity).

This module also revisits chunking from Module 1. We can use embeddings for an advanced chunking strategy: grouping sentences by semantic similarity instead of only by fixed character windows.

## Prerequisites

- The dataset in `dataset/astronomy_docs/` must already exist.
- Ollama must be running locally.
- The embedding model `embeddinggemma` must be available locally.

If you are on a fresh machine:

```bash
python3 scripts/generate_astronomy_data.py
ollama pull embeddinggemma
```

## Key Concepts
- **Embeddings**: A list of numbers (vector) that captures the semantic meaning of a piece of text.
- **Vector Database**: A specialized database like ChromaDB that stores these vectors and allows for fast "Nearest Neighbor" queries.
- **Embedding Models**: A large language model that takes text in and outputs embeddings representing the text instead of new text.

### First ChromaDB Interaction

Before wiring ChromaDB into the full exercise, it helps to see the smallest useful loop:
- create a database client
- create a database collection
- insert one or two records
- inspect what is stored in the database

Example:

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db", settings=chromadb.config.Settings(anonymized_telemetry=False))

# Reset collection
try:
    client.delete_collection("astronomy")
except Exception:
    pass  # Ignores error if collection doesn't exist yet

# hnsw:space is 'hierarchical navigable small world', an algorithm for nearest neighbor searching.
# cosine is cosine distance(a,b) = 1 - (a \dot b)/(|a| * |b|) = [0, 2] 
collection = client.get_or_create_collection(
    name="astronomy",
    metadata={"hnsw:space": "cosine"},
)

collection.upsert(
    ids=["demo-1", "demo-2"],
    documents=[
        "Black holes are regions where gravity is extremely strong.",
        "Exoplanets are planets that orbit stars beyond our solar system.",
    ],
    embeddings=[
        [0.1, 0.2, 0.3],
        [0.4, 0.2, 0.9],
    ],
)

stored = collection.get()
print("IDs:", stored["ids"])
print("Documents:", stored["documents"])

results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3]],
    n_results=2,
    include=["documents", "distances"],
)

for doc, distance in zip(results["documents"][0], results["distances"][0]):
    similarity = 1 - distance
    print(f"similarity={similarity:.4f} | {doc}")
```

> ⚠️ **Note on Vector Dimensions:**
> ChromaDB locks a collection's dimension to the first vector inserted. If you run the 3D demo code (`[0.1, 0.2, 0.3]`) and later try inserting 768D embeddings into the same `astronomy` collection, it will raise an `InvalidArgumentError`. 
>
> **Fix:** Delete the local database folder (`rm -rf ./chroma_db`) or use `client.delete_collection("astronomy")` before switching embedding models.

In the actual exercise, embeddings come from `ollama.embed(...)` rather than a hard-coded list.
When querying ChromaDB, ask for `include=["documents", "distances"]` to inspect how close each match is to the query vector.

## Handling Document Updates & Stale Data

When re-indexing documents or crawling web pages without wiping the database, standard updates can leave orphan chunks in ChromaDB if a document shrinks or gets deleted. Follow these strategies to keep your vector database clean:

### 1. Document-Level Cleanups (Local Files)
When updating a known file or document, delete existing chunks for that specific `source` metadata before performing an `upsert`:

```python
# Remove old chunks associated with this file before upserting new ones
collection.delete(where={"source": filename})

# Proceed with chunking and upserting the updated document
collection.upsert(ids=new_ids, documents=new_texts, ...)
```

### What An Embedding Looks Like

An embedding is literally a vector of numbers. For example, a model might turn a short sentence into something conceptually like:

```text
"Black holes bend spacetime"
-> [0.12, -0.44, 0.08, 0.91, -0.07, ...]
```

The real vector is much longer (e.g. 256, 512, 1024 numbers) than this toy example, but the idea is the same:
- each number is one dimension in the vector
- the full vector represents the meaning of the text
- similar text tends to produce vectors that are close together in vector space

That is why a vector database can answer a semantic question like "Tell me about black holes" even if the stored chunk does not use exactly the same wording.

### Building On Module 1

Module 1 introduced fixed-size chunking, cutting every 150 characters, because it is easy to understand and implement.

In this module, we can introduce **semantic chunking**, grouping nearby sentences that mean similar things. 
Semantic chunking uses embeddings to compare sentences as vectors and decide chunk boundaries based on topic changes.

## Exercises
1. `vector_db_ex.py`: Implement `semantic_chunk_text()` so you can chunk documents using embeddings instead of only fixed windows.
2. Extend the same file to:
   - initialize a local ChromaDB client
   - create or get an `astronomy` collection
   - insert chunk text plus embeddings
   - inspect the collection contents with `collection.get()`
   - perform a search for `"What is a black hole?"`
3. When you query, print the similarity score alongside each retrieved chunk so you can compare the ranking.

### Suggested Staging

To keep the exercise focused, it helps to build Module 2 in two passes:

1. First pass: keep `get_chunks()` simple and return only chunk strings.
   This isolates semantic chunking, embedding generation, and basic ChromaDB usage.
2. Second pass: promote each chunk into a richer record with an `id`, `text`,
   and `metadata` dictionary.
   This is the pattern used in `vector_db_sln.py`, and it is what later makes
   metadata-aware retrieval and filtering possible in Module 3.

That second pass matters because metadata is attached to the document before or
while chunk records are created, not after retrieval. Even if `get_chunks()`
loops over all files and returns one list at the end, each item in that list can
still be a structured record that keeps the chunk text and its source metadata together.

Suggested command:

```bash
python3 module_02_vector_storage/vector_db_ex.py
```

## Solutions
See `vector_db_sln.py` for a complete implementation including **semantic chunking**, persistent storage, and **Metadata enrichment** (adding dates and categories to documents).

The solution now separates:
- loading full documents plus document-level metadata
- converting those documents into chunk records
- inserting chunk text, embeddings, and metadata into ChromaDB

That split is intentional for workshop readability: it makes the metadata handoff visible instead of hiding it inside one larger helper.

Suggested command:

```bash
python3 module_02_vector_storage/vector_db_sln.py
```

Expected outcome:
- A `./chroma_db/` directory should be created locally.
- You should be able to inspect stored IDs and documents before running retrieval.
- The stored chunks should follow sentence/topic boundaries more naturally than raw fixed-size slices.
- The sample query should return text mentioning supermassive black holes or black holes in general, with visible similarity scores.

Why this matters for later modules:
- Module 3 and Module 5 both assume the `astronomy` collection already exists in `./chroma_db/`.
