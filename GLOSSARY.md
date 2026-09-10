# RAG Glossary

This glossary collects the key terms used across the tutorial `README.md` files and explains why each term matters in a Retrieval-Augmented Generation (RAG) system.

## A

### Answer Grounding
**Definition:** The practice of making sure a generated answer is supported by retrieved source material rather than unsupported model guesses.

**Why it matters for RAG:** Grounding is one of the main reasons to use RAG at all. It reduces hallucinations and makes answers easier to trust and inspect.

## C

### ChromaDB
**Definition:** A vector database used in this tutorial to store embeddings, documents, and metadata for later retrieval.

**Why it matters for RAG:** RAG needs a place to store vectorized document chunks and search them efficiently when a user asks a question.

### Chunk
**Definition:** A smaller segment of a larger document created so the system can index and retrieve manageable pieces of text.

**Why it matters for RAG:** Retrieval usually works better on focused passages than on entire documents. Chunks are the units the retriever can actually search and return.

### Chunking
**Definition:** The process of dividing a document into smaller segments before embedding and storage.

**Why it matters for RAG:** Chunking directly affects retrieval quality. Poor chunking can hide relevant facts, split useful context, or retrieve noisy passages.

### Context
**Definition:** The retrieved text passed into the model prompt alongside the user’s question.

**Why it matters for RAG:** Context is the evidence the model is supposed to use when answering. Without useful context, the system falls back to general model knowledge.

### Context Injection
**Definition:** Inserting retrieved documents or passages into the prompt before sending it to the language model.

**Why it matters for RAG:** This is the core mechanism of RAG. Retrieval only helps if the model actually receives the retrieved text in its prompt.

### Cosine Similarity / Cosine Distance
**Definition:** A mathematical way to compare how close two vectors are in direction. This tutorial uses cosine distance in ChromaDB and interprets nearer vectors as more similar.

**Why it matters for RAG:** Similarity scoring is what lets the system find chunks that are semantically related to a query even when the wording differs.

## E

### Embedding
**Definition:** A vector of numbers representing the semantic meaning of a piece of text.

**Why it matters for RAG:** Embeddings are what make semantic retrieval possible. They turn text into something the vector database can compare numerically.

### Embedding Model
**Definition:** A model that takes text as input and returns embeddings instead of generated text.

**Why it matters for RAG:** The quality of retrieval depends heavily on the quality and consistency of the embeddings produced for documents and queries.

### Evaluation
**Definition:** The process of checking whether a RAG system is producing useful, relevant, and supported answers.

**Why it matters for RAG:** A pipeline that runs is not necessarily a pipeline that works. Evaluation is how you detect weak retrieval, hallucinations, and prompt issues.

## F

### Faithfulness
**Definition:** A measure of whether the answer is actually supported by the retrieved context.

**Why it matters for RAG:** RAG is supposed to answer from retrieved evidence. If faithfulness is low, the system is not reliably using its sources.

### Full-Document Retrieval
**Definition:** A retrieval strategy that returns or reconstructs a whole document rather than only isolated chunks.

**Why it matters for RAG:** Some questions need broader context than a small chunk provides. Full-document retrieval can help when local passages alone are too narrow.

## G

### Generation
**Definition:** The step where the language model produces a natural-language answer from the prompt and retrieved context.

**Why it matters for RAG:** Retrieval finds evidence, but generation turns that evidence into a usable response for the user.

### Granularity
**Definition:** The size and level of detail used for chunks.

**Why it matters for RAG:** Granularity controls the tradeoff between precision and context. Small chunks can be precise; larger chunks can preserve more surrounding meaning.

### Ground Truth
**Definition:** A known correct answer or reference answer used for evaluation.

**Why it matters for RAG:** You need a trusted reference point to judge whether your system is answering correctly or only sounding plausible.

## H

### Hallucination
**Definition:** A generated claim that is incorrect, unsupported, or invented by the model.

**Why it matters for RAG:** RAG is often adopted specifically to reduce hallucinations by anchoring answers in retrieved material.

## L

### Lexical Baseline
**Definition:** A simple evaluation method based on word overlap between a generated answer and a reference answer.

**Why it matters for RAG:** It provides a quick, understandable baseline before moving to richer evaluation methods, even though it is limited.

### Local LLM
**Definition:** A large language model running on the user’s own machine rather than through a remote API.

**Why it matters for RAG:** This tutorial’s pipeline is local-first. A local LLM makes the workshop more privacy-friendly and easier to run offline after setup.

## M

### Metadata
**Definition:** Structured fields attached to documents or chunks, such as source, date, or category.

**Why it matters for RAG:** Metadata enables filtering, traceability, and more targeted retrieval than similarity search alone.

### Metadata Filtering
**Definition:** Restricting retrieval results using metadata constraints such as dates, sources, or categories.

**Why it matters for RAG:** Some questions are not just about topic similarity. They also depend on time period, document type, or source constraints.

## N

### Nearest Neighbor Search
**Definition:** A method for finding the stored vectors most similar to a query vector.

**Why it matters for RAG:** This is the retrieval engine behind vector search. It is how the system selects likely relevant chunks from the corpus.

## O

### Ollama
**Definition:** The local model runtime used in this tutorial for both embeddings and chat generation.

**Why it matters for RAG:** Ollama is the interface this tutorial uses to run the embedding model and the answer-generating model locally.

### Overlap
**Definition:** Reusing some text from one chunk in the next chunk during chunking.

**Why it matters for RAG:** Overlap helps preserve continuity across boundaries so important information is less likely to be split apart.

## P

### Persistent Vector Store
**Definition:** A vector database saved to disk so its contents remain available across runs.

**Why it matters for RAG:** Persistence lets you build the index once and reuse it later instead of regenerating embeddings every time.

### Prompt Assembly
**Definition:** The construction of the final prompt from system instructions, retrieved context, and the user’s question.

**Why it matters for RAG:** Good retrieval can still fail if the final prompt is poorly structured or does not clearly tell the model how to use the context.

### Pydantic
**Definition:** A Python library for defining and validating structured data schemas.

**Why it matters for RAG:** In this tutorial, Pydantic helps make evaluation outputs structured and machine-readable instead of loose free-form text.

## Q

### Qualitative Evaluation
**Definition:** Human review of system outputs using judgment and task-specific criteria.

**Why it matters for RAG:** Not every failure shows up in a numeric metric. Human inspection catches issues like misleading phrasing or weak explanations.

### Quantitative Evaluation
**Definition:** Measurement of system performance using explicit scores or metrics.

**Why it matters for RAG:** Quantitative checks make it easier to compare versions of a pipeline and see whether changes improve results.

### Query
**Definition:** The user’s question or search input sent into the retrieval pipeline.

**Why it matters for RAG:** The query is what gets embedded and matched against stored vectors to find relevant context.

## R

### RAG (Retrieval-Augmented Generation)
**Definition:** A pattern where a system retrieves relevant source material first and then gives that material to a language model to help generate an answer.

**Why it matters for RAG:** It is the central concept of the tutorial. The full workflow exists to combine retrieval with generation so answers are more specific and better supported.

### Relevance
**Definition:** A measure of whether an answer responds to the user’s question.

**Why it matters for RAG:** An answer can be faithful to the context and still fail if it does not address what the user asked.

### Retrieval
**Definition:** The step that searches stored document representations and returns the most relevant matches for a query.

**Why it matters for RAG:** Retrieval determines which evidence the model sees. If retrieval is poor, generation quality usually drops with it.

### Reranking
**Definition:** An extra step that reorders retrieved results using a stronger or more task-specific scoring method.

**Why it matters for RAG:** Basic similarity search is often not enough in real systems. Reranking can improve which chunks are surfaced first.

## S

### Semantic Chunking
**Definition:** Chunking based on meaning or topic boundaries rather than only fixed character counts.

**Why it matters for RAG:** Semantic chunking often produces cleaner retrieval units because the text inside each chunk is more coherent.

### Semantic Similarity
**Definition:** Similarity in meaning rather than exact word overlap.

**Why it matters for RAG:** Users rarely phrase questions exactly like the source text. Semantic similarity helps the retriever match meaning across different wording.

### Similarity Retrieval
**Definition:** Retrieving documents or chunks by comparing vector similarity between the query and stored embeddings.

**Why it matters for RAG:** This is the main retrieval strategy used in the tutorial’s vector-store pipeline.

### Streamlit
**Definition:** A Python framework for building lightweight interactive web apps.

**Why it matters for RAG:** Streamlit provides a simple way to expose the RAG pipeline as a chat interface and show retrieved snippets to users.

### Structured Output
**Definition:** Returning data in a defined schema such as JSON rather than free-form prose.

**Why it matters for RAG:** Structured outputs make evaluation results easier to log, compare, filter, and visualize programmatically.

### System Prompt
**Definition:** High-level instructions that tell the model how it should behave when generating an answer.

**Why it matters for RAG:** The system prompt can tell the model to rely on retrieved context, admit uncertainty, and avoid unsupported claims.

## V

### Vector
**Definition:** An ordered list of numbers representing a document or query in embedding space.

**Why it matters for RAG:** Vectors are the numerical form that makes similarity search possible.

### Vector Database
**Definition:** A database optimized for storing vectors and running nearest-neighbor similarity queries over them.

**Why it matters for RAG:** RAG systems need fast vector search to retrieve relevant passages from a document collection.

### Vector Store
**Definition:** The stored collection of vectors, documents, and often metadata used for retrieval.

**Why it matters for RAG:** The vector store is the searchable memory of the RAG system.

## W

### Word-Overlap Baseline
**Definition:** A simple metric that compares shared words between a generated answer and a reference answer.

**Why it matters for RAG:** It offers a quick first-pass evaluation method, even though it cannot fully measure factual correctness or grounding.
