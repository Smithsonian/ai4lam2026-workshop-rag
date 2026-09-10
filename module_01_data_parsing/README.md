# Module 1: Data Parsing and Chunking

In this module, you will learn how to prepare raw text data for a RAG pipeline. Since LLMs have limited context windows, we cannot pass thousands of pages at once. We must break the documents into smaller "chunks."

## Prerequisites

Before starting this module, generate the sample dataset:

```bash
python3 scripts/generate_astronomy_data.py
```

## Key Concepts
- **Chunking**: Dividing a document into smaller segments.
- **Overlap**: Including some text from the previous chunk in the next one to maintain semantic continuity.
- **Granularity**: The size of the chunks affects what the retriever finds. Small chunks = precise but potentially missing context; Large chunks = more context but noise.

## Exercises
1. `parsing_ex.py`: Implement a fixed-size window chunking algorithm with overlap.
2. Try running it on the astronomy dataset in `dataset/astronomy_docs`.

Suggested command:

```bash
python3 module_01_data_parsing/parsing_ex.py
```

Expected outcome:
- The script should print each Markdown filename and the number of generated chunks.
- With the provided sample data and `chunk_size=100`, each file should produce multiple chunks rather than a single full-document output.

## Solutions
See `parsing_sln.py` for an improved implementation of fixed-size chunking that avoids splitting in the middle of words.

Suggested command:

```bash
python3 module_01_data_parsing/parsing_sln.py
```

What to look for:
- The fixed-size chunker may break in the middle of words
- The solution chunker should respect word boundaries.

Things to think about:
- All the different punctuation types
- Structured text such as tables and headings for tables
