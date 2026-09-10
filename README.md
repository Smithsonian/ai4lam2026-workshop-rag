# Implementing Secure, On-Premise Retrieval-Augmented Generation (RAG) for Mission Archives and Custom Datasets

Gregory Ciccarelli  
gregory.ciccarelli@sao.si.edu  
Center for Astrophysics | Harvard & Smithsonian (CfA)  

As Large Language Models (LLMs) become standard tools for software development and data discovery, technical and scientific archives face a dual challenge: the risk of AI "hallucinations" in niche domains and the security concerns of uploading proprietary, sensitive, or pre-publication documentation to commercial cloud providers. This tutorial provides a hands-on guide to implementing a secure Retrieval-Augmented Generation (RAG) pipeline entirely on-premise. This session features a practical case study using a publicly available dataset, the architecture and techniques taught are fundamentally data-agnostic and directly transferable to any organization's localized data.

By the end of this session, attendees will walk away with a repeatable, blueprint for building a private "Knowledge Assistant" tailored to their own data sources.

https://www.conftool.org/fantastic-futures-2026/index.php?page=browseSessions&form_date=2026-09-15&form_session=102&presentations=hide

# This Repository
This repository is a hands-on introduction to building a small local Retrieval-Augmented Generation (RAG) system. It is designed as a workshop/tutorial.

You will work through a complete path:
1. Parse and chunk source documents.
2. Create local embeddings.
3. Store and query vectors with ChromaDB.
4. Send retrieved context to a local LLM through Ollama.
5. Evaluate outputs.
6. Optionally expose the pipeline in a Streamlit chat UI.

## What This Repository Covers

This tutorial currently uses a small synthetic astronomy corpus in Markdown format. It demonstrates the core RAG workflow, but it does not try to cover every document type, production deployment pattern, or full benchmarking workflow.

## Architecture At A Glance

```text
Markdown documents
    -> chunking
    -> embedding model (`embeddinggemma` via Ollama)
    -> ChromaDB vector store
    -> similarity retrieval
    -> prompt assembly
    -> local Gemma LLM via Ollama
    -> answer + optional evaluation/UI
```

## Prerequisites

- Python 3.10+ is recommended.
- Ollama must be installed locally and running.
- The chat model used in this tutorial is `gemma4:e4b`.
- The embedding model used in this tutorial is `embeddinggemma`.

## Quickstart

### Download Dependencies, Then Perform all Processing Locally

Use this path on a fresh machine that still has temporary internet access for package/model downloads.

Run this code block to setup the directory pre-requisites for
all the modules.

```bash
# Download the workshop code
git clone https://github.com/Smithsonian/ai4lam2026-workshop-rag
# Create workshop python environment
cd ai4lam2026-workshop-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Download & install ollama from: https://ollama.com/download
# Start the ollama server
ollama serve
# Download the LLMs
ollama pull gemma4:e4b
ollama pull embeddinggemma
# Define settings
source scripts/setup_env.sh
```

`scripts/setup_env.sh` sets `OLLAMA_KEEP_ALIVE=-1`,
which tells Ollama to keep the model loaded instead of unloading it after idle time.
Run `ollama ps` to see processor information about currently loaded models.

### Cleanup environment setup
deactivate
rm -rf .venv

Expected outcomes:
- `scripts/generate_astronomy_data.py` creates `dataset/astronomy_docs/` with 5 Markdown files.

## Suggested Learning Path

For each module:
1. Read the module `README.md`.
2. Try the exercise file (`*_ex.py`).
3. Compare with the solution (`*_sln.py`).
4. Run the solution before moving to the next dependent module.

## Tutorial Structure

The workshop is divided into five modules:

- **Module 1: Data Parsing** (`module_01_data_parsing/`)  
  Learn chunking strategies for small source documents.
- **Module 2: Vector Storage** (`module_02_vector_storage/`)  
  Build embeddings and a local ChromaDB collection.
- **Module 3: RAG Pipeline** (`module_03_retrieval_augmented_generation/`)  
  Retrieve relevant chunks and pass them to Ollama.
- **Module 4: Evaluation** (`module_04_evaluation/`)  
  Inspect answer quality with simple automated checks.
- **Module 5: Streamlit Interface** (`module_05_streamlit_interface/`)  
  Wrap the pipeline in a small local chat application.

## Module Dependencies

- Modules 1 and 2 require the generated dataset in `dataset/astronomy_docs/`.
- Modules 3 and 5 require the `astronomy` collection in `./chroma_db/`, which is created by running the Module 2 solution.
- Module 4 is easiest to understand after Module 3, because it assumes you already have a RAG answer to inspect.

## Privacy And Security Notes

This repository is intended for local execution:
- ChromaDB telemetry is disabled in `scripts/setup_env.sh`.
- Streamlit usage stats are disabled by `.streamlit/config.toml`.  `scripts/setup_env.sh` creates it, and Module 5 creates that file automatically if it is missing.
- Ollama runs against `http://localhost:11434`.

Two practical caveats:
- The first-time installation of Python packages and model assets is not offline unless you have already mirrored or preloaded them.
- The examples show local-only components, but this repository should still be understood as a tutorial, not a hardened production deployment guide.
