# Module 05: Building a User Interface with Streamlit

In this final module, we move from running Python scripts to creating an interactive application. We use **Streamlit**, a framework designed for machine learning and data applications.

## Prerequisites

- Module 2 must already have created `./chroma_db/`.
- Ollama must be running locally.
- The Ollama models `gemma4:e4b` and `embeddinggemma` must already exist locally.
- Module 5 will create `.streamlit/config.toml` itself if it is missing so Streamlit usage stats stay disabled even when `scripts/setup_env.sh` was not sourced.

## Objectives
- Learn how to create a chat interface using `st.chat_message` and `st.chat_input`.
- Manage conversation state across multiple user interactions using `st.session_state`.
- Integrate the RAG pipeline (retrieval + generation) into an asynchronous-feeling UI.
- Display retrieved context for transparency (so users can verify where information came from).

## Core Concepts

### 1. Session State
Streamlit apps rerun the entire script every time a user interacts with a widget. To keep track of the chat history, we use `st.session_state`, which acts like a dictionary that persists across reruns.

### 2. The Chat Loop
A typical AI chatbot UI follows this loop:
1. **Render History**: Display previous messages from session state.
2. **Capture Input**: Wait for user input via `st.chat_input`.
3. **Process & Respond**: Retrieve context, generate an answer, and append both to history.
4. **Re-render**: Streamlit automatically updates the UI.

## How to Run
1. Ensure Ollama is running locally and has `gemma4:e4b` and `embeddinggemma` available.
2. Ensure the vector store already exists:
   ```bash
   python3 module_02_vector_storage/vector_db_sln.py
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   streamlit run module_05_streamlit_interface/app.py
   ```

Expected outcome:
- A local chat app opens in the browser.
- Asking an astronomy question should stream an answer into the chat UI plus show an expandable list of retrieved snippets.
- If the database is missing, the app should stop with an error explaining that Module 2 must be run first.

## Exercises
- **Customization**: Change the system prompt in `app.py` to change the personality of the astronomy assistant.
- **Context Limit**: Modify the retrieval function to allow users to select how many documents (`k`) are retrieved via a slider in the sidebar.
- **Streaming**: Inspect how `st.write_stream` is used and experiment with different presentation patterns, such as a cursor, partial status text, or source previews while generation is in progress.
