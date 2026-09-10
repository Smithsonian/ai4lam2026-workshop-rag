#!/bin/bash

# Disable Telemetry for various tools
export ANONYMIZED_TELEMETRY=False # ChromaDB
export HF_HUB_OFFLINE=1          # Hugging Face Offline mode (optional: revert to 0 if models are needed from hugging face.  HF_HUB_OFFLINE is not relevant for the tutorial because all models are from ollama.)
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_KEEP_ALIVE=-1

# Streamlit telemetry is usually disabled via config file. 
# Create a .streamlit/config.toml file
mkdir -p .streamlit
cat <<EOF > .streamlit/config.toml
[browser]
gatherUsageStats = false
EOF

echo "Telemetry disabled and environment variables set."
echo "Note: Please source this script or add these exports to your ~/.bashrc"
