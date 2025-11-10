import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Model Selection ---
# Change these values to swap models easily

# Supported: "gemini", "gpt"
LLM_PROVIDER = "gemini"

# Supported: "default" (all-miniLm-l6), "openai" (text-embedder-small), "PubMedBert" (NeuML/pubmedbert-base-embeddings)
EMBEDDER_PROVIDER = "PubMedBert"

# --- Model Names ---
LLM_MODELS = {
    "gemini": "gemini-2.5-flash",
    "gpt": "gpt-3.5-turbo"
}

EMBEDDER_MODELS = {
    "default": "all-MiniLM-L6-v2",
    "PubMedBert" : "NeuML/pubmedbert-base-embeddings",
    "openai": "text-embedder-small",
}

# --- Vector Store Config ---
PINECONE_INDEX_NAME = "sqbot-index" # <<< IMPORTANT: Change this
EMBEDDING_DIMENSIONS = {
    "default": 384,  # all-MiniLM-L6-v2 dimension
    "PubMedBert": 768, # PubMedBert dimension
    "openai": 1536,  # text-embedder-small dimension
}

# --- Data Processing Config ---
DATA_DIRECTORY = "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# --- RAG Config ---
TOP_K_RESULTS = 10