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
EMBEDDER_PROVIDER = "MedEmbed"

# --- Model Names ---
LLM_MODELS = {
    "gemini": "gemini-2.5-flash",
    "gpt": "gpt-3.5-turbo"
}

EMBEDDER_MODELS = {
    "default": "all-MiniLM-L6-v2",
    "PubMedBert" : "NeuML/pubmedbert-base-embeddings",
    "openai": "text-embedder-small",
    "MedEmbed" : "abhinand/MedEmbed-base-v0.1"
}

# --- Vector Store Config ---
PINECONE_INDEX_NAME = "medembed-index"
EMBEDDING_DIMENSIONS = {
    "default": 384,  # all-MiniLM-L6-v2 dimension
    "PubMedBert": 768, # PubMedBert dimension
    "openai": 1536,  # text-embedder-small dimension
    "MedEmbed": 768  # MedEmbed dimension
}

# --- Data Processing Config ---
DATA_DIR = "data"
OUTPUT_FILE = "data_cache/embedded_nodes_hybrid.jsonl"

CHUNK_SIZE = 350
OVERLAP = 30

# --- RAG Config ---
TOP_K_RESULTS = 8

USE_LLM_METADATA=True

FILTERS = [
    # General medicine types
    "painkiller", "fever reducer", "fever", "antibiotic", "cold medicine",
    "allergy medicine", "vitamin", "supplement", "sleep aid",
    "steroid", "anti inflammatory", "cough medicine", "stomach medicine",
    "heart_medicine", "diabetes medicine", "eye medicine", "ear medicine",
    "skin cream", "nasal spray", "antidepressant", "antiviral", "cold", "cough",
    "gastrics", "headache", "infection", "inflammation",

    # Common medical document sections
    "dosage", "usage", "side_effects", "warnings", "precautions",
    "interactions", "storage", "indications", "contraindications"
]


# -----------------------------
# Section labels for structured metadata
# -----------------------------
SECTION_LABELS = [
    "brand generic",
    "composition",
    "pharmacology",
    "indications",
    "dosage",
    "usage",
    "contraindications",
    "warnings",
    "side effects",
    "interactions",
    "overdose",
    "storage",
    "how supplied",
    "other",
]