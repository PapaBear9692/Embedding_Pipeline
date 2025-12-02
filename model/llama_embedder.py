# file: 2
# embedder.py
import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"

PINECONE_INDEX_NAME = "medicine-chatbot-sample-data-llama"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_NAMESPACE = None


def init_settings_and_storage():
    """Initialize embeddings, LLM, and Pinecone store (no chunking here)."""
    load_dotenv(ENV_PATH)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not pinecone_api_key or not google_api_key:
        raise ValueError("Missing PINECONE_API_KEY or GOOGLE_API_KEY in .env")

    # Embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device="cpu",  # change to "cuda" if you have GPU
    )

    # LLM
    Settings.llm = Gemini(
        model=GEMINI_MODEL_NAME,
        api_key=google_api_key,
        temperature=0.1,
    )

    # No node_parser needed because we’ll create nodes manually

    # Detect embedding dimension
    dummy_embed = Settings.embed_model.get_text_embedding("hello world")
    embedding_dim = len(dummy_embed)
    print(f"Embedding dimension: {embedding_dim}")

    # Pinecone setup
    pc = Pinecone(api_key=pinecone_api_key)
    existing = [idx["name"] for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
    else:
        print(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'")

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_NAMESPACE,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context
