# app_config.py
import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

EMBED_MODEL_NAME = "abhinand/MedEmbed-base-v0.1"
EMBEDDING_DIM = 768

PINECONE_INDEX_NAME = "medicine-chatbot-sample-data-llama-v1"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_NAMESPACE = None


def init_settings_and_storage():
    """Initialize embeddings, automatic chunking, and Pinecone vector store."""
    # Load environment variables
    load_dotenv(ENV_PATH)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in .env")

    # Embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device="cpu",  # change to "cuda" if you have GPU
    )

    # Automatic chunking / node parsing
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,    # max characters per chunk (tune as needed)
        chunk_overlap=50,  # overlap between chunks for context continuity
    )

    # Pinecone setup
    pc = Pinecone(api_key=pinecone_api_key)

    # Depending on your pinecone client version, list_indexes() typically
    # returns an object with a .names() method that lists index names.
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
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
