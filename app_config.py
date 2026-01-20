import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from prime_node import get_prime_nodes

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

EMBED_MODEL_NAME = "abhinand/MedEmbed-base-v0.1"  # or -large
EMBEDDING_DIM = 768

PINECONE_INDEX_NAME = "sqbot-data-index"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_NAMESPACE = None


def upsert_prime_nodes(pinecone_index) -> None:
    """
    Ensure BOTH Prime nodes exist:
      - Prime_Node_Pharma
      - Prime_Node_Herbal
      - Prime_Node_Agrovet
    """
    records = get_prime_nodes()

    for r in records:
        if len(r["values"]) != EMBEDDING_DIM:
            raise ValueError(
                f"{r['id']} embedding length {len(r['values'])} != EMBEDDING_DIM {EMBEDDING_DIM}"
            )

    pinecone_index.upsert(
        vectors=records,
        namespace=PINECONE_NAMESPACE,
    )


def init_settings_and_storage():
    load_dotenv(ENV_PATH)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in .env")

    # LlamaIndex settings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device="cpu",  # or "cuda"
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=30,
    )

    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )

        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        upsert_prime_nodes(pinecone_index)
        print("Prime nodes inserted (Pharma + Herbal + Agrovet).")

    else:
        print(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'")
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_NAMESPACE,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context
