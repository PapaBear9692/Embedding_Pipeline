import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.llms.openai_like import OpenAILike


from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

EMBED_MODEL_NAME = "abhinand/MedEmbed-base-v0.1"  # or -large

GEMINI_MODEL_NAME = "models/gemini-2.5-flash"
# GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite"

# =========== Groq LLaMA-3 configuration ================ #
# #LLAMA_MODEL_NAME = "llama-3.1-8b-instant"
# #LLAMA_MODEL_NAME = "openai/gpt-oss-120b"
# LLAMA_MODEL_NAME = "qwen/qwen3-32b"
# LLAMA_API_BASE = "https://api.groq.com/openai/v1"
# ======================================================= #

PINECONE_INDEX_NAME = "medbot"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_NAMESPACE = None  # or a string if you want one

def init_settings_and_storage():
    load_dotenv(ENV_PATH)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not pinecone_api_key or not google_api_key:
        raise ValueError("Missing PINECONE_API_KEY or GOOGLE_API_KEY in .env")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device="cpu",  # or "cuda"
    )

    Settings.llm = GoogleGenAI(
        model=GEMINI_MODEL_NAME,
        api_key=google_api_key,
        temperature=0.5,
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=30,
    )

    #================================================#
    # pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # llama_api_key = os.getenv("LLAMA_API_KEY")

    # if not pinecone_api_key or not llama_api_key:
    #     raise ValueError("Missing PINECONE_API_KEY or LLAMA_API_KEY in .env")
    #  # Embedding model stays the same
    # Settings.embed_model = HuggingFaceEmbedding(
    #     model_name=EMBED_MODEL_NAME,
    #     device="cpu",  # or "cuda" if you have GPU
    # )

    # # Use Groq LLaMA-3 via OpenAI-compatible API
    # Settings.llm = OpenAILike(
    #     model=LLAMA_MODEL_NAME,       # "llama3-8b-8192"
    #     api_base=LLAMA_API_BASE,      # "https://api.groq.com/openai/v1"
    #     api_key=llama_api_key,
    #     temperature=0.5,
    #     is_chat_model=True,
    # )

    # Settings.node_parser = SentenceSplitter(
    #     chunk_size=512,
    #     chunk_overlap=50,
    # )
    #================================================#

    dummy_embed = Settings.embed_model.get_text_embedding("hello world")
    embedding_dim = len(dummy_embed)
    print(f"Embedding dimension: {embedding_dim}")

    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
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
    return storage_context, pinecone_index
