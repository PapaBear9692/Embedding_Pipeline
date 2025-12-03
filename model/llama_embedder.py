# llama_embedder.py
import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.bm25 import BM25Embedding
from llama_index.llms.gemini import Gemini
from llama_index.core.extractors import SimpleMetadataExtractor

from app_config import FILTERS


ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

DEFAULT_DENSE_MODEL = "abhinand/MedEmbed-base-v0.1"
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash"


class HybridEmbedder:
    """
    Dense + Sparse embedding + optional structured metadata extraction (Gemini).
    Updated to match the initialization style of file:2, but without Pinecone.
    """

    def __init__(
        self,
        dense_model: str = DEFAULT_DENSE_MODEL,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        use_structured_metadata: bool = False,
    ):
        # Load environment variables
        load_dotenv(ENV_PATH)

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if use_structured_metadata and not google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY in .env")

        # ---------- Embedding Models ----------
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=dense_model,
            device="cpu",  # change if needed
        )

        # Sparse model (BM25)
        self.sparse = BM25Embedding()

        # Also keep dense model instance for convenience
        self.dense = Settings.embed_model

        # ---------- Optional Gemini metadata extractor ----------
        self.use_structured_metadata = use_structured_metadata
        self.extractor = None

        if use_structured_metadata:
            llm = Gemini(
                model=gemini_model,
                api_key=google_api_key,
                temperature=0.1,
            )

            # LLM chooses metadata fields from FILTERS
            self.extractor = SimpleMetadataExtractor(
                llm=llm,
                fields=["filter_by"],
                allowed_values=FILTERS,
                multiple=True,
            )

        # Print embedding dim for debugging (same style as file:2)
        test_vec = self.dense.get_text_embedding("hello world")
        print(f"[HybridEmbedder] Dense embedding dimension: {len(test_vec)}")

    # ---------- Dense + Sparse embedding ----------
    def embed_text(self, text: str):
        return {
            "dense": self.dense.get_text_embedding(text),
            "sparse": self.sparse.get_text_embedding(text),
        }

    # ---------- Gemini metadata extraction ----------
    def extract_structured_metadata(self, text: str):
        if not self.use_structured_metadata or not self.extractor:
            return {}
        return self.extractor.extract(text)
