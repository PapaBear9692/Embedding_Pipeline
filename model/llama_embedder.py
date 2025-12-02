# llama_embedder.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.bm25 import BM25Embedding
from llama_index.core.extractors import SimpleMetadataExtractor
from app_config import FILTERS


class HybridEmbedder:
    """
    Dense + Sparse embedding + optional LLM metadata extraction.
    """
    def __init__(
        self,
        dense_model="abhinand/MedEmbed-base-v0.1",
        use_structured_metadata=False,
        model_name=None,
        google_api_key=None
    ):
        self.dense = HuggingFaceEmbedding(model_name=dense_model)
        self.sparse = BM25Embedding()
        self.use_structured_metadata = use_structured_metadata

        if use_structured_metadata:
            if not google_api_key:
                raise ValueError("Google API key required for metadata extraction")
            
            from langchain.chat_models import ChatGoogleGemini
            
            llm = ChatGoogleGemini(api_key=google_api_key, model=model_name)

            # Only one field "filter_by", LLM picks one or more from FILTERS
            self.extractor = SimpleMetadataExtractor(
                llm=llm,
                fields=["filter_by"],
                allowed_values=FILTERS,
                multiple=True  # allow LLM to pick multiple if relevant
            )

    def embed_text(self, text: str):
        return {
            "dense": self.dense.get_text_embedding(text),
            "sparse": self.sparse.get_text_embedding(text),
        }

    def extract_structured_metadata(self, text: str):
        if not self.use_structured_metadata:
            return {}
        return self.extractor.extract(text)
