# llama_hybrid_embedder.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.bm25 import BM25Retriever
from app_config import FILTERS

class HybridEmbedder:
    """
    Dense + Sparse embedding + optional LLM metadata extraction.
    """
    def __init__(self, dense_model=None, use_structured_metadata=False, model_name=None, google_api_key=None):
        self.dense_model = dense_model
        self.use_structured_metadata = use_structured_metadata

        
        self.dense = HuggingFaceEmbedding(model_name=model_name)

        # Sparse embedding retriever
        self.sparse = BM25Retriever.from_defaults(nodes=[])

        # Optional structured metadata extraction
        if use_structured_metadata:
            if not google_api_key:
                raise ValueError("Google API key required for metadata extraction")
            llm = Gemini(api_key=google_api_key, model=model_name)
            self.extractor = SimpleMetadataExtractor(
                llm=llm,
                fields=["filter_by"],
                allowed_values=FILTERS,
                multiple=True
            )

    def embed_text(self, text: str):
        dense_vec = self.dense.get_text_embedding(text)
        sparse_vec = self.sparse.get_sparse_vector(text)
        return {"dense": dense_vec, "sparse": sparse_vec}

    def extract_structured_metadata(self, text: str):
        if not self.use_structured_metadata:
            return {}
        return self.extractor.extract(text)
