# llama_embedder.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.bm25 import BM25Embedding
from llama_index.core.extractors import SimpleMetadataExtractor


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
            self.extractor = SimpleMetadataExtractor(
                llm=llm,
                fields=["drug_name", "side_effects", "dosage", "drug_for_general_term", "interactions", "precautions","warnings", "usage_instructions", "storage_instructions"]
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
