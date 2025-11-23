from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
from app_config import EMBEDDER_MODELS, EMBEDDING_DIMENSIONS, OPENAI_API_KEY

class BaseEmbedder:
    """Base class for all embedding providers."""
    def embed_documents(self, texts):
        raise NotImplementedError
    
    def embed_query(self, text):
        raise NotImplementedError
        
    def get_dimension(self):
        raise NotImplementedError

class HuggingFaceEmbedder(BaseEmbedder):
    """Wrapper for Sentence Transformer models (including default and PubMedBert)."""
    def __init__(self, model_name, dimension):
        # LangChain's SentenceTransformerEmbeddings handles the model loading
        self.embedder = SentenceTransformerEmbeddings(
            model_name=model_name
        )
        self._dimension = dimension

    def embed_documents(self, texts):
        return self.embedder.embed_documents(texts)

    def embed_query(self, text):
        return self.embedder.embed_query(text)
        
    def get_dimension(self):
        return self._dimension

class OpenAIEmbedder(BaseEmbedder):
    """Wrapper for OpenAI embedding models."""
    def __init__(self, model_name, dimension, api_key):
        if not api_key:
            raise ValueError("OpenAI API Key not found.")
            
        self.embedder = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
        self._dimension = dimension

    def embed_documents(self, texts):
        # OpenAI returns embeddings directly as a list of lists (embeddings for documents)
        return self.embedder.embed_documents(texts)

    def embed_query(self, text):
        # OpenAI returns a single embedding for the query
        return self.embedder.embed_query(text)
        
    def get_dimension(self):
        return self._dimension

def get_embedder(provider: str) -> BaseEmbedder:
    """
    Factory function to get the correct embedder implementation based on the provider name.
    """
    if provider == "default":
        model_name = EMBEDDER_MODELS["default"]
        dimension = EMBEDDING_DIMENSIONS["default"]
        print(f"Using default HuggingFace Embedder: {model_name}")
        return HuggingFaceEmbedder(model_name, dimension)
    
    # Logic to support the PubMedBert model
    elif provider == "PubMedBert":
        model_name = EMBEDDER_MODELS["PubMedBert"]
        dimension = EMBEDDING_DIMENSIONS["PubMedBert"]
        print(f"Using PubMedBert Embedder: {model_name}")
        return HuggingFaceEmbedder(model_name, dimension)
    
    elif provider == "MedEmbed":
        model_name = EMBEDDER_MODELS["MedEmbed"]
        dimension = EMBEDDING_DIMENSIONS["MedEmbed"]
        print(f"Using MedEmbed Embedder: {model_name}")
        return HuggingFaceEmbedder(model_name, dimension)

    elif provider == "openai":
        model_name = EMBEDDER_MODELS["openai"]
        dimension = EMBEDDING_DIMENSIONS["openai"]
        print(f"Using OpenAI Embedder: {model_name}")
        return OpenAIEmbedder(model_name, dimension, OPENAI_API_KEY)
    
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")
