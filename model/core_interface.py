from abc import ABC, abstractmethod
from typing import List, Dict, Any

class EmbeddingInterface(ABC):
    """Interface for an embedding model."""
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Returns the embedding dimension (e.g., 384, 768)."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts (for indexing)."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query text (for retrieval)."""
        pass

class LLMInterface(ABC):
    """Interface for a Language Model."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: List[Dict[str, Any]]) -> str:
        """Generates a response based on a prompt and retrieved context."""
        pass

class VectorStoreInterface(ABC):
    """Interface for a vector database."""
    
    @abstractmethod
    def create_index_if_not_exists(self, dimension: int):
        """Checks for and creates the vector index if needed."""
        pass
        
    @abstractmethod
    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """Upserts documents and their embeddings into the store."""
        pass
        
    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Queries the store for similar documents."""
        pass