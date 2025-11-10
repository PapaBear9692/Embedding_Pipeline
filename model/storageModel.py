from typing import List, Dict, Any
from model.core_interface import VectorStoreInterface
from pinecone import Pinecone, ServerlessSpec
from app_config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)

class PineconeStorage(VectorStoreInterface):
    """Implementation for Pinecone vector store."""
    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment.")
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.index = None # Will be set after connection
        print(f"PineconeStorage initialized for index: '{self.index_name}'")

    def connect_to_index(self):
        """Connects the index object to an existing index."""
        try:
            self.index = self.pc.Index(self.index_name)
            print(f"Successfully connected to Pinecone index: '{self.index_name}'")
        except Exception as e:
            print(f"Error connecting to Pinecone index '{self.index_name}': {e}")
            print("Please ensure the index exists (run the indexing script).")
            raise

    def create_index_if_not_exists(self, dimension: int):
        """Creates the Pinecone index if it doesn't already exist."""
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}' with dimension {dimension}...")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print("Index created successfully. Waiting for it to be ready...")
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    import time
                    time.sleep(5)
                print("Index is ready.")
            except Exception as e:
                print(f"Error creating Pinecone index: {e}")
                raise
        else:
            print(f"Index '{self.index_name}' already exists.")
        
        # Connect to the index after creation/verification
        self.connect_to_index()

    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """Upserts documents and their embeddings in batches."""
        if not self.index:
            raise ConnectionError("Pinecone index not initialized. Call connect_to_index() first.")
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                print(f"Upserting batch {i//batch_size + 1}...")
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error upserting batch: {e}")
        print("Upsert complete.")

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Queries the vector store and returns metadata."""
        if not self.index:
            raise ConnectionError("Pinecone index not initialized.")
            
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            # Return the metadata of the matches
            return [match.get('metadata', {}) for match in results.get('matches', [])]
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []