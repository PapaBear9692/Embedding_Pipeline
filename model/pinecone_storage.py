# pinecone_storage.py
from pinecone import Pinecone, ServerlessSpec
from app_config import PINECONE_API_KEY, PINECONE_INDEX_NAME


class PineconeStorage:
    """
    Handles Pinecone connection and hybrid (dense + sparse) upserts.
    """

    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("Missing Pinecone API key")

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.index = None

    def connect(self):
        """Connect to an existing Pinecone index."""
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")

    def create_index_if_not_exists(self, dense_dim: int):
        """Create a hybrid-enabled Pinecone index if it doesn't exist."""
        existing = self.pc.list_indexes().names()

        if self.index_name not in existing:
            print(f"Creating Pinecone hybrid index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dense_dim,     # dense vector dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Index created successfully.")
        else:
            print(f"Index '{self.index_name}' already exists.")

        self.connect()

    def upsert_batch(self, vectors):
        """Upsert a batch of hybrid vectors (dense + sparse)."""
        if not self.index:
            raise RuntimeError("Index not connected. Call connect() first.")

        self.index.upsert(vectors=vectors)
        print(f"Upserted batch of {len(vectors)} hybrid vectors.")
