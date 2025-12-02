# pinecone_storage.py
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from app_config import PINECONE_API_KEY, PINECONE_INDEX_NAME


class PineconeStorage:
    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing")

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.index = None

    def connect(self):
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")

    def create_index_if_not_exists(self, dimension: int):
        existing = self.pc.list_indexes().names()
        if self.index_name not in existing:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            print(f"Index '{self.index_name}' already exists.")

        self.connect()

    def upsert_vectors(self, batch: List[Dict[str, Any]]):
        if self.index is None:
            raise RuntimeError("Index not connected")

        self.index.upsert(vectors=batch)
