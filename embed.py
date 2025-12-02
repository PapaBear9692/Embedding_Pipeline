# embed.py
import uuid
import json
from pathlib import Path

from llama_data_loader import load_documents, chunk_documents
from llama_embedder import LlamaEmbedder


DATA_DIR = "data"
OUTPUT_PATH = "data_cache/embedded_chunks.jsonl"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 30


def embed_and_save():
    print("📌 Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents")

    print("📌 Chunking...")
    nodes = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(nodes)} chunks")

    print("📌 Loading embedder...")
    embedder = LlamaEmbedder(provider="hf",  model_name="abhinand/MedEmbed-base-v0.1")  # provider="openai" / "hf"

    print("📌 Embedding...")
    for node in nodes:
        node.embedding = embedder.embed(node.get_content())

    print("📌 Saving JSONL...")
    Path("data_cache").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for node in nodes:
            record = {
                "id": node.node_id or str(uuid.uuid4()),
                "text": node.get_content(),
                "embedding": node.embedding,
                "metadata": node.metadata,
            }
            f.write(json.dumps(record) + "\n")

    print(f"✅ Done — saved {len(nodes)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    embed_and_save()
