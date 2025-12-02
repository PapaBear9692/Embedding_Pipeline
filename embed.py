# embed.py
import json
import uuid
from pathlib import Path

from llama_data_loader import load_documents, chunk_documents
from llama_embedder import HybridEmbedder


DATA_DIR = "data"
OUTPUT_PATH = "data_cache/embedded_nodes_hybrid.jsonl"

CHUNK_SIZE = 400
OVERLAP = 30


def embed_and_save():
    print("# Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} docs")

    print("# Chunking...")
    nodes = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"Created {len(nodes)} nodes")

    print("# Initializing embedder (Hybrid Dense + Sparse + Metadata)...")
    embedder = HybridEmbedder(
        dense_model="abhinand/MedEmbed-base-v0.1",
        use_structured_metadata=False  # set True if needed
    )

    print("# Embedding nodes...")
    Path("data_cache").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for node in nodes:
            text = node.get_content()
            emb = embedder.embed_text(text)

            metadata = node.metadata

            # Optional LLM metadata extraction
            if embedder.use_structured_metadata:
                metadata.update(embedder.extract_structured_metadata(text))

            record = {
                "id": node.node_id or str(uuid.uuid4()),
                "text": text,
                "dense": emb["dense"],
                "sparse": emb["sparse"],
                "metadata": metadata,
                "relationships": {
                    rel.name: info.node_id
                    for rel, info in node.relationships.items()
                },
            }

            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved {len(nodes)} hybrid-embedded nodes → {OUTPUT_PATH}")


if __name__ == "__main__":
    embed_and_save()
