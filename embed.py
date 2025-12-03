# embed.py
import json
import uuid
from pathlib import Path
from model.llama_data_loader import load_documents, chunk_documents
from model.llama_embedder import HybridEmbedder
from app_config import (
    DATA_DIR,
    EMBEDDER_MODELS,
    EMBEDDER_PROVIDER,
    OUTPUT_PATH,
    CHUNK_SIZE,
    OVERLAP,
    USE_LLM_METADATA,
)




def embed_and_save():
    print("# Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents")

    print("# Chunking...")
    nodes = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"Created {len(nodes)} chunks")

    print("# Initializing HybridEmbedder...")
    embedder = HybridEmbedder(
        dense_model=EMBEDDER_MODELS[EMBEDDER_PROVIDER],
        use_structured_metadata=USE_LLM_METADATA,
    )

    print("# Embedding nodes...")
    Path("data_cache").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for node in nodes:
            text = node.get_content()

            # Hybrid embedding
            emb = embedder.embed_text(text)

            # Base metadata from loader/chunker
            metadata = dict(node.metadata)

            # Optional: structured metadata extraction (Gemini)
            if embedder.use_structured_metadata:
                extracted = embedder.extract_structured_metadata(text)
                metadata.update(extracted)

            record = {
                "id": node.node_id or str(uuid.uuid4()),
                "text": text,
                "dense": emb["dense"],
                "sparse": emb["sparse"],
                "metadata": metadata,
                "relationships": {
                rel.name: getattr(info, "node_id", info.get("node_id") if isinstance(info, dict) else None)
                for rel, info in node.relationships.items()
            },

            }

            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved {len(nodes)} hybrid-embedded nodes → {OUTPUT_PATH}")


if __name__ == "__main__":
    embed_and_save()
