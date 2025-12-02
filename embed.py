# embed.py
import json
import uuid
from pathlib import Path
from app_config import DATA_DIR, EMBEDDER_MODELS, EMBEDDER_PROVIDER, GOOGLE_API_KEY, LLM_MODELS, LLM_PROVIDER, OUTPUT_PATH, CHUNK_SIZE, OVERLAP, USE_LLM_METADATA
from llama_data_loader import load_documents, chunk_documents
from llama_embedder import HybridEmbedder



def embed_and_save():
    print("# Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} docs")

    print("# Chunking...")
    nodes = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"Created {len(nodes)} nodes")

    print("# Initializing embedder (Hybrid Dense + Sparse + Metadata)...")
    embedder = HybridEmbedder(
        dense_model=EMBEDDER_MODELS[EMBEDDER_PROVIDER],
        use_structured_metadata=USE_LLM_METADATA,
        model_name= LLM_MODELS[LLM_PROVIDER],
        google_api_key=GOOGLE_API_KEY
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
