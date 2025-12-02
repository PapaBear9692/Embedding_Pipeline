# embed.py
import json
from pathlib import Path
import uuid
from llama_hybrid_embedder import HybridEmbedder
from llama_data_loader import load_documents, chunk_documents
from app_config import (
    DATA_DIR,
    EMBEDDER_MODELS,
    EMBEDDER_PROVIDER,
    GOOGLE_API_KEY,
    LLM_MODELS,
    LLM_PROVIDER,
    OUTPUT_PATH,
    CHUNK_SIZE,
    OVERLAP,
    USE_LLM_METADATA,
)

# 1. Load documents
print("# Loading documents...")
docs = load_documents(DATA_DIR)
print(f"Loaded {len(docs)} documents")

# 2. Chunk documents
print("# Chunking...")
nodes = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
print(f"Created {len(nodes)} nodes")

# 3. Initialize embedder
print("# Initializing embedder...")
embedder = HybridEmbedder(
    dense_model=EMBEDDER_MODELS[EMBEDDER_PROVIDER],
    use_structured_metadata=USE_LLM_METADATA,
    model_name=LLM_MODELS[LLM_PROVIDER],
    google_api_key=GOOGLE_API_KEY,
)

# 4. Embed nodes & extract metadata
print("# Embedding nodes...")
Path("data_cache").mkdir(exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for node in nodes:
        text = node.get_content()
        emb = embedder.embed_text(text)

        # Optional LLM metadata
        if embedder.use_structured_metadata:
            node.metadata.update(embedder.extract_structured_metadata(text))

        record = {
            "id": node.node_id or str(uuid.uuid4()),
            "text": text,
            "dense": emb["dense"],
            "sparse": emb["sparse"],
            "metadata": node.metadata,
            "relationships": {rel.name: info.node_id for rel, info in node.relationships.items()},
        }
        f.write(json.dumps(record) + "\n")

print(f"✅ Saved {len(nodes)} nodes → {OUTPUT_PATH}")
