import logging
import time
from typing import Optional
import json
from pathlib import Path

from llama_index.core import VectorStoreIndex

from dataloader import load_documents
from config import init_settings_and_storage, PINECONE_NAMESPACE

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

EXPORT_PATH = Path(r"E:\work\Medicine-Chatbot\data\embedded_data.json")


def export_index_to_json(pinecone_index):
    """Export all vectors (embeddings + metadata + text) from Pinecone to a JSON file."""
    exported = []
    # For Pinecone v3, None usually means the "default" namespace (empty string)
    namespace = PINECONE_NAMESPACE or ""

    # iterate IDs in index
    for id_batch in pinecone_index.list(namespace=namespace):
        if not id_batch:
            continue

        # Fetch full vectors for this batch
        response = pinecone_index.fetch(ids=id_batch, namespace=namespace)
        vectors = response.vectors or {}

        for vid, vec in vectors.items():
            # vec is a pinecone.data.Vector object
            metadata = getattr(vec, "metadata", {}) or {}
            embedding = getattr(vec, "values", []) or []

            # Try to get text from metadata (LlamaIndex usually stores text there)
            text = metadata.get("text") or metadata.get("content") or ""

            # If still empty, try to parse text out of _node_content (LlamaIndex packed it there)
            if not text and "_node_content" in metadata:
                try:
                    node_content = json.loads(metadata["_node_content"])
                    text = node_content.get("text", "") or text
                except Exception:
                    pass  # if parsing fails, just leave text as-is

            # (Optional) if you don't want gigantic _node_content in your final JSON, you can drop it:
            # metadata = {k: v for k, v in metadata.items() if k != "_node_content"}

            exported.append(
                {
                    "id": vid,
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=4, ensure_ascii=False)

    logging.info("Exported %d vectors to %s", len(exported), EXPORT_PATH)


def build_index() -> Optional[VectorStoreIndex]:
    start = time.time()
    logging.info("Starting index build process...")

    logging.info("Loading documents...")
    documents = load_documents()
    if not documents:
        logging.warning("No documents found. Aborting index build.")
        return None

    logging.info(
        "Loaded %d documents. Initializing embeddings, LLM, and Pinecone storage context...",
        len(documents),
    )
    storage_context, pinecone_index = init_settings_and_storage()

    logging.info("Building VectorStoreIndex with automatic chunking...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    logging.info("Exporting embeddings from Pinecone to JSON...")
    # export_index_to_json(pinecone_index)

    elapsed = time.time() - start
    logging.info("Index built successfully in %.2f seconds.", elapsed)
    logging.info("You can now query this index from your LlamaIndex chatbot.")

    return index


if __name__ == "__main__":
    build_index()
