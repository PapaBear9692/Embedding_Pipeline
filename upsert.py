# store-index.py
import logging
import time
from typing import Optional

from llama_index.core import VectorStoreIndex

from dataloader import load_documents
from app_config import init_settings_and_storage


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


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
    storage_context = init_settings_and_storage()

    logging.info("Building VectorStoreIndex with automatic chunking...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    elapsed = time.time() - start
    logging.info("Index built successfully in %.2f seconds.", elapsed)
    logging.info("You can now query this index from your LlamaIndex chatbot.")

    return index


if __name__ == "__main__":
    build_index()
