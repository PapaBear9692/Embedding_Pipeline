# file: 3
# store-index.py
import logging
import time
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from dataloader import load_documents
from embedder import init_settings_and_storage


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def build_index(
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Optional[VectorStoreIndex]:
    """
    Build a Pinecone-backed VectorStoreIndex using LlamaIndex.

    Steps:
    - Load documents via `dataloader.load_documents()`
    - Split them into smaller nodes (chunks) for better retrieval
    - Store embeddings in Pinecone via `embedder.init_settings_and_storage()`
    """
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

    # Chunk documents into smaller nodes for better retrieval granularity
    logging.info(
        "Splitting %d documents into nodes (chunk_size=%d, chunk_overlap=%d)...",
        len(documents),
        chunk_size,
        chunk_overlap,
    )
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    if not nodes:
        logging.warning("No nodes were created from documents. Aborting index build.")
        return None

    logging.info("Created %d nodes. Building VectorStoreIndex in Pinecone...", len(nodes))
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    elapsed = time.time() - start
    logging.info("Index built successfully in %.2f seconds.", elapsed)
    logging.info("You can now query this index from your LlamaIndex chatbot.")

    return index


if __name__ == "__main__":
    build_index()
