# upsert.py
from typing import Optional

from llama_index.core import VectorStoreIndex

from dataloader import load_documents
from app_config import init_settings_and_storage


def build_index() -> Optional[VectorStoreIndex]:

    documents = load_documents()
    if not documents:
        return None

    storage_context = init_settings_and_storage()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    return index


if __name__ == "__main__":
    build_index()
