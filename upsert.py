# upsert.py
from typing import Optional

from llama_index.core import VectorStoreIndex

from dataloader import load_documents
from app_config import init_settings_and_storage


def update_list(storage_context, documents):
    vector_store = storage_context.vector_store
    pinecone_index = vector_store._pinecone_index
    node_id = "0be0688d-4df4-4f2c-9d9c-25efb2447d6f#086f1d93-589f-4f77-b465-413e1c9bef43"

    # 1) collect filenames from documents
    filenames = [
        d.metadata.get("file_name")
        for d in documents
        if d.metadata.get("file_name")
    ]

    if not filenames:
        return

    # make one clean text block
    text_to_append = "\n".join(filenames)

    # 2) fetch existing node
    res = pinecone_index.fetch(
        ids=[node_id],
        namespace=vector_store.namespace,
    )

    if node_id not in res.vectors:
        raise ValueError("Node ID not found in Pinecone")

    meta = res.vectors[node_id]["metadata"]

    # 3) append text
    current_text = meta.get("_node_content", "")
    updated_text = current_text.rstrip() + "\n" + text_to_append

    # 4) update metadata only
    pinecone_index.update(
        id=node_id,
        set_metadata={"_node_content": updated_text},
        namespace=vector_store.namespace,
    )


def build_index(job_dir: str) -> Optional[VectorStoreIndex]:

    documents = load_documents(job_dir)
    if not documents:
        return None

    storage_context = init_settings_and_storage()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    update_list(storage_context, documents)

    return index


# if __name__ == "__main__":
#     build_index()
