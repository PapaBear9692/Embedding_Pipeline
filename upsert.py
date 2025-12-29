import json
import shutil
from typing import Optional, Tuple
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings


from dataloader import load_documents
from app_config import init_settings_and_storage


def update_list(storage_context, documents):
    vector_store = storage_context.vector_store
    pinecone_index = vector_store._pinecone_index
    node_id = "f3f7bfd0-45d6-450f-82ec-b529d242a8b4#893da7cd-66a7-4978-8094-5f5c7b6b8696"

    # 1) collect product names from filenames (remove .pdf)
    new_items = [
        Path(d.metadata.get("file_name")).stem
        for d in documents
        if d.metadata.get("file_name")
    ]
    new_items = [x.strip() for x in new_items if x.strip()]
    if not new_items:
        return

    # 2) fetch existing node
    res = pinecone_index.fetch(
        ids=[node_id],
        namespace=vector_store.namespace,
    )
    if node_id not in res.vectors:
        raise ValueError("Node ID not found in Pinecone")

    vec = res.vectors[node_id]
    meta = getattr(vec, "metadata", None) or vec.get("metadata", {}) or {}

    raw = meta.get("_node_content")
    if not raw:
        raise ValueError("Missing _node_content")

    # 3) parse node JSON
    node_obj = json.loads(raw)
    text = node_obj.get("text", "") or ""

    # 4) split existing comma-separated items
    # normalize line breaks -> spaces, then split by comma
    existing_items = [
        x.strip()
        for x in text.replace("\n", " ").split(",")
        if x.strip()
    ]

    # 5) merge (case-insensitive, keep original order)
    seen = {x.lower(): x for x in existing_items}
    for item in new_items:
        if item.lower() not in seen:
            seen[item.lower()] = item

    merged_items = list(seen.values())

    # 6) rebuild text exactly as comma-separated
    node_obj["text"] = ", ".join(merged_items)

    # 7) write back
    pinecone_index.update(
        id=node_id,
        set_metadata={
            "_node_content": json.dumps(node_obj, ensure_ascii=False)
        },
        namespace=vector_store.namespace,
    )


def cleanup_train_data():
    # CLEANUP: delete contents of train_data, keep folder
    TRAIN_DATA_DIR = Path(__file__).resolve().parent / "data" / "train_data"
    if TRAIN_DATA_DIR.exists() and TRAIN_DATA_DIR.is_dir():
        for item in TRAIN_DATA_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def build_index() -> Optional[Tuple[VectorStoreIndex, int]]:
    print("Building index...")
    documents = load_documents()
    if not documents:
        return None

    storage_context = init_settings_and_storage()
    print("Created VectorStore Index...")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    chunk_count = len(nodes)

    #update_list(storage_context, documents)
    cleanup_train_data()

    return index, chunk_count


if __name__ == "__main__":
    build_index()
