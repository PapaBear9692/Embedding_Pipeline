import json
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

from dataloader import load_documents
from app_config import init_settings_and_storage
from ocr import run_ocr


ALLOWED_TYPES = {"pharma", "herbal", "agrovet"}


def _normalize_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t = str(t).strip().lower()
    return t if t in ALLOWED_TYPES else None


def _iter_types(train_type: Optional[str]) -> List[str]:
    """
    If train_type is provided -> [train_type]
    If None -> all types
    """
    t = _normalize_type(train_type)
    return [t] if t else ["pharma", "herbal", "agrovet"]


def _cap_type(t: str) -> str:
    return t.strip().lower().capitalize()


def _prime_node_id(train_type: str) -> str:
    return f"Prime_Node_{_cap_type(train_type)}"


def update_list(storage_context, documents, train_type: str) -> None:
    """
    Update exactly ONE prime node (for the given train_type) using the documents passed in.
    """
    vector_store = storage_context.vector_store
    pinecone_index = vector_store._pinecone_index
    namespace = vector_store.namespace

    t = _normalize_type(train_type)
    if not t:
        raise ValueError(f"Invalid train_type for update_list: {train_type}")

    node_id = _prime_node_id(t)

    # 1) collect product names from filenames (remove .pdf)
    new_items = [
        Path(d.metadata.get("file_name")).stem
        for d in documents
        if d.metadata.get("file_name")
    ]
    new_items = [x.strip() for x in new_items if x and x.strip()]
    if not new_items:
        return

    # 2) fetch existing node
    res = pinecone_index.fetch(ids=[node_id], namespace=namespace)
    if node_id not in getattr(res, "vectors", {}):
        raise ValueError(f"Node ID not found in Pinecone: {node_id}")

    vec = res.vectors[node_id]
    meta = getattr(vec, "metadata", None) or vec.get("metadata", {}) or {}

    raw = meta.get("_node_content")
    if not raw:
        raise ValueError(f"Missing _node_content for node: {node_id}")

    # 3) parse node JSON
    node_obj = json.loads(raw)
    text = node_obj.get("text", "") or ""

    # 4) split existing comma-separated items
    existing_items = [
        x.strip()
        for x in text.replace("\n", " ").split(",")
        if x.strip()
    ]

    # dedupe new_items in the same run (prevents Nexum,Nexum,Nexum)
    new_items_unique: List[str] = []
    seen_new = set()
    for it in new_items:
        k = it.lower()
        if k not in seen_new:
            seen_new.add(k)
            new_items_unique.append(it)

    # 5) merge (case-insensitive, keep original order)
    seen = {x.lower(): x for x in existing_items}
    for item in new_items_unique:
        if item.lower() not in seen:
            seen[item.lower()] = item

    merged_items = list(seen.values())

    # 6) rebuild text exactly as comma-separated
    node_obj["text"] = ", ".join(merged_items)

    # 7) write back
    pinecone_index.update(
        id=node_id,
        set_metadata={"_node_content": json.dumps(node_obj, ensure_ascii=False)},
        namespace=namespace,
    )


def cleanup_train_data(train_type: str | None = None) -> None:
    base = Path(__file__).resolve().parent / "data" / "train_data"
    if not base.exists():
        return

    train_type = _normalize_type(train_type)

    if train_type:
        target = base / _cap_type(train_type)
        if target.exists():
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
        return

    for item in base.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)



def build_index(train_type: str | None = None) -> Optional[Tuple[VectorStoreIndex, int]]:
    types_to_run = _iter_types(train_type)

    print(f"Building index for types: {types_to_run}")

    storage_context = init_settings_and_storage()

    total_chunks = 0
    last_index: Optional[VectorStoreIndex] = None

    for t in types_to_run:
        run_ocr(train_type=t)

        documents = load_documents(train_type=t)
        if not documents:
            print(f"[{t}] No documents found; skipping.")
            cleanup_train_data(train_type=t)
            continue

        # stamp type
        for d in documents:
            d.metadata["product_type"] = t

        # embed this batch
        last_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )

        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        total_chunks += len(nodes)

        update_list(storage_context, documents, train_type=t)
        cleanup_train_data(train_type=t)

        print(f"[{t}] Done. Chunks added: {len(nodes)}")

    
    if last_index is None:
        return None

    return last_index, total_chunks



if __name__ == "__main__":
    build_index()
