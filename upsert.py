import json
import shutil
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

from dataloader import load_documents
from app_config import init_settings_and_storage
from ocr import run_ocr


ALLOWED_TYPES = {"pharma", "herbal"}


def _cap_type(t: str) -> str:
    """'pharma' -> 'Pharma'"""
    return t.strip().lower().capitalize()


def _normalize_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t = t.strip().lower()
    return t if t in ALLOWED_TYPES else None


def _prime_node_id_for_type(train_type: str) -> str:
    # Prime_Node_Pharma / Prime_Node_Herbal
    return f"Prime_Node_{_cap_type(train_type)}"


def _extract_items_from_docs(documents) -> List[str]:
    # Collect product names from metadata file_name (remove .pdf)
    items = []
    for d in documents:
        fn = (d.metadata.get("file_name") or "").strip()
        if not fn:
            continue
        items.append(Path(fn).stem.strip())
    return [x for x in items if x]


def _merge_comma_list(existing_text: str, new_items: List[str]) -> str:
    existing_items = [
        x.strip()
        for x in (existing_text or "").replace("\n", " ").split(",")
        if x.strip()
    ]

    seen = {x.lower(): x for x in existing_items}
    for item in new_items:
        key = item.lower()
        if key not in seen:
            seen[key] = item

    merged_items = list(seen.values())
    return ", ".join(merged_items)


def update_list(storage_context, documents, train_type: str | None = None) -> None:
    """
    Updates Prime node(s) with product names.
    - If train_type is provided: updates only that node.
    - If train_type is None: groups docs by d.metadata['product_type'] and updates each type node.
    """
    vector_store = storage_context.vector_store
    pinecone_index = vector_store._pinecone_index
    namespace = vector_store.namespace

    train_type = _normalize_type(train_type)

    # Group docs by type
    docs_by_type: Dict[str, List] = {}

    if train_type:
        docs_by_type[train_type] = documents
    else:
        # Expect dataloader to set product_type per doc when loading both types,
        # but we guard anyway.
        for d in documents:
            t = _normalize_type(d.metadata.get("product_type"))
            if not t:
                continue
            docs_by_type.setdefault(t, []).append(d)

    for t, docs in docs_by_type.items():
        new_items = _extract_items_from_docs(docs)
        if not new_items:
            continue

        node_id = _prime_node_id_for_type(t)

        # Fetch existing node
        res = pinecone_index.fetch(ids=[node_id], namespace=namespace)
        if node_id not in getattr(res, "vectors", {}):
            # If you prefer hard-fail, change this to raise ValueError(...)
            print(f"WARNING: Prime node not found in Pinecone: {node_id} (skipping list update)")
            continue

        vec = res.vectors[node_id]
        meta = getattr(vec, "metadata", None) or vec.get("metadata", {}) or {}

        raw = meta.get("_node_content")
        if not raw:
            print(f"WARNING: Missing _node_content for node {node_id} (skipping list update)")
            continue

        # Parse node JSON
        try:
            node_obj = json.loads(raw)
        except Exception:
            print(f"WARNING: Could not parse _node_content JSON for node {node_id} (skipping list update)")
            continue

        text = node_obj.get("text", "") or ""

        # Merge and write back
        node_obj["text"] = _merge_comma_list(text, new_items)

        pinecone_index.update(
            id=node_id,
            set_metadata={"_node_content": json.dumps(node_obj, ensure_ascii=False)},
            namespace=namespace,
        )


def cleanup_train_data(train_type: str | None = None) -> None:
    """
    Deletes OCR output after embedding.
    - If train_type provided: deletes only data/train_data/<Type>
    - If None: deletes everything under data/train_data (both types)
    """
    base = Path(__file__).resolve().parent / "data" / "train_data"
    if not base.exists() or not base.is_dir():
        return

    train_type = _normalize_type(train_type)

    if train_type:
        target = base / _cap_type(train_type)
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
        return

    # None -> wipe everything inside train_data
    for item in base.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)


def build_index(train_type: str | None = None) -> Optional[Tuple[VectorStoreIndex, int]]:
    """
    Manual training: pass train_type='pharma' or 'herbal' -> only that folder processed.
    Crawler training: call build_index() with None -> process both types (ocr/loader should support this).
    """
    train_type = _normalize_type(train_type)

    print("Building index...")
    run_ocr(train_type=train_type)

    documents = load_documents(train_type=train_type)
    if not documents:
        return None

    # Stamp product_type for manual uploads (crawler path should already set per-doc type in loader)
    if train_type:
        for d in documents:
            d.metadata["product_type"] = train_type

    storage_context = init_settings_and_storage()
    print("Created VectorStore Index...")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    chunk_count = len(nodes)

    update_list(storage_context, documents, train_type=train_type)
    cleanup_train_data(train_type=train_type)

    return index, chunk_count


if __name__ == "__main__":
    build_index()
