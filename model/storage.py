# storage.py

import os
import json
from typing import Optional, Tuple, Dict, Any, List

from pinecone import Pinecone, ServerlessSpec

from app_config import (
    OUTPUT_FILE,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    BATCH_SIZE
)



# checkpoint file lives next to the OUTPUT_FILE by default
DEFAULT_CHECKPOINT_FILE = os.path.join(
    os.path.dirname(OUTPUT_FILE),
    "upsert_checkpoint.json"
)

def iter_json_records(jsonl_path: str):
    """
    Iterate over JSON objects in a file where each object is pretty-printed
    (indent=2) and separated by one or more blank lines.

    This matches the format produced by save_nodes_to_jsonl(), which writes:

        json.dump(..., indent=2)
        f.write("\\n\\n")
    """
    buffer_lines = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            # blank line = end of one JSON object
            if not line.strip():
                if buffer_lines:
                    raw = "".join(buffer_lines)
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError as e:
                        print(f"[WARN] Skipping malformed JSON block: {e}")
                    buffer_lines = []
                continue

            buffer_lines.append(line)

    # last block (no trailing blank line)
    if buffer_lines:
        raw = "".join(buffer_lines)
        try:
            yield json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping malformed JSON block at EOF: {e}")


# -----------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------
def load_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT_FILE) -> int:
    """
    Load the last processed line index from checkpoint JSON.
    Return 0 if no checkpoint exists.
    """
    if not os.path.exists(checkpoint_path):
        return 0

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("last_line", 0))
    except Exception:
        # If checkpoint is corrupted, start from scratch
        return 0


def save_checkpoint(last_line: int, checkpoint_path: str = DEFAULT_CHECKPOINT_FILE) -> None:
    """
    Save the last processed line index into checkpoint JSON.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    data = {"last_line": int(last_line)}
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def inspect_first_vector(jsonl_path: str) -> Tuple[int, bool]:
    """
    Inspect the JSON file to get:
    - dense embedding dimension
    - whether sparse_values are present (for sanity check / logging)
    """
    for record in iter_json_records(jsonl_path):
        embedding = record.get("embedding") or []
        sparse_values = record.get("sparse_values")
        dim = len(embedding)
        has_sparse = bool(sparse_values)
        if dim == 0:
            raise ValueError(
                "First non-empty record has no 'embedding' or zero dimension."
            )
        return dim, has_sparse

    raise ValueError("JSON file appears to be empty or malformed.")


def get_or_create_index(pc: Pinecone, jsonl_path: str):
    """
    Get an existing Pinecone index, or create it if it doesn't exist.
    Automatically detects dense embedding dimension from the JSONL file.
    For serverless indexes, all metadata is indexed by default and can be used
    for filtering (no metadata_config needed).
    """
    # Check if index already exists
    existing = {idx["name"] for idx in pc.list_indexes()}

    if PINECONE_INDEX_NAME in existing:
        print(f"Using existing index: {PINECONE_INDEX_NAME}")
        return pc.Index(PINECONE_INDEX_NAME)

    # Otherwise, create a new index
    print(f"Index '{PINECONE_INDEX_NAME}' does not exist. Creating...")

    # Detect dimension + sparse presence for logging
    dim, has_sparse = inspect_first_vector(jsonl_path)
    print(f"- Detected dense dimension: {dim}")
    print(f"- Sparse values present: {has_sparse}")

    # Create serverless index (hybrid-ready; sparse just works at upsert/query)
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=dim,
        metric="dotproduct",  # good for transformer embeddings + hybrid
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
        # deletion_protection="disabled",  # optional, if you want to set it
    )

    print("Waiting for index to be ready...")
    # For serverless, readiness is usually quick; we can just proceed.
    return pc.Index(PINECONE_INDEX_NAME)

# -----------------------------------------------------------------------
# JSONL reading & upsert logic
# -----------------------------------------------------------------------
def build_vector_from_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a JSONL record into a Pinecone upsert vector dict.

    Expected record format (from save_nodes_to_jsonl):
    {
        "id": str,
        "text": str,
        "metadata": {...},
        "embedding": [...],
        "sparse_values": {
            "indices": [...],
            "values": [...]
        }
    }
    """
    _id = record.get("id")
    embedding = record.get("embedding")
    metadata = record.get("metadata", {}) or {}
    sparse_values = record.get("sparse_values")

    if not _id or not isinstance(embedding, list) or len(embedding) == 0:
        # skip malformed records
        return None

    # remove redundant sparse_values from metadata if present
    metadata.pop("sparse_values", None)

    vector = {
        "id": _id,
        "values": embedding,
        "metadata": metadata,
    }

    if sparse_values:
        # Pinecone hybrid: use 'sparse_values' field with {"indices": [...], "values": [...]}
        vector["sparse_values"] = sparse_values

    return vector


def upsert_from_jsonl(
    index,
    jsonl_path: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT_FILE,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Stream through the JSON file, resuming from the last checkpoint,
    and upsert vectors to Pinecone in batches.

    The checkpoint stores the index of the last successfully processed record
    (0-based), NOT the file line number.
    """
    last_record_idx = load_checkpoint(checkpoint_path)
    print(f"Resuming from record index: {last_record_idx}")

    batch: List[Dict[str, Any]] = []
    current_record_idx = -1

    for current_record_idx, record in enumerate(iter_json_records(jsonl_path)):
        # Skip already processed records
        if current_record_idx <= last_record_idx:
            continue

        vector = build_vector_from_record(record)
        if vector is None:
            print(
                f"[WARN] Skipping record with missing id/embedding at index "
                f"{current_record_idx}"
            )
            continue

        batch.append(vector)

        # If batch is full, upsert and save checkpoint
        if len(batch) >= batch_size:
            print(
                f"Upserting batch ending at record {current_record_idx} "
                f"(size={len(batch)})..."
            )
            try:
                index.upsert(vectors=batch)
            except Exception as e:
                print(f"[ERROR] Upsert failed at record {current_record_idx}: {e}")
                print(
                    "You can rerun the script to resume from "
                    "the last successful checkpoint."
                )
                return

            save_checkpoint(current_record_idx, checkpoint_path)
            batch = []

    # Flush any remaining vectors
    if batch:
        print(
            f"Upserting final batch ending at record {current_record_idx} "
            f"(size={len(batch)})..."
        )
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"[ERROR] Final upsert failed at record {current_record_idx}: {e}")
            print(
                "You can rerun the script to resume from "
                "the last successful checkpoint."
            )
            return

        save_checkpoint(current_record_idx, checkpoint_path)

    print("Upsert complete. All available records processed.")
