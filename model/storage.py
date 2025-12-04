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
    BATCH_SIZE,
)


# -----------------------------------------------------------------------
# Checkpoint config
# -----------------------------------------------------------------------

# Checkpoint file lives next to the OUTPUT_FILE by default
DEFAULT_CHECKPOINT_FILE = os.path.join(
    os.path.dirname(OUTPUT_FILE),
    "upsert_checkpoint.json",
)


# -----------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT_FILE) -> int:

    if not os.path.exists(checkpoint_path):
        return 0

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("last_line", 0))
    except Exception:
        # If checkpoint is corrupted, start from scratch
        return 0


def save_checkpoint(last_record_idx: int, checkpoint_path: str = DEFAULT_CHECKPOINT_FILE) -> None:

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    data = {"last_line": int(last_record_idx)}
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------------------------------------------------
# Index inspection & creation
# -----------------------------------------------------------------------

def inspect_first_vector(jsonl_path: str) -> Tuple[int, bool]:

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            embedding = record.get("embedding") or []
            sparse_values = record.get("sparse_values")
            dim = len(embedding)
            has_sparse = bool(sparse_values)

            if dim == 0:
                raise ValueError(
                    "First non-empty record has no 'embedding' or zero dimension."
                )

            return dim, has_sparse

    raise ValueError("JSONL file appears to be empty or malformed.")


def get_or_create_index(pc: Pinecone, jsonl_path: str):

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
        # deletion_protection="disabled",  # optional
    )

    print("Waiting for index to be ready...")
    # For serverless, readiness is usually quick; we can just proceed.
    return pc.Index(PINECONE_INDEX_NAME)


# -----------------------------------------------------------------------
# JSONL reading & upsert logic
# -----------------------------------------------------------------------

def build_vector_from_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:

    _id = record.get("id")
    embedding = record.get("embedding")
    metadata = record.get("metadata", {}) or {}
    sparse_values = record.get("sparse_values")

    # Basic validation
    if not _id or not isinstance(embedding, list) or len(embedding) == 0:
        return None

    vector: Dict[str, Any] = {
        "id": _id,
        "values": embedding,
        "metadata": metadata,
    }

    if sparse_values:
        vector["sparse_values"] = sparse_values

    return vector


def upsert_from_jsonl(
    index,
    jsonl_path: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT_FILE,
    batch_size: int = BATCH_SIZE,
) -> None:

    last_record_idx = load_checkpoint(checkpoint_path)
    print(f"Resuming from record index: {last_record_idx}")

    batch: List[Dict[str, Any]] = []
    current_record_idx = -1

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for current_record_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Skip already processed records
            if current_record_idx <= last_record_idx:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON at record {current_record_idx}: {e}")
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
