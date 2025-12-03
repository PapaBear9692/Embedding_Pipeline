# upsert.py

import os

from pinecone import Pinecone

from app_config import (
    OUTPUT_FILE,
    PINECONE_API_KEY,
    BATCH_SIZE,
)

from model.storage import (
    get_or_create_index,
    upsert_from_jsonl,
    DEFAULT_CHECKPOINT_FILE,
)


def main():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in app_config.")

    if not os.path.exists(OUTPUT_FILE):
        raise FileNotFoundError(f"JSONL file not found at: {OUTPUT_FILE}")

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index = get_or_create_index(pc, OUTPUT_FILE)

    print(f"Starting upsert from {OUTPUT_FILE}")
    upsert_from_jsonl(
        index=index,
        jsonl_path=OUTPUT_FILE,
        checkpoint_path=DEFAULT_CHECKPOINT_FILE,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
