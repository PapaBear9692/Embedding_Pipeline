# upsert.py
import json
import time
from pathlib import Path

from pinecone_storage import PineconeStorage


# ---- Retry Helper ----
def retry(fn, retries=3, delay=5, backoff=2):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[Retry {attempt}/{retries}] {e}")
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= backoff


# ---- Main Pipeline ----
def upsert_embedded_chunks(
        input_path="data_cache/embedded_chunks.jsonl",
        log_path="data_cache/uploaded_ids.json",
        batch_size=50,
        max_retries=3):
    print("🚀 Starting Pinecone Upsertion")

    input_file = Path(input_path)
    log_file = Path(log_path)
    log_file.parent.mkdir(exist_ok=True)

    # Load resume log
    uploaded_ids = set()
    if log_file.exists():
        uploaded_ids = set(json.loads(log_file.read_text()))
        print(f"Resuming — already uploaded: {len(uploaded_ids)} chunks")

    # Load embedded chunks
    if not input_file.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    print("Loading embedded chunks...")
    records = [json.loads(line) for line in input_file.read_text().splitlines()]
    print(f"Loaded {len(records)} records")

    # Pinecone setup
    storage = PineconeStorage()
    storage.connect()

    pending_batch = []
    batch_ids = []

    for idx, rec in enumerate(records, start=1):
        if rec["id"] in uploaded_ids:
            continue

        metadata = rec.get("metadata", {})
        metadata["text"] = rec.get("text", "")

        pending_batch.append({
            "id": rec["id"],
            "values": rec["embedding"],
            "metadata": metadata,
        })
        batch_ids.append(rec["id"])

        # If batch is full → upload
        if len(pending_batch) >= batch_size:
            print(f"⬆️ Upserting batch of {len(pending_batch)} vectors...")
            retry(lambda: storage.upsert_vectors(pending_batch), retries=max_retries)
            uploaded_ids.update(batch_ids)

            log_file.write_text(json.dumps(list(uploaded_ids), indent=2))
            pending_batch.clear()
            batch_ids.clear()

    # Final leftover batch
    if pending_batch:
        print(f"⬆️ Upserting final batch ({len(pending_batch)})...")
        retry(lambda: storage.upsert_vectors(pending_batch), retries=max_retries)
        uploaded_ids.update(batch_ids)
        log_file.write_text(json.dumps(list(uploaded_ids), indent=2))

    print(f"🎉 Finished. Total uploaded: {len(uploaded_ids)}")


if __name__ == "__main__":
    upsert_embedded_chunks()
