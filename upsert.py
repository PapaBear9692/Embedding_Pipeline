# upsert.py
import json
import time
from pathlib import Path

from pinecone_storage import PineconeStorage


# ---------- Retry Helper ----------
def retry(fn, retries=3, delay=5, backoff=2):
    for i in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[Retry {i}/{retries}] {e}")
            if i == retries:
                raise
            time.sleep(delay)
            delay *= backoff


# ---------- Hybrid Upsert Pipeline ----------
def hybrid_upsert(
    input_path="data_cache/embedded_nodes_hybrid.jsonl",
    log_path="data_cache/uploaded_hybrid_ids.json",
    batch_size=50,
    max_retries=3
):
    print("🚀 Starting Hybrid Dense + Sparse Upsert...")

    input_file = Path(input_path)
    log_file = Path(log_path)
    log_file.parent.mkdir(exist_ok=True)

    # Load resume log
    uploaded_ids = set()
    if log_file.exists():
        uploaded_ids = set(json.loads(log_file.read_text()))
        print(f"Resuming — already uploaded: {len(uploaded_ids)} chunks")

    # Load embedded hybrid chunks
    if not input_file.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    print("📌 Loading hybrid-embedded records...")
    records = [json.loads(line) for line in input_file.read_text().splitlines()]
    print(f"Loaded {len(records)} nodes")

    # Pinecone connection
    storage = PineconeStorage()
    storage.connect()

    pending = []
    pending_ids = []

    for rec in records:
        rid = rec["id"]
        if rid in uploaded_ids:
            continue

        dense_vec = rec["dense"]
        sparse_vec = rec["sparse"]
        metadata = rec["metadata"]

        vector_obj = {
            "id": rid,
            "values": dense_vec,
            "sparse_values": sparse_vec,
            "metadata": metadata,
        }

        pending.append(vector_obj)
        pending_ids.append(rid)

        if len(pending) >= batch_size:
            print(f"⬆️ Upserting hybrid batch of {len(pending)}...")
            retry(lambda: storage.upsert_batch(pending), retries=max_retries)

            uploaded_ids.update(pending_ids)
            log_file.write_text(json.dumps(list(uploaded_ids), indent=2))

            pending.clear()
            pending_ids.clear()

    # Final leftover batch
    if pending:
        print(f"⬆️ Upserting final hybrid batch ({len(pending)})...")
        retry(lambda: storage.upsert_batch(pending), retries=max_retries)

        uploaded_ids.update(pending_ids)
        log_file.write_text(json.dumps(list(uploaded_ids), indent=2))

    print(f"🎉 Hybrid Upsert Complete! Total uploaded: {len(uploaded_ids)}")


if __name__ == "__main__":
    hybrid_upsert()
