# metadata_and_upsert_resumable_retry.py
import json
import time
from pathlib import Path
from model.llm_meta_generator import LLMMetaGenerator
from model.storageModel import PineconeStorage

# ---------- Helper: retry wrapper ----------
def retry(func, retries=3, delay=5, backoff=2, exception_type=Exception):
    """
    Retry helper for transient errors.
    """
    for attempt in range(1, retries + 1):
        try:
            return func()
        except exception_type as e:
            print(f"[Retry {attempt}/{retries}] {func.__name__} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
                delay *= backoff
            else:
                print(f" Failed after {retries} retries.")
                raise

# ---------- Main pipeline ----------
def generate_metadata_and_upsert(
    input_path="data_cache/embedded_chunks.jsonl",
    log_path="data_cache/uploaded_ids.json",
    batch_size=5,
    delay=10,
    max_retries=3
):
    print("--- Starting Metadata Generation + Pinecone Upsert (Resumable + Retry) ---")

    input_file = Path(input_path)
    log_file = Path(log_path)
    log_file.parent.mkdir(exist_ok=True)

    # 1. Load uploaded IDs
    uploaded_ids = set()
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            uploaded_ids = set(json.load(f))
        print(f"Resuming from previous run — {len(uploaded_ids)} chunks already uploaded.")

    # 2. Load embedded chunks
    if not input_file.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(records)} embedded chunks.")

    # 3. Initialize components
    meta_generator = LLMMetaGenerator()
    storage = PineconeStorage()
    storage.connect_to_index()

    vectors_to_upsert, uploaded_now = [], []

    # 4. Process each record
    for i, record in enumerate(records, start=1):
        if record["id"] in uploaded_ids:
            continue

        print(f"[{i}/{len(records)}] Generating metadata...")

        # Retry metadata generation
        def generate_meta():
            return meta_generator.generate_metadata(record["text"])

        try:
            llm_metadata = retry(generate_meta, retries=max_retries, delay=5)
        except Exception as e:
            print(f" Skipping chunk due to repeated LLM failure: {e}")
            continue

        metadata = {
            "text": record["text"],
            "source": record.get("source", "unknown"),
            "page": record.get("page", None),
            **llm_metadata,
        }

        vectors_to_upsert.append({
            "id": record["id"],
            "values": record["embedding"],
            "metadata": metadata
        })
        uploaded_now.append(record["id"])

        print(f"  ↳ Topic: {llm_metadata.get('topic', 'N/A')}")
        time.sleep(delay)

        # Batch upsert
        if len(vectors_to_upsert) >= batch_size:
            print(f" Upserting batch of {len(vectors_to_upsert)} vectors...")
            try:
                retry(lambda: storage.upsert_vectors(vectors_to_upsert), retries=max_retries, delay=5)
                uploaded_ids.update(uploaded_now)
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(list(uploaded_ids), f, indent=2)
                vectors_to_upsert.clear()
                uploaded_now.clear()
            except Exception as e:
                print(f" Batch upsert failed repeatedly: {e}")

    # Final flush
    if vectors_to_upsert:
        print(f" Upserting final {len(vectors_to_upsert)} vectors...")
        try:
            retry(lambda: storage.upsert_vectors(vectors_to_upsert), retries=max_retries, delay=5)
            uploaded_ids.update(uploaded_now)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(list(uploaded_ids), f, indent=2)
        except Exception as e:
            print(f" Final upsert failed: {e}")

    print(f" Completed. Total uploaded: {len(uploaded_ids)}")
    print("--- Resumable + Retry Pipeline Finished ---")


if __name__ == "__main__":
    generate_metadata_and_upsert()
