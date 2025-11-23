# embed_and_save.py
import uuid
import json
from pathlib import Path
from model.dataLoaderModel import DataLoader
from model.embedderModel import get_embedder
from app_config import EMBEDDER_PROVIDER

def embed_and_save(output_path="data_cache/embedded_chunks.jsonl"):
    print("--- Starting Embedding Extraction Pipeline ---")

    # 1. Initialize components
    data_loader = DataLoader()
    embedder = get_embedder(EMBEDDER_PROVIDER)
    print(f"Embedder loaded ({EMBEDDER_PROVIDER})")

    # 2. Load documents
    documents = data_loader.load_documents()
    if not documents:
        print("No documents found. Aborting...")
        return

    print(f"Loaded {len(documents)} documents.")

    # 3. Split documents into chunks
    chunks = data_loader.split_documents(documents)
    if not chunks:
        print("No chunks created. Aborting...")
        return

    print(f"Split into {len(chunks)} chunks...")

    # 4. Embed chunks
    print(f"Embedding {len(chunks)} chunks... This may take a while.")
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed_documents(chunk_texts)
    print("Embedding complete.")

    # 5. Save to JSONL
    Path("data_cache").mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk, emb in zip(chunks, embeddings):

            source_path = chunk.metadata.get("source", "unknown")
            pdf_name = Path(source_path).name  # e.g., "drugA.pdf"
            pdf_name = pdf_name.replace("_sup_", "").replace(".pdf", "")

            # Prepend filename to the text content
            combined_text = f"[{pdf_name}]\n{chunk.page_content}"

            record = {
                "id": str(uuid.uuid4()),
                "text": combined_text,               
                "embedding": emb,
                "source": source_path,
                "page": chunk.metadata.get("page", None),
                "Drug Name": pdf_name
            }

            f.write(json.dumps(record) + "\n")


    print(f"✅ Saved {len(chunks)} embedded chunks to {output_path}")
    print("--- Embedding Extraction Complete ---")


if __name__ == "__main__":
    embed_and_save()
