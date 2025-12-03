from app_config import DATA_DIR, OUTPUT_FILE

from model.data_loader import (
    create_llm,
    load_documents,
    chunk_documents,
    preprocess_nodes,
    prefix_source_in_text,
    enrich_metadata,
    normalize_tags_and_index,
)

from model.embedder import (
    embed_nodes,
    add_splade_sparse_vectors,
    save_nodes_to_jsonl,
)


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # Create LLM ONCE and reuse it
    llm = create_llm()

    print("Loading documents...")
    documents = load_documents(DATA_DIR)

    print("Chunking documents...")
    nodes = chunk_documents(documents)

    print("Cleaning text...")
    nodes = preprocess_nodes(nodes)

    print("Adding filename prefix to chunks...")
    nodes = prefix_source_in_text(nodes)

    print("Enriching metadata (filter_by + section) with a single LLM call per chunk...")
    nodes = enrich_metadata(nodes, llm)

    print("Normalizing tags and adding doc_id/chunk_index/drug_name/keywords...")
    nodes = normalize_tags_and_index(nodes)

    print("Generating dense embeddings...")
    nodes = embed_nodes(nodes)

    print("Generating SPLADE sparse vectors...")
    nodes = add_splade_sparse_vectors(nodes)

    print(f"Saving nodes to: {OUTPUT_FILE}")
    save_nodes_to_jsonl(nodes, OUTPUT_FILE)

    print("Pipeline complete!")


if __name__ == "__main__":
    main()
