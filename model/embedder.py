import os
import json

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone_text.sparse import SpladeEncoder

from app_config import EMBEDDER_MODELS, EMBEDDER_PROVIDER


# -----------------------------
# Step 4a: Dense Embeddings
# -----------------------------
def embed_nodes(nodes, model_name=None):
    if model_name is None:
        model_name = EMBEDDER_MODELS[EMBEDDER_PROVIDER]

    embedder = HuggingFaceEmbedding(model_name=model_name)
    for node in nodes:
        node.embedding = embedder.get_text_embedding(node.text)
    return nodes


# -----------------------------
# Step 4b: SPLADE Sparse Embeddings (NEW)
# -----------------------------
def add_splade_sparse_vectors(nodes):
    """
    Generate SPLADE sparse vectors for each node.text and attach them under
    metadata['sparse_values'] so they can be saved to JSONL and later upserted
    into Pinecone.
    """
    splade = SpladeEncoder()

    texts = [node.text for node in nodes]
    sparse_vectors = splade.encode_documents(texts)
    # sparse_vectors is a list of dicts: {"indices": [...], "values": [...]}

    for node, sv in zip(nodes, sparse_vectors):
        # Put sparse vector into metadata (Pydantic-safe)
        node.metadata["sparse_values"] = sv

    return nodes


# -----------------------------
# Step 5: Save nodes (JSONL)
# -----------------------------
def save_nodes_to_jsonl(nodes, output_file: str):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for node in nodes:
            metadata = dict(node.metadata or {})
            metadata["text"] = node.text
            sparse_values = getattr(node, "sparse_values", None) or metadata.get("sparse_values")
            metadata.pop("sparse_values", None)
            record = {
                "id": node.node_id,
                "embedding": node.embedding,
                "metadata": metadata,
            }

            if sparse_values is not None:
                record["sparse_values"] = sparse_values
            
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
