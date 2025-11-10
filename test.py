import os
from pinecone import Pinecone
from app_config import EMBEDDER_MODELS, EMBEDDING_DIMENSIONS, OPENAI_API_KEY
from model.embedderModel import get_embedder

def main():
    # Initialize embedder
    provider = input("Enter embedder provider (default / PubMedBert / openai): ").strip() or "PubMedBert"
    embedder = get_embedder(provider)

    # Initialize Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("Missing Pinecone API key. Set PINECONE_API_KEY in your environment.")
    
    INDEX_NAME = "sqbot-index"  # change this to your Pinecone index name
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    print("\nPinecone search app ready! Type 'exit' to quit.\n")

    while True:
        query = input("Enter query: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Embed query
        query_vector = embedder.embed_query(query)

        # Query Pinecone
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        print("\nTop matching chunks:")
        if not results['matches']:
            print("No matches found.")
        else:
            for match in results['matches']:
                meta = match.get("metadata", {})
                snippet = meta.get("text", "")[:120] + "..." if "text" in meta else ""
                print(f"ID: {match['id']}  |  Score: {match['score']:.4f}")
                if snippet:
                    print(f"Snippet: {snippet}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()
