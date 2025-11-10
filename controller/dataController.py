import uuid
from pathlib import Path
from model.dataLoaderModel import DataLoader
from model.embedderModel import get_embedder
from model.storageModel import PineconeStorage
from model.llm_meta_generator import LLMMetaGenerator
from app_config import EMBEDDER_PROVIDER
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import time

def run_indexing_pipeline():

    print("--- Starting Data Indexing Pipeline ---")

    try:
        # 1. Initialize Components
        print("# Initializing components..............")
        data_loader = DataLoader()
        print("Data Loader loaded...")
        embedder = get_embedder(EMBEDDER_PROVIDER)
        print("Embedder loaded...")
        storage = PineconeStorage()
        print("Storage loaded...")
        meta_generator = LLMMetaGenerator()
        print("Meta Generator loaded...")

        # 2. Load Documents
        print(f"# Loading documents from '{data_loader.data_dir}'..........")
        documents = []
        for file in Path(data_loader.data_dir).rglob("*"):
            if file.suffix.lower() == ".pdf":
                documents.extend(PyPDFLoader(str(file)).load())
            elif file.suffix.lower() == ".txt":
                documents.extend(TextLoader(str(file)).load())

        if not documents:
            print(f"No documents found in '{data_loader.data_dir}'. Aborting....")
            return

        print(f"Loaded {len(documents)} documents/pages.")

        # 3. Split Documents into Chunks
        print("# Splitting documents into chunks..........")
        chunks = data_loader.split_documents(documents)
        if not chunks:
            print("No chunks were created. Aborting...")
            return
        print(f"Split into {len(chunks)} chunks...")

        # 4. Ensure Pinecone Index Exists
        dimension = embedder.get_dimension()
        print(f"# Checking if Pinecone index exists (dimension: {dimension})..........")
        storage.create_index_if_not_exists(dimension)

        # 5. Embed Chunks
        print(f"# Embedding {len(chunks)} chunks.......... (this may take a while)")
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.embed_documents(chunk_texts)
        print("Embedding complete.....")

        # 6. Generate Metadata via LLM
        print("# Generating LLM-based metadata for each chunk..........")
        vectors_to_upsert = []

        for chunk, embedding in zip(chunks, embeddings):
            vector_id = str(uuid.uuid4())

            # Generate contextual metadata
            llm_metadata = meta_generator.generate_metadata(chunk.page_content)
            print(f"{llm_metadata}/n")    
            time.sleep(10)

            # Combine default + LLM metadata
            metadata = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", None),
                **llm_metadata,
            }

            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

        # 7. Upload to Pinecone
        print(f"# Upserting {len(vectors_to_upsert)} vectors to Pinecone..........")
        storage.upsert_vectors(vectors_to_upsert)

        print("--- Data Indexing Pipeline Finished Successfully ---")

    except Exception as e:
        print(f"FATAL ERROR during indexing: {e}")
        import traceback
        traceback.print_exc()
        print("--- Data Indexing Pipeline Aborted ---")