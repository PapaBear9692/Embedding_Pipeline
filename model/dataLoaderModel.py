import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app_config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIRECTORY

class DataLoader:
    """
    Handles loading all .txt and .pdf files from the data directory
    and splitting them into manageable chunks.
    """
    def __init__(self, data_dir: str = DATA_DIRECTORY):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def load_documents(self) -> List[Document]:
        """Loads all .txt and .pdf files from the specified directory."""
        if not os.path.exists(self.data_dir):
            print(f"Error: Data directory '{self.data_dir}' not found.")
            return []

        print(f"Loading documents from {self.data_dir}...")

        # Load both .pdf and .txt files
        pdf_loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            use_multithreading=True,
            show_progress=True,
        )
        txt_loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            use_multithreading=True,
            show_progress=True,
        )

        documents = pdf_loader.load() + txt_loader.load()
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits loaded documents into smaller chunks."""
        if not documents:
            return []
        
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        return chunks
