import os
from typing import List

from tqdm import tqdm

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from app_config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIRECTORY


# ---------------------------------------------------------------
# SAFE TEXT EXTRACTION FOR ALL DOCTR FORMATS
# ---------------------------------------------------------------
def extract_doctr_text(page: dict) -> str:
    lines = []
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            line_text = " ".join(
                word.get("value", "")
                for word in line.get("words", [])
                if "value" in word
            )
            if line_text.strip():
                lines.append(line_text)
    return "\n".join(lines)


# ---------------------------------------------------------------
# OCR FUNCTION (SEQUENTIAL)
# ---------------------------------------------------------------
def run_doctr_ocr(pdf_path: str, use_gpu: bool) -> List[Document]:
    predictor = ocr_predictor(pretrained=True, assume_straight_pages=False)
    if use_gpu:
        predictor = predictor.to("cuda")

    doc = DocumentFile.from_pdf(pdf_path)
    results = predictor(doc).export()

    docs = []
    for i, page in enumerate(results["pages"]):
        text = extract_doctr_text(page)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path, "page": i}
            )
        )
    return docs


# ---------------------------------------------------------------
# MAIN DATA LOADER
# ---------------------------------------------------------------
class DataLoader:

    def __init__(self, data_dir: str = DATA_DIRECTORY, use_gpu: bool = False):
        self.data_dir = data_dir
        self.use_gpu = use_gpu
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def _detect_scanned_pdfs(self, pdf_docs: List[Document]) -> List[str]:
        scanned = []
        for d in pdf_docs:
            if not d.page_content or len(d.page_content.strip()) < 20:
                scanned.append(d.metadata["source"])
        return scanned

    def load_documents(self) -> List[Document]:
        if not os.path.exists(self.data_dir):
            print(f"Error: Data directory '{self.data_dir}' not found.")
            return []

        print(f"Loading documents from {self.data_dir}...")

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

        pdf_docs = pdf_loader.load()
        txt_docs = txt_loader.load()

        documents = []

        scanned_paths = self._detect_scanned_pdfs(pdf_docs)

        # Add normal PDFs
        for d in pdf_docs:
            if d.metadata["source"] not in scanned_paths:
                documents.append(d)

        # Sequential OCR for scanned PDFs
        if scanned_paths:
            print(f"\nDetected {len(scanned_paths)} scanned PDFs → running DocTR OCR...")
            print(f"GPU Mode: {'ON (CUDA)' if self.use_gpu else 'OFF (CPU)'}\n")
            for pdf_path in tqdm(scanned_paths, desc="OCR Progress", ncols=90):
                ocr_docs = run_doctr_ocr(pdf_path, self.use_gpu)
                documents.extend(ocr_docs)

        # Add TXT files
        documents.extend(txt_docs)

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        print(f"Splitting {len(documents)} documents into chunks...")
        return self.text_splitter.split_documents(documents)
