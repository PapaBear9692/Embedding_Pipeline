import os
from typing import List
from multiprocessing import Pool, cpu_count

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
    """
    Safely extract text regardless of DocTR version/structure.
    Supports:
    - page["blocks"][...]["lines"][...]["words"][...]["value"]
    - missing or empty keys
    """
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
# WORKER: RUNS INSIDE MULTIPROCESSING
# ---------------------------------------------------------------
def doctr_ocr_worker(args):
    pdf_path, use_gpu = args

    # Create predictor in worker process
    predictor = ocr_predictor(pretrained=True, assume_straight_pages=False)

    if use_gpu:
        predictor = predictor.to("cuda")

    # Load PDF pages as images
    doc = DocumentFile.from_pdf(pdf_path)

    # Run OCR
    results = predictor(doc).export()

    docs = []

    for i, page in enumerate(results["pages"]):
        text = extract_doctr_text(page)

        docs.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path, "page": i},
            )
        )

    return docs



# ---------------------------------------------------------------
# MAIN DATA LOADER
# ---------------------------------------------------------------
class DataLoader:
    """
    Loads all documents (.txt + .pdf),
    performs DocTR OCR on scanned PDFs,
    and splits docs into chunks.
    """

    def __init__(self, data_dir: str = DATA_DIRECTORY, use_gpu: bool = False):
        self.data_dir = data_dir
        self.use_gpu = use_gpu

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    # -----------------------------------------------------------
    # Detect scanned PDFs (no text extracted via PyPDFLoader)
    # -----------------------------------------------------------
    def _detect_scanned_pdfs(self, pdf_docs: List[Document]) -> List[str]:
        scanned = []
        for d in pdf_docs:
            if not d.page_content or len(d.page_content.strip()) < 20:
                scanned.append(d.metadata["source"])
        return scanned

    # -----------------------------------------------------------
    # Load documents with OCR fallback
    # -----------------------------------------------------------
    def load_documents(self) -> List[Document]:
        if not os.path.exists(self.data_dir):
            print(f"Error: Data directory '{self.data_dir}' not found.")
            return []

        print(f"Loading documents from {self.data_dir}...")

        # Loaders (normal)
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

        # Detect scanned PDFs
        scanned_paths = self._detect_scanned_pdfs(pdf_docs)

        # Add normal PDFs
        for d in pdf_docs:
            if d.metadata["source"] not in scanned_paths:
                documents.append(d)

        # -------------------------------------------------------
        # Parallel OCR for scanned PDFs
        # -------------------------------------------------------
        if scanned_paths:
            print(f"\nDetected {len(scanned_paths)} scanned PDFs → running DocTR OCR...")
            print(f"GPU Mode: {'ON (CUDA)' if self.use_gpu else 'OFF (CPU)'}")

            worker_count = max(cpu_count() - 1, 1)
            print(f"Using {worker_count} parallel OCR workers...\n")

            with Pool(worker_count) as pool:
                results = list(
                    tqdm(
                        pool.imap(
                            doctr_ocr_worker,
                            [(path, self.use_gpu) for path in scanned_paths],
                        ),
                        total=len(scanned_paths),
                        desc="OCR Progress",
                        ncols=90,
                    )
                )

            # Flatten list
            for doc_list in results:
                documents.extend(doc_list)

        # Add TXT files
        documents.extend(txt_docs)

        return documents

    # -----------------------------------------------------------
    # Split into chunks (same as before)
    # -----------------------------------------------------------
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        print(f"Splitting {len(documents)} documents into chunks...")
        return self.text_splitter.split_documents(documents)
