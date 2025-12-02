# llama_data_loader.py
import os
from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


# ---- OCR helper ----
def extract_doctr_text(page: dict) -> str:
    lines = []
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            text = " ".join(word.get("value", "") for word in line.get("words", []))
            if text.strip():
                lines.append(text)
    return "\n".join(lines)


def run_doctr_ocr(pdf_path: str) -> List[Document]:
    predictor = ocr_predictor(pretrained=True, assume_straight_pages=False)
    doc = DocumentFile.from_pdf(pdf_path)
    results = predictor(doc).export()

    docs = []
    for i, page in enumerate(results["pages"]):
        text = extract_doctr_text(page)
        docs.append(
            Document(text, metadata={"source": pdf_path, "page": i})
        )
    return docs


# ---- Main Loader ----
def load_documents(data_dir: str) -> List[Document]:
    docs = []

    for path in Path(data_dir).rglob("*.pdf"):
        try:
            from llama_index.readers.file import PDFReader
            parsed = PDFReader().load_data(str(path))
            if parsed:
                docs.extend(parsed)
                continue
        except:
            pass

        # If PDFReader fails, fallback to OCR
        ocr_docs = run_doctr_ocr(str(path))
        docs.extend(ocr_docs)

    for path in Path(data_dir).rglob("*.txt"):
        text = Path(path).read_text(errors="ignore")
        docs.append(
            Document(text, metadata={"source": str(path)})
        )

    return docs


# ---- Chunking (SentenceSplitter) ----
def chunk_documents(documents: List[Document], chunk_size: int, overlap: int):
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.get_nodes_from_documents(documents)
