# llama_data_loader.py
import re
import os
from pathlib import Path
from typing import List
from uuid import uuid4

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import RelatedNodeInfo, NodeRelationship


PDF_DIR = Path("data/pdfs")  # same style as your new loader


# ---------- Extractors (same behavior as your new code) ----------
def extract_product_name(text: str, filename: str):
    first_line = text.strip().split("\n")[0]
    if first_line and len(first_line) < 120:
        return first_line
    return Path(filename).stem


def extract_usage(text: str):
    match = re.search(r"(Indication|Uses?|Usage)[\s:\-]+(.{10,300})",
                      text, re.IGNORECASE)
    return match.group(2).strip() if match else None


def normalize_name(filename: str):
    name = Path(filename).stem.lower()
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]+", "", name)
    return name


# ---------- DOCTR OCR ----------
def extract_doctr_text(page: dict) -> str:
    lines = []
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            text_line = " ".join(word.get("value", "") for word in line.get("words", []))
            if text_line.strip():
                lines.append(text_line)
    return "\n".join(lines)


def run_doctr_ocr(path: str) -> List[Document]:
    predictor = ocr_predictor(
        pretrained=True,
        assume_straight_pages=True,
        detect_kwargs={"rotated_bbox": False},
        reco_kwargs={"beam_size": 5},
    )

    doc = DocumentFile.from_pdf(path)
    results = predictor(doc).export()
    out = []

    for page_idx, page in enumerate(results["pages"]):
        text = extract_doctr_text(page)

        out.append(
            Document(
                text,
                metadata={
                    "source": path,
                    "page": page_idx,
                    "file_name": Path(path).name
                },
                doc_id=str(uuid4()),
            )
        )

    return out


# ---------- Loader (updated to use SimpleDirectoryReader first) ----------
def load_documents(data_dir: str = str(PDF_DIR)) -> List[Document]:
    documents = []

    # ---------- FIRST TRY STANDARD LOADER ----------
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        required_exts=[".pdf"],
        recursive=False,
    )

    try:
        docs = reader.load_data()
        for d in docs:
            text = d.text or ""
            file_path = d.metadata.get("file_path", "")
            filename = Path(file_path).name if file_path else "unknown.pdf"

            # add same metadata fields as your new loader
            d.metadata.update(
                {
                    "product_name": extract_product_name(text, filename),
                    "usage": extract_usage(text),
                    "file_name": filename,
                    "normalized_name": normalize_name(filename),
                }
            )

        documents.extend(docs)

    except Exception:
        pass  # if reader fails → continue to OCR fallback

    # ---------- OCR FOR SCANNED PDFs ----------
    for pdf_path in Path(data_dir).glob("*.pdf"):
        # Skip already loaded PDFs (SimpleDirectoryReader succeeded)
        if any(str(pdf_path) == d.metadata.get("file_path") for d in documents):
            continue

        # OCR fallback
        ocr_docs = run_doctr_ocr(str(pdf_path))
        for d in ocr_docs:
            text = d.text
            filename = Path(pdf_path).name

            d.metadata.update(
                {
                    "product_name": extract_product_name(text, filename),
                    "usage": extract_usage(text),
                    "file_name": filename,
                    "normalized_name": normalize_name(filename),
                }
            )

        documents.extend(ocr_docs)

    # ---------- Load plain .txt files ----------
    for txt_path in Path(data_dir).glob("*.txt"):
        text = Path(txt_path).read_text(errors="ignore")
        filename = txt_path.name

        documents.append(
            Document(
                text,
                metadata={
                    "source": str(txt_path),
                    "file_name": filename,
                    "product_name": extract_product_name(text, filename),
                    "usage": extract_usage(text),
                    "normalized_name": normalize_name(filename),
                },
                doc_id=str(uuid4()),
            )
        )

    return documents


# ---------- Chunking with Node Relationships ----------
def chunk_documents(documents: List[Document], chunk_size=800, overlap=100):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes:
        node_doc_id = node.metadata.get("doc_id")
        if node_doc_id:
            node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=node_doc_id
            )

    return nodes
