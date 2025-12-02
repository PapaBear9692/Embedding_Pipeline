# llama_data_loader.py
import os
from pathlib import Path
from typing import List
from uuid import uuid4
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import RelatedNodeInfo, NodeRelationship
from llama_index.core import SimpleDirectoryReader

def extract_doctr_text(page: dict) -> str:
    lines = []
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            line_text = " ".join(word.get("value", "") for word in line.get("words", []))
            if line_text.strip():
                lines.append(line_text)
    return "\n".join(lines)

def run_doctr_ocr(path: str) -> List[Document]:
    predictor = ocr_predictor(pretrained=True, assume_straight_pages=True,
                              detect_kwargs={"rotated_bbox": False},
                              reco_kwargs={"beam_size": 5})
    doc = DocumentFile.from_pdf(path)
    results = predictor(doc).export()
    out = []
    for page_idx, page in enumerate(results["pages"]):
        text = extract_doctr_text(page)
        out.append(Document(text, metadata={"source": path, "page": page_idx, "doc_id": str(uuid4())}))
    return out

def load_documents(data_dir: str) -> List[Document]:
    docs = []
    for path in Path(data_dir).rglob("*.pdf"):
        try:
            from llama_index.readers.file import PDFReader
            extracted = PDFReader().load_data(str(path))
            if extracted:
                for d in extracted:
                    if "doc_id" not in d.metadata:
                        d.metadata["doc_id"] = str(uuid4())
                docs.extend(extracted)
                continue
        except Exception:
            pass
        docs.extend(run_doctr_ocr(str(path)))
    for path in Path(data_dir).rglob("*.txt"):
        text = Path(path).read_text(errors="ignore")
        docs.append(Document(text, metadata={"source": str(path), "doc_id": str(uuid4())}))
    return docs

def chunk_documents(documents: List[Document], chunk_size=800, overlap=100):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    for node in nodes:
        node_doc_id = node.metadata.get("doc_id")
        if node_doc_id:
            node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=node_doc_id)
    return nodes
