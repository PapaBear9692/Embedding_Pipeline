# file: 1
import re
from pathlib import Path

from llama_index.core import SimpleDirectoryReader


# ---------- CONFIG ----------
ROOT_DIR = Path(__file__).resolve().parent
PDF_DIR = ROOT_DIR / "data" / "train_data"

# ---------- HELPERS ----------
def normalize_name(filename: str) -> str:
    """Remove leading numeric IDs, normalize and slugify for IDs."""
    base = Path(filename).stem
    base = re.sub(r"^\d+[_\-]*", "", base).strip()
    base = base.lower()
    base = re.sub(r"\s+", "-", base)
    return re.sub(r"[^a-z0-9\-]", "", base)


def extract_product_name(text: str, filename: str) -> str:
    """Try 'Brand name:' in text, fallback to pretty version of filename."""
    match = re.search(r"Brand\s*name\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip()
    return normalize_name(filename).replace("-", " ")


def extract_usage(text: str) -> str:
    """Extract 'Usage:' one-line summary if present."""
    match = re.search(r"Usage\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1).splitlines()[0].strip())
    return ""


# ---------- DOCUMENT LOADER ----------
def load_documents():
    """Load PDFs with SimpleDirectoryReader and enrich metadata."""
    if not PDF_DIR.is_dir():
        raise FileNotFoundError(f"PDF_DIR does not exist: {PDF_DIR}")

    reader = SimpleDirectoryReader(
        input_dir=str(PDF_DIR),
        required_exts=[".pdf"],
        recursive=False,
    )

    docs = reader.load_data()
    if not docs:
        raise ValueError(f"No PDF documents found in {PDF_DIR}")

    for d in docs:
        text = d.text or ""
        file_path = d.metadata.get("file_path", "")
        filename = Path(file_path).name if file_path else "unknown.pdf"

        d.metadata.update(
            {
                "product_name": extract_product_name(text, filename),
                "usage": extract_usage(text),
                "file_name": filename,
                "normalized_name": normalize_name(filename),
            }
        )

    print(f"Loaded {len(docs)} documents from {PDF_DIR}")
    return docs
