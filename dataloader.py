import re
from pathlib import Path
from typing import Optional, List

from llama_index.core import SimpleDirectoryReader

# ---------- CONFIG ----------
ROOT_DIR = Path(__file__).resolve().parent
TRAIN_DATA_DIR = ROOT_DIR / "data" / "train_data"

ALLOWED_TYPES = {"pharma", "herbal", "agrovet"}


# ---------- HELPERS ----------
def normalize_name(filename: str) -> str:
    """Remove leading numeric IDs, normalize and slugify for IDs."""
    base = Path(filename).stem
    base = re.sub(r"^\d+[_\-]*", "", base).strip()
    base = base.lower()
    base = re.sub(r"\s+", "-", base)
    return re.sub(r"[^a-z0-9\-]", "", base)


def prettify_filename(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_product_name(text: str, filename: str) -> str:
    """Try 'Brand name:' in text, fallback to a prettified filename."""
    match = re.search(r"Brand\s*name\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip()
    return prettify_filename(filename)


def extract_usage(text: str) -> str:
    """Extract 'Usage:' one-line summary if present."""
    match = re.search(r"Usage\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1).splitlines()[0].strip())
    return ""


def _normalize_type(train_type: Optional[str]) -> Optional[str]:
    if not train_type:
        return None
    t = train_type.strip().lower()
    return t if t in ALLOWED_TYPES else None


def _cap_type(t: str) -> str:
    return t.strip().lower().capitalize()


def _infer_product_type_from_path(file_path: str) -> Optional[str]:
    """
    Infer type from path like .../data/train_data/Pharma/<file>.pdf
    Returns 'pharma' or 'herbal' if detected.
    """
    p = Path(file_path)
    parts = [x.lower() for x in p.parts]
    if "train_data" in parts:
        i = parts.index("train_data")
        if i + 1 < len(parts):
            maybe = parts[i + 1]
            if maybe in ALLOWED_TYPES:
                return maybe
            # also handle capitalized folder names
            maybe2 = maybe.lower()
            if maybe2 in ALLOWED_TYPES:
                return maybe2
    return None


# ---------- DOCUMENT LOADER ----------
def load_documents(train_type: str | None = None):
    """
    Load PDFs/DOCX/TXT with SimpleDirectoryReader and enrich metadata.

    train_type:
      - 'pharma' or 'herbal' or 'agrovet' -> only that folder
      - None -> load all (recursive)
    """
    t = _normalize_type(train_type)

    if t:
        pdf_dir = TRAIN_DATA_DIR / _cap_type(t)
        recursive = False
    else:
        pdf_dir = TRAIN_DATA_DIR
        recursive = True  # so it picks up train_data/Pharma + train_data/Herbal

    print(f"Loading uploaded documents from: {pdf_dir} (recursive={recursive})")

    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"Train data directory does not exist: {pdf_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(pdf_dir),
        required_exts=[".pdf", ".docx", ".txt"],
        recursive=recursive,
    )

    docs = reader.load_data()
    if not docs:
        raise ValueError(f"No documents found in {pdf_dir}")

    print(f"Loaded {len(docs)} documents... enriching metadata")

    for d in docs:
        text = d.text or ""
        file_path = d.metadata.get("file_path", "")
        filename = Path(file_path).name if file_path else "unknown"

        # Determine product type
        if t:
            product_type = t
        else:
            product_type = _infer_product_type_from_path(file_path) or ""

        d.metadata.update(
            {
                "product_name": extract_product_name(text, filename),
                "usage": extract_usage(text),
                "file_name": prettify_filename(filename),
                "normalized_name": normalize_name(filename),
                "product_type": product_type,  # 'pharma' / 'herbal' (or "" if unknown)
            }
        )

    print(f"Loaded {len(docs)} documents from {pdf_dir}")
    return docs
