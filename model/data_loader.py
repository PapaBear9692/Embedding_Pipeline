import os
import re
import json
from time import sleep
from collections import defaultdict

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.file import PDFReader

from app_config import (
    DATA_DIR,
    FILTERS,
    GOOGLE_API_KEY,
    LLM_MODELS,
    LLM_PROVIDER,
    CHUNK_SIZE,
    OVERLAP,
    SECTION_LABELS,
)


# -----------------------------
# LLM factory
# -----------------------------
def create_llm() -> GoogleGenAI:
    """Create and return a configured GoogleGenAI LLM instance."""
    return GoogleGenAI(
        model=LLM_MODELS[LLM_PROVIDER],
        api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )


# -----------------------------
# Step 0: Text cleaning
# -----------------------------
def clean_text(text: str) -> str:
    """Normalize leaflet text for better embeddings and BM25."""
    # Replace funky bullets
    text = text.replace("", "-").replace("•", "-")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Turn hard-wrapped lines into spaces (basic)
    text = re.sub(r" ?\n ?", " ", text)

    return text.strip()


def preprocess_nodes(nodes):
    for node in nodes:
        node.text = clean_text(node.text)
    return nodes


# -----------------------------
# Step 1: Load PDF documents safely
# -----------------------------
def load_documents(folder_path: str = DATA_DIR):
    reader = SimpleDirectoryReader(
        input_dir=folder_path,
        required_exts=[".pdf"],
        file_extractor={".pdf": PDFReader()},
        recursive=False,
    )
    return reader.load_data()


# -----------------------------
# Step 2: Chunk documents
# -----------------------------
def chunk_documents(documents):
    parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
    )
    return parser.get_nodes_from_documents(documents)


def prefix_source_in_text(nodes):
    """Prefix each chunk text with [filename - page X] for better traceability."""
    for node in nodes:
        fname = node.metadata.get("file_name", "Unknown")
        page = node.metadata.get("page_label")
        prefix = f"[{fname}"
        if page:
            prefix += f" - page {page}"
        prefix += "] "

        if not node.text.startswith(prefix):
            node.text = prefix + node.text
    return nodes


# -----------------------------
# Step 3: Single LLM metadata extraction (filter_by + section)
# -----------------------------
def generate_metadata(llm: GoogleGenAI, text: str) -> dict:
    prompt = (
        "You are a strict JSON generator for a medicine leaflet classification task.\n\n"
        "Given the following text, you must return ONLY a JSON object with this exact shape:\n"
        "{\n"
        '  "filter_by": [...],\n'
        '  "section": "..." \n'
        "}\n\n"
        "Rules:\n"
        f"- 'filter_by' must be a JSON array of zero or more values from this allowed list: {FILTERS}\n"
        f"- 'section' must be ONE string from this allowed list: {SECTION_LABELS}\n"
        "- If no filter_by category applies, use []\n"
        "- If no section fits well, use 'other'\n\n"
        "Respond with ONLY the JSON object, no extra text.\n\n"
        f"Text:\n{text}\n"
    )

    try:
        response = llm.complete(prompt)
        raw = response.text

        data = json.loads(raw)

        # --- filter_by processing ---
        fb = data.get("filter_by", [])
        if isinstance(fb, str):
            fb = [fb]
        if not isinstance(fb, list):
            fb = []

        # keep only allowed filter values
        fb = [v for v in fb if v in FILTERS]

        # --- section processing ---
        section = data.get("section", "other")
        if not isinstance(section, str) or section not in SECTION_LABELS:
            section = "other"

    except Exception:
        fb = []
        section = "other"

    return {
        "filter_by": fb,
        "section": section,
    }


def enrich_metadata(nodes, llm: GoogleGenAI):
    """Enrich each node's metadata using a single LLM call per chunk."""
    for node in nodes:
        meta = generate_metadata(llm, node.text)
        node.metadata.update(meta)
        sleep(0.5)  # small delay to avoid rate limits
    return nodes


# -----------------------------
# Step 3b: Normalize tags & add doc_id / chunk_index / drug_name / keywords
# -----------------------------
def normalize_tags_and_index(nodes):
    counters = defaultdict(int)

    for node in nodes:
        # --- doc_id from filename ---
        fname = node.metadata.get("file_name", "unknown")
        base_name = os.path.basename(fname)
        doc_id, _ = os.path.splitext(base_name)
        node.metadata["doc_id"] = doc_id

        # --- chunk_index per doc ---
        chunk_idx = counters[doc_id]
        counters[doc_id] += 1
        node.metadata["chunk_index"] = chunk_idx

        # --- drug_name (simple: same as doc_id) ---
        node.metadata["drug_name"] = doc_id

        # --- normalize filter_by tags ---
        tags = node.metadata.get("filter_by", [])
        if isinstance(tags, str):
            tags = [tags]

        if isinstance(tags, list):
            normed = {
                str(t).strip().lower()
                for t in tags
                if isinstance(t, (str, bytes))
            }
            filter_by = sorted(normed)
        else:
            filter_by = []

        node.metadata["filter_by"] = filter_by

        # --- keywords for BM25 / hybrid search ---
        keywords = set(filter_by)

        section = node.metadata.get("section")
        if isinstance(section, str) and section:
            keywords.add(section.lower())

        # include doc_id / drug_name for lexical matches
        keywords.add(doc_id.lower())

        node.metadata["keywords"] = sorted(keywords)

    return nodes
