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
    # Show allowed values as proper JSON arrays to the model
    filters_json = json.dumps(FILTERS, ensure_ascii=False)
    sections_json = json.dumps(SECTION_LABELS, ensure_ascii=False)

    # Build simple examples based on your config
    example_filter_nonempty = FILTERS[:2] if len(FILTERS) >= 2 else FILTERS
    example_section_nonempty = SECTION_LABELS[0] if SECTION_LABELS else "other"
    example_section_other = "other" if "other" in SECTION_LABELS else (
        SECTION_LABELS[0] if SECTION_LABELS else "other"
    )

    prompt = (
        "You are a strict JSON generator for a medicine leaflet classification task.\n\n"
        "Given the following text, you must return ONLY a JSON object with this exact shape:\n"
        "{\n"
        '  "filter_by": [...],\n'
        '  "section": "..." \n'
        "}\n\n"
        "Allowed values:\n"
        f"- 'filter_by' values MUST be chosen only from this list: {filters_json}\n"
        f"- 'section' MUST be one of: {sections_json}\n"
        "- If no filter_by category applies, use []\n"
        "- If no section fits well, use \"other\".\n\n"
        "Examples (these are just examples, not answers):\n\n"
        "Example 1:\n"
        "Text: \"Dosage for adults and children over 12 years...\"\n"
        "Output:\n"
        f"{{\n  \"filter_by\": {json.dumps(example_filter_nonempty)},\n"
        f"  \"section\": {json.dumps(example_section_nonempty)}\n}}\n\n"
        "Example 2:\n"
        "Text: \"This leaflet contains general information about the medicine...\"\n"
        "Output:\n"
        f"{{\n  \"filter_by\": [],\n"
        f"  \"section\": {json.dumps(example_section_other)}\n}}\n\n"
        "Now analyze the following text and respond with ONLY the JSON object, "
        "with no explanations, no extra keys, and no markdown code fences.\n\n"
        f"Text:\n{text}\n"
    )

    try:
        response = llm.complete(prompt)
        raw = response.text.strip()

        # --- Handle ```json ... ``` wrappers if model ignores instructions ---
        if raw.startswith("```"):
            # Strip leading ``` or ```json
            first_fence_end = raw.find("```", 3)
            if first_fence_end != -1:
                inner = raw[first_fence_end + 3 :]
                second_fence = inner.rfind("```")
                if second_fence != -1:
                    raw = inner[:second_fence].strip()
                else:
                    raw = inner.strip()

        data = json.loads(raw)

        # --- filter_by processing (keep only allowed values) ---
        fb = data.get("filter_by", [])
        if isinstance(fb, str):
            fb = [fb]
        if not isinstance(fb, list):
            fb = []

        fb = [v for v in fb if v in FILTERS]

        # --- section processing (fallback to 'other') ---
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
