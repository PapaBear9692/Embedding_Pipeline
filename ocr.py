# ocr.py
"""
Layout Parser JSON-first pipeline:
  1) Read all PDFs from ./ocr/
  2) Send each PDF to Google Document AI (Layout Parser processor)
  3) Convert OCR output to JSON and save: ./data/<name>.layout.json
  4) Build a readable PDF from JSON (not from document.text): ./data/<name>.ocr.pdf

Run:
  python ocr.py

Requirements:
  pip install google-cloud-documentai python-dotenv reportlab

.env (same folder as this file):
  GCP_PROJECT_ID=...
  GCP_DOCAI_PROCESSOR_ID=...
  GCP_DOCAI_LOCATION=us or eu
  GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json

Folders:
  ./ocr/   input PDFs
  ./data/  output PDFs + JSON
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1
from google.protobuf.json_format import MessageToDict

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Preformatted


# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

IN_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "data" / "train_data"


# -----------------------------
# PDF formatting helpers
# -----------------------------
_HEADING_RE = re.compile(r"^\s*(?:[A-Z][A-Z0-9 \-/&(),.%]{3,}|\d+(?:\.\d+)*\s+\S.+|.+:\s*)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[\.\)])\s+")


def _cleanup_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 90:
        return False
    if s.endswith(".") and len(s.split()) > 3:
        return False
    return bool(_HEADING_RE.match(s))


def _export_text_as_readable_pdf(title: str, text: str, out_pdf: Path) -> None:
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "OCRTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=12,
    )
    style_h = ParagraphStyle(
        "OCRHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=10,
        spaceAfter=6,
    )
    style_body = ParagraphStyle(
        "OCRBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceBefore=2,
        spaceAfter=6,
    )

    # Monospace style for tables (no reflow)
    style_table = ParagraphStyle(
        "OCRTableMono",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=9.5,
        leading=12,
        spaceBefore=6,
        spaceAfter=6,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title=title,
    )

    cleaned = _cleanup_text(text)
    if not cleaned:
        cleaned = "(No text detected)"

    flow = [Paragraph(_escape_xml(title), style_title), Spacer(1, 6)]

    # Split by lines so we can treat table blocks specially
    lines = cleaned.splitlines()

    in_table = False
    table_lines = []
    para_lines = []

    def flush_para():
        nonlocal para_lines
        if not para_lines:
            return
        block = "\n".join(para_lines).strip()
        para_lines = []
        if not block:
            return

        # Split into paragraph blocks by blank lines
        blocks = re.split(r"\n\s*\n", block)
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            ls = [ln.strip() for ln in b.splitlines() if ln.strip()]
            if not ls:
                continue

            # heading?
            if len(ls) == 1 and _looks_like_heading(ls[0]) and not _BULLET_RE.match(ls[0]):
                flow.append(Paragraph(_escape_xml(ls[0]), style_h))
                continue

            # bullet block?
            bullet_count = sum(1 for ln in ls if _BULLET_RE.match(ln))
            if bullet_count >= max(2, int(0.6 * len(ls))):
                for ln in ls:
                    if _BULLET_RE.match(ln):
                        content = _BULLET_RE.sub("", ln).strip()
                        flow.append(Paragraph(_escape_xml(content), style_body, bulletText="•"))
                    else:
                        flow.append(Paragraph(_escape_xml(ln), style_body))
                flow.append(Spacer(1, 4))
                continue

            # normal paragraph (join lines to avoid weird wraps)
            paragraph = " ".join(ls)
            flow.append(Paragraph(_escape_xml(paragraph), style_body))

    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        # Keep as-is; Preformatted preserves spacing/newlines
        table_text = "\n".join(table_lines).rstrip()
        table_lines = []
        if not table_text.strip():
            return

        # Add a small label and render monospace
        flow.append(Paragraph(_escape_xml("TABLE"), style_h))
        flow.append(Preformatted(_escape_xml(table_text), style_table))
        flow.append(Spacer(1, 6))

    for ln in lines:
        raw = ln.rstrip("\n")

        if raw.strip() == "[TABLE]":
            flush_para()
            in_table = True
            table_lines = []
            continue

        if raw.strip() == "[/TABLE]":
            in_table = False
            flush_table()
            continue

        if in_table:
            # render raw table lines as-is; don't strip hard
            table_lines.append(raw)
        else:
            para_lines.append(raw)

    flush_para()
    if in_table:
        flush_table()

    doc.build(flow)



# -----------------------------
# JSON-first extraction (Layout Parser)
# -----------------------------

def _extract_text_from_layout_json(doc: dict) -> str:
    """
    Build nicely formatted plain text from Layout Parser snake_case JSON.
    - Uses text_block.type_ (heading vs paragraph)
    - Preserves nesting (headings + their child blocks)
    - Prints tables row-by-row with a separator
    """

    layout = doc.get("document_layout") or {}
    blocks = layout.get("blocks") or []
    if not blocks:
        return ""

    out = []

    def fix_text(s: str) -> str:
        s = str(s or "")
        # common docAI / OCR formatting artifacts
        s = s.replace("^{\\circ}", "°").replace("\\circ", "°")
        s = s.replace("~", " ")  # overdose line had 6-10~gm -> 6-10 gm
        s = re.sub(r"[ \t]+", " ", s).strip()
        return s

    def add_line(line: str = "") -> None:
        out.append(line)

    def render_text_block(tb: dict, level: int) -> None:
        text = fix_text(tb.get("text", ""))
        typ = (tb.get("type_") or "").strip()

        if not text and not tb.get("blocks"):
            return

        # Headings: uppercase + blank lines around for readability
        if typ.startswith("heading"):
            if out and out[-1] != "":
                add_line("")
            add_line(text.upper())
            add_line("")
        else:
            # paragraph
            if text:
                add_line(text)

        # Render children (nested blocks)
        children = tb.get("blocks") or []
        if children:
            render_blocks(children, level + 1)

    def cell_text_from_cell(cell: dict) -> str:
        parts = []
        for cb in cell.get("blocks") or []:
            ctb = cb.get("text_block")
            if ctb:
                t = fix_text(ctb.get("text", ""))
                if t:
                    parts.append(t)
        # join pieces inside a cell
        return " ".join(parts).strip()

    def render_table(tab: dict) -> None:
        body_rows = tab.get("body_rows") or []
        if not body_rows:
            return

        if out and out[-1] != "":
            add_line("")
        add_line("[TABLE]")

        for row in body_rows:
            cells = row.get("cells") or []
            row_cells = [cell_text_from_cell(c) for c in cells]
            # keep empty cells but trim trailing empties
            while row_cells and not row_cells[-1]:
                row_cells.pop()

            if any(row_cells):
                add_line(" | ".join(row_cells))

        add_line("[/TABLE]")
        add_line("")

    def render_blocks(bs: list, level: int) -> None:
        for b in bs:
            # text block
            tb = b.get("text_block")
            if tb:
                render_text_block(tb, level)
                continue

            # table block
            tab = b.get("table_block")
            if tab:
                render_table(tab)
                continue

            # (optional) list_block if your output ever contains it
            lb = b.get("list_block")
            if lb:
                if out and out[-1] != "":
                    add_line("")
                for item in lb.get("list_entries") or []:
                    t = fix_text(item.get("text", ""))
                    if t:
                        add_line(f"• {t}")
                add_line("")
                continue

    render_blocks(blocks, level=0)

    # Post-clean:
    text = "\n".join(out)

    # Remove 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Join broken lines that are obviously wrapped mid-sentence
    # (optional but helps: avoids "Ace XR\nTablet:" style breaks)
    text = re.sub(r"([a-z0-9,;:])\n([a-z])", r"\1 \2", text)

    return text


# -----------------------------
# Document AI (close to Google's sample)
# -----------------------------
def _get_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing {name} in .env")
    return v


def _init_docai_client():
    project_id = _get_env("GCP_PROJECT_ID")
    processor_id = _get_env("GCP_DOCAI_PROCESSOR_ID")
    location = _get_env("GCP_DOCAI_LOCATION")

    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai_v1.DocumentProcessorServiceClient(client_options=opts)

    full_processor_name = client.processor_path(project_id, location, processor_id)
    processor = client.get_processor(request=documentai_v1.GetProcessorRequest(name=full_processor_name))
    return client, processor


def _process_pdf_bytes(client, processor, pdf_bytes: bytes) -> documentai_v1.Document:
    raw_document = documentai_v1.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    req = documentai_v1.ProcessRequest(name=processor.name, raw_document=raw_document)
    result = client.process_document(request=req)
    return result.document

def _cleanup_ocr_inputs(in_dir: Path) -> None:
    """Delete all PDF files from the OCR input folder."""
    for p in in_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            try:
                p.unlink()
            except Exception as e:
                print(f"WARNING: failed to delete OCR input {p.name}: {e}")


def _cleanup_train_json(train_dir: Path) -> None:
    """Delete all JSON files from the train folder."""
    for p in train_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            try:
                p.unlink()
            except Exception as e:
                print(f"WARNING: failed to delete train JSON {p.name}: {e}")


# -----------------------------
# Main
# -----------------------------
def run_ocr() -> None:
    load_dotenv(ENV_PATH)

    # ensure credentials env exists; SDK uses it automatically
    _ = _get_env("GOOGLE_APPLICATION_CREDENTIALS")

    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in IN_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"No PDFs found in: {IN_DIR}")
        return

    client, processor = _init_docai_client()

    print(f"Found {len(pdfs)} PDF(s) in {IN_DIR}")
    print(f"Using processor: {processor.name}")

    ok = 0
    for pdf_path in pdfs:
        try:
            print(f"\nOCR -> {pdf_path.name}")

            document = _process_pdf_bytes(client, processor, pdf_path.read_bytes())

            # Convert to JSON dict and save (this is the "truth" for Layout Parser)
            doc_json: Dict[str, Any] = MessageToDict(document._pb, preserving_proto_field_name=True)  # type: ignore[attr-defined]
            json_out = OUT_DIR / f"{pdf_path.stem}.layout.json"
            json_out.write_text(json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8")

            # Build PDF from JSON (prefer chunkedDocument)
            text = _extract_text_from_layout_json(doc_json)

            # Debug counters
            chunks = (doc_json.get("chunkedDocument") or {}).get("chunks") or []
            blocks = (doc_json.get("documentLayout") or {}).get("blocks") or []
            pages = doc_json.get("pages") or []
            print(f"Pages: {len(pages)} | Layout blocks: {len(blocks)} | Chunks: {len(chunks)}")
            print("Extracted text length:", len(text))
            print("Extracted preview:", repr(text[:200]))

            out_pdf = OUT_DIR / f"{pdf_path.stem}.pdf"
            _export_text_as_readable_pdf(pdf_path.stem, text, out_pdf)

            print(f"Saved JSON: {json_out}")
            print(f"Saved PDF : {out_pdf}")
            ok += 1
        except Exception as e:
            print(f"FAILED: {pdf_path.name} -> {e}")

    print(f"\nDone. Successfully processed {ok}/{len(pdfs)} file(s).")
    _cleanup_ocr_inputs(IN_DIR)
    _cleanup_train_json(OUT_DIR)


    
if __name__ == "__main__":
    run_ocr()
