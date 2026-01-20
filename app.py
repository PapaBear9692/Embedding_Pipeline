import os
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from upsert import build_index
from dataCrawler import dataCrawler

ROOT_DIR = Path(__file__).resolve().parent

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Base data dirs (type-aware)
DATA_DIR = ROOT_DIR / "data"
PHARMA_DIR = DATA_DIR / "Pharma"
HERBAL_DIR = DATA_DIR / "Herbal"
AGROVET_DIR = DATA_DIR / "Agrovet"

_embed_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def _count_pdfs_in_type_folders() -> set[str]:
    """Return set of all PDF filenames found under data/Pharma and data/Herbal."""
    PHARMA_DIR.mkdir(parents=True, exist_ok=True)
    HERBAL_DIR.mkdir(parents=True, exist_ok=True)
    AGROVET_DIR.mkdir(parents=True, exist_ok=True)

    pharma = {p.name for p in PHARMA_DIR.glob("*.pdf")}
    herbal = {p.name for p in HERBAL_DIR.glob("*.pdf")}
    agrovet = {p.name for p in AGROVET_DIR.glob("*.pdf")}
    return pharma | herbal | agrovet


def create_app():
    app = Flask(__name__)
    print("Flask app running...")

    # Optional: limit upload size (e.g., 100MB total request)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    @app.get("/")
    def home():
        print("Rendering home page...")
        return render_template("index.html")

    @app.post("/api/train")
    def ingest():
        print("Received ingest request...")

        train_type = (request.form.get("train_type") or "").strip().lower()
        if train_type not in {"pharma", "herbal", "agrovet"}:
            return jsonify(
                {"error": "Invalid or missing product type. Select Pharma, Herbal, or Agrovet first."}
            ), 400

        upload_dir = DATA_DIR / train_type.capitalize()
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Frontend sends "files" (multiple)
        if "files" not in request.files:
            return jsonify({"error": "No files field in form-data (expected key: files)"}), 400

        files = request.files.getlist("files")
        print(f"Files received: {len(files)}")
        if not files:
            return jsonify({"error": "No files received"}), 400

        saved = 0
        skipped = 0

        for f in files:
            if not f or not f.filename:
                skipped += 1
                continue

            filename = secure_filename(f.filename)
            if not filename:
                skipped += 1
                continue

            if not allowed_file(filename):
                skipped += 1
                continue

            out_path = upload_dir / filename
            f.save(out_path)
            saved += 1

        if saved == 0:
            return jsonify(
                {"error": "No valid files to ingest", "files": 0, "chunks": 0, "skipped": skipped}
            ), 400

        chunk_count = 0
        try:
            result = build_index(train_type=train_type)
            if result is None:
                chunk_count = 0
            else:
                _, chunk_count = result

        except Exception as e:
            return jsonify(
                {"error": str(e), "files": saved, "chunks": chunk_count, "skipped": skipped}
            ), 500

        print("Training complete.")
        return jsonify({"files": saved, "chunks": chunk_count, "skipped": skipped})

    @app.post("/api/crawl")
    def embed_from_crawl():
        # Prevent overlapping runs
        if not _embed_lock.acquire(blocking=False):
            return jsonify({"error": "Embedding pipeline already running"}), 409

        saved = 0
        skipped = 0
        chunk_count = 0

        try:
            # Count PDFs before (across both type folders)
            before_pdfs = _count_pdfs_in_type_folders()

            # 1) auto-download into data/Pharma and data/Herbal (crawler decides)
            dataCrawler()

            # Count PDFs after -> compute "saved" as new filenames
            after_pdfs = _count_pdfs_in_type_folders()
            saved = len(after_pdfs - before_pdfs)

            if saved == 0:
                return jsonify(
                    {"error": "No New PDF files to ingest", "files": 0, "chunks": 0, "skipped": skipped}
                ), 400

            # 2) embed/build index for BOTH types
            result = build_index()  # train_type=None -> process both
            if result is None:
                chunk_count = 0
            else:
                _, chunk_count = result

            return jsonify({"files": saved, "chunks": chunk_count, "skipped": skipped})

        except Exception as e:
            return jsonify(
                {"error": str(e), "files": saved, "chunks": chunk_count, "skipped": skipped}
            ), 500

        finally:
            _embed_lock.release()

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)
    #app.run(host="127.0.0.1", port=5000, debug=True)
