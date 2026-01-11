import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from upsert import build_index  # calls your existing pipeline

ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "ocr"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def create_app():

    app = Flask(__name__)
    print("Flask app running...")

    # Optional: limit upload size (e.g., 50MB total request)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def home():
        print("Rendering home page...")
        return render_template("index.html")

    @app.post("/api/train")
    def ingest():
        print("Received ingest request...")
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

            out_path = UPLOAD_DIR / filename
            f.save(out_path)
            saved += 1

        if saved == 0:
            return jsonify({"error": "No valid PDF files to ingest", "files": 0, "chunks": 0, "skipped": skipped}), 400
        chunk_count = 0
        try:
            result = build_index()
            if result is None:
                chunk_count = 0
            else:
                index, chunk_count = result

        except Exception as e:
            return jsonify({"error": str(e), "files": saved, "chunks": chunk_count, "skipped": skipped}), 500
        print("Training complete.")
        return jsonify({"files": saved, "chunks": chunk_count, "skipped": skipped})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)
    #app.run(host="127.0.0.1", port=5000, debug=True)
