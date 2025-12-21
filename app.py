# app.py
import os
import uuid
import shutil
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import upsert

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = BASE_DIR / "data" / "train_data"
TRAIN_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# app.py (fixed parts)

@app.post("/upload")
def upload():
    if "files" not in request.files:
        return jsonify({"status": "error", "message": "Use multipart key 'files'"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"status": "error", "message": "No files uploaded"}), 400

    job_id = uuid.uuid4().hex

    
    job_dir = TRAIN_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved, rejected = [], []
    for f in files:
        if not f or not f.filename:
            continue

        filename = secure_filename(f.filename)
        if not allowed_file(filename):
            rejected.append({"file": f.filename, "reason": "Only PDF allowed"})
            continue

        dst = job_dir / filename

        # avoid overwrite inside same job
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            i = 1
            while True:
                candidate = job_dir / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1

        f.save(dst)
        saved.append(str(dst))

    return jsonify({"status": "ok", "job_id": job_id, "saved": saved, "rejected": rejected})


@app.post("/upsert/<job_id>")
def run_upsert(job_id: str):
    
    job_dir = TRAIN_ROOT / str(job_id)

    if not job_dir.exists() or not job_dir.is_dir():
        return jsonify({"status": "error", "message": "Invalid job_id"}), 404

    pdfs = list(job_dir.glob("*.pdf"))
    if not pdfs:
        return jsonify({"status": "error", "message": "No PDFs found for this job"}), 400

    try:
        # This matches your dataloader.py expectation (job_dir is the folder name)
        result = upsert.build_index(str(job_id))

        shutil.rmtree(job_dir)

        return jsonify({
            "status": "ok",
            "message": "Upsert completed",
            "job_id": job_id,
            "pdf_count": len(pdfs),
            "result_type": type(result).__name__ if result is not None else None
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "job_id": job_id}), 500



if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
