import csv
import io
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from pymongo import MongoClient
from bson import ObjectId
from gridfs import GridFSBucket
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
EVAL = ROOT / "data" / "evaluation"

GOLD_MANIFEST = META / "gold_query_manifest.csv"
PROTO_MANIFEST = META / "prototype_unique_manifest.csv"
TOP5_FILE = EVAL / "dinov2_top5_results.csv"
LABEL_FILE = META / "verified_matching_pairs.csv"

load_dotenv(ROOT / ".env")

app = Flask(__name__)
client = None
db = None
buckets = {}


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def filename_key(value):
    return (value or "").strip().replace("\\", "/").split("/")[-1].casefold()


def load_manifest(path):
    rows = read_csv(path)
    return {
        filename_key(row.get("filename")): row for row in rows if row.get("filename")
    }


def load_dashboard_data():
    gold_manifest = load_manifest(GOLD_MANIFEST)
    proto_manifest = load_manifest(PROTO_MANIFEST)
    top_rows = read_csv(TOP5_FILE)

    try:
        label_rows = read_csv(LABEL_FILE)
    except FileNotFoundError:
        label_rows = []

    # A verified pair is indexed by its two filenames.
    labels = {}
    for row in label_rows:
        gold_name = filename_key(row.get("gold_filename"))
        proto_name = filename_key(row.get("prototype_filename"))
        label = (row.get("label") or "").strip()

        if gold_name and proto_name and label in {"0", "1"}:
            labels[(gold_name, proto_name)] = {
                "label": int(label),
                "match_type": row.get("match_type", ""),
                "verifier": row.get("verifier", ""),
                "notes": row.get("notes", ""),
            }

    grouped = {}
    for row in top_rows:
        gold_name = (row.get("gold_filename") or "").strip()
        proto_name = (row.get("prototype_filename") or "").strip()
        if not gold_name or not proto_name:
            continue

        gkey = filename_key(gold_name)
        pkey = filename_key(proto_name)

        try:
            rank = int(row.get("rank", "0"))
            score = float(row.get("cosine_similarity", "0"))
        except (TypeError, ValueError):
            continue

        if gkey not in gold_manifest or pkey not in proto_manifest:
            continue

        verified = labels.get((gkey, pkey))
        item = {
            "rank": rank,
            "prototype_filename": proto_name,
            "score": score,
            "verified_label": verified["label"] if verified else None,
            "match_type": verified["match_type"] if verified else "",
            "gold_image_url": ("/api/image?kind=gold&filename=" + gold_name),
            "prototype_image_url": ("/api/image?kind=prototype&filename=" + proto_name),
        }

        grouped.setdefault(gold_name, []).append(item)

    results = []
    for gold_name, matches in grouped.items():
        matches.sort(key=lambda item: item["rank"])
        results.append(
            {
                "gold_filename": gold_name,
                "gold_image_url": "/api/image?kind=gold&filename=" + gold_name,
                "matches": matches[:5],
            }
        )

    return results, gold_manifest, proto_manifest


def get_db():
    global client, db

    if db is not None:
        return db

    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URL")
    if not uri:
        raise RuntimeError(
            "MongoDB URI not found. Check MONGO_URI, MONGODB_URI, or MONGO_URL in .env."
        )

    db_name = os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DB") or "jewelmatch"
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[db_name]
    return db


def read_image_bytes(row):
    source = (row.get("source") or "").strip().lower()

    if source == "local":
        raw_path = (row.get("local_path") or "").strip()
        if not raw_path:
            raise FileNotFoundError("No local image path in manifest.")

        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        return path.read_bytes()

    if source == "gridfs":
        database = get_db()
        bucket_name = (row.get("gridfs_bucket") or "fs").strip()

        if bucket_name not in buckets:
            buckets[bucket_name] = GridFSBucket(database, bucket_name=bucket_name)

        raw_id = (row.get("gridfs_file_id") or "").strip()
        if not raw_id:
            raise FileNotFoundError("No GridFS file ID in manifest.")

        try:
            file_id = ObjectId(raw_id)
        except Exception:
            file_id = raw_id

        return buckets[bucket_name].open_download_stream(file_id).read()

    raise ValueError(f"Unsupported image source: {source}")


@app.get("/api/health")
def health():
    try:
        results, _, _ = load_dashboard_data()
        return jsonify(
            {
                "ok": True,
                "gold_queries": len(results),
                "top5_file": TOP5_FILE.is_file(),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/dashboard")
def dashboard():
    try:
        results, _, _ = load_dashboard_data()
        return jsonify({"results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/image")
def image_route():
    kind = (request.args.get("kind") or "").strip().lower()
    filename = (request.args.get("filename") or "").strip()

    if kind not in {"gold", "prototype"} or not filename:
        return jsonify({"error": "Provide kind=gold/prototype and filename."}), 400

    try:
        gold_manifest = load_manifest(GOLD_MANIFEST)
        proto_manifest = load_manifest(PROTO_MANIFEST)
        manifest = gold_manifest if kind == "gold" else proto_manifest

        row = manifest.get(filename_key(filename))
        if row is None:
            return jsonify({"error": "Image not found in manifest."}), 404

        data = read_image_bytes(row)

        # Return a normalised display image without changing the source file.
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            output = io.BytesIO()
            im.save(output, format="JPEG", quality=90)
            output.seek(0)

        return send_file(output, mimetype="image/jpeg", max_age=0)

    except Exception as exc:
        app.logger.exception("Could not serve image")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("Jewellery Model 3 - Baseline Dashboard")
    print("Read-only: no MongoDB writes are performed.")
    app.run(host="127.0.0.1", port=5002, debug=False)
