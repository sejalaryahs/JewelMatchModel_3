import csv
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
OUT = ROOT / "data" / "embeddings"

GOLD_MANIFEST = META / "gold_query_manifest.csv"
PROTO_MANIFEST = META / "prototype_unique_manifest.csv"

MODEL_NAME = "facebook/dinov2-small"
DEVICE = torch.device("cpu")


def read_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # Only use images that passed dataset validation.
    return [r for r in rows if r.get("status", "").strip().lower() == "valid"]


def get_gridfs_image(row, db, client_cache):
    bucket_name = (row.get("gridfs_bucket") or "fs").strip()

    if bucket_name not in client_cache:
        from gridfs import GridFSBucket

        client_cache[bucket_name] = GridFSBucket(db, bucket_name=bucket_name)

    file_id = (row.get("gridfs_file_id") or "").strip()
    if not file_id:
        raise ValueError(
            f"Missing GridFS file ID for {row.get('filename', 'unknown image')}"
        )

    # MongoDB-generated IDs are normally ObjectIds. Also allow string IDs.
    try:
        lookup_id = ObjectId(file_id)
    except Exception:
        lookup_id = file_id

    data = client_cache[bucket_name].open_download_stream(lookup_id).read()
    return Image.open(io.BytesIO(data))


def load_image(row, db, bucket_cache):
    source = (row.get("source") or "").strip().lower()

    if source == "local":
        path_text = (row.get("local_path") or "").strip()
        if not path_text:
            raise ValueError(f"Missing local_path for {row.get('filename')}")

        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path)

    elif source == "gridfs":
        image = get_gridfs_image(row, db, bucket_cache)

    else:
        raise ValueError(
            f"Unsupported image source '{source}' for {row.get('filename')}"
        )

    # Apply EXIF orientation and ensure consistent RGB input.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.load()
    return image


def make_embeddings(rows, processor, model, db, bucket_cache, label):
    vectors = []
    successful_rows = []
    errors = []

    total = len(rows)
    print(f"\nProcessing {label}: {total} images")

    for index, row in enumerate(rows, start=1):
        filename = row.get("filename") or f"image_{index}"

        try:
            image = load_image(row, db, bucket_cache)
            inputs = processor(images=image, return_tensors="pt")

            # CPU-only inference; no gradient or training.
            with torch.inference_mode():
                outputs = model(**inputs)
                # DINOv2 CLS token is the image-level representation.
                vector = outputs.last_hidden_state[:, 0, :]
                vector = F.normalize(vector, p=2, dim=1)

            vectors.append(vector[0].cpu().numpy().astype(np.float32))
            successful_rows.append(row)

            print(f"[{index}/{total}] OK  {filename}")

        except Exception as exc:
            errors.append((filename, str(exc)))
            print(f"[{index}/{total}] ERROR  {filename}: {exc}")

    if not vectors:
        raise RuntimeError(f"No embeddings were generated for {label}.")

    return np.stack(vectors), successful_rows, errors


def save_metadata(path, rows):
    if not rows:
        return

    # Preserve all manifest fields for traceability.
    columns = list(dict.fromkeys(key for row in rows for key in row.keys()))

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    load_dotenv(ROOT / ".env")

    print("=" * 64)
    print("JEWELLERY MODEL 3 - DINOv2 SMALL BASELINE")
    print("=" * 64)
    print("Project:", ROOT)
    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)
    print("This script does not train or fine-tune the model.")
    print("It does not write to MongoDB.")

    gold_rows = read_manifest(GOLD_MANIFEST)
    proto_rows = read_manifest(PROTO_MANIFEST)

    print(f"\nValid gold images: {len(gold_rows)}")
    print(f"Valid prototype images: {len(proto_rows)}")

    if not gold_rows or not proto_rows:
        raise RuntimeError(
            "Gold and prototype manifests must both contain valid images."
        )

    mongo_uri = (
        os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URL")
    )

    if any((r.get("source") or "").lower() == "gridfs" for r in proto_rows):
        if not mongo_uri:
            raise RuntimeError(
                "Prototype images use GridFS, but no MongoDB URI was found in .env. "
                "Check MONGO_URI, MONGODB_URI, or MONGO_URL."
            )

    db_name = os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DB") or "jewelmatch"
    client = None
    db = None
    bucket_cache = {}

    try:
        if mongo_uri:
            # Read-only use: this script only downloads GridFS file contents.
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
            client.admin.command("ping")
            db = client[db_name]
            print(f"MongoDB connection OK; database: {db_name}")

        print("\nLoading pretrained DINOv2 Small...")
        print("The first run may download model files into your Hugging Face cache.")

        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()

        print("Model loaded.")
        print("Embedding dimension:", model.config.hidden_size)

        gold_vectors, gold_ok, gold_errors = make_embeddings(
            gold_rows, processor, model, db, bucket_cache, "gold queries"
        )

        proto_vectors, proto_ok, proto_errors = make_embeddings(
            proto_rows, processor, model, db, bucket_cache, "prototype catalogue"
        )

        OUT.mkdir(parents=True, exist_ok=True)

        gold_npz = OUT / "dinov2_small_gold_embeddings.npz"
        proto_npz = OUT / "dinov2_small_prototype_embeddings.npz"

        np.savez_compressed(
            gold_npz,
            embeddings=gold_vectors,
            filenames=np.asarray([r.get("filename", "") for r in gold_ok], dtype=str),
            sha256=np.asarray([r.get("sha256", "") for r in gold_ok], dtype=str),
        )

        np.savez_compressed(
            proto_npz,
            embeddings=proto_vectors,
            filenames=np.asarray([r.get("filename", "") for r in proto_ok], dtype=str),
            sha256=np.asarray([r.get("sha256", "") for r in proto_ok], dtype=str),
        )

        save_metadata(OUT / "dinov2_small_gold_metadata.csv", gold_ok)
        save_metadata(OUT / "dinov2_small_prototype_metadata.csv", proto_ok)

        print("\n" + "=" * 64)
        print("BASELINE EMBEDDING SUMMARY")
        print("=" * 64)
        print(f"Gold embeddings:      {len(gold_ok)}")
        print(f"Prototype embeddings: {len(proto_ok)}")
        print(f"Embedding dimension:  {gold_vectors.shape[1]}")
        print(f"Gold output:          {gold_npz}")
        print(f"Prototype output:     {proto_npz}")
        print(f"Metadata folder:      {OUT}")

        if gold_errors or proto_errors:
            print("\nWARNING: Some images could not be processed.")
            for filename, error in gold_errors + proto_errors:
                print(f"  {filename}: {error}")
            print("Resolve these errors before evaluating the full catalogue.")
        else:
            print("\nSUCCESS: Every manifest image was embedded.")

        print("\nNo model training occurred. No MongoDB records were changed.")

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        raise
