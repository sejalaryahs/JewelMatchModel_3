import os
import io
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from gridfs import GridFSBucket
from bson import ObjectId
from transformers import AutoImageProcessor, AutoModel

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "facebook/dinov2-small"

DATABASE_NAME = os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DB") or "jewelmatch"

SOURCE_COLLECTION = "prototype"
EMBEDDING_COLLECTION = "prototype_dinov2_embeddings_v1"
GRIDFS_BUCKET_NAME = "fs"

BATCH_SIZE = 4
DEVICE = torch.device("cpu")


# ============================================================
# MONGODB CONNECTION
# ============================================================


def get_mongo_uri():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URL")

    if not uri:
        raise RuntimeError(
            "MongoDB URI not found. Check the URI variable in your .env file."
        )

    return uri


# ============================================================
# IMAGE LOADING
# ============================================================


def load_gridfs_image(bucket, image_gridfs_id):
    """Load an image directly from GridFS without writing it to disk."""

    if not isinstance(image_gridfs_id, ObjectId):
        image_gridfs_id = ObjectId(str(image_gridfs_id))

    stream = bucket.open_download_stream(image_gridfs_id)
    try:
        image_bytes = stream.read()
    finally:
        stream.close()

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


# ============================================================
# DINOv2 EMBEDDING
# ============================================================


def create_embedding(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")

    # CPU-only inference
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

        # DINOv2 CLS token representation
        embedding = outputs.last_hidden_state[:, 0, :]

        # Normalize so cosine similarity can be calculated by dot product
        embedding = F.normalize(embedding, p=2, dim=1)

    vector = embedding[0].cpu().tolist()

    if len(vector) != 384:
        raise RuntimeError(f"Unexpected embedding size: {len(vector)}; expected 384.")

    return vector


# ============================================================
# MAIN BUILDER
# ============================================================


def main():
    print("=" * 65)
    print("JEWELLERY MODEL 3 - MONGODB DINOv2 EMBEDDING BUILDER")
    print("=" * 65)

    print(f"Database: {DATABASE_NAME}")
    print(f"Source collection: {SOURCE_COLLECTION}")
    print(f"Destination collection: {EMBEDDING_COLLECTION}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    client = MongoClient(
        get_mongo_uri(),
        serverSelectionTimeoutMS=10000,
    )

    try:
        client.admin.command("ping")
        db = client[DATABASE_NAME]

        source = db[SOURCE_COLLECTION]
        destination = db[EMBEDDING_COLLECTION]
        bucket = GridFSBucket(db, bucket_name=GRIDFS_BUCKET_NAME)

        # Create an index in the NEW collection only.
        destination.create_index(
            "catalogue_id",
            unique=True,
        )

        records = list(source.find({}).sort("id", 1))

        if not records:
            print("No prototype records found.")
            return

        print(f"\nFound {len(records)} prototype catalogue records.")

        print("\nLoading DINOv2 Small model...")
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()

        operations = []
        processed = 0
        skipped = 0

        for index, record in enumerate(records, start=1):
            catalogue_id = record.get("id")
            image_gridfs_id = record.get("image_gridfs_id")

            if not catalogue_id:
                print(f"[SKIP] Record {record.get('_id')} has no catalogue ID.")
                skipped += 1
                continue

            if not image_gridfs_id:
                print(f"[SKIP] {catalogue_id}: no image_gridfs_id.")
                skipped += 1
                continue

            try:
                image = load_gridfs_image(bucket, image_gridfs_id)
                embedding = create_embedding(image, processor, model)

                document = {
                    "catalogue_id": str(catalogue_id),
                    "catalogue_name": record.get("name", ""),
                    "catalogue_type": record.get("type", ""),
                    "catalogue_collection": record.get("collection", ""),
                    "source_document_id": record["_id"],
                    "image_gridfs_id": image_gridfs_id,
                    "filename": record.get("filename") or record.get("image", ""),
                    "image_path": record.get("image_path", ""),
                    "model_name": MODEL_NAME,
                    "embedding_dimension": len(embedding),
                    "embedding": embedding,
                    "created_at": datetime.now(timezone.utc),
                }

                operations.append(
                    UpdateOne(
                        {"catalogue_id": str(catalogue_id)},
                        {"$set": document},
                        upsert=True,
                    )
                )

                processed += 1
                print(
                    f"[{index}/{len(records)}] "
                    f"Embedded {catalogue_id} | "
                    f"{document['catalogue_name']} | "
                    f"{len(embedding)} dimensions"
                )

                # Write small batches to MongoDB.
                if len(operations) >= BATCH_SIZE:
                    result = destination.bulk_write(
                        operations,
                        ordered=False,
                    )
                    print(
                        f"  Saved batch: "
                        f"{result.upserted_count} inserted, "
                        f"{result.modified_count} updated."
                    )
                    operations.clear()

            except Exception as exc:
                skipped += 1
                print(f"[ERROR] {catalogue_id}: {exc}")

        if operations:
            result = destination.bulk_write(operations, ordered=False)
            print(
                f"Saved final batch: "
                f"{result.upserted_count} inserted, "
                f"{result.modified_count} updated."
            )

        print("\n" + "=" * 65)
        print("EMBEDDING BUILD SUMMARY")
        print("=" * 65)
        print(f"Source catalogue records: {len(records)}")
        print(f"Successfully embedded: {processed}")
        print(f"Skipped or failed: {skipped}")
        print(
            "Documents in embedding collection:",
            destination.count_documents({}),
        )
        print(f"Embedding collection: {EMBEDDING_COLLECTION}")
        print("\nExisting prototype records were not modified.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
