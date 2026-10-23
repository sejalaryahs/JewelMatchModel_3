import os
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from gridfs import GridFS
from transformers import AutoImageProcessor, AutoModel

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DATABASE = os.getenv("MONGODB_DATABASE")
GRIDFS_BUCKET = os.getenv("GRIDFS_BUCKET", "fs")

SOURCE_COLLECTION = "prototype"
EMBEDDING_COLLECTION = "prototype_dinov2_embeddings_v1"

MODEL_NAME = "facebook/dinov2-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_GRID_SIZE = 14

if not MONGO_URI:
    raise RuntimeError("MONGODB_URI is missing from the root .env file.")

if not MONGO_DATABASE:
    raise RuntimeError("MONGODB_DATABASE is missing from the root .env file.")

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=20000)
db = client[MONGO_DATABASE]

source = db[SOURCE_COLLECTION]
embeddings = db[EMBEDDING_COLLECTION]
fs_bucket = GridFS(db, collection=GRIDFS_BUCKET)

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------


def load_rgb(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def make_views(image):
    """Create multiple views without modifying the stored source image."""
    views = [("full", image)]
    width, height = image.size

    for fraction, name in ((0.85, "crop85"), (0.70, "crop70")):
        crop_width = max(1, int(width * fraction))
        crop_height = max(1, int(height * fraction))

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2

        views.append(
            (name, image.crop((left, top, left + crop_width, top + crop_height)))
        )

    for angle in (90, 180, 270):
        views.append((f"rotation{angle}", image.rotate(angle, expand=True)))

    return views


# --------------------------------------------------
# MODEL
# --------------------------------------------------

print("Loading DINOv2:", MODEL_NAME)
print("Device:", DEVICE)

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()


def extract_features(images):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state.float()

    # Global CLS embeddings: (views, 384)
    global_features = F.normalize(hidden[:, 0, :], p=2, dim=-1)

    # Patch embeddings
    patches = hidden[:, 1:, :]
    batch, count, dimension = patches.shape

    side = int(count**0.5)
    if side * side != count:
        raise ValueError(f"Unexpected patch count: {count}")

    patches = patches.transpose(1, 2).reshape(batch, dimension, side, side)

    if side != PATCH_GRID_SIZE:
        patches = F.adaptive_avg_pool2d(patches, (PATCH_GRID_SIZE, PATCH_GRID_SIZE))

    patches = patches.flatten(2).transpose(1, 2)
    patches = F.normalize(patches, p=2, dim=-1)

    return (
        global_features.cpu().numpy().astype(np.float32),
        patches.cpu().numpy().astype(np.float32),
    )


# --------------------------------------------------
# BUILD EMBEDDINGS
# --------------------------------------------------


def main():
    records = list(source.find({}))
    print("Prototype records found:", len(records))
    print("Existing embedding collection:", EMBEDDING_COLLECTION)

    operations = []
    processed = 0
    skipped = 0

    try:
        for index, record in enumerate(records, start=1):
            source_document_id = record["_id"]
            gridfs_id = record.get("image_gridfs_id")
            catalogue_id = None

            try:
                # Match the source document to its EXISTING embedding.
                existing = embeddings.find_one(
                    {"source_document_id": source_document_id},
                    {
                        "_id": 1,
                        "catalogue_id": 1,
                        "catalogue_name": 1,
                        "catalogue_collection": 1,
                        "catalogue_type": 1,
                        "filename": 1,
                    },
                )

                if existing is None:
                    print(
                        f"[SKIP] No existing embedding for source "
                        f"{source_document_id}"
                    )
                    skipped += 1
                    continue

                catalogue_id = existing.get("catalogue_id")

                if not catalogue_id:
                    print(
                        f"[SKIP] Existing embedding has no catalogue_id: "
                        f"{source_document_id}"
                    )
                    skipped += 1
                    continue

                if not gridfs_id:
                    print(f"[SKIP] {catalogue_id}: missing image_gridfs_id")
                    skipped += 1
                    continue

                # Read the original image from GridFS.
                image_bytes = fs_bucket.get(gridfs_id).read()
                image = load_rgb(image_bytes)

                named_views = make_views(image)
                view_names = [name for name, _ in named_views]
                view_images = [im for _, im in named_views]

                global_features, patch_features = extract_features(view_images)

                document = {
                    "catalogue_id": catalogue_id,
                    "source_document_id": source_document_id,
                    "catalogue_collection": existing.get(
                        "catalogue_collection", "Prototype"
                    ),
                    "catalogue_name": existing.get("catalogue_name", str(catalogue_id)),
                    "catalogue_type": existing.get("catalogue_type"),
                    "filename": existing.get("filename", record.get("filename", "")),
                    "image_gridfs_id": gridfs_id,
                    "model_name": MODEL_NAME,
                    "embedding_method": "dinov2_multiview_global_patch_v1",
                    "embedding_dimension": int(global_features.shape[1]),
                    "embedding": global_features[0].tolist(),
                    "global_embedding": global_features[0].tolist(),
                    "global_embeddings": global_features.tolist(),
                    "patch_embeddings": patch_features.tolist(),
                    "patch_count": int(patch_features.shape[1]),
                    "patch_grid_size": PATCH_GRID_SIZE,
                    "view_names": view_names,
                    "updated_at": datetime.now(timezone.utc),
                }

                # Update the exact existing document; never upsert.
                operations.append(
                    UpdateOne(
                        {"_id": existing["_id"]}, {"$set": document}, upsert=False
                    )
                )

                processed += 1
                print(f"[{index}/{len(records)}] Prepared {catalogue_id}")

                if len(operations) >= 10:
                    result = embeddings.bulk_write(operations, ordered=False)
                    print("Existing records updated:", result.modified_count)
                    operations.clear()

            except Exception as exc:
                skipped += 1
                print(f"[ERROR] {catalogue_id or source_document_id}: {exc}")

        if operations:
            result = embeddings.bulk_write(operations, ordered=False)
            print("Existing records updated:", result.modified_count)
            operations.clear()

        print("\nEmbedding build finished.")
        print("Successfully prepared:", processed)
        print("Skipped or failed:", skipped)
        print("Target collection:", EMBEDDING_COLLECTION)
        print("Source catalogue modified: No")
        print("New embedding documents inserted: No")

    finally:
        client.close()


if __name__ == "__main__":
    main()
