import io
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pymongo import MongoClient
from gridfs import GridFSBucket
from bson import ObjectId
from transformers import AutoImageProcessor, AutoModel

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# IMPORTANT: Define MODEL_NAME before loading the model.
MODEL_NAME = "facebook/dinov2-small"

DB_NAME = (
    os.getenv("MONGODB_DATABASE")
    or os.getenv("MONGO_DB_NAME")
    or os.getenv("MONGODB_DB")
    or "jewelmatch"
)

EMBEDDING_COLLECTION = "prototype_dinov2_embeddings_v1"

TOP_K = 5
DEVICE = torch.device("cpu")
PATCH_GRID_SIZE = 14
EMBEDDING_DIMENSION = 384

GRIDFS_BUCKET = os.getenv("GRIDFS_BUCKET", "fs")


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

CORS(
    app,
    origins=os.getenv("CORS_ORIGINS", "*").split(","),
)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# ============================================================
# MONGODB CONNECTION
# ============================================================


def get_mongo_uri():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URL")

    if not uri:
        raise RuntimeError("MongoDB URI is missing from environment variables.")

    return uri


client = MongoClient(
    get_mongo_uri(),
    serverSelectionTimeoutMS=15000,
)

db = client[DB_NAME]
embeddings = db[EMBEDDING_COLLECTION]

gridfs_bucket = GridFSBucket(
    db,
    bucket_name=GRIDFS_BUCKET,
)


# ============================================================
# LOAD DINOV2 MODEL
# ============================================================

print("Loading DINOv2 model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

print("DINOv2 model loaded.")
print(f"MongoDB database: {DB_NAME}")
print(f"Embedding collection: {EMBEDDING_COLLECTION}")
print(f"GridFS bucket: {GRIDFS_BUCKET}")
print(f"Top K: {TOP_K}")


# ============================================================
# IMAGE PROCESSING
# ============================================================


def load_rgb(raw):
    with Image.open(io.BytesIO(raw)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def make_views(image):
    """
    Create the same three views used by the embedding builder:
    1. Original image
    2. Center crop at 85%
    3. Center crop at 70%

    The builder also stores 90, 180, and 270 degree rotations.
    """

    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size

    views = [image]

    for scale in (0.85, 0.70):
        crop_w = max(1, int(width * scale))
        crop_h = max(1, int(height * scale))

        left = (width - crop_w) // 2
        top = (height - crop_h) // 2

        crop = image.crop(
            (
                left,
                top,
                left + crop_w,
                top + crop_h,
            )
        )

        views.append(crop)

    # Match the embedding builder's rotation order.
    views.extend(
        [
            image.rotate(90, expand=True),
            image.rotate(180, expand=True),
            image.rotate(270, expand=True),
        ]
    )

    return views


# ============================================================
# DINOV2 FEATURE EXTRACTION
# ============================================================


def extract_view(image):
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model(**inputs)

        tokens = output.last_hidden_state

        # Global CLS-token embedding.
        global_vector = F.normalize(
            tokens[:, 0, :],
            p=2,
            dim=-1,
        )

        # Patch-token embeddings.
        patch_tokens = tokens[:, 1:, :]

        count = patch_tokens.shape[1]
        grid = int(count**0.5)

        if grid * grid != count:
            raise RuntimeError(f"Unexpected patch count: {count}")

        channels = patch_tokens.shape[-1]

        feature_map = patch_tokens.transpose(1, 2).reshape(
            1,
            channels,
            grid,
            grid,
        )

        patches = F.adaptive_avg_pool2d(
            feature_map,
            (PATCH_GRID_SIZE, PATCH_GRID_SIZE),
        )

        patches = patches.flatten(2).transpose(1, 2)

        patches = F.normalize(
            patches,
            p=2,
            dim=-1,
        )

    return (
        global_vector[0].cpu().numpy().astype(np.float32),
        patches[0].cpu().numpy().astype(np.float32),
    )


def extract_features(image):
    pairs = [extract_view(view) for view in make_views(image)]

    global_embeddings = np.stack([pair[0] for pair in pairs])

    patch_embeddings = np.stack([pair[1] for pair in pairs])

    return global_embeddings, patch_embeddings


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================


def normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(
        vectors,
        axis=-1,
        keepdims=True,
    )

    return vectors / np.maximum(norms, 1e-12)


def compare_patches(query, stored):
    query = normalize_rows(query)
    stored = normalize_rows(stored)

    matrix = query @ stored.T

    forward = float(matrix.max(axis=1).mean())
    backward = float(matrix.max(axis=0).mean())

    return (forward + backward) / 2


def compare_features(
    query_globals,
    query_patches,
    stored_globals,
    stored_patches,
):
    query_globals = normalize_rows(query_globals)
    stored_globals = normalize_rows(stored_globals)

    # Compare all query views against all catalogue views.
    global_matrix = query_globals @ stored_globals.T

    original_score = float(global_matrix[0, 0])

    alternate_score = float(global_matrix[1:, 1:].max())

    global_score = 0.65 * original_score + 0.35 * alternate_score

    # Original image patch similarity.
    patch_full = compare_patches(
        query_patches[0],
        stored_patches[0],
    )

    # Center-crop comparisons.
    patch_crop_scores = [
        compare_patches(
            query_patches[q],
            stored_patches[s],
        )
        for q in (1, 2)
        for s in (1, 2)
    ]

    # Rotation comparisons.
    patch_rotation_scores = [
        compare_patches(
            query_patches[q],
            stored_patches[s],
        )
        for q in (3, 4, 5)
        for s in (3, 4, 5)
    ]

    best_crop_score = max(patch_crop_scores)
    best_rotation_score = max(patch_rotation_scores)

    patch_score = (
        0.50 * patch_full + 0.25 * best_crop_score + 0.25 * best_rotation_score
    )

    score = 0.55 * global_score + 0.45 * patch_score

    return score, global_score, patch_score


# ============================================================
# HOME ROUTE
# ============================================================


@app.get("/")
def index():
    return jsonify(
        {
            "service": "Jewellery Model 3 API",
            "status": "running",
            "top_k": TOP_K,
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/api/health")
def health():
    try:
        client.admin.command("ping")

        return jsonify(
            {
                "status": "ok",
                "model": MODEL_NAME,
                "database": DB_NAME,
                "embedding_collection": EMBEDDING_COLLECTION,
                "embedding_count": embeddings.count_documents({}),
                "top_k": TOP_K,
            }
        )

    except Exception:
        app.logger.exception("Health check failed")

        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Database unavailable",
                }
            ),
            503,
        )


# ============================================================
# JEWELLERY IMAGE MATCHING
# ============================================================


@app.post("/api/match")
def match():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image."}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "No image selected."}), 400

    try:
        raw = uploaded.read()

        if not raw:
            return jsonify({"error": "The uploaded image is empty."}), 400

        query_image = load_rgb(raw)

        query_globals, query_patches = extract_features(query_image)

    except Exception:
        app.logger.exception("Query image processing failed")

        return jsonify({"error": "Unable to process this image."}), 400

    try:
        docs = list(
            embeddings.find(
                {
                    "model_name": MODEL_NAME,
                    "embedding_dimension": EMBEDDING_DIMENSION,
                }
            )
        )

        if not docs:
            return jsonify({"error": "No prototype embeddings found."}), 503

        results = []
        skipped = 0

        for doc in docs:
            try:
                stored_globals = np.asarray(
                    doc["global_embeddings"],
                    dtype=np.float32,
                )

                stored_patches = np.asarray(
                    doc["patch_embeddings"],
                    dtype=np.float32,
                )

                if stored_globals.shape != (
                    6,
                    EMBEDDING_DIMENSION,
                ):
                    skipped += 1
                    continue

                if (
                    stored_patches.ndim != 3
                    or stored_patches.shape[0] != 6
                    or stored_patches.shape[1] != 196
                    or stored_patches.shape[2] != EMBEDDING_DIMENSION
                ):
                    skipped += 1
                    continue

                score, global_score, patch_score = compare_features(
                    query_globals,
                    query_patches,
                    stored_globals,
                    stored_patches,
                )

                gridfs_id = doc.get("image_gridfs_id")

                results.append(
                    {
                        "catalogueId": str(doc.get("catalogue_id", "")),
                        "name": (doc.get("catalogue_name") or doc.get("filename", "")),
                        "type": doc.get("catalogue_type", ""),
                        "filename": doc.get("filename", ""),
                        "imageUrl": (
                            f"/api/prototype-image/{gridfs_id}" if gridfs_id else None
                        ),
                        "similarity": round(
                            float(score),
                            6,
                        ),
                        "similarityPercent": round(
                            float(np.clip(score, 0, 1)) * 100,
                            2,
                        ),
                        "globalSimilarity": round(
                            float(global_score),
                            6,
                        ),
                        "patchSimilarity": round(
                            float(patch_score),
                            6,
                        ),
                    }
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                skipped += 1

                app.logger.warning(
                    "Skipping incompatible embedding document: %s",
                    doc.get("_id"),
                )

        results.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        top = results[:TOP_K]

        if not top:
            return (
                jsonify(
                    {
                        "error": (
                            "No compatible six-view embeddings found. "
                            "Check the existing embedding documents."
                        ),
                        "skippedEmbeddings": skipped,
                    }
                ),
                503,
            )

        return jsonify(
            {
                "queryFilename": uploaded.filename,
                "model": MODEL_NAME,
                "count": len(top),
                "matches": top,
                "totalPrototypesScored": len(results),
                "skippedEmbeddings": skipped,
            }
        )

    except Exception:
        app.logger.exception("Matching failed")

        return jsonify({"error": "Matching failed. Check server logs."}), 500


# ============================================================
# SERVE CATALOGUE IMAGES FROM GRIDFS
# ============================================================


@app.get("/api/prototype-image/<gridfs_id>")
def prototype_image(gridfs_id):
    try:
        object_id = ObjectId(gridfs_id)

        with gridfs_bucket.open_download_stream(object_id) as stream:
            raw = stream.read()

        image = load_rgb(raw)

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=90,
        )

        output.seek(0)

        return send_file(
            output,
            mimetype="image/jpeg",
        )

    except Exception:
        app.logger.exception("Unable to retrieve prototype image")

        return jsonify({"error": "Prototype image not found."}), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
