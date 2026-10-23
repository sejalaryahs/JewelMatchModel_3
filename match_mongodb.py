import os
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import AutoImageProcessor, AutoModel

# ============================================================
# CONFIGURATION - EXISTING JEWELLERY MODEL 3
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "facebook/dinov2-small"
DATABASE_NAME = os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DB") or "jewelmatch"

EMBEDDING_COLLECTION = "prototype_dinov2_embeddings_v1"

TOP_K = 8
DEVICE = torch.device("cpu")

GLOBAL_WEIGHT = 0.60
PATCH_WEIGHT = 0.40


def get_mongo_uri():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URL")

    if not uri:
        raise RuntimeError("MongoDB URI not found. Check the existing .env file.")

    return uri


# ============================================================
# IMAGE LOADING
# ============================================================


def load_query_image(image_path):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def load_query_image_bytes(image_bytes):
    """Load an image upload without saving it to disk."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


# ============================================================
# DINOv2 FEATURE EXTRACTION
# ============================================================


def create_features(image, processor, model):
    """
    Returns:
      global_vector: normalized CLS embedding, shape (384,)
      patch_vectors: normalized patch embeddings, shape (N, 384)
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        tokens = outputs.last_hidden_state

        global_embedding = F.normalize(
            tokens[:, 0, :],
            p=2,
            dim=-1,
        )

        patch_embeddings = F.normalize(
            tokens[:, 1:, :],
            p=2,
            dim=-1,
        )

    global_vector = global_embedding[0].cpu().numpy().astype(np.float32)

    patch_vectors = patch_embeddings[0].cpu().numpy().astype(np.float32)

    if global_vector.shape != (384,):
        raise RuntimeError(f"Expected 384 dimensions, got {global_vector.shape}")

    return global_vector, patch_vectors


# Keep this function for compatibility with any existing imports.
def create_embedding(image, processor, model):
    global_vector, _ = create_features(image, processor, model)
    return global_vector


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================


def cosine_similarity(vector_a, vector_b):
    a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vector_b, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        return None

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-12:
        return None

    return float(np.dot(a, b) / denominator)


def patch_similarity(query_patches, stored_patches):
    """
    Symmetric local matching.

    Every query patch finds its closest stored patch, and every stored
    patch finds its closest query patch. Both directions contribute.
    """
    try:
        query_patches = np.asarray(query_patches, dtype=np.float32)
        stored_patches = np.asarray(stored_patches, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if query_patches.ndim != 2 or stored_patches.ndim != 2:
        return None

    if len(query_patches) == 0 or len(stored_patches) == 0:
        return None

    if query_patches.shape[1] != stored_patches.shape[1]:
        return None

    # Ensure vectors are normalized even if stored data is not.
    query_norms = np.linalg.norm(query_patches, axis=1, keepdims=True)
    stored_norms = np.linalg.norm(stored_patches, axis=1, keepdims=True)

    query_patches = query_patches / np.maximum(query_norms, 1e-12)
    stored_patches = stored_patches / np.maximum(stored_norms, 1e-12)

    similarity_matrix = query_patches @ stored_patches.T

    query_to_stored = float(np.max(similarity_matrix, axis=1).mean())
    stored_to_query = float(np.max(similarity_matrix, axis=0).mean())

    return (query_to_stored + stored_to_query) / 2.0


def similarity_percent(score):
    # Ranking score, not a probability of a correct match.
    return round(float(np.clip(score, 0.0, 1.0)) * 100, 2)


# ============================================================
# MATCHING
# ============================================================


def find_matches(query_image, processor, model, collection):
    query_global, query_patches = create_features(query_image, processor, model)

    records = list(
        collection.find(
            {
                "model_name": MODEL_NAME,
                "embedding_dimension": 384,
            },
            {
                "_id": 0,
                "catalogue_id": 1,
                "catalogue_name": 1,
                "catalogue_type": 1,
                "catalogue_collection": 1,
                "filename": 1,
                "image_path": 1,
                "image_gridfs_id": 1,
                "embedding": 1,
                "global_embedding": 1,
                "patch_embeddings": 1,
            },
        )
    )

    if not records:
        raise RuntimeError(
            "No DINOv2 embeddings found in "
            f"{EMBEDDING_COLLECTION}. Check the builder and database."
        )

    results = []

    for record in records:
        stored_global = record.get("global_embedding")

        if stored_global is None:
            stored_global = record.get("embedding", [])

        try:
            stored_global = np.asarray(stored_global, dtype=np.float32)
        except (TypeError, ValueError):
            continue

        if stored_global.shape != (384,):
            print(
                "Skipping invalid global embedding for "
                f"{record.get('catalogue_id', 'unknown')}"
            )
            continue

        global_score = cosine_similarity(query_global, stored_global)

        if global_score is None:
            continue

        stored_patches = record.get("patch_embeddings")
        local_score = None

        if stored_patches is not None:
            local_score = patch_similarity(query_patches, stored_patches)

        if local_score is None:
            final_score = global_score
            method = "global_only"
        else:
            final_score = GLOBAL_WEIGHT * global_score + PATCH_WEIGHT * local_score
            method = "global_plus_patches"

        results.append(
            {
                "catalogue_id": record.get("catalogue_id", ""),
                "catalogue_name": record.get("catalogue_name", ""),
                "catalogue_type": record.get("catalogue_type", ""),
                "catalogue_collection": record.get("catalogue_collection", ""),
                "filename": record.get("filename", ""),
                "image_path": record.get("image_path", ""),
                "image_gridfs_id": str(record.get("image_gridfs_id", "")),
                "similarity": float(final_score),
                "similarity_percent": similarity_percent(final_score),
                "global_similarity": float(global_score),
                "patch_similarity": (
                    float(local_score) if local_score is not None else None
                ),
                "matching_method": method,
            }
        )

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:TOP_K]


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python match_mongodb.py "path\\to\\gold_query.jpg"')
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print(f"ERROR: Query image not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("JEWELLERY MODEL 3 - IMPROVED DINOv2 MATCHER")
    print("=" * 70)
    print(f"Query image: {image_path.name}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Embedding collection: {EMBEDDING_COLLECTION}")
    print(f"Device: {DEVICE}")

    client = MongoClient(
        get_mongo_uri(),
        serverSelectionTimeoutMS=10000,
    )

    try:
        client.admin.command("ping")
        collection = client[DATABASE_NAME][EMBEDDING_COLLECTION]

        print("\nLoading DINOv2 Small...")
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

        query_image = load_query_image(image_path)

        print("Generating query features...")
        matches = find_matches(
            query_image,
            processor,
            model,
            collection,
        )

        print("\nTOP MATCHES")
        print("-" * 70)

        for rank, result in enumerate(matches, start=1):
            print(f"\nRank: {rank}")
            print(f"Catalogue ID: {result['catalogue_id']}")
            print(f"Name: {result['catalogue_name']}")
            print(f"Type: {result['catalogue_type']}")
            print(f"Image: {result['filename']}")
            print(
                f"Combined score: {result['similarity']:.4f} "
                f"({result['similarity_percent']:.2f}%)"
            )
            print(f"Global score: {result['global_similarity']:.4f}")

            if result["patch_similarity"] is not None:
                print(f"Patch score: {result['patch_similarity']:.4f}")
            else:
                print("Patch score: unavailable; using global score")

            print(f"Method: {result['matching_method']}")
            print(f"GridFS ID: {result['image_gridfs_id']}")

        print("\nQuery image processed in memory.")
        print("No query image was saved to MongoDB.")

        if matches and all(
            result["matching_method"] == "global_only" for result in matches
        ):
            print(
                "\nNOTE: Patch features are not available for these "
                "results. Update the existing embedding builder before "
                "expecting local patch matching."
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()
