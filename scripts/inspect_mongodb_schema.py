"""
Inspect MongoDB schema for Jewellery Model 3.

Read-only script:
- Does not insert, update, or delete documents.
- Does not delete or modify GridFS images.
- Prints collection names, sample fields, and prototype records.
"""

import os
from collections import Counter

from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFSBucket
from bson import ObjectId

# --------------------------------------------------
# 1. Load MongoDB connection settings
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

load_dotenv(ENV_PATH)

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URL")

DATABASE_NAME = os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DB")

# Use the known project database name if it is not set in .env.
if not DATABASE_NAME:
    DATABASE_NAME = "jewelmatch"


def print_separator():
    print("-" * 75)


def safe_value(value):
    """Convert BSON values to readable text."""
    if isinstance(value, ObjectId):
        return str(value)
    return repr(value)


def main():
    if not MONGO_URI:
        print("ERROR: MongoDB URI was not found in .env.")
        print("Expected MONGO_URI, MONGODB_URI, or MONGO_URL.")
        return

    client = None

    try:
        print_separator()
        print("JEWELLERY MODEL 3 - MONGODB SCHEMA INSPECTION")
        print_separator()
        print(f"Environment file: {ENV_PATH}")
        print(f"Database: {DATABASE_NAME}")
        print("Mode: READ ONLY")
        print()

        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )

        # Verify the connection.
        client.admin.command("ping")
        print("MongoDB connection: SUCCESS")

        db = client[DATABASE_NAME]

        # --------------------------------------------------
        # 2. List collections
        # --------------------------------------------------

        collection_names = db.list_collection_names()

        print()
        print_separator()
        print("COLLECTIONS")
        print_separator()

        if not collection_names:
            print("No collections found in this database.")

        for name in sorted(collection_names):
            collection = db[name]
            try:
                count = collection.count_documents({})
                print(f"{name}: {count} documents")
            except Exception as exc:
                print(f"{name}: unable to count documents ({exc})")

        # --------------------------------------------------
        # 3. Inspect document fields and examples
        # --------------------------------------------------

        print()
        print_separator()
        print("COLLECTION SCHEMA SAMPLES")
        print_separator()

        for name in sorted(collection_names):
            # Skip GridFS binary storage collections in this section.
            if name.endswith(".chunks"):
                continue

            collection = db[name]

            # Inspect a maximum of 3 documents per collection.
            samples = list(collection.find({}).limit(3))

            print(f"\nCollection: {name}")

            if not samples:
                print("  No documents.")
                continue

            all_fields = Counter()
            for document in samples:
                all_fields.update(document.keys())

            print("  Fields seen in sample documents:")
            for field in sorted(all_fields):
                print(f"    - {field}")

            for index, document in enumerate(samples, start=1):
                print(f"\n  Sample document {index}:")
                for key, value in document.items():
                    # Avoid printing connection strings or credentials.
                    if key.lower() in {
                        "password",
                        "token",
                        "secret",
                        "apikey",
                        "api_key",
                    }:
                        value = "[REDACTED]"

                    print(f"    {key}: {safe_value(value)}")

        # --------------------------------------------------
        # 4. Inspect likely prototype catalogue collections
        # --------------------------------------------------

        print()
        print_separator()
        print("LIKELY PROTOTYPE CATALOGUE RECORDS")
        print_separator()

        candidate_names = [
            name
            for name in collection_names
            if not name.endswith((".chunks", ".files"))
            and any(
                word in name.lower()
                for word in (
                    "prototype",
                    "jewellery",
                    "jewelry",
                    "catalogue",
                    "catalog",
                )
            )
        ]

        if not candidate_names:
            print(
                "No collection matched common prototype/catalogue names. "
                "Review the collection list above."
            )

        for name in candidate_names:
            collection = db[name]
            print(f"\nCollection: {name}")

            # Look for likely prototype documents without changing anything.
            query = {
                "$or": [
                    {"type": {"$regex": "LP", "$options": "i"}},
                    {"category": {"$regex": "prototype", "$options": "i"}},
                    {"collection": {"$regex": "prototype", "$options": "i"}},
                ]
            }

            try:
                matches = list(collection.find(query).limit(10))
            except Exception:
                matches = []

            print(f"Sample matching records: {len(matches)}")

            for document in matches:
                print(
                    "  ",
                    {
                        key: safe_value(value)
                        for key, value in document.items()
                        if key.lower()
                        not in {"password", "token", "secret", "apikey", "api_key"}
                    },
                )

        # --------------------------------------------------
        # 5. Inspect GridFS metadata and links
        # --------------------------------------------------

        print()
        print_separator()
        print("GRIDFS IMAGE STORAGE")
        print_separator()

        try:
            fs = GridFSBucket(db, bucket_name="fs")
            files_collection = db["fs.files"]

            total_gridfs_files = files_collection.count_documents({})
            print(f"GridFS bucket: fs")
            print(f"Total GridFS files: {total_gridfs_files}")

            print("\nSample GridFS file metadata:")

            for document in files_collection.find({}).limit(5):
                print(
                    "  ",
                    {
                        key: safe_value(value)
                        for key, value in document.items()
                        if key not in {"metadata"}
                    },
                )

                if document.get("metadata") is not None:
                    print("    metadata:", safe_value(document["metadata"]))

            # Count catalogue documents with common GridFS link fields.
            print("\nPossible GridFS references in catalogue collections:")

            link_fields = {
                "image_id",
                "gridfs_id",
                "file_id",
                "image",
                "image_file_id",
                "gridfs_file_id",
            }

            for name in sorted(collection_names):
                if name.endswith((".chunks", ".files")):
                    continue

                collection = db[name]
                sample = collection.find_one({})

                if not sample:
                    continue

                matching_fields = [
                    key for key in sample.keys() if key.lower() in link_fields
                ]

                if matching_fields:
                    print(f"\n  Collection: {name}")
                    print(f"  Possible link fields: {matching_fields}")

                    for document in collection.find({}).limit(5):
                        print(
                            "    ",
                            {
                                "_id": safe_value(document.get("_id")),
                                **{
                                    field: safe_value(document.get(field))
                                    for field in matching_fields
                                },
                            },
                        )

        except Exception as exc:
            print(f"GridFS inspection warning: {exc}")

        print()
        print_separator()
        print("INSPECTION COMPLETED")
        print("No database records or images were changed.")
        print_separator()

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        print("Check your .env settings, MongoDB connection, and database access.")

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
