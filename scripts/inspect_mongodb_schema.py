import os
from collections import Counter
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFSBucket


def main():
    load_dotenv(override=True)

    uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE", "jewelmatch")
    bucket_name = os.getenv("GRIDFS_BUCKET", "fs")

    if not uri:
        raise RuntimeError("MONGODB_URI is missing. Check your local .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=10000)

    try:
        client.admin.command("ping")
        db = client[database_name]

        print(f"Connected to database: {database_name}")
        print("Read-only inspection; no records will be changed.")

        # Inspect the prototype collection without printing document values.
        collection = db["prototype"]
        total = collection.count_documents({})

        print(f"\n=== PROTOTYPE COLLECTION: {total} DOCUMENTS ===")

        key_counts = Counter()
        type_counts = Counter()

        for doc in collection.find({}, {"_id": 1}).limit(500):
            full_doc = collection.find_one({"_id": doc["_id"]})
            if not full_doc:
                continue

            for key, value in full_doc.items():
                key_counts[key] += 1
                type_counts[(key, type(value).__name__)] += 1

        print("Field presence counts:")
        for key, count in sorted(key_counts.items()):
            print(f"  {key}: present in {count} documents")

        print("\nField types:")
        for (key, value_type), count in sorted(type_counts.items()):
            print(f"  {key} [{value_type}]: {count} documents")

        # Inspect GridFS metadata only; do not download image bytes.
        files_collection = db[f"{bucket_name}.files"]
        gridfs_count = files_collection.count_documents({})

        print(f"\n=== GRIDFS BUCKET '{bucket_name}': {gridfs_count} FILE RECORDS ===")

        for record in files_collection.find(
            {}, {"filename": 1, "metadata": 1, "contentType": 1}
        ).limit(1000):
            metadata = record.get("metadata") or {}
            print(
                "filename={!r}; metadata_keys={}; contentType={!r}".format(
                    record.get("filename"),
                    sorted(metadata.keys()),
                    record.get("contentType"),
                )
            )

        print("\nInspection complete.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
