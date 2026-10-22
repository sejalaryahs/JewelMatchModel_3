import os
from dotenv import load_dotenv
from pymongo import MongoClient


def main():
    # Load MongoDB settings from the project's .env file
    load_dotenv(override=True)

    uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE", "jewelmatch")
    bucket_name = os.getenv("GRIDFS_BUCKET", "fs")

    if not uri:
        raise RuntimeError("MONGODB_URI is missing. Check your .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=10000)

    try:
        # Test the MongoDB connection
        client.admin.command("ping")
        db = client[database_name]

        print(f"Connected to database: {database_name}")
        print("Read-only inspection: no records will be changed.")

        # Read GridFS file IDs and filenames
        files_collection = db[f"{bucket_name}.files"]

        gridfs_files = {
            str(record["_id"]): record.get("filename", "")
            for record in files_collection.find({}, {"filename": 1})
        }

        # Read prototype catalogue documents
        prototype_collection = db["prototype"]
        total_documents = prototype_collection.count_documents({})

        print(f"\nGridFS file records: {len(gridfs_files)}")
        print(f"Prototype documents: {total_documents}")

        linked_count = 0
        missing_count = 0

        print("\n=== PROTOTYPE DOCUMENT TO GRIDFS LINKS ===")

        fields = {
            "_id": 1,
            "id": 1,
            "name": 1,
            "type": 1,
            "collection": 1,
            "filename": 1,
            "image_path": 1,
            "image_gridfs_id": 1,
        }

        for doc in prototype_collection.find({}, fields).sort("id", 1):
            gridfs_id = doc.get("image_gridfs_id")

            if gridfs_id is not None:
                gridfs_id_text = str(gridfs_id)
                matched_filename = gridfs_files.get(gridfs_id_text)
            else:
                matched_filename = None

            if matched_filename:
                linked_count += 1
                status = "LINKED"
            else:
                missing_count += 1
                status = "NO MATCHING GRIDFS FILE ID"

            print(f"\nStatus: {status}")
            print(f"  Catalogue ID: {doc.get('id', '')}")
            print(f"  Name: {doc.get('name', '')}")
            print(f"  Type: {doc.get('type', '')}")
            print(f"  Collection: {doc.get('collection', '')}")
            print(f"  Document filename: {doc.get('filename', '')}")
            print(f"  Image path: {doc.get('image_path', '')}")
            print(f"  GridFS filename: " f"{matched_filename or '(not found)'}")

        print("\n=== SUMMARY ===")
        print(f"Total prototype documents: {total_documents}")
        print(f"Documents linked to GridFS files: {linked_count}")
        print(f"Documents without matching GridFS IDs: {missing_count}")
        print("\nInspection complete. No data was modified.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
