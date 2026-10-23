import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URL")
db_name = os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DB") or "jewelmatch"

if not uri:
    raise RuntimeError("MongoDB URI is missing from .env")

client = MongoClient(uri, serverSelectionTimeoutMS=10000)

try:
    client.admin.command("ping")
    db = client[db_name]

    print(f"Database: {db_name}")
    print(f"Prototype documents: {db['prototype'].count_documents({})}")
    print(f"GridFS files: {db['fs.files'].count_documents({})}")

    print("\n--- PROTOTYPE DOCUMENT SAMPLES ---")
    for doc in db["prototype"].find({}).limit(5):
        # Exclude large legacy embedding fields from the output.
        sample = {k: v for k, v in doc.items() if "embedding" not in k.lower()}
        print(sample)

    print("\n--- GRIDFS PROTOTYPE FILES ---")
    query = {
        "collection": {
            "$regex": "^prototype$",
            "$options": "i",
        }
    }

    files = list(db["fs.files"].find(query).limit(50))
    print(f"Files with collection=prototype: {len(files)}")

    for f in files:
        print(
            {
                "_id": str(f["_id"]),
                "filename": f.get("filename"),
                "jewellery_id": f.get("jewellery_id"),
                "jewellery_type": f.get("jewellery_type"),
                "image_path": f.get("image_path"),
                "collection": f.get("collection"),
            }
        )

    print("\n--- MATCHING CATALOGUE IDS ---")
    catalogue_ids = {
        str(doc.get("jewellery_id"))
        for doc in db["prototype"].find(
            {"jewellery_id": {"$exists": True}},
            {"jewellery_id": 1},
        )
    }

    linked = 0
    for f in db["fs.files"].find(query):
        jewellery_id = f.get("jewellery_id")
        if jewellery_id is not None and str(jewellery_id) in catalogue_ids:
            linked += 1

    print(f"GridFS files whose jewellery_id appears in prototype: {linked}")

    print("\nRead-only inspection completed. No data was changed.")

finally:
    client.close()
