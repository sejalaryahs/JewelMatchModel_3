
import csv
import hashlib
import os
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

LOCAL_DIRS = {
    "gold": ROOT / "data" / "raw" / "gold",
    "prototype": ROOT / "data" / "raw" / "prototype",
}
OUTPUT = ROOT / "data" / "metadata" / "image_manifest.csv"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

FIELDS = [
    "category", "source", "filename", "item_id", "design_id",
    "local_path", "gridfs_file_id", "gridfs_bucket", "sha256",
    "width", "height", "format", "duplicate_count", "status",
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def get_category(metadata, filename):
    aliases = {
        "gold": "gold",
        "gold jewellery": "gold",
        "gold_jewellery": "gold",
        "prototype": "prototype",
        "prototypes": "prototype",
        "prototype jewellery": "prototype",
        "prototype_jewellery": "prototype",
    }

    for key in ("category", "type", "catalogue_type"):
        value = aliases.get(str(metadata.get(key, "")).strip().lower())
        if value:
            return value

    # Support names such as gold/J001.jpg or prototype/J017.jpg.
    parts = filename.replace("\\", "/").split("/")
    return aliases.get(parts[0].strip().lower(), "")


def build_record(category, source, filename, data, **extra):
    record = {field: "" for field in FIELDS}
    record.update({
        "category": category,
        "source": source,
        "filename": filename,
        "item_id": Path(filename).stem,
        "design_id": "",
        "duplicate_count": 0,
        "status": "valid",
        **extra,
    })
    record["sha256"] = sha256(data)

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            record["width"], record["height"] = image.size
            record["format"] = image.format or ""

        if not record["category"]:
            record["status"] = "needs_category"
        elif record["width"] < 1 or record["height"] < 1:
            record["status"] = "invalid_dimensions"

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        record["status"] = f"invalid_image: {exc}"

    return record


def scan_local():
    records = []

    for category, folder in LOCAL_DIRS.items():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n[LOCAL] {category}: {folder}")

        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name.lower() == ".gitkeep":
                continue

            # Inspect extensionless files too. Pillow will verify the
            # actual contents instead of trusting the filename extension.
            try:
                data = path.read_bytes()
                record = build_record(
                    category, "local", path.name, data,
                    item_id=path.stem,
                    local_path=str(path.resolve()),
                )
                records.append(record)
                print(f"  {record['status']}: {path.name}")

            except OSError as exc:
                print(f"  READ ERROR: {path.name}: {exc}")

    return records


def scan_gridfs():
    uri = os.getenv("MONGODB_URI", "").strip()
    db_name = os.getenv("MONGODB_DATABASE", "").strip()
    bucket = os.getenv("GRIDFS_BUCKET", "fs").strip() or "fs"

    if not uri or not db_name:
        print("\n[GRIDFS] Skipped: configure MONGODB_URI and "
              "MONGODB_DATABASE in .env.")
        return []

    records = []
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)

    try:
        client.admin.command("ping")
        db = client[db_name]
        gridfs = GridFS(db, collection=bucket)

        print(f"\n[GRIDFS] Database: {db_name}; bucket: {bucket}")

        for file in gridfs.find():
            filename = str(file.filename or "")
            if Path(filename).suffix.lower() not in EXTENSIONS:
                continue

            metadata = file.metadata or {}
            category = get_category(metadata, filename)

            try:
                data = file.read()
                record = build_record(
                    category, "gridfs", filename, data,
                    item_id=str(metadata.get("item_id", Path(filename).stem)),
                    design_id=str(metadata.get("design_id", "")),
                    gridfs_file_id=str(file._id),
                    gridfs_bucket=bucket,
                )
                records.append(record)
                print(f"  {record['status']}: {filename} "
                      f"(category={category or 'unknown'})")

            except Exception as exc:
                print(f"  READ ERROR: {filename}: {exc}")

    finally:
        client.close()

    return records


def main():
    records = scan_local()
    records.extend(scan_gridfs())

    # Mark exact duplicate content for review; do not delete or skip it.
    counts = Counter(
        row["sha256"] for row in records if row["sha256"]
    )
    for row in records:
        row["duplicate_count"] = counts.get(row["sha256"], 0)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 55)
    print("JEWELLERY MODEL 3 - DATASET INVENTORY")
    print("=" * 55)
    print(f"Total records: {len(records)}")
    print("Valid categorized:", sum(r["status"] == "valid" for r in records))
    print("Needs category:", sum(r["status"] == "needs_category" for r in records))
    print("Invalid images:", sum(r["status"].startswith("invalid_image") for r in records))
    print("Exact duplicate groups:",
          sum(n > 1 for n in counts.values()))
    print("Manifest:", OUTPUT)
    print("Original images were not modified or deleted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Inventory failed: {exc}", file=sys.stderr)
        sys.exit(1)
