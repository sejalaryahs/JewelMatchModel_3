import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "prototype_unique_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "prototype_catalogue_review.csv"


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("The prototype manifest contains no records.")

    # Only IDs explicitly present in the manifest are preserved.
    # Never turn a filename into a catalogue ID automatically.
    output_rows = []

    for row in rows:
        existing_id = (row.get("item_id") or "").strip()

        # GridFS records already linked to MongoDB catalogue items
        # can preserve their known IDs.
        if existing_id:
            status = "existing_catalogue_id"
            confirm_id = existing_id
        else:
            status = "needs_catalogue_id_review"
            confirm_id = ""

        output_rows.append(
            {
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "filename": row.get("filename", ""),
                "existing_item_id": existing_id,
                "gridfs_file_id": row.get("gridfs_file_id", ""),
                "gridfs_bucket": row.get("gridfs_bucket", ""),
                "local_path": row.get("local_path", ""),
                "sha256": row.get("sha256", ""),
                "catalogue_id_to_confirm": confirm_id,
                "review_status": status,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "category",
        "source",
        "filename",
        "existing_item_id",
        "gridfs_file_id",
        "gridfs_bucket",
        "local_path",
        "sha256",
        "catalogue_id_to_confirm",
        "review_status",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    existing_count = sum(
        row["review_status"] == "existing_catalogue_id" for row in output_rows
    )
    review_count = sum(
        row["review_status"] == "needs_catalogue_id_review" for row in output_rows
    )

    print("=== PROTOTYPE CATALOGUE REVIEW ===")
    print(f"Total unique image records: {len(output_rows)}")
    print(f"Records with manifest item IDs: {existing_count}")
    print(f"Records needing ID review: {review_count}")
    print(f"Created: {OUTPUT_PATH}")
    print("No source images or database records were changed.")


if __name__ == "__main__":
    main()
