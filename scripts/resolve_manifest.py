import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "metadata" / "image_manifest.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "metadata" / "image_manifest_resolved.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError("The manifest has no CSV header.")

    # Index known prototype image hashes.
    prototype_hashes = {
        row["sha256"]
        for row in rows
        if row.get("category", "").strip().lower() == "prototype"
        and row.get("sha256", "").strip()
    }

    resolved_count = 0
    still_unknown = []

    for row in rows:
        if row.get("source") == "gridfs" and row.get("status") == "needs_category":
            image_hash = row.get("sha256", "").strip()

            if image_hash and image_hash in prototype_hashes:
                row["category"] = "prototype"
                row["status"] = "duplicate_content_category_resolved"
                resolved_count += 1
            else:
                still_unknown.append(row.get("filename", ""))

    # Save to a separate file. Never overwrite the original manifest.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=== MANIFEST RESOLUTION SUMMARY ===")
    print(f"Original records: {len(rows)}")
    print(f"Exact-match categories resolved: {resolved_count}")
    print(f"Records still needing category review: {len(still_unknown)}")

    for filename in still_unknown:
        print(f"  REVIEW: {filename}")

    print(f"\nNew manifest: {OUTPUT_FILE}")
    print("Original manifest and images were not modified.")


if __name__ == "__main__":
    main()
