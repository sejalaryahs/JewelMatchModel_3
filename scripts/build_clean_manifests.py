import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "metadata" / "image_manifest_resolved.csv"
OUTPUT_DIR = ROOT / "data" / "metadata"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_priority(row):
    # Prefer a GridFS prototype image over a local copy of identical content.
    # This does not assign a catalogue ID where one is not known.
    source = row.get("source", "").lower()
    filename = row.get("filename", "").replace("\\", "/").lower()

    if source == "gridfs" and filename.startswith("prototype/"):
        return 0
    if source == "gridfs":
        return 1
    if source == "local":
        return 2
    return 3


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Resolved manifest not found: {INPUT}")

    fieldnames, rows = read_csv(INPUT)

    gold_rows = []
    prototype_candidates = []
    review_rows = []

    for row in rows:
        category = row.get("category", "").strip().lower()
        status = row.get("status", "").strip().lower()

        if category == "gold":
            gold_rows.append(row)
        elif category == "prototype":
            prototype_candidates.append(row)
        else:
            review_rows.append(row)

    # Deduplicate prototype image content by SHA-256.
    # Keep the preferred source record and preserve other locations in a
    # separate duplicate report. Records with no hash are not silently merged.
    groups = {}
    unhashed = []

    for row in prototype_candidates:
        digest = row.get("sha256", "").strip()

        if not digest:
            unhashed.append(row)
            continue

        groups.setdefault(digest, []).append(row)

    unique_prototypes = []
    duplicate_rows = []

    for digest, group in groups.items():
        ordered = sorted(group, key=source_priority)
        canonical = ordered[0].copy()
        canonical["duplicate_count"] = str(len(group))
        unique_prototypes.append(canonical)

        for duplicate in ordered[1:]:
            duplicate_copy = duplicate.copy()
            duplicate_copy["duplicate_of_sha256"] = digest
            duplicate_rows.append(duplicate_copy)

    for row in unhashed:
        row_copy = row.copy()
        row_copy["duplicate_count"] = "1"
        unique_prototypes.append(row_copy)

    # Stable output order for easy review.
    unique_prototypes.sort(
        key=lambda row: (
            row.get("filename", "").lower(),
            row.get("sha256", ""),
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        OUTPUT_DIR / "gold_query_manifest.csv",
        fieldnames,
        gold_rows,
    )
    write_csv(
        OUTPUT_DIR / "prototype_unique_manifest.csv",
        fieldnames,
        unique_prototypes,
    )

    duplicate_fields = list(fieldnames)
    if "duplicate_of_sha256" not in duplicate_fields:
        duplicate_fields.append("duplicate_of_sha256")

    write_csv(
        OUTPUT_DIR / "prototype_exact_duplicates.csv",
        duplicate_fields,
        duplicate_rows,
    )
    write_csv(
        OUTPUT_DIR / "category_review_required.csv",
        fieldnames,
        review_rows,
    )

    print("\n=== CLEAN MANIFEST SUMMARY ===")
    print(f"Gold query records: {len(gold_rows)}")
    print(f"Prototype records before deduplication: {len(prototype_candidates)}")
    print(f"Unique prototype image contents: {len(unique_prototypes)}")
    print(f"Duplicate prototype records listed separately: {len(duplicate_rows)}")
    print(f"Records needing category review: {len(review_rows)}")

    print("\nCreated:")
    print("  data/metadata/gold_query_manifest.csv")
    print("  data/metadata/prototype_unique_manifest.csv")
    print("  data/metadata/prototype_exact_duplicates.csv")
    print("  data/metadata/category_review_required.csv")
    print("\nNo images or MongoDB records were changed.")


if __name__ == "__main__":
    main()
