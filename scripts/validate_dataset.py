from pathlib import Path
from collections import defaultdict
import hashlib
import sys

from PIL import Image, UnidentifiedImageError
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file for exact duplicate detection."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    if not CONFIG_PATH.is_file():
        print(f"ERROR: Configuration file not found: {CONFIG_PATH}")
        return 1

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    allowed_extensions = {
        extension.lower()
        for extension in data_config["allowed_extensions"]
    }

    folders = {
        "gold": PROJECT_ROOT / data_config["raw_gold_dir"],
        "prototype": PROJECT_ROOT / data_config["raw_prototype_dir"],
    }

    errors = []
    duplicates = defaultdict(list)
    total_images = 0

    print("=" * 65)
    print("JEWELLERY MODEL 3 - DATASET VALIDATION")
    print("=" * 65)

    for category, folder in folders.items():
        print(f"\n[{category.upper()}]")
        print(f"Folder: {folder}")

        if not folder.is_dir():
            message = f"Dataset folder does not exist: {folder}"
            print(f"ERROR: {message}")
            errors.append(message)
            continue

        files = sorted(
            path for path in folder.rglob("*")
            if path.is_file()
            and path.name.lower() != ".gitkeep"
        )

        if not files:
            print("No files found. This is acceptable for an empty dataset.")
            continue

        valid_count = 0

        for path in files:
            total_images += 1

            if path.suffix.lower() not in allowed_extensions:
                message = f"Unsupported extension: {path.relative_to(PROJECT_ROOT)}"
                print(f"ERROR: {message}")
                errors.append(message)
                continue

            try:
                # Verify the file structure first.
                with Image.open(path) as image:
                    image.verify()

                # Reopen the file to inspect its dimensions.
                with Image.open(path) as image:
                    width, height = image.size
                    image_format = image.format

                if width < 1 or height < 1:
                    message = f"Invalid image dimensions: {path}"
                    print(f"ERROR: {message}")
                    errors.append(message)
                    continue

                file_hash = sha256_file(path)
                duplicates[file_hash].append(
                    (category, path, width, height)
                )

                valid_count += 1
                print(
                    f"OK: {path.relative_to(PROJECT_ROOT)} "
                    f"| {width}x{height} | {image_format}"
                )

            except (UnidentifiedImageError, OSError, ValueError) as exc:
                message = f"Unreadable image: {path} ({exc})"
                print(f"ERROR: {message}")
                errors.append(message)

        print(f"Files checked: {len(files)}")
        print(f"Valid images: {valid_count}")

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    print(f"Files encountered: {total_images}")
    print(f"Exact duplicate groups: {sum(1 for items in duplicates.values() if len(items) > 1)}")

    for items in duplicates.values():
        if len(items) > 1:
            print("\nPotential exact duplicates:")
            for category, path, width, height in items:
                print(f"  [{category}] {path.relative_to(PROJECT_ROOT)}")

    if errors:
        print(f"\nValidation errors: {len(errors)}")
        print("Review the errors above before using these files.")
        return 1

    print("\nNo validation errors found.")
    print("Note: duplicate files are reported, never deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
