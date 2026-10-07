# ---------------------------------------------------------------------------
# Check that a data directory matches the project's dataset layout.
#
# Expected layout (torchvision ImageFolder style, see const.py):
#
#     <data-dir>/
#         train/0/...      train/1/...
#         validate/0/...   validate/1/...
#         test/0/...       test/1/...
#
# Every split (TRAIN_DIRNAME/VALIDATE_DIRNAME/TEST_DIRNAME) must exist with a
# "0" and a "1" folder (CLASS_NAMES), and each of those must hold at least one
# image. Corrupted images are detected with Pillow (open + full decode); with
# --delete they are removed after the scan, never while iterating.
#
# Usage:
#     python check_dataset.py --data-dir <dir> [--delete]
#
# Options:
#     --data-dir PATH   required: dataset root holding train/validate/test
#     --delete          delete the corrupted images found while scanning
#
# Exit status:
#     0  layout is valid (and, with --delete, no corrupted images remain)
#     1  invalid layout
#
# Examples:
#     python check_dataset.py --data-dir ../data
#     python check_dataset.py --data-dir ../data --delete
# ---------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError

from const import CLASS_NAMES, TEST_DIRNAME, TRAIN_DIRNAME, VALIDATE_DIRNAME, configure_logging, log
from utils import existing_dir

IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_DIRNAMES: Tuple[str, ...] = (TRAIN_DIRNAME, VALIDATE_DIRNAME, TEST_DIRNAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the dataset layout and optionally drop corrupted images."
    )
    parser.add_argument(
        "--data-dir",
        type=existing_dir,
        required=True,
        help="dataset root holding train/validate/test",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="delete the corrupted images found while scanning",
    )
    return parser.parse_args()


def list_images(directory: Path) -> List[Path]:
    """Every image-looking file directly inside directory, sorted."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def is_image_ok(path: Path) -> bool:
    """Fully decode with Pillow; returns False when the file is corrupted."""
    try:
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        log.warning("corrupted image %s: %s", path, exc)
        return False
    return True


def check_structure(data_dir: Path) -> bool:
    """Verify <split>/<class>/ exists for every split and class, non-empty."""
    valid: bool = True

    for split_dirname in SPLIT_DIRNAMES:
        split_dir: Path = data_dir / split_dirname
        if not split_dir.is_dir():
            log.error("missing split directory: %s", split_dir)
            valid = False
            continue

        for class_name in CLASS_NAMES:
            class_dir: Path = split_dir / class_name
            if not class_dir.is_dir():
                log.error("missing class directory: %s", class_dir)
                valid = False
                continue

            images: List[Path] = list_images(class_dir)
            if not images:
                log.error("no images in %s", class_dir)
                valid = False
            else:
                log.info("%s: %d image(s)", class_dir, len(images))

    return valid


def remove_corrupted(data_dir: Path) -> int:
    """Find corrupted images, then delete them after the scan. Returns the count."""
    corrupted: List[Path] = []

    # Collect first, delete later, so no directory is modified while iterating.
    for split_dirname in SPLIT_DIRNAMES:
        for class_name in CLASS_NAMES:
            class_dir: Path = data_dir / split_dirname / class_name
            if not class_dir.is_dir():
                continue
            for image_path in list_images(class_dir):
                if not is_image_ok(image_path):
                    corrupted.append(image_path)

    if not corrupted:
        log.info("no corrupted images found")
        return 0

    # Delete the marked files only now that the scan is complete.
    log.warning("deleting %d corrupted image(s)", len(corrupted))
    for path in corrupted:
        try:
            path.unlink()
            log.info("deleted %s", path)
        except OSError as exc:
            log.error("could not delete %s: %s", path, exc)

    return len(corrupted)


def main() -> None:
    args = parse_args()

    configure_logging()

    data_dir: Path = args.data_dir
    log.info("data_dir=%s delete=%s", data_dir, args.delete)

    if not check_structure(data_dir):
        log.error("invalid dataset structure")
        sys.exit(1)

    if args.delete:
        removed: int = remove_corrupted(data_dir)
        log.info("removed %d corrupted image(s)", removed)
    else:
        log.info("skipping the image scan (pass --delete to scan and remove corrupted images)")

    log.info("dataset OK")


if __name__ == "__main__":
    main()
