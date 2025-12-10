"""
Check for duplicate images across train/val/test splits by hashing file contents.

Usage:
    python3 scripts/check_duplicates.py --data_dir data --output results/duplicate_report.json
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def hash_file(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA1 hash of a file."""
    sha1 = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def collect_hashes(data_dir: Path) -> Dict[str, List[str]]:
    """
    Walk train/val/test and collect hashes.

    Returns:
        dict mapping hash -> list of relative paths
    """
    hash_map: Dict[str, List[str]] = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for img_path in split_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            file_hash = hash_file(img_path)
            rel = img_path.relative_to(data_dir).as_posix()
            hash_map.setdefault(file_hash, []).append(rel)
    return hash_map


def main():
    parser = argparse.ArgumentParser(description="Check duplicate images across splits.")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root with train/val/test")
    parser.add_argument("--output", type=str, default="results/duplicate_report.json", help="Where to save the report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    print(f"Hashing images under {data_dir} ...")
    hash_map = collect_hashes(data_dir)

    # Find duplicates (hash appears more than once, especially across splits)
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    summary = {
        "total_images": sum(len(v) for v in hash_map.values()),
        "unique_images": len(hash_map),
        "duplicate_hashes": len(duplicates),
    }

    print("\nSummary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    # Write detailed report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "duplicates": duplicates}
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {output_path}")

    # If duplicates exist, print a few examples
    if duplicates:
        print("\nSample duplicates (up to 5 hashes):")
        for i, (h, paths) in enumerate(duplicates.items()):
            if i >= 5:
                break
            print(f"- hash {h[:10]}... ({len(paths)} files)")
            for p in paths[:5]:
                print(f"  * {p}")


if __name__ == "__main__":
    main()

