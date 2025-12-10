"""
Remove duplicate images across train/val/test splits by hashing file contents.

Default behavior:
- Keeps one copy per hash using split priority: train > val > test
- Deletes other copies and logs what was removed

Usage:
    python3 scripts/cleanup_duplicates.py --data_dir data --report results/duplicate_report.json --dry_run
    python3 scripts/cleanup_duplicates.py --data_dir data --report results/duplicate_report.json
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_PRIORITY = ["train", "val", "test"]  # order to keep


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


def collect_hashes(data_dir: Path) -> Dict[str, List[Path]]:
    """Walk splits and collect hashes."""
    hash_map: Dict[str, List[Path]] = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for img_path in split_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            file_hash = hash_file(img_path)
            hash_map.setdefault(file_hash, []).append(img_path)
    return hash_map


def select_keep_and_remove(paths: List[Path]) -> Tuple[Path, List[Path]]:
    """
    Choose which path to keep based on split priority, return (keep, remove_list).
    """
    if not paths:
        return None, []

    def priority(p: Path) -> Tuple[int, str]:
        rel_parts = p.parts
        # find split in parts
        split = None
        for part in rel_parts:
            if part in SPLIT_PRIORITY:
                split = part
                break
        idx = SPLIT_PRIORITY.index(split) if split in SPLIT_PRIORITY else len(SPLIT_PRIORITY)
        return idx, str(p)

    paths_sorted = sorted(paths, key=priority)
    keep = paths_sorted[0]
    remove = paths_sorted[1:]
    return keep, remove


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate images across splits.")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root with train/val/test")
    parser.add_argument("--report", type=str, default="results/duplicate_cleanup.json", help="Where to save cleanup report")
    parser.add_argument("--dry_run", action="store_true", help="Do not delete files, just report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    print(f"Hashing images under {data_dir} ...")
    hash_map = collect_hashes(data_dir)

    duplicates_info = []
    total_removed = 0

    for file_hash, paths in hash_map.items():
        if len(paths) <= 1:
            continue
        keep, remove_list = select_keep_and_remove(paths)
        if not keep:
            continue
        for rm in remove_list:
            total_removed += 1
            if not args.dry_run:
                try:
                    rm.unlink()
                except Exception as e:
                    print(f"Failed to delete {rm}: {e}")
        duplicates_info.append(
            {
                "hash": file_hash,
                "keep": str(keep.relative_to(data_dir)),
                "removed": [str(r.relative_to(data_dir)) for r in remove_list],
            }
        )

    summary = {
        "total_hashes": len(hash_map),
        "total_duplicates": len(duplicates_info),
        "files_removed": total_removed if not args.dry_run else 0,
        "dry_run": args.dry_run,
    }

    print("\nSummary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    output_path = Path(args.report)
    # If a directory is provided, write a default filename inside it
    if output_path.is_dir():
        output_path = output_path / "duplicate_cleanup.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "details": duplicates_info}
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {output_path}")

    if args.dry_run and duplicates_info:
        print("\nDry run: no files deleted. Re-run without --dry_run to apply.")


if __name__ == "__main__":
    main()

