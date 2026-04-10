from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def ensure_link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copytree(src, dst)
    else:
        os.symlink(src, dst, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=str,
        default=str(Path.home() / "webshop_data"),
        help="Existing WebShop asset root containing data/ and search_engine/.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy assets instead of creating symlinks. This may require >11GB.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    assets_root = repo_root / "assets" / "webshop"
    src_root = Path(args.source_root).resolve()
    src_data = src_root / "data"
    src_search = src_root / "search_engine"
    if not src_data.exists():
        src_data = src_root
    if not src_search.exists():
        src_search = src_root
    if not src_data.exists():
        raise FileNotFoundError(f"Missing source data directory: {src_data}")
    if not src_search.exists():
        raise FileNotFoundError(f"Missing source search_engine directory: {src_search}")

    dst_data = assets_root / "data"
    dst_search = assets_root / "search_engine"
    ensure_link_or_copy(src_data, dst_data, copy_files=args.copy)
    ensure_link_or_copy(src_search, dst_search, copy_files=args.copy)

    print(f"assets_root={assets_root}")
    print(f"data={'copy' if args.copy else 'symlink'} -> {dst_data}")
    print(f"search_engine={'copy' if args.copy else 'symlink'} -> {dst_search}")


if __name__ == "__main__":
    main()
