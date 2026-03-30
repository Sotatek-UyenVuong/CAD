"""
prepare_dataset.py
------------------
Merge all 4 labeled YOLO folders and split into train / val / test sets.
Uses stratified split to ensure rare classes appear in all splits.

Output structure:
  dataset/
    images/  train/  val/  test/
    labels/  train/  val/  test/
    cad_dataset.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE_FOLDERS: list[str] = [
    "rendered_竣工図（新綱島スクエア　建築意匠図）_dpi300",
    "rendered_竣工図（新綱島スクエア　電気設備図）_dpi300",
    "rendered_竣工図（新綱島スクエア　構造図）_dpi300",
    "rendered_竣工図（新綱島スクエア　機械設備図）_dpi300",
]

CLASSES: list[str] = ["text", "table", "title_block", "diagram"]

# Drop class IDs from source (original indices in classes.txt)
DROP_CLASS_IDS: set[int] = {4}          # 4=deleted_part
CLASS_ID_REMAP: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3}

SPLIT_RATIO = (0.80, 0.10, 0.10)

ImageEntry = tuple[Path, Path | None]


def collect_all_images(src_dir: Path) -> list[ImageEntry]:
    entries: list[ImageEntry] = []
    for img_path in sorted(src_dir.glob("*.png")):
        lbl_path = src_dir / f"{img_path.stem}.txt"
        entries.append((img_path, lbl_path if lbl_path.exists() else None))
    return entries


def split_entries(
    entries: list[ImageEntry],
    ratio: tuple[float, float, float],
    seed: int = 42,
) -> tuple[list[ImageEntry], list[ImageEntry], list[ImageEntry]]:
    """Stratified split: rare classes spread proportionally across all splits."""
    rng = random.Random(seed)

    rare_threshold = max(10, int(len(entries) * 0.05))
    class_to_entries: dict[int, list[ImageEntry]] = {}
    for img_path, lbl_path in entries:
        if lbl_path is None or not lbl_path.exists():
            continue
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                if cls_id not in DROP_CLASS_IDS:
                    cls_id = CLASS_ID_REMAP.get(cls_id, cls_id)
                    class_to_entries.setdefault(cls_id, [])
                    if (img_path, lbl_path) not in class_to_entries[cls_id]:
                        class_to_entries[cls_id].append((img_path, lbl_path))

    rare_entries: set[tuple] = set()
    for cls_id, cls_imgs in class_to_entries.items():
        if len(cls_imgs) <= rare_threshold:
            for e in cls_imgs:
                rare_entries.add(e)

    rare_list   = [e for e in entries if e in rare_entries]
    normal_list = [e for e in entries if e not in rare_entries]
    rng.shuffle(rare_list)
    rng.shuffle(normal_list)

    def _split(lst: list[ImageEntry], force: bool = False) -> tuple[list, list, list]:
        n = len(lst)
        if n == 0:
            return [], [], []
        if force and n >= 3:
            n_val  = max(1, int(n * ratio[1]))
            n_test = max(1, int(n * ratio[2]))
            n_train = max(1, n - n_val - n_test)
        elif n >= 3:
            n_train = max(1, int(n * ratio[0]))
            n_val   = max(1, int(n * ratio[1]))
        elif n == 2:
            n_train, n_val = 1, 1
        else:
            n_train, n_val = n, 0
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    r_tr, r_va, r_te = _split(rare_list, force=True)
    n_tr, n_va, n_te = _split(normal_list)

    train = r_tr + n_tr; rng.shuffle(train)
    val   = r_va + n_va; rng.shuffle(val)
    test  = r_te + n_te; rng.shuffle(test)
    return train, val, test


def _parse_labels(lbl_path: Path | None) -> list[str]:
    if lbl_path is None or not lbl_path.exists():
        return []
    out = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        if cls_id in DROP_CLASS_IDS:
            continue
        new_id = CLASS_ID_REMAP.get(cls_id, cls_id)
        out.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return out


def copy_split(
    entries: list[ImageEntry],
    split: str,
    dst_root: Path,
    src_tag: str,
) -> tuple[int, int]:
    img_dst = dst_root / "images" / split
    lbl_dst = dst_root / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    labeled = bg = 0
    for img_path, lbl_path in entries:
        stem = f"{src_tag}__{img_path.stem}"
        shutil.copy2(img_path, img_dst / f"{stem}.png")
        lines = _parse_labels(lbl_path)
        (lbl_dst / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if lines:
            labeled += 1
        else:
            bg += 1
    return labeled, bg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    dst = ROOT / "dataset"
    if dst.exists():
        if args.overwrite:
            print(f"⚠  Removing existing directory: {dst}")
            shutil.rmtree(dst)
        else:
            sys.exit(f"ERROR: {dst} already exists. Use --overwrite to replace.")

    totals: dict[str, tuple[int, int, int]] = {}

    for folder_name in SOURCE_FOLDERS:
        src_dir = ROOT / folder_name
        if not src_dir.exists():
            print(f"⚠  Skipping missing folder: {folder_name}")
            continue

        entries = collect_all_images(src_dir)
        train_e, val_e, test_e = split_entries(entries, SPLIT_RATIO, seed=args.seed)
        src_tag = f"folder{SOURCE_FOLDERS.index(folder_name)}"

        tr_l, tr_b = copy_split(train_e, "train", dst, src_tag)
        va_l, va_b = copy_split(val_e,   "val",   dst, src_tag)
        te_l, te_b = copy_split(test_e,  "test",  dst, src_tag)

        total = len(entries)
        labeled = tr_l + va_l + te_l
        bg      = tr_b + va_b + te_b
        totals[folder_name] = (len(train_e), len(val_e), len(test_e))
        print(f"✓ {folder_name:<55} total {total:>4}  (labeled {labeled} | bg {bg:>3})  →  "
              f"train {len(train_e):>3}  val {len(val_e):>3}  test {len(test_e):>3}")

    # Write YAML
    yaml_content = f"""# CAD Drawing Dataset  –  YOLO format
# Generated by tools/prepare_dataset.py

path: {dst}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASSES)}
names:
"""
    for i, name in enumerate(CLASSES):
        yaml_content += f"  {i}: {name}\n"
    (dst / "cad_dataset.yaml").write_text(yaml_content, encoding="utf-8")

    tr = sum(v[0] for v in totals.values())
    va = sum(v[1] for v in totals.values())
    te = sum(v[2] for v in totals.values())
    print(f"\n✓ Dataset written to:  {dst}")
    print(f"✓ Config YAML:         {dst / 'cad_dataset.yaml'}")
    print(f"\n  Split summary:\n    train : {tr:>5}\n    val   : {va:>5}\n    test  : {te:>5}")


if __name__ == "__main__":
    main()
