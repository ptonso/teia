from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


SPLITS: tuple[str, ...] = ("train", "val", "test")


def parse_target_dir(description: str) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target-dir", required=True, help="Directory where the dataset root will be written.")
    args = parser.parse_args()
    return Path(args.target_dir).resolve()


def reset_target_dir(target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_image(path: Path, color: tuple[int, int, int], *, size: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((size * 0.25, size * 0.25, size * 0.75, size * 0.75), outline=(255, 255, 255), width=3)
    draw.line((size * 0.2, size * 0.8, size * 0.8, size * 0.2), fill=(32, 32, 32), width=2)
    image.save(path)


def write_mask(path: Path, *, size: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((size * 0.28, size * 0.28, size * 0.76, size * 0.76), fill=1)
    mask.save(path)


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def split_counts() -> dict[str, int]:
    return {"train": 4, "val": 2, "test": 2}


def colors() -> Iterable[tuple[int, int, int]]:
    palette = (
        (166, 67, 82),
        (55, 126, 184),
        (77, 175, 74),
        (152, 78, 163),
        (255, 127, 0),
        (64, 141, 148),
        (190, 122, 52),
        (91, 91, 91),
    )
    while True:
        yield from palette


def write_yolo_data_yaml(target_dir: Path, *, task: str, names: list[str], kpt_shape: tuple[int, int] | None = None) -> None:
    lines = [
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(names)}",
        f"names: [{', '.join(repr(name) for name in names)}]",
        f"task: {task}",
    ]
    if kpt_shape is not None:
        lines.append(f"kpt_shape: [{kpt_shape[0]}, {kpt_shape[1]}]")
    write_text(target_dir / "data.yaml", "\n".join(lines))
