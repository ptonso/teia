"""Write colored shapes on a black canvas as a detection dataset in one of four formats: python shapes.py <root> --format det|pose|seg|obb.

Each 64 px image holds one or two shapes of two classes (``red``, ``green``) whose centers fall in different cells of a 4 x 4 grid.
``det``, ``pose`` and ``seg`` write the YOLO format (<root>/<split>/{images,labels}/<id> and data.yaml): rectangles for ``det`` and
``pose`` (the four corners are the keypoints) and ellipses with their outline polygon for ``seg``. ``obb`` writes rotated rectangles as
LabelMe polygons next to each image (<root>/<split>/<id>.{png,json}).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE, GRID = 64, 4
COLORS = {"red": (255, 40, 40), "green": (40, 255, 40)}
NAMES = list(COLORS)


def scene(rng: np.random.Generator, rotate: bool) -> list[dict]:
    shapes, taken = [], set()
    for _ in range(int(rng.integers(1, 3))):
        for _ in range(100):
            w, h = rng.uniform(14, 26, 2)
            angle = rng.uniform(0, math.pi / 2) if rotate else 0.0
            half = (abs(w * math.cos(angle)) + abs(h * math.sin(angle))) / 2, (abs(w * math.sin(angle)) + abs(h * math.cos(angle))) / 2
            cx, cy = (rng.uniform(half[0] + 1, SIZE - half[0] - 1), rng.uniform(half[1] + 1, SIZE - half[1] - 1))
            cell = (int(cy * GRID / SIZE), int(cx * GRID / SIZE))
            if cell not in taken:
                taken.add(cell)
                shapes.append({"cls": int(rng.integers(0, 2)), "cx": cx, "cy": cy, "w": w, "h": h, "angle": angle})
                break
    return shapes


def corners(s: dict) -> list[tuple[float, float]]:
    cos, sin = math.cos(s["angle"]), math.sin(s["angle"])
    return [(s["cx"] + x * cos - y * sin, s["cy"] + x * sin + y * cos) for x, y in [(-s["w"] / 2, -s["h"] / 2), (s["w"] / 2, -s["h"] / 2), (s["w"] / 2, s["h"] / 2), (-s["w"] / 2, s["h"] / 2)]]


def ellipse(s: dict, points: int = 16) -> list[tuple[float, float]]:
    return [(s["cx"] + s["w"] / 2 * math.cos(t), s["cy"] + s["h"] / 2 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, points, endpoint=False)]


def draw(shapes: list[dict], fmt: str) -> Image.Image:
    image = Image.new("RGB", (SIZE, SIZE))
    canvas = ImageDraw.Draw(image)
    for s in shapes:
        canvas.polygon(ellipse(s) if fmt == "seg" else corners(s), fill=COLORS[NAMES[s["cls"]]])
    return image


def label_line(s: dict, fmt: str) -> str:
    box = f"{s['cx'] / SIZE:.6f} {s['cy'] / SIZE:.6f} {s['w'] / SIZE:.6f} {s['h'] / SIZE:.6f}"
    if fmt == "pose":
        return f"{s['cls']} {box} " + " ".join(f"{x / SIZE:.6f} {y / SIZE:.6f} 2" for x, y in corners(s))
    if fmt == "seg":
        return f"{s['cls']} " + " ".join(f"{x / SIZE:.6f} {y / SIZE:.6f}" for x, y in ellipse(s))
    return f"{s['cls']} {box}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--format", choices=["det", "pose", "seg", "obb"], required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for split, count in {"train": 600, "val": 100, "test": 100}.items():
        for idx in range(count):
            shapes = scene(rng, rotate=args.format == "obb")
            image = draw(shapes, args.format)
            if args.format == "obb":
                folder = args.root / split
                folder.mkdir(parents=True, exist_ok=True)
                image.save(folder / f"{split}_{idx:03d}.png")
                payload = {"version": "5.8.3", "flags": {}, "imagePath": f"{split}_{idx:03d}.png", "imageHeight": SIZE, "imageWidth": SIZE, "imageData": None,
                           "shapes": [{"label": NAMES[s["cls"]], "points": [[round(x, 2), round(y, 2)] for x, y in corners(s)], "shape_type": "polygon", "flags": {}} for s in shapes]}
                (folder / f"{split}_{idx:03d}.json").write_text(json.dumps(payload))
            else:
                (args.root / split / "images").mkdir(parents=True, exist_ok=True)
                (args.root / split / "labels").mkdir(parents=True, exist_ok=True)
                image.save(args.root / split / "images" / f"{split}_{idx:03d}.png")
                (args.root / split / "labels" / f"{split}_{idx:03d}.txt").write_text("\n".join(label_line(s, args.format) for s in shapes) + "\n")
    if args.format != "obb":
        head = f"task: {dict(pose='pose', seg='segment').get(args.format, 'detect')}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 2\nnames: {NAMES}\n"
        (args.root / "data.yaml").write_text(head + ("kpt_shape: [4, 3]\n" if args.format == "pose" else ""))
    print(f"{args.format} shapes written to {args.root}")


if __name__ == "__main__":
    main()
