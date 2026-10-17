"""Synthesize tiny boards for graph node classification, in both edge formats.

Every board is a green canvas with component boxes on a coarse grid, wires between neighbouring boxes, and one
class per box: ``present`` (a coloured body), ``missing`` (an empty footprint: two pads and no body) or
``background`` (a box over bare canvas).

``--graph-signal`` removes the appearance cue: a missing box is drawn exactly like a present one, and the only
evidence is an orange leaf part wired to it alone. Then a vision-only model cannot beat the class prior, and a
model that reads neighbours can be exact.

Writes, under ``--target-dir``: ``images/`` and ``coco/{train,val,test}.json`` + ``graph.json`` (COCO
boxes, edges keyed by image), and ``labelme/<split>/`` (LabelMe JSON with an ``edges`` key, image beside it).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE, GRID = 192, 4
CELL = SIZE // GRID
COUNTS = {"train": 24, "val": 8, "test": 8}
BODIES = [(70, 90, 200), (200, 170, 60), (150, 80, 160)]
ORANGE = (255, 130, 0)
CANVAS, WIRE, PAD = (30, 110, 60), (20, 20, 20), (210, 210, 210)
CATEGORIES = ["present", "missing", "background"]


def make_board(rng: np.random.Generator, graph_signal: bool):
    n_main = int(rng.integers(6, 9))
    kinds = list(rng.choice(CATEGORIES, size=n_main, p=[0.65, 0.25, 0.10]))
    twins = [i for i, kind in enumerate(kinds) if kind == "missing"] if graph_signal else []
    if graph_signal:
        kinds = ["present" if kind == "background" else kind for kind in kinds]
    total = n_main + len(twins)
    cells = rng.choice(GRID * GRID, size=total, replace=False)
    boxes = []
    for cell in cells:
        w, h = (int(v) for v in rng.integers(18, 28, size=2))
        x = int(cell % GRID) * CELL + int(rng.integers(0, CELL - w))
        y = int(cell // GRID) * CELL + int(rng.integers(0, CELL - h))
        boxes.append([x, y, w, h])
    labels = kinds + ["present"] * len(twins)
    centres = np.array([[x + w / 2, y + h / 2] for x, y, w, h in boxes])
    edges = {(a, b) for a in range(n_main) for b in np.argsort(np.linalg.norm(centres[:n_main] - centres[a], axis=1))[1:3] if a != b and b < n_main}
    edges = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    edges |= {(m, n_main + t) for t, m in enumerate(twins)}
    image = Image.new("RGB", (SIZE, SIZE), CANVAS)
    draw = ImageDraw.Draw(image)
    for a, b in edges:
        draw.line([tuple(centres[a]), tuple(centres[b])], fill=WIRE, width=2)
    for i, ((x, y, w, h), label) in enumerate(zip(boxes, labels)):
        appearance = "present" if graph_signal and label == "missing" else label
        if appearance == "present":
            color = ORANGE if i >= n_main else BODIES[int(rng.integers(len(BODIES)))]
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=WIRE)
        elif appearance == "missing":
            draw.rectangle([x, y + h // 3, x + 4, y + 2 * h // 3], fill=PAD)
            draw.rectangle([x + w - 4, y + h // 3, x + w, y + 2 * h // 3], fill=PAD)
    return image, boxes, labels, sorted(edges)


def write_coco(root: Path, split: str, boards, graph: dict) -> None:
    images, annotations = [], []
    for i, (name, boxes, labels, edges) in enumerate(boards):
        images.append({"id": i, "file_name": name, "width": SIZE, "height": SIZE})
        ids = []
        for box, label in zip(boxes, labels):
            annotation = {"id": len(annotations), "image_id": i, "category_id": CATEGORIES.index(label), "bbox": box}
            ids.append(annotation["id"])
            annotations.append(annotation)
        graph[name] = {"edges": [{"src": ids[a], "dst": ids[b]} for a, b in edges]}
    categories = [{"id": i, "name": name, "supercategory": name} for i, name in enumerate(CATEGORIES)]
    (root / "coco" / f"{split}.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))


def write_labelme(root: Path, split: str, name: str, boxes, labels, edges) -> None:
    shapes = [
        {"label": label, "shape_type": "rectangle", "points": [[x, y], [x + w, y + h]]}
        for (x, y, w, h), label in zip(boxes, labels)
    ]
    payload = {"imagePath": name, "shapes": shapes, "edges": [{"src": a, "dst": b} for a, b in edges]}
    (root / "labelme" / split / f"{Path(name).stem}.json").write_text(json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny synthetic boards with graphs.")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--graph-signal", action="store_true", help="hide the appearance cue of missing boxes")
    args = parser.parse_args()
    root = Path(args.target_dir).resolve()
    shutil.rmtree(root, ignore_errors=True)
    for sub in ("images", "coco", *(f"labelme/{s}" for s in COUNTS)):
        (root / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    graph: dict = {}
    for split, count in COUNTS.items():
        boards = []
        for i in range(count):
            image, boxes, labels, edges = make_board(rng, args.graph_signal)
            name = f"{split}_{i:02d}.png"
            image.save(root / "images" / name)
            image.save(root / "labelme" / split / name)
            write_labelme(root, split, name, boxes, labels, edges)
            boards.append((name, boxes, labels, edges))
        write_coco(root, split, boards, graph)
    (root / "graph.json").write_text(json.dumps(graph))


if __name__ == "__main__":
    main()
