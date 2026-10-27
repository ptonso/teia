"""Write scikit-learn's bundled 8x8 digits as a segmentation set: <root>/<split>/{images,masks}/<id>.png.

The mask labels each pixel ``stroke`` (1) where the digit has ink and ``background`` (0) elsewhere.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

root = Path(sys.argv[1])
data = load_digits()
order = np.random.default_rng(0).permutation(len(data.target))
for split, window in {"train": slice(0, 1200), "val": slice(1200, 1500), "test": slice(1500, None)}.items():
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "masks").mkdir(parents=True, exist_ok=True)
    for idx in order[window]:
        pixels = (data.images[idx] * 255 / 16).astype(np.uint8)
        Image.fromarray(pixels).convert("RGB").save(root / split / "images" / f"{idx:05d}.png")
        Image.fromarray((pixels > 0).astype(np.uint8)).save(root / split / "masks" / f"{idx:05d}.png")
(root / "data.yaml").write_text(
    "names: [background, stroke]\n"
    + "".join(f"{s}:\n  images: {s}/images\n  masks: {s}/masks\n" for s in ("train", "val", "test"))
)
print(f"digits written to {root}")
