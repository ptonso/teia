"""Write scikit-learn's bundled 8x8 digits as an image folder: <root>/<split>/<class>/<id>.png."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

root = Path(sys.argv[1])
data = load_digits()
order = np.random.default_rng(0).permutation(len(data.target))
for split, window in {"train": slice(0, 1200), "val": slice(1200, 1500), "test": slice(1500, None)}.items():
    for idx in order[window]:
        out = root / split / str(data.target[idx])
        out.mkdir(parents=True, exist_ok=True)
        pixels = (data.images[idx] * 255 / 16).astype(np.uint8)
        Image.fromarray(pixels).convert("RGB").save(out / f"{idx:05d}.png")
print(f"digits written to {root}")
