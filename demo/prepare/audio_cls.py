"""Write synthetic tones as an audio class folder: <root>/<split>/<label>/<id>.wav.

The pitch is the class (``low``, ``high``). With ``--multi-label`` each clip is also filed under a duration label
(``short``, ``long``), so one stem sits in several folders as the multi-label reader expects.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(sys.argv[1])
multi_label = "--multi-label" in sys.argv[2:]
rate = 16000
pitch = {"low": 220.0, "high": 880.0}
rng = np.random.default_rng(0)
for split, count in {"train": 120, "val": 30, "test": 30}.items():
    for idx in range(count):
        name = ("low", "high")[idx % 2]
        length = ("short", "long")[(idx // 2) % 2]
        t = np.arange(int(rate * (0.5 if length == "short" else 1.0))) / rate
        wave = 0.5 * np.sin(2 * np.pi * pitch[name] * t) + 0.01 * rng.standard_normal(t.shape)
        for label in (name, length) if multi_label else (name,):
            out = root / split / label
            out.mkdir(parents=True, exist_ok=True)
            sf.write(out / f"{split}_{idx:03d}.wav", wave.astype(np.float32), rate)
print(f"tones written to {root}")
