"""Write a survival table (exponential event times from a linear log-hazard, right-censored, in whole weeks) as <root>/{train,val,test}.csv plus data.yaml.

Whole-week times make tied times common, as in real follow-up data.
"""

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
beta = np.array([0.8, -0.5, 0.3, 0.0])
for split, n in {"train": 600, "val": 150, "test": 150}.items():
    features = rng.normal(size=(n, 4))
    event_time = rng.exponential(1.0 / (0.05 * np.exp(features @ beta)))
    censor_time = rng.uniform(5, 60, n)
    time = np.maximum(np.ceil(np.minimum(event_time, censor_time)), 1).astype(int)
    event = (event_time <= censor_time).astype(int)
    with open(root / f"{split}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f0", "f1", "f2", "f3", "week", "arrest"])
        writer.writerows([[round(float(v), 4) for v in row] + [int(t), int(e)] for row, t, e in zip(features, time, event)])
(root / "data.yaml").write_text("task: survival\ntime_col: week\nevent_col: arrest\nsplits:\n  train: train.csv\n  val: val.csv\n  test: test.csv\n")
print(f"table written to {root}")
