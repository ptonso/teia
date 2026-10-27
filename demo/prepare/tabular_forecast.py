"""Write a monthly forecasting table (trend, yearly season and noise for three routes) as <root>/{train,val,test}.csv plus data.yaml.

The target column is not an input of a window datamodule, so each row also carries last month's value as the covariate
``prev_passengers``. Each route is split by time: months 0-95 train, 96-119 validation, 120-143 test.
"""

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
months = np.arange(144)
splits = {"train": months < 96, "val": (months >= 96) & (months < 120), "test": months >= 120}
rows = {name: [] for name in splits}
for route, level in {"A-B": 100.0, "A-C": 150.0, "B-C": 80.0}.items():
    passengers = (level + 0.5 * months + 20 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 2, len(months))) / 100
    previous = np.concatenate([passengers[:1], passengers[:-1]])
    for name, mask in splits.items():
        for m in months[mask]:
            rows[name].append([route, f"{2010 + m // 12}-{m % 12 + 1:02d}-01", round(float(previous[m]), 4), round(float(passengers[m]), 4)])
for name, data in rows.items():
    with open(root / f"{name}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["route", "month", "prev_passengers", "passengers"])
        writer.writerows(data)
(root / "data.yaml").write_text(
    "task: forecast\ntargets: [passengers]\nsplits:\n  train: train.csv\n  val: val.csv\n  test: test.csv\n"
    "forecast:\n  entity_col: route\n  time_col: month\n  lookback: 12\n  horizon: 3\n"
)
print(f"table written to {root}")
