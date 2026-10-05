"""Synthetic sales data with PLANTED anomalies + ground-truth labels."""
import json, sys
import numpy as np, pandas as pd


def make(path="evals/sales.csv", labels="evals/labels.json", seed=7, n_anom=10):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=300)
    rows = [(d.date().isoformat(), reg, int(rng.normal(100, 8)),
             round(float(rng.normal(5000, 350)), 2))
            for d in days for reg in ("north", "south")]
    df = pd.DataFrame(rows, columns=["date", "region", "units", "revenue"])
    idx = sorted(rng.choice(len(df), n_anom, replace=False).tolist())
    for k, i in enumerate(idx):
        df.loc[i, "revenue"] = round(float(df.loc[i, "revenue"] * (4.5 if k % 2 == 0 else 0.1)), 2)
    df.to_csv(path, index=False)
    json.dump({"rows": idx, "column": "revenue"}, open(labels, "w"))
    return idx


if __name__ == "__main__":
    print("planted anomaly rows:", make(*sys.argv[1:3]))
