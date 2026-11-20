"""Compare the scorer with the published 2024 TAP2 workbook."""
import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.stats import pearsonr, spearmanr

from tap_score import METRICS, calculate, prepare

ALIASES = {"arcitumomab": "arcitumumab"}

# Green/amber and amber/red limits from the 2024 paper, Table 1.
CUTOFFS_TAP2 = [(37, 42, 55, 63), (95.58, 110.11, 168.06, 201.59),
                (None, None, 1.32, 4.22), (None, None, 2.00, 4.42),
                (-30.60, -6.00, None, None)]


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower().split(" (", 1)[0])


def score(path):
    return calculate(prepare(path))


def flag(score, bounds):
    score = round(float(score), 2)
    red_low, amber_low, amber_high, red_high = bounds
    if (red_low is not None and score < red_low) or (red_high is not None and score > red_high):
        return "red"
    if ((amber_low is not None and score <= amber_low)
            or (amber_high is not None and score >= amber_high)):
        return "amber"
    return "green"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("pdb_folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("tap2_2024_benchmark.csv"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    workbook = load_workbook(args.xlsx, data_only=True, read_only=True)
    rows = list(workbook.active.values)[1:]
    workbook.close()
    models = {normalize(p.stem.removesuffix("_model")): p for p in args.pdb_folder.glob("*.pdb")}
    names = [row[0] for row in rows]
    model_keys = [normalize(name) for name in names]
    paths = [models[key if key in models else ALIASES.get(key, key)] for key in model_keys]
    reference = np.array([row[1:] for row in rows], dtype=float)
    if len(rows) != 664:
        parser.error(f"expected 664 reference rows, got {len(rows)}")
    with ProcessPoolExecutor(args.workers) as pool:
        predicted = np.array(list(pool.map(score, paths)))

    development = np.random.default_rng(2024).choice(len(rows), size=442, replace=False)
    holdout = ~np.isin(np.arange(len(rows)), development)
    flags = []
    with args.out.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["therapeutic", "model", "split", *(
            f"{metric}_{suffix}" for metric in METRICS for suffix in
            ("reference", "reconstructed", "reference_flag", "reconstructed_flag")
        )])
        for name, path, held, observed, actual in zip(names, paths, holdout, reference, predicted):
            row = [name, path.name, "holdout" if held else "development"]
            agreement = []
            for expected, calculated, bounds in zip(observed, actual, CUTOFFS_TAP2):
                original_color, computed_color = flag(expected, bounds), flag(calculated, bounds)
                row.extend([round(expected, 4), round(calculated, 4), original_color, computed_color])
                agreement.append(original_color == computed_color)
            writer.writerow(row)
            flags.append(agreement)

    flags = np.array(flags)
    for title, subset in [("Holdout", holdout), ("All", np.ones(len(rows), dtype=bool))]:
        print(f"\n{title} ({subset.sum()} models)")
        print("metric,Pearson,Spearman,MAE,bias,match_to_4dp,matching_flags")
        for j, metric in enumerate(METRICS):
            x, y = predicted[subset, j], reference[subset, j]
            match = np.isclose(np.round(x, 4), np.round(y, 4), rtol=0, atol=1e-8).sum()
            diff = x - y
            print(f"{metric},{pearsonr(x,y).statistic:.4f},{spearmanr(x,y).statistic:.4f},"
                  f"{abs(diff).mean():.4f},{diff.mean():.4f},{match},{flags[subset,j].sum()}")


if __name__ == "__main__":
    main()
