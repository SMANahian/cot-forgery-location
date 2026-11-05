"""Success rate per cell and the differences between them, with conservative 95% CIs.

    python src/analyze.py paper_style

CIs come from a two-way cluster bootstrap: each replicate resamples the 50 conditions and the 20
request phrasings, which every cell shares. Unless cluster effects are large, this makes the
intervals wider than they need to be.
"""
import argparse
import json

import numpy as np

from experiment import CELLS, DATASETS, ROOT

# Weights over CELLS. Cases: 1 = user/styled, 2 = user/plain, 3 = real/styled, 4 = real/plain.
CONTRASTS = {
    "style in user message (1 - 2)": [0, 1, -1, 0, 0],
    "style in real reasoning (3 - 4)": [0, 0, 0, 1, -1],
    "location, styled (3 - 1)": [0, -1, 0, 1, 0],
    "location, plain (4 - 2)": [0, 0, -1, 0, 1],
    "style x location ((3 - 4) - (1 - 2))": [0, -1, 1, 1, -1],
}


def load_grid(rows):
    """Successes as an array of cells x conditions x phrasings."""
    grid = np.zeros((len(CELLS), 50, 20))
    assert len(rows) == grid.size, "results are incomplete"
    for r in rows:
        grid[CELLS.index(r["cell"]), r["slot"], r["phrasing"]] = r["success"]
    return grid


def bootstrap(grid, replicates=2000):
    rng = np.random.default_rng(0)
    rates = []
    for _ in range(replicates):
        conditions = rng.integers(0, 50, 50)
        phrasings = rng.integers(0, 20, 20)
        rates.append(grid[:, conditions][:, :, phrasings].mean(axis=(1, 2)))
    return np.array(rates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    args = parser.parse_args()

    rows = json.loads((ROOT / "results" / f"{args.dataset}.json").read_text())
    grid = load_grid(rows)
    rates, boot = grid.mean(axis=(1, 2)), bootstrap(grid)

    def show(name, weights):
        low, high = np.percentile(100 * boot @ weights, [2.5, 97.5])
        print(f"{name:38s}{100 * rates @ weights:+7.1f}  [{low:+.1f}, {high:+.1f}]")

    for cell, weights in zip(CELLS, np.eye(len(CELLS))):
        show(cell, weights)
    print()
    for name, weights in CONTRASTS.items():
        show(name, np.array(weights))
    no_answer = {cell: sum(r["cell"] == cell and r["answer"] is None for r in rows) for cell in CELLS}
    print("\nno final answer:", no_answer)


if __name__ == "__main__":
    main()
