"""Success rate per cell and the differences between them, with 95% CIs.

    python src/analyze.py gpt-oss-20b paper_style
    python src/analyze.py gpt-oss-20b authentic --vs paper_style

CIs come from a two-way cluster bootstrap: each replicate resamples the 50 conditions and the 20
request phrasings, which every cell and dataset shares. Unless cluster effects are large, this makes the
intervals wider than they need to be. A cell at 0% or 100% does not vary across replicates, so a row that
uses one is marked * and its interval leaves out that cell's uncertainty. --vs compares the style effect in
each location with another run's, using the same replicates for both. paper_style_h200 is paper_style run
again on the authentic runs' GPU. Every number prints at a fixed width, so a file's size says nothing.
"""
import argparse
import json

import numpy as np

from experiment import CELLS, DATASETS, MODELS, ROOT

RUNS = DATASETS + ["paper_style_h200"]

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


def bootstrap(grids, replicates=2000):
    """Rates per replicate, grid and cell. Every grid is resampled with the same draws."""
    rng = np.random.default_rng(0)
    rates = []
    for _ in range(replicates):
        conditions = rng.integers(0, 50, 50)
        phrasings = rng.integers(0, 20, 20)
        rates.append([grid[:, conditions][:, :, phrasings].mean(axis=(1, 2)) for grid in grids])
    return np.array(rates)


def show(name, value, boot, edge):
    """One row; edge marks a row that uses a cell at 0% or 100%."""
    low, high = np.percentile(100 * boot, [2.5, 97.5])
    print(f"{name:38s}{100 * value:+7.1f}  [{low:+6.1f}, {high:+6.1f}]{' *' if edge else '  '}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("dataset", choices=RUNS)
    parser.add_argument("--vs", choices=RUNS)
    args = parser.parse_args()

    names = [args.dataset] + ([args.vs] if args.vs else [])
    runs = [json.loads((ROOT / "results" / args.model / f"{name}.json").read_text()) for name in names]
    grids = [load_grid(rows) for rows in runs]
    rates, boot = [grid.mean(axis=(1, 2)) for grid in grids], bootstrap(grids)
    edges = [(r == 0) | (r == 1) for r in rates]  # per run and cell

    for cell, weights in zip(CELLS, np.eye(len(CELLS))):
        show(cell, rates[0] @ weights, boot[:, 0] @ weights, (edges[0] & (weights != 0)).any())
    print()
    for name, weights in CONTRASTS.items():
        w = np.array(weights)
        show(name, rates[0] @ w, boot[:, 0] @ w, (edges[0] & (w != 0)).any())
    print()
    for name, rows in zip(names, runs):
        counts = " ".join(f"{cell}={sum(r['cell'] == cell and r['answer'] is None for r in rows):4d}" for cell in CELLS)
        print(f"no final answer, {name}: {counts}")

    if args.vs:
        print(f"\n{args.dataset} minus {args.vs}:")
        for location in ["user", "real"]:
            styled, plain = np.eye(len(CELLS))[[CELLS.index(f"{location}/styled"), CELLS.index(f"{location}/plain")]]
            style = styled - plain
            edge = lambda w: any((e & (w != 0)).any() for e in edges)
            show(f"style effect in {location}", (rates[0] - rates[1]) @ style, (boot[:, 0] - boot[:, 1]) @ style, edge(style))
            # Datasets are compared through the style effect; the raw plain-rate difference only shows where it comes from.
            show(f"raw {location}/plain difference (diagnostic)", (rates[0] - rates[1]) @ plain, (boot[:, 0] - boot[:, 1]) @ plain, edge(plain))
    print("\n* a cell in this row is at 0% or 100%, so its interval leaves out that cell's uncertainty")


if __name__ == "__main__":
    main()
