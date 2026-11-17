"""Runs to read, drawn with a fixed seed before anyone has seen the results.

    python src/sample.py gemma-4-31b authentic > results/sealed/gemma-4-31b_authentic_sample.txt

Up to 10 successes per forged cell (does the answer give the code, or refuse while printing it?) and up to 10
failures per real cell (does the model object that the user never said they meet the condition?).
"""
import argparse
import json
import random

from analyze import RUNS
from experiment import CELLS, MODELS, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("dataset", choices=RUNS)
    args = parser.parse_args()

    rows = json.loads((ROOT / "results" / args.model / f"{args.dataset}.json").read_text())
    rng = random.Random(0)
    for cell in CELLS[1:]:
        for success in [True] + ([False] if cell.startswith("real") else []):
            pool = [r for r in rows if r["cell"] == cell and r["success"] == success]
            for r in rng.sample(pool, min(10, len(pool))):
                print(f"===== {cell}, {'success' if success else 'failure'}, condition {r['slot']}, phrasing {r['phrasing']}")
                print(f"--- request\n{r['request']}\n--- answer\n{r['answer']}\n--- end of the completion\n{r['raw'][-600:]}\n")


if __name__ == "__main__":
    main()
