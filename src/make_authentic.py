"""Collect gpt-oss's real reasoning for the `authentic` dataset. Needs a GPU.

    python src/make_authentic.py    # -> data/forgeries/authentic.json

The model gets a version of the task where sharing really is allowed, so it writes the same
argument in its own words. Each condition keeps the first sample that makes the argument, gives
the code in its answer, and doesn't repeat the code in its reasoning. The `plain` field stays
empty until each passage has been destyled and read.
"""
import json

from transformers import AutoTokenizer

from experiment import MODEL, ROOT, build_items, is_success, load_task, parse_answer, render
from run import generate

SAMPLES = 64


def parse_reasoning(raw):
    return raw.partition("<|channel|>analysis<|message|>")[2].split("<|")[0].strip()


def usable(reasoning, answer, item):
    low = reasoning.lower()
    makes_argument = "polic" in low and all(k in low for k in item["keywords"])
    hides_code = item["word"] not in low and item["digits"] not in low
    return is_success(answer, item) and makes_argument and hides_code


def permissive_rule(task, item):
    code = f"{item['word']}-{item['digits']}"
    return task["permissive_rule"].format(code=code, policy=item["policy"], meets=item["meets"])


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    task = load_task()
    items = [item for item in build_items(task) if item["phrasing"] == 0]  # one per condition
    prompts = [render(tok, permissive_rule(task, it), it["request"]) for it in items for _ in range(SAMPLES)]
    raws = generate(tok, prompts)

    rows = []
    for i, item in enumerate(items):
        samples = [(parse_reasoning(raw), parse_answer(raw)) for raw in raws[i * SAMPLES:(i + 1) * SAMPLES]]
        passing = [reasoning for reasoning, answer in samples if usable(reasoning, answer, item)]
        if passing:
            rows.append({
                "slot": item["slot"],
                "policy": item["policy"],
                "styled": passing[0],
                "plain": None,
            })
        print(f"condition {item['slot']:2d}: {len(passing)}/{SAMPLES} samples usable")

    assert len(rows) == len(items), "some conditions have no passage: raise SAMPLES and rerun"
    with open(ROOT / "data/forgeries/authentic.json", "w") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
