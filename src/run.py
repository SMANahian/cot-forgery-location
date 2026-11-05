"""Generate every cell for one dataset and save the results.

    python src/run.py paper_style --dry     # print one prompt per cell, no GPU
    python src/run.py paper_style --smoke   # 91 items per cell -> results/paper_style_smoke.json
    python src/run.py paper_style           # 1,000 items per cell -> results/paper_style.json
"""
import argparse
import json

from transformers import AutoTokenizer

from experiment import (
    CELLS,
    DATASETS,
    MODEL,
    ROOT,
    build_items,
    build_prompt,
    is_success,
    load_forgeries,
    load_task,
    parse_answer,
)


def generate(tok, prompts):
    from vllm import LLM, SamplingParams  # imported here so --dry works without a GPU

    llm = LLM(model=MODEL, max_model_len=12288, gpu_memory_utilization=0.90, seed=0)
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=8192, skip_special_tokens=False)
    requests = [{"prompt_token_ids": tok(p, add_special_tokens=False)["input_ids"]} for p in prompts]
    outputs = llm.generate(requests, params)
    return [tok.decode(list(out.outputs[0].token_ids), skip_special_tokens=False) for out in outputs]


def save(rows, name):
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"{name}.json", "w") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)


def print_summary(rows):
    for cell in CELLS:
        cell_rows = [r for r in rows if r["cell"] == cell]
        successes = sum(r["success"] for r in cell_rows)
        no_answer = sum(r["answer"] is None for r in cell_rows)
        print(f"{cell:12s} success {successes}/{len(cell_rows)}, no final answer {no_answer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    task = load_task()
    forgeries = load_forgeries(args.dataset)
    items = build_items(task)
    if args.smoke:
        items = items[::11]  # spread over all conditions and phrasings
    jobs = [(cell, item) for cell in CELLS for item in items]
    prompts = [build_prompt(tok, item, cell, task, forgeries) for cell, item in jobs]

    if args.dry:
        for i, cell in enumerate(CELLS):
            print(f"\n===== {cell} =====\n{prompts[i * len(items)]}")
        return

    rows = []
    for (cell, item), prompt, raw in zip(jobs, prompts, generate(tok, prompts)):
        answer = parse_answer(raw)
        rows.append({
            **item,
            "cell": cell,
            "prompt": prompt,
            "raw": raw,
            "answer": answer,
            "success": is_success(answer, item),
        })
    save(rows, args.dataset + ("_smoke" if args.smoke else ""))
    print_summary(rows)


if __name__ == "__main__":
    main()
