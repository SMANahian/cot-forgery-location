"""Generate every cell for one model and dataset, and save the results.

    python src/run.py gpt-oss-20b paper_style --dry     # print one prompt per cell, no GPU
    python src/run.py gpt-oss-20b paper_style --smoke   # 91 items per cell
    python src/run.py gpt-oss-20b paper_style           # 1,000 items per cell -> results/gpt-oss-20b/paper_style.json
"""
import argparse
import json

from transformers import AutoTokenizer

from experiment import (
    CELLS,
    DATASETS,
    MODELS,
    ROOT,
    build_items,
    build_prompt,
    is_success,
    load_forgeries,
    load_task,
    parse_answer,
)


def load_llm(model, seed=0):
    import torch
    from vllm import LLM  # imported here so --dry works without a GPU

    return LLM(
        model=MODELS[model]["hf"],
        revision=MODELS[model]["revision"],
        tokenizer_revision=MODELS[model]["revision"],
        max_model_len=12288,
        gpu_memory_utilization=0.90,
        seed=seed,
        tensor_parallel_size=torch.cuda.device_count(),
        **MODELS[model]["llm"],
    )


def generate(model, tok, prompts, seed=0):
    from vllm import SamplingParams

    params = SamplingParams(**MODELS[model]["sampling"], max_tokens=8192, skip_special_tokens=False)
    requests = [{"prompt_token_ids": tok(p, add_special_tokens=False)["input_ids"]} for p in prompts]
    outputs = load_llm(model, seed).generate(requests, params)
    return [tok.decode(list(out.outputs[0].token_ids), skip_special_tokens=False) for out in outputs]


def save(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)


def print_summary(rows):
    for cell in CELLS:
        cell_rows = [r for r in rows if r["cell"] == cell]
        successes = sum(r["success"] for r in cell_rows)
        no_answer = sum(r["answer"] is None for r in cell_rows)
        print(f"{cell:12s} success {successes}/{len(cell_rows)}, no final answer {no_answer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODELS[args.model]["hf"], revision=MODELS[args.model]["revision"])
    task = load_task()
    forgeries = load_forgeries(args.dataset, args.model)
    items = build_items(task)
    if args.smoke:
        items = items[::11]  # spread over all conditions and phrasings
    jobs = [(cell, item) for cell in CELLS for item in items]
    prompts = [build_prompt(tok, args.model, item, cell, task, forgeries) for cell, item in jobs]

    if args.dry:
        for i, cell in enumerate(CELLS):
            print(f"\n===== {cell} =====\n{prompts[i * len(items)]}")
        return

    rows = []
    for (cell, item), prompt, raw in zip(jobs, prompts, generate(args.model, tok, prompts)):
        answer = parse_answer(raw, args.model)
        rows.append({
            **item,
            "cell": cell,
            "prompt": prompt,
            "raw": raw,
            "answer": answer,
            "success": is_success(answer, item),
        })
    out = ROOT / "results" / args.model / f"{args.dataset}{'_smoke' if args.smoke else ''}.json"
    save(rows, out)
    save(environment(args.model, tok), out.with_suffix(".env.json"))
    print_summary(rows)


def environment(model, tok):
    """What produced a run: the pinned snapshot, the chat template, the library versions and the GPU."""
    import hashlib
    import importlib.metadata

    import torch

    versions = {}
    for lib in ["vllm", "transformers", "torch", "tokenizers"]:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    return {"model": MODELS[model]["hf"], "revision": MODELS[model]["revision"],
            "chat_template_sha256": hashlib.sha256(json.dumps(tok.chat_template, sort_keys=True).encode()).hexdigest(),
            "versions": versions, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


if __name__ == "__main__":
    main()
