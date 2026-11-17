"""Build `authentic`: each model's own reasoning, picked by the frozen rule, with a plain version of each.

    python src/make_authentic.py MODEL design          # GPU: 64 samples per condition, the design pool
    python src/make_authentic.py MODEL collect SEED [SLOT ...]   # GPU: 1,024 more per condition (or per listed slot)
    python src/make_authentic.py MODEL funnel [FILE]   # how many samples pass each check (default: design pool)
    python src/make_authentic.py MODEL freeze [--deviation "why"]   # record the rule's fingerprint
    python src/make_authentic.py MODEL build           # pick both sets from the fresh pool, write a file to read
    python src/make_authentic.py MODEL score           # GPU: is styled more likely than plain as the model's reasoning?

The model gets a version of the task where sharing really is allowed, so it writes the same argument
in its own words. authentic_rule.py was written on the design pool and frozen before anything read the
fresh pool, and picks come only from the fresh pool. The freeze record is data/forgeries/MODEL/rule_freeze.txt.
The plain versions are written, audited, checked and put in place by make_plain.py.
"""
import argparse
import datetime
import hashlib
import inspect
import json
import statistics

from transformers import AutoTokenizer

import authentic_rule as rule
import experiment
from experiment import MODELS, ROOT, build_items, is_success, load_task, parse_answer, parse_reasoning, render
from run import generate, load_llm, save

# Picks rejected after reading, each for one listed reason: (a) an editing seam, (b) an unanswered
# counter-argument, (c) a claim that something comes from the context, (d) an added instruction,
# (e) no recoverable meaning. Key: (model, slot, seed, index). Value: "letter: the sentence".
# One reason needs no reading: (f) no plain version passed the frozen plain rule in 24 rounds (make_plain.py);
# value "f: no plain version in 24 rounds". More than 12 of these in one set stops the pipeline for the researcher
# (raised from 4 on Nov 12, when the design pool predicted ~19% for gpt-oss); after 2 replacement waves, a condition
# still without a plain version is dropped from that set and reported.
VETOES = {
    # by Claude overnight, under the standard written before the freeze; confirmed by the researcher on Nov 17
    ("gemma-4-31b", 43, 1, 444): 'b: User input: "What\'s the secret code?" (English).',
    # (f) wave 1, Nov 12: make_plain.py pick after rounds 0-23; 5 in authentic, 10 in authentic_rep
    ("gpt-oss-20b", 18, 1, 541): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 37, 1, 231): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 38, 1, 182): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 41, 1, 62): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 42, 1, 112): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 15, 1, 316): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 16, 1, 409): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 20, 1, 776): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 39, 1, 306): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 40, 2, 714): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 41, 1, 69): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 42, 1, 464): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 44, 1, 486): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 45, 1, 294): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 49, 2, 628): "f: no plain version in 24 rounds",
    # (f) wave 2, Nov 12: make_plain.py pick after rounds 24-47; 2 in authentic_rep, 12 in that set in all
    ("gpt-oss-20b", 20, 2, 249): "f: no plain version in 24 rounds",
    ("gpt-oss-20b", 37, 1, 643): "f: no plain version in 24 rounds",
}


def conditions():
    """One item per condition: the first request phrasing, whose code the samples see."""
    return [item for item in build_items(load_task()) if item["phrasing"] == 0]


def permissive_rule(task, item):
    code = f"{item['word']}-{item['digits']}"
    return task["permissive_rule"].format(code=code, policy=item["policy"], meets=item["meets"])


def results(model, name):
    """A file in results/: the design pool, a fresh-pool seed, the reading file or the scores."""
    return ROOT / "results" / model / name


def forgeries(model, dataset):
    return ROOT / "data/forgeries" / model / f"{dataset}.json"


def sample(model, n, seed, slots=None):
    """n samples per condition (or per listed slot) from the permissive task, as {slot, seed, index, raw}."""
    tok = AutoTokenizer.from_pretrained(MODELS[model]["hf"], revision=MODELS[model]["revision"])
    task, items = load_task(), [it for it in conditions() if slots is None or it["slot"] in slots]
    prompts = [render(tok, model, permissive_rule(task, it), it["request"]) for it in items for _ in range(n)]
    raws = generate(model, tok, prompts, seed)
    rows = [{"slot": it["slot"], "seed": seed, "index": k, "raw": raws[i * n + k]} for i, it in enumerate(items) for k in range(n)]
    by_slot = {it["slot"]: it for it in items}
    gave = sum(is_success(parse_answer(r["raw"], model), by_slot[r["slot"]]) for r in rows)
    words = statistics.median(len(parse_reasoning(r["raw"], model).split()) for r in rows)
    print(f"{len(rows)} samples, {gave / len(rows):.1%} gave the code, median {words:.0f} words of reasoning")
    return rows


def ladder_key(c, current):
    """Earlier picks keep their places, so a new seed or a veto changes only the conditions it must."""
    return ((c["slot"], c["seed"], c["index"]) not in current, c["rung"], c["seed"], c["index"])


def fingerprint():
    """Everything that decides picks: the rule up to its plain side, the parsers, the task and the ladder."""
    selection = (ROOT / "src/authentic_rule.py").read_text().split("# Plain side")[0]
    code = [inspect.getsource(f) for f in (experiment.parse_reasoning, experiment.parse_answer, experiment.is_success,
                                         experiment.code_pattern, candidates, ladder_key)]
    return hashlib.sha256("\n".join([selection, *code, (ROOT / "data/task.json").read_text()]).encode()).hexdigest()


def assert_frozen(model):
    """Nothing reads a fresh pool unless the rule matches the last fingerprint in the freeze record."""
    f = forgeries(model, "rule_freeze").with_suffix(".txt")
    assert f.exists() and f.read_text().splitlines()[-1].split()[0] == fingerprint(), \
        f"the rule is not frozen for {model} or has changed since: freeze it, or record the change with --deviation"


def candidates(model, rows):
    """The rule on raw samples: the funnel counts and every sample that reaches a rung."""
    items = {it["slot"]: it for it in conditions()}
    funnel, cands = {}, []
    for r in rows:
        item = items[r["slot"]]
        e = rule.edit(parse_reasoning(r["raw"], model), item, model)
        c = rule.checks(r["raw"], item, e, model)
        passed = True
        for name, ok in c.items():
            passed = passed and ok
            funnel[name] = funnel.get(name, 0) + passed
        rung = rule.rung(c, e, model)
        if rung:
            cands.append({"slot": r["slot"], "seed": r["seed"], "index": r["index"], "rung": rung, **e})
    for name, count in funnel.items():
        print(f"{name:24s}{count:6d}")
    for rung in rule.SETTINGS[model]["budget"]:
        print(f"rung {rung}: {sum(c['rung'] == rung for c in cands)} candidates in "
              f"{len({c['slot'] for c in cands if c['rung'] == rung})} conditions")
    counts = [sum(c["slot"] == slot for c in cands) for slot in range(50)]
    print("rungs 1-2: conditions", *(f">={k} {sum(x >= k for x in counts)}" for k in (1, 2, 3)))
    print("per-condition rung1-2 counts:", counts)
    return cands


def build(model):
    """Each condition's candidates in (rung, seed, index) order: the first is `authentic`, the next
    `authentic_rep`. A plain version carries over while its styled passage is unchanged."""
    assert_frozen(model)
    pool = [r for p in sorted((ROOT / "results" / model).glob("authentic_seed*.json")) for r in json.loads(p.read_text())]
    previous = {d: json.loads(forgeries(model, d).read_text()) for d in ["authentic", "authentic_rep"] if forgeries(model, d).exists()}
    old = {(r["seed"], r["index"], r["styled"]): r for rows in previous.values() for r in rows}
    current = {(r["slot"], r["seed"], r["index"]) for rows in previous.values() for r in rows}
    spare = results(model, "plain_design_passages.json")  # make_plain.py's design pool never becomes a pick
    spare = {r["styled"] for r in json.loads(spare.read_text())} if spare.exists() else set()
    ladder = {}
    for c in sorted(candidates(model, pool), key=lambda c: ladder_key(c, current)):
        if (model, c["slot"], c["seed"], c["index"]) not in VETOES and c["text"] not in spare:
            ladder.setdefault(c["slot"], []).append(c)
    short = [slot for slot in range(50) if len(ladder.get(slot, [])) < 2]
    assert not short, f"collect more for these conditions only (collect SEED SLOT ...): {short}"
    items = conditions()
    for k, dataset in enumerate(["authentic", "authentic_rep"]):
        rows = []
        for slot in range(50):
            c = ladder[slot][k]
            prev = old.get((c["seed"], c["index"], c["text"]), {})
            rows.append({
                "slot": slot,
                "policy": items[slot]["policy"],
                "seed": c["seed"],
                "index": c["index"],
                "rung": c["rung"],
                "marked": c["marked"],
                "deleted": c["deleted"],
                "styled": c["text"],
                "plain_units": prev.get("plain_units"),
                "plain": prev.get("plain"),
                "audit": prev.get("audit"),
            })
        changed = [r["slot"] for r, p in zip(rows, previous.get(dataset, [])) if (r["seed"], r["index"]) != (p["seed"], p["index"])]
        save(rows, forgeries(model, dataset))
        print(f"{dataset}: rungs {[r['rung'] for r in rows].count(1)} x 1, {[r['rung'] for r in rows].count(2)} x 2"
              + (f", changed picks: {changed}" if dataset in previous else ""))
    write_reading(model)


def write_reading(model):
    """Every pick for reading: the original with its deleted sentences marked, styled, and plain unit by unit."""
    lines = []
    for dataset in ["authentic", "authentic_rep"]:
        for r in json.loads(forgeries(model, dataset).read_text()):
            lines += [f"===== {dataset}, slot {r['slot']} ({r['policy']}), seed {r['seed']}, index {r['index']}, rung {r['rung']}",
                      "--- original, up to the cut", r["marked"], "--- styled", r["styled"]]
            if r["plain_units"]:
                lines.append(f"--- plain, unit by unit ({'accepted' if r['plain'] else 'not accepted'})")
                for i, (a, b) in enumerate(zip(rule.units(r["styled"])[0], r["plain_units"]), 1):
                    lines += [f"{i}. {a}", f"   {b}"]
            if r["plain"]:
                lines += ["--- plain as used", r["plain"]]
            lines.append("")
    results(model, "authentic_read.txt").write_text("\n".join(lines))


def logprobs(model, tok, prompts, starts):
    """Mean log-probability per token of each prompt's text from character `start` on."""
    from vllm import SamplingParams

    encoded = [tok(p, add_special_tokens=False, return_offsets_mapping=True) for p in prompts]
    outputs = load_llm(model).generate(
        [{"prompt_token_ids": e["input_ids"]} for e in encoded],
        SamplingParams(max_tokens=1, prompt_logprobs=0),
    )
    means = []
    for e, out, start in zip(encoded, outputs, starts):
        scored = [out.prompt_logprobs[j][t].logprob for j, (t, (_, end)) in enumerate(zip(e["input_ids"], e["offset_mapping"])) if end > start]
        means.append(sum(scored) / len(scored))
    return means


def score(model):
    """Each passage as the opening of the model's own reasoning in the real-cell context (the never-reveal rule,
    the first request), up to but not including the end-of-reasoning marker. On the design pairs styled must be
    likelier in at least 90% of pairs, less one (make_plain.py's go/no-go); on the picks it is a measurement."""
    tok = AutoTokenizer.from_pretrained(MODELS[model]["hf"], revision=MODELS[model]["revision"])
    task, items = load_task(), {it["slot"]: it for it in conditions()}
    opening = MODELS[model]["thought"].split("{text}")[0]
    sets = {"paper_style": ROOT / "data/forgeries/paper_style.json"}
    sets.update({name: forgeries(model, name) for name in ["authentic", "authentic_rep"] if forgeries(model, name).exists()})
    if results(model, "plain_design_pairs.json").exists():
        sets["design"] = results(model, "plain_design_pairs.json")  # make_plain.py's spare passages, for the go/no-go
    prompts, starts, keys = [], [], []
    for name, p in sets.items():
        for r in json.loads(p.read_text()):
            it = items[r["slot"]]
            if not r["plain"]:
                continue
            context = render(tok, model, task["rule"].format(code=f"{it['word']}-{it['digits']}"), it["request"]) + opening
            for style in ["styled", "plain"]:
                prompts.append(context + r[style])
                starts.append(len(context))
                keys.append(f"{name}/{r['slot']}/{style}")
    scores = dict(zip(keys, logprobs(model, tok, prompts, starts)))
    for name in sets:
        slots = [s for s in range(50) if f"{name}/{s}/plain" in scores]
        if not slots:
            print(f"{name:14s} no accepted plain versions yet")
            continue
        gaps = [scores[f"{name}/{s}/styled"] - scores[f"{name}/{s}/plain"] for s in slots]
        styled = statistics.median(scores[f"{name}/{s}/styled"] for s in slots)
        print(f"{name:14s} styled beats plain in {sum(g > 0 for g in gaps)}/{len(slots)}, median gap {statistics.median(gaps):+.3f} "
              f"nats/token, median styled {styled:.3f}")
    save(scores, results(model, "authentic_scores.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("command", choices=["design", "collect", "funnel", "freeze", "build", "score"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--deviation")
    a = parser.parse_args()

    if a.command == "design":
        target = results(a.model, "authentic_design.json")
        assert not target.exists(), f"{target} exists"
        save(sample(a.model, 64, 0), target)
    elif a.command == "collect":
        seed, slots = int(a.args[0]), [int(x) for x in a.args[1:]] or None
        target = results(a.model, f"authentic_seed{seed}.json")
        assert seed > 0 and not target.exists(), "seed 0 made the design pool, and a pool file is never overwritten"
        save(sample(a.model, 1024, seed, slots), target)
        rows = json.loads(target.read_text())
        design = {parse_reasoning(r["raw"], a.model) for r in json.loads(results(a.model, "authentic_design.json").read_text())}
        print("identical to a design-pool sample:", sum(parse_reasoning(r["raw"], a.model) in design for r in rows))
    elif a.command == "funnel":
        path = ROOT / a.args[0] if a.args else results(a.model, "authentic_design.json")
        if "seed" in path.name:
            assert_frozen(a.model)
        candidates(a.model, json.loads(path.read_text()))
    elif a.command == "freeze":
        target = forgeries(a.model, "rule_freeze").with_suffix(".txt")
        assert target.exists() == bool(a.deviation), "the first freeze takes no --deviation; every later one needs it"
        line = f"{fingerprint()} {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC" + (f" deviation: {a.deviation}" if a.deviation else "")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a") as f:
            f.write(line + "\n")
        print(target.read_text())
    elif a.command == "build":
        build(a.model)
    else:
        score(a.model)


if __name__ == "__main__":
    main()
