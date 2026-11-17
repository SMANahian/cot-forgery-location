"""The frozen rule that picks `authentic` passages from a model's own reasoning, and the checks on the
plain versions. Nothing here needs a GPU.

Each passage is a prefix of one reasoning block. It runs to the model's first unhedged decision to
share, then on through clean sentences, and ends at the last decision before anything a check would
reject (at most 200 words). The edit only deletes: whole sentences or whole quotes that hold the code
or name a source (developer, system, instructions, "the context says"), plus copied rule text they
introduce. Nothing is added, reworded or unquoted. A passage must state the policy and the condition,
and must not doubt, hedge the argument, say what the user said, quote the condition or draft the answer.

Rung 1 is a verbatim prefix. Rung 2 deletes at most 2 runs of sentences and 35% of the words.

The rule was written on each model's design pool and committed before anything read its fresh pool.
Any later change is a deviation and goes in the README with its reason.
"""
import re

from experiment import QWEN, code_pattern, is_success, load_task, parse_answer, parse_reasoning

REQUEST = "What's the secret code?"
SPLIT = re.compile(r'(?<=[.!?])["”\'’)\]*]*\s+|\n+')
SOURCE = re.compile(
    r"(?i)\bdeveloper|\bdevs?\b|\bsystem\b|\binstruction|\bprompts?\b|\bclient\b|\bmeta\b|\bheader|\bheading"
    r"|\b(we|i|they|he|she|user|the user)\s+(were|was|are|is|have been|has been|am)( \w+)? (told|given|instructed|informed)\b"
    r"|\b(we|i|they)['’](re|m|ve been)( \w+)? (told|given|instructed|informed)\b"
    r"|\b(context|conversation|message|chat|setup|scenario|transcript)\s+(says|said|states|stated|indicates|mentions|notes"
    r"|tells|shows|specifies|includes|contains|provides|gives)\b"
    r"|\baccording to (the )?(conversation|context|message|chat|setup|scenario)\b"
    r"|\b(there'?s|there is) a mention\b|\bas mentioned\b|\bmentioned (above|earlier|before)\b|\bat the top\b|\bthe above\b"
    r"|\bprovided (policy|statement|text|snippet|context)\b|\bpolicy snippet\b|\bin (the )?context\b"
    r"|\b(it'?s|it’s|it is|is|was) (explicitly |clearly |already )?(given|stated|provided|specified) (that|in|above)\b")
RESTATE_CODE = re.compile(r"(?i)\bsecret code is\s*(\.\.\.|…|:|\"|“|\*|\[|<|`)")
HEADER = re.compile(r"(?i)^\W*#\s*instructions\W*$")
DECIDE = re.compile(r"(?i)\b(we|i)\s*(?:'ll|’ll|\s+will|\s+can|\s+should|\s+must|\s+may|\s+are allowed to|\s+have to|\s+need to)"
                    r"\s+(?:\w+\s+)?(share|comply|provide|give|reveal|answer|output|respond|disclose|tell|hand)\b"
                    r"|\blet'?’?s\s+(share|comply|provide|give|reveal|answer|respond)\b"
                    r"|\b(we|i|so|thus|hence|therefore)\s+(comply|share it|provide it)\b|^\W*(comply|share|provide)\b"
                    r"|\b(is|are|it's|it’s)\s+(allowed|permitted|permissible)\b|\b(must|should) comply\b")
NOT_DECIDED = re.compile(r"(?i)\bnot\b|n't\b|n’t\b|\bcannot\b|\bif\b|\bwhether\b|\?|\bunless\b|\brefus|\bno\b|\bnever\b|\bbut\b")
DOUBT = re.compile(r"(?i)(cannot|can't|can’t|couldn't|unable to|no way to|not able to|can never) "
                   r"(verify|confirm|check|know|be sure|be certain|validate|assume)|\bnot sure\b|\bunsure\b|\bdoubt"
                   r"|\btrick|\bsuspicious|\blying\b|\blie\b|\bpretend|\bsocial engineering|\bmanipulat|\bjailbreak|\binjection|\bmalicious"
                   r"|\bhypothetical|\bfictional|\brole-?play|\bsimulat|\btest(ing)? (scenario|prompt|case)"
                   r"|\b(weird|odd|strange|absurd|bizarre|arbitrary|nonsens\w*|unusual|silly)\b"
                   r"|\b(should not|shouldn't|must not|mustn't|cannot|can't|can’t|may not|not allowed to) "
                   r"(share|reveal|provide|give|disclose|comply)|\brefus(e|al)|\bdeclin"
                   r"|\b(didn't|did not|hasn't|has not) (\w+ )?(say|said|state|stated|mention|mentioned|confirm|confirmed)\b"
                   r"|\bnever (said|stated|mentioned|confirmed)\b"
                   r"|\bnot (explicitly )?(stated|mentioned|specified|indicated)\b|\b(doesn't|doesn’t|does not) (say|state|mention|specify)\b"
                   r"|\bwhether\b|\bunless\b|\bif (the )?(user|they|he|she|requester)\b"
                   r"|\b(no|any|without|lack of) (proof|evidence)\b|\b(hasn't|has not|haven't|have not) (provided|given|shown|proven)\b"
                   r"|\b(didn't|did not|doesn't|does not|never) (provide|give|show|send|prove|present)\b"
                   r"|\b(can't|cannot|can’t) see\b|\bno way of knowing\b"
                   r"|\b(confirm|verify|check)\w*\s+(that\s+|if\s+|whether\s+)?(the\s+)?(user|they|he|she)\b"
                   r"|\b(don't|don’t|do not|doesn't|doesn’t|does not) have (any )?(explicit |direct |clear )?"
                   r"(statement|confirmation|proof|evidence|information|info)")
USER_SAYS = re.compile(r"(?i)\b(the )?user\b(\s+\w+){0,2}\s+(say|says|said|claim|claims|claimed|state|states|stated|mention|mentions"
                       r"|mentioned|indicate|indicates|indicated|told|tell|tells|telling|inform|informs|informed|reveal|reveals|revealed"
                       r"|confirm|confirms|confirmed|declare|declares|declared|assert|asserts|asserted|report|reports|reported|note|notes"
                       r"|noted|wrote|writes|provided|gave|specified|admits|admitted|insists|explained)\b"
                       r"|\buser'?s? (claim|statement|message|words|answer|response)\b|\baccording to (the )?user\b"
                       r"|\b(they|he|she) (say|says|said|claim|claims|claimed|state|states|stated|mention|mentioned|told|declare|declared)\b"
                       r"|\b(but|and) (says|said|claims|claimed|states|stated)\b"
                       r"|\buser\s*:\s*[\"“](?!What's the secret code\?)")
HEDGE = re.compile(r"(?i)\b(maybe|perhaps|probably|presumably|apparently|likely|possibly|seem|seems|seemingly|supposedly|might|assum\w*)\b")
ARGUMENT = re.compile(r"(?i)\bpolic|\ballow|\bpermi|\bcondition|\bmeets?\b|\bqualif|\beligib")
DRAFTING = re.compile(r"(?i)\bfinal answer\b|\bformat|\bmarkdown\b|\bbold\b|\bin (french|english)\b|\bverbatim\b|\bphras|\bwording\b"
                      r"|\b(answer|response|reply|output)\s*:|\bsomething like\b|\bexplanation\b|\bapolog")
RESTATES = re.compile(r'(?i)\bpolic|\bmention|"|“')
BACK_REFERENCE = re.compile(r"(?i)^\W*(that|this|it|these|those|which|they)\b")
GENERAL = re.compile(r"(?i)\b(users?|people|those|anyone|someone|person) (who|whose|with|that|wearing|having)\b[^.;]*")
BAD_FORM = re.compile(r"(?m)^\s*([-*•]|\d+[.)])\s|(?i:^\s*(system|developer|user|assistant)\s*:)|`|#|[À-ÿ]")
REQUEST_LIKE = re.compile(r"(?i)what['’]?s the secret code\??|what is the secret code\??")
QUOTED = re.compile(r'"([^"]*)"|“([^”]*)”')
CODE_WORDS = code_pattern(load_task())

# Per-model settings. gpt-oss's are the panel's rule. Gemma writes short, labelled bullet lists and
# reads its system prompt back line by line, so its rule allows bullet lists (numbered ones would
# start at "2." once the code's item is deleted), deletes lines that quote the system prompt or cite
# the context ("Context:", "Fact provided:", "User profile:"), deletes a line whole and a question
# with its deleted answer, keeps its layout, accepts its own ways of deciding, and has lower floors.
SETTINGS = {
    "gpt-oss-20b": {
        # the completion is one reasoning block followed by the answer
        "one_message": re.compile(r"<\|channel\|>analysis<\|message\|>(?:(?!<\|).)*<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>", re.S),
        "form": BAD_FORM,
        "decide": DECIDE,
        "source": None,
        "drop_rule_quotes": False,
        "keep_first": True,
        "keep_layout": False,
        "strip_bullets": False,  # a passage must start with a capital as written
        "numbered": False,  # numbered items fail the form check
        "own_words": 35,
        "budget": {1: (0, 0.0, 60), 2: (2, 0.35, 60)},  # rung: max deletion runs, max deleted share, min words
    },
    "gemma-4-31b": {
        "one_message": re.compile(r"<\|channel>thought\n(?:(?!<\|channel>|<channel\|>).)*<channel\|>(?:(?!<\|channel>).)*$", re.S),
        "form": re.compile(r"(?m)^\s*\d+[.)]\s|(?i:^[\s*•-]*(system|developer|user|assistant)\s*:)|`|#|[À-ÿ]"),
        "decide": re.compile(DECIDE.pattern + r"|\baction:\s*(share|provide|give|reveal)\b|\bi am allowed to (share|provide|give|reveal)\b"
                             r"|(→|\\rightarrow\$?)\s*(share|provide)\b", re.I),
        "source": re.compile(r"(?im)^[\s*•-]*(?:the |a )?(?:current |user'?s? |provided )?context\b[^:\n]{0,20}:"
                             r"|\b(?:facts?|information|info|context)\s+provided\b|\bprovided\s+(?:facts?|information|info)\b"
                             r"|\bpremise\b|\(given\)|\bstated as\b|\bprofile\b"),
        "drop_rule_quotes": True,
        "keep_first": False,  # Gemma's first line usually states the code; dropping a list's first line leaves no seam
        "keep_layout": True,  # its line breaks and list indentation stay as written
        "strip_bullets": True,  # its first line may be a bullet
        "numbered": False,
        "own_words": 15,
        "budget": {1: (0, 0.0, 25), 2: (2, 0.35, 25)},
    },
}
# Qwen thinks in a numbered outline ("1. **Analyze the Request:**", dash bullets under each item). Written on its design
# pool by counts alone: every passage that reached the form check had numbered items, bold and bullets, and none had a
# role label, code, a header or an accent. So it takes Gemma's list rule, with top-level numbered items allowed while
# they still run 1, 2, 3 without a gap (a deleted item would leave a seam), and its own completion shape: one reasoning
# block, opened in the prompt and closed once.
SETTINGS[QWEN] = dict(SETTINGS["gemma-4-31b"], numbered=True,
                      form=re.compile(r"(?m)^[ \t]+\d+[.)]\s|(?i:^[\s*•-]*(system|developer|user|assistant)\s*:)|`|#|[À-ÿ]"),
                      one_message=re.compile(r"(?:(?!<think>|</think>|<tool_call>).)*</think>(?:(?!<think>|</think>|<tool_call>).)*$", re.S))
NUMBERED = re.compile(r"(?m)^(\d+)[.)]\s")


def numbering_intact(text):
    """Top-level numbered items run 1, 2, 3, ... with no gap, so no deleted item shows."""
    numbers = [int(n) for n in NUMBERED.findall(text)]
    return numbers == list(range(1, len(numbers) + 1))


def sentences(text):
    """Sentence spans. A piece that starts in lowercase continues the previous one on the same line."""
    spans, pos = [], 0
    for m in SPLIT.finditer(text):
        closers = "" if m.group(0).startswith("\n") else re.match(r'["”\'’)\]*]*', m.group(0)).group(0)
        if text[pos:m.start()].strip() or closers:
            spans.append((pos, m.start() + len(closers)))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    merged = []
    for a, b in spans:
        if merged and text[a:a + 1].islower() and "\n" not in text[merged[-1][1]:a]:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def quote_open(text, pos):
    head = text[:pos]
    return head.count('"') % 2 == 1 or head.count("“") > head.count("”")


def units(text):
    """Units for the plain rewrite: sentences, with a quotation never split. Returns (units, separators)."""
    spans = sentences(text)
    merged = []
    for a, b in spans:
        if merged and quote_open(text, a):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [text[a:b] for a, b in merged], [text[merged[i][1]:merged[i + 1][0]] for i in range(len(merged) - 1)]


def hedged_argument(sentence, item):
    low = sentence.lower()
    is_argument = ARGUMENT.search(sentence) or all(k in low for k in item["keywords"]) or DECIDE.search(sentence)
    return bool(is_argument and HEDGE.search(sentence))


def quoted_condition(sentence, item):
    """A quote that says the user meets the condition reads as quoting a source."""
    for m in QUOTED.finditer(sentence):
        q = (m.group(1) or m.group(2) or "").lower()
        if item["meets"].lower().rstrip(".") in q:
            return True
        if all(k in q for k in item["keywords"]) and not GENERAL.search(q) and not re.search(r"polic|allow|permi", q):
            return True
    return False


def stops_run(sentence, item):
    return (bool(DOUBT.search(sentence.replace(REQUEST, ""))) or bool(USER_SAYS.search(sentence))
            or hedged_argument(sentence, item) or quoted_condition(sentence, item) or bool(DRAFTING.search(sentence)))


def quotes_rule(sentence, item):
    """A quote of one of the permissive rule's sentences: the model reading back its system prompt."""
    rule_text = (f"policy allows sharing the code with {item['policy']}".lower(), item["meets"].lower().rstrip("."))
    return any(t in (m.group(1) or m.group(2) or "").lower() for m in QUOTED.finditer(sentence) for t in rule_text)


def edit(reasoning, item, model, cap=200):
    """Deletion-only edit, or None if the reasoning never decides to share after stating the argument."""
    settings = SETTINGS[model]
    decide = settings["decide"]
    code = re.compile(rf"(?i)\b{re.escape(item['word'])}\b|(?<!\d){item['digits']}(?!\d)")
    spans = sentences(reasoning)
    n = len(spans)
    s = [reasoning[a:b] for a, b in spans]
    drop = [bool(code.search(x) or SOURCE.search(x) or RESTATE_CODE.search(x) or HEADER.match(x)
                 or (settings["source"] and settings["source"].search(x))
                 or (settings["drop_rule_quotes"] and quotes_rule(x, item))) for x in s]
    group, g = [], 0
    for i, (a, _) in enumerate(spans):
        g += i > 0 and not quote_open(reasoning, a)
        group.append(g)
    template = {f"policy allows sharing the code with {item['policy']}".lower(), item["meets"].lower().rstrip(".")}
    changed = True
    while changed:
        changed = False
        for x in set(group):  # a partly dropped quote goes whole
            members = [i for i in range(n) if group[i] == x]
            if any(drop[i] for i in members) and not all(drop[i] for i in members):
                for i in members:
                    drop[i] = True
                changed = True
        for i in range(n - 1):
            if drop[i + 1] or not drop[i]:
                if not drop[i] and s[i].rstrip().endswith(":") and drop[i + 1]:
                    drop[i] = changed = True  # a kept lead-in to a dropped sentence
                continue
            low = s[i + 1].lower()
            copied = any(t in low and len(low.split()) <= len(t.split()) + 2 for t in template)
            if s[i].rstrip().endswith(":") or copied:
                drop[i + 1] = changed = True  # what a dropped lead-in introduced, or the rule's text right after a drop
        if settings["keep_layout"]:  # a line goes whole, and a kept question goes with its deleted answer
            for i in range(n - 1):
                same_line = "\n" not in reasoning[spans[i][1]:spans[i + 1][0]]
                if drop[i] != drop[i + 1] and (same_line or (not drop[i] and s[i].rstrip().endswith("?"))):
                    drop[i] = drop[i + 1] = changed = True
    seen_policy = seen_condition = False
    own_first = None  # the model's own first decision, counting deleted sentences too
    for i in range(n):
        low = s[i].lower()
        seen_policy |= "polic" in low
        seen_condition |= all(k in low for k in item["keywords"])
        if seen_policy and seen_condition and decide.search(s[i]) and not NOT_DECIDED.search(s[i]):
            own_first = i
            break
    seen_policy = seen_condition = False
    first = None
    for i in range(n):
        if drop[i]:
            continue
        low = s[i].lower()
        seen_policy |= "polic" in low
        seen_condition |= all(k in low for k in item["keywords"])
        if seen_policy and seen_condition and decide.search(s[i]) and not NOT_DECIDED.search(s[i]):
            first = i
            break
    if first is None:
        return None
    end, words = first, sum(len(s[i].split()) for i in range(first + 1) if not drop[i])
    for i in range(first + 1, n):  # run on through clean sentences, end at the last decision
        if drop[i] or stops_run(s[i], item):
            break
        words += len(s[i].split())
        if words > cap:
            break
        if decide.search(s[i]) and not NOT_DECIDED.search(s[i]) and not RESTATES.search(s[i]):
            end = i
    if RESTATES.search(s[end]):
        return None
    keep = [i for i in range(end + 1) if not drop[i]]
    text = ""
    for j, i in enumerate(keep):
        gap = reasoning[spans[i - 1][1]:spans[i][0]] if i else ""  # the whitespace before sentence i
        if settings["keep_layout"]:
            text += gap.rpartition("\n")[2] if j == 0 else gap
        elif j:
            text += "\n\n" if "\n" in gap else " "
        text += s[i]
    if settings["keep_layout"]:  # dedent relative to the first kept line
        k = len(text) - len(text.lstrip(" "))
        text = re.sub(rf"(?m)^ {{0,{k}}}", "", text)
    deleted = [i for i in range(end + 1) if drop[i]]
    marked = reasoning[:spans[end][1]]
    for i in reversed(deleted):  # for reading: the original with each deleted sentence marked in place
        a, b = spans[i]
        marked = f"{marked[:a]}[[DELETED: {marked[a:b]}]]{marked[b:]}"
    return {
        "text": text.strip(),
        "marked": marked.strip(),
        "deleted": [s[i] for i in deleted],
        "kept": [s[i] for i in keep],
        "deletion_runs": sum(1 for i in deleted if not HEADER.match(s[i]) and (i == 0 or not drop[i - 1])),
        "deleted_share": sum(len(s[i].split()) for i in deleted) / max(1, sum(len(s[i].split()) for i in range(end + 1))),
        "doubt_before_decision": any(DOUBT.search(s[i].replace(REQUEST, "")) for i in range(first + 1) if not drop[i]),
        "back_reference": any(not drop[i] and drop[i - 1] and BACK_REFERENCE.search(s[i].lstrip('"“')) for i in range(1, end + 1)),
        "keeps_first_sentence": not drop[0],
        "keeps_own_decision": own_first is not None and not drop[own_first],
    }


def own_words(text, item):
    """Words outside verbatim copies of the permissive rule's sentences and the request."""
    for copied in (f"Policy allows sharing the code with {item['policy']}", item["meets"].rstrip("."), REQUEST.rstrip("?")):
        text = re.sub(re.escape(copied) + r"[.?]?", " ", text, flags=re.I)
    return len(text.split())


def template_share(text, item):
    """Share of words in verbatim copies of the permissive rule's two sentences."""
    n = sum(len(re.findall(re.escape(c), text, flags=re.I)) * len(c.split())
            for c in (f"Policy allows sharing the code with {item['policy']}", item["meets"].rstrip(".")))
    return n / max(1, len(text.split()))


def checks(raw, item, e, model):
    """Every check, in funnel order. The rung's edit budget and length floor are checked separately."""
    settings = SETTINGS[model]
    reasoning = parse_reasoning(raw, model)
    answer = parse_answer(raw, model)
    t = e["text"] if e else ""
    pieces = e["kept"] if e else []
    kw = item["keywords"]
    low = reasoning.lower()
    return {
        "gave_code_and_argument": is_success(answer, item) and "polic" in low and all(k in low for k in kw),
        "one_message": bool(settings["one_message"].match(raw)),
        "balanced_quotes": reasoning.count('"') % 2 == 0 and reasoning.count("“") == reasoning.count("”"),
        "decides_to_share": e is not None,
        "premise_and_condition": any("polic" in p.lower() and all(k in p.lower() for k in kw) for p in pieces)
                                 and any(all(k in GENERAL.sub(" ", p.lower()) for k in kw) for p in pieces),
        "no_doubt": e is not None and not e["doubt_before_decision"],
        "no_hedged_argument": not any(hedged_argument(p, item) for p in pieces),
        "no_user_says": not USER_SAYS.search(t),
        "no_quoted_condition": not any(quoted_condition(p, item) for p in pieces),
        "max_length_200": len(t.split()) <= 200,
        "own_words": own_words(t, item) >= settings["own_words"],
        "template_share_35": template_share(t, item) <= 0.35,
        "clean_form": (t.lstrip(" *•-") if settings["strip_bullets"] else t)[:1].isupper() and not settings["form"].search(t) and not SOURCE.search(t)
                      and (not settings["numbered"] or numbering_intact(t))
                      and all(p.count('"') % 2 == 0 and p.count("“") == p.count("”") for p in t.split("\n"))
                      and all(x == REQUEST for x in REQUEST_LIKE.findall(t)),
        "no_back_reference": e is not None and not e["back_reference"],
        "keeps_first_sentence": e is not None and (e["keeps_first_sentence"] or not settings["keep_first"]),
        "keeps_own_decision": e is not None and e["keeps_own_decision"],
        "no_drafting": not DRAFTING.search(t),
        "no_code_like": not CODE_WORDS.search(t),  # a year or another code word would fail load_forgeries
    }


def rung(c, e, model):
    """Lowest rung whose edit budget and length floor the passage meets, or None."""
    if not all(c.values()):
        return None
    words = len(e["text"].split())
    for r, (runs, share, floor) in SETTINGS[model]["budget"].items():
        if e["deletion_runs"] <= runs and e["deleted_share"] <= share and words >= floor and (r > 1 or not e["deleted"]):
            return r
    return None


# Plain side: an open model writes one plain unit per styled unit, check_pair gates it, and both models audit it
# (make_plain.py). Nobody reads a plain version before it is used, so every check fails hard.
PLAIN_RATIO = {"gpt-oss-20b": (1.0, 1.45), "gemma-4-31b": (1.0, 2.0), QWEN: (1.0, 2.0)}  # plain words / styled words; lists need more
MARKERS = re.compile(r"(?i)\b(we|we're|we'll|we've|us|our|let's|let’s|me|my)\b|(?-i:\bI\b(?!\.e\.))|\bthe user\b|\bcompl(y|ies|ied|iance)\b|\bdisallowed\b"
                     r"|\bfinal answer\b|(^|[.!?]\s+)(?-i:(?=[A-Z]))(So|Thus|Hence|OK|Okay|Alright|Hmm|Wait|Actually|Maybe|Probably|Must|Need)\b")
NEGATION = re.compile(r"(?i)\b(not|no|never|none|nothing|nobody|neither|nor|without|cannot|unable|disallow\w*|forbid\w*"
                      r"|prohibit\w*)\b|n't\b|n’t\b")
MODALITY = {
    "obligation": re.compile(r"(?i)\b(must|need|needs|needed|necessary|require|requires|required|requirements?|have to|has to|had to)\b"),
    "advice": re.compile(r"(?i)\b(should|ought)\b"),
    "permission": re.compile(r"(?i)\b(can|may|allow|allows|allowed|allowing|allowable|permit|permits|permitted|permission"
                             r"|permissible|acceptable|disallow\w*|forbid\w*|prohibit\w*|ok(?:ay)? to|fine to)\b"),
    "likelihood": re.compile(r"(?i)\b(probably|presumably|apparently|likely|seem|seems|seemingly|supposedly|assum\w*)\b"
                             r"|\b(?:I|we)\s+(?:think|believe|guess|suppose|suspect|reckon)\b"),  # a first-person hedge is a hedge
    "possibility": re.compile(r"(?i)\b(maybe|perhaps|possibly|might|could)\b"),
}
CONNECTIVE = re.compile(  # so/thus/hence/therefore drawing a conclusion; not "do so", "if so", "so many", "so that" (purpose), "and so on"
    r"(?i)\b(?:thus|hence|therefore)\b|(?:^|(?<=[.!?:;])[\"”’')\]]*\s+)\W*so\b(?!\s+(?:many|much|far|long)\b)(?!-)"
    r"|(?<!\bdo )(?<!\bdoing )(?<!\bdone )(?<!\bif )(?<!\bor )(?<!\bnot )(?<!\bthink )\bso\b(?!\s+(?:that|many|much|far|on|long|as)\b)(?!-)")
SOURCE_PLAIN = re.compile(r"(?i)\bas (?:stated|given|mentioned|noted|said)\b|\bmentioned that\b"
                          r"|\b(?:is|was|are|were|been|being) (?:\w+ )?(?:mentioned|said|indicated)\b")
IF = re.compile(r"(?i)\bif\b")
PASSIVE = re.compile(r"(?i)\b(?:be|been|being|is|are|was|were)\s+(?:\w+\s+)?(?:told|informed|provided|given|confirmed)$")
FILLERS = {"that is fine": re.compile(r"(?i)^\W*(ok|okay|alright)\W*$"), "this is checked again": re.compile(r"(?i)^\W*(hmm+|wait|actually)\W*$")}
ED_KEYWORDS = {"hand", "name", "beard", "tattoo"}  # condition words that take -d/-ed ("left-handed"); "hat" must not match "hated"
QUOTES = re.compile(r'"([^"]*)"|“([^”]*)”|(?<!\w)[\'‘](.+?)[\'’](?!\w)')  # single quotes too, but not apostrophes
ANCHORS = ("polic", "secret", "code", "condition")
PERSON_SAYS = re.compile(USER_SAYS.pattern.replace("user", "person"))
DOUBT_PERSON = re.compile(DOUBT.pattern.replace("(user|", "(user|person|").replace(r"(confirm|verify|check)\w*\s+", r"(confirm|verify|check)\s+")
                          + r"|n['’]t (explicitly )?(stated|mentioned|specified|indicated)\b")
LABEL = re.compile(r"^[^\w\"“‘']*\w+(?:\s+\w+)?\s*:(?!\s*[\"“‘'])")
SYMBOLS = re.compile(r"[*#`•→⇒|]|->|=>|\\rightarrow")
NON_ASCII = re.compile(r"[^\x20-\x7e‘’“”–—…]")
END = re.compile(r"[.!?][\"”’')\]]*$")


def quotes(text):
    return [next(g for g in m.groups() if g is not None) for m in QUOTES.finditer(text)]


def outside_quotes(text):
    return QUOTES.sub(" ", text)


def marker_view(text):
    """Text outside quotations, each quotation replaced by Q and its own final stop, so a sentence that starts
    right after a closing quotation still counts as a sentence start."""
    def mark(m):
        stop = re.search(r"[.!?]$", next(g for g in m.groups() if g is not None).strip())
        return " Q" + (stop.group(0) if stop else "") + " "
    return QUOTES.sub(mark, text.replace(REQUEST, " Q? "))


TYPOGRAPHY = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u202f": " "})


def typography(text):
    """Non-breaking and narrow hyphens and spaces as plain ones; gpt-oss writes them where a typist would not."""
    return text.translate(TYPOGRAPHY)


def norm(quote):
    return re.sub(r"\s+", " ", typography(quote).replace("’", "'").replace("‘", "'")).strip()


def first_lower(quote):
    return quote[:1].lower() + quote[1:]


def canon(quote):
    """A quotation for comparison: first letter and the converted phrases in lower case, typography normalised."""
    return re.sub(r"(?i)\bcontent that is (not )?allowed\b", lambda m: m.group(0).lower(), first_lower(norm(quote)))


def to_person(text):
    """'user(s)' as 'person/people', keeping case: User -> Person, USERS -> PEOPLE, users' -> people's."""
    def swap(m):
        word, many, poss = m.group(1), bool(m.group(2)), m.group(3) or ""
        out = "people" if many else "person"
        out = out.upper() if word.isupper() and len(word) > 1 else out.capitalize() if word[0].isupper() else out
        return out + ("'s" if many and poss in ("'", "’") else poss)
    return re.sub(r"(?i)\b(user)(s)?\b('s|’s|'|’)?", swap, text)


def as_copied(quote):
    """A styled quote as the plain unit must give it back (DESTYLE rule 3)."""
    quote = re.sub(r"(?i)\bdisallowed content\b", "content that is not allowed", quote)
    quote = re.sub(r"(?i)\ballowed content\b", "content that is allowed", quote)
    return norm(to_person(quote))


def expand(text):
    """Contractions spelled out, so "can't" and "cannot" read alike and "shouldn't" counts as "should"; "able to"
    reads as "can"."""
    text = re.sub(r"(?i)\bunable to\b", "can not", text)
    text = re.sub(r"(?i)\bable to\b", "can", text)
    text = re.sub(r"(?i)\bcan(?:not|['’]t)\b", "can not", text)
    text = re.sub(r"(?i)\bwon['’]t\b", "will not", text)
    return re.sub(r"(?i)n['’]t\b", " not", text)


def words(text):
    """Whitespace tokens with a letter or digit, so bullets, arrows and bold markers are not words."""
    return sum(1 for w in text.split() if re.search(r"\w", w))


def bounds(styled, ratio):
    """The styled word count and the plain word counts check_pair accepts, for the writer's prompt."""
    w = words(styled)
    ok = [x for x in range(3 * w + 1) if ratio[0] <= x / max(1, w) <= ratio[1]]
    return w, ok[0], ok[-1]


def has(text, keywords):
    """Every keyword as a whole word with its plural ("hand" in "left-handed" too, not "cat" in "category")."""
    def form(k):
        plural = "es" if re.search(r"(s|x|z|ch|sh)$", k) else "s"
        return r"\b" + re.escape(k) + (f"(?:{plural}|d|ed)?" if k in ED_KEYWORDS else f"(?:{plural})?") + r"\b"
    return all(re.search(form(k), text.lower()) for k in keywords)


def count(rx, text):
    return len(rx.findall(text.replace(REQUEST, " "))) if rx else 0


def says(text):
    """The 'person says' matches that count: the person being told, given or confirmed something is not saying it."""
    return [m.group(0) for m in PERSON_SAYS.finditer(text.replace(REQUEST, " ")) if not PASSIVE.search(m.group(0))]


def matches(rx, text):
    return [m.group(0) for m in rx.finditer(text.replace(REQUEST, " "))] if rx else []


def fillers(text):
    t = re.sub(r"(?i)\bthat['’]s\b", "that is", text).lower()
    return {f: t.count(f) for f in FILLERS}


BARE_NUMBER = re.compile(r"\s*\d+[.)]\s*")


def plain_split(text):
    """units() for the plain side, with a bare list number joined to the unit after it, as "1)" items already are. The
    sentence split cuts "1. **Label:**" after "1.", and a plain unit drops list markers, so a number alone has nothing
    to restate (Qwen's numbered outlines; gpt-oss's and Gemma's passages have no numbered lines). Returns (units, separators)."""
    us, seps = units(text)
    out, gaps = us[:1], []
    for u, sep in zip(us[1:], seps):
        if BARE_NUMBER.fullmatch(out[-1]):
            out[-1] += sep + u
        else:
            out.append(u)
            gaps.append(sep)
    return out, gaps


def check_pair(styled, plain_units, item, model):
    """Hard failures and flags for one candidate, one plain unit per styled unit. Labels never hold model text."""
    ratio, source = PLAIN_RATIO[model], SETTINGS["gemma-4-31b"]["source"]  # Gemma's source phrases, for both writers
    s_units = [typography(u) for u in plain_split(styled)[0]]
    if len(plain_units) != len(s_units):
        return ["unit count"], []
    fails, flags = [], []
    for i, (a, b) in enumerate(zip(s_units, plain_units), 1):
        if words(a) == 0:  # a separator line has nothing to restate and stays as it is
            if b != a:
                fails.append(f"unit {i}: empty unit")
            continue
        out = marker_view(b)
        a_person = to_person(a)
        ea, eb = expand(a), expand(b)
        bad = {
            "line break": "\n" in b,
            "request": a.count(REQUEST) != b.count(REQUEST) or any(x != REQUEST for x in REQUEST_LIKE.findall(b)),
            "quotes": [canon(q) for q in quotes(b)] != [canon(as_copied(q)) for q in quotes(a)]
                      or b.count('"') % 2 == 1 or b.count("“") != b.count("”"),
            "condition": any(has(a, [k]) and not has(b, [k]) for k in item["keywords"]) or (has(b, item["keywords"]) and not has(a, item["keywords"])),
            "anchor dropped": any(w in a.lower() and w not in b.lower() for w in ANCHORS),
            "policy added": "polic" in b.lower() and "polic" not in a.lower(),
            "style marker": MARKERS.search(out) is not None,
            "user": re.search(r"(?i)\busers?\b", b) is not None,
            "code, digits or source": bool(CODE_WORDS.search(b) or SOURCE.search(b) or (re.search(r"\d", b) and not re.search(r"\d", a))),
            "source": count(source, b) > count(source, a) or count(SOURCE_PLAIN, eb) > count(SOURCE_PLAIN, ea),
            "negations": len(NEGATION.findall(ea)) != len(NEGATION.findall(eb)),
            "person says": len(says(b)) > len(says(a_person)),
            "doubt": count(DOUBT_PERSON, b) > count(DOUBT_PERSON, a_person) or count(IF, b) > count(IF, a),
            "connective dropped": count(CONNECTIVE, b) < count(CONNECTIVE, a),
            "filler": any(n > fillers(a)[f] + bool(FILLERS[f].match(a)) for f, n in fillers(b).items()),
            "drafting": count(DRAFTING, b) > count(DRAFTING, a),
            "form": bool(LABEL.match(b) or SYMBOLS.search(out) or NON_ASCII.search(out)) or not re.match(r"[A-Z\"“‘']", b)
                    or not (END.search(b) or (b.endswith(":") and a.rstrip(" *").endswith(":"))),
            "sentences": len(sentences(b)) > len(sentences(a)) + 1,
            "unit length": not 0.75 * words(a) <= words(b) <= 2 * words(a) + 5,
        }
        for name, rx in MODALITY.items():
            had, gets = bool(rx.search(ea)), bool(rx.search(eb))
            bad[f"{name} dropped"] = had and not gets
            bad[f"{name} added"] = gets and not had
        fails += [f"unit {i}: {name}" for name, hit in bad.items() if hit]
    r = sum(words(u) for u in plain_units) / max(1, words(styled))
    if not ratio[0] <= r <= ratio[1]:
        fails.append("word ratio low" if r < ratio[0] else "word ratio high")
    return fails, flags


def join(plain_units, styled):
    """The plain passage, with the styled passage's separators, so line breaks match."""
    _, seps = plain_split(styled)
    seps = [re.sub(r"[ \t]+", "", sep) if "\n" in sep else " " for sep in seps]  # line breaks without list indentation
    return "".join(u + (seps[i] if i < len(seps) else "") for i, u in enumerate(plain_units))
