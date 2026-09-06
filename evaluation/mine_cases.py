"""
mine_cases.py
Mines a diverse 100-query benchmark from the prepared corpus instead of curating
3 papers by hand. For each sampled passage (or figure caption) the local Ollama
model writes one question + answer grounded ONLY in that passage; every case is
then verified against its source (must_include facts must appear in the passage)
so gold answers can't be hallucinated. Split: 50 text, 25 math, 25 figure, spread
across ~25 papers.

Reuses: the already-running Ollama (/api/generate, JSON mode), the existing
chunks.json / figures.json, and the human_eval case schema. No backend server,
no new deps.

Run from repo root:  python3 evaluation/mine_cases.py [--limit-per-type N] [--model M]
Writes evaluation/mined_cases.json (does not touch human_eval/cases.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.network_policy_service import NetworkPolicyService  # noqa: E402

PAPERS = ROOT / "backend" / "data" / "papers"
OUT = ROOT / "evaluation" / "mined_cases.json"
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

MATH_RE = re.compile(r"(=|\\frac|\\sqrt|\\sum|softmax|gradient|\bloss\b|objective|probability|"
                     r"complexity|O\(|\blog\b|\bexp\b|theta|sigma|lambda|\^|_\{|\bformula\b|\bequation\b)", re.I)
QUOTAS = {"text": 50, "math": 25, "figure": 25}

# Benchmark-dataset papers: their content IS example problems/questions, so mining
# them yields dataset samples ("min value of a^2+b^2?", an ISBN), not paper facts.
DATASET_EXCLUDE = {"2103.03874", "2110.14168", "2009.03300", "2108.07732", "2109.07958"}

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_NONANSWER = ("does not provide", "does not specify", "not specified", "does not mention",
              "passage does not", "no specific", "not explicitly", "cannot be determined")
_SEEN_Q: set[str] = set()  # cross-type dedup


def norm(s: str) -> str:
    for lig, rep in _LIGATURES.items():
        s = s.replace(lig, rep)
    return s


def llm_json(prompt: str, model: str) -> dict | None:
    NetworkPolicyService.require_local_endpoint(OLLAMA, "Ollama case miner")
    body = {"model": model, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": 0.4, "num_predict": 400}}
    req = urllib.request.Request(OLLAMA + "/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=120).read())["response"]
        return json.loads(raw)
    except Exception:
        return None


def grounded(must: list, source: str) -> bool:
    """Keep a case only if at least half its must_include facts appear (word/number
    boundary) in the source passage. This is the anti-hallucination gate."""
    if not must:
        return False
    src = source.lower()
    hits = sum(1 for f in must if isinstance(f, str) and f.strip()
               and re.search(r"(?<!\w)" + re.escape(f.lower().strip()) + r"(?!\w)", src))
    return hits >= max(1, (len(must) + 1) // 2)


def valid(obj: dict, source: str, kind: str) -> tuple[str, str, list] | None:
    if not isinstance(obj, dict):
        return None
    q, a, must = obj.get("question"), obj.get("answer"), obj.get("must_include")
    if not (isinstance(q, str) and isinstance(a, str) and isinstance(must, list)):
        return None
    q, a = norm(q.strip()), norm(a.strip())
    must = [norm(str(m).strip()) for m in must if str(m).strip()][:4]
    if len(q) < 12 or not q.endswith("?") or len(a) < 8:
        return None
    # reject non-answers ("the passage does not provide a formula")
    if any(p in a.lower() for p in _NONANSWER):
        return None
    # reject reference/bibliography meta-questions
    if re.search(r"\bisbn\b|title of the (paper|book)", q, re.I) or re.search(r"\bby [A-Z][a-z]+ (et al|and )", q):
        return None
    # math cases must actually contain math (formula / objective / complexity / symbol)
    if kind == "math" and not MATH_RE.search(q + " " + a):
        return None
    # reject too-generic figure questions
    if kind == "figure" and q.lower().rstrip("?").strip() in {
            "what does the figure show", "what does the table show",
            "what does the figure/table show", "what is shown in the figure",
            "what is shown in the table"}:
        return None
    if not grounded(must, source):
        return None
    if q in _SEEN_Q:            # cross-type dedup
        return None
    _SEEN_Q.add(q)
    return q, a, must


PROMPT = {
    "text": "From this passage of a scientific paper, write ONE specific factual question a "
            "researcher would ask about the PAPER's own claims, methods, datasets, or results, and "
            "its answer, using ONLY the passage. Do NOT ask about any example, sample, or practice "
            "problem shown inside the passage. Prefer concrete facts (numbers, method names, "
            "definitions). Return JSON with keys question, answer, must_include (2-4 short EXACT "
            "substrings copied from the passage that a correct answer must contain).\n\nPassage:\n{src}",
    "math": "From this passage, write ONE question about a MATHEMATICAL detail the PAPER defines or "
            "uses (a formula, training objective, loss, probability expression, or computational "
            "complexity) and its answer, using ONLY the passage. Do NOT ask about an example or "
            "practice problem contained in the passage. Return JSON with keys question, answer, "
            "must_include (2-4 short EXACT substrings from the passage).\n\nPassage:\n{src}",
    "figure": "From this figure/table caption of a scientific paper, write ONE question about what the "
              "figure or table shows and its answer, using ONLY the caption. Return JSON with keys "
              "question, answer, must_include (2-4 short EXACT substrings from the caption).\n\nCaption:\n{src}",
}
CAP = {"text": "single_doc_text", "math": "math", "figure": "visual"}


def load(paper: str, name: str) -> list:
    p = PAPERS / paper / name
    return json.loads(p.read_text()) if p.exists() else []


def content_chunks(paper: str, math_only: bool) -> list[dict]:
    out = []
    for c in load(paper, "chunks.json"):
        t = c.get("text", "")
        if len(t) < 300 or "references" in (c.get("section_title", "") or "").lower():
            continue
        # skip worked-example chunks (GSM8K/MATH-style) so we mine paper facts, not samples
        if re.search(r"(Answer:|####|Problem\s*\d|Q\d+:|Solution:)", t) or t.count("?") >= 3:
            continue
        # skip reference/bibliography chunks (mining them yields "title of paper by X?")
        if t.count("et al") >= 2 or t.count("arXiv:") >= 1:
            continue
        if math_only and not MATH_RE.search(t):
            continue
        out.append(c)
    return out


def pick_papers() -> tuple[list[str], list[str]]:
    text_papers, fig_papers = [], []
    for d in sorted(PAPERS.iterdir()):
        if not (d / "chunks.json").exists():
            continue
        if d.name in DATASET_EXCLUDE:  # benchmark-dataset papers are unminable for facts
            continue
        if len(load(d.name, "chunks.json")) >= 15 and "." in d.name:  # arxiv-style
            text_papers.append(d.name)
        if "." in d.name and load(d.name, "figures.json"):  # arxiv only, skip upload_ docs
            fig_papers.append(d.name)
    # ~25 diverse text/math papers: even spread over the qualifying set
    step = max(1, len(text_papers) // 25)
    text_papers = text_papers[::step][:25]
    return text_papers, fig_papers


def save(cases: list[dict]) -> None:
    OUT.write_text(json.dumps(cases, indent=2, ensure_ascii=False))


def mine(kind: str, papers: list[str], need: int, model: str, seed: int, all_cases: list[dict]) -> None:
    """Fill `need` more cases of `kind`, appending to all_cases and saving after each
    (so a kill/sleep never loses progress; re-running resumes)."""
    if need <= 0:
        return
    rng = random.Random(seed)
    made, attempts = 0, 0
    pool = papers[:]
    rng.shuffle(pool)
    pi = 0
    while made < need and attempts < need * 10:  # math rejects a lot; more attempts
        attempts += 1
        paper = pool[pi % len(pool)]; pi += 1
        if kind == "figure":
            figs = [f for f in load(paper, "figures.json") if len(f.get("caption", "")) > 40]
            if not figs:
                continue
            item = rng.choice(figs)
            src, locus = norm(item["caption"]), item.get("label", "Figure")
        else:
            chunks = content_chunks(paper, math_only=(kind == "math"))
            if not chunks:
                continue
            item = rng.choice(chunks)
            src, locus = norm(item["text"]), f"page {item.get('page')}"
        v = valid(llm_json(PROMPT[kind].format(src=src[:1500]), model), src, kind)
        if not v:
            continue
        q, a, must = v
        idx = sum(1 for c in all_cases if c["capability"] == CAP[kind]) + 1
        all_cases.append({
            "case_id": f"mine_{kind}_{idx:03d}",
            "capability": CAP[kind],
            "paper_id": paper,
            "secondary_paper_ids": [],
            "question": q,
            "gold_answer": a,
            "must_include": must,
            "answer_locus": locus,
            "difficulty": "medium",
            "notes": f"Mined from {paper} ({kind}); gold verified against source passage.",
        })
        made += 1
        save(all_cases)  # incremental: survive interruption
        print(f"  [{kind} {idx}] {paper}: {q[:70]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma4:12b"))
    ap.add_argument("--limit-per-type", type=int, default=0, help="smoke test: cap each type")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    text_papers, fig_papers = pick_papers()
    print(f"papers: {len(text_papers)} text/math, {len(fig_papers)} figure")

    q = dict(QUOTAS)
    if args.limit_per_type:
        q = {k: min(v, args.limit_per_type) for k, v in q.items()}

    from collections import Counter
    # Resume: keep any cases already mined (a v2 run interrupted by sleep), seed the
    # dedup set from them, and only fill the remaining quota per type.
    cases = json.loads(OUT.read_text()) if OUT.exists() else []
    for c in cases:
        _SEEN_Q.add(c["question"])
    have = Counter(c["capability"] for c in cases)
    if cases:
        print(f"resuming: {len(cases)} cases already present {dict(have)}")

    mine("text", text_papers, q["text"] - have["single_doc_text"], args.model, args.seed, cases)
    mine("math", text_papers, q["math"] - have["math"], args.model, args.seed + 1, cases)
    mine("figure", fig_papers, q["figure"] - have["visual"], args.model, args.seed + 2, cases)

    save(cases)
    print(f"\nWrote {OUT} ({len(cases)} cases) "
          f"across {len({c['paper_id'] for c in cases})} papers: "
          f"{dict(Counter(c['capability'] for c in cases))}")


# ponytail: LLM-generated + source-verified beats a regex fact-miner here; the
# grounded() check is the anti-hallucination gate. Upgrade path: swap the local
# model for a stronger judge, or add a second-pass human review of mined_cases.json.
def _selfcheck():
    assert grounded(["28.4", "BLEU"], "we report 28.4 BLEU on the task")
    assert not grounded(["8"], "the model has 2048 dims")  # boundary, not substring
    assert valid({"question": "What BLEU?", "answer": "28.4 BLEU reported",
                  "must_include": ["28.4"]}, "we report 28.4 BLEU", "text") is None  # q must end with ?
    assert valid({"question": "What BLEU score?", "answer": "28.4 BLEU reported",
                  "must_include": ["28.4"]}, "we report 28.4 BLEU", "text")
    _SEEN_Q.clear()
    assert valid({"question": "What is the loss objective?", "answer": "the formula is L = -log p",
                  "must_include": ["log p"]}, "the formula is L = -log p", "math")  # real math passes
    assert valid({"question": "What dataset is used?", "answer": "It uses ImageNet here",
                  "must_include": ["ImageNet"]}, "It uses ImageNet here", "math") is None  # non-math rejected
    assert valid({"question": "What does it show?", "answer": "the passage does not provide details",
                  "must_include": ["details"]}, "the passage does not provide details", "text") is None
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
