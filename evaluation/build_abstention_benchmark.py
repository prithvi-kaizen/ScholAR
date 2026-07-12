"""
build_abstention_benchmark.py
Construct provably-unanswerable questions to test whether ScholAR abstains (declines) instead
of fabricating when the answer is NOT in the document. This is the acid test for the paper's
grounding claim, and no such negatives exist in the current benchmarks (every case is answerable).

Construction (cross-paper): each of the 100 diverse cases carries a paper-specific question plus
`must_include` facts mined verbatim from that paper. We pair a question with a DIFFERENT target
paper and keep it only if BOTH hold:
  (a) NONE of the case's must_include substrings appears (normalized) anywhere in the target
      paper's chunks -> the specific fact is provably absent from the target, and
  (b) the question itself contains at least one must_include substring -> the question names the
      absent entity, so it is unambiguously about content the target paper does not have (guards
      against the question being answerable by some different-but-valid answer in the target).
The correct behavior for a grounded system is to abstain.

Caveat (disclosed in the paper): these negatives are auto-constructed from mined facts and are
synthetic; the absent fact is provably not in the target paper, but the set is small.

Emits: evaluation/abstention_cases.json
Run from repo root:  python3 evaluation/build_abstention_benchmark.py [--selfcheck]
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "backend" / "data" / "papers"
CASES = PROJECT_ROOT / "evaluation" / "human_eval" / "cases.json"
OUT = PROJECT_ROOT / "evaluation" / "abstention_cases.json"

N_TARGET = 20       # negatives to emit
MAX_PER_PAPER = 2   # cap negatives sharing a target paper, to spread coverage


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def paper_text(paper_id: str) -> str:
    chunks = json.loads((DATA_DIR / paper_id / "chunks.json").read_text())
    return _norm(" ".join(c.get("text", "") for c in chunks))


def question_facts(question: str, must: list[str]) -> list[str]:
    """must_include substrings that appear in the question itself (the named absent entities)."""
    q = _norm(question)
    return [m for m in must if _norm(m) in q]


def absent_in(must: list[str], target_txt: str) -> bool:
    """True iff none of the must substrings occur in the target paper."""
    return not any(_norm(m) in target_txt for m in must)


def build() -> tuple[list[dict], dict]:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    papers = sorted({c["paper_id"] for c in cases})
    text_of = {p: paper_text(p) for p in papers}

    rng = random.Random(20260712)  # deterministic
    order = cases[:]
    rng.shuffle(order)

    out: list[dict] = []
    per_paper: dict[str, int] = {}
    considered = 0
    for c in order:
        if len(out) >= N_TARGET:
            break
        must = c.get("must_include") or []
        named = question_facts(c["question"], must)
        if not named:                      # guard (b): question must name an absent entity
            continue
        considered += 1
        # candidate targets: any other paper where the fact is provably absent, honoring the cap
        cands = [p for p in papers
                 if p != c["paper_id"]
                 and per_paper.get(p, 0) < MAX_PER_PAPER
                 and absent_in(must, text_of[p])]
        if not cands:
            continue
        target = rng.choice(cands)
        per_paper[target] = per_paper.get(target, 0) + 1
        out.append({
            "id": f"abst_{len(out)+1:02d}",
            "source_case_id": c["case_id"],
            "source_paper": c["paper_id"],
            "target_paper": target,
            "capability": c.get("capability"),
            "question": c["question"],
            "absent_facts": named,
        })
    stats = {"emitted": len(out), "sources_with_named_fact": considered, "total_sources": len(cases)}
    return out, stats


def _verify(cases: list[dict]) -> None:
    """Re-prove absence + naming for every emitted case (independent of build ordering)."""
    text_cache: dict[str, str] = {}
    for c in cases:
        tp = c["target_paper"]
        txt = text_cache.setdefault(tp, paper_text(tp))
        for f in c["absent_facts"]:
            assert _norm(f) not in txt, f"leak: {f!r} present in {tp} ({c['id']})"
            assert _norm(f) in _norm(c["question"]), f"unnamed fact {f!r} ({c['id']})"


def _selfcheck() -> None:
    assert question_facts("what is BM25 k1?", ["BM25 k1", "b of 0.72"]) == ["BM25 k1"]
    assert absent_in(["zzz-not-here"], "some paper text about transformers")
    assert not absent_in(["transformers"], "some paper text about transformers")
    print("unit selfcheck OK")
    cases, stats = build()
    assert cases, "no negatives built"
    _verify(cases)
    assert 1 <= max(_pp(cases).values()) <= MAX_PER_PAPER
    print(f"build selfcheck OK: {stats}")


def _pp(cases: list[dict]) -> dict:
    d: dict[str, int] = {}
    for c in cases:
        d[c["target_paper"]] = d.get(c["target_paper"], 0) + 1
    return d


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        sys.exit()
    cases, stats = build()
    _verify(cases)
    OUT.write_text(json.dumps(cases, indent=2, ensure_ascii=False))
    caps = {}
    for c in cases:
        caps[c["capability"]] = caps.get(c["capability"], 0) + 1
    print(f"emitted {stats['emitted']} negatives "
          f"(from {stats['sources_with_named_fact']}/{stats['total_sources']} usable sources)")
    print(f"capabilities: {caps}")
    print(f"target papers: {len(_pp(cases))} distinct")
    print(f"wrote {OUT.name}")
