"""
rescore_faithfulness_judge.py
Re-score the already-generated answers with the REAL entailment judge (llm_entailment.py)
instead of the cosine proxy, addressing the reviewers' decisive objection. Uses the 350 stored
answers in human_eval/answers.json (4 local models x 100 diverse cases across 25 papers), so it
(a) needs no regeneration and no model thrashing, and (b) reports faithfulness on the DIVERSE set
rather than the 3 famous papers the old headline used.

For each answer: decompose its prose into atoms and judge each against the cited evidence chunks
(entail/neutral/CONTRADICT), and grade each inline citation Supported/Partial/Unsupported. The
judge (default qwen3.5:9b) can fire contradiction; cosine could not.

Resumable per model. Run (Ollama up, no backend needed):
    python3 evaluation/rescore_faithfulness_judge.py --model gemma4:12b
    ... qwen3.5:9b / llama3.1:8b / mistral:7b   (or --model all)
Writes: evaluation/results/faithfulness_judged.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["FAITH_JUDGE"] = "llm"  # force the LLM-judge path in the shared scorer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import nli_faithfulness as nli  # noqa: E402  (honours FAITH_JUDGE=llm set above)

ANS = ROOT / "evaluation" / "human_eval" / "answers.json"
DATA = ROOT / "backend" / "data" / "papers"
OUT = ROOT / "evaluation" / "results" / "faithfulness_judged.json"

_SCORER = nli.NLIFaithfulnessScorer()


def answer_prose(answer: str) -> str:
    m = re.search(r"\*\*\s*Answer\s*\*\*(.*?)(?:\*\*\s*Evidence\s*\*\*|\Z)", answer,
                  re.IGNORECASE | re.DOTALL)
    prose = m.group(1) if m else answer
    return re.sub(r"\[\d+\]", "", prose).strip()


def sentences_with_citations(answer: str) -> list[tuple[str, list[int]]]:
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", answer.replace("\n", " ")):
        refs = [int(n) for n in re.findall(r"\[(\d+)\]", sent)]
        if refs:
            out.append((re.sub(r"\[\d+\]", "", sent).strip(), refs))
    return out


def chunk_text_of(paper_id: str, chunk_id: str, cache: dict) -> str:
    key = paper_id
    if key not in cache:
        cache[key] = {c["chunk_id"]: c.get("text", "")
                      for c in json.loads((DATA / paper_id / "chunks.json").read_text())}
    return cache[key].get(chunk_id, "")


def score_answer(rec: dict, cache: dict) -> dict | None:
    cits = rec.get("citations") or []
    if rec.get("error") or not cits:
        return None
    cited_texts = [chunk_text_of(rec["paper_id"], c.get("chunk_id"), cache) or c.get("quote", "")
                   for c in cits]
    cited_texts = [t for t in cited_texts if t]
    prose = answer_prose(rec["answer"])
    gen = _SCORER.score_full(prose, [{"text": t} for t in cited_texts], top_k=len(cited_texts) or 1)
    n_atoms = max(gen["n_atoms"], 1)

    by_ref = {c.get("ref_id"): c for c in cits}
    labels = []
    for sent, refs in sentences_with_citations(rec["answer"]):
        for ref in refs:
            c = by_ref.get(ref)
            if not c:
                continue
            ct = chunk_text_of(rec["paper_id"], c.get("chunk_id"), cache) or c.get("quote", "")
            if not ct:
                continue
            best = _SCORER.score_claim_vs_chunks(sent, [ct], top_k=1)
            labels.append({"ENTAILMENT": "Supported", "NEUTRAL": "Partial",
                           "CONTRADICTION": "Unsupported"}[best["label"]])
    return {"case_id": rec["case_id"], "model": rec["model"],
            "gen_faithfulness": gen["cfs"], "n_atoms": n_atoms,
            "n_contradicted": gen["n_contradicted"],
            "hallucination_rate": round(gen["n_contradicted"] / n_atoms, 3),
            "cit_supported": labels.count("Supported"),
            "cit_partial": labels.count("Partial"),
            "cit_unsupported": labels.count("Unsupported")}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    sup = sum(r["cit_supported"] for r in rows)
    par = sum(r["cit_partial"] for r in rows)
    uns = sum(r["cit_unsupported"] for r in rows)
    tot = sup + par + uns
    return {"n_answers": n,
            "mean_gen_faithfulness": round(sum(r["gen_faithfulness"] for r in rows) / n, 3),
            "mean_hallucination_rate": round(sum(r["hallucination_rate"] for r in rows) / n, 3),
            "answers_with_a_contradiction": sum(1 for r in rows if r["n_contradicted"] > 0),
            "citations_checked": tot,
            "citation_support": {"supported": sup, "partial": par, "unsupported": uns,
                                 "supported_pct": round(sup / tot, 3) if tot else None}}


CHECKPOINT_EVERY = 10  # write after this many freshly-scored answers (survives kills)


def _save(blob: dict) -> None:
    blob["generated_at"] = datetime.now().isoformat(timespec="seconds")
    for m, rows in blob["raw"].items():
        if rows:
            blob["summary"][m] = summarize(rows)
    OUT.write_text(json.dumps(blob, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    answers = json.loads(ANS.read_text())
    models = sorted({a["model"] for a in answers}) if args.model == "all" else [args.model]

    blob = {"judge": os.getenv("FAITH_JUDGE_MODEL", "qwen3.5:9b"), "summary": {}, "raw": {}}
    if OUT.exists():
        blob = json.loads(OUT.read_text())
    blob.setdefault("raw", {})
    blob.setdefault("summary", {})
    done = {(m, r["case_id"]) for m, rows in blob["raw"].items() for r in rows}
    print(f"resume: {len(done)} answers already scored")

    cache: dict = {}
    fresh = 0
    todo = [a for a in answers if a["model"] in models and (a["model"], a["case_id"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to score this run: {len(todo)}")
    for j, rec in enumerate(todo, 1):
        r = score_answer(rec, cache)
        if r:
            blob["raw"].setdefault(rec["model"], []).append(r)
            fresh += 1
            if fresh % CHECKPOINT_EVERY == 0:
                _save(blob)
                print(f"  checkpoint {j}/{len(todo)} (saved {sum(len(v) for v in blob['raw'].values())} total)")
    _save(blob)
    for m, s in blob["summary"].items():
        c = s["citation_support"]
        print(f"== {m}: faith {s['mean_gen_faithfulness']}  halluc {s['mean_hallucination_rate']}  "
              f"contra-answers {s['answers_with_a_contradiction']}/{s['n_answers']}  "
              f"cites {c['supported']}/{c['partial']}/{c['unsupported']} (sup {c['supported_pct']})")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(todo)-0} attempted, {fresh} scored this run)")


if __name__ == "__main__":
    main()
