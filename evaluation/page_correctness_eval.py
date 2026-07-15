"""
page_correctness_eval.py
Measure the paper's MOTIVATING failure mode, which reviewers correctly noted is never measured:
"chat systems invent page references and attribute quotes to places they do not appear."

The comparison's `invalid_page_rate` only checks whether a cited page number is IN RANGE, not
whether the cited page actually contains the claim. This checks the latter: for every cited
(sentence, page) pair in the stored comparison answers, does that page's text support the sentence?
The local LLM entailment judge decides. ScholAR's citations resolve to the evidence chunk's own
page, so its page-correctness is high by construction; the freeform baselines are the real test.

Reuses stored answers in comparison_results.json (no re-generation). Judge = llm_entailment.

Run:  FAITH_JUDGE not needed; judge is called directly.
      python3 evaluation/page_correctness_eval.py [--systems pdfchat,scholar] [--limit 40]
Writes: evaluation/results/page_correctness_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from llm_entailment import classify, JUDGE_MODEL  # noqa: E402

RES = ROOT / "evaluation" / "results"
DATA = ROOT / "backend" / "data" / "papers"
OUT = RES / "page_correctness_results.json"


def page_text_map(paper_id: str) -> dict[int, str]:
    pages = json.loads((DATA / paper_id / "pages.json").read_text())
    out = {}
    for p in pages:
        n = p.get("page")
        if isinstance(n, int):
            out[n] = p.get("text", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="pdfchat,scholar")
    ap.add_argument("--limit", type=int, default=0, help="cases per system (0 = all)")
    args = ap.parse_args()
    systems = [s.strip() for s in args.systems.split(",")]

    rows = json.loads((RES / "comparison_results.json").read_text())["rows"]
    by_sys = defaultdict(list)
    for r in rows.values():
        if r["system"] in systems:
            by_sys[r["system"]].append(r)

    result = {"judge": JUDGE_MODEL, "systems": {}}
    pcache: dict[str, dict[int, str]] = {}
    for sysname in systems:
        recs = by_sys[sysname]
        if args.limit:
            recs = recs[:args.limit]
        total = supported = 0
        detail = []
        for i, r in enumerate(recs, 1):
            pm = pcache.setdefault(r["paper_id"], page_text_map(r["paper_id"]))
            for sent, pages in (r.get("cited") or []):
                for p in pages:
                    ptext = pm.get(int(p)) if p is not None else None
                    if not ptext:
                        continue  # out-of-range pages are already counted 0 in invalid_page_rate
                    label, _ = classify(sent, ptext)
                    ok = label == "ENTAILMENT"
                    total += 1
                    supported += int(ok)
                    detail.append({"case": r["case_id"], "page": p, "label": label})
            if i % 20 == 0:
                print(f"  [{sysname}] {i}/{len(recs)}  running page-correctness {supported}/{total}")
        rate = round(supported / total, 3) if total else None
        result["systems"][sysname] = {"cited_pages_checked": total,
                                      "page_supported": supported, "page_correctness": rate}
        print(f"== {sysname}: page-correctness {rate}  ({supported}/{total} cited pages support the claim)")

    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("Note: 'invalid_page_rate' in the comparison is 0 for all systems (pages in range); "
          "this measures whether the page actually SUPPORTS the cited claim, the real failure mode.")


if __name__ == "__main__":
    main()
