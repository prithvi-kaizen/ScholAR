"""
build_m3sciqa.py
Assemble M3SciQA's *locality* task into a case file we can evaluate against.

Why this exists: M3SciQA (Li et al., 2024) poses exactly ScholAR's multi-document localization
task -- given an anchor paper and a question about one of its figures, find the reference paper
in the anchor's bibliography that answers it -- and publishes a baseline table (their Table 2,
MRR) that we can be measured against. It is the external baseline our multi-doc result lacked.

Three joins have to be done carefully (each was verified against the raw data):
  1. Candidate references carry only {title, abstract}, no arXiv id, while the gold label is an
     arXiv id. So we resolve gold arXiv id -> title via the arXiv API and match on a normalized
     title to recover the gold's index in the candidate list.
  2. `retrieval_paper.json` is keyed by *rephrased* questions (only 6/296 join with
     locality.jsonl), so we join on `anchor_id` and use the full cluster bibliography instead.
  3. Their reported random floor (0.126) does not match our pool size, so we do NOT assume it:
     run_m3sciqa_eval.py computes the random floor empirically on the exact pool it ranks.

Prereq: the dataset clone (done once, ~74MB, gitignored):
    git clone --depth 1 https://github.com/yale-nlp/M3SciQA.git evaluation/m3sciqa/data/M3SciQA

Run from repo root:  python3 evaluation/m3sciqa/build_m3sciqa.py [--selfcheck]
Writes: evaluation/m3sciqa/m3sciqa_cases.json
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "M3SciQA"
LOCALITY = SRC / "data" / "locality.jsonl"
CLUSTERS = SRC / "paper_cluster_S2_content.json"
FIGDIR = SRC / "data" / "locality"
OUT = HERE / "m3sciqa_cases.json"
TITLE_CACHE = HERE / "arxiv_titles.json"

ARXIV_API = "https://export.arxiv.org/api/query?id_list={}&max_results=100"
ATOM = "{http://www.w3.org/2005/Atom}"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", norm(s)).strip()


def fetch_titles(arxiv_ids: list[str]) -> dict[str, str]:
    """arXiv id -> title, batched, cached on disk (the API is the only network call here)."""
    cache = json.loads(TITLE_CACHE.read_text()) if TITLE_CACHE.exists() else {}
    missing = [i for i in arxiv_ids if i not in cache]
    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        url = ARXIV_API.format(",".join(batch))
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for entry in root.findall(f"{ATOM}entry"):
            idu = entry.findtext(f"{ATOM}id") or ""
            m = re.search(r"abs/([^v]+)", idu)
            title = squash(entry.findtext(f"{ATOM}title") or "")
            if m and title:
                cache[m.group(1)] = title
        print(f"  fetched titles {i + len(batch)}/{len(missing)}")
        time.sleep(3)  # arXiv asks for a 3s gap between requests
    TITLE_CACHE.write_text(json.dumps(cache, indent=1))
    return cache


def match_index(gold_title: str, candidates: list[dict]) -> int | None:
    """Index of the gold paper in the candidate list, by normalized title (exact, then fuzzy)."""
    g = squash(gold_title)
    titles = [squash(c.get("title", "")) for c in candidates]
    if g in titles:
        return titles.index(g)
    best, best_i = 0.0, None
    for i, t in enumerate(titles):
        r = SequenceMatcher(None, g, t).ratio()
        if r > best:
            best, best_i = r, i
    return best_i if best >= 0.90 else None


def build() -> tuple[list[dict], dict]:
    cases = [json.loads(l) for l in LOCALITY.read_text().splitlines() if l.strip()]
    clusters = json.loads(CLUSTERS.read_text())

    titles = fetch_titles(sorted({c["reference_id"] for c in cases}))

    out, unresolved, no_fig = [], 0, 0
    for n, c in enumerate(cases):
        anchor, gold = c["anchor_id"], c["reference_id"]
        cands = clusters.get(anchor) or []
        gold_title = titles.get(gold)
        idx = match_index(gold_title, cands) if (gold_title and cands) else None
        if idx is None:
            unresolved += 1
            continue
        fig = FIGDIR / anchor / Path(c["evidence_anchor"]).name
        if not fig.exists():
            no_fig += 1
        out.append({
            "case_id": f"m3_{n:03d}",
            "anchor_id": anchor,
            "question": c["question_anchor"],
            "modal": c.get("modal"),
            "figure_path": str(fig.relative_to(ROOT)) if fig.exists() else None,
            "gold_arxiv": gold,
            "gold_title": gold_title,
            "gold_index": idx,
            "candidates": [{"title": x.get("title", ""), "abstract": x.get("abstract", "")}
                           for x in cands],
        })
    stats = {"total": len(cases), "built": len(out), "unresolved_gold": unresolved,
             "missing_figure": no_fig,
             "pool_mean": round(sum(len(o["candidates"]) for o in out) / max(len(out), 1), 1)}
    return out, stats


def _selfcheck(cases: list[dict]) -> None:
    assert cases, "no cases built"
    for c in cases:
        cands = c["candidates"]
        i = c["gold_index"]
        assert 0 <= i < len(cands), f"{c['case_id']}: gold_index out of range"
        # the gold index must actually point at the gold paper
        assert SequenceMatcher(None, squash(c["gold_title"]),
                               squash(cands[i]["title"])).ratio() >= 0.90, \
            f"{c['case_id']}: gold_index does not point at the gold title"
    print("selfcheck OK")


if __name__ == "__main__":
    if not LOCALITY.exists():
        sys.exit(f"missing dataset. Run:\n  git clone --depth 1 "
                 f"https://github.com/yale-nlp/M3SciQA.git {SRC.relative_to(ROOT)}")
    cases, stats = build()
    _selfcheck(cases)
    if "--selfcheck" in sys.argv:
        sys.exit()
    OUT.write_text(json.dumps(cases, indent=2))
    print(f"\nbuilt {stats['built']}/{stats['total']} locality cases")
    print(f"  unresolved gold (dropped, disclosed): {stats['unresolved_gold']}")
    print(f"  cases missing a figure image        : {stats['missing_figure']}")
    print(f"  mean candidate pool                 : {stats['pool_mean']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
