"""
compute_scores.py
Aggregates human-evaluation score exports into the metrics defined in
HUMAN_EVAL_DESIGN.md Section 9. Pure standard library (no numpy/scipy), so it runs
in any Python 3.11+ environment.

Inputs:
  - one or more evaluator exports (evaluation/human_eval/exports/*.json), each the
    file produced by the "Export scores" button in score_sheet.html
  - evaluation/human_eval/cases.json (capability + link to the faithfulness benchmark)
  - evaluation/results/faithfulness_eval_results_v3.json (automated NLI-CFS)

Outputs:
  - evaluation/human_eval/human_eval_results.json
  - evaluation/human_eval/human_eval_report.md

Run from repo root:  python3 evaluation/human_eval/compute_scores.py
Options:  --exports DIR (default evaluation/human_eval/exports)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = HERE / "cases.json"
FAITH = ROOT / "evaluation" / "results" / "faithfulness_eval_results_v3.json"
OUT_JSON = HERE / "human_eval_results.json"
OUT_MD = HERE / "human_eval_report.md"

DIMS = ["relevance", "coverage", "faithfulness", "usefulness"]


# ---- pure-stdlib statistics ----
def mean(xs): return st.fmean(xs) if xs else None
def stdev(xs): return st.pstdev(xs) if len(xs) > 1 else 0.0
def r3(x): return round(x, 3) if x is not None else None  # safe round (pearson etc. can return None)


def pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx, my = st.fmean(x), st.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return None
    return pearson(_ranks(x), _ranks(y))


def cohens_kappa(a, b):
    """Cohen's kappa for two lists of categorical labels (paired)."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    cats = sorted({c for p in pairs for c in p})
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for x, _ in pairs if x == c) / n
        pb = sum(1 for _, y in pairs if y == c) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _gammainc_p(a, x):
    """Regularized lower incomplete gamma P(a,x), for chi-square CDF."""
    if x <= 0:
        return 0.0
    if x < a + 1:  # series
        term = 1.0 / a
        s = term
        n = a
        for _ in range(200):
            n += 1
            term *= x / n
            s += term
            if abs(term) < abs(s) * 1e-12:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for Q, then P = 1 - Q
    tiny = 1e-30
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1 - q


def chi2_sf(x, df):
    """Survival function (upper tail p-value) of chi-square."""
    return max(0.0, 1.0 - _gammainc_p(df / 2.0, x / 2.0))


def friedman(matrix):
    """Friedman test. matrix: rows = subjects (questions), cols = treatments (models).
    Returns (Q statistic, df, p-value). Rows with any None are skipped."""
    rows = [r for r in matrix if all(v is not None for v in r)]
    n = len(rows)
    k = len(matrix[0]) if matrix else 0
    if n < 2 or k < 2:
        return None
    rank_sums = [0.0] * k
    tie_term = 0.0  # sum over all rows of sum(t^3 - t) for each group of t tied ranks
    for r in rows:
        rr = _ranks(list(r))
        for j in range(k):
            rank_sums[j] += rr[j]
        # count tie-group sizes in this row's ranks
        from collections import Counter
        for t in Counter(rr).values():
            if t > 1:
                tie_term += t ** 3 - t
    q = (12.0 / (n * k * (k + 1))) * sum(rj ** 2 for rj in rank_sums) - 3 * n * (k + 1)
    # Tie correction: with tied ranks (common in Likert data) the uncorrected Q is
    # downward-biased, inflating the p-value toward "no difference". Divide by the
    # correction factor. On untied data tie_term is 0 and the factor is 1.0.
    denom = n * (k ** 3 - k)
    correction = 1.0 - (tie_term / denom) if denom else 1.0
    if correction > 0:
        q = q / correction
    df = k - 1
    return {"Q": round(q, 4), "df": df, "p_value": round(chi2_sf(q, df), 5),
            "n_questions": n, "tie_corrected": tie_term > 0}


# ---- data loading ----
def load_exports(exports_dir: Path):
    files = sorted(exports_dir.glob("*.json")) if exports_dir.exists() else []
    evaluators = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        evaluators.append({"name": d.get("evaluator") or f.stem, "scores": d.get("scores", [])})
    return evaluators


def citation_stats(citations):
    labels = [c.get("label") for c in citations if c.get("label")]
    sup = labels.count("Supported")
    par = labels.count("Partial")
    uns = labels.count("Unsupported")
    return sup, par, uns


def count_missing(text):
    if not text or not text.strip():
        return 0
    parts = [p for p in text.replace(";", "\n").splitlines() if p.strip()]
    return len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default=str(HERE / "exports"))
    args = ap.parse_args()

    cases = {c["case_id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))}
    evaluators = load_exports(Path(args.exports))
    if not evaluators:
        raise SystemExit(f"No evaluator exports found in {args.exports}. "
                         "Place the exported *.json score files there.")

    # pool all (evaluator, case, model) rows
    rows = []
    for ev in evaluators:
        for s in ev["scores"]:
            s = dict(s)
            s["evaluator"] = ev["name"]
            rows.append(s)

    models = sorted({r["model"] for r in rows})
    caps = sorted({r["capability"] for r in rows})

    # 1. per-model mean Likert (overall + per capability)
    per_model = {}
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        entry = {"n_answers": len(mrows)}
        for d in DIMS:
            vals = [r[d] for r in mrows if r.get(d) is not None]
            entry[d] = {"mean": round(mean(vals), 3) if vals else None,
                        "stdev": round(stdev(vals), 3) if vals else None, "n": len(vals)}
        # citation metrics
        sup = par = uns = miss = ncit = 0
        for r in mrows:
            s, p, u = citation_stats(r.get("citations", []))
            sup += s; par += p; uns += u; ncit += (s + p + u)
            miss += count_missing(r.get("missing_citations", ""))
        # Precision: of the citations the model emitted, the fraction that hold up.
        #   strict  = Supported / all citations; lenient gives Partial half credit.
        # Recall: of the claims that should be cited, the fraction actually cited
        #   with support. A cited-worthy claim is either an emitted citation that is
        #   Supported or Partial (present and at least partly grounded) or a
        #   missing-citation claim (should have been cited but was not). Unsupported
        #   citations are NOT counted as recalled (a hallucinated citation is not a
        #   satisfied citation-worthy claim), which is the C4 fix.
        prec_strict = sup / ncit if ncit else None
        prec_len = (sup + 0.5 * par) / ncit if ncit else None
        cited_worthy_covered = sup + par
        recall = cited_worthy_covered / (cited_worthy_covered + miss) if (cited_worthy_covered + miss) else None
        # F1 over the two consistently-defined quantities (strict precision, recall).
        if prec_strict is not None and recall is not None:
            f1 = (2 * prec_strict * recall / (prec_strict + recall)) if (prec_strict + recall) else 0.0
        else:
            f1 = None
        entry["citations"] = {
            "supported": sup, "partial": par, "unsupported": uns, "missing": miss,
            "precision_strict": round(prec_strict, 3) if prec_strict is not None else None,
            "precision_lenient": round(prec_len, 3) if prec_len is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
        }
        # per-capability faithfulness mean
        entry["faithfulness_by_capability"] = {}
        for cap in caps:
            vals = [r["faithfulness"] for r in mrows if r["capability"] == cap and r.get("faithfulness") is not None]
            if vals:
                entry["faithfulness_by_capability"][cap] = round(mean(vals), 3)
        per_model[m] = entry

    # 2. ranking summary
    ranking = {m: {"first_place": 0, "ranks": []} for m in models}
    by_case_ev = defaultdict(list)
    for r in rows:
        by_case_ev[(r["evaluator"], r["case_id"])].append(r)
    for group in by_case_ev.values():
        for r in group:
            if r.get("rank") is not None:
                ranking[r["model"]]["ranks"].append(r["rank"])
                if r["rank"] == 1:
                    ranking[r["model"]]["first_place"] += 1
    ranking_summary = {m: {"first_place": v["first_place"],
                           "mean_rank": round(mean(v["ranks"]), 3) if v["ranks"] else None,
                           "n_ranked": len(v["ranks"])} for m, v in ranking.items()}

    # 3. Friedman across models (faithfulness), per question averaged over evaluators
    def matrix_for(field):
        mat = []
        case_ids = sorted({r["case_id"] for r in rows})
        for cid in case_ids:
            row = []
            for m in models:
                vals = [r[field] for r in rows if r["case_id"] == cid and r["model"] == m and r.get(field) is not None]
                row.append(mean(vals) if vals else None)
            mat.append(row)
        return mat
    friedman_faith = friedman(matrix_for("faithfulness"))

    # citation-support rate per (case, model) for Friedman
    def support_rate(r):
        s, p, u = citation_stats(r.get("citations", []))
        tot = s + p + u
        return (s / tot) if tot else None
    sr_matrix = []
    for cid in sorted({r["case_id"] for r in rows}):
        row = []
        for m in models:
            vals = [support_rate(r) for r in rows if r["case_id"] == cid and r["model"] == m]
            vals = [v for v in vals if v is not None]
            row.append(mean(vals) if vals else None)
        sr_matrix.append(row)
    friedman_support = friedman(sr_matrix)

    # 4. inter-annotator agreement (if >=2 evaluators)
    agreement = None
    if len(evaluators) >= 2:
        a, b = evaluators[0], evaluators[1]
        aidx = {(s["case_id"], s["model"]): s for s in a["scores"]}
        bidx = {(s["case_id"], s["model"]): s for s in b["scores"]}
        common = sorted(set(aidx) & set(bidx))
        faith_a = [aidx[k]["faithfulness"] for k in common if aidx[k].get("faithfulness") is not None and bidx[k].get("faithfulness") is not None]
        faith_b = [bidx[k]["faithfulness"] for k in common if aidx[k].get("faithfulness") is not None and bidx[k].get("faithfulness") is not None]
        cit_a, cit_b = [], []
        for k in common:
            ca = {c["ref_id"]: c["label"] for c in aidx[k].get("citations", [])}
            cb = {c["ref_id"]: c["label"] for c in bidx[k].get("citations", [])}
            for ref in set(ca) & set(cb):
                cit_a.append(ca[ref]); cit_b.append(cb[ref])
        agreement = {
            "overlap_answers": len(common),
            "faithfulness_pearson": r3(pearson(faith_a, faith_b)) if len(faith_a) >= 2 else None,
            "citation_label_kappa": r3(cohens_kappa(cit_a, cit_b)) if cit_a else None,
            "evaluators": [a["name"], b["name"]],
        }

    # 5. human faithfulness vs automated NLI-CFS (single-doc cases linked to the benchmark)
    validation = None
    if FAITH.exists():
        faith = json.loads(FAITH.read_text(encoding="utf-8"))
        auto = {}
        for sys_key in ("hybrid", "bm25"):
            for rr in faith.get(sys_key, {}).get("results", []):
                auto.setdefault(rr["case_id"], {})[sys_key] = rr
        hx, ax_cfs, ax_nli = [], [], []
        for cid, case in cases.items():
            fid = case.get("source_faithfulness_id")
            if not fid or fid not in auto or "hybrid" not in auto[fid]:
                continue
            hvals = [r["faithfulness"] for r in rows if r["case_id"] == cid and r.get("faithfulness") is not None]
            if not hvals:
                continue
            hx.append(mean(hvals))
            ax_cfs.append(auto[fid]["hybrid"]["combined_cfs"])
            ax_nli.append(auto[fid]["hybrid"]["nli_cfs"])
        if len(hx) >= 3:
            validation = {
                "n_cases": len(hx),
                "human_faithfulness_vs_combined_cfs": {
                    "pearson": r3(pearson(hx, ax_cfs)), "spearman": r3(spearman(hx, ax_cfs))},
                "human_faithfulness_vs_nli_cfs": {
                    "pearson": r3(pearson(hx, ax_nli)), "spearman": r3(spearman(hx, ax_nli))},
            }

    results = {
        "n_evaluators": len(evaluators),
        "n_scored_answers": len(rows),
        "models": models,
        "per_model": per_model,
        "ranking": ranking_summary,
        "friedman_faithfulness": friedman_faith,
        "friedman_citation_support": friedman_support,
        "inter_annotator_agreement": agreement,
        "validation_vs_automated_nli_cfs": validation,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(results)
    print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")


def write_report(r):
    L = ["# ScholAR Human Evaluation: Results", "",
         f"Evaluators: {r['n_evaluators']}. Scored answers: {r['n_scored_answers']}. "
         f"Models: {', '.join(r['models'])}.", "",
         "## Per-model answer quality (mean 1-5) and citation quality", "",
         "| Model | Relevance | Coverage | Faithfulness | Usefulness | Cite prec | Cite recall | Cite F1 | Sup/Par/Uns |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m in r["models"]:
        e = r["per_model"][m]
        c = e["citations"]
        L.append(f"| {m} | {e['relevance']['mean']} | {e['coverage']['mean']} | "
                 f"{e['faithfulness']['mean']} | {e['usefulness']['mean']} | "
                 f"{c['precision_strict']} | {c['recall']} | {c['f1']} | "
                 f"{c['supported']}/{c['partial']}/{c['unsupported']} |")
    L += ["", "## Ranking (1 = best)", "", "| Model | First-place | Mean rank | N |", "|---|---:|---:|---:|"]
    for m in r["models"]:
        rk = r["ranking"][m]
        L.append(f"| {m} | {rk['first_place']} | {rk['mean_rank']} | {rk['n_ranked']} |")
    L += ["", "## Model-agnostic tests (Friedman across models, same questions)", ""]
    ff = r["friedman_faithfulness"]; fs = r["friedman_citation_support"]
    if ff:
        L.append(f"- Faithfulness: Q={ff['Q']}, df={ff['df']}, p={ff['p_value']} over {ff['n_questions']} questions. "
                 f"{'No significant difference across models (supports model-agnostic grounding).' if ff['p_value'] > 0.05 else 'Significant difference across models (report which model differs).'}")
    if fs:
        L.append(f"- Citation-support rate: Q={fs['Q']}, df={fs['df']}, p={fs['p_value']} over {fs['n_questions']} questions.")
    if r["inter_annotator_agreement"]:
        a = r["inter_annotator_agreement"]
        L += ["", "## Inter-annotator agreement", "",
              f"- Overlap answers: {a['overlap_answers']} ({', '.join(a['evaluators'])})",
              f"- Faithfulness Pearson: {a['faithfulness_pearson']}",
              f"- Citation-label Cohen's kappa: {a['citation_label_kappa']}"]
    if r["validation_vs_automated_nli_cfs"]:
        v = r["validation_vs_automated_nli_cfs"]
        L += ["", "## Validation: human faithfulness vs automated NLI-CFS", "",
              f"On {v['n_cases']} single-document cases linked to the automated faithfulness benchmark:",
              f"- Human faithfulness vs Combined CFS: Pearson {v['human_faithfulness_vs_combined_cfs']['pearson']}, "
              f"Spearman {v['human_faithfulness_vs_combined_cfs']['spearman']}",
              f"- Human faithfulness vs NLI-CFS: Pearson {v['human_faithfulness_vs_nli_cfs']['pearson']}, "
              f"Spearman {v['human_faithfulness_vs_nli_cfs']['spearman']}",
              "", "A strong positive correlation indicates the automated metric tracks expert judgment."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
