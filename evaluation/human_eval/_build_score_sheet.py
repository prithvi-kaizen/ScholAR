"""
_build_score_sheet.py
Builds a self-contained score_sheet.html from cases.json and answers.json.
Data is embedded at build time (no external requests, opens offline via file://).
Per question, the model answers are shuffled and blinded (Answer A, B, C, D); the
real model is retained in the embedded data for un-blinded export only.

Run from repo root:  python3 evaluation/human_eval/_build_score_sheet.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.json"
ANSWERS = HERE / "answers.json"
OUT = HERE / "score_sheet.html"

SEED = 20260706  # fixed so the blinded ordering is reproducible across rebuilds


def build_payload() -> list[dict]:
    cases = {c["case_id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))}
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    by_case: dict[str, list[dict]] = {}
    for a in answers:
        by_case.setdefault(a["case_id"], []).append(a)

    rng = random.Random(SEED)
    payload = []
    for case_id in cases:
        alist = by_case.get(case_id, [])
        if not alist:
            continue
        order = alist[:]
        rng.shuffle(order)
        case = cases[case_id]
        blinded = []
        for i, a in enumerate(order):
            blinded.append({
                "label": chr(ord("A") + i),
                "model": a["model"],  # hidden in UI, used only on export
                "answer": a["answer"],
                "error": a.get("error"),
                "citations": [
                    {"ref_id": c.get("ref_id"), "page": c.get("page"),
                     "section_title": c.get("section_title"), "quote": c.get("quote")}
                    for c in a.get("citations", [])
                ],
            })
        payload.append({
            "case_id": case_id,
            "capability": case["capability"],
            "question": case["question"],
            "gold_answer": case["gold_answer"],
            "must_include": case["must_include"],
            "answer_locus": case["answer_locus"],
            "answers": blinded,
        })
    return payload


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScholAR Human Evaluation</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; line-height: 1.5;
         background: #f6f7f9; color: #14171a; }
  header { position: sticky; top: 0; background: #14171a; color: #fff; padding: 12px 20px;
           display: flex; align-items: center; gap: 16px; flex-wrap: wrap; z-index: 5; }
  header h1 { font-size: 16px; margin: 0; }
  header .sp { flex: 1; }
  header input { padding: 6px 8px; border-radius: 6px; border: 1px solid #444; background: #222; color: #fff; }
  button { padding: 8px 14px; border-radius: 8px; border: 0; background: #2d7d46; color: #fff;
           font-weight: 600; cursor: pointer; }
  button.sec { background: #555; }
  main { max-width: 960px; margin: 0 auto; padding: 20px; }
  .case { background: #fff; border: 1px solid #e2e5e9; border-radius: 12px; padding: 18px; margin-bottom: 22px; }
  .cap { display: inline-block; font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
         background: #eef; color: #334; border-radius: 999px; padding: 2px 10px; margin-bottom: 8px; }
  .q { font-size: 18px; font-weight: 700; margin: 4px 0 10px; }
  .ref { background: #f3f8f4; border-left: 3px solid #2d7d46; padding: 8px 12px; border-radius: 6px;
         font-size: 14px; margin-bottom: 6px; }
  .must li { font-size: 13px; }
  .answer { border: 1px solid #dfe3e8; border-radius: 10px; padding: 14px; margin-top: 14px; }
  .answer h3 { margin: 0 0 8px; font-size: 15px; }
  .atext { white-space: pre-wrap; font-size: 14px; background: #fafbfc; padding: 10px; border-radius: 6px; }
  .cit { font-size: 13px; border-top: 1px dashed #dfe3e8; padding: 8px 0; display: grid;
         grid-template-columns: 1fr auto; gap: 8px; align-items: start; }
  .cit .quote { color: #333; }
  .cit .pg { color: #777; font-size: 12px; }
  .likert { display: flex; gap: 14px; flex-wrap: wrap; margin: 10px 0; }
  .likert label { font-size: 13px; }
  select, textarea { font: inherit; padding: 4px 6px; border-radius: 6px; border: 1px solid #ccc; }
  textarea { width: 100%; min-height: 40px; margin-top: 4px; }
  .rankrow { margin-top: 8px; font-size: 13px; }
  .status { font-size: 13px; color: #cfc; }
  @media (prefers-color-scheme: dark) {
    body { background: #0e1113; color: #e8eaed; }
    .case, .answer { background: #1a1e21; border-color: #2a2f34; }
    .atext { background: #14171a; } .ref { background: #14231a; }
    .cap { background: #223; color: #cde; } select, textarea { background: #14171a; color: #e8eaed; border-color: #333; }
  }
</style>
</head>
<body>
<header>
  <h1>ScholAR Human Evaluation</h1>
  <label style="font-size:13px">Evaluator: <input id="evaluator" placeholder="your name"></label>
  <span class="sp"></span>
  <span class="status" id="status"></span>
  <button class="sec" onclick="clearAll()">Reset</button>
  <button onclick="exportJSON()">Export scores</button>
</header>
<main id="root"></main>
<script>
const DATA = __DATA__;
const KEY = "scholar_human_eval_v1";
const DIMS = [["relevance","Q1 Relevance"],["coverage","Q2 Coverage"],
              ["faithfulness","Q3 Faithfulness"],["usefulness","Q4 Usefulness"]];
let state = JSON.parse(localStorage.getItem(KEY) || "{}");

function save() { localStorage.setItem(KEY, JSON.stringify(state)); flash(); }
function flash() { const s=document.getElementById("status"); s.textContent="saved "+new Date().toLocaleTimeString(); }
function k(caseId, label) { return caseId+"::"+label; }

function setVal(caseId, label, field, val) {
  const kk=k(caseId,label); state[kk]=state[kk]||{}; state[kk][field]=val; save();
}
function setCit(caseId, label, ref, val) {
  const kk=k(caseId,label); state[kk]=state[kk]||{}; state[kk].citations=state[kk].citations||{};
  state[kk].citations[ref]=val; save();
}

function render() {
  document.getElementById("evaluator").value = state.__evaluator || "";
  const root = document.getElementById("root");
  root.innerHTML = "";
  DATA.forEach((c, idx) => {
    const div = document.createElement("div"); div.className="case";
    let html = `<span class="cap">${c.capability}</span>`
      + `<div class="q">${idx+1}. ${esc(c.question)}</div>`
      + `<div class="ref"><b>Reference:</b> ${esc(c.gold_answer)}</div>`
      + `<div class="ref"><b>Must include:</b> ${c.must_include.map(esc).join("; ")} `
      + `<span class="pg">(${esc(c.answer_locus)})</span></div>`;
    c.answers.forEach(a => {
      const kk = k(c.case_id, a.label); const st = state[kk]||{};
      html += `<div class="answer"><h3>Answer ${a.label}</h3>`;
      html += `<div class="atext">${esc(a.answer || "(no answer / error)")}</div>`;
      if (a.citations.length) {
        html += `<div style="margin-top:8px"><b style="font-size:13px">Citations</b>`;
        a.citations.forEach(ct => {
          const cur = (st.citations||{})[ct.ref_id] || "";
          html += `<div class="cit"><div><span class="pg">[${ct.ref_id}] p.${ct.page} `
            + `${esc(ct.section_title||"")}</span><br><span class="quote">${esc(ct.quote||"")}</span></div>`
            + `<select onchange="setCit('${c.case_id}','${a.label}',${ct.ref_id},this.value)">`
            + opt(["","Supported","Partial","Unsupported"], cur) + `</select></div>`;
        });
        html += `</div>`;
      }
      html += `<div class="likert">`;
      DIMS.forEach(([f,lab]) => {
        html += `<label>${lab}: <select onchange="setVal('${c.case_id}','${a.label}','${f}',this.value)">`
          + opt(["","1","2","3","4","5"], st[f]||"") + `</select></label>`;
      });
      html += `</div>`;
      html += `<label style="font-size:13px">Q6 Missing citations (claims that should cite but do not):`
        + `<textarea onchange="setVal('${c.case_id}','${a.label}','missing',this.value)">`
        + esc(st.missing||"") + `</textarea></label>`;
      const n = c.answers.length;
      html += `<div class="rankrow">Q7 Rank (1 = best of the ${n}): `
        + `<select onchange="setVal('${c.case_id}','${a.label}','rank',this.value)">`
        + opt([""].concat(Array.from({length:n},(_,i)=>String(i+1))), st.rank||"") + `</select></div>`;
      html += `</div>`;
    });
    div.innerHTML = html; root.appendChild(div);
  });
}
function opt(vals, cur) { return vals.map(v=>`<option value="${v}" ${v===cur?"selected":""}>${v||"-"}</option>`).join(""); }
function esc(s){ return String(s).replace(/[&<>"]/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[m])); }

document.getElementById("evaluator").addEventListener("change", e=>{ state.__evaluator=e.target.value; save(); });

function exportJSON() {
  const rows = [];
  DATA.forEach(c => c.answers.forEach(a => {
    const st = state[k(c.case_id,a.label)] || {};
    rows.push({
      case_id: c.case_id, capability: c.capability, model: a.model, blind_label: a.label,
      relevance: num(st.relevance), coverage: num(st.coverage),
      faithfulness: num(st.faithfulness), usefulness: num(st.usefulness),
      rank: num(st.rank), missing_citations: st.missing || "",
      citations: a.citations.map(ct => ({ ref_id: ct.ref_id, label: (st.citations||{})[ct.ref_id] || "" }))
    });
  }));
  const out = { evaluator: state.__evaluator || "", exported_at: new Date().toISOString(), scores: rows };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:"application/json"});
  const url = URL.createObjectURL(blob); const a=document.createElement("a");
  a.href=url; a.download = "human_eval_scores_" + (state.__evaluator||"anon").replace(/\\W+/g,"_") + ".json";
  a.click(); URL.revokeObjectURL(url);
}
function num(v){ return v===undefined||v===""?null:Number(v); }
function clearAll(){ if(confirm("Clear all your scores?")){ state={}; localStorage.removeItem(KEY); render(); } }

render();
</script>
</body>
</html>
"""


def main() -> None:
    if not ANSWERS.exists():
        raise SystemExit("answers.json not found. Run generate_answers.py first.")
    payload = build_payload()
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    OUT.write_text(html, encoding="utf-8")
    n_ans = sum(len(c["answers"]) for c in payload)
    print(f"Wrote {OUT} ({len(payload)} questions, {n_ans} answers to score)")


if __name__ == "__main__":
    main()
