# ScholAR Evaluation Suite

Automated evaluation for the ScholAR pipeline: retrieval, answer faithfulness, visual
grounding, multi-document localization, a resource-matched comparison against local
baselines, abstention on unanswerable questions, and per-model efficiency. The pipeline
is model-agnostic, so the generation-based evaluations run the
**same** ScholAR pipeline across four local models (`qwen3.5:9b`, `gemma4:12b`,
`llama3.1:8b`, `mistral:7b`); only the generation model changes.

Two families of data are used. Small, manually labeled benchmarks over three landmark
papers (*Attention Is All You Need*, *RAG*, *LLaMA*) drive the metrics that need
ground-truth chunk labels. A diverse 100-case benchmark (50 text, 25 mathematical, 25
figure/table, across 25 papers; mined and source-verified by `mine_cases.py`) backs the
human study and the generation-faithfulness matrix. The labeled benchmarks are
intentionally small: enough for real ablations, not for a broad research claim.

## Benchmarks and scripts

| Script | Measures | Model / backend |
|---|---|---|
| `run_retrieval_eval.py` | Retrieval R@k, MRR, NDCG@k: keyword vs BM25 vs dense vs hybrid (14 hand-labeled; pass `--cases evaluation/benchmark_cases_scaled.json --tag scaled` for the 100-case set) | none |
| `run_faithfulness_eval.py` | Retrieval-support CFS: gold claim vs retrieval, BM25 vs hybrid (51 hand-labeled; pass `--cases evaluation/faithfulness_cases_scaled.json --tag scaled` for the 100-case set) | none |
| `run_generation_faithfulness_eval.py` | Faithfulness of the **generated answer** vs its context (single model) | backend + model |
| `run_generation_faithfulness_matrix.py` | The above across the 4 models, accumulated per run (resumable) | backend + models |
| `run_visual_eval.py` | Figure/table routing R@5 + answer-quality proxy (18 cases) | model |
| `run_visual_caption_ablation.py` | Caption-only vs full-vision, paired | model |
| `run_multidoc_eval.py` | Cross-document localization R@k, MRR (10 arXiv-resolvable cases) | none |
| `run_multidoc_bounds_eval.py` | Oracle and random-floor bounds for the multi-doc task | none |
| `run_comparison_eval.py` | ScholAR vs pdfchat / vanilla-RAG / PaperQA2-RCS on shared cases (`--rescore` re-scores offline) | model |
| `run_efficiency_eval.py` | Per-model latency, throughput, and memory footprint of the answering path (`--model`, resumable) | model |
| `run_abstention_eval.py` | Abstention vs fabrication on provably-unanswerable questions, per model (`--model`, `--rescore`) | model |
| `m3sciqa/build_m3sciqa.py` | Assembles M3SciQA's locality task (297 labeled cases) into our schema | none |
| `m3sciqa/run_m3sciqa_eval.py` | Multi-doc localization vs M3SciQA's published baselines (`--tier text\|vision`) | vision tier: model |
| `mine_cases.py` | Mines + source-verifies the diverse 100-case benchmark | model |
| `build_scaled_benchmark.py` | Auto-labels the 100-case set into the scaled retrieval + faithfulness benchmarks | none |
| `build_abstention_benchmark.py` | Builds the 20 provably-unanswerable cross-paper negatives | none |

Shared components: `embedder.py` (local all-MiniLM-L6-v2, pure PyTorch), `hybrid_retrieval.py`
(BM25 + dense + RRF), `nli_faithfulness.py` (cosine-based retrieval-support scorer), and
`llm_entailment.py` (local generated-claim entailment judge). Ground-truth files:
`benchmark_cases.json` / `faithfulness_cases.json` (3-paper hand-labeled anchors),
`benchmark_cases_scaled.json` / `faithfulness_cases_scaled.json` (auto-labeled 25-paper),
`abstention_cases.json` (unanswerable negatives), `visual_benchmark.json`, `multidoc_benchmark.json`.

## Headline results (traceable to `results/*.json`)

The latest entailment-judge rescoring supersedes the older cosine-proxy headline
numbers for generated-answer faithfulness. The cosine matrix is retained for
provenance and as evidence of why metric validation matters.

- **Retrieval (scaled, 100 cases / 25 papers):** BM25-primary R@5 0.93, MRR 0.861; plain BM25 within noise (0.94, 0.863); dense-only trails (0.74, 0.572). At scale BM25 beats dense, reversing the 3-paper anchor where dense topped a tiny set, which is why ScholAR keeps BM25 primary.
- **Retrieval-support CFS (scaled):** BM25 0.785, Hybrid 0.782; SCHR@5 0.860 → 0.750; 93/100 and 92/100 faithful.
- **Generation faithfulness across the 4 models (350 answers):** the local entailment judge measures a modest 0.594 to 0.645 mean faithfulness, with 93 of 350 answers containing at least one contradicted atom. Citation support ranges from 0.643 to 0.856. The older cosine proxy reported 0.719 to 0.951 and materially overstated the result.
- **Faithfulness negative control (20 true and corrupted claims):** the entailment judge catches 90% of corruptions and marks 75% as contradictions, compared with 50% and 0% for the cosine scorer. The judge also falsely marks 25% of true claims unfaithful, so human validation remains necessary.
- **Local comparison (qwen3.5:9b, 91 shared cases):** under the entailment judge, ScholAR scores 0.453 generation faithfulness, below vanilla RAG at 0.735 and PaperQA2-style RCS at 0.779. ScholAR retains the highest must-include answer recall at 0.563 versus 0.550 for PaperQA2-style, but that difference is not significant under paired bootstrap intervals.
- **Page support:** ScholAR's page is valid by construction, but only 249 of 378 audited cited pages support the attached claim (0.659), compared with 202 of 283 for PDF-chat (0.714). The mechanism guarantees provenance, not entailment.
- **Abstention (20 unanswerable questions):** qwen3.5 and llama3.1 decline 100%, gemma4 95%, mistral 90%; the rare non-abstentions are a benign minibatch-size overlap, not free invention.
- **Efficiency (per query, Apple Silicon 18 GB):** BM25 retrieval ~40 ms (model-independent); end-to-end 6 to 11 s; 6 to 8 GB loaded; 16 to 27 tok/s across the four models.
- **Visual:** correct figure/table routing on all 18 pilot cases, zero caption fallback.
- **Multi-doc localization (M3SciQA, 297 labeled locality cases):** text-only BM25 sits near chance (MRR 0.180 vs our empirical floor 0.121; their BM25 is 0.127 vs floor 0.126). Resolving the anchor **figure** with the local multimodal model first lifts MRR to **0.474** (qwen3.5:9b) / 0.455 (gemma4:12b): 3.3× the best open-source LMM they report (0.144), past GPT-4V (0.400), and within 0.026 of GPT-4o (0.500). Expert humans remain ahead at 0.796.

## Requirements

`run_retrieval_eval.py`, `run_multidoc_*` run on `requirements.txt` alone. The scorer-based
and generation-based scripts additionally need `numpy`, `torch`, `safetensors` for the local
encoder: `pip install -r evaluation/requirements.txt`. Generation scripts need Ollama running
with the relevant model(s) pulled; `run_generation_faithfulness_*` also need the backend
(`make backend`) and the anchor papers prepared under `backend/data/papers/`.

Run everything from the repo root. Results are written to `evaluation/results/`. If a rerun
differs from a committed result, treat it as a reproducibility signal to investigate, not
something to silently overwrite.

## Human evaluation

The blinded, multi-model, citation-grounded human study lives in `human_eval/`; see
`human_eval/README.md` for the run order and `human_eval/HUMAN_EVAL_DESIGN.md` for the instrument.
