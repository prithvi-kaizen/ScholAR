# ScholAR Evaluation Suite

Automated evaluation for the ScholAR pipeline: retrieval, answer faithfulness, visual
grounding, multi-document localization, and a resource-matched comparison against local
baselines. The pipeline is model-agnostic, so the generation-based evaluations run the
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
| `run_retrieval_eval.py` | Retrieval R@k, MRR: keyword vs BM25 vs dense vs hybrid (14 cases) | none |
| `run_faithfulness_eval.py` | Retrieval-support CFS: gold claim vs retrieval, BM25 vs hybrid (51 cases) | none |
| `run_generation_faithfulness_eval.py` | Faithfulness of the **generated answer** vs its context (single model) | backend + model |
| `run_generation_faithfulness_matrix.py` | The above across the 4 models, accumulated per run (resumable) | backend + models |
| `run_visual_eval.py` | Figure/table routing R@5 + answer-quality proxy (18 cases) | model |
| `run_visual_caption_ablation.py` | Caption-only vs full-vision, paired | model |
| `run_multidoc_eval.py` | Cross-document localization R@k, MRR (10 arXiv-resolvable cases) | none |
| `run_multidoc_bounds_eval.py` | Oracle and random-floor bounds for the multi-doc task | none |
| `run_comparison_eval.py` | ScholAR vs pdfchat / vanilla-RAG / PaperQA2-RCS on shared cases | model |
| `mine_cases.py` | Mines + source-verifies the diverse 100-case benchmark | model |

Shared components: `embedder.py` (local all-MiniLM-L6-v2, pure PyTorch), `hybrid_retrieval.py`
(BM25 + dense + RRF), `nli_faithfulness.py` (SummaC-ZS-style scorer). Ground-truth files:
`benchmark_cases.json` (retrieval), `faithfulness_cases.json` (oracle claims),
`visual_benchmark.json`, `multidoc_benchmark.json`.

## Headline results (traceable to `results/*.json`)

- **Retrieval:** BM25-primary R@5 0.929, MRR 0.788; dense-only tops this small set (R@5 1.000).
- **Retrieval-support CFS:** BM25 0.807, Hybrid 0.827; SCHR@5 0.824 → 0.922; 48/51 and 49/51 faithful.
- **Generation faithfulness (single model, gemma4:12b, 51 cases):** mean 0.971, contradiction rate 0.0, 94% of citations fully supported (188/12/0 of 200).
- **Visual:** correct figure/table routing on all 18 pilot cases, zero caption fallback.
- **Multi-doc localization:** R@5 0.50, MRR 0.183, at or below the random floor (R@5 0.625) — the clearest open problem.

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

The model-agnostic, citation-grounded human study lives in `human_eval/` — see
`human_eval/README.md` for the run order and `human_eval/HUMAN_EVAL_DESIGN.md` for the instrument.
