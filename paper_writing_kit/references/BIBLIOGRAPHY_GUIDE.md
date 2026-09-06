# ScholAR Bibliography & Citation Guide

This guide maps each entry in `scholar_master_references.bib` to its specific section and rhetorical role in the EACL manuscript.

---

## 1. Multi-Modal Document Retrieval & Visual Document Understanding (VDU)

| Key | Citation | Paper Context & How to Cite |
| :--- | :--- | :--- |
| `faysse2024colpali` | Faysse et al. (2024) | Cite in **Section 1 (Intro)**, **Section 2 (Related Work)**, and **Section 3 (Visual Retrieval)**. Frame ScholAR's Channel 4 (page-level ColPali/ColQwen late interaction) as preserving typographic layout without OCR loss, but contrast with our hybrid design that adds Section 3 AST semantic scoping. |
| `blecher2023nougat` | Blecher et al. (2023) | Cite in **Section 2 (Related Work)** when discussing OCR and layout parsing limitations (hallucinated math tables in end-to-end vision-to-text models). |
| `khattab2020colbert` | Khattab & Zaharia (2020) | Cite in **Section 3 (Method - Retrieval)** for the MaxSim late-interaction scoring operator $\sum_{q \in Q} \max_{d \in D} (E_q \cdot E_d^\top)$. |
| `santhanam2022colbertv2` | Santhanam et al. (2022) | Cite for residual compression and token pruning strategies in scalable multi-vector retrieval. |

---

## 2. Textual Retrieval & Hybrid Search

| Key | Citation | Paper Context & How to Cite |
| :--- | :--- | :--- |
| `robertson2009bm25` | Robertson & Zaragoza (2009) | Cite in **Section 3 (Channel 1: Lexical Sparse BM25)**. ScholAR relies on BM25 for strict symbol matching (chemical formulas, variable names, hyperparameters like $\beta_1=0.9$). |
| `chen2024bgem3` | Chen et al. (2024) | Cite in **Section 3 (Channel 2: Dense Semantic BGE-M3)**. Used for 1024-d passage and chunk embeddings capturing conceptual paraphrasing. |
| `lewis2020rag` | Lewis et al. (2020) | Foundational RAG citation in **Section 1 (Introduction)**. |

---

## 3. Adaptive Routing, Reasoning & Graph Retrieval

| Key | Citation | Paper Context & How to Cite |
| :--- | :--- | :--- |
| `jeong2024adaptiverag` | Jeong et al. (2024) | Cite in **Section 3 (Adaptive Complexity Routing)**. ScholAR advances Adaptive-RAG by extending routing from binary retrieval/no-retrieval to a 5-tier pipeline ($L_1$ Direct Factoid to $L_5$ Cross-Document Multi-Hop). |
| `edge2024graphrag` | Edge et al. (2024) | Cite in **Section 3 (Active Cross-Modal Graph Expansion)**. Contrast ScholAR's localized AST-anchored graph traversal with heavy offline global knowledge graphs. |
| `asai2023selfrag` | Asai et al. (2024) | Cite in **Section 4 (Verification & Guardrails)**. ScholAR's Conservative Verifier implements self-reflective critique to reject ungrounded or hallucinated answers. |
| `yan2024crag` | Yan et al. (2024) | Cite in **Section 4 (Self-Correction & Fallbacks)**. ScholAR triggers automated query reformulation and graph re-expansion if verification confidence $< 0.65$. |
| `sun2025mlr` | Sun et al. (2025) | Cite in **Section 4 (Multi-Level Reasoning Framework)**. Structured multi-level reasoning guiding decomposition into atomic sub-questions. |

---

## 4. Evaluation Benchmarks & Factuality

| Key | Citation | Paper Context & How to Cite |
| :--- | :--- | :--- |
| `dasigi2021qasper` | Dasigi et al. (2021) | Main scientific question answering benchmark in **Section 5 (Evaluation)**. ScholAR evaluates factual answer extraction against ground-truth paper evidence. |
| `wadden2020scifact` | Wadden et al. (2020) | Scientific claim verification benchmark in **Section 5**. Evaluates sentence-level precision/recall and rationale selection. |
| `min2023factscore` | Min et al. (2023) | Atomic factuality evaluation metric in **Section 5 & Section 6**. Used to measure claim-level precision of LLM answers. |
| `li2024m3sciqa` | Li et al. (2024) | Multi-modal multi-document scientific QA benchmark. Validates ScholAR's cross-paper synthesis and chart/table question answering. |

---

## 5. Foundation Models & Exemplar Testbeds

| Key | Citation | Paper Context & How to Cite |
| :--- | :--- | :--- |
| `bai2024qwen25` | Qwen Team (2024) | Backbone local SLM (`Qwen2.5-9B-Instruct` / `Qwen3.5-9B`) enabling 100% on-premise air-gapped inference. |
| `dubey2024llama3` | Dubey et al. (2024) | Backbone fast local SLM (`Llama-3.2-3B-Instruct`) evaluated in low-latency industrial deployment profiles. |
| `vaswani2017attention` | Vaswani et al. (2017) | Used as canonical testbed paper (`arXiv:1706.03762`) in multi-hop evaluation case studies. |
| `he2016deep` | He et al. (2016) | Used as canonical benchmark paper (`arXiv:1512.03385`) for table extraction and ablation queries. |
