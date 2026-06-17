# ScholAR — Research Roadmap & Thesis Checklist
**Target Venue:** AAAI-27 (The Forty-First AAAI Conference on Artificial Intelligence)
**Target Submission Window:** Approximately August 2026

---

## Related Work Filed

- [x] Read **SlideTailor** (AAAI-26): *"Personalized Presentation Slide Generation for Scientific Papers"*
  - NUS; Wenzheng Zeng, Mingyu Ouyang, Langyuan Cui, Hwee Tou Ng
  - Key ideas: implicit preference distillation, chain-of-speech alignment, preference-guided slide layout
  - **Connection to ScholAR:** SlideTailor reads papers as raw PDFs using text/image extraction. Their upstream
    bottleneck is the same one we are solving — faithful, structured extraction from PDFs. Their benchmark (PSP)
    is built on AI-conference papers only. A properly structured RAG grounding layer could plug directly into
    their pipeline to make citation evidence extractable and verifiable.

---

## Direction 1 — Generation Evaluation (Faithfulness & Citation Quality)

> **Goal:** Move from retrieval-only evaluation (Recall@K, MRR) to a full end-to-end evaluation of
> the generated answer quality, specifically measuring whether model claims are faithfully grounded
> in the PDF evidence.

### 1.1 Faithfulness Metric (NLI-based) ✅ COMPLETED (v3 — Research Grade)
- [x] Survey NLI-based faithfulness metrics: ALIGNSCORE, TRUE, SummaC, MiniCheck, FActScore
- [x] Implement **NLI-CFS** — three-tier faithfulness pipeline:
      - Tier 1: Token-level NLI (SummaC/AlignScore-style): atomic decomposition + entailment classification
      - Tier 2: SCR (Semantic Coverage Rate) using all-MiniLM-L6-v2 loaded from local HF cache via torch
      - Tier 3: KFP (Key Fact Presence) — numeric + technical term matching
      - Combined: CFS = 0.50 × NLI + 0.30 × SCR + 0.20 × KFP
- [x] Implement **Hybrid BM25+Dense+RRF** retrieval (Reciprocal Rank Fusion, k=60)
      using all-MiniLM-L6-v2 for dense ranking without sentence-transformers library
- [x] Expand benchmark from 20 → **51 oracle-claim cases** across 3 papers
      (15 Attention, 12 RAG, 24 LLaMA; 8 claim types fully balanced)
- [x] **Results (v3):**
      - BM25-primary: Combined NLI-CFS = 0.658, SCHR@5 = 0.863 (33/51 FAITHFUL)
      - Hybrid BM25+Dense+RRF: NLI-CFS = **0.667**, SCHR@5 = **0.922** (+6.9 pp)
      - Conceptual claims: 0.810 (best); Result-number: 0.469 (hardest)
- [x] Full ablation: CFS monotonically improves Top-1 (0.548) → Top-5 (0.658/0.667)
- [x] LaTeX paper updated with real results, new tables, and SummaC/AlignScore/RRF citations
- [x] PDF compiled: `paper/scholar_aaai27.pdf` (5 pages, clean compile)


### 1.2 LLM-as-Judge Answer Evaluation
- [ ] Build a prompt template for LLM-as-judge scoring (factuality, grounding, completeness) 
      using Groq `llama-3.3-70b` as the judge
- [ ] Evaluate judge agreement with NLI-based CFS scores on a 20-sample overlap
- [ ] Write `evaluation/run_answer_eval.py` to handle both NLI and LLM-judge modes
- [ ] Document evaluation framework in `evaluation/README.md`

### 1.3 Hallucination Detection
- [ ] Identify the top-3 failure modes causing hallucinated citations in the current system
- [ ] Test whether enforcing the `E1`, `E2` evidence ID scheme reduces hallucination vs. free-form citation
- [ ] Quantify hallucination rate reduction as a research contribution

---

## Direction 2 — Visual Document Retrieval & Structured Extraction

> **Goal:** Replace raw PyMuPDF plain-text extraction with layout-aware, multimodal document parsing.
> Evaluate how structured extraction (tables, figures, formulas) improves retrieval accuracy vs. flat text.

### 2.1 Structured Markdown Extraction (Flat Text → Markdown)
- [ ] Integrate a structured parser: evaluate **Nougat** (Meta) or **Grobid** as alternatives to PyMuPDF
- [ ] Compare output quality on 3 benchmark papers (Attention Is All You Need, RAG, LLaMA)
  - Criteria: Table fidelity (are table rows preserved?), equation structure (LaTeX vs. plain text), figure captions
- [ ] Implement a `markdown_chunking_service.py` that wraps the new parser output into the same
      `chunk_id → text → page` schema used by the existing retriever
- [ ] **Experiment 1:** Run the 14-case retrieval benchmark on markdown-chunked text vs. plain-text chunked text
- [ ] Report Recall@K and MRR delta between `flat_text` and `structured_markdown` chunking
- [ ] Write the experiment design as a formal ablation study section

### 2.2 Visual Page Retrieval (ColPali / Image-Embedded Retrieval)
- [ ] Read the ColPali paper: *"Efficient Document Retrieval with Vision Language Models"*
- [ ] Set up `colpali-engine` locally (or via Hugging Face Inference API)
- [ ] Build a `visual_retrieval_service.py` that indexes each PDF page as a visual token embedding
      using a PaliGemma-based VLM
- [ ] **Experiment 2:** Compare visual page retrieval vs. BM25 text retrieval for:
  - Figure-based questions (e.g., *"What does Figure 3 show?"*)
  - Table-based questions (e.g., *"What BLEU score does the Transformer achieve?"*)
  - Dense equation questions (e.g., *"What is the attention function formula?"*)
- [ ] Create a specialized test set of 20 figure/table/equation-heavy queries for this experiment
- [ ] Design a hybrid retrieval mode: BM25 text + ColPali visual re-ranker
- [ ] Benchmark hybrid vs. BM25-only and ColPali-only

---

## Direction 3 — Equation & Mathematical Semantics

> **Goal:** Enable retrieval and comprehension of mathematical formulas and equations in technical papers —
> a class of content that BM25 tokenizers and standard embedding models completely ignore.

### 3.1 Equation Detection & Extraction
- [ ] Audit PyMuPDF extraction quality on math-heavy papers (e.g., Attention Is All You Need, equations in the LLaMA paper)
- [ ] Evaluate whether **Nougat** preserves LaTeX syntax during extraction
- [ ] Implement a preprocessing step to detect inline vs. display equations and tag them separately in chunks

### 3.2 Specialized Math Tokenization
- [ ] Study math-aware tokenization strategies: MathBERT, token-level LaTeX parsing
- [ ] Implement a `math_tokenizer` in `retrieval_service.py` that augments BM25 with symbol-level tokens
      (e.g., `\sigma`, `\frac`, `\text{softmax}`) alongside natural language terms
- [ ] **Experiment 3:** Run retrieval benchmark on 10 math-specific queries not covered in the current 14-case set:
  - *"What is the softmax attention formula?"*
  - *"How is the loss function defined in the paper?"*
  - *"What are the model size equations?"*
- [ ] Report math-query Recall@K with and without the math tokenizer

### 3.3 Formula Rendering in the Study Panel
- [ ] Integrate MathJax or KaTeX into the frontend to render LaTeX strings returned by the backend
- [ ] Surface equations as a dedicated "Equations" section in the Study Panel UI
- [ ] Allow equation-level citation highlights (draw bounding boxes around formula regions in the PDF viewer)

---

## Direction 4 — Multi-Document Tracing (Citation Graph)

> **Goal:** Extend ScholAR from single-paper study to a multi-document reasoning tool where
> questions about comparative claims can pull evidence from both the primary paper and its cited works.
> Inspired by Connected Papers (https://www.connectedpapers.com/).

### 4.1 Citation Graph Construction
- [ ] Build a citation parser for arXiv papers using the **Semantic Scholar API** (no API key required for S2)
- [ ] On paper preparation, automatically fetch the paper's top-cited references using arXiv/S2
- [ ] Store a `citations.json` per paper with: title, arXiv ID, S2 paper ID, abstract, year, citation count
- [ ] Build a lightweight graph visualization in the frontend (nodes = papers, edges = citation relationships)
      — look at `d3.js` force-directed graph or a library like `vis-network`

### 4.2 Multi-Paper Retrieval
- [ ] Extend `retrieve_chunks()` to optionally accept a list of paper IDs (primary + cited)
- [ ] When a query contains comparison signals (e.g., *"How does this compare to [12]?"*), trigger
      multi-paper retrieval that fetches relevant chunks from the referenced paper too
- [ ] Build a `cross_paper_retriever.py` that merges ranked chunks from multiple papers and
      labels each result with its source paper
- [ ] **Experiment 4:** Evaluate cross-paper retrieval on 10 comparative queries that reference a known citation:
  - Ground truth: correct chunk from the *cited* paper, not the primary paper
  - Metric: Cross-paper Recall@5 and MRR

### 4.3 Connected Papers Graph View
- [ ] Design a graph view page in the frontend:
  - Node = paper (primary paper highlighted in a distinct color)
  - Edge = citation relationship (directional)
  - Node hover = title, abstract snippet, year
  - Node click = open that paper in the study workspace
- [ ] Optionally color-code nodes by domain / recency / citation count
- [ ] Wire graph data through a new backend endpoint: `GET /api/papers/{paper_id}/citation-graph`

### 4.4 Research Claim Lineage
- [ ] When a model answer cites a claim, check whether that claim originated in the primary paper
      or was inherited from a reference — label citations as `[Primary]` or `[Cited: Paper Title]`
- [ ] Expose this claim lineage in the Study Panel as a "Claim Trace" UI element

---

## Conference Submission Preparation (AAAI-27)

- [ ] Identify the most novel and evaluable direction for the primary paper contribution
- [ ] Draft paper abstract (target: positioning as a grounded scientific reading comprehension system
      with structured multi-modal retrieval)
- [ ] Write the related work section covering: RAG systems, document layout analysis, scientific QA,
      citation networks, SlideTailor (AAAI-26), ColPali, BEIR benchmark, SciFact
- [ ] Run all proposed experiments and fill results tables
- [ ] Design system figures: architecture diagram, retrieval pipeline flowchart, citation graph example
- [ ] Set up arXiv preprint
- [ ] Prepare camera-ready version per AAAI-27 formatting guidelines
- [ ] Submit by AAAI-27 abstract deadline (check: https://aaai.org/conference/aaai/aaai-27/)
