# ScholAR: EACL 2027 System Demonstration Video Script

**Target Length:** 2 minutes 30 seconds (150 seconds maximum)  
**Tone:** Concise, technical, demonstrative  
**Focus:** Proving real, inspectable evidence localization on local hardware

---

### [0:00 – 0:20] 1. The Problem: Black-Box Document Assistants
- **Visual:** Split screen showing a traditional "Chat with PDF" tool outputting generic text with unverified citations versus a dense scientific paper with complex equations and multi-column figures.
- **Narrative (Voiceover):**  
  *"Modern researchers increasingly rely on document assistants to navigate dense scientific publications. Yet most tools treat PDFs as flattened plain text, discard spatial coordinates, and hallucinate citations without inspectability. Moreover, sending confidential manuscripts or unreleased preprints to closed cloud APIs introduces serious privacy risks. We present **ScholAR**, an open-source, local-first assistant designed for inspectable, source-grounded scientific question answering."*

---

### [0:20 – 0:40] 2. Ingestion & Strict-Local Privacy Boundary
- **Visual:** User drags a research PDF into the ScholAR home screen. Real-time progress indicators display AST parsing, layout analysis, and bounding-box extraction. Terminal shows the local network policy active.
- **Narrative (Voiceover):**  
  *"ScholAR operates under a strict-local network boundary. Document processing, retrieval, and inference execute entirely on consumer hardware via local models like Qwen 3.5. During ingestion, our dual-engine pipeline extracts hierarchical document ASTs, indexes text chunks with their exact page bounding boxes, and rasterizes high-resolution crops for every figure and table. All artifacts are committed atomically to a local SQLite catalog."*

---

### [0:40 – 1:30] 3. Querying, Cited Synthesis & Evidence Inspection
- **Visual:** User types a question: *"What optimizer and learning rate schedule are used?"* The assistant generates a concise, cited response with inline chips `[1]`, `[2]`. The user clicks citation chip `[1]`; the PDF viewer navigates to the cited page and source region.
- **Narrative (Voiceover):**  
  *"When a user submits a query, ScholAR fuses BM25 lexical search and dense embeddings using Reciprocal Rank Fusion. Our budgeting service selects optimal evidence within the workstation's memory limits. The local model synthesizes an answer annotated with citation markers. Crucially, the model does not invent page numbers. Clicking any citation chip instantly commands the synchronized PDF viewer to navigate to the exact page and highlight the supporting bounding box directly in the original layout."*

---

### [1:30 – 2:00] 4. Multimodal Figures, Verification & Evidence Graph
- **Visual:** User asks: *"How does the generator architecture compare in Figure 2?"* ScholAR answers with reference to Figure 2. Clicking the citation displays an isolated high-resolution crop of the architecture diagram alongside the text. The user then clicks 'Evidence Graph', opening an interactive visual graph connecting query nodes, retrieved chunks, and cited claims.
- **Narrative (Voiceover):**  
  *"ScholAR handles multimodal scientific evidence seamlessly. For visual queries, it extracts and displays localized figure and table crops alongside textual captions. Behind every response, our post-generation verifier performs deterministic lexical and numerical consistency checks, remapping or pruning unsupported markers. Users can inspect the full reasoning graph through the Evidence Graph Modal, reviewing which candidate chunks were retrieved, prioritized, and cited."*

---

### [2:00 – 2:25] 5. Audit Traces, Performance & Availability
- **Visual:** User clicks 'Export Trace' and downloads a JSON execution receipt showing query latency, model digests, and verifier diffs. The measured local profile (reported in the paper) has $p_{50}=27.34$ seconds and $p_{95}=149.54$ seconds; the video demonstrates the trace rather than promising a fixed response time. Screen transitions to the open GitHub repository.
- **Narrative (Voiceover):**  
  *"Every interaction produces a JSON audit trace detailing retrieval scores, raw completions, and verifier decisions. The paper reports a measured local profile rather than a fixed response-time promise: on the named 18-gigabyte workstation, end-to-end latency has a median of 27.34 seconds and a 95th percentile of 149.54 seconds. ScholAR is open-source under the MIT license, including backend pipelines, the React frontend, and the evaluation harnesses."*

---

### [2:25 – 2:30] 6. Conclusion
- **Visual:** Title card with paper title, author, repository URL, package URL, and video QR code. Replace `PACKAGE_URL_REQUIRED_BEFORE_SUBMISSION` with the immutable release URL before recording the final version.
- **Narrative (Voiceover):**  
  *"Inspectable, auditable, and private scientific document QA. Try ScholAR today at our open-source repository. Thank you."*
