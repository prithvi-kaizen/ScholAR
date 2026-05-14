# ScholAR Final Report

## Project title

ScholAR: A Local First Research Paper Study Assistant

## 1. The real problem

Research papers are hard to study, especially for students who are still learning how to read technical work. A paper usually has many layers at once. There is the main idea, the motivation, the method, the math, the experiments, the results, the limitations, and the connection to prior work. Even when the paper is important, it is easy to get lost.

This problem becomes worse in machine learning and NLP papers because the writing often assumes background knowledge. A student may understand some parts of the introduction but then struggle when the paper moves into architecture details, datasets, metrics, ablations, or result tables. Many students end up reading passively. They highlight text, copy notes, or ask a general chatbot for help. The issue is that a general chatbot can answer confidently without being grounded in the paper. That creates a trust problem. If the system gives an answer but the user cannot see where the answer came from, the answer is not very useful for real studying.

ScholAR was built to address this problem in a practical way. The goal was not to build a huge research platform. The goal was to build a working assistant that lets a student search or upload a paper, view the PDF, generate a study plan, ask questions, and see cited evidence from the paper.

The main idea is simple:

- Keep the paper visible while the AI answers.
- Use retrieval before generation so the model sees relevant paper context.
- Make study goals specific to the paper instead of generic.
- Let the user use either a Groq API model or a local Ollama Qwen model.
- Keep the system understandable enough to debug and evaluate.

This matters because the real user need is not just summarization. The real need is supported understanding. A good research paper assistant should help the reader ask better questions, trace answers back to the PDF, and notice what the paper does not prove.

## 2. What was built

ScholAR is a full stack GenAI application. The frontend is built with Next.js, TypeScript, and Tailwind CSS. The backend is built with FastAPI and Python. PDFs are processed with PyMuPDF. arXiv search is handled through the arXiv API. Local model support is handled through Ollama, with Qwen 3.5 9B as the default local model. Groq support was added so the user can switch to a stronger cloud model when needed. (llama-3.3-70b-versatile)

The current system supports these main flows:

1. Search for papers from arXiv.
2. Select a paper and prepare it for study.
3. Upload a custom PDF and prepare it for study.
4. Extract paper text page by page.
5. Chunk the paper into retrievable passages.
6. View the paper side by side with the study assistant.
7. Generate 8 paper-specific study goals.
8. Generate recursive subquestions under each study goal.
9. Ask questions about the paper.
10. Retrieve relevant evidence chunks.
11. Generate an answer from Groq or local Qwen.
12. Show cited references and let the user click them to jump back to the PDF.
13. Fall back to local Qwen when Groq rate limits are reached.

The system is intentionally local first. The backend stores prepared papers in local folders under `backend/data/papers`. Each paper has a PDF, metadata, extracted pages, chunks, and generated study goals. This keeps the project easy to inspect. It also makes the system explainable because the whole pipeline is visible on disk.

The frontend was designed around a split study workspace. The left side shows the PDF pages. The right side shows the ScholAR study panel and chat. This is important because the AI answer should not replace the paper. The answer should sit next to the paper.

## 3. Technical approach

The diagram below shows the actual implemented flow of the system.

![ScholAR architecture and flow](../architecture/ScholAR_architecture_flow.png)

The backend uses a retrieval augmented generation flow. The model does not receive the whole paper every time. Instead, ScholAR retrieves relevant chunks first and sends those chunks to the model as paper evidence.

The paper processing flow is:

1. Download or receive the PDF.
2. Save it locally as `paper.pdf`.
3. Extract text page by page with PyMuPDF.
4. Save extracted pages as `pages.json`.
5. Split the pages into chunks.
6. Add chunk metadata such as page number, section title, paragraph text, and chunk type when possible.
7. Save chunks as `chunks.json`.

The retrieval layer started as simple keyword overlap. Later it was improved into a hybrid retrieval system. After evaluation, the design was changed again because BM25 was the most reliable method on the project benchmark. The current retrieval is BM25-primary. That means BM25 does the main ranking, and the other signals only make small reranking adjustments.

- BM25-style lexical scoring as the main retrieval signal.
- Lightweight hashed embedding similarity as a small tie-breaker.
- Query expansion for common research terms such as method, result, contribution, architecture, experiment, and limitation.
- Page hints when the user or study goal mentions specific pages.
- Small reranking boosts for useful paper phrases such as “we propose”, “we introduce”, “we show”, and “we find”.
- Small section or chunk-type boosts when the query asks about methods, results, experiments, or limitations.

This retrieval layer is still simple compared to a production system with learned embeddings and a cross-encoder reranker, but it is a meaningful improvement over plain keyword matching. More importantly, it follows the evaluation result instead of forcing a more complicated method when BM25 is already strong.

The answer generation prompt asks the model to answer in a structured way. The model is asked to use paper evidence first, cite evidence IDs, and avoid inventing page citations. The frontend converts evidence IDs into clean numbered references. This was done because page-only citations were confusing and sometimes looked less formal. Numbered references are closer to a research style, and the reference panel can show the actual passage text.

The system also supports two model providers:

- Local Qwen through Ollama.
- Groq API through a stronger hosted model.

The goal is to keep both modes consistent. Groq is usually faster and stronger, but it can hit rate limits. Local Qwen is slower but keeps the system usable when API limits are reached. This is why the app includes a provider toggle and Groq limit fallback behavior.

## 4. What worked

The biggest thing that worked was the end-to-end study flow. A user can search or upload a paper, prepare it, view it, generate goals, and ask questions in the same interface. This is important because many GenAI class projects only show one isolated feature. ScholAR became a real study workflow instead of just a chat demo.

The split PDF and assistant layout also worked well. It matches the actual reading behavior of a student. The user can look at the PDF and the AI response at the same time. This makes the tool feel more like a reading companion than a detached chatbot.

Paper-specific study goals were another strong part. Early versions used generic study goals such as “summarize core idea” and “explain methodology”. That was too shallow. The improved version generates goals that reflect the actual paper. For example, for a RAG paper, the goals focus on retrieval augmented generation, parametric and non-parametric memory, training, and evaluation. For the Transformer paper, the goals focus on attention, encoder-decoder design, positional encoding, parallelization, and translation results.

The recursive breakdown is also useful. Instead of giving only 8 high-level goals, the system can break each goal into subquestions. This is closer to how someone actually studies a paper. A goal like “Understand methodology” is too broad by itself. It becomes more useful when broken into questions like:

- What problem does the method solve?
- What are the main components?
- What assumptions does it make?
- What evidence supports it?
- What are its limitations?

The Groq and local model toggle also worked as a practical feature. Groq improves answer quality when available. Local Qwen keeps the app usable when the API is unavailable or rate-limited. This is important for a classroom project because demos should not completely depend on one external API.

The quantitative evaluation also worked as a real project artifact. Instead of only saying “retrieval improved”, the project now has a benchmark script, benchmark cases, raw results, and a written evaluation report. This makes the project much stronger for final submission.

## 5. What failed or was weaker than expected

The hardest part was citations. At first, the model sometimes produced page citations that looked correct but were not reliable enough. This happened because the model was allowed to write page numbers directly. That is risky because the model may copy a number from the prompt or infer a page number incorrectly. The fix was to make citations evidence-based. The backend provides evidence IDs, and the model is asked to cite only those IDs. The app then turns those IDs into numbered references.

Another issue was citation highlighting. Sometimes the cited text really existed in the PDF, but the frontend did not highlight it correctly. This happened because PDF text extraction and PDF visual text are not always identical. Line breaks, ligatures, hyphenation, and spacing can make exact string search fail. For example, a sentence may exist in the PDF, but PyMuPDF search may not find the full sentence as one exact string. This made it look like the citation was wrong even when the evidence text was present.

This is a real limitation of PDF-based systems. PDFs are layout documents, not clean semantic documents. The project improved this by adding more flexible quote matching, but citation highlighting still needs more work before it can be considered robust.

Another weakness was local model latency. Local Qwen can produce good answers, but it is slower than Groq and sometimes times out on longer prompts. This is expected because the local machine has limited compute compared to hosted inference. The solution was to shorten local prompts, limit evidence size, and show clearer failure messages. Still, local mode is not as smooth as Groq mode.

Search quality also needed improvement. arXiv search can return surprising results because it depends on the query and arXiv matching behavior. The project improved search handling, but search is still not the main technical contribution. The main contribution is the study and retrieval workflow after a paper is selected.

The biggest technical limitation is that the retrieval benchmark is still small. The first version of the hybrid retriever did not beat BM25-only on every metric. That was a useful failure because it changed the decision. The current system now uses BM25 as the primary retrieval method and keeps hybrid signals as small reranking boosts.

## 6. Quantitative evaluation

For the final evaluation, the project uses a small retrieval benchmark. This benchmark measures whether ScholAR retrieves the right evidence chunks before the model answers. This is the right place to start because the quality of the model answer depends heavily on retrieval. If retrieval gives the wrong chunks, the model may answer with weak or incorrect evidence.

The benchmark uses 14 manually checked test cases from 3 prepared papers:

- `1706.03762`: Attention Is All You Need.
- `2005.11401`: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- `2302.13971`: LLaMA: Open and Efficient Foundation Language Models.

Each test case includes:

- A realistic user question.
- The paper ID.
- The chunk IDs that contain relevant evidence.
- The expected pages.
- A short reason explaining what the case tests.

The benchmark covers main idea questions, method questions, architecture questions, result table questions, training details, human evaluation, safety, bias, toxicity, carbon footprint, and page-hint questions.

Four retrieval settings were compared:

| System                           | What it means                                                  |
| -------------------------------- | -------------------------------------------------------------- |
| `keyword_overlap`              | A simple baseline using token overlap.                         |
| `bm25_only`                    | A BM25-style lexical retrieval baseline.                       |
| `bm25_primary_no_page_hints`   | Current ScholAR retrieval without page hints.                  |
| `bm25_primary_with_page_hints` | Current ScholAR retrieval with page hints when they are given. |

The metrics were:

| Metric   | Meaning                                                            |
| -------- | ------------------------------------------------------------------ |
| Recall@1 | The first retrieved chunk is relevant.                             |
| Recall@3 | At least one of the first 3 chunks is relevant.                    |
| Recall@5 | At least one of the first 5 chunks is relevant.                    |
| MRR      | Mean reciprocal rank. Higher means relevant chunks appear earlier. |

The results were:

| System                           | Cases | Recall@1 | Recall@3 | Recall@5 |   MRR |
| -------------------------------- | ----: | -------: | -------: | -------: | ----: |
| `keyword_overlap`              |    14 |    0.571 |    0.786 |    0.929 | 0.687 |
| `bm25_only`                    |    14 |    0.714 |    0.929 |    1.000 | 0.812 |
| `bm25_primary_no_page_hints`   |    14 |    0.714 |    0.929 |    1.000 | 0.812 |
| `bm25_primary_with_page_hints` |    14 |    0.714 |    0.929 |    1.000 | 0.812 |

The important result is that BM25 was the strongest and most reliable retrieval method on this benchmark. The earlier hybrid-primary version had slightly better top-rank behavior in one run, but it missed one result-table case in Recall@5. That was not acceptable because a paper assistant needs reliable evidence more than a fancy scoring formula.

Because of that, the project changed the actual retrieval design. ScholAR now uses BM25 as the main retriever. Semantic similarity, page hints, section hints, and paper phrase boosts are still present, but they are small reranking signals. They should help when candidates are close, but they should not overpower BM25.

After this change, the current BM25-primary system matched BM25-only on the 14-case benchmark:

- Recall@1: 0.714.
- Recall@3: 0.929.
- Recall@5: 1.000.
- MRR: 0.812.

This means the current system found at least one correct evidence chunk in the top 5 for every test case. It also means the previous failure case, `rag_qa_results`, was fixed by making BM25 dominant.

The page-hint ablation did not change the aggregate metric on this small benchmark after BM25 became dominant. That does not mean page hints are useless. It means the current benchmark is too small to prove their value. Page hints are still reasonable to keep because they are intuitive for study goals and user questions that mention specific pages.

## 7. Why the results matter

The evaluation connects directly to the real problem. ScholAR is supposed to help students understand papers with grounded evidence. If the system retrieves the wrong passages, it cannot reliably teach the paper. The retrieval benchmark measures the part of the system that decides what evidence the model sees.

The results show that ScholAR has a real technical direction:

- Keyword retrieval is simple but weaker at ranking.
- BM25 is the strongest tested retrieval backbone for this project.
- Hybrid signals should be used carefully as reranking aids, not as the main signal.
- Page hints are useful conceptually, but this small benchmark does not yet prove a measurable gain after switching to BM25-primary retrieval.
- Table and result retrieval improved after BM25 became dominant.

This is an honest and useful result. It does not claim the system is solved. It shows where the system works and where it needs work.

For a final class project, this is enough to show a real comparison and ablation. For a research conference direction, the next version would need a larger benchmark, better retrieval models, and stronger citation faithfulness evaluation.

## 8. What should be improved next

The first improvement should still be retrieval, but the direction is clearer now. BM25 should remain the backbone, especially for result tables and exact metric questions. Future work should test whether learned embeddings or a reranker can improve over BM25 without hurting Recall@5.

The second improvement should be citation faithfulness. The system should check whether every cited quote can be found in the extracted paper text and ideally highlighted in the rendered PDF. If a citation cannot be matched, it should not be shown as confident evidence.

The third improvement should be a larger benchmark. The current benchmark has 14 cases across 3 papers. A stronger final benchmark should have at least 75 to 150 cases across more papers and more research areas. It should include method, results, equations, limitations, ablations, and implementation questions.

The fourth improvement should be answer-level evaluation. Retrieval evaluation is the right first step, but eventually the project should also measure answer quality. Possible answer metrics include citation precision, answer completeness, unsupported claim rate, and human preference between Groq and local Qwen answers.

The fifth improvement should be a better local model experience. Local Qwen should use shorter prompts, stronger evidence selection, and possibly streaming output. That would make local mode feel more reliable.

## 9. Final reflection

ScholAR worked best when it treated the model as a study assistant, not as a replacement for the paper. The strongest part of the project is the full workflow: prepare the PDF, view the paper, generate a paper-specific study plan, ask questions, and inspect cited evidence.

The project also showed why grounded AI systems are hard. The model is only one part of the system. Retrieval, chunking, PDF extraction, citation formatting, highlighting, UI design, and fallback behavior all affect whether the user trusts the answer.

The most honest conclusion is this: ScholAR is a working MVP with a real evaluation, but it is not yet a finished research system. It has a strong direction. It helps users study papers in a more structured way, and the evaluation gives clear evidence about what works and what still needs improvement.

The next version should focus less on adding more features and more on making the core loop reliable:

1. Retrieve the right evidence.
2. Generate a clear explanation.
3. Cite only what is actually supported.
4. Highlight the cited passage in the PDF.
5. Help the user move from paper reading to real understanding.

That loop is the real problem ScholAR is trying to solve.
