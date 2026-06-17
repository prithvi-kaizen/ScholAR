# ScholAR Final Report

**Project:** ScholAR, a local first research paper study assistant  
**Goal:** Help students read technical research papers with a local AI assistant that shows evidence from the paper.

## 1. Problem and Motivation

Reading research papers is difficult for students, early researchers, and builders who are trying to understand a new technical area. A paper is not just a long article. It contains a research problem, motivation, method, assumptions, experiments, results, limitations, and connections to earlier work. These parts are often mixed together, and the writing assumes that the reader already knows the field.

This is especially true for machine learning and NLP papers. A student may understand the abstract but get stuck when the paper moves into model architecture, retrieval methods, training details, datasets, evaluation metrics, ablation studies, or result tables. The reader may also miss the difference between what the authors actually prove and what the paper only suggests.

General chatbots can help explain papers, but they create a trust problem. If a chatbot answers from memory or general internet knowledge, the answer may sound confident while not being grounded in the paper. For studying, this is a serious issue. A student needs to know where an answer came from. They should be able to click a citation, inspect the actual paper passage, and decide whether the answer is supported.

ScholAR was built for this problem. The project is a working GenAI research paper assistant. It lets the user search arXiv papers or upload a custom PDF, prepare the paper locally, view the PDF, generate a study plan, ask questions, and inspect cited evidence from the paper. The goal is not to replace reading. The goal is to make reading more structured and less lonely.

The main motivation is supported understanding. ScholAR should help the user move from passive reading to active study. Instead of only asking “summarize this paper”, the user should be able to ask:

- What is the paper’s main contribution?
- What method does it use?
- What evidence supports the method?
- What experiments were run?
- What limitations should I be careful about?
- What would an implementation plan look like?

This connects directly to the real problem. Students do not only need shorter summaries. They need a tool that helps them read carefully, stay grounded in the PDF, and build a mental model of the paper.

## 2. Method

ScholAR is a full stack GenAI application. The frontend is built with Next.js, TypeScript, and Tailwind CSS. The backend is built with FastAPI and Python. PDFs are processed with PyMuPDF. arXiv search is handled through the arXiv API. Local model support uses Ollama with Qwen. Cloud model support uses Groq, with `llama-3.3-70b-versatile` configured as the Groq model.

The system supports two main ways to start studying a paper:

1. Search arXiv, select a paper, and prepare it.
2. Upload a custom PDF and prepare it.

After a paper is prepared, the user studies it in a split workspace. The left side shows the PDF. The right side shows the ScholAR assistant, study goals, chat, model toggle, and cited references.

The diagram below shows the implemented system flow.

![ScholAR architecture and flow](../architecture/ScholAR_architecture_flow.png)

The backend processing pipeline works as follows:

1. The user searches arXiv or uploads a PDF.
2. The backend saves the PDF locally as `paper.pdf`.
3. PyMuPDF extracts text page by page.
4. The backend writes `metadata.json`, `pages.json`, and `chunks.json`.
5. The chunking service creates page-preserving chunks.
6. Each chunk stores text, page number, character offsets, and available section metadata.
7. The frontend displays rendered PDF page images.
8. The study panel calls backend APIs for study goals and chat.

The local storage structure is simple on purpose. A prepared paper lives under:

```text
backend/data/papers/{paper_id}/
```

Each prepared paper includes:

```text
paper.pdf
metadata.json
pages.json
chunks.json
goals_canonical_*.json
```

This makes the system easy to inspect and debug. It also avoids a database dependency for the current project stage.

The main AI flow is retrieval augmented generation. The model does not receive the full paper every time. Instead, ScholAR retrieves relevant chunks from `chunks.json`, builds a grounded prompt, and asks the selected model to answer from that evidence.

The final retrieval design is BM25-primary. Earlier versions used a more aggressive hybrid scoring method, but the evaluation showed that BM25 was the most reliable tested retriever. Because of that, ScholAR now uses BM25 as the main retrieval signal and keeps other signals as small reranking boosts.

The retrieval method includes:

- BM25 lexical scoring as the primary ranking signal.
- Lightweight hashed semantic overlap as a small tie-breaker.
- Query expansion for terms like method, result, contribution, architecture, experiment, and limitation.
- Page hints when the user or study goal mentions specific pages.
- Small boosts for research phrases like “we propose”, “we introduce”, “we show”, and “we find”.
- Small boosts for section or chunk type when the query asks about methods, results, experiments, or limitations.

The generation layer supports two providers:

- **Local Qwen through Ollama:** useful for local, private, offline-style study, but slower.
- **Groq API:** useful for stronger and faster answers, but depends on API availability and rate limits.

If Groq hits a rate limit, the frontend warns the user and allows switching to local Qwen. This is important because the tool should not completely fail during a demo or study session.

The study plan feature generates 8 paper-specific study goals. Each goal can include recursive subquestions, cited evidence, limitations, and implementation notes. This is meant to guide the student through the paper instead of giving only one flat summary.

The citation design changed during development. Early versions allowed the model to write page citations directly. That was risky because the model could invent or misuse page numbers. The final design gives the model evidence IDs such as `E1` and `E2`. The frontend then converts those IDs into numbered references like `[1]` and `[2]`. The reference panel shows the supporting passage and lets the user click back to the PDF.

## 3. Experiments and Results

The evaluation focuses on retrieval quality. This is the right first experiment because ScholAR depends on retrieval before generation. If retrieval gives the model the wrong chunks, the answer and citations can be wrong even if the language model is strong.

The benchmark uses 14 manually checked retrieval cases from 3 prepared papers:

- `1706.03762`: *Attention Is All You Need*.
- `2005.11401`: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- `2302.13971`: *LLaMA: Open and Efficient Foundation Language Models*.

Each test case contains:

- A realistic user question.
- A paper ID.
- The relevant chunk IDs that contain the answer evidence.
- Expected pages.
- A short explanation of what the case tests.

The benchmark covers these query types:

- Main idea and contribution.
- Method and architecture.
- Training or implementation details.
- Result tables and benchmark numbers.
- Human evaluation.
- Safety, bias, toxicity, and carbon footprint.
- Page-hint questions.

Four retrieval settings were compared:

| System | Description |
|---|---|
| `keyword_overlap` | Simple token overlap baseline. |
| `bm25_only` | BM25-style lexical retrieval baseline. |
| `bm25_primary_no_page_hints` | Current ScholAR retriever without page hints. |
| `bm25_primary_with_page_hints` | Current ScholAR retriever with page hints. |

The metrics are:

| Metric | Meaning |
|---|---|
| Recall@1 | The first retrieved chunk is relevant. |
| Recall@3 | At least one of the first 3 retrieved chunks is relevant. |
| Recall@5 | At least one of the first 5 retrieved chunks is relevant. |
| MRR | Mean reciprocal rank. Higher means the first relevant chunk appears earlier. |

The final results were:

| System | Cases | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| `keyword_overlap` | 14 | 0.571 | 0.786 | 0.929 | 0.687 |
| `bm25_only` | 14 | 0.714 | 0.929 | 1.000 | 0.812 |
| `bm25_primary_no_page_hints` | 14 | 0.714 | 0.929 | 1.000 | 0.812 |
| `bm25_primary_with_page_hints` | 14 | 0.714 | 0.929 | 1.000 | 0.812 |

The main result is that BM25 was the strongest tested retrieval backbone. Keyword overlap was weaker, especially on Recall@1 and MRR. The BM25-only baseline found at least one relevant chunk in the top 5 for every test case.

This result changed the project. An earlier hybrid-primary version had slightly better top-rank behavior in one run, but it missed an important result-table case in Recall@5. The missed case was `rag_qa_results`, where the question asked about open-domain QA scores in the RAG paper. The correct evidence was in `chunk_006`, but the older hybrid-primary retriever did not return it in the top 5. Since correct evidence is more important than a fancy scoring formula, the final system was changed to BM25-primary retrieval.

After this change, the current BM25-primary retriever matched BM25-only on the 14-case benchmark:

- Recall@1: 0.714.
- Recall@3: 0.929.
- Recall@5: 1.000.
- MRR: 0.812.

The page-hint ablation did not improve the aggregate score in this small benchmark. Both `bm25_primary_no_page_hints` and `bm25_primary_with_page_hints` scored the same. This does not prove that page hints are useless. It means this benchmark is too small to measure their benefit. Page hints are still useful in the interface because study goals and user questions often mention specific pages.

The experiment also shows a limitation. The benchmark has only 14 cases from 3 papers. This is enough for a course project comparison and ablation, but it is not enough for a strong research claim. A future version should evaluate 75 to 150 cases across more papers and more paper types.

## 4. Analysis

The strongest part of ScholAR is the end-to-end study workflow. The user can search or upload a paper, prepare it, view it, generate a paper-specific study plan, ask questions, and inspect cited evidence. This makes the tool feel like a real study assistant instead of a disconnected chatbot.

The split-screen interface also helps with the original problem. The user does not need to leave the paper to ask questions. The PDF stays visible, and the assistant sits next to it. This supports careful reading because the user can compare the model answer with the paper.

The study goals worked better after they became paper-specific. Generic goals like “summarize the core idea” were too shallow. Paper-specific goals are more useful because they reflect the actual topic, method, and evaluation of the paper. Recursive subquestions also helped because they turn broad goals into smaller study tasks.

The Groq and local Qwen toggle is useful, but it comes with tradeoffs. Groq gives stronger and faster answers, but it depends on API limits. Local Qwen is more private and keeps the app usable without Groq, but it is slower and can time out on longer prompts. The current fallback behavior is practical, but local model latency is still a weakness.

The biggest failure during development was citation reliability. The model sometimes produced citations that looked correct but were not actually safe enough. This happened when the model was allowed to write page citations directly. The fix was to make the backend control evidence IDs and make the frontend display numbered references. This made citations more formal and less dependent on the model inventing page numbers.

Citation highlighting was also difficult. Sometimes the cited text exists in the extracted paper text, but the PDF renderer cannot highlight it correctly. This happens because PDF text extraction does not always match the visual PDF exactly. Line breaks, ligatures, hyphenation, and spacing can break exact matching. This is a real issue with PDF-based AI systems, not only with ScholAR. The system improved by using more flexible matching, but highlighting still needs more work.

The retrieval evaluation gave the most useful technical lesson. More complex does not automatically mean better. BM25 worked better than the first hybrid-primary scoring method for this project. That is why the final system uses BM25 as the backbone. This is a good outcome because the system changed based on evidence.

Does ScholAR actually help with the stated problem? I think the answer is yes for the current project scope, but not perfectly. It helps because it keeps the paper visible, generates study goals, answers from retrieved paper chunks, and shows cited evidence. It does not fully solve the problem because citation highlighting is not perfect, local model speed is limited, and the evaluation benchmark is still small.

The next version should focus on reliability instead of adding many new features. The core loop should be:

1. Retrieve the right evidence.
2. Generate a clear answer.
3. Cite only supported claims.
4. Highlight the cited passage in the PDF.
5. Help the user understand the paper more deeply.

That loop is the real value of the project.

## 5. References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. *Attention Is All You Need*. NeurIPS, 2017. https://arxiv.org/abs/1706.03762

[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS, 2020. https://arxiv.org/abs/2005.11401

[3] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. *LLaMA: Open and Efficient Foundation Language Models*. arXiv, 2023. https://arxiv.org/abs/2302.13971

[4] Stephen Robertson and Hugo Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 2009. https://doi.org/10.1561/1500000019

[5] Qwen Team. *Qwen Technical Report*. arXiv, 2023. https://arxiv.org/abs/2309.16609
