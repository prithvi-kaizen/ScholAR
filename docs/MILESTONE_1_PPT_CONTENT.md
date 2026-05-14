# ScholAR Milestone 1 PPT Content

Use this as the slide-by-slide content for your Milestone 1 presentation. Keep the slides concise, and explain the details verbally using the speaker notes.

## Slide 1: Title

**ScholAR: Local LLM Research Paper Assistant**

Milestone 1 Presentation

Team/Presenter: Prithviraj Sangramsinh Patil

**One-line pitch:** ScholAR helps users search, open, read, and understand research papers using a local AI study assistant.

**Speaker notes:**
ScholAR is inspired by paperbreakdown.com, but the current focus is a local-first MVP. The system helps a student or researcher move from paper discovery to structured study without sending paper content to an external cloud LLM.

## Slide 2: Problem Statement

**Research papers are hard to study efficiently.**

- Papers are long, dense, and filled with domain-specific language.
- Readers often understand the abstract but struggle with methods, experiments, and implementation details.
- Normal PDF readers do not guide learning.
- General chatbots may not stay grounded in the paper and may raise privacy concerns.

**Speaker notes:**
The main problem is not only finding papers. The harder part is turning a selected paper into an understandable study session. ScholAR targets students, researchers, and AI engineers who need to learn papers faster.

## Slide 3: Project Goal

**Goal:** Build a local research paper assistant that supports the complete first study workflow.

- Search papers from arXiv.
- Select and prepare a paper.
- Extract paper text locally.
- View the paper in the browser.
- Generate study goals.
- Ask questions with answers grounded in paper chunks.

**Speaker notes:**
Milestone 1 focuses on proving the end-to-end workflow. Advanced features like authentication, recommendation systems, fine-tuning, and knowledge graphs are intentionally left for later.

## Slide 4: Milestone 1 Scope

**Completed in Milestone 1**

- arXiv search using the public Atom API.
- Local PDF download.
- Page-wise text extraction using PyMuPDF.
- Simple chunking with page references.
- Local JSON-based paper storage.
- Local Ollama/Qwen integration.
- Study goals generation.
- Grounded chat with cited evidence.
- Next.js dark UI with search, paper modal, PDF viewer, study panel, and chat.

**Out of scope for this milestone**

- User login.
- Cloud deployment.
- Multi-user database.
- Learned embedding-based retrieval.
- Similar paper recommendations.
- Fine-tuning.

## Slide 5: High-Level Architecture

```text
User
  |
  v
Next.js Frontend
  |-- Home/Search UI
  |-- Paper Modal
  |-- PDF Viewer
  |-- Study Goals + Chat
  |
  v
FastAPI Backend
  |-- arXiv Search Service
  |-- PDF Download/Extraction Service
  |-- Chunking Service
  |-- Retrieval Service
  |-- Ollama Service
  |
  v
Local Storage + Local Ollama Model
```

**Speaker notes:**
The frontend is responsible for the user experience. The backend handles all paper processing and model communication. The local model is accessed through Ollama, and prepared papers are stored in local files under `backend/data/papers`.

## Slide 6: Frontend Architecture

**Frontend stack**

- Next.js with TypeScript.
- Tailwind CSS.
- Lucide icons.
- Browser localStorage for recently viewed papers and bookmarks.

**Main frontend parts**

- `app/page.tsx`: Home page and paper search.
- `PaperCard`: Shows paper summary cards.
- `PaperModal`: Displays paper details and starts paper preparation.
- `app/paper/[id]/page.tsx`: Study workspace.
- `PdfViewer`: Renders full paper pages.
- `StudyPanel`: Shows study goals and quick-start actions.
- `ChatBox`: Sends paper questions to the backend.

**Speaker notes:**
The UI has two major screens: the discovery page and the study workspace. The study workspace uses a split layout: PDF on the left and AI assistant on the right.

## Slide 7: Backend Architecture

**Backend stack**

- FastAPI.
- Pydantic request models.
- httpx for async HTTP calls.
- PyMuPDF for PDF extraction and rendering.
- Ollama API for local model generation.
- Local JSON file storage.

**Core backend modules**

- `main.py`: API routes.
- `arxiv_service.py`: arXiv search, caching, fallback results, ranking.
- `pdf_service.py`: PDF download, text extraction, page rendering.
- `chunking_service.py`: Page-preserving text chunks.
- `retrieval_service.py`: BM25-primary retrieval with lightweight reranking.
- `ollama_service.py`: Local model calls and fallback study goals.

## Slide 8: Data Pipeline

```text
1. User searches a topic
2. Backend queries arXiv
3. User selects a paper
4. Backend downloads paper PDF
5. PyMuPDF extracts page-wise text
6. Text is split into chunks
7. Metadata, pages, and chunks are saved locally
8. Frontend opens the study workspace
9. User reads pages and asks questions
10. Backend retrieves relevant chunks and asks Ollama
11. Answer returns with cited references
```

**Speaker notes:**
The system keeps the pipeline simple and explainable. Every prepared paper has local files for metadata, pages, chunks, and the PDF, which makes debugging easy.

## Slide 9: API Endpoints

**Backend API**

- `GET /health`: Check backend and Ollama status.
- `GET /api/search?q=...`: Search arXiv papers.
- `POST /api/papers/prepare`: Download, extract, chunk, and save a paper.
- `GET /api/papers/{paper_id}`: Return prepared paper metadata.
- `GET /api/papers/{paper_id}/pdf`: Return local PDF file.
- `GET /api/papers/{paper_id}/page/{page_number}.png`: Render a page image.
- `POST /api/papers/{paper_id}/study-goals`: Generate or load study goals.
- `POST /api/papers/{paper_id}/chat`: Answer questions using retrieved context.

**Speaker notes:**
The backend API is small but covers the complete flow needed for Milestone 1.

## Slide 10: Retrieval and Local AI

**Current approach**

- Paper pages are split into chunks.
- User query is tokenized.
- Chunks are scored with BM25 as the main signal.
- Small reranking boosts use page hints, section hints, phrase hints, and lightweight semantic overlap.
- Top chunks are sent to Ollama as context.
- The model is instructed to answer only from paper context.
- Citations include page numbers and short quotes.

**Why this is useful for Milestone 1**

- Easy to explain.
- Easy to debug.
- Works without vector databases.
- Keeps paper content local.

**Speaker notes:**
This is not a full embedding-based RAG system yet. It is a transparent first version that proves the study assistant workflow.

## Slide 11: User Workflow Demo

**Demo steps**

1. Open the ScholAR home page.
2. Search for a topic like `retrieval augmented generation`.
3. Open a paper card.
4. Click **Study with AI**.
5. Wait for PDF preparation.
6. View the paper in the split workspace.
7. Open study goals.
8. Ask a question such as: `What is the main contribution of this paper?`
9. Show answer and cited references.

**Speaker notes:**
This slide can be used as the live demo script. If the model is slow, use an already prepared paper like `Attention Is All You Need`.

## Slide 12: Preliminary Results

**Tested flow**

- Search to paper selection works.
- PDF download and extraction works.
- Full paper page rendering works.
- Study goals are generated or fallback goals are shown.
- Chat returns answers grounded in retrieved chunks.
- Page citations are returned with answers.

**Example test paper**

- `Attention Is All You Need`
- Prepared locally.
- Pages extracted.
- Chunks generated.
- AI study chat tested with citation output.

## Slide 13: Current Limitations

- Retrieval is BM25-primary, not learned embedding-based.
- PDF extraction quality depends on the PDF text layer.
- Ollama can be slow on CPU-only machines.
- Study goal generation has a timeout and may use fallback goals.
- Storage is local JSON, not a multi-user database.
- PDF toolbar has some placeholder controls.
- No deployment or authentication yet.

**Speaker notes:**
These limitations are acceptable for Milestone 1 because the goal was to build a working MVP and identify what needs improvement next.

## Slide 14: Next Steps

**Milestone 2 improvements**

- Test learned embeddings or reranking against the BM25-primary baseline.
- Improve citation accuracy and source highlighting.
- Add streaming chat responses.
- Add PDF text search and better navigation.
- Add a proper database for saved papers and sessions.
- Add evaluation questions for study goals.
- Test with more research papers.
- Prepare deployment path after local MVP is stable.

## Slide 15: Conclusion

**Milestone 1 outcome**

- ScholAR now has a working local-first research paper study workflow.
- The app can search arXiv, prepare papers, render PDFs, generate study goals, and answer grounded questions.
- The architecture is simple, modular, and ready for better retrieval and evaluation in the next milestone.

**Closing line:**
ScholAR proves that a local AI assistant can turn a research paper from a static PDF into an interactive study session.
