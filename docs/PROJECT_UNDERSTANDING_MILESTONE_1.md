# ScholAR Project Understanding: Milestone 1

## 1. Project Summary

ScholAR is a local-first research paper assistant. It helps a user search for research papers from arXiv, select a paper, download and process its PDF, view the full paper in a browser, generate study goals, and ask questions using a local Ollama/Qwen model.

The project is inspired by paperbreakdown.com, but Milestone 1 focuses on building a simple working MVP instead of a full production platform.

## 2. Problem Being Solved

Research papers are difficult to study because they are long, technical, and often assume that the reader already understands the background. A normal PDF reader only displays the document. It does not help the user understand the problem, method, experiments, limitations, or implementation direction.

ScholAR solves this by turning a paper into an interactive study workspace:

1. The user finds a paper.
2. The system prepares the paper locally.
3. The user reads the PDF.
4. The assistant provides study goals.
5. The user asks paper-specific questions.
6. The assistant answers using extracted paper context and cited evidence.

## 3. Milestone 1 Objective

The objective of Milestone 1 is to prove the complete end-to-end workflow:

```text
Search paper -> Select paper -> Download PDF -> Extract text -> Chunk text
-> View PDF -> Generate study goals -> Ask grounded questions
```

This milestone is intentionally local-first and simple. It does not include login, cloud deployment, advanced recommendations, fine-tuning, or a production database.

## 4. Technology Stack

### Frontend

- Next.js
- TypeScript
- Tailwind CSS
- Lucide React icons
- Browser localStorage

### Backend

- FastAPI
- Pydantic
- httpx
- PyMuPDF
- python-dotenv
- Local JSON storage

### AI Model Layer

- Ollama
- Default model: `qwen3:9b`
- Optional model override through `OLLAMA_MODEL`

### External Source

- arXiv public Atom API

## 5. Repository Structure

```text
project-paperbreakdown/
  README.md
  requirements.txt
  backend/
    main.py
    services/
      arxiv_service.py
      pdf_service.py
      chunking_service.py
      retrieval_service.py
      ollama_service.py
    data/
      papers/
    results/
      preliminary_results.md
  frontend/
    app/
      page.tsx
      paper/[id]/page.tsx
      layout.tsx
      globals.css
    components/
      SearchBar.tsx
      PaperCard.tsx
      PaperModal.tsx
      PdfViewer.tsx
      StudyPanel.tsx
      StudyGoals.tsx
      ChatBox.tsx
    types/
      paper.ts
    package.json
  output/
    pdf/
      ScholAR_Milestone_1.md
      ScholAR_Milestone_1.pdf
  docs/
    MILESTONE_1_PPT_CONTENT.md
    PROJECT_UNDERSTANDING_MILESTONE_1.md
```

## 6. High-Level Architecture

```text
                        +----------------------+
                        |        User          |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |   Next.js Frontend   |
                        |  Search + Study UI   |
                        +----------+-----------+
                                   |
                      HTTP requests to backend API
                                   |
                                   v
                        +----------------------+
                        |   FastAPI Backend    |
                        +----------+-----------+
                                   |
        +--------------------------+--------------------------+
        |                          |                          |
        v                          v                          v
+----------------+        +----------------+        +----------------+
| arXiv Service  |        | PDF Service    |        | Ollama Service |
| Search papers  |        | Download/render|        | Local LLM      |
+----------------+        +----------------+        +----------------+
        |                          |
        v                          v
+----------------+        +----------------+
| Search Cache   |        | Local Paper    |
| JSON           |        | JSON + PDF     |
+----------------+        +----------------+
```

## 7. Frontend Flow

### 7.1 Home Page

File: `frontend/app/page.tsx`

The home page is the paper discovery screen. It includes:

- Search bar.
- Editor's picks.
- Popular and New tabs.
- Paper cards.
- Recently viewed papers.
- Paper details modal.
- Local bookmarks.

Important behavior:

1. User enters a query.
2. Frontend calls `GET /api/search`.
3. Backend returns arXiv paper metadata.
4. Results are displayed as paper cards.
5. User clicks a paper card to open the modal.

### 7.2 Paper Modal

File: `frontend/components/PaperModal.tsx`

The paper modal shows:

- Paper title.
- Authors.
- Year.
- Categories.
- Abstract.
- Bookmark button.
- Share button.
- arXiv page link.
- **Study with AI** button.

When the user clicks **Study with AI**:

1. Frontend sends the selected paper metadata to `POST /api/papers/prepare`.
2. Backend downloads and processes the PDF.
3. Backend returns a local `paper_id`.
4. Frontend navigates to `/paper/{paper_id}`.

### 7.3 Study Workspace

File: `frontend/app/paper/[id]/page.tsx`

The study workspace is split into two columns:

- Left side: PDF viewer.
- Right side: AI study panel.

This is the main reading experience.

### 7.4 PDF Viewer

File: `frontend/components/PdfViewer.tsx`

The PDF viewer:

- Loads paper metadata from `GET /api/papers/{paper_id}`.
- Gets total page count.
- Renders every PDF page as a PNG image.
- Supports previous/next page navigation.
- Tracks current page while scrolling.
- Supports zoom in and zoom out.

Each page image is loaded from:

```text
GET /api/papers/{paper_id}/page/{page_number}.png
```

### 7.5 Study Panel

File: `frontend/components/StudyPanel.tsx`

The study panel includes:

- Study goals tab.
- Quick start tab.
- Chat box.
- Local Qwen session label.

When the panel loads:

1. It shows default study goals.
2. It calls `POST /api/papers/{paper_id}/study-goals`.
3. If Ollama is available, backend generates paper-specific goals.
4. If Ollama is unavailable or slow, fallback goals are shown.

### 7.6 Chat Box

File: `frontend/components/ChatBox.tsx`

The chat box:

- Accepts user questions.
- Sends questions to `POST /api/papers/{paper_id}/chat`.
- Displays assistant answers.
- Displays returned page citation badges.
- Handles loading and error states.

## 8. Backend Flow

File: `backend/main.py`

The backend exposes the API used by the frontend. It coordinates search, paper preparation, PDF rendering, study goals, and chat.

### 8.1 Health Check

Endpoint:

```text
GET /health
```

Returns:

- Backend status.
- Whether Ollama is available.
- Current model name.

### 8.2 Search Papers

Endpoint:

```text
GET /api/search?q={query}&max_results=12
```

Service:

```text
backend/services/arxiv_service.py
```

Search process:

1. Clean and tokenize the query.
2. Build arXiv search queries.
3. Rate-limit requests to avoid arXiv throttling.
4. Parse arXiv Atom XML.
5. Rerank papers locally.
6. Use search cache when possible.
7. Use fallback papers if arXiv is unavailable and the query matches known papers.

Returned metadata includes:

- `id`
- `title`
- `authors`
- `year`
- `published`
- `summary`
- `categories`
- `pdf_url`
- `abs_url`

### 8.3 Prepare Paper

Endpoint:

```text
POST /api/papers/prepare
```

Services:

```text
backend/services/pdf_service.py
backend/services/chunking_service.py
```

Input:

- Paper metadata from the frontend.

Process:

1. Create a safe local paper ID.
2. Create a folder under `backend/data/papers/{paper_id}`.
3. Save paper metadata as `metadata.json`.
4. Download the PDF as `paper.pdf`.
5. Extract text page by page using PyMuPDF.
6. Save extracted pages as `pages.json`.
7. Split extracted text into chunks.
8. Save chunks as `chunks.json`.

Output:

```json
{
  "paper_id": "1706.03762",
  "metadata": {},
  "pages": 15,
  "chunks": 15
}
```

### 8.4 Local Paper Storage

Each prepared paper is stored as:

```text
backend/data/papers/{paper_id}/
  metadata.json
  paper.pdf
  pages.json
  chunks.json
  goals.json
```

`goals.json` is created after study goals are generated. If study goals already exist, the backend loads them instead of regenerating them.

### 8.5 PDF Text Extraction

Service:

```text
backend/services/pdf_service.py
```

PyMuPDF opens the downloaded PDF and extracts text page by page. Each page is stored with:

```json
{
  "page": 1,
  "text": "Extracted page text..."
}
```

This page-wise structure is important because citations need page numbers.

### 8.6 PDF Page Rendering

Endpoint:

```text
GET /api/papers/{paper_id}/page/{page_number}.png
```

The backend renders a requested PDF page into PNG bytes using PyMuPDF. The frontend displays these PNG pages in the PDF viewer.

This approach avoids relying on browser PDF plugins and gives a consistent reading view.

### 8.7 Chunking

Service:

```text
backend/services/chunking_service.py
```

Chunking method:

- Each page is split into word chunks.
- Target chunk size is about 1400 words.
- Overlap is about 120 words.
- Page number is preserved for every chunk.

Example chunk:

```json
{
  "chunk_id": "chunk_001",
  "page": 1,
  "text": "Chunk text...",
  "char_start": 0,
  "char_end": 5000
}
```

### 8.8 Retrieval

Service:

```text
backend/services/retrieval_service.py
```

Current retrieval is BM25-primary:

1. Tokenize user question.
2. Remove stop words.
3. Tokenize every chunk.
4. Compute BM25-style lexical scores.
5. Apply small reranking boosts for page hints, section hints, phrase hints, and lightweight semantic overlap.
6. Select the top chunks for the model prompt.

This is not a learned embedding retrieval system yet. It is intentionally BM25-first because the project evaluation showed BM25 was the most reliable tested grounding method.

### 8.9 Study Goals

Endpoint:

```text
POST /api/papers/{paper_id}/study-goals
```

Service:

```text
backend/services/ollama_service.py
```

Flow:

1. Load paper metadata and chunks.
2. If `goals.json` exists, return saved goals.
3. Check if Ollama is available.
4. If Ollama is available, ask the local model to generate exactly 8 study goals.
5. Save generated goals to `goals.json`.
6. If Ollama is unavailable or times out, return fallback goals.

Fallback goals include:

- Define problem and motivation.
- Summarize core idea.
- Explain methodology.
- Identify algorithm or architecture.
- Understand experimental setup.
- Report key results.
- Discuss limitations.
- Convert paper into implementation plan.

### 8.10 Chat

Endpoint:

```text
POST /api/papers/{paper_id}/chat
```

Input:

```json
{
  "message": "What is the main contribution?",
  "history": []
}
```

Flow:

1. Validate that the message is not empty.
2. Load `chunks.json`.
3. Retrieve top matching chunks.
4. Build citations from selected chunks.
5. Build a grounded prompt for Ollama.
6. Ask Ollama to answer only from the provided context.
7. Return the answer and citations.
8. If Ollama is unavailable, return an extractive fallback answer from retrieved chunks.

Output:

```json
{
  "answer": "The paper proposes...",
  "citations": [
    {
      "page": 2,
      "chunk_id": "chunk_003",
      "quote": "Relevant sentence..."
    }
  ]
}
```

## 9. Complete User Journey

### Step 1: Start Backend

```bash
cd project-paperbreakdown
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Start Frontend

```bash
cd project-paperbreakdown/frontend
npm run dev
```

Open:

```text
http://localhost:3000
```

### Step 3: Start Ollama

```bash
ollama serve
ollama pull qwen3:9b
```

Optional smaller model:

```bash
ollama pull qwen2.5:7b
```

### Step 4: Search Paper

Example query:

```text
retrieval augmented generation survey
```

The frontend calls the backend search endpoint and displays arXiv results.

### Step 5: Open Paper Modal

Click any paper card to view:

- Abstract.
- Authors.
- Categories.
- arXiv link.
- Bookmark and share controls.
- Study with AI action.

### Step 6: Prepare Paper

Click **Study with AI**.

The backend:

- Downloads the PDF.
- Extracts text.
- Splits text into chunks.
- Saves all local files.
- Returns the local paper ID.

### Step 7: Study Paper

The app opens:

```text
/paper/{paper_id}
```

The user sees:

- Rendered PDF pages.
- Study goals.
- Quick-start actions.
- AI chat.

### Step 8: Ask Questions

Example questions:

- What is the main contribution of this paper?
- Explain the methodology in simple terms.
- What datasets and metrics are used?
- What are the limitations?
- How can I implement this paper?

The assistant answers using retrieved chunks and cited evidence.

## 10. Data and Control Flow

```text
Frontend SearchBar
  -> GET /api/search
  -> arXiv API
  -> Search results returned
  -> Paper cards displayed

PaperModal Study with AI
  -> POST /api/papers/prepare
  -> PDF downloaded
  -> pages.json created
  -> chunks.json created
  -> frontend navigates to study page

Study page
  -> GET /api/papers/{paper_id}
  -> GET page PNGs
  -> POST /study-goals
  -> POST /chat
  -> answer + citations displayed
```

## 11. What Is Completed Till Now

Completed features:

- Functional backend API.
- arXiv paper search.
- arXiv search caching and fallback logic.
- PDF download.
- PDF validation.
- Page-wise text extraction.
- PDF page rendering as images.
- Chunk generation.
- Local JSON storage.
- Study goal generation using Ollama.
- Fallback study goals.
- Grounded chat endpoint.
- Extractive fallback chat answer.
- Search home page.
- Paper cards and modal.
- Recently viewed papers.
- Bookmark storage through localStorage.
- Full paper study page.
- Scrollable rendered PDF pages.
- Study goals panel.
- Quick start panel.
- Chat interface with citations.

## 12. Preliminary Result

The project has been tested with a known paper:

```text
Attention Is All You Need
```

Observed result:

- The paper can be found or selected.
- The PDF can be prepared locally.
- Text is extracted page by page.
- Chunks are generated.
- The PDF pages render in the browser.
- Study goals load.
- Questions return grounded responses with cited references.

## 13. Current Limitations

- Retrieval is BM25-primary instead of learned embedding-based retrieval.
- The model response depends on local Ollama speed and hardware.
- If Ollama is unavailable, the app uses fallback responses.
- PDF extraction depends on the quality of the PDF text layer.
- Local JSON storage is not suitable for multi-user production.
- Chat history is sent from the frontend but not deeply used in retrieval yet.
- PDF search input exists visually but is not fully implemented.
- No login, deployment, cloud sync, or user library database yet.

## 14. Why This Architecture Is Good for Milestone 1

The current architecture is good for Milestone 1 because:

- It proves the complete workflow.
- It is easy to run locally.
- It avoids cloud LLM privacy concerns.
- It is modular.
- Each service has a clear responsibility.
- The storage format is easy to inspect.
- The retrieval logic is transparent.
- It can be upgraded later without rewriting the whole app.

## 15. Next Milestone Plan

Recommended Milestone 2 tasks:

1. Expand the retrieval benchmark to more papers and query types.
2. Test whether learned embeddings or reranking improve over BM25 without lowering Recall@5.
3. Improve citation accuracy by linking answer spans to source pages.
4. Add streaming responses for better UX.
5. Implement PDF text search.
6. Add user notes and highlights.
7. Add paper library management.
8. Add evaluation questions after study goals.
9. Test with more papers and save benchmark results.
10. Prepare deployment strategy after local stability improves.

## 16. Presentation Demo Script

Use this flow during the milestone demo:

1. Show the home page and explain that ScholAR starts with arXiv paper discovery.
2. Search for `retrieval augmented generation`.
3. Open a paper card and explain the metadata shown.
4. Click **Study with AI** and explain that the backend downloads and processes the PDF locally.
5. Open the study workspace.
6. Show the PDF viewer and page navigation.
7. Show study goals and explain that they are generated by a local Qwen model or fallback logic.
8. Ask: `What is the main contribution of this paper?`
9. Show the answer and page citation badges.
10. Explain limitations and next milestone improvements.

## 17. Final Summary

Milestone 1 successfully implements a working local research paper assistant. The system can search arXiv, prepare a paper, render the PDF, generate study goals, and answer questions using local context. The project is not production-ready yet, but the foundation is strong and ready for better retrieval, citation quality, and study tools in the next milestone.
