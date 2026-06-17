# Domain Note

## 1. Real Problem

Research papers are dense, long, and full of assumptions. Students, engineers, and researchers often need to understand a paper quickly enough to decide whether it is worth deeper reading, but the path from abstract to implementation is rarely direct.

## 2. Who It Affects

This affects graduate students, independent researchers, applied AI engineers, startup teams, and practitioners who need to read unfamiliar papers under time pressure. It is especially painful for people working across fields, where terminology and evaluation norms change from paper to paper.

## 3. Why Existing Solutions Fall Short

Generic PDF readers show the document but do not help structure learning. General chatbots can summarize uploaded text, but they often hide source grounding, require cloud upload, or produce explanations that drift away from the paper. Academic search tools help discovery, but they usually stop at metadata and citations instead of turning a selected paper into a guided study session.

## 4. Proposed Approach

The project combines arXiv search, local PDF processing, simple retrieval, and a local Ollama Qwen model. A user searches for a paper, prepares it locally, views the PDF, generates study goals, and asks questions answered only from retrieved paper chunks with page citations.

## 5. Why The Approach Is Justified

Milestone 1 keeps the system explainable and private. arXiv search provides broad access to open research papers. PyMuPDF gives practical page-level extraction. Simple chunking and keyword retrieval are easy to inspect and debug. Ollama allows local inference without sending papers to a hosted LLM. This is enough to validate the core workflow before adding embeddings, better citation alignment, or advanced study features.
