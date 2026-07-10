from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
STUDY_GOAL_PROMPT_VERSION = "study-goals-v8-recursive"


def _ollama_options(temperature: float) -> dict[str, Any]:
    return {
        "temperature": temperature,
        "top_p": float(os.getenv("OLLAMA_TOP_P", "0.9")),
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16000")),
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "1650")),
}


async def ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def generate(prompt: str, temperature: float = 0.2, images: list[str] | None = None, model: str | None = None) -> str:
    payload: dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": _ollama_options(temperature),
    }
    if images:
        payload["images"] = images
    async with httpx.AsyncClient(timeout=float(os.getenv("OLLAMA_TIMEOUT", "240")), trust_env=False) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def _source_pages_from_chunks(chunks: list[dict[str, Any]] | None, fallback: int = 1) -> list[int]:
    pages = []
    for chunk in chunks or []:
        page = chunk.get("page")
        if isinstance(page, int) and page not in pages:
            pages.append(page)
        if len(pages) >= 3:
            break
    return pages or [fallback]


def _chunk_context(chunks: list[dict[str, Any]] | None, max_chars: int = 9000) -> str:
    text = " ".join(chunk.get("text", "") for chunk in chunks or [])
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if len(sentence.strip()) > 40]


def _find_sentence(text: str, patterns: tuple[str, ...], fallback: str) -> str:
    banned = (
        "permission",
        "copyright",
        "provided proper attribution",
        "journalistic or scholarly works",
        "arxiv:",
        "equal contribution",
    )
    for sentence in _sentences(text):
        lowered = sentence.lower()
        if any(term in lowered for term in banned):
            continue
        if any(pattern in lowered for pattern in patterns):
            return sentence[:360]
    return fallback[:360]


def _title_hint(title: str) -> str:
    if ":" in title:
        lead = title.split(":", 1)[0].strip()
        if len(lead) >= 8:
            return lead
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9-]+", title) if len(word) > 2]
    return " ".join(words[:7]) or "the paper"


def fallback_goals(metadata: dict[str, Any] | None = None, chunks: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    paper_title = (metadata or {}).get("title") or "this paper"
    pages = _source_pages_from_chunks(chunks)
    summary = (metadata or {}).get("summary") or ""
    context = f"{summary} {_chunk_context(chunks)}"
    hint = _title_hint(paper_title)

    motivation = _find_sentence(
        context,
        ("dominant sequence", "recurrent", "convolutional", "problem", "motivation", "challenge", "bottleneck", "constraint"),
        summary or f"Identify the research problem that motivates {paper_title}.",
    )
    contribution = _find_sentence(
        context,
        ("we propose", "we introduce", "we present", "new simple network architecture", "central contribution", "contribution"),
        summary or f"Identify the central contribution of {paper_title}.",
    )
    method = _find_sentence(
        context,
        ("probe battery", "framework", "architecture", "algorithm", "method", "model", "mechanism", "attention", "retrieval"),
        contribution,
    )
    experiments = _find_sentence(
        context,
        ("across", "evaluate", "experiment", "evaluation", "dataset", "benchmark", "baseline", "training", "truthfulqa", "nq-open"),
        "Locate the datasets, baselines, metrics, and evaluation protocol used by the paper.",
    )
    results = _find_sentence(
        context,
        ("mean", "rate", "result", "achieve", "outperform", "state-of-the-art", "bleu", "accuracy", "improve", "p <", "χ"),
        "Locate the paper's strongest quantitative or qualitative evidence.",
    )
    limitations = _find_sentence(
        context,
        ("limitation", "future work", "fail", "assumption", "however", "although", "not determine", "does not"),
        "Look for assumptions, scope limits, missing experiments, or cases where the method may not apply.",
    )

    goals = [
        (
            f"Understand the {hint} problem setting",
            f"Use this motivation as the anchor: {motivation} Study what failure mode or research gap the paper is isolating, why existing evaluation is insufficient, and what practical risk the authors want the reader to notice.",
        ),
        (
            f"Explain the paper's main contribution",
            f"Use the paper's own framing: {contribution} Identify the new framework, representation, model, or evaluation protocol, and separate the paper's actual contribution from background concepts it builds on.",
        ),
        (
            "Trace the method step by step",
            f"Use this method anchor: {method} Break the approach into inputs, transformations, decision rules, outputs, and validation checks so the reader can reproduce the reasoning without relying on vague summary language.",
        ),
        (
            "Map the framework components and data flow",
            f"Create a component map using this clue: {method} Name the important stages, how information moves between them, and where measurements, probes, metrics, or model/API calls enter the pipeline.",
        ),
        (
            "Inspect the experimental setup and controls",
            f"Use the evaluation context as your checklist: {experiments} Record datasets, models, prompts, metrics, baselines, thresholds, and control conditions, then explain why each is needed for a fair test.",
        ),
        (
            "Interpret the key quantitative findings",
            f"Ground the result summary in paper evidence: {results} Extract the important numbers or comparisons, explain what changed, and decide whether the evidence supports the paper's central claim.",
        ),
        (
            "Critique assumptions and limitations",
            f"Start with this limitation clue: {limitations} Separate limitations stated by the authors from additional concerns about scope, data, compute, measurement validity, external validity, or generalization.",
        ),
        (
            "Convert the paper into an implementation plan",
            f"Translate the paper into build steps grounded in its method and evaluation: {method} Define required inputs, preprocessing, model/API calls, scoring rules, tests, expected outputs, and failure cases to check before coding.",
        ),
    ]

    fallback_subquestions = [
        "What problem does this part of the paper solve?",
        "What are the main components or claims?",
        "What assumptions does it make?",
        "What evidence in the paper supports it?",
    ]

    fallback_chunks = chunks or []
    return [
        {
            "id": f"goal_{index}",
            "title": goal_title,
            "description": description,
            "source_pages": pages,
            "subquestions": [
                {
                    "id": f"goal_{index}_q{sub_index}",
                    "question": question,
                    "evidence_chunks": _evidence_for_question(f"{goal_title}. {description}. {question}", fallback_chunks),
                }
                for sub_index, question in enumerate(fallback_subquestions, start=1)
            ],
            "status": "not_started",
        }
        for index, (goal_title, description) in enumerate(goals, start=1)
    ]


def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _evidence_for_question(question: str, chunks: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    from backend.services.retrieval_service import extract_page_hints, retrieve_chunks, short_quote

    selected = retrieve_chunks(question, chunks, limit=limit, preferred_pages=extract_page_hints(question))
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "page": chunk.get("page"),
            "section_title": chunk.get("section_title", "Body"),
            "chunk_type": chunk.get("chunk_type", "body"),
            "quote": _clean_evidence_quote(short_quote(chunk, question, max_length=360), chunk),
        }
        for chunk in selected
    ]


def _clean_evidence_quote(quote: str, chunk: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", quote).strip()
    if chunk.get("chunk_type") == "abstract" and "Abstract" in chunk.get("text", ""):
        after_abstract = chunk.get("text", "").split("Abstract", 1)[-1]
        sentences = _sentences(after_abstract)
        if sentences:
            return sentences[0][:300]
    noisy_terms = ("patrick lewis", "facebook ai research", "university college london", "new york university")
    if any(term in text.lower() for term in noisy_terms):
        sentences = _sentences(chunk.get("text", ""))
        for sentence in sentences:
            lowered = sentence.lower()
            if not any(term in lowered for term in noisy_terms):
                return sentence[:300]
    return text[:300]


def _normalize_subquestions(goal_id: str, raw: Any, title: str, description: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    defaults = [
        f"What problem does {title} solve in this paper?",
        f"What are the main components or claims behind {title}?",
        f"What assumptions or design choices does this part rely on?",
        f"What evidence supports this part of the paper?",
    ]
    questions: list[str] = []
    if isinstance(raw, list):
        for item in raw[:5]:
            if isinstance(item, str):
                questions.append(item.strip())
            elif isinstance(item, dict):
                questions.append(str(item.get("question") or item.get("title") or "").strip())
    questions = [question for question in questions if question]
    if len(questions) < 3:
        questions.extend(defaults[len(questions) :])
    normalized = []
    for index, question in enumerate(questions[:5], start=1):
        evidence_query = f"{title}. {description}. {question}"
        normalized.append(
            {
                "id": f"{goal_id}_q{index}",
                "question": question,
                "evidence_chunks": _evidence_for_question(evidence_query, chunks),
            }
        )
    return normalized


async def generate_study_goals(
    metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
    generate_func=generate,
    provider: str = "local",
) -> list[dict[str, Any]]:
    selected_chunks = _select_study_goal_chunks(chunks)
    char_limit = 1800 if provider == "local" else 2600
    chunk_limit = 7 if provider == "local" else 10
    context = "\n\n".join(
        f"[chunk: {chunk.get('chunk_id')} | p. {chunk.get('page')}]\n{chunk.get('text', '')[:char_limit]}"
        for chunk in selected_chunks[:chunk_limit]
    )
    prompt = f"""
You are designing a serious guided study plan for one specific research paper.

Generate exactly 8 paper-specific study goals. These must NOT be generic checklist items.
Each goal should help the reader understand a concrete part of this paper: its problem, claims, method, mechanism, experiments, results, limitations, and implementation implications.

Return only a valid JSON array with exactly 8 objects.
Each object must contain:
- title: a short, specific goal title tied to this paper
- description: 2 to 4 detailed sentences explaining what the reader should learn, including paper-specific concepts, datasets, mechanisms, equations, architecture parts, claims, or results when available
- subquestions: 3 to 5 precise subquestions that recursively break the goal into smaller research-reading questions
- source_pages: 1 to 4 page numbers from the provided chunks where the reader should look first
- status: "not_started"

Rules:
- Use only the title, abstract, and paper context below.
- Do not use generic titles like "Explain methodology" unless you add the paper's actual method or concept.
- If the provided context is thin, say what should be verified in the paper instead of inventing facts.
- Page numbers must come from the context markers.
- Make the plan precise enough that clicking a goal can produce a useful tutoring explanation.
- Subquestions should ask about problem, components, assumptions, evidence, experiments, results, limitations, or implementation details depending on the goal.
- Do not include markdown fences or prose outside the JSON.

Title: {metadata.get("title")}
Authors: {", ".join(metadata.get("authors", []))}
Abstract: {metadata.get("summary")}

Paper context:
{context}
""".strip()

    response = await generate_func(prompt, temperature=0.1)
    parsed = _extract_json_array(response) or []
    goals: list[dict[str, Any]] = []
    for index, item in enumerate(parsed[:8], start=1):
        if not isinstance(item, dict):
            continue
        source_pages = item.get("source_pages") or [1]
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        if not title or not description:
            continue
        goals.append(
            {
                "id": f"goal_{index}",
                "title": title,
                "description": description,
                "source_pages": source_pages if isinstance(source_pages, list) else [source_pages],
                "subquestions": _normalize_subquestions(
                    f"goal_{index}",
                    item.get("subquestions"),
                    title,
                    description,
                    chunks,
                ),
                "status": "not_started",
            }
        )

    if len(goals) != 8:
        defaults = fallback_goals(metadata, selected_chunks)
        goals.extend(defaults[len(goals) :])
    return goals[:8]


def _select_study_goal_chunks(chunks: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    if not chunks:
        return []

    important_terms = (
        "abstract introduction method methodology approach architecture algorithm model training "
        "experiment evaluation results ablation limitation discussion conclusion dataset benchmark"
    ).split()
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "").lower()
        score = sum(1 for term in important_terms if term in text)
        if index < 3:
            score += 4
        if any(word in text for word in ("we propose", "we introduce", "our method", "our approach")):
            score += 3
        scored.append((score, -index, chunk))

    first_chunks = chunks[:3]
    ranked = [chunk for _, _, chunk in sorted(scored, reverse=True)]
    selected: list[dict[str, Any]] = []
    for chunk in first_chunks + ranked:
        chunk_id = chunk.get("chunk_id")
        if chunk_id not in {item.get("chunk_id") for item in selected}:
            selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected
