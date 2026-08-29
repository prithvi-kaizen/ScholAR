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
    except Exception:
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


async def generate_stream(
    prompt: str,
    temperature: float = 0.2,
    images: list[str] | None = None,
    model: str | None = None,
):
    """Async generator yielding streaming response tokens from local Ollama instance."""
    payload: dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": _ollama_options(temperature),
    }
    if images:
        payload["images"] = images
    async with httpx.AsyncClient(timeout=float(os.getenv("OLLAMA_TIMEOUT", "240")), trust_env=False) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get("response", "")
                        if data.get("done", False):
                            break
                    except Exception:
                        continue


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


def fallback_goals(
    metadata: dict[str, Any] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    figures: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    paper_title = (metadata or {}).get("title") or "this paper"
    pages = _source_pages_from_chunks(chunks)
    summary = (metadata or {}).get("summary") or ""
    context = f"{summary} {_chunk_context(chunks)}"
    hint = _title_hint(paper_title)

    # Discover sections and figures
    sections = list(dict.fromkeys(
        c.get("section_title") for c in (chunks or [])
        if c.get("section_title") and c.get("section_title") not in ("Body", "Abstract")
    ))[:5]
    table_labels = [f.get("label") for f in (figures or []) if "table" in (f.get("label") or "").lower() or f.get("figure_type") == "table"]
    figure_labels = [f.get("label") for f in (figures or []) if "figure" in (f.get("label") or "").lower() or f.get("figure_type") == "figure"]

    table_ref = table_labels[0] if table_labels else "Key Tables"
    figure_ref = figure_labels[0] if figure_labels else "Figure 1"
    sec_ref = sections[0] if sections else "Methodology"

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

    goal_specs = [
        # Phase 1: Foundation
        {
            "phase": "Foundation",
            "difficulty": "Foundational",
            "title": f"Dissect the {hint} Problem & Failure Modes",
            "description": f"Analyze the core problem setup: {motivation} Study what limitations in previous methods the authors isolate, and why standard baselines fall short.",
            "estimated_minutes": 5,
            "target_evidence": ["Introduction", "Abstract"],
            "key_takeaways": ["Core bottleneck in prior approaches", "Specific motivation for this paper"],
            "subquestions": [
                f"What exact research gap or bottleneck does {hint} address?",
                "Why were prior state-of-the-art methods inadequate for this challenge?",
                "What assumptions does the paper make about the data or task setting?",
            ],
        },
        {
            "phase": "Foundation",
            "difficulty": "Foundational",
            "title": "Analyze the Core Contribution & Breakthrough Claims",
            "description": f"Examine the paper's primary hypothesis: {contribution} Distinguish the novel representation or architecture from standard building blocks.",
            "estimated_minutes": 6,
            "target_evidence": ["Section 1 & 2", figure_ref],
            "key_takeaways": ["Central theoretical or empirical claim", "Key conceptual novelty"],
            "subquestions": [
                "What is the single most important conceptual innovation introduced?",
                "How do the authors differentiate their work from closely related baselines?",
                "What primary performance or efficiency improvements are claimed?",
            ],
        },
        # Phase 2: Architecture
        {
            "phase": "Architecture",
            "difficulty": "Intermediate",
            "title": f"Trace Technical Architecture & Tensor Flow ({figure_ref})",
            "description": f"Break down the end-to-end mechanism: {method} Map inputs, intermediate layer transformations, attention or convolutional operations, and output heads.",
            "estimated_minutes": 8,
            "target_evidence": [figure_ref, sec_ref],
            "key_takeaways": ["Step-by-step layer transformations", "Mathematical mechanics and tensor flow"],
            "subquestions": [
                f"What are the precise inputs, layer operations, and outputs shown in {figure_ref}?",
                "How do tokens/features propagate through the forward pass?",
                "What role do positional encodings, normalization, or residual connections play?",
            ],
        },
        {
            "phase": "Architecture",
            "difficulty": "Advanced",
            "title": "Deconstruct Loss Functions & Training Objectives",
            "description": f"Examine optimization dynamics: {method} Understand the loss formulations, objective functions (e.g. cross-entropy, MLM, diffusion noise estimation), and optimization hyperparameters.",
            "estimated_minutes": 8,
            "target_evidence": ["Methodology", "Optimization"],
            "key_takeaways": ["Exact loss formulation", "Training objective tradeoff"],
            "subquestions": [
                "What exact loss function or objective is optimized during training?",
                "How are hyperparameters (learning rate schedule, warmups, batch size) configured?",
                "What regularization techniques prevent overfitting or instability?",
            ],
        },
        # Phase 3: Benchmarks
        {
            "phase": "Benchmarks",
            "difficulty": "Intermediate",
            "title": f"Evaluate Experimental Setup & Baselines ({table_ref})",
            "description": f"Audit the empirical testing protocol: {experiments} Verify benchmark datasets, metric formulations, baseline configurations, and dataset splits.",
            "estimated_minutes": 7,
            "target_evidence": [table_ref, "Experiments"],
            "key_takeaways": ["Benchmark datasets and evaluation metrics", "Fairness of baseline comparisons"],
            "subquestions": [
                "What datasets and benchmark suites are evaluated?",
                "What evaluation metrics (e.g. BLEU, Accuracy, F1, FID) are reported?",
                "Are the baseline comparisons matched in parameter count and training compute?",
            ],
        },
        {
            "phase": "Benchmarks",
            "difficulty": "Advanced",
            "title": f"Interpret Benchmark Gains & Ablation Deltas ({table_ref})",
            "description": f"Analyze quantitative score differentials: {results} Measure metric deltas across model variants, ablations, and compute-efficiency tradeoffs.",
            "estimated_minutes": 8,
            "target_evidence": [table_ref, "Ablations"],
            "key_takeaways": ["Statistically significant metric deltas", "Key findings from ablation studies"],
            "subquestions": [
                f"What are the top-line metric gains achieved in {table_ref}?",
                "Which component ablations cause the largest drops in performance?",
                "How does performance scale with model size and training tokens/compute?",
            ],
        },
        # Phase 4: Implementation
        {
            "phase": "Implementation",
            "difficulty": "Intermediate",
            "title": "Critique Assumptions, Failure Modes & Limitations",
            "description": f"Identify vulnerabilities and constraints: {limitations} Analyze computational costs, dataset biases, failure cases, and unproven claims.",
            "estimated_minutes": 6,
            "target_evidence": ["Limitations", "Discussion"],
            "key_takeaways": ["Scope constraints and edge cases", "Computational and memory overhead"],
            "subquestions": [
                "What explicit limitations or failure modes do the authors acknowledge?",
                "Under what data distributions or domain shifts might the model fail?",
                "What memory or compute bottlenecks restrict deployment in production?",
            ],
        },
        {
            "phase": "Implementation",
            "difficulty": "Advanced",
            "title": "Synthesize a Practical Engineering Build & Code Plan",
            "description": f"Translate research theory into an executable PyTorch implementation roadmap: Define data loaders, tensor modules, optimizer setup, validation tests, and scaling checkpoints.",
            "estimated_minutes": 10,
            "target_evidence": ["Method", "Appendix / Code"],
            "key_takeaways": ["PyTorch module architecture and tensor shapes", "Step-by-step reproduction recipe"],
            "subquestions": [
                "What PyTorch modules and tensor shapes are needed for the core layers?",
                "How should the training loop, loss calculation, and backprop be structured?",
                "What unit tests and sanity checks should be run before full-scale training?",
            ],
        },
    ]

    fallback_chunks = chunks or []
    return [
        {
            "id": f"goal_{index}",
            "title": spec["title"],
            "description": spec["description"],
            "phase": spec["phase"],
            "difficulty": spec["difficulty"],
            "estimated_minutes": spec["estimated_minutes"],
            "target_evidence": spec["target_evidence"],
            "key_takeaways": spec["key_takeaways"],
            "source_pages": pages,
            "subquestions": [
                {
                    "id": f"goal_{index}_q{sub_index}",
                    "question": q,
                    "evidence_chunks": _evidence_for_question(f"{spec['title']}. {spec['description']}. {q}", fallback_chunks),
                }
                for sub_index, q in enumerate(spec["subquestions"], start=1)
            ],
            "status": "not_started",
        }
        for index, spec in enumerate(goal_specs, start=1)
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
    figures: list[dict[str, Any]] | None = None,
    sections: list[str] | None = None,
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

    # Discover sections and figures
    if not sections:
        sections = list(dict.fromkeys(
            c.get("section_title") for c in chunks
            if c.get("section_title") and c.get("section_title") not in ("Body", "Abstract")
        ))[:6]

    sec_context = ", ".join(sections) if sections else "Introduction, Methodology, Experiments, Results, Discussion"

    fig_items = []
    for f in (figures or [])[:6]:
        lbl = f.get("label") or "Figure"
        cap = (f.get("caption") or "")[:80]
        fig_items.append(f"{lbl} ({cap})" if cap else lbl)
    fig_context = "; ".join(fig_items) if fig_items else "Key architecture diagrams and empirical result tables"

    prompt = f"""
You are designing a state-of-the-art guided research study curriculum for this specific paper.

Generate exactly 8 paper-specific study goals organized into 4 pedagogical learning phases:
- Phase 1 (Goals 1-2): "Foundation" -> Problem motivation, research gap, core breakthrough claims
- Phase 2 (Goals 3-4): "Architecture" -> Technical mechanism, tensor flow, loss formulations, algorithms (reference figures like {fig_context[:50]})
- Phase 3 (Goals 5-6): "Benchmarks" -> Experimental setups, datasets, metric tables, quantitative gains, ablations
- Phase 4 (Goals 7-8): "Implementation" -> Practical limitations, failure modes, PyTorch/build implementation recipe

Return ONLY a valid JSON array with exactly 8 objects.
Each object MUST contain:
- id: "goal_1" .. "goal_8"
- phase: "Foundation" | "Architecture" | "Benchmarks" | "Implementation"
- difficulty: "Foundational" | "Intermediate" | "Advanced"
- title: a concise, highly specific title tied to this paper's actual mechanisms/tables/equations
- description: 2 to 3 detailed sentences explaining what the reader should master
- estimated_minutes: integer (5, 6, 8, 10)
- target_evidence: array of strings naming sections or figures to inspect (e.g. ["Section 3.1", "Table 1"])
- key_takeaways: array of 2 short bullet points highlighting core takeaways
- subquestions: array of 3 to 4 precise research-reading questions
- source_pages: array of 1 to 3 integer page numbers from the provided context markers
- status: "not_started"

Paper Title: {metadata.get("title")}
Authors: {", ".join(metadata.get("authors", []))}
Abstract: {metadata.get("summary")}
Key Sections: {sec_context}
Key Visual Evidence: {fig_context}

Paper text context:
{context}
""".strip()

    response = await generate_func(prompt, temperature=0.1)
    parsed = _extract_json_array(response) or []
    goals: list[dict[str, Any]] = []
    
    phases_order = ["Foundation", "Foundation", "Architecture", "Architecture", "Benchmarks", "Benchmarks", "Implementation", "Implementation"]
    diff_order = ["Foundational", "Foundational", "Intermediate", "Advanced", "Intermediate", "Advanced", "Intermediate", "Advanced"]

    for index, item in enumerate(parsed[:8], start=1):
        if not isinstance(item, dict):
            continue
        source_pages = item.get("source_pages") or [1]
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        if not title or not description:
            continue

        assigned_phase = item.get("phase") or phases_order[index - 1]
        assigned_diff = item.get("difficulty") or diff_order[index - 1]
        est_min = item.get("estimated_minutes") or (6 if assigned_diff == "Foundational" else 8)
        target_ev = item.get("target_evidence") or [sections[0] if sections else "Methodology"]
        takeaways = item.get("key_takeaways") or ["Core theoretical mechanism", "Empirical insight"]

        goals.append(
            {
                "id": f"goal_{index}",
                "title": title,
                "description": description,
                "phase": assigned_phase,
                "difficulty": assigned_diff,
                "estimated_minutes": est_min,
                "target_evidence": target_ev if isinstance(target_ev, list) else [str(target_ev)],
                "key_takeaways": takeaways if isinstance(takeaways, list) else [str(takeaways)],
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
        defaults = fallback_goals(metadata, selected_chunks, figures)
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
