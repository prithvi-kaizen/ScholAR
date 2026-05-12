from __future__ import annotations

from backend.services.ollama_service import generate

MAX_DEPTH = 3
CONFIDENCE_THRESHOLD = 0.7


async def _score_confidence(answer: str) -> float:
    """Ask the model to rate its own confidence from 0.0 to 1.0."""
    prompt = (
        "Rate your confidence in the following answer from 0.0 to 1.0. "
        "Reply with just a single decimal number and nothing else.\n\n"
        f"Answer: {answer}"
    )
    try:
        raw = await generate(prompt)
        score = float(raw.strip().split()[0])
        return max(0.0, min(1.0, score))  # clamp to [0, 1]
    except (ValueError, IndexError):
        # if model doesn't return a clean number, assume uncertain
        return 0.5


async def recursive_generate(
    prompt: str,
    context: str,
    question: str,
    all_chunks: list[dict] | None = None,
    history: list[dict] | None = None,
    depth: int = 0,
    previous_answer: str = "",
) -> tuple[str, int, float]:
    """
    Recursively refine an answer up to MAX_DEPTH times.
    - Uses model-scored confidence instead of keyword matching
    - Stops early if the answer isn't changing
    - Re-retrieves chunks based on the previous answer
    - Accepts conversation history for better grounding
    Returns (answer, depth_reached, final_confidence_score).
    """
    print(f"[RLM] depth={depth}")

    if depth >= MAX_DEPTH:
        answer = await generate(prompt)
        confidence = await _score_confidence(answer)
        print(f"[RLM] depth={depth} | MAX DEPTH reached | confidence={confidence:.2f} | preview='{answer[:80]}...'")
        return answer, depth, confidence

    # Prepend conversation history to the prompt if available
    history_block = ""
    if history:
        history_block = "\n".join(
            f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
            for turn in history[-4:]  # last 4 turns only
        )
        prompt = f"Conversation so far:\n{history_block}\n\n{prompt}"

    answer = await generate(prompt)
    confidence = await _score_confidence(answer)

    print(f"[RLM] depth={depth} | confident={confidence >= CONFIDENCE_THRESHOLD} | confidence={confidence:.2f} | preview='{answer[:80]}...'")

    print("=== HISTORY ===")
    print(history)

    # Stop if confident enough
    if confidence >= CONFIDENCE_THRESHOLD:
        return answer, depth, confidence

    # Stop if answer hasn't changed — no point looping
    if previous_answer and answer.strip() == previous_answer.strip():
        print(f"[RLM] depth={depth} | answer unchanged, stopping early")
        return answer, depth, confidence

    # Re-retrieve chunks based on the previous answer for better context
    new_context = context
    if all_chunks:
        from backend.services.retrieval_service import retrieve_chunks
        new_chunks = retrieve_chunks(answer, all_chunks, limit=4)
        if new_chunks:
            new_context = "\n\n".join(
                f"[{chunk.get('chunk_id')} | p. {chunk.get('page')}]\n{chunk.get('text', '')[:2400]}"
                for chunk in new_chunks
            )

    refined_prompt = f"""
You previously gave this answer (confidence score: {confidence:.2f}):
\"\"\"{answer}\"\"\"

It seems incomplete. Using the updated paper context below, try again with deeper reasoning.
Be more specific and cite page numbers like [p. N].

Context:
{new_context}

Question: {question}
""".strip()

    return await recursive_generate(
        prompt=refined_prompt,
        context=new_context,
        question=question,
        all_chunks=all_chunks,
        history=history,
        depth=depth + 1,
        previous_answer=answer,
    )