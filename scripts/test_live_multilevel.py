import asyncio
import sys
from backend.main import ChatInput, _answer_request_from_chat
from backend.services.answer_pipeline import AnswerPipelineService

async def main():
    query = (
        "How does isolating attention to a single head compare to using multiple parallel heads "
        "in terms of translation quality, and what structural limitation in the parallel projection "
        "mechanism explains why single-head attention underperforms?"
    )
    print("=== Testing Query (No Table/Figure Numbers Mentioned) ===", flush=True)
    print("Query:", query, flush=True)

    chat_input = ChatInput(message=query, model="qwen3.5:9b")
    req = _answer_request_from_chat("1706.03762", chat_input)

    print("\nExecuting AnswerPipelineService.answer...", flush=True)
    trace = await AnswerPipelineService.answer(req)
    resp = trace.to_chat_response()

    print("\n=== STATUS ===", resp.get("status") or trace.status.value, flush=True)
    print("Reasoning level:", resp.get("reasoning_level"), flush=True)
    print("Latency ms:", trace.latency_ms, flush=True)
    print("\n=== FINAL ANSWER ===\n", resp.get("answer") or trace.final_answer, flush=True)

    print("\n=== CITATIONS ===", flush=True)
    for c in resp.get("citations", []):
        print(f"  [{c.get('marker')}] {c.get('label')} (page {c.get('page')}) - {c.get('quote', '')[:70]}", flush=True)

    print("\n=== SELECTED VISUAL CROPS ===", flush=True)
    meta = resp.get("figure_label") or trace.response_metadata.get("figure_label")
    print("Figures/Tables passed to Vision Model:", meta, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
