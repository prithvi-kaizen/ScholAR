"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { Loader2, Send } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "../types/paper";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

interface QueuedPrompt {
  id: number;
  text: string;
}

interface ChatBoxProps {
  paperId: string;
  queuedPrompt: QueuedPrompt | null;
}

export function ChatBox({ paperId, queuedPrompt }: ChatBoxProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const handledPrompt = useRef<number | null>(null);

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const userMessage: ChatMessage = { role: "user", content: trimmed };
    const history = [...messages, userMessage].map((message) => ({
      role: message.role,
      content: message.content
    }));
    setMessages((current) => [...current, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, history })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "Chat failed");
      }
      const payload = await response.json();
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: payload.answer,
          citations: payload.citations ?? []
        }
      ]);
    } catch (caught) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: caught instanceof Error ? caught.message : "Could not reach the local study assistant."
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!queuedPrompt || handledPrompt.current === queuedPrompt.id) return;
    handledPrompt.current = queuedPrompt.id;
    void sendMessage(queuedPrompt.text);
  }, [queuedPrompt]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage(input);
  }

  return (
    <div className="border-t border-line bg-ink p-4">
      <div className="mb-3 max-h-72 space-y-3 overflow-y-auto pr-1">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={message.role === "user" ? "text-right" : "text-left"}>
            <div
              className={`inline-block max-w-[92%] rounded-lg border px-3 py-2 text-sm leading-6 ${
                message.role === "user"
                  ? "border-acid/30 bg-acid/10 text-white"
                  : "border-line bg-panel text-zinc-200"
              }`}
            >
              {message.role === "assistant" ? (
  <div className="prose prose-invert prose-sm max-w-none">
    <ReactMarkdown>{message.content}</ReactMarkdown>
  </div>
) : (
  message.content
)}
              {message.citations?.length ? (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {message.citations.map((citation) => (
                    <span key={`${citation.chunk_id}-${citation.page}`} className="rounded-md border border-line px-2 py-0.5 text-xs text-zinc-400">
                      p. {citation.page}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        ))}
        {loading ? (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Loader2 size={16} className="animate-spin" />
            Thinking locally with Qwen...
          </div>
        ) : null}
      </div>
      <form onSubmit={handleSubmit} className="flex items-center gap-2 rounded-lg border border-line bg-panel px-3 py-2">
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask anything..."
          className="min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-zinc-500"
        />
        <span className="hidden rounded-md border border-line px-2 py-1 text-xs text-zinc-500 sm:inline">local Qwen</span>
        <button disabled={loading} className="rounded-md bg-white p-2 text-black transition hover:bg-acid disabled:opacity-60" aria-label="Send message">
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}