"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { AlertTriangle, BookOpen, Loader2, Send } from "lucide-react";
import type { ChatMessage, Citation } from "../types/paper";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

interface QueuedPrompt {
  id: number;
  text: string;
}

interface ChatBoxProps {
  paperId: string;
  queuedPrompt: QueuedPrompt | null;
  provider: "local" | "groq";
  onProviderChange: (provider: "local" | "groq") => void;
  onCitationClick: (citation: Citation) => void;
  expanded: boolean;
  onChatActivity: () => void;
}

function renderInline(text: string, citations: Citation[] = [], onCitationClick?: (citation: Citation) => void) {
  const citationByRef = new Map(citations.map((citation, index) => [String(citation.ref_id ?? index + 1), citation]));
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d+\])/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
    }
    const citationMatch = part.match(/^\[(\d+)\]$/);
    if (citationMatch) {
      const citation = citationByRef.get(citationMatch[1]);
      if (!citation) return <span key={index}>{part}</span>;
      return (
        <button
          key={index}
          type="button"
          onClick={() => onCitationClick?.(citation)}
          className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded border border-blue-400/60 bg-blue-500/20 px-1.5 align-baseline text-[11px] font-semibold leading-none text-blue-100 transition hover:bg-blue-500/35"
          title={`Open cited passage on page ${citation.page}`}
        >
          {citation.ref_id ?? citationMatch[1]}
        </button>
      );
    }
    return <span key={index}>{part}</span>;
  });
}

function renderAnswer(content: string, citations: Citation[] = [], onCitationClick?: (citation: Citation) => void) {
  const cleaned = content.replace(/^#{1,6}\s*/gm, "").trim();
  const lines = cleaned.split(/\n+/).map((line) => line.trim()).filter(Boolean);
  return (
    <div className="space-y-2">
      {lines.map((line, index) => {
        const bullet = line.match(/^[-*]\s+(.+)/);
        const numbered = line.match(/^\d+[.)]\s+(.+)/);
        const heading = line.match(/^\*\*([^*]+)\*\*:?\s*(.*)$/);

        if (heading) {
          return (
            <div key={index} className="pt-1">
              <div className="font-semibold text-white">{heading[1]}</div>
              {heading[2] ? <div className="mt-1 text-zinc-200">{renderInline(heading[2], citations, onCitationClick)}</div> : null}
            </div>
          );
        }

        if (bullet || numbered) {
          return (
            <div key={index} className="flex gap-2 text-zinc-200">
              <span className="mt-[0.65em] h-1.5 w-1.5 shrink-0 rounded-full bg-acid/80" />
              <span>{renderInline((bullet?.[1] ?? numbered?.[1] ?? line).trim(), citations, onCitationClick)}</span>
            </div>
          );
        }

        return <p key={index} className="text-zinc-200">{renderInline(line, citations, onCitationClick)}</p>;
      })}
    </div>
  );
}

function quotePreview(quote: string) {
  return quote.length > 145 ? `${quote.slice(0, 145)}...` : quote;
}

function sectionLabel(citation: Citation) {
  return citation.section_title || citation.chunk_type || "Paper";
}

export function ChatBox({ paperId, queuedPrompt, provider, onProviderChange, onCitationClick, expanded, onChatActivity }: ChatBoxProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const handledPrompt = useRef<number | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);

  async function sendMessage(text: string, providerOverride?: "local" | "groq") {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const activeProvider = providerOverride ?? provider;
    const userMessage: ChatMessage = { role: "user", content: trimmed };
    onChatActivity();
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
        body: JSON.stringify({ message: trimmed, history, provider: activeProvider, web_search: true })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "Chat failed");
      }
      const payload = await response.json();
      if (payload.provider_error) {
        setMessages((current) => [
          ...current,
          {
            role: "assistant",
            content: payload.message ?? "The selected provider is unavailable.",
            provider_error: payload.provider_error,
            retry_text: trimmed,
            web_results: payload.web_results ?? [],
            used_web_search: Boolean(payload.used_web_search)
          }
        ]);
        return;
      }
      const providerNote =
        payload.requested_provider && payload.provider && payload.requested_provider !== payload.provider
          ? `Using ${payload.model} because ${payload.requested_provider === "groq" ? "Groq API is not configured" : "the requested provider is unavailable"}.\n\n`
          : "";
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `${providerNote}${payload.answer}`,
          citations: payload.citations ?? [],
          web_results: payload.web_results ?? [],
          used_web_search: Boolean(payload.used_web_search)
        }
      ]);
    } catch (caught) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: caught instanceof Error ? caught.message : "Could not reach the study assistant."
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

  useEffect(() => {
    const node = messagesRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [messages, loading, expanded]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage(input);
  }

  return (
    <div className={`min-h-0 border-t border-line bg-ink p-4 ${expanded ? "flex flex-1 flex-col" : ""}`}>
      <div ref={messagesRef} className={`mb-3 space-y-3 overflow-y-auto pr-1 ${expanded ? "min-h-0 flex-1" : "max-h-72"}`}>
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={message.role === "user" ? "text-right" : "text-left"}>
            <div
              className={`inline-block max-w-[92%] rounded-lg border px-3 py-2 text-sm leading-6 ${
                message.role === "user"
                  ? "border-acid/30 bg-acid/10 text-white"
                  : "border-line bg-panel text-zinc-200"
              }`}
            >
              {message.provider_error ? (
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <AlertTriangle size={17} className="mt-1 shrink-0 text-amber-300" />
                    <div>
                      <div className="font-semibold text-white">
                        {message.provider_error === "groq_rate_limit" ? "Groq limit reached" : "Model response failed"}
                      </div>
                      <p className="mt-1 text-zinc-300">{message.content}</p>
                    </div>
                  </div>
                  {message.provider_error === "groq_rate_limit" && message.retry_text ? (
                    <button
                      type="button"
                      onClick={() => {
                        onProviderChange("local");
                        void sendMessage(message.retry_text ?? "", "local");
                      }}
                      className="rounded-md border border-acid/40 bg-acid/15 px-3 py-2 text-xs font-semibold text-acid transition hover:bg-acid hover:text-black"
                    >
                      Use Local Qwen and retry
                    </button>
                  ) : null}
                </div>
              ) : message.role === "assistant" ? renderAnswer(message.content, message.citations ?? [], onCitationClick) : message.content}
              {message.citations?.length ? (
                <div className="mt-4 rounded-lg border border-line bg-ink/70 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="inline-flex items-center gap-2 text-sm font-semibold text-zinc-200">
                      <BookOpen size={15} className="text-zinc-400" />
                      Cited references
                    </div>
                    <span className="text-xs text-zinc-500">{message.citations.length}</span>
                  </div>
                  <div className="mt-3 space-y-3">
                  {message.citations.map((citation) => (
                    <div key={`${citation.chunk_id}-${citation.page}-${citation.quote}`} className="space-y-1">
                      <div className="text-xs font-medium text-zinc-400">{sectionLabel(citation)}:</div>
                      <button
                        onClick={() => onCitationClick(citation)}
                        className="flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left text-xs text-zinc-400 transition hover:bg-blue-500/10 hover:text-blue-100"
                      >
                        <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-blue-400" />
                        <span>
                          <span className="mr-1 font-semibold text-blue-200">[{citation.ref_id ?? citation.page}]</span>
                          {quotePreview(citation.quote)}
                        </span>
                      </button>
                    </div>
                  ))}
                  </div>
                </div>
              ) : null}
              {message.web_results?.length ? (
                <div className="mt-3 space-y-1.5 border-t border-line pt-3">
                  <div className="text-xs font-medium uppercase tracking-[0.16em] text-zinc-500">Web sources</div>
                  {message.web_results.slice(0, 4).map((result, sourceIndex) => (
                    <a
                      key={`${result.id}-${result.url}`}
                      href={result.url}
                      target="_blank"
                      rel="noreferrer"
                      className="block rounded-md border border-line px-2 py-1.5 text-xs text-zinc-400 hover:border-zinc-500 hover:text-white"
                    >
                      [web:{sourceIndex + 1}] {result.title}
                    </a>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        ))}
        {loading ? (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Loader2 size={16} className="animate-spin" />
            {provider === "groq" ? "Thinking with Groq, searching web if needed..." : "Thinking locally with Qwen, searching web if needed..."}
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
        <span className="hidden rounded-md border border-line px-2 py-1 text-xs text-zinc-500 sm:inline">
          {provider === "groq" ? "Groq API + web" : "local Qwen + web"}
        </span>
        <button disabled={loading} className="rounded-md bg-white p-2 text-black transition hover:bg-acid disabled:opacity-60" aria-label="Send message">
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}
