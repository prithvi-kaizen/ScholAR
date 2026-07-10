"use client";

import { FormEvent, useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  AlertTriangle, BookOpen, Camera,
  Send, Sparkles, Zap, FileText, GitCompare, BarChart2
} from "lucide-react";
import katex from "katex";
import type { ChatMessage, Citation } from "../types/paper";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

interface QueuedPrompt {
  id: number;
  text: string;
}

interface ChatBoxProps {
  paperId: string;
  queuedPrompt: QueuedPrompt | null;
  onCitationClick: (citation: Citation) => void;
  expanded: boolean;
  onChatActivity: () => void;
  secondaryPaperIds?: string[];
}

const SUGGESTED_QUESTIONS = [
  { icon: FileText,   text: "What is the main contribution of this paper?" },
  { icon: BarChart2,  text: "What does Figure 1 show?" },
  { icon: GitCompare, text: "How does this compare to prior work?" },
  { icon: Sparkles,   text: "Explain the methodology in simple terms." },
];

function TypingDots() {
  return (
    <span className="inline-flex items-end gap-[3px] h-4 ml-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="block h-1.5 w-1.5 rounded-full bg-zinc-400"
          style={{ animation: `scholar-dot 1.2s ease-in-out ${i * 0.2}s infinite` }}
        />
      ))}
    </span>
  );
}

function InlineMath({ expr }: { expr: string }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(expr, { throwOnError: false, output: "html", trust: false });
    } catch {
      return null;
    }
  }, [expr]);
  if (!html) return <span>${expr}$</span>;
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

// A single-$ span is only math if it looks like math: it contains a LaTeX
// control char, or has no internal whitespace (e.g. $x$, $d_k$, $\tau$).
// This keeps currency like "$50 million and $10" from being rendered as KaTeX.
function looksLikeMath(inner: string): boolean {
  if (/[\\^_{}]/.test(inner)) return true;
  return !/\s/.test(inner);
}

function renderInline(
  text: string,
  citations: Citation[] = [],
  onCitationClick?: (c: Citation) => void
) {
  const citByRef = new Map(
    citations.map((c, i) => [String(c.ref_id ?? i + 1), c])
  );
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d+\]|\$\$[^$\n]+\$\$|\$[^$\n]+\$)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("$$") && part.endsWith("$$") && part.length > 4) {
      return <InlineMath key={i} expr={part.slice(2, -2)} />;
    }
    if (part.startsWith("$") && part.endsWith("$") && part.length > 2 && looksLikeMath(part.slice(1, -1))) {
      return <InlineMath key={i} expr={part.slice(1, -1)} />;
    }
    const m = part.match(/^\[(\d+)\]$/);
    if (m) {
      const cit = citByRef.get(m[1]);
      if (!cit) return <span key={i}>{part}</span>;
      return (
        <button
          key={i}
          type="button"
          onClick={() => onCitationClick?.(cit)}
          className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded border border-blue-400/60 bg-blue-500/20 px-1.5 align-baseline text-[11px] font-semibold leading-none text-blue-100 transition hover:bg-blue-500/35"
          title={`Go to page ${cit.page}`}
        >
          {cit.ref_id ?? m[1]}
        </button>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function renderAnswer(
  content: string,
  citations: Citation[] = [],
  onCitationClick?: (c: Citation) => void
) {
  const cleaned = content.replace(/^#{1,6}\s*/gm, "").trim();
  const lines = cleaned.split(/\n+/).map((l) => l.trim()).filter(Boolean);
  return (
    <div className="space-y-2">
      {lines.map((line, i) => {
        const bullet  = line.match(/^[-*]\s+(.+)/);
        const numbered = line.match(/^\d+[.)]\s+(.+)/);
        const heading  = line.match(/^\*\*([^*]+)\*\*:?\s*(.*)$/);
        if (heading) return (
          <div key={i} className="pt-1">
            <div className="font-semibold text-white">{heading[1]}</div>
            {heading[2] && (
              <div className="mt-1 text-zinc-200">
                {renderInline(heading[2], citations, onCitationClick)}
              </div>
            )}
          </div>
        );
        if (bullet || numbered) return (
          <div key={i} className="flex gap-2 text-zinc-200">
            <span className="mt-[0.65em] h-1.5 w-1.5 shrink-0 rounded-full bg-acid/80" />
            <span>{renderInline((bullet?.[1] ?? numbered?.[1] ?? line).trim(), citations, onCitationClick)}</span>
          </div>
        );
        return <p key={i} className="text-zinc-200">{renderInline(line, citations, onCitationClick)}</p>;
      })}
    </div>
  );
}

function quotePreview(q: string) {
  return q.length > 145 ? q.slice(0, 145) + "…" : q;
}

function sectionLabel(c: Citation) {
  if (c.is_figure) return c.label || c.section_title || "Figure";
  return c.section_title || c.chunk_type || "Paper";
}

function ModelBadge({ model, vision }: { model?: string; vision?: boolean }) {
  if (!model && !vision) return null;
  return (
    <span className={`ml-2 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
      vision ? "bg-teal-500/15 text-teal-400" : "bg-zinc-700 text-zinc-400"
    }`}>
      {vision ? <><Camera size={9} />Vision</> : <><Zap size={9} />{model}</>}
    </span>
  );
}

function FigureThumbnail({ imageUrl, label, caption, onClick }: {
  imageUrl: string; label: string; caption: string; onClick?: () => void;
}) {
  const [loaded, setLoaded] = useState(false);
  const [error,  setError]  = useState(false);
  return (
    <div className="mt-3 overflow-hidden rounded-xl border border-teal-500/30 bg-teal-900/10">
      <div className="flex items-center gap-2 border-b border-teal-500/20 px-3 py-2">
        <Camera size={13} className="shrink-0 text-teal-400" />
        <span className="text-xs font-semibold text-teal-300">{label}</span>
      </div>
      {!error ? (
        <button
          type="button"
          onClick={onClick}
          className="block w-full text-left transition-opacity hover:opacity-90"
          title={`View ${label}`}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`${backendUrl}${imageUrl}`}
            alt={label}
            onLoad={() => setLoaded(true)}
            onError={() => setError(true)}
            className={`w-full max-h-64 bg-white/5 object-contain transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
          />
        </button>
      ) : (
        <div className="px-3 py-2 text-xs text-zinc-500">Image unavailable</div>
      )}
      {caption && (
        <p className="border-t border-teal-500/10 px-3 py-2 text-[11px] leading-relaxed text-zinc-400">
          {caption.length > 200 ? caption.slice(0, 200) + "…" : caption}
        </p>
      )}
    </div>
  );
}

export function ChatBox({
  paperId,
  queuedPrompt,
  onCitationClick,
  expanded,
  onChatActivity,
  secondaryPaperIds = [],
}: ChatBoxProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const handledPrompt = useRef<number | null>(null);
  const messagesRef   = useRef<HTMLDivElement | null>(null);
  const inputRef      = useRef<HTMLInputElement | null>(null);

  const sendMessage = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = { role: "user", content: trimmed };
    onChatActivity();
    const history = [...messages, userMsg].map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          history,
          secondary_paper_ids: secondaryPaperIds,
        }),
      });
      if (!res.ok) {
        const p = await res.json().catch(() => ({}));
        throw new Error(p.detail ?? "Chat failed");
      }
      const payload = await res.json();

      if (payload.error) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: payload.message ?? "The local model is unavailable.",
            error: true,
          },
        ]);
        return;
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: payload.answer,
          citations: payload.citations ?? [],
          vision: Boolean(payload.vision),
          vision_fallback: Boolean(payload.vision_fallback),
          figure_id: payload.figure_id ?? undefined,
          figure_label: payload.figure_label ?? undefined,
          figure_image_url: payload.figure_image_url ?? undefined,
          model: payload.model ?? undefined,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "Could not reach the study assistant.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [loading, messages, paperId, secondaryPaperIds, onChatActivity]);

  useEffect(() => {
    if (!queuedPrompt || handledPrompt.current === queuedPrompt.id || loading) return;
    handledPrompt.current = queuedPrompt.id;
    void sendMessage(queuedPrompt.text);
  }, [queuedPrompt, sendMessage, loading]);

  useEffect(() => {
    const node = messagesRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [messages, loading, expanded]);

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    void sendMessage(input);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void sendMessage(input);
    }
  }

  const isEmpty = messages.length === 0 && !loading;

  return (
    <div className={`min-h-0 border-t border-line bg-ink p-4 ${expanded ? "flex flex-1 flex-col" : ""}`}>
      {/* Scrollable message area */}
      <div
        ref={messagesRef}
        className={`mb-3 space-y-4 overflow-y-auto pr-1 ${expanded ? "min-h-0 flex-1" : "max-h-72"}`}
      >
        {/* ── Empty state ── */}
        {isEmpty && (
          <div className="flex flex-col items-center gap-5 py-6 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-acid/10 ring-1 ring-acid/20">
              <Sparkles size={22} className="text-acid" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">Ask anything about this paper</p>
              <p className="mt-1 text-xs text-zinc-500">
                {secondaryPaperIds.length > 0
                  ? `Multi-doc mode active · ${secondaryPaperIds.length} reference${secondaryPaperIds.length > 1 ? "s" : ""} loaded`
                  : "Figures, tables, methods, comparisons — all grounded in the source"}
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2 w-full max-w-sm">
              {SUGGESTED_QUESTIONS.map(({ icon: Icon, text }) => (
                <button
                  key={text}
                  type="button"
                  onClick={() => void sendMessage(text)}
                  className="flex items-start gap-2 rounded-lg border border-line bg-panel px-3 py-2 text-left text-xs text-zinc-300 transition hover:border-acid/40 hover:bg-acid/5 hover:text-white"
                >
                  <Icon size={13} className="mt-0.5 shrink-0 text-acid/70" />
                  <span className="leading-snug">{text}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Messages ── */}
        {messages.map((msg, idx) => (
          <div
            key={`${msg.role}-${idx}`}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            style={{ animation: "scholar-slide-in 0.18s ease-out" }}
          >
            <div
              className={`max-w-[92%] rounded-2xl border px-4 py-3 text-sm leading-6 ${
                msg.role === "user"
                  ? "rounded-br-sm border-acid/30 bg-acid/10 text-white"
                  : "rounded-bl-sm border-line bg-panel text-zinc-200"
              }`}
            >
              {/* Error */}
              {msg.error ? (
                <div className="flex items-start gap-2">
                  <AlertTriangle size={17} className="mt-1 shrink-0 text-amber-300" />
                  <div>
                    <div className="font-semibold text-white">Model response failed</div>
                    <p className="mt-1 text-zinc-300">{msg.content}</p>
                  </div>
                </div>
              ) : msg.role === "assistant" ? (
                <>
                  {/* Model badge */}
                  {(msg.model || msg.vision) && (
                    <div className="mb-2 flex items-center">
                      <ModelBadge model={msg.model} vision={msg.vision} />
                      {msg.vision_fallback && (
                        <span className="ml-2 text-[10px] text-amber-400">(caption fallback)</span>
                      )}
                    </div>
                  )}
                  {renderAnswer(msg.content, msg.citations ?? [], onCitationClick)}
                </>
              ) : (
                msg.content
              )}

              {/* Citations panel */}
              {(msg.citations?.length ?? 0) > 0 && (
                <div className="mt-4 rounded-xl border border-line bg-ink/70 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="inline-flex items-center gap-2 text-sm font-semibold text-zinc-200">
                      <BookOpen size={14} className="text-zinc-400" />
                      Sources
                    </div>
                    <span className="rounded-full bg-zinc-700 px-2 py-0.5 text-[11px] text-zinc-300">
                      {msg.citations!.length}
                    </span>
                  </div>
                  <div className="mt-3 space-y-2">
                    {msg.citations!.map((cit, ci) => (
                      <div key={`${cit.chunk_id}-${cit.page}-${ci}`} className="space-y-1">
                        <div className="flex items-center gap-1.5">
                          {cit.is_figure && <Camera size={11} className="text-teal-400 shrink-0" />}
                          <span className={`text-[11px] font-medium ${cit.is_figure ? "text-teal-400" : "text-zinc-400"}`}>
                            {sectionLabel(cit)}
                          </span>
                          <span className="text-[11px] text-zinc-600">· p.{cit.page}</span>
                          {cit.source_paper_id && cit.source_paper_id !== paperId && (
                            <span className="rounded bg-acid/20 px-1.5 py-0.5 text-[10px] font-semibold text-acid">ref</span>
                          )}
                        </div>
                        {cit.is_figure && cit.image_file && msg.figure_image_url ? (
                          <FigureThumbnail
                            imageUrl={msg.figure_image_url}
                            label={cit.label || sectionLabel(cit)}
                            caption={cit.caption || cit.quote}
                            onClick={() => onCitationClick(cit)}
                          />
                        ) : (
                          <button
                            onClick={() => onCitationClick(cit)}
                            className="flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-zinc-400 transition hover:bg-blue-500/10 hover:text-blue-100"
                          >
                            <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-blue-400" />
                            <span>
                              <span className="mr-1 font-semibold text-blue-200">[{cit.ref_id ?? ci + 1}]</span>
                              {quotePreview(cit.quote)}
                            </span>
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>
          </div>
        ))}

        {/* ── Typing indicator ── */}
        {loading && (
          <div className="flex justify-start">
            <div className="flex items-center gap-2 rounded-2xl rounded-bl-sm border border-line bg-panel px-4 py-3 text-sm text-zinc-400">
              <span>Thinking locally</span>
              <TypingDots />
            </div>
          </div>
        )}
      </div>

      {/* ── Input bar ── */}
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2 rounded-xl border border-line bg-panel px-3 py-2 transition focus-within:border-zinc-500"
      >
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything…"
          className="min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-zinc-500"
        />
        <span className="hidden rounded-md border border-line px-2 py-1 text-[11px] text-zinc-600 sm:inline">
          Local · ⌘↵
        </span>
        <button
          disabled={loading || !input.trim()}
          className="flex h-8 w-8 items-center justify-center rounded-lg bg-acid text-black transition hover:bg-acid/80 disabled:opacity-40"
          aria-label="Send message"
        >
          <Send size={15} />
        </button>
      </form>
    </div>
  );
}
