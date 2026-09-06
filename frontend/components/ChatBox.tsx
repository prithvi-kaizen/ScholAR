"use client";

import { FormEvent, useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  AlertTriangle, BookOpen, Camera,
  Send, Sparkles, Zap, FileText, GitCompare, BarChart2,
  Copy, RotateCcw, Crop, X, Download,
  Layers, Network, Calculator, ShieldCheck, ChevronRight
} from "lucide-react";
import katex from "katex";
import type {
  ChatMessage,
  Citation,
  CustomSnippet,
  ReasoningPathStep,
} from "../types/paper";
import type { ChatRequest } from "../types/api";
import {
  getApiErrorMessage,
  isChatResponse,
  isExportReasoningResponse,
  isFigureListResponse,
} from "../types/api";

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
  activeSnippet?: CustomSnippet | null;
  onDismissSnippet?: () => void;
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

// Clean pseudo-LaTeX model names like $\text{BERT}_{\text{BASE}}$ into human-readable BERT-Base
function cleanLatexArtifacts(text: string): string {
  return text
    .replace(/\$\\text\{([^}]+)\}_\{?\\text\{([^}]+)\}?\}\$/g, "$1-$2")
    .replace(/\$\\text\{([^}]+)\}_\{([^}]+)\}\$/g, "$1-$2")
    .replace(/\$\\text\{([^}]+)\}\$/g, "$1")
    .replace(/\\text\{([^}]+)\}/g, "$1");
}

function looksLikeMath(inner: string): boolean {
  // If it is just plain words with no math symbols, do not treat as math
  if (/^[a-zA-Z0-9_\-\s]+$/.test(inner) && !/[=+\-*/\\^_{}<>]/.test(inner)) {
    return false;
  }
  if (/[\\^_{}<>=+\-*/]/.test(inner)) return true;
  return !/\s/.test(inner);
}

function renderInline(
  text: string,
  citations: Citation[] = [],
  onCitationClick?: (c: Citation) => void
): React.ReactNode[] {
  const normalizedText = cleanLatexArtifacts(text);
  const citByRef = new Map(
    citations.map((c, i) => [String(c.ref_id ?? i + 1), c])
  );
  const parts = normalizedText.split(/(\*\*[^*]+\*\*|\[\d+\]|\$\$[^$\n]+\$\$|\$[^$\n]+\$)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-white">
          {renderInline(part.slice(2, -2), citations, onCitationClick)}
        </strong>
      );
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

      let badgeClasses = "border-blue-400/60 bg-blue-500/20 text-blue-100 hover:bg-blue-500/35";
      let statusPrefix = "Source";
      if (cit.verification === "SUPPORTED") {
        badgeClasses = "border-emerald-400/60 bg-emerald-500/20 text-emerald-100 hover:bg-emerald-500/35";
        statusPrefix = "Verified Evidence";
      } else if (cit.verification === "PARTIAL" || cit.verification === "PARTIALLY_SUPPORTED") {
        badgeClasses = "border-amber-400/60 bg-amber-500/20 text-amber-100 hover:bg-amber-500/35";
        statusPrefix = "Partial Evidence";
      } else if (cit.verification === "CONTRADICTED") {
        badgeClasses = "border-rose-400/60 bg-rose-500/20 text-rose-100 hover:bg-rose-500/35";
        statusPrefix = "Contradicted / Mismatch";
      }

      return (
        <button
          key={i}
          type="button"
          onClick={() => onCitationClick?.(cit)}
          className={`mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1.5 align-baseline text-[11px] font-semibold leading-none transition ${badgeClasses}`}
          title={`[${statusPrefix}] Page ${cit.page}: "${cit.quote?.slice(0, 120) || "Click to jump to page"}"`}
        >
          {cit.ref_id ?? m[1]}
        </button>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function CodeBlock({ code, language }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="my-3 overflow-hidden rounded-xl border border-zinc-700/70 bg-zinc-950/90 shadow-sm">
      <div className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/80 px-3 py-1.5 text-[11px] text-zinc-400">
        <span className="font-mono uppercase tracking-wider">{language || "code"}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="rounded px-2 py-0.5 text-[10px] font-medium text-zinc-300 transition hover:bg-zinc-800 hover:text-white"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 font-mono text-xs leading-relaxed text-emerald-300">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function MarkdownTable({
  rawLines,
  citations,
  onCitationClick,
}: {
  rawLines: string[];
  citations: Citation[];
  onCitationClick?: (c: Citation) => void;
}) {
  const parsedRows = rawLines.map((l) =>
    l
      .trim()
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((c) => c.trim())
  );
  if (parsedRows.length < 2) return null;

  // Filter out separator lines like |---|---|
  const headerRow = parsedRows[0];
  const bodyRows = parsedRows.slice(1).filter((r) => !r.every((c) => /^[:\s-]+$/.test(c)));

  return (
    <div className="my-3 overflow-x-auto rounded-xl border border-zinc-700/70 bg-zinc-900/50 shadow-inner">
      <table className="w-full border-collapse text-left text-xs">
        <thead className="border-b border-zinc-700 bg-zinc-800/90 text-[11px] font-semibold text-zinc-200">
          <tr>
            {headerRow.map((cell, idx) => (
              <th key={idx} className="px-3 py-2 border-r border-zinc-700/50 last:border-r-0">
                {renderInline(cell, citations, onCitationClick)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/70 text-zinc-300">
          {bodyRows.map((row, rIdx) => (
            <tr key={rIdx} className="transition-colors hover:bg-zinc-800/30 odd:bg-zinc-900/30">
              {row.map((cell, cIdx) => (
                <td key={cIdx} className="px-3 py-2 border-r border-zinc-800/50 last:border-r-0 text-zinc-200 font-mono text-[11.5px]">
                  {renderInline(cell, citations, onCitationClick)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function renderAnswer(
  content: string,
  citations: Citation[] = [],
  onCitationClick?: (c: Citation) => void
) {
  const cleaned = content.replace(/^#{1,6}\s*/gm, "").trim();

  // 1. Extract fenced code blocks
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  const blocks: Array<{ type: "code" | "text"; content: string; language?: string }> = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = codeBlockRegex.exec(cleaned)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({ type: "text", content: cleaned.slice(lastIndex, match.index) });
    }
    blocks.push({ type: "code", content: match[2].trim(), language: match[1] || "code" });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < cleaned.length) {
    blocks.push({ type: "text", content: cleaned.slice(lastIndex) });
  }

  return (
    <div className="space-y-3">
      {blocks.map((block, bIdx) => {
        if (block.type === "code") {
          return <CodeBlock key={bIdx} code={block.content} language={block.language} />;
        }

        // 2. Process text block: group contiguous table lines
        const lines = block.content.split("\n").map((l) => l.trim()).filter(Boolean);
        const elements: React.ReactNode[] = [];
        let tableBuffer: string[] = [];

        const flushTable = (key: string) => {
          if (tableBuffer.length > 0) {
            elements.push(
              <MarkdownTable
                key={key}
                rawLines={[...tableBuffer]}
                citations={citations}
                onCitationClick={onCitationClick}
              />
            );
            tableBuffer = [];
          }
        };

        lines.forEach((line, i) => {
          const isTableRow = line.startsWith("|") && line.endsWith("|");
          if (isTableRow) {
            tableBuffer.push(line);
            return;
          }

          flushTable(`table-${bIdx}-${i}`);

          const bullet = line.match(/^[-*]\s+(.+)/);
          const numbered = line.match(/^\d+[.)]\s+(.+)/);
          const heading = line.match(/^\*\*([^*]+)\*\*:?\s*(.*)$/);

          if (heading) {
            elements.push(
              <div key={`head-${bIdx}-${i}`} className="pt-1.5">
                <div className="font-semibold text-white tracking-wide">{heading[1]}</div>
                {heading[2] && (
                  <div className="mt-1 text-zinc-200">
                    {renderInline(heading[2], citations, onCitationClick)}
                  </div>
                )}
              </div>
            );
          } else if (bullet || numbered) {
            elements.push(
              <div key={`bullet-${bIdx}-${i}`} className="flex gap-2 text-zinc-200">
                <span className="mt-[0.65em] h-1.5 w-1.5 shrink-0 rounded-full bg-acid/80" />
                <span>{renderInline((bullet?.[1] ?? numbered?.[1] ?? line).trim(), citations, onCitationClick)}</span>
              </div>
            );
          } else {
            elements.push(
              <p key={`p-${bIdx}-${i}`} className="text-zinc-200 leading-relaxed">
                {renderInline(line, citations, onCitationClick)}
              </p>
            );
          }
        });

        flushTable(`table-end-${bIdx}`);

        return <div key={bIdx} className="space-y-2">{elements}</div>;
      })}
    </div>
  );
}

function quotePreview(q?: string | null) {
  if (!q) return "";
  return q.length > 145 ? q.slice(0, 145) + "…" : q;
}

function sectionLabel(c: Citation) {
  if (c.is_figure) return c.label || c.section_title || "Figure";
  return c.section_title || c.chunk_type || "Paper";
}

function ReasoningBadge({ level }: { level?: string }) {
  if (!level) return null;
  const config: Record<string, { label: string; bg: string; text: string }> = {
    L1_DIRECT_LOOKUP: { label: "L1 Direct Lookup", bg: "bg-blue-500/15", text: "text-blue-400" },
    L2_SAME_SECTION: { label: "L2 Same-Section", bg: "bg-cyan-500/15", text: "text-cyan-400" },
    L3_CROSS_SECTION: { label: "L3 Cross-Section", bg: "bg-purple-500/15", text: "text-purple-400" },
    L4_CROSS_MODAL: { label: "L4 Cross-Modal", bg: "bg-emerald-500/15", text: "text-emerald-400" },
    L5_MULTI_HOP_SYNTHESIS: { label: "L5 Multi-Hop Synthesis", bg: "bg-amber-500/15", text: "text-amber-400" },
  };
  const c = config[level] || { label: level.replace(/_/g, " "), bg: "bg-zinc-700/50", text: "text-zinc-300" };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide ${c.bg} ${c.text}`}>
      <Layers size={9} />
      {c.label}
    </span>
  );
}

import { EvidenceGraphModal } from "./EvidenceGraphModal";

function ReasoningTrail({ steps, onStepClick, onOpenGraph }: {
  steps?: ReasoningPathStep[];
  onStepClick?: (citation: Citation) => void;
  onOpenGraph?: () => void;
}) {
  if (!steps || steps.length <= 1) return null;
  return (
    <div className="mt-2.5 rounded-lg border border-purple-500/20 bg-purple-950/20 p-2.5 text-xs">
      <div className="flex items-center justify-between font-medium text-purple-300 text-[11px] mb-1.5">
        <div className="flex items-center gap-1.5">
          <Network size={12} className="text-purple-400 shrink-0" />
          <span>Evidence Reasoning Chain</span>
        </div>
        {onOpenGraph && (
          <button
            type="button"
            onClick={onOpenGraph}
            className="flex items-center gap-1 text-[10px] text-purple-300 hover:text-white transition font-normal"
          >
            <span>View Graph</span>
            <ChevronRight size={10} />
          </button>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-1 text-[11px] text-zinc-300">
        {steps.map((step, idx) => (
          <div key={step.step_index} className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => onStepClick?.({
                page: step.page,
                chunk_id: step.evidence_id,
                section_title: step.section,
                chunk_type: step.modality,
                quote: step.claim_contribution,
              })}
              className="inline-flex items-center gap-1 rounded bg-purple-500/20 px-1.5 py-0.5 font-mono text-[10px] text-purple-200 hover:bg-purple-500/30 transition text-left"
              title={`${step.reasoning_mode ? `[${step.reasoning_mode}] ` : ""}${step.subgoal ? `${step.subgoal} — ` : ""}${step.claim_contribution}`}
            >
              <span className="font-semibold">{step.step_index}.</span>
              {step.reasoning_mode && (
                <span className="text-[9px] uppercase tracking-wider font-semibold text-purple-300 bg-purple-900/50 px-1 rounded">
                  {step.reasoning_mode.replace("ProblemUnderstanding", "Understand").replace("CaseAnalysis", "Cases")}
                </span>
              )}
              {step.section || step.evidence_id} (p.{step.page})
            </button>
            {idx < steps.length - 1 && <ChevronRight size={10} className="text-zinc-500 shrink-0" />}
          </div>
        ))}
      </div>
    </div>
  );
}

function NumericPlanBadge({ plan }: { plan?: { operation: string; computed_value: number; formatted_value: string; formatted_statement: string; is_exact: boolean } }) {
  if (!plan) return null;
  return (
    <div className="mt-2.5 rounded-lg border border-emerald-500/20 bg-emerald-950/20 p-2.5 text-xs text-zinc-200">
      <div className="flex items-center gap-1.5 font-medium text-emerald-300 text-[11px] mb-1">
        <Calculator size={12} className="text-emerald-400 shrink-0" />
        <span>Deterministic Tabular Arithmetic</span>
        {plan.is_exact && <span className="ml-auto rounded bg-emerald-500/30 px-1 text-[9px] font-mono text-emerald-200">EXACT</span>}
      </div>
      <p className="text-[11px] leading-relaxed text-zinc-300">{plan.formatted_statement}</p>
    </div>
  );
}

function VerificationReportBadge({ report }: { report?: { overall_supported: boolean; supported_count: number; unsupported_count: number; contradicted_count: number; has_abstained: boolean } }) {
  if (!report) return null;
  return (
    <div className="mt-2 flex items-center gap-2 text-[10px] text-zinc-400">
      <ShieldCheck size={11} className={report.overall_supported ? "text-emerald-400" : "text-amber-400"} />
      <span>
        {report.supported_count} verified
        {report.unsupported_count > 0 && ` · ${report.unsupported_count} caveats`}
        {report.contradicted_count > 0 && ` · ${report.contradicted_count} contradictions`}
      </span>
    </div>
  );
}

function VerificationBadge({ label }: { label?: string }) {
  if (!label) return null;
  if (label === "SUPPORTED") {
    return <span className="ml-auto rounded bg-emerald-500/20 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-300">Verified</span>;
  }
  if (label === "PARTIAL" || label === "PARTIALLY_SUPPORTED") {
    return <span className="ml-auto rounded bg-amber-500/20 px-1.5 py-0.5 text-[9px] font-semibold text-amber-300">Partial</span>;
  }
  if (label === "CONTRADICTED") {
    return <span className="ml-auto rounded bg-rose-500/20 px-1.5 py-0.5 text-[9px] font-semibold text-rose-300">Contradicted</span>;
  }
  return <span className="ml-auto rounded bg-zinc-700/50 px-1.5 py-0.5 text-[9px] font-semibold text-zinc-400">Uncertain</span>;
}

function ModelBadge({ model, vision, routeType, reasoningLevel }: { model?: string; vision?: boolean; routeType?: string; reasoningLevel?: string }) {
  if (!model && !vision && !routeType && !reasoningLevel) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {reasoningLevel && <ReasoningBadge level={reasoningLevel} />}
      {(model || vision) && (
        <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
          vision ? "bg-teal-500/15 text-teal-400" : "bg-zinc-700 text-zinc-400"
        }`}>
          {vision ? <><Camera size={9} />Vision</> : <><Zap size={9} />{model}</>}
        </span>
      )}
      {routeType && (
        <span className="rounded-full bg-indigo-500/15 px-2 py-0.5 text-[10px] font-medium capitalize text-indigo-300">
          {routeType.replace(/_/g, " ").toLowerCase()}
        </span>
      )}
    </div>
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
  activeSnippet = null,
  onDismissSnippet,
}: ChatBoxProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [copiedMsgIdx, setCopiedMsgIdx] = useState<number | null>(null);
  const [dynamicPrompts, setDynamicPrompts] = useState<{ icon: typeof FileText; text: string }[]>([]);
  const [graphModal, setGraphModal] = useState<{
    isOpen: boolean;
    query: string;
    level?: string;
    steps?: ReasoningPathStep[];
  }>({
    isOpen: false,
    query: "",
  });
  const handledPrompt = useRef<number | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Focus input whenever a new snippet is attached
  useEffect(() => {
    if (activeSnippet) {
      inputRef.current?.focus();
    }
  }, [activeSnippet]);

  // Fetch dynamic suggestions based on paper figures / tables
  useEffect(() => {
    let cancelled = false;
    async function loadPaperFigures() {
      try {
        const res = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/figures`);
        if (!res.ok) throw new Error();
        const data: unknown = await res.json();
        if (!isFigureListResponse(data)) {
          throw new Error("Invalid figures response");
        }
        if (cancelled) return;

        const figs = data.figures;
        const prompts: { icon: typeof FileText; text: string }[] = [];
        const tables = figs.filter((f) => f.figure_type === "table" || f.label.toLowerCase().includes("table"));
        const figures = figs.filter((f) => f.figure_type === "figure" || f.label.toLowerCase().includes("figure"));

        if (tables.length > 0) {
          prompts.push({ icon: BarChart2, text: `Explain ${tables[0].label}` });
        }
        if (figures.length >= 2) {
          prompts.push({ icon: GitCompare, text: `Compare ${figures[0].label} and ${figures[1].label}` });
        } else if (figures.length === 1) {
          prompts.push({ icon: Camera, text: `What does ${figures[0].label} show?` });
        }
        prompts.push({ icon: Sparkles, text: "Explain the core methodology in detail" });
        prompts.push({ icon: FileText, text: "What is the main contribution of this paper?" });

        setDynamicPrompts(prompts.slice(0, 4));
      } catch {
        if (!cancelled) setDynamicPrompts(SUGGESTED_QUESTIONS);
      }
    }
    void loadPaperFigures();
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  const exportReport = async (msg: ChatMessage, format: "markdown" | "latex") => {
    try {
      const msgIdx = messages.indexOf(msg);
      const userQ = [...messages].slice(0, msgIdx).reverse().find((m) => m.role === "user")?.content || "Scientific Query";
      const res = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/export/reasoning`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userQ,
          answer: msg.content,
          format,
          reasoning_level: msg.reasoning_level || "L5_MULTI_HOP_SYNTHESIS",
          steps: msg.reasoning_steps || [],
          numeric_plan: msg.numeric_plan || null,
          verification_report: msg.verification_report || null,
        }),
      });
      if (!res.ok) throw new Error();
      const data: unknown = await res.json();
      if (!isExportReasoningResponse(data)) {
        throw new Error("Invalid reasoning export response");
      }
      const blob = new Blob([data.content], { type: format === "latex" ? "application/x-latex" : "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = data.filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // Fallback
    }
  };

  const copyMessage = (idx: number, content: string, citations: Citation[] = []) => {
    let textToCopy = content;
    if (citations.length > 0) {
      textToCopy += "\n\n### Sources:\n" + citations.map((c, i) => `[${c.ref_id ?? i + 1}] Page ${c.page}: ${c.quote || c.label}`).join("\n");
    }
    navigator.clipboard.writeText(textToCopy);
    setCopiedMsgIdx(idx);
    setTimeout(() => setCopiedMsgIdx(null), 2000);
  };

  const exportStudyNotes = () => {
    if (messages.length === 0) return;
    const dateStr = new Date().toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
    let md = `# ScholAR Study Notes: Paper ${paperId}\n*Exported on ${dateStr}*\n\n---\n\n`;

    messages.forEach((m) => {
      if (m.role === "user") {
        md += `## Question: ${m.content}\n\n`;
      } else {
        md += `### ScholAR Response\n${m.content}\n\n`;
        if (m.citations && m.citations.length > 0) {
          md += `**Cited Sources:**\n`;
          m.citations.forEach((c, ci) => {
            md += `- **[${c.ref_id ?? ci + 1}]** Page ${c.page} (${c.section_title || c.label}): "${c.quote || c.caption}"\n`;
          });
          md += "\n";
        }
        md += "---\n\n";
      }
    });

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${paperId}_Study_Notes.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

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
      const payloadBody: ChatRequest = {
        message: trimmed,
        history,
        secondary_paper_ids: secondaryPaperIds,
      };

      if (activeSnippet) {
        payloadBody.snippet_id = activeSnippet.id;
        payloadBody.snippet_page = activeSnippet.page;
        payloadBody.snippet_bbox = activeSnippet.bbox;
        payloadBody.snippet_text = activeSnippet.text;
        onDismissSnippet?.();
      }

      const res = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payloadBody),
      });
      if (!res.ok) {
        const errorPayload: unknown = await res.json().catch(() => null);
        throw new Error(getApiErrorMessage(errorPayload, "Chat failed"));
      }
      const payload: unknown = await res.json();
      if (!isChatResponse(payload)) {
        throw new Error("The chat service returned an invalid response.");
      }

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
          citations: payload.citations,
          abstained: payload.abstained,
          uncertainty_reason: payload.uncertainty_reason,
          route_type: payload.route_type,
          capability_mode: payload.capability_mode,
          vision: payload.vision,
          vision_fallback: payload.vision_fallback,
          is_snippet: payload.is_snippet,
          snippet_id: payload.snippet_id ?? undefined,
          figure_id: payload.figure_id ?? undefined,
          figure_label: payload.figure_label ?? undefined,
          figure_image_url: payload.figure_image_url ?? undefined,
          model: payload.model,
          reasoning_level: payload.reasoning_level,
          reasoning_steps: payload.reasoning_steps,
          numeric_plan: payload.numeric_plan ?? undefined,
          verification_report: payload.verification_report ?? undefined,
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
  }, [loading, messages, paperId, secondaryPaperIds, activeSnippet, onDismissSnippet, onChatActivity]);

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
      {/* Top action header for export */}
      {messages.length > 0 && (
        <div className="mb-2 flex items-center justify-between border-b border-line/60 pb-2 text-xs">
          <span className="text-zinc-500 font-medium">{messages.length} messages</span>
          <button
            type="button"
            onClick={exportStudyNotes}
            className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1 text-[11px] font-medium text-zinc-300 transition hover:border-zinc-500 hover:text-white"
            title="Download full Q&A and citations as a Markdown study note"
          >
            <FileText size={12} className="text-acid" />
            Export Notes (.md)
          </button>
        </div>
      )}

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
                  ? `Multi-doc mode active: ${secondaryPaperIds.length} reference${secondaryPaperIds.length > 1 ? "s" : ""} loaded`
                  : "Figures, tables, methods, comparisons: all grounded in the source"}
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2 w-full max-w-sm">
              {(dynamicPrompts.length > 0 ? dynamicPrompts : SUGGESTED_QUESTIONS).map(({ icon: Icon, text }) => (
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
                  {/* Model, reasoning & route badge */}
                  {(msg.model || msg.vision || msg.route_type || msg.reasoning_level) && (
                    <div className="mb-2 flex items-center">
                      <ModelBadge
                        model={msg.model}
                        vision={msg.vision}
                        routeType={msg.route_type}
                        reasoningLevel={msg.reasoning_level}
                      />
                      {msg.vision_fallback && (
                        <span className="ml-2 text-[10px] text-amber-400">(caption fallback)</span>
                      )}
                    </div>
                  )}
                  {msg.numeric_plan && <NumericPlanBadge plan={msg.numeric_plan} />}
                  {msg.reasoning_steps && (
                    <ReasoningTrail
                      steps={msg.reasoning_steps}
                      onStepClick={onCitationClick}
                      onOpenGraph={() =>
                        setGraphModal({
                          isOpen: true,
                          query: messages[idx - 1]?.content || "Multi-Level Evidence Query",
                          level: msg.reasoning_level,
                          steps: msg.reasoning_steps,
                        })
                      }
                    />
                  )}
                  <div className="mt-1">
                    {renderAnswer(msg.content, msg.citations ?? [], onCitationClick)}
                  </div>
                  {msg.verification_report && <VerificationReportBadge report={msg.verification_report} />}
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
                          <VerificationBadge label={cit.verification} />
                        </div>
                        {cit.is_figure && (cit.figure_id || cit.image_file) ? (
                          <FigureThumbnail
                            imageUrl={cit.image_url || (cit.figure_id ? `/api/papers/${cit.source_paper_id || paperId}/figures/${cit.figure_id}.png` : (msg.figure_image_url || ""))}
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

              {/* Assistant Message Actions */}
              {msg.role === "assistant" && !msg.error && (
                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-line/40 pt-2 text-[11px] text-zinc-500">
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => copyMessage(idx, msg.content, msg.citations)}
                      className="inline-flex items-center gap-1 rounded px-2 py-0.5 transition hover:bg-zinc-800 hover:text-zinc-200"
                      title="Copy answer & citations"
                    >
                      <Copy size={11} />
                      {copiedMsgIdx === idx ? "Copied!" : "Copy"}
                    </button>
                    <button
                      type="button"
                      onClick={() => exportReport(msg, "markdown")}
                      className="inline-flex items-center gap-1 rounded px-2 py-0.5 transition hover:bg-zinc-800 hover:text-zinc-200"
                      title="Export verified reasoning report as Markdown"
                    >
                      <Download size={11} />
                      .md
                    </button>
                    <button
                      type="button"
                      onClick={() => exportReport(msg, "latex")}
                      className="inline-flex items-center gap-1 rounded px-2 py-0.5 transition hover:bg-zinc-800 hover:text-zinc-200"
                      title="Export verified reasoning report as LaTeX document"
                    >
                      <Download size={11} />
                      .tex
                    </button>
                  </div>
                  {idx === messages.length - 1 && !loading && (
                    <button
                      type="button"
                      onClick={() => {
                        const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
                        if (lastUserMsg) void sendMessage(lastUserMsg.content);
                      }}
                      className="inline-flex items-center gap-1 rounded px-2 py-0.5 transition hover:bg-zinc-800 hover:text-zinc-200"
                      title="Regenerate this answer"
                    >
                      <RotateCcw size={11} />
                      Retry
                    </button>
                  )}
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

      {/* ── Active Snippet Attachment Card ── */}
      {activeSnippet && (
        <div className="mb-2 flex items-center justify-between rounded-xl border border-blue-500/40 bg-blue-500/10 p-2.5 shadow-md backdrop-blur-sm">
          <div className="flex items-center gap-3 min-w-0">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${backendUrl}${activeSnippet.imageUrl}`}
              alt="Snippet preview"
              className="h-11 w-16 shrink-0 rounded-md border border-blue-400/40 bg-black/60 object-contain shadow-inner"
            />
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 font-semibold text-blue-200 text-xs">
                <Crop size={13} className="text-blue-400" />
                <span>Attached Snippet · Page {activeSnippet.page}</span>
              </div>
              <p className="truncate text-[11px] text-zinc-400 mt-0.5">
                {activeSnippet.text ? activeSnippet.text.slice(0, 90) : "Visual region selected from PDF"}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onDismissSnippet}
            className="rounded-md p-1.5 text-zinc-400 transition hover:bg-white/10 hover:text-white"
            title="Dismiss snippet"
          >
            <X size={15} />
          </button>
        </div>
      )}

      {/* ── Snippet Quick Suggestions ── */}
      {activeSnippet && (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {[
            "Explain this equation step-by-step",
            "What does this chart show?",
            "Summarize the key takeaway from this region",
            "Convert this pseudocode to Python",
          ].map((promptText) => (
            <button
              key={promptText}
              type="button"
              onClick={() => void sendMessage(promptText)}
              className="flex items-center gap-1 rounded-full border border-blue-400/30 bg-blue-500/10 px-2.5 py-1 text-[11px] font-medium text-blue-200 transition hover:bg-blue-500/20 hover:text-white"
            >
              <Sparkles size={11} className="text-blue-400" />
              <span>{promptText}</span>
            </button>
          ))}
        </div>
      )}

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
          placeholder={activeSnippet ? `Ask anything about this snippet on page ${activeSnippet.page}…` : "Ask anything…"}
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

      {/* Interactive Evidence Graph Modal */}
      <EvidenceGraphModal
        isOpen={graphModal.isOpen}
        onClose={() => setGraphModal((prev) => ({ ...prev, isOpen: false }))}
        query={graphModal.query}
        reasoningLevel={graphModal.level}
        reasoningSteps={graphModal.steps}
        onNodeClick={onCitationClick}
      />
    </div>
  );
}
