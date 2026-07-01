"use client";

import { useEffect, useState, useCallback } from "react";
import {
  BookOpen, ChevronDown, ChevronUp, ExternalLink,
  Loader2, Plus, CheckCircle2, AlertTriangle, Download
} from "lucide-react";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export interface Reference {
  ref_number: number;
  title: string;
  authors: string[];
  year: string;
  abstract: string;
  arxiv_id: string | null;
  s2_paper_id: string | null;
  pdf_url: string;
  abs_url: string;
  ingested: boolean;
  secondary_local_id: string | null;
}

interface ReferencesPanelProps {
  paperId: string;
  onSecondaryPaperIngested: (secondaryLocalId: string) => void;
  activeSecondaryIds: Set<string>;
  /** When true, renders inline (no collapse toggle) — used inside the References tab */
  inline?: boolean;
}

type IngestStatus = "idle" | "loading" | "done" | "error";
type IngestState  = Record<number, IngestStatus>;

function SkeletonCard() {
  return (
    <div className="rounded-xl border border-line bg-panel p-3 animate-pulse">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 space-y-2">
          <div className="h-3 w-4/5 rounded bg-zinc-700" />
          <div className="h-2.5 w-2/5 rounded bg-zinc-800" />
        </div>
        <div className="h-6 w-20 shrink-0 rounded-lg bg-zinc-800" />
      </div>
      <div className="mt-2 space-y-1.5">
        <div className="h-2 w-full rounded bg-zinc-800" />
        <div className="h-2 w-3/4 rounded bg-zinc-800" />
      </div>
    </div>
  );
}

export function ReferencesPanel({
  paperId,
  onSecondaryPaperIngested,
  activeSecondaryIds,
  inline = false,
}: ReferencesPanelProps) {
  const [open,          setOpen]          = useState(inline); // auto-open when inline
  const [refs,          setRefs]          = useState<Reference[]>([]);
  const [loading,       setLoading]       = useState(false);
  const [uploadWarning, setUploadWarning] = useState(false);
  const [ingestState,   setIngestState]   = useState<IngestState>({});
  const [ingestMessage, setIngestMessage] = useState<Record<number, string>>({});
  const [error,         setError]         = useState<string | null>(null);
  const [batchLoading,  setBatchLoading]  = useState(false);

  const loadRefs = useCallback(async () => {
    if (refs.length > 0) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${backendUrl}/api/papers/${encodeURIComponent(paperId)}/references`
      );
      if (!res.ok) throw new Error("Could not load references");
      const payload = await res.json();
      setRefs(payload.references ?? []);
      setUploadWarning(Boolean(payload.upload_warning));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load references");
    } finally {
      setLoading(false);
    }
  }, [paperId, refs.length]);

  useEffect(() => {
    if (open) void loadRefs();
  }, [open, loadRefs]);

  async function ingestRef(refIndex: number) {
    setIngestState((prev) => ({ ...prev, [refIndex]: "loading" }));
    setIngestMessage((prev) => ({ ...prev, [refIndex]: "Downloading PDF…" }));

    // Update message after 4s to give estimated time feedback
    const timer = setTimeout(() => {
      setIngestMessage((prev) => ({ ...prev, [refIndex]: "Processing pages…" }));
    }, 4000);

    try {
      const res = await fetch(
        `${backendUrl}/api/papers/${encodeURIComponent(paperId)}/references/${refIndex}/ingest`,
        { method: "POST" }
      );
      clearTimeout(timer);
      if (!res.ok) {
        const p = await res.json().catch(() => ({}));
        throw new Error(p.detail ?? "Ingest failed");
      }
      const payload = await res.json();
      const secId: string = payload.secondary_paper_id;
      setRefs((prev) =>
        prev.map((ref, i) =>
          i === refIndex ? { ...ref, ingested: true, secondary_local_id: secId } : ref
        )
      );
      setIngestState((prev) => ({ ...prev, [refIndex]: "done" }));
      setIngestMessage((prev) => ({ ...prev, [refIndex]: "" }));
      onSecondaryPaperIngested(secId);
    } catch (err) {
      clearTimeout(timer);
      setIngestState((prev) => ({ ...prev, [refIndex]: "error" }));
      setIngestMessage((prev) => ({ ...prev, [refIndex]: "" }));
      console.error("Ingest error:", err);
    }
  }

  async function ingestAll() {
    const pending = refsWithPdf
      .map((ref) => refs.indexOf(ref))
      .filter((i) => {
        const ref = refs[i];
        const isActive = ref.secondary_local_id
          ? activeSecondaryIds.has(ref.secondary_local_id)
          : false;
        return !isActive && ingestState[i] !== "done" && ingestState[i] !== "loading";
      });

    if (pending.length === 0) return;
    setBatchLoading(true);
    for (const idx of pending) {
      await ingestRef(idx);
    }
    setBatchLoading(false);
  }

  const refsWithPdf = refs.filter((r) => r.pdf_url);
  const refsNoPdf   = refs.filter((r) => !r.pdf_url);
  const activeCount = activeSecondaryIds.size;
  const loadableCount = refsWithPdf.length;
  const unloadedCount = refsWithPdf.filter((r) => {
    const isActive = r.secondary_local_id
      ? activeSecondaryIds.has(r.secondary_local_id)
      : false;
    return !isActive;
  }).length;

  return (
    <div className={inline ? "bg-ink" : "border-t border-line bg-ink"}>
      {/* Header toggle — only shown when used in collapsed sidebar mode */}
      {!inline && (
        <button
          id="references-panel-toggle"
          type="button"
          onClick={() => setOpen((prev) => !prev)}
          className="flex w-full items-center justify-between px-5 py-3 text-sm font-medium text-zinc-300 transition hover:text-white"
          aria-expanded={open}
        >
          <span className="flex items-center gap-2">
            <BookOpen size={15} className="text-acid" />
            References
            {loadableCount > 0 && (
              <span className="rounded-full bg-zinc-700 px-2 py-0.5 text-xs text-zinc-300">
                {loadableCount} available
              </span>
            )}
            {activeCount > 0 && (
              <span className="rounded-full bg-acid/20 px-2 py-0.5 text-xs font-semibold text-acid">
                {activeCount} active
              </span>
            )}
          </span>
          {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>
      )}

      {open && (
        <div className={inline ? "px-4 py-4" : "max-h-80 overflow-y-auto px-5 pb-4"}>
          {/* Upload warning */}
          {uploadWarning && (
            <div className="mb-3 flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-300">
              <AlertTriangle size={13} className="mt-0.5 shrink-0" />
              <span>
                <strong>Uploaded PDF</strong> — references resolved via Semantic Scholar title search.
                {refs.length > 0
                  ? ` Found ${refs.length} reference${refs.length > 1 ? "s" : ""}.`
                  : " Some references may be missing."}
              </span>
            </div>
          )}

          {/* Skeleton loading */}
          {loading && (
            <div className="space-y-2 py-2">
              {[0, 1, 2].map((i) => <SkeletonCard key={i} />)}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
              {error}
            </div>
          )}

          {/* Empty state */}
          {!loading && !error && refs.length === 0 && (
            <div className="py-6 text-center text-sm text-zinc-500">
              No references found.
              <br />
              <span className="text-xs">
                The paper may not be indexed by Semantic Scholar.
              </span>
            </div>
          )}

          {/* Batch load button */}
          {!loading && unloadedCount > 1 && (
            <div className="mb-3 flex items-center justify-between gap-2">
              <span className="text-xs text-zinc-500">
                {unloadedCount} papers available to load
              </span>
              <button
                type="button"
                onClick={ingestAll}
                disabled={batchLoading}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-600 bg-panel px-3 py-1.5 text-xs text-zinc-300 transition hover:border-acid hover:text-acid disabled:opacity-50"
              >
                {batchLoading ? (
                  <Loader2 size={11} className="animate-spin" />
                ) : (
                  <Download size={11} />
                )}
                Load all
              </button>
            </div>
          )}

          {/* Reference cards */}
          {!loading && refsWithPdf.length > 0 && (
            <div className="space-y-2">
              {refsWithPdf.map((ref) => {
                const originalIndex = refs.indexOf(ref);
                const state    = ingestState[originalIndex] ?? "idle";
                const message  = ingestMessage[originalIndex] ?? "";
                const isActive = ref.secondary_local_id
                  ? activeSecondaryIds.has(ref.secondary_local_id)
                  : false;

                return (
                  <div
                    key={`ref-${originalIndex}`}
                    id={`ref-card-${originalIndex}`}
                    className={`rounded-xl border p-3 transition ${
                      isActive
                        ? "border-acid/40 bg-acid/5"
                        : "border-line bg-panel hover:border-zinc-600"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-semibold leading-snug text-white">
                          {ref.title || "Untitled reference"}
                        </div>
                        <div className="mt-0.5 truncate text-[11px] text-zinc-500">
                          {ref.authors.slice(0, 2).join(", ")}
                          {ref.authors.length > 2 ? " et al." : ""}
                          {ref.year ? ` · ${ref.year}` : ""}
                          {ref.arxiv_id && (
                            <span className="ml-1.5 text-zinc-600">arXiv:{ref.arxiv_id}</span>
                          )}
                        </div>
                      </div>

                      <div className="flex shrink-0 items-center gap-1.5">
                        {ref.abs_url && (
                          <a
                            href={ref.abs_url}
                            target="_blank"
                            rel="noreferrer"
                            title="Open on arXiv"
                            className="rounded p-1 text-zinc-600 hover:text-white"
                          >
                            <ExternalLink size={12} />
                          </a>
                        )}

                        {isActive ? (
                          <span className="flex items-center gap-1 rounded-lg bg-acid/20 px-2 py-1 text-[11px] font-semibold text-acid">
                            <CheckCircle2 size={11} />
                            Active
                          </span>
                        ) : state === "loading" ? (
                          <span className="flex items-center gap-1.5 rounded-lg border border-line px-2 py-1 text-[11px] text-zinc-400">
                            <Loader2 size={11} className="animate-spin" />
                            {message || "Loading…"}
                          </span>
                        ) : state === "error" ? (
                          <button
                            type="button"
                            onClick={() => ingestRef(originalIndex)}
                            className="rounded-lg border border-red-500/40 bg-red-500/10 px-2 py-1 text-[11px] text-red-400 hover:bg-red-500/20"
                          >
                            Retry
                          </button>
                        ) : (
                          <button
                            type="button"
                            id={`ingest-ref-${originalIndex}`}
                            onClick={() => ingestRef(originalIndex)}
                            className="flex items-center gap-1 rounded-lg border border-zinc-600 bg-panel px-2 py-1 text-[11px] text-zinc-300 transition hover:border-acid hover:text-acid"
                            title="Download and add to session"
                          >
                            <Plus size={11} />
                            Load
                          </button>
                        )}
                      </div>
                    </div>

                    {ref.abstract && (
                      <p className="mt-1.5 line-clamp-2 text-[11px] leading-relaxed text-zinc-500">
                        {ref.abstract}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* No-PDF references */}
          {!loading && refsNoPdf.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-zinc-600 hover:text-zinc-400">
                {refsNoPdf.length} reference{refsNoPdf.length > 1 ? "s" : ""} without downloadable PDF
              </summary>
              <div className="mt-2 space-y-1">
                {refsNoPdf.map((ref, i) => (
                  <div key={`nopdf-${i}`} className="text-xs text-zinc-600">
                    {ref.title || "Untitled"}{ref.year ? ` (${ref.year})` : ""}
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
