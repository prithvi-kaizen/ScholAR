"use client";

import { useEffect, useState, useCallback } from "react";
import { BookOpen, ChevronDown, ChevronUp, ExternalLink, Loader2, Plus, CheckCircle2, AlertTriangle } from "lucide-react";

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
  /** Called when a secondary paper is successfully ingested */
  onSecondaryPaperIngested: (secondaryLocalId: string) => void;
  /** Set of already-active secondary paper IDs */
  activeSecondaryIds: Set<string>;
}

interface IngestState {
  [refIndex: number]: "idle" | "loading" | "done" | "error";
}

export function ReferencesPanel({ paperId, onSecondaryPaperIngested, activeSecondaryIds }: ReferencesPanelProps) {
  const [open, setOpen] = useState(false);
  const [refs, setRefs] = useState<Reference[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploadWarning, setUploadWarning] = useState(false);
  const [ingestState, setIngestState] = useState<IngestState>({});
  const [error, setError] = useState<string | null>(null);

  const loadRefs = useCallback(async () => {
    if (refs.length > 0) return; // already loaded
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${backendUrl}/api/papers/${encodeURIComponent(paperId)}/references`
      );
      if (!response.ok) throw new Error("Could not load references");
      const payload = await response.json();
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
    try {
      const response = await fetch(
        `${backendUrl}/api/papers/${encodeURIComponent(paperId)}/references/${refIndex}/ingest`,
        { method: "POST" }
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "Ingest failed");
      }
      const payload = await response.json();
      const secId: string = payload.secondary_paper_id;

      // Update local ref state
      setRefs((prev) =>
        prev.map((ref, idx) =>
          idx === refIndex ? { ...ref, ingested: true, secondary_local_id: secId } : ref
        )
      );
      setIngestState((prev) => ({ ...prev, [refIndex]: "done" }));
      onSecondaryPaperIngested(secId);
    } catch (err) {
      setIngestState((prev) => ({ ...prev, [refIndex]: "error" }));
      console.error("Ingest error:", err);
    }
  }

  const refsWithPdf = refs.filter((r) => r.pdf_url);
  const refsNoPdf   = refs.filter((r) => !r.pdf_url);

  return (
    <div className="border-t border-line bg-ink">
      {/* Collapsed header */}
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
          {refs.length > 0 && (
            <span className="rounded-full bg-zinc-700 px-2 py-0.5 text-xs text-zinc-300">
              {refsWithPdf.length} loadable
            </span>
          )}
          {activeSecondaryIds.size > 0 && (
            <span className="rounded-full bg-acid/20 px-2 py-0.5 text-xs text-acid">
              {activeSecondaryIds.size} active
            </span>
          )}
        </span>
        {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>

      {open && (
        <div className="max-h-72 overflow-y-auto px-5 pb-4">
          {uploadWarning && (
            <div className="mb-3 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-2 text-xs text-amber-300">
              <AlertTriangle size={13} className="mt-0.5 shrink-0" />
              <span>
                This is an uploaded PDF. Reference resolution used arXiv title-search (best-effort).
                Some references may be missing or mismatched.
              </span>
            </div>
          )}

          {loading && (
            <div className="flex items-center gap-2 py-4 text-sm text-zinc-400">
              <Loader2 size={15} className="animate-spin" />
              Resolving references via Semantic Scholar…
            </div>
          )}

          {error && (
            <div className="py-3 text-sm text-red-400">{error}</div>
          )}

          {!loading && !error && refs.length === 0 && (
            <div className="py-3 text-sm text-zinc-500">
              No references found. The paper may not be indexed by Semantic Scholar.
            </div>
          )}

          {!loading && refsWithPdf.length > 0 && (
            <div className="space-y-2">
              {refsWithPdf.map((ref, rawIndex) => {
                // rawIndex is index within refsWithPdf; we need the original index in `refs`
                const originalIndex = refs.indexOf(ref);
                const state = ingestState[originalIndex] ?? "idle";
                const isActive = ref.secondary_local_id
                  ? activeSecondaryIds.has(ref.secondary_local_id)
                  : false;

                return (
                  <div
                    key={`ref-${originalIndex}`}
                    id={`ref-card-${originalIndex}`}
                    className={`rounded-lg border p-3 transition ${
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
                          {ref.authors.slice(0, 3).join(", ")}
                          {ref.authors.length > 3 ? " et al." : ""}
                          {ref.year ? ` · ${ref.year}` : ""}
                        </div>
                      </div>

                      <div className="flex shrink-0 items-center gap-1.5">
                        {ref.abs_url && (
                          <a
                            href={ref.abs_url}
                            target="_blank"
                            rel="noreferrer"
                            title="Open on arXiv"
                            className="rounded p-1 text-zinc-500 hover:text-white"
                          >
                            <ExternalLink size={13} />
                          </a>
                        )}

                        {isActive ? (
                          <span className="flex items-center gap-1 rounded-md bg-acid/20 px-2 py-1 text-[11px] font-semibold text-acid">
                            <CheckCircle2 size={12} />
                            Active
                          </span>
                        ) : state === "loading" ? (
                          <span className="flex items-center gap-1 rounded-md border border-line px-2 py-1 text-[11px] text-zinc-400">
                            <Loader2 size={12} className="animate-spin" />
                            Loading…
                          </span>
                        ) : state === "error" ? (
                          <button
                            type="button"
                            onClick={() => ingestRef(originalIndex)}
                            className="rounded-md border border-red-500/40 bg-red-500/10 px-2 py-1 text-[11px] text-red-400 hover:bg-red-500/20"
                          >
                            Retry
                          </button>
                        ) : (
                          <button
                            type="button"
                            id={`ingest-ref-${originalIndex}`}
                            onClick={() => ingestRef(originalIndex)}
                            className="flex items-center gap-1 rounded-md border border-zinc-600 bg-panel px-2 py-1 text-[11px] text-zinc-300 transition hover:border-acid hover:text-acid"
                            title="Download and add this paper to the session"
                          >
                            <Plus size={12} />
                            Load paper
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

          {!loading && refsNoPdf.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-zinc-600 hover:text-zinc-400">
                {refsNoPdf.length} references without a downloadable PDF
              </summary>
              <div className="mt-2 space-y-1">
                {refsNoPdf.map((ref, idx) => (
                  <div key={`nopdf-${idx}`} className="text-xs text-zinc-600">
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
