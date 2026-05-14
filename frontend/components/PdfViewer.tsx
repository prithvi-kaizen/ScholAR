"use client";

import { UIEvent, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, FileText, ZoomIn, ZoomOut } from "lucide-react";
import type { Citation } from "../types/paper";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

interface PdfViewerProps {
  paperId: string;
  activeCitation: Citation | null;
}

export function PdfViewer({ paperId, activeCitation }: PdfViewerProps) {
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [zoom, setZoom] = useState(1.65);
  const [title, setTitle] = useState("Research Paper");
  const [imageError, setImageError] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadMetadata() {
      try {
        const response = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}`);
        if (!response.ok) throw new Error("Paper metadata is unavailable");
        const metadata = await response.json();
        if (cancelled) return;
        setTotalPages(Math.max(Number(metadata.pages ?? 1), 1));
        setTitle(metadata.title ?? "Research Paper");
      } catch {
        if (!cancelled) setTotalPages(1);
      }
    }
    void loadMetadata();
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  const pages = useMemo(() => Array.from({ length: totalPages }, (_, index) => index + 1), [totalPages]);

  useEffect(() => {
    if (!activeCitation?.page) return;
    scrollToPage(activeCitation.page);
  }, [activeCitation]);

  function pageImageUrl(pageNumber: number) {
    const params = new URLSearchParams({ zoom: zoom.toFixed(2) });
    if (activeCitation?.page === pageNumber && activeCitation.quote) {
      params.set("highlight", activeCitation.quote);
      params.set("highlightVersion", "2");
    }
    return `${backendUrl}/api/papers/${encodeURIComponent(paperId)}/page/${pageNumber}.png?${params.toString()}`;
  }

  function scrollToPage(pageNumber: number) {
    const safePage = Math.min(Math.max(pageNumber, 1), totalPages);
    setPage(safePage);
    setImageError("");
    document.getElementById(`pdf-page-${safePage}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function handleScroll(event: UIEvent<HTMLDivElement>) {
    const container = event.currentTarget;
    const pageNodes = Array.from(container.querySelectorAll<HTMLElement>("[data-page-number]"));
    let closestPage = page;
    let closestDistance = Number.POSITIVE_INFINITY;
    for (const node of pageNodes) {
      const distance = Math.abs(node.offsetTop - container.scrollTop - 24);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestPage = Number(node.dataset.pageNumber);
      }
    }
    if (closestPage !== page) setPage(closestPage);
  }

  return (
    <section className="grid min-h-0 grid-cols-[54px_minmax(0,1fr)] border-r border-line bg-panelSoft">
      <aside className="flex flex-col items-center gap-3 border-r border-line bg-panel py-3">
        <button className="grid h-10 w-10 place-items-center rounded-md border border-blue-400 bg-blue-500/20 text-blue-100" aria-label="Pages">
          <FileText size={18} />
        </button>
      </aside>

      <div className="flex min-h-0 flex-col">
        <div className="flex h-16 shrink-0 flex-wrap items-center justify-between gap-3 border-b border-line bg-panelSoft px-4">
          <div className="flex items-center gap-2 text-sm text-zinc-300">
            <button onClick={() => scrollToPage(page - 1)} className="rounded-md p-2 text-zinc-400 hover:bg-white/5 hover:text-white" aria-label="Previous page">
              <ChevronLeft size={18} />
            </button>
            <span className="rounded-md border border-line bg-panel px-4 py-2 text-white">{page}</span>
            <span className="text-zinc-500">/ {totalPages}</span>
            <button onClick={() => scrollToPage(page + 1)} className="rounded-md p-2 text-zinc-400 hover:bg-white/5 hover:text-white" aria-label="Next page">
              <ChevronRight size={18} />
            </button>
          </div>

          <div className="flex items-center gap-3 text-sm text-zinc-400">
            <button onClick={() => setZoom((value) => Math.max(value - 0.15, 0.9))} className="rounded-md p-2 hover:bg-white/5 hover:text-white" aria-label="Zoom out">
              <ZoomOut size={17} />
            </button>
            <span className="w-14 text-center">{Math.round((zoom / 1.65) * 100)}%</span>
            <button onClick={() => setZoom((value) => Math.min(value + 0.15, 2.7))} className="rounded-md p-2 hover:bg-white/5 hover:text-white" aria-label="Zoom in">
              <ZoomIn size={17} />
            </button>
          </div>
        </div>

        <div className="flex h-12 shrink-0 items-center border-b border-line bg-panel px-4">
          <div className="min-w-0 truncate text-sm font-medium text-zinc-300">{title}</div>
        </div>

        <div ref={scrollRef} onScroll={handleScroll} className="min-h-0 flex-1 overflow-auto bg-[rgb(var(--color-paper-bg))]">
          <div className="mx-auto flex min-h-full w-full flex-col items-center gap-6 px-4 py-6">
            {imageError ? (
              <div className="mt-16 max-w-md rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-100">
                {imageError}
              </div>
            ) : (
              pages.map((pageNumber) => (
                <div
                  id={`pdf-page-${pageNumber}`}
                  key={`${pageNumber}-${zoom}`}
                  data-page-number={pageNumber}
                  className="w-full scroll-mt-6"
                >
                  <div className="mb-2 text-center text-xs font-medium text-zinc-500">
                    Page {pageNumber}
                    {activeCitation?.page === pageNumber ? (
                      <span className="ml-2 rounded-md border border-blue-400/40 bg-blue-500/15 px-2 py-0.5 text-blue-200">
                        highlighted citation
                      </span>
                    ) : null}
                  </div>
                  <img
                    src={pageImageUrl(pageNumber)}
                    alt={`Page ${pageNumber} of ${title}`}
                    loading={pageNumber <= 2 ? "eager" : "lazy"}
                    onError={() => setImageError("Could not render this PDF page. Try preparing the paper again.")}
                    className={`mx-auto h-auto max-w-full bg-white shadow-2xl shadow-black/60 ${
                      activeCitation?.page === pageNumber ? "ring-2 ring-blue-400" : ""
                    }`}
                  />
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
