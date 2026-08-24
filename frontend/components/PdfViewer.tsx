"use client";

import { UIEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Crop,
  FileText,
  MessageSquare,
  Sparkles,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { Citation, CustomSnippet } from "../types/paper";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

interface PdfViewerProps {
  paperId: string;
  activeCitation: Citation | null;
  onSnippetToChat?: (snippet: CustomSnippet) => void;
  activeSnippet?: CustomSnippet | null;
}

export function PdfViewer({ paperId, activeCitation, onSnippetToChat, activeSnippet }: PdfViewerProps) {
  const [page, setPage] = useState(1);
  const [pageInput, setPageInput] = useState("1");
  const [isEditingPage, setIsEditingPage] = useState(false);
  const [totalPages, setTotalPages] = useState(1);
  const [zoom, setZoom] = useState(1.65);
  const [title, setTitle] = useState("Research Paper");
  const [erroredPages, setErroredPages] = useState<Set<number>>(new Set());

  // Interactive Snippet / Marquee tool state
  const [isSnipMode, setIsSnipMode] = useState(false);
  const [snipDrag, setSnipDrag] = useState<{
    pageNumber: number;
    startX: number;
    startY: number;
    currX: number;
    currY: number;
  } | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<{
    pageNumber: number;
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);
  const [isCropping, setIsCropping] = useState(false);
  const [snipCopied, setSnipCopied] = useState(false);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pageInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadMetadata() {
      try {
        const response = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}`);
        if (!response.ok) throw new Error("Paper metadata is unavailable");
        const metadata = await response.json();
        if (cancelled) return;
        const total = Math.max(Number(metadata.pages ?? 1), 1);
        setTotalPages(total);
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

  const scrollToPage = useCallback((pageNumber: number) => {
    const safePage = Math.min(Math.max(pageNumber, 1), totalPages);
    setPage(safePage);
    setPageInput(String(safePage));
    setIsEditingPage(false);
    setErroredPages(new Set());
    document.getElementById(`pdf-page-${safePage}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [totalPages]);

  useEffect(() => {
    if (!activeCitation?.page) return;
    scrollToPage(activeCitation.page);
  }, [activeCitation, scrollToPage]);

  // Keyboard navigation and snip shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't capture when typing in text inputs
      if (["INPUT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName)) return;

      if (e.key === "ArrowLeft" || e.key === "j" || e.key === "J") {
        e.preventDefault();
        scrollToPage(page - 1);
      } else if (e.key === "ArrowRight" || e.key === "k" || e.key === "K") {
        e.preventDefault();
        scrollToPage(page + 1);
      } else if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        setZoom((v) => Math.min(v + 0.15, 2.7));
      } else if (e.key === "-") {
        e.preventDefault();
        setZoom((v) => Math.max(v - 0.15, 0.9));
      } else if (e.key === "0") {
        e.preventDefault();
        setZoom(1.65);
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        setIsSnipMode((prev) => !prev);
        setSelectedRegion(null);
        setSnipDrag(null);
      } else if (e.key === "Escape") {
        if (selectedRegion) {
          setSelectedRegion(null);
        } else if (isSnipMode) {
          setIsSnipMode(false);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [page, scrollToPage, isSnipMode, selectedRegion]);

  function handlePageInputSubmit(e: React.FormEvent) {
    e.preventDefault();
    const parsed = parseInt(pageInput, 10);
    if (!isNaN(parsed) && parsed >= 1 && parsed <= totalPages) {
      scrollToPage(parsed);
    } else {
      setPageInput(String(page));
      setIsEditingPage(false);
    }
  }

  function pageImageUrl(pageNumber: number) {
    const params = new URLSearchParams({ zoom: zoom.toFixed(2) });
    if (activeCitation?.page === pageNumber && activeCitation.quote) {
      params.set("highlight", activeCitation.quote);
      params.set("highlightVersion", "2");
    }
    return `${backendUrl}/api/papers/${encodeURIComponent(paperId)}/page/${pageNumber}.png?${params.toString()}`;
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
    if (closestPage !== page) {
      setPage(closestPage);
      if (!isEditingPage) setPageInput(String(closestPage));
    }
  }

  // Snip drag calculations
  const handlePageMouseDown = (pageNumber: number, e: React.MouseEvent<HTMLDivElement>) => {
    if (!isSnipMode) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min((e.clientX - rect.left) / rect.width, 1));
    const y = Math.max(0, Math.min((e.clientY - rect.top) / rect.height, 1));
    setSelectedRegion(null);
    setSnipDrag({
      pageNumber,
      startX: x,
      startY: y,
      currX: x,
      currY: y,
    });
  };

  const handlePageMouseMove = (pageNumber: number, e: React.MouseEvent<HTMLDivElement>) => {
    if (!isSnipMode || !snipDrag || snipDrag.pageNumber !== pageNumber) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min((e.clientX - rect.left) / rect.width, 1));
    const y = Math.max(0, Math.min((e.clientY - rect.top) / rect.height, 1));
    setSnipDrag((prev) => (prev ? { ...prev, currX: x, currY: y } : null));
  };

  const handlePageMouseUp = (pageNumber: number) => {
    if (!isSnipMode || !snipDrag || snipDrag.pageNumber !== pageNumber) return;
    const x0 = Math.min(snipDrag.startX, snipDrag.currX);
    const y0 = Math.min(snipDrag.startY, snipDrag.currY);
    const x1 = Math.max(snipDrag.startX, snipDrag.currX);
    const y1 = Math.max(snipDrag.startY, snipDrag.currY);

    if (x1 - x0 > 0.02 && y1 - y0 > 0.015) {
      setSelectedRegion({ pageNumber, x0, y0, x1, y1 });
    }
    setSnipDrag(null);
  };

  // Action: Crop & send snippet to chat
  const handleSendSnippetToChat = async () => {
    if (!selectedRegion) return;
    setIsCropping(true);
    try {
      const resp = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/snippets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          page: selectedRegion.pageNumber,
          bbox: [selectedRegion.x0, selectedRegion.y0, selectedRegion.x1, selectedRegion.y1],
          zoom: 3.0,
        }),
      });
      if (!resp.ok) throw new Error("Failed to crop snippet");
      const data = await resp.json();
      onSnippetToChat?.({
        id: data.snippet_id,
        page: selectedRegion.pageNumber,
        bbox: [selectedRegion.x0, selectedRegion.y0, selectedRegion.x1, selectedRegion.y1],
        imageUrl: data.image_url,
        text: data.text,
      });
      setSelectedRegion(null);
      setIsSnipMode(false);
    } catch (err) {
      console.error("Snippet crop error:", err);
    } finally {
      setIsCropping(false);
    }
  };

  // Action: Copy snippet image to clipboard
  const handleCopySnippetImage = async () => {
    if (!selectedRegion) return;
    setIsCropping(true);
    try {
      const resp = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/snippets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          page: selectedRegion.pageNumber,
          bbox: [selectedRegion.x0, selectedRegion.y0, selectedRegion.x1, selectedRegion.y1],
          zoom: 3.0,
        }),
      });
      if (!resp.ok) throw new Error("Failed to crop snippet");
      const data = await resp.json();
      const imgResp = await fetch(`${backendUrl}${data.image_url}`);
      const blob = await imgResp.blob();
      await navigator.clipboard.write([
        new ClipboardItem({ [blob.type]: blob }),
      ]);
      setSnipCopied(true);
      setTimeout(() => setSnipCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy image:", err);
    } finally {
      setIsCropping(false);
    }
  };

  return (
    <section className="grid min-h-0 h-full grid-cols-[54px_minmax(0,1fr)] border-r border-line bg-panelSoft">
      <aside className="flex flex-col items-center gap-3 border-r border-line bg-panel py-3">
        <button
          className="grid h-10 w-10 place-items-center rounded-md border border-blue-400 bg-blue-500/20 text-blue-100"
          aria-label="Pages"
          title="PDF Page View"
        >
          <FileText size={18} />
        </button>
      </aside>

      <div className="flex min-h-0 flex-col">
        <div className="flex h-16 shrink-0 flex-wrap items-center justify-between gap-3 border-b border-line bg-panelSoft px-4">
          <div className="flex items-center gap-2 text-sm text-zinc-300">
            <button
              onClick={() => scrollToPage(page - 1)}
              className="rounded-md p-2 text-zinc-400 transition hover:bg-white/5 hover:text-white"
              aria-label="Previous page (J or Left Arrow)"
              title="Previous page (J / Left Arrow)"
            >
              <ChevronLeft size={18} />
            </button>

            {isEditingPage ? (
              <form onSubmit={handlePageInputSubmit} className="inline-block">
                <input
                  ref={pageInputRef}
                  type="text"
                  value={pageInput}
                  onChange={(e) => setPageInput(e.target.value)}
                  onBlur={handlePageInputSubmit}
                  className="w-14 rounded-md border border-blue-500 bg-panel px-2 py-1 text-center font-medium text-white shadow-inner focus:outline-none"
                  autoFocus
                />
              </form>
            ) : (
              <button
                onClick={() => {
                  setIsEditingPage(true);
                  setTimeout(() => pageInputRef.current?.select(), 50);
                }}
                className="rounded-md border border-line bg-panel px-3.5 py-1.5 font-medium text-white transition hover:border-zinc-500"
                title="Click to jump to page"
              >
                {page}
              </button>
            )}

            <span className="text-zinc-500 font-medium">/ {totalPages}</span>
            <button
              onClick={() => scrollToPage(page + 1)}
              className="rounded-md p-2 text-zinc-400 transition hover:bg-white/5 hover:text-white"
              aria-label="Next page (K or Right Arrow)"
              title="Next page (K / Right Arrow)"
            >
              <ChevronRight size={18} />
            </button>
          </div>

          {/* Center / Right Toolbar: Snip Tool and Zoom */}
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            {/* Snip Region / Screenshot Tool */}
            <button
              onClick={() => {
                setIsSnipMode((prev) => !prev);
                setSelectedRegion(null);
                setSnipDrag(null);
              }}
              className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold transition ${
                isSnipMode
                  ? "border border-blue-500 bg-blue-500/25 text-blue-200 shadow-sm ring-2 ring-blue-500/20"
                  : "border border-line/60 bg-panel text-zinc-300 hover:border-zinc-500 hover:text-white"
              }`}
              title="Snip Region (S) - Click & drag on PDF to ask about a specific equation or chart"
            >
              <Crop size={14} className={isSnipMode ? "animate-pulse text-blue-300" : ""} />
              <span>{isSnipMode ? "Snipping Active (Esc)" : "Snip Region"}</span>
            </button>

            <div className="h-4 w-px bg-line/60 mx-1" />

            <button
              onClick={() => setZoom((value) => Math.max(value - 0.15, 0.9))}
              className="rounded-md p-2 transition hover:bg-white/5 hover:text-white"
              aria-label="Zoom out (-)"
              title="Zoom out (-)"
            >
              <ZoomOut size={17} />
            </button>
            <button
              onClick={() => setZoom(1.65)}
              className="w-16 rounded-md px-2 py-1 text-center text-xs font-medium text-zinc-300 transition hover:bg-zinc-800"
              title="Reset Zoom (0)"
            >
              {Math.round((zoom / 1.65) * 100)}%
            </button>
            <button
              onClick={() => setZoom((value) => Math.min(value + 0.15, 2.7))}
              className="rounded-md p-2 transition hover:bg-white/5 hover:text-white"
              aria-label="Zoom in (+)"
              title="Zoom in (+)"
            >
              <ZoomIn size={17} />
            </button>
          </div>
        </div>

        <div className="flex h-12 shrink-0 items-center justify-between border-b border-line bg-panel px-4">
          <div className="min-w-0 truncate text-sm font-medium text-zinc-300">{title}</div>
          {isSnipMode && (
            <div className="flex items-center gap-1.5 text-xs text-blue-400 font-medium">
              <Sparkles size={13} />
              <span>Click and drag on any page to select an equation, chart, or text block</span>
            </div>
          )}
        </div>

        <div ref={scrollRef} onScroll={handleScroll} className="min-h-0 flex-1 overflow-auto bg-[rgb(var(--color-paper-bg))]">
          <div className="mx-auto flex min-h-full w-full flex-col items-center gap-6 px-4 py-6">
            {pages.map((pageNumber) => {
              const isDraggingOnThisPage = snipDrag && snipDrag.pageNumber === pageNumber;
              const dragBox = isDraggingOnThisPage
                ? {
                    left: `${Math.min(snipDrag.startX, snipDrag.currX) * 100}%`,
                    top: `${Math.min(snipDrag.startY, snipDrag.currY) * 100}%`,
                    width: `${Math.abs(snipDrag.currX - snipDrag.startX) * 100}%`,
                    height: `${Math.abs(snipDrag.currY - snipDrag.startY) * 100}%`,
                  }
                : null;

              const isSelectedOnThisPage = selectedRegion && selectedRegion.pageNumber === pageNumber;
              const selectedBox = isSelectedOnThisPage
                ? {
                    left: `${selectedRegion.x0 * 100}%`,
                    top: `${selectedRegion.y0 * 100}%`,
                    width: `${(selectedRegion.x1 - selectedRegion.x0) * 100}%`,
                    height: `${(selectedRegion.y1 - selectedRegion.y0) * 100}%`,
                  }
                : null;

              const isSnippetHighlight = activeSnippet && activeSnippet.page === pageNumber;

              return (
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
                    {isSnippetHighlight && (
                      <span className="ml-2 rounded-md border border-amber-400/40 bg-amber-500/15 px-2 py-0.5 text-amber-200">
                        active snippet
                      </span>
                    )}
                  </div>
                  {erroredPages.has(pageNumber) ? (
                    <div className="mx-auto max-w-md rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-100">
                      Could not render page {pageNumber}. Try preparing the paper again.
                    </div>
                  ) : (
                    <div
                      onMouseDown={(e) => handlePageMouseDown(pageNumber, e)}
                      onMouseMove={(e) => handlePageMouseMove(pageNumber, e)}
                      onMouseUp={() => handlePageMouseUp(pageNumber)}
                      className={`relative mx-auto inline-block max-w-full select-none ${
                        isSnipMode ? "cursor-crosshair" : ""
                      }`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={pageImageUrl(pageNumber)}
                        alt={`Page ${pageNumber} of ${title}`}
                        loading={pageNumber <= 2 ? "eager" : "lazy"}
                        onError={() => setErroredPages((prev) => new Set(prev).add(pageNumber))}
                        draggable={false}
                        className={`block h-auto max-w-full bg-white shadow-2xl shadow-black/60 transition-all ${
                          activeCitation?.page === pageNumber ? "ring-2 ring-blue-400" : ""
                        }`}
                      />

                      {/* Active Citation Bounding Box */}
                      {activeCitation?.page === pageNumber && activeCitation.bbox_normalized && (
                        <div
                          className="pointer-events-none absolute rounded border-2 border-dashed border-amber-400 bg-amber-400/10 transition-all duration-300"
                          style={{
                            left: `${activeCitation.bbox_normalized.x0 * 100}%`,
                            top: `${activeCitation.bbox_normalized.y0 * 100}%`,
                            width: `${(activeCitation.bbox_normalized.x1 - activeCitation.bbox_normalized.x0) * 100}%`,
                            height: `${(activeCitation.bbox_normalized.y1 - activeCitation.bbox_normalized.y0) * 100}%`,
                          }}
                        >
                          <span className="absolute -top-5 left-0 rounded bg-amber-500 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-black shadow">
                            {activeCitation.label || "Figure"}
                          </span>
                        </div>
                      )}

                      {/* Active Snippet Bounding Box */}
                      {isSnippetHighlight && activeSnippet && (
                        <div
                          className="pointer-events-none absolute rounded border-2 border-blue-400 bg-blue-400/15 shadow-[0_0_15px_rgba(59,130,246,0.5)] transition-all duration-300"
                          style={{
                            left: `${activeSnippet.bbox[0] * 100}%`,
                            top: `${activeSnippet.bbox[1] * 100}%`,
                            width: `${(activeSnippet.bbox[2] - activeSnippet.bbox[0]) * 100}%`,
                            height: `${(activeSnippet.bbox[3] - activeSnippet.bbox[1]) * 100}%`,
                          }}
                        >
                          <span className="absolute -top-5 left-0 rounded bg-blue-500 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white shadow">
                            Snippet
                          </span>
                        </div>
                      )}

                      {/* Subregion Grounding Overlays */}
                      {activeCitation?.page === pageNumber &&
                        activeCitation.subregions?.map((sub, sIdx) => (
                          <div
                            key={sub.region_id ?? sIdx}
                            className="pointer-events-none absolute rounded border-2 border-emerald-400 bg-emerald-400/20 shadow-[0_0_12px_rgba(52,211,153,0.6)] transition-all duration-300"
                            style={{
                              left: `${sub.bbox.x0 * 100}%`,
                              top: `${sub.bbox.y0 * 100}%`,
                              width: `${(sub.bbox.x1 - sub.bbox.x0) * 100}%`,
                              height: `${(sub.bbox.y1 - sub.bbox.y0) * 100}%`,
                            }}
                          >
                            <span className="absolute -bottom-5 right-0 rounded bg-emerald-500 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-black shadow">
                              {sub.role || "subregion"}
                            </span>
                          </div>
                        ))}

                      {/* Active Drag Marquee Box */}
                      {isDraggingOnThisPage && dragBox && (
                        <div
                          className="pointer-events-none absolute border-2 border-blue-400 bg-blue-500/20 shadow-lg transition-none"
                          style={dragBox}
                        />
                      )}

                      {/* Completed Selection Box with Floating Action Bar */}
                      {isSelectedOnThisPage && selectedBox && (
                        <div
                          className="absolute border-2 border-dashed border-blue-400 bg-blue-500/20 shadow-[0_0_20px_rgba(59,130,246,0.4)]"
                          style={selectedBox}
                        >
                          {/* Floating Action Menu Bar */}
                          <div
                            className="absolute -bottom-12 left-1/2 -translate-x-1/2 flex items-center gap-1.5 rounded-lg border border-line bg-zinc-900/95 p-1.5 shadow-2xl backdrop-blur-md z-30 whitespace-nowrap"
                            onMouseDown={(e) => e.stopPropagation()}
                          >
                            <button
                              type="button"
                              onClick={handleSendSnippetToChat}
                              disabled={isCropping}
                              className="flex items-center gap-1.5 rounded-md bg-blue-600 px-2.5 py-1 text-xs font-semibold text-white shadow transition hover:bg-blue-500 disabled:opacity-50"
                            >
                              <MessageSquare size={13} />
                              <span>{isCropping ? "Cropping..." : "Ask ScholAR"}</span>
                            </button>

                            <button
                              type="button"
                              onClick={handleCopySnippetImage}
                              disabled={isCropping}
                              className="flex items-center gap-1 rounded-md border border-line bg-zinc-800 px-2 py-1 text-xs font-medium text-zinc-200 transition hover:bg-zinc-700 disabled:opacity-50"
                              title="Copy cropped image to clipboard"
                            >
                              {snipCopied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
                              <span>{snipCopied ? "Copied!" : "Copy"}</span>
                            </button>

                            <button
                              type="button"
                              onClick={() => setSelectedRegion(null)}
                              className="rounded-md p-1 text-zinc-400 transition hover:bg-zinc-800 hover:text-white"
                              title="Cancel selection (Esc)"
                            >
                              <X size={14} />
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
