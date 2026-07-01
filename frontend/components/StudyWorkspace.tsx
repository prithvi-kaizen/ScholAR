"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Citation } from "../types/paper";
import { PdfViewer } from "./PdfViewer";
import { StudyPanel } from "./StudyPanel";

interface StudyWorkspaceProps {
  paperId: string;
}

const MIN_PDF_WIDTH  = 340;
const MIN_CHAT_WIDTH = 360;
const HANDLE_WIDTH   = 6;

export function StudyWorkspace({ paperId }: StudyWorkspaceProps) {
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [pdfWidthPct, setPdfWidthPct] = useState(52); // 52% for PDF by default
  const dragging = useRef(false);
  const startX   = useRef(0);
  const startPct = useRef(52);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const totalW = containerRef.current.clientWidth - HANDLE_WIDTH;
    const delta  = e.clientX - startX.current;
    const newPct = startPct.current + (delta / totalW) * 100;
    // Clamp so neither pane gets smaller than its minimum
    const minPdfPct  = (MIN_PDF_WIDTH  / totalW) * 100;
    const minChatPct = 100 - (MIN_CHAT_WIDTH / totalW) * 100;
    setPdfWidthPct(Math.min(Math.max(newPct, minPdfPct), minChatPct));
  }, []);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  function startDrag(e: React.MouseEvent) {
    e.preventDefault();
    dragging.current  = true;
    startX.current    = e.clientX;
    startPct.current  = pdfWidthPct;
    document.body.style.cursor     = "col-resize";
    document.body.style.userSelect = "none";
  }

  return (
    <div ref={containerRef} className="flex min-h-0 flex-1 overflow-hidden">
      {/* PDF pane */}
      <div style={{ width: `${pdfWidthPct}%` }} className="min-h-0 min-w-0 shrink-0 flex flex-col">
        <PdfViewer paperId={paperId} activeCitation={activeCitation} />
      </div>

      {/* Drag handle */}
      <div
        onMouseDown={startDrag}
        className="group relative z-10 flex shrink-0 cursor-col-resize items-center justify-center bg-ink hover:bg-acid/10 transition-colors"
        style={{ width: HANDLE_WIDTH }}
        title="Drag to resize"
      >
        <div className="h-10 w-[2px] rounded-full bg-zinc-700 group-hover:bg-acid/60 transition-colors" />
      </div>

      {/* Study / Chat pane */}
      <div className="min-h-0 min-w-0 flex-1 flex flex-col">
        <StudyPanel paperId={paperId} onCitationClick={setActiveCitation} />
      </div>
    </div>
  );
}
