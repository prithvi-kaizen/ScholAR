"use client";

import { ExternalLink, Share2, Sparkles, Star, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type { Paper } from "../types/paper";
import { Badge } from "./Badge";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

interface PaperModalProps {
  paper: Paper | null;
  onClose: () => void;
  onBookmark: (paper: Paper) => void;
  onViewed: (paper: Paper) => void;
}

export function PaperModal({ paper, onClose, onBookmark, onViewed }: PaperModalProps) {
  const router = useRouter();
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState("");

  if (!paper) return null;
  const currentPaper = paper;

  async function preparePaper() {
    setPreparing(true);
    setError("");
    onViewed(currentPaper);
    try {
      const response = await fetch(`${backendUrl}/api/papers/prepare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentPaper)
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "Could not prepare paper");
      }
      const payload = (await response.json()) as { paper_id: string };
      router.push(`/paper/${encodeURIComponent(payload.paper_id)}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not prepare paper");
    } finally {
      setPreparing(false);
    }
  }

  function sharePaper() {
    void navigator.clipboard?.writeText(currentPaper.abs_url);
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/75 px-4 py-8 backdrop-blur-sm">
      <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-line bg-panel shadow-glow">
        <div className="flex items-start justify-between gap-4 border-b border-line p-5">
          <div>
            <div className="mb-3 flex flex-wrap gap-1.5">
              {currentPaper.categories.slice(0, 5).map((category) => (
                <Badge key={`${currentPaper.id}-${category}`} label={category} />
              ))}
            </div>
            <h2 className="text-xl font-semibold leading-8 text-white">{currentPaper.title}</h2>
            <p className="mt-2 text-sm text-zinc-400">
              {currentPaper.year} · {currentPaper.authors.join(", ") || "Unknown authors"}
            </p>
          </div>
          <button onClick={onClose} className="rounded-md border border-line p-2 text-zinc-400 hover:text-white" aria-label="Close modal">
            <X size={18} />
          </button>
        </div>

        <div className="space-y-5 p-5">
          <p className="text-sm leading-7 text-zinc-300">{currentPaper.summary}</p>
          {error ? <p className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</p> : null}
          <div className="flex flex-wrap gap-2">
            <button onClick={() => onBookmark(currentPaper)} className="inline-flex items-center gap-2 rounded-md border border-line px-3 py-2 text-sm text-zinc-200 hover:border-zinc-500">
              <Star size={16} />
              Bookmark
            </button>
            <button onClick={sharePaper} className="inline-flex items-center gap-2 rounded-md border border-line px-3 py-2 text-sm text-zinc-200 hover:border-zinc-500">
              <Share2 size={16} />
              Share
            </button>
            <a
              href={currentPaper.abs_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-md border border-line px-3 py-2 text-sm text-zinc-200 hover:border-zinc-500"
            >
              <ExternalLink size={16} />
              Go to Paper Page
            </a>
            <button
              onClick={preparePaper}
              disabled={preparing}
              className="inline-flex items-center gap-2 rounded-md bg-white px-3 py-2 text-sm font-semibold text-black hover:bg-acid disabled:cursor-not-allowed disabled:opacity-60"
            >
              <Sparkles size={16} />
              {preparing ? "Preparing paper..." : "Study with AI"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
