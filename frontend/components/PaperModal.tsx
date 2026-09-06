"use client";

import { ExternalLink, Sparkles, Star, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { getApiErrorMessage } from "../types/api";
import type { Paper } from "../types/paper";
import { useNetworkPolicy } from "../lib/networkPolicy";
import { Badge } from "./Badge";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

const prepareSteps = [
  "Downloading PDF from arXiv & verifying integrity",
  "Extracting text, formulas & building AST index",
  "Rasterizing 3× high-resolution tables & figures",
  "Synthesizing 8-goal recursive study roadmap",
  "Opening personalized workspace",
];

interface PaperModalProps {
  paper: Paper | null;
  onClose: () => void;
  onBookmark: (paper: Paper) => void;
  isBookmarked: boolean;
  onViewed: (paper: Paper) => void;
}

export function PaperModal({ paper, onClose, onBookmark, isBookmarked, onViewed }: PaperModalProps) {
  const router = useRouter();
  const [preparing, setPreparing] = useState(false);
  const [prepareStep, setPrepareStep] = useState(0);
  const [error, setError] = useState("");
  const { policy } = useNetworkPolicy();

  if (!paper) return null;
  const currentPaper = paper;

  async function preparePaper() {
    setPreparing(true);
    setPrepareStep(0);
    setError("");
    onViewed(currentPaper);
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), 90_000);
    try {
      setPrepareStep(1);
      const response = await fetch(`${backendUrl}/api/papers/prepare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentPaper),
        signal: timeoutController.signal
      });
      if (!response.ok) {
        const payload: unknown = await response.json().catch(() => ({}));
        throw new Error(getApiErrorMessage(payload, "Could not prepare paper"));
      }
      setPrepareStep(2);
      const payload = (await response.json()) as { paper_id: string };
      setPrepareStep(3);
      // Pre-warm study goals so the workspace opens with real, paper-specific
      // goals already cached instead of generic placeholders.
      await fetch(`${backendUrl}/api/papers/${encodeURIComponent(payload.paper_id)}/study-goals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      }).catch(() => null);
      setPrepareStep(4);
      router.push(`/paper/${encodeURIComponent(payload.paper_id)}`);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        setError("Preparing the paper timed out. Please try again.");
      } else {
        setError(caught instanceof Error ? caught.message : "Could not prepare paper");
      }
      setPreparing(false);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  if (preparing) {
    return (
      <div className="fixed inset-0 z-50 grid place-items-center bg-black/80 px-4 backdrop-blur-sm">
        <div className="w-full max-w-lg rounded-lg border border-line bg-panel p-6 shadow-glow">
          <div className="mb-5 flex items-center gap-4">
            <div className="relative h-14 w-14 shrink-0">
              <div className="absolute inset-0 rounded-full border-2 border-acid/20" />
              <div className="absolute inset-0 animate-spin rounded-full border-2 border-acid border-t-transparent" />
              <div className="absolute inset-3 rounded-full bg-acid/15" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-acid">Preparing study workspace</p>
              <h2 className="mt-1 text-xl font-semibold text-white">{prepareSteps[prepareStep]}</h2>
              <p className="mt-1 text-sm text-zinc-400">Reading the paper and generating study goals tailored to it.</p>
            </div>
          </div>
          <div className="mb-4 h-2 overflow-hidden rounded-full bg-white/10">
            <div className="h-full rounded-full bg-acid transition-all duration-500" style={{ width: `${((prepareStep + 1) / prepareSteps.length) * 100}%` }} />
          </div>
          <div className="space-y-2">
            {prepareSteps.map((step, index) => (
              <div key={step} className={`flex items-center gap-2 text-sm ${index <= prepareStep ? "text-zinc-100" : "text-zinc-600"}`}>
                <span className={`h-2 w-2 rounded-full ${index <= prepareStep ? "bg-acid" : "bg-zinc-700"}`} />
                {step}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
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
          {policy ? (
            <p className={`rounded-md border px-3 py-2 text-xs ${
              policy.mode === "strict-local"
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
                : "border-amber-500/30 bg-amber-500/10 text-amber-200"
            }`}>
              {policy.mode === "strict-local"
                ? "Strict-local: prepared papers open locally; an uncached paper requires a separate acquisition-enabled session."
                : "Acquisition-enabled: preparing an uncached paper may download its PDF from the listed external source."}
            </p>
          ) : null}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => onBookmark(currentPaper)}
              className={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm hover:border-zinc-500 ${
                isBookmarked ? "border-acid/40 bg-acid/10 text-acid" : "border-line text-zinc-200"
              }`}
            >
              <Star size={16} fill={isBookmarked ? "currentColor" : "none"} />
              {isBookmarked ? "Bookmarked" : "Bookmark"}
            </button>
            {currentPaper.abs_url ? (
              <a
                href={currentPaper.abs_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-md border border-line px-3 py-2 text-sm text-zinc-200 hover:border-zinc-500"
              >
                <ExternalLink size={16} />
                Go to Paper Page
              </a>
            ) : null}
            <button
              onClick={preparePaper}
              className="inline-flex items-center gap-2 rounded-md bg-white px-3 py-2 text-sm font-semibold text-black hover:bg-acid"
            >
              <Sparkles size={16} />
              Study with AI
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
