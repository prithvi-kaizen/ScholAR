"use client";

import { useEffect, useMemo, useState } from "react";
import { ActionTiles } from "../components/ActionTiles";
import { Navbar } from "../components/Navbar";
import { PaperCard } from "../components/PaperCard";
import { PaperModal } from "../components/PaperModal";
import { SearchBar } from "../components/SearchBar";
import type { Paper } from "../types/paper";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const RECENT_KEY = "scholar_recent";
const BOOKMARK_KEY = "scholar_bookmarks";

const editorPicks: Paper[] = [
  {
    id: "1706.03762",
    title: "Attention Is All You Need",
    authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    year: "2017",
    summary:
      "The Transformer architecture replaces recurrent sequence modeling with attention mechanisms and becomes a foundation for modern language models.",
    categories: ["cs.CL", "cs.LG"],
    pdf_url: "https://arxiv.org/pdf/1706.03762",
    abs_url: "https://arxiv.org/abs/1706.03762"
  },
  {
    id: "2005.11401",
    title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    authors: ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
    year: "2020",
    summary:
      "RAG combines parametric generation with non-parametric retrieval for tasks where factual grounding matters.",
    categories: ["cs.CL", "cs.AI"],
    pdf_url: "https://arxiv.org/pdf/2005.11401",
    abs_url: "https://arxiv.org/abs/2005.11401"
  },
  {
    id: "2302.13971",
    title: "LLaMA: Open and Efficient Foundation Language Models",
    authors: ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard"],
    year: "2023",
    summary:
      "LLaMA introduces a family of efficient foundation language models trained on public data at multiple scales.",
    categories: ["cs.CL", "cs.LG"],
    pdf_url: "https://arxiv.org/pdf/2302.13971",
    abs_url: "https://arxiv.org/abs/2302.13971"
  }
];

function readPapers(key: string): Paper[] {
  try {
    const stored = window.localStorage.getItem(key);
    return stored ? (JSON.parse(stored) as Paper[]) : [];
  } catch {
    return [];
  }
}

function writePapers(key: string, papers: Paper[]) {
  window.localStorage.setItem(key, JSON.stringify(papers));
}

export default function HomePage() {
  const [papers, setPapers] = useState<Paper[]>(editorPicks);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [recentlyViewed, setRecentlyViewed] = useState<Paper[]>([]);
  const [activeTab, setActiveTab] = useState("Editor's Picks");
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setRecentlyViewed(readPapers(RECENT_KEY));
  }, []);

  const visiblePapers = useMemo(() => {
    if (searched) return papers;
    if (activeTab === "Popular") return [...editorPicks].reverse();
    if (activeTab === "New") return [...editorPicks].sort((a, b) => Number(b.year) - Number(a.year));
    return editorPicks;
  }, [activeTab, papers, searched]);

  function rememberPaper(paper: Paper) {
    const next = [paper, ...recentlyViewed.filter((item) => item.id !== paper.id)].slice(0, 6);
    setRecentlyViewed(next);
    writePapers(RECENT_KEY, next);
  }

  function handleSelect(paper: Paper) {
    setSelectedPaper(paper);
    rememberPaper(paper);
  }

  function bookmarkPaper(paper: Paper) {
    const bookmarks = readPapers(BOOKMARK_KEY);
    const next = [paper, ...bookmarks.filter((item) => item.id !== paper.id)];
    writePapers(BOOKMARK_KEY, next);
  }

  async function search(query: string) {
    const trimmed = query.trim();
    if (!trimmed) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${backendUrl}/api/search?q=${encodeURIComponent(trimmed)}&max_results=12`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "Search failed");
      }
      const payload = (await response.json()) as { papers: Paper[] };
      setPapers(payload.papers ?? []);
      setSearched(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Search failed");
      setSearched(false);
      setPapers(editorPicks);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-ink text-white">
      <Navbar />
      <section className="mx-auto w-full max-w-7xl px-5 py-8">
        <div className="mb-8">
          <SearchBar onSearch={search} loading={loading} />
        </div>

        <div className="mb-8">
          <ActionTiles />
        </div>

        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex rounded-lg border border-line bg-panel p-1">
            {["Editor's Picks", "Popular", "New"].map((tab) => (
              <button
                key={tab}
                onClick={() => {
                  setActiveTab(tab);
                  if (!searched) setPapers(editorPicks);
                }}
                className={`rounded-md px-4 py-2 text-sm transition ${
                  activeTab === tab && !searched ? "bg-white text-black" : "text-zinc-400 hover:text-white"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
          {searched ? <span className="text-sm text-zinc-500">{papers.length} arXiv results</span> : null}
        </div>

        {error ? <div className="mb-5 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div> : null}

        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {[...Array(8)].map((_, index) => (
              <div key={index} className="h-48 animate-pulse rounded-lg border border-line bg-panel" />
            ))}
          </div>
        ) : visiblePapers.length ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {visiblePapers.map((paper) => (
              <PaperCard key={paper.id} paper={paper} onSelect={handleSelect} />
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-line bg-panel px-6 py-12 text-center text-zinc-400">
            No papers found. Try a different arXiv query.
          </div>
        )}

        <section className="mt-10">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Recently Viewed Papers</h2>
            <span className="text-xs text-zinc-500">Stored locally</span>
          </div>
          {recentlyViewed.length ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {recentlyViewed.slice(0, 3).map((paper) => (
                <PaperCard key={`recent-${paper.id}`} paper={paper} onSelect={handleSelect} />
              ))}
            </div>
          ) : (
            <div className="rounded-lg border border-line bg-panel px-5 py-6 text-sm text-zinc-500">
              Open a paper to start your local reading trail.
            </div>
          )}
        </section>
      </section>

      <PaperModal paper={selectedPaper} onClose={() => setSelectedPaper(null)} onBookmark={bookmarkPaper} onViewed={rememberPaper} />
    </main>
  );
}
