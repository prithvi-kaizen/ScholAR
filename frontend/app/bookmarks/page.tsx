"use client";

import { useEffect, useState } from "react";
import { Bookmark, Trash2 } from "lucide-react";
import Link from "next/link";
import { Navbar } from "../../components/Navbar";
import type { Paper } from "../../types/paper";

const BOOKMARK_KEY = "scholar_bookmarks";

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

export default function BookmarksPage() {
  const [bookmarks, setBookmarks] = useState<Paper[]>([]);

  useEffect(() => {
    setBookmarks(readPapers(BOOKMARK_KEY));
  }, []);

  function removeBookmark(paper: Paper) {
    const next = bookmarks.filter((b) => b.id !== paper.id);
    setBookmarks(next);
    writePapers(BOOKMARK_KEY, next);
  }

  return (
    <main className="min-h-screen bg-ink text-white">
      <Navbar />
      <section className="mx-auto w-full max-w-5xl px-5 py-10">
        <div className="mb-8 flex items-center gap-3">
          <Bookmark size={22} className="text-acid" />
          <h1 className="text-2xl font-semibold text-white">Bookmarks</h1>
          <span className="rounded-md bg-white/5 px-2 py-0.5 text-xs text-zinc-400">
            {bookmarks.length} saved
          </span>
        </div>

        {bookmarks.length === 0 ? (
          <div className="rounded-lg border border-line bg-panel px-6 py-16 text-center">
            <Bookmark size={32} className="mx-auto mb-3 text-zinc-600" />
            <p className="text-sm text-zinc-500">No bookmarks yet.</p>
            <p className="mt-1 text-xs text-zinc-600">Bookmark a paper from the homepage to see it here.</p>
            <Link href="/" className="mt-4 inline-block text-xs text-acid hover:underline">
              Back to homepage
            </Link>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {bookmarks.map((paper) => (
              <div
                key={paper.id}
                className="flex flex-col justify-between rounded-lg border border-line bg-panel p-5 transition hover:border-zinc-500"
              >
                <div>
                  <div className="mb-2 flex flex-wrap gap-1.5">
                    {paper.categories.slice(0, 2).map((cat) => (
                      <span key={cat} className="rounded-md bg-white/5 px-2 py-0.5 text-xs text-zinc-400">
                        {cat}
                      </span>
                    ))}
                  </div>
                  <Link href={`/paper/${paper.id}`}>
                    <h2 className="text-sm font-semibold text-white hover:text-acid transition cursor-pointer">
                      {paper.title}
                    </h2>
                  </Link>
                  {paper.authors.length > 0 && (
                    <p className="mt-1 text-xs text-zinc-500">
                      {paper.authors.slice(0, 2).join(", ")}
                    </p>
                  )}
                  {paper.summary && (
                    <p className="mt-2 text-xs leading-5 text-zinc-400 line-clamp-3">
                      {paper.summary}
                    </p>
                  )}
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <span className="text-xs text-zinc-600">{paper.year}</span>
                  <button
                    onClick={() => removeBookmark(paper)}
                    className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 transition"
                  >
                    <Trash2 size={13} />
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}