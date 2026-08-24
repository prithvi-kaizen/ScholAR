"use client";

import { useState } from "react";
import Link from "next/link";
import { HelpCircle } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { ShortcutsModal } from "./ShortcutsModal";

export function Navbar() {
  const [showShortcuts, setShowShortcuts] = useState(false);

  return (
    <>
      <nav className="flex h-16 items-center justify-between border-b border-line bg-ink/95 px-5">
        <Link href="/" className="flex items-center gap-3 rounded-md outline-none transition hover:opacity-80 focus-visible:ring-2 focus-visible:ring-acid">
          <div className="grid h-8 w-8 place-items-center rounded-md border border-acid/40 bg-acid/10 text-sm font-bold text-acid">
            AR
          </div>
          <span className="text-base font-semibold text-white">ScholAR</span>
        </Link>
        <div className="flex items-center gap-3">
          <Link
            href="/compare"
            className="flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs text-zinc-300 transition hover:border-purple-500/50 hover:text-white"
          >
            <span className="font-semibold text-purple-400">Cross-Paper</span>
            <span>Compare</span>
          </Link>
          <Link
            href="/benchmark"
            className="flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs text-zinc-300 transition hover:border-acid/50 hover:text-white"
          >
            <span className="font-semibold text-acid">EACL &apos;27</span>
            <span>Benchmark</span>
          </Link>
          <Link
            href="/telemetry"
            className="flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs text-zinc-300 transition hover:border-emerald-500/50 hover:text-white"
          >
            <span className="font-semibold text-emerald-400">System</span>
            <span>Telemetry</span>
          </Link>
          <button
            type="button"
            onClick={() => setShowShortcuts(true)}
            className="flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs text-zinc-400 transition hover:border-zinc-500 hover:text-white"
            title="Keyboard shortcuts (?)"
          >
            <HelpCircle size={14} />
            <span className="hidden sm:inline">Shortcuts</span>
            <kbd className="rounded border border-zinc-700 bg-zinc-800 px-1 py-0.2 text-[10px] font-mono text-zinc-300">?</kbd>
          </button>
          <ThemeToggle />
        </div>
      </nav>
      <ShortcutsModal isOpen={showShortcuts} onClose={() => setShowShortcuts(false)} />
    </>
  );
}
