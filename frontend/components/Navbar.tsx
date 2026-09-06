"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { HelpCircle, GitCompare, BarChart3, Activity, ShieldCheck } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { ShortcutsModal } from "./ShortcutsModal";
import { useNetworkPolicy } from "../lib/networkPolicy";

export function Navbar() {
  const [showShortcuts, setShowShortcuts] = useState(false);
  const pathname = usePathname();
  const { policy, loading: policyLoading, error: policyError } = useNetworkPolicy();

  const policyTitle = policy
    ? `${policy.mode}; missing assets: ${policy.missing_assets.join(", ") || "none"}; external-network actions: ${policy.actions.filter((action) => action.requires_external_network).map((action) => action.action).join(", ")}`
    : policyError ?? "Loading network policy";

  const isCompare = pathname === "/compare";
  const isBenchmark = pathname === "/benchmark";
  const isTelemetry = pathname === "/telemetry";

  return (
    <>
      <nav className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-line/60 bg-ink/85 px-6 backdrop-blur-md transition-colors">
        <div className="flex items-center gap-6">
          <Link
            href="/"
            className="flex items-center gap-3 rounded-xl outline-none transition-all hover:opacity-90 focus-visible:ring-2 focus-visible:ring-acid"
          >
            <div className="grid h-9 w-9 place-items-center rounded-xl border border-acid/30 bg-acid/10 text-sm font-black text-acid shadow-sm shadow-acid/10 transition-transform hover:scale-105">
              AR
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
                ScholAR
                <span className="rounded bg-acid/20 px-1.5 py-0.2 text-[9px] font-mono font-semibold uppercase text-acid tracking-wide">
                  v1.0
                </span>
              </span>
              <span className="text-[10px] text-zinc-400 font-medium hidden sm:inline">
                Local Multi-Level Reasoning Engine
              </span>
            </div>
          </Link>

          <div
            className={`hidden lg:flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${
              policy?.mode === "strict-local"
                ? "border-emerald-500/20 bg-emerald-950/30 text-emerald-400"
                : "border-amber-500/20 bg-amber-950/30 text-amber-300"
            }`}
            title={policyTitle}
          >
            <ShieldCheck size={13} />
            <span>
              {policyLoading
                ? "Checking network policy"
                : policy?.mode === "strict-local"
                  ? "Strict-local analysis"
                  : policy?.mode === "acquisition-enabled"
                    ? "Acquisition enabled"
                    : "Policy unavailable"}
            </span>
            {policy && policy.missing_assets.length > 0 ? (
              <span className="text-[10px] opacity-75">· {policy.missing_assets.length} assets missing</span>
            ) : null}
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          <Link
            href="/compare"
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold transition-all ${
              isCompare
                ? "border-purple-500/50 bg-purple-500/15 text-white shadow-sm shadow-purple-500/10"
                : "border-line/60 bg-panel/80 text-zinc-400 hover:border-purple-500/40 hover:text-zinc-200"
            }`}
          >
            <GitCompare size={13} className={isCompare ? "text-purple-400" : "text-zinc-400"} />
            <span className="hidden md:inline">Cross-Paper</span>
            <span>Compare</span>
          </Link>

          <Link
            href="/benchmark"
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold transition-all ${
              isBenchmark
                ? "border-acid/50 bg-acid/15 text-white shadow-sm shadow-acid/10"
                : "border-line/60 bg-panel/80 text-zinc-400 hover:border-acid/40 hover:text-zinc-200"
            }`}
          >
            <BarChart3 size={13} className={isBenchmark ? "text-acid" : "text-zinc-400"} />
            <span className="font-semibold text-acid">EACL &apos;27</span>
            <span className="hidden sm:inline">Benchmark</span>
          </Link>

          <Link
            href="/telemetry"
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold transition-all ${
              isTelemetry
                ? "border-emerald-500/50 bg-emerald-500/15 text-white shadow-sm shadow-emerald-500/10"
                : "border-line/60 bg-panel/80 text-zinc-400 hover:border-emerald-500/40 hover:text-zinc-200"
            }`}
          >
            <Activity size={13} className={isTelemetry ? "text-emerald-400" : "text-zinc-400"} />
            <span className="hidden md:inline">System</span>
            <span>Telemetry</span>
          </Link>

          <div className="h-4 w-px bg-line/60 mx-1 hidden sm:block" />

          <button
            type="button"
            onClick={() => setShowShortcuts(true)}
            className="flex items-center gap-1.5 rounded-xl border border-line/60 bg-panel/80 px-2.5 py-1.5 text-xs text-zinc-400 transition-all hover:border-zinc-500 hover:text-white"
            title="Keyboard shortcuts (?)"
          >
            <HelpCircle size={14} />
            <kbd className="hidden sm:inline rounded border border-zinc-700 bg-zinc-800/80 px-1 py-0.2 text-[10px] font-mono text-zinc-300">?</kbd>
          </button>
          <ThemeToggle />
        </div>
      </nav>
      <ShortcutsModal isOpen={showShortcuts} onClose={() => setShowShortcuts(false)} />
    </>
  );
}
