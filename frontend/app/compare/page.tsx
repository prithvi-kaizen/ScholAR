"use client";

import { useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  GitCompare,
  Network,
  BookOpen,
  Send,
  Layers,
  Sparkles,
  ArrowRight,
  Download,
  CheckCircle2,
  Table as TableIcon,
} from "lucide-react";
import { Navbar } from "../../components/Navbar";

const BENCHMARK_PAPERS = [
  { id: "1706.03762", title: "Attention Is All You Need (Vaswani et al., 2017)", year: "2017" },
  { id: "2112.10752", title: "High-Res Latent Diffusion Models (Rombach et al., 2022)", year: "2022" },
  { id: "1412.6980", title: "Adam: Stochastic Optimization (Kingma & Ba, 2014)", year: "2014" },
  { id: "1406.2661", title: "Generative Adversarial Nets (Goodfellow et al., 2014)", year: "2014" },
  { id: "2406.08394", title: "VisionLLM v2: Multimodal Model (Wu et al., 2024)", year: "2024" },
  { id: "2104.08663", title: "BEIR: Zero-shot Information Retrieval (Thakur et al., 2021)", year: "2021" },
  { id: "2603.14257", title: "Inter-doc Multi-hop Scientific QA (2026)", year: "2026" },
  { id: "2025.emnlp-main.77", title: "LLM Cross-Document Multi-Entity QA (2025)", year: "2025" },
  { id: "yale_thesis_1003", title: "Towards Multimodal Multi-Doc Understanding (2024)", year: "2024" },
  { id: "2410.00526", title: "PaperQA2: Superhuman Scientific Search (2024)", year: "2024" },
];

export default function CrossPaperComparePage() {
  const [primaryPaperId, setPrimaryPaperId] = useState<string>("1706.03762");
  const [secondaryPaperId, setSecondaryPaperId] = useState<string>("2112.10752");
  const [query, setQuery] = useState<string>(
    "How does the self-attention mechanism in Vaswani 2017 compare with the cross-attention conditioning in Rombach 2022?"
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [graphResult, setGraphResult] = useState<any | null>(null);

  const handleCompare = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}/api/reasoning/cross-document`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          primary_paper_id: primaryPaperId,
          secondary_paper_ids: [secondaryPaperId],
        }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setGraphResult(data);
    } catch {
      // Fallback sample
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-ink text-zinc-100 flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-10 space-y-8">
        {/* Header */}
        <div className="space-y-3">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-400 hover:text-white transition"
          >
            <ArrowLeft size={14} />
            Back to ScholAR Workspace
          </Link>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-purple-400">
            <GitCompare size={14} />
            Multi-Document Reasoning Linker
          </div>
          <h1 className="text-3xl font-bold text-white tracking-tight">
            Cross-Paper Evidence Synthesis & Comparison
          </h1>
          <p className="text-sm text-zinc-400 max-w-2xl leading-relaxed">
            Construct unified multi-hop evidence graphs across multiple research papers simultaneously.
          </p>
        </div>

        {/* Paper Selection Card */}
        <div className="rounded-2xl border border-line bg-panel p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Primary Paper (Paper A)
              </label>
              <select
                value={primaryPaperId}
                onChange={(e) => setPrimaryPaperId(e.target.value)}
                className="w-full rounded-xl border border-line bg-zinc-900 px-4 py-2.5 text-xs text-white outline-none focus:border-purple-500"
              >
                {BENCHMARK_PAPERS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Comparison Paper (Paper B)
              </label>
              <select
                value={secondaryPaperId}
                onChange={(e) => setSecondaryPaperId(e.target.value)}
                className="w-full rounded-xl border border-line bg-zinc-900 px-4 py-2.5 text-xs text-white outline-none focus:border-purple-500"
              >
                {BENCHMARK_PAPERS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Comparative Prompt Input */}
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Comparative Research Question
            </label>
            <div className="flex items-center gap-2 rounded-xl border border-line bg-zinc-900 px-3 py-2">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a cross-document comparative question..."
                className="min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-zinc-500"
              />
              <button
                onClick={handleCompare}
                disabled={loading || !query.trim()}
                className="flex items-center gap-1.5 rounded-lg bg-purple-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-purple-500 disabled:opacity-40 shadow-lg shadow-purple-600/30"
              >
                <Sparkles size={13} />
                <span>{loading ? "Synthesizing..." : "Synthesize Graph"}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Results Graph Display */}
        {graphResult && (
          <div className="rounded-2xl border border-line bg-panel p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Network size={16} className="text-purple-400" />
                Cross-Document Evidence Graph ({graphResult.graph?.nodes?.length || 0} Nodes, {graphResult.graph?.edges?.length || 0} Directed Edges)
              </div>
              <span className="rounded-full bg-purple-500/20 px-3 py-1 text-xs font-semibold text-purple-300">
                Unified Cross-Paper DAG
              </span>
            </div>

            {/* Nodes Flow */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {graphResult.graph?.nodes?.map((node: any, idx: number) => (
                <div key={node.node_id || idx} className="rounded-xl border border-line bg-zinc-900/60 p-4 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-white font-mono">{node.node_id}</span>
                    <span className="rounded bg-purple-500/20 px-2 py-0.5 text-[10px] font-mono text-purple-300">
                      Doc: {node.document_id} · p.{node.page}
                    </span>
                  </div>
                  <div className="text-xs text-zinc-400 font-medium">{node.section}</div>
                  <p className="text-xs text-zinc-300 leading-relaxed line-clamp-3">
                    {node.text_preview}
                  </p>
                </div>
              ))}
            </div>

            {/* Directed Cross-Paper Bridges */}
            <div className="space-y-2 border-t border-line pt-4">
              <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Cross-Paper Conceptual Bridges
              </div>
              <div className="space-y-2">
                {graphResult.graph?.edges?.map((edge: any, eIdx: number) => (
                  <div key={eIdx} className="flex items-center gap-2 rounded-lg bg-zinc-900/40 p-2.5 text-xs text-zinc-300">
                    <span className="font-mono text-purple-300">{edge.source_id}</span>
                    <ArrowRight size={12} className="text-zinc-500 shrink-0" />
                    <span className="font-mono text-purple-300">{edge.target_id}</span>
                    <span className="text-zinc-500">·</span>
                    <span className="text-zinc-400">{edge.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
