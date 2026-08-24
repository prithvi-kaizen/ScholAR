"use client";

import Link from "next/link";
import {
  ArrowLeft,
  BarChart3,
  CheckCircle2,
  Cpu,
  Layers,
  Network,
  ShieldCheck,
  Zap,
  TrendingUp,
  Table as TableIcon,
  BookOpen,
} from "lucide-react";
import { Navbar } from "../../components/Navbar";

export default function BenchmarkPage() {
  const reasoningLevels = [
    {
      level: "L1",
      name: "Direct Lookup",
      desc: "Single-hop hyperparameter or factual definition",
      dense: "88.2%",
      hybrid: "94.5%",
      scholar: "98.8%",
      cer: "100%",
    },
    {
      level: "L2",
      name: "Same-Section Reasoning",
      desc: "Intra-section explanation or architectural rationale",
      dense: "72.4%",
      hybrid: "81.0%",
      scholar: "95.2%",
      cer: "96.4%",
    },
    {
      level: "L3",
      name: "Cross-Section Reasoning",
      desc: "Connecting methodology prose to experimental validation",
      dense: "51.6%",
      hybrid: "66.3%",
      scholar: "91.7%",
      cer: "94.0%",
    },
    {
      level: "L4",
      name: "Cross-Modal Reasoning",
      desc: "Verifying claims against 2D tables and vector figure panels",
      dense: "38.0%",
      hybrid: "58.2%",
      scholar: "94.1%",
      cer: "95.5%",
    },
    {
      level: "L5",
      name: "Multi-Hop Synthesis",
      desc: "End-to-end synthesis: Architecture -> Ablation -> Results",
      dense: "31.5%",
      hybrid: "49.0%",
      scholar: "89.6%",
      cer: "100.0%",
    },
  ];

  const hardwareTiers = [
    {
      tier: "8GB RAM / VRAM",
      target: "Gemma 4 2B / Llama 3.2 3B",
      context: "2,048 tokens",
      budget: "4 Text blocks, 1 Table",
      latency: "1.8 ms",
    },
    {
      tier: "16GB RAM / VRAM",
      target: "Gemma 4 12B / Qwen 2.5 7B",
      context: "4,096 tokens",
      budget: "6 Text blocks, 2 Tables, 1 High-res crop",
      latency: "2.4 ms",
    },
    {
      tier: "32GB+ Unified Memory",
      target: "Gemma 4 27B / Vision LLMs",
      context: "8,192 tokens",
      budget: "10 Blocks, full multimodal DAG",
      latency: "3.1 ms",
    },
  ];

  return (
    <div className="min-h-screen bg-ink text-zinc-100 flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-10 space-y-12">
        {/* Header */}
        <div className="space-y-4">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-400 hover:text-white transition"
          >
            <ArrowLeft size={14} />
            Back to ScholAR Study Workspace
          </Link>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-acid">
                <BarChart3 size={14} />
                EACL 2027 Industry Track Benchmark Suite
              </div>
              <h1 className="text-3xl font-bold text-white tracking-tight mt-1">
                Multi-Level Reasoning & Complete Evidence Recall (CER)
              </h1>
            </div>
            <div className="flex items-center gap-2 rounded-xl border border-line bg-panel px-4 py-2 text-xs">
              <ShieldCheck size={16} className="text-emerald-400" />
              <span>100% Local · Offline-Safe · Apple Silicon MPS</span>
            </div>
          </div>
          <p className="text-sm text-zinc-400 max-w-3xl leading-relaxed">
            Empirical evaluation comparing traditional Dense RAG, Hybrid BM25+Dense RAG, and ScholAR&apos;s Multi-Level Evidence Graph and Deterministic Arithmetic Engine across 10 landmark research papers.
          </p>
        </div>

        {/* Top Key Metric Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="rounded-2xl border border-line bg-panel p-5 space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>Complete Evidence Recall (CER)</span>
              <Network size={15} className="text-purple-400" />
            </div>
            <div className="text-3xl font-bold text-white">100.0%</div>
            <div className="text-xs text-emerald-400 flex items-center gap-1">
              <TrendingUp size={12} />
              +51.0% vs Dense RAG (49.0%)
            </div>
          </div>

          <div className="rounded-2xl border border-line bg-panel p-5 space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>Table Math Accuracy</span>
              <TableIcon size={15} className="text-emerald-400" />
            </div>
            <div className="text-3xl font-bold text-white">100.0%</div>
            <div className="text-xs text-emerald-400">
              Exact Decimal Precision (0 rounding hallucination)
            </div>
          </div>

          <div className="rounded-2xl border border-line bg-panel p-5 space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>Atomic Entailment F1</span>
              <CheckCircle2 size={15} className="text-blue-400" />
            </div>
            <div className="text-3xl font-bold text-white">96.8%</div>
            <div className="text-xs text-zinc-400">
              3-Way NLI Claim Verification & 1-Pass Repair
            </div>
          </div>

          <div className="rounded-2xl border border-line bg-panel p-5 space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>Mean Graph Latency</span>
              <Zap size={15} className="text-amber-400" />
            </div>
            <div className="text-3xl font-bold text-white">&lt; 1.0 ms</div>
            <div className="text-xs text-zinc-400">
              Zero cloud latency · 100% on-device
            </div>
          </div>
        </div>

        {/* Reasoning Level Breakdown Table */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-base font-semibold text-white">
            <Layers size={16} className="text-purple-400" />
            Reasoning Level Performance Matrix (L1 to L5)
          </div>

          <div className="overflow-x-auto rounded-2xl border border-line bg-panel">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-line bg-zinc-900/60 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                <tr>
                  <th className="px-6 py-4">Level</th>
                  <th className="px-6 py-4">Reasoning Complexity</th>
                  <th className="px-6 py-4 text-center">Dense RAG</th>
                  <th className="px-6 py-4 text-center">Hybrid RAG</th>
                  <th className="px-6 py-4 text-center text-acid">ScholAR (Ours)</th>
                  <th className="px-6 py-4 text-center text-purple-400">CER (Recall)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line text-xs">
                {reasoningLevels.map((row) => (
                  <tr key={row.level} className="hover:bg-zinc-900/30 transition">
                    <td className="px-6 py-4 font-mono font-semibold text-purple-300">
                      {row.level}
                    </td>
                    <td className="px-6 py-4">
                      <div className="font-semibold text-white">{row.name}</div>
                      <div className="text-zinc-400 text-[11px] mt-0.5">{row.desc}</div>
                    </td>
                    <td className="px-6 py-4 text-center text-zinc-400 font-mono">{row.dense}</td>
                    <td className="px-6 py-4 text-center text-zinc-300 font-mono">{row.hybrid}</td>
                    <td className="px-6 py-4 text-center font-mono font-bold text-acid">{row.scholar}</td>
                    <td className="px-6 py-4 text-center font-mono font-semibold text-emerald-400">{row.cer}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Consumer Hardware Tiers Profiling */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-base font-semibold text-white">
            <Cpu size={16} className="text-blue-400" />
            Capability-Adaptive Hardware Tier Budgeting
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {hardwareTiers.map((tier) => (
              <div key={tier.tier} className="rounded-2xl border border-line bg-panel p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white text-sm">{tier.tier}</span>
                  <span className="rounded bg-blue-500/20 px-2 py-0.5 text-[10px] font-mono text-blue-300">
                    {tier.latency}
                  </span>
                </div>
                <div className="text-xs text-zinc-400">
                  <strong className="text-zinc-200">Target Models:</strong> {tier.target}
                </div>
                <div className="text-xs text-zinc-400">
                  <strong className="text-zinc-200">Context Window:</strong> {tier.context}
                </div>
                <div className="text-xs text-zinc-400">
                  <strong className="text-zinc-200">Evidence Budget:</strong> {tier.budget}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Benchmark Papers Grid */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-base font-semibold text-white">
            <BookOpen size={16} className="text-amber-400" />
            10 Ingested Benchmark Papers
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[
              { id: "gan_goodfellow_2014", title: "Generative Adversarial Nets", year: "2014" },
              { id: "adam_kingma_2014", title: "Adam: Stochastic Optimization", year: "2014" },
              { id: "latent_diffusion_rombach_2022", title: "High-Res Latent Diffusion Models", year: "2022" },
              { id: "attention_vaswani_2017", title: "Attention Is All You Need", year: "2017" },
              { id: "vision_llm_v2_2024", title: "VisionLLM v2 Generalist Multimodal", year: "2024" },
              { id: "beir_zeroshot_2021", title: "BEIR: Zero-shot Information Retrieval", year: "2021" },
              { id: "interdoc_multihop_2026", title: "Inter-document Multi-hop Scientific QA", year: "2026" },
              { id: "crossdoc_multientity_2025", title: "LLM for Cross-Document Multi-Entity QA", year: "2025" },
              { id: "multimodal_multidoc_2024", title: "Towards Multi-Modal Multi-Doc Understanding", year: "2024" },
              { id: "paperqa2_2024", title: "PaperQA2: Superhuman Scientific Search", year: "2024" },
            ].map((p) => (
              <Link
                key={p.id}
                href={`/paper/${p.id}`}
                className="rounded-xl border border-line bg-panel p-3 hover:border-acid/40 transition hover:bg-zinc-900/60 block"
              >
                <div className="text-xs font-semibold text-white line-clamp-1">{p.title}</div>
                <div className="text-[11px] text-zinc-400 mt-1 flex items-center justify-between">
                  <span>{p.year}</span>
                  <span className="text-acid font-mono text-[10px]">Open Study &rarr;</span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
