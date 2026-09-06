"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowLeft,
  Cpu,
  Database,
  Network,
  ShieldCheck,
  Zap,
  CheckCircle2,
  Calculator,
  RefreshCw,
} from "lucide-react";
import { Navbar } from "../../components/Navbar";
import type { SystemDiagnostic, TelemetryTrace } from "../../types/api";
import { isSystemDiagnostic, isTelemetryTrace } from "../../types/api";

export default function TelemetryPage() {
  const [diagnostics, setDiagnostics] = useState<SystemDiagnostic | null>(null);
  const [traces, setTraces] = useState<TelemetryTrace[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedTrace, setSelectedTrace] = useState<TelemetryTrace | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const [diagRes, tracesRes] = await Promise.all([
        fetch(`${backendUrl}/api/system/health-diagnostic`),
        fetch(`${backendUrl}/api/telemetry/traces`),
      ]);
      if (diagRes.ok) {
        const diagnosticData: unknown = await diagRes.json();
        setDiagnostics(isSystemDiagnostic(diagnosticData) ? diagnosticData : null);
      }
      if (tracesRes.ok) {
        const traceData: unknown = await tracesRes.json();
        const validTraces = Array.isArray(traceData) ? traceData.filter(isTelemetryTrace) : [];
        setTraces(validTraces);
        setSelectedTrace(validTraces[0] ?? null);
      }
    } catch {
      setDiagnostics(null);
      setTraces([]);
      setSelectedTrace(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-ink text-zinc-100 flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-10 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-400 hover:text-white transition"
            >
              <ArrowLeft size={14} />
              Back to ScholAR Workspace
            </Link>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-emerald-400">
              <Activity size={14} />
              Enterprise Telemetry & Audit Logs
            </div>
            <h1 className="text-3xl font-bold text-white tracking-tight">
              System Diagnostics & Reasoning Traces
            </h1>
          </div>
          <button
            onClick={() => void fetchData()}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-xl border border-line bg-panel px-3 py-2 text-xs font-semibold text-zinc-300 hover:border-zinc-500 hover:text-white transition"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
            <span>Refresh Diagnostics</span>
          </button>
        </div>

        {/* Diagnostics Grid */}
        {diagnostics && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-2xl border border-line bg-panel p-5 space-y-2">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Hardware Acceleration</span>
                <Cpu size={15} className="text-blue-400" />
              </div>
              <div className="text-sm font-bold text-white line-clamp-1">
                {diagnostics.acceleration?.device}
              </div>
              <div className="text-xs text-emerald-400 flex items-center gap-1">
                <CheckCircle2 size={12} />
                PyTorch {diagnostics.acceleration?.torch_version}
              </div>
            </div>

            <div className="rounded-2xl border border-line bg-panel p-5 space-y-2">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>System Memory & Tier</span>
                <Zap size={15} className="text-amber-400" />
              </div>
              <div className="text-2xl font-bold text-white">
                {diagnostics.memory?.hardware_tier} Tier
              </div>
              <div className="text-xs text-zinc-400">
                {diagnostics.memory?.available_ram_gb} GB free / {diagnostics.memory?.total_ram_gb} GB total
              </div>
            </div>

            <div className="rounded-2xl border border-line bg-panel p-5 space-y-2">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Vector Embedding Storage</span>
                <Database size={15} className="text-purple-400" />
              </div>
              <div className="text-2xl font-bold text-white">
                {diagnostics.storage?.cached_embeddings_count} Papers
              </div>
              <div className="text-xs text-zinc-400">
                Dense `.npy` embeddings cached locally
              </div>
            </div>

            <div className="rounded-2xl border border-line bg-panel p-5 space-y-2">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Local Generator</span>
                <ShieldCheck size={15} className="text-emerald-400" />
              </div>
              <div className="text-sm font-bold text-white line-clamp-1">
                {diagnostics.local_llm?.active_model}
              </div>
              <div className="text-xs text-emerald-400">
                {diagnostics.local_llm?.mode}
              </div>
            </div>
          </div>
        )}

        {/* Traces Explorer */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-base font-semibold text-white">
              <Network size={16} className="text-purple-400" />
              Recorded Audit Traces ({traces.length})
            </div>
            <span className="text-xs text-zinc-500 font-mono">backend/data/traces/</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Traces List */}
            <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
              {traces.length === 0 ? (
                <div className="rounded-xl border border-line bg-panel p-6 text-center text-xs text-zinc-500">
                  No telemetry traces recorded yet. Ask a question in the study workspace to generate traces.
                </div>
              ) : (
                traces.map((t) => {
                  const isSelected = selectedTrace?.trace_id === t.trace_id;
                  return (
                    <div
                      key={t.trace_id}
                      onClick={() => setSelectedTrace(t)}
                      className={`cursor-pointer rounded-xl border p-3.5 transition-all ${
                        isSelected
                          ? "border-purple-500/60 bg-purple-950/30"
                          : "border-line bg-panel hover:border-zinc-700"
                      }`}
                    >
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-mono text-purple-300">{t.trace_id}</span>
                        <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                          {t.reasoning_level}
                        </span>
                      </div>
                      <div className="mt-1.5 text-xs font-medium text-white line-clamp-1">
                        {t.query}
                      </div>
                      <div className="mt-1 flex items-center justify-between text-[10px] text-zinc-500">
                        <span>{t.paper_id}</span>
                        <span>{t.latency_ms} ms</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Selected Trace Details Drawer */}
            <div className="lg:col-span-2 rounded-2xl border border-line bg-panel p-6 space-y-6">
              {selectedTrace ? (
                <div className="space-y-6">
                  <div className="flex items-center justify-between border-b border-line pb-4">
                    <div>
                      <div className="text-[11px] font-mono text-purple-400">{selectedTrace.trace_id}</div>
                      <h2 className="text-base font-semibold text-white mt-0.5">&ldquo;{selectedTrace.query}&rdquo;</h2>
                    </div>
                    <span className="rounded-full bg-purple-500/20 px-3 py-1 text-xs font-semibold text-purple-300">
                      {selectedTrace.reasoning_level}
                    </span>
                  </div>

                  {/* Decomposed Subqueries */}
                  {selectedTrace.subqueries?.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                        Decomposed Subqueries
                      </div>
                      <div className="space-y-1.5">
                        {selectedTrace.subqueries.map((sq) => (
                          <div key={sq.subquery_id} className="rounded-lg bg-zinc-900/60 p-2.5 text-xs text-zinc-300 flex items-center gap-2">
                            <span className="rounded bg-purple-500/20 px-1.5 py-0.5 font-mono text-[10px] text-purple-300 font-semibold">{sq.subquery_id}</span>
                            <span>{sq.query_text}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Tabular Math Proof */}
                  {selectedTrace.numeric_plan && (
                    <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-4 space-y-2">
                      <div className="flex items-center gap-2 text-xs font-semibold text-emerald-300">
                        <Calculator size={14} />
                        Deterministic Tabular Arithmetic
                      </div>
                      <p className="text-xs text-zinc-200 leading-relaxed">
                        {selectedTrace.numeric_plan.formatted_statement}
                      </p>
                    </div>
                  )}

                  {/* Reasoning Path Steps */}
                  {selectedTrace.reasoning_path?.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                        Evidence Reasoning Path Steps ({selectedTrace.reasoning_path.length})
                      </div>
                      <div className="space-y-2">
                        {selectedTrace.reasoning_path.map((step) => (
                          <div key={step.step_index} className="rounded-xl border border-line bg-zinc-900/40 p-3 text-xs space-y-1">
                            <div className="flex items-center justify-between text-zinc-400">
                              <span className="font-semibold text-white">Step {step.step_index}: {step.section || step.evidence_id}</span>
                              <span>Page {step.page} · <strong className="capitalize">{step.modality}</strong></span>
                            </div>
                            <p className="text-zinc-300 text-[11px] leading-relaxed">
                              {step.claim_contribution}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12 text-xs text-zinc-500">
                  Select a trace from the left panel to inspect full execution details.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
