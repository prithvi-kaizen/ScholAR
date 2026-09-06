"use client";

import { useEffect, useState } from "react";
import {
  X,
  Network,
  Layers,
  ArrowRight,
  BookOpen,
  Camera,
  Table as TableIcon,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";
import type { Citation, ReasoningPathStep } from "../types/paper";

interface EvidenceGraphModalProps {
  isOpen: boolean;
  onClose: () => void;
  query: string;
  reasoningLevel?: string;
  reasoningSteps?: ReasoningPathStep[];
  onNodeClick: (citation: Citation) => void;
}

export function EvidenceGraphModal({
  isOpen,
  onClose,
  query,
  reasoningLevel = "L5_MULTI_HOP_SYNTHESIS",
  reasoningSteps = [],
  onNodeClick,
}: EvidenceGraphModalProps) {
  const [selectedStepIdx, setSelectedStepIdx] = useState<number>(0);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const currentStep = reasoningSteps[selectedStepIdx] || reasoningSteps[0];

  const getModalityIcon = (modality: string) => {
    switch (modality.toLowerCase()) {
      case "table":
        return <TableIcon size={14} className="text-emerald-400" />;
      case "visual":
      case "figure":
        return <Camera size={14} className="text-purple-400" />;
      default:
        return <BookOpen size={14} className="text-blue-400" />;
    }
  };

  const getRoleBadge = (role: string) => {
    switch (role) {
      case "method_definition":
        return <span className="rounded bg-blue-500/20 px-2 py-0.5 text-[10px] font-semibold text-blue-300">1. Mechanism</span>;
      case "ablation_support":
        return <span className="rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold text-amber-300">2. Ablation Support</span>;
      case "final_result":
        return <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">3. Empirical Result</span>;
      default:
        return <span className="rounded bg-zinc-700 px-2 py-0.5 text-[10px] font-semibold text-zinc-300">Context</span>;
    }
  };

  const getMLRModeBadge = (mode?: string, role?: string) => {
    switch (mode) {
      case "ProblemUnderstanding":
        return <span className="rounded bg-sky-500/20 px-2 py-0.5 text-[10px] font-semibold text-sky-300">Problem Understanding</span>;
      case "Planning":
        return <span className="rounded bg-indigo-500/20 px-2 py-0.5 text-[10px] font-semibold text-indigo-300">Planning</span>;
      case "Recall":
        return <span className="rounded bg-blue-500/20 px-2 py-0.5 text-[10px] font-semibold text-blue-300">Recall</span>;
      case "Derivation":
        return <span className="rounded bg-cyan-500/20 px-2 py-0.5 text-[10px] font-semibold text-cyan-300">Derivation</span>;
      case "Calculation":
        return <span className="rounded bg-teal-500/20 px-2 py-0.5 text-[10px] font-semibold text-teal-300">Calculation</span>;
      case "CaseAnalysis":
        return <span className="rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold text-amber-300">Case Analysis</span>;
      case "Verification":
        return <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">Verification</span>;
      case "ErrorCorrection":
      case "Backtracking":
        return <span className="rounded bg-rose-500/20 px-2 py-0.5 text-[10px] font-semibold text-rose-300">{mode}</span>;
      case "Synthesis":
      case "Finalization":
        return <span className="rounded bg-violet-500/20 px-2 py-0.5 text-[10px] font-semibold text-violet-300">{mode}</span>;
      default:
        return getRoleBadge(role || "primary_evidence");
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="relative flex flex-col w-full max-w-4xl max-h-[85vh] rounded-2xl border border-line bg-zinc-950 shadow-2xl overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-line px-6 py-4 bg-zinc-900/50">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-500/20 text-purple-400">
              <Network size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-white">Multi-Level Evidence Graph</h2>
                <span className="rounded-full bg-purple-500/20 px-2 py-0.5 text-[10px] font-semibold text-purple-300">
                  {reasoningLevel.replace(/_/g, " ")}
                </span>
              </div>
              <p className="text-xs text-zinc-400 mt-0.5 truncate max-w-xl">
                &ldquo;{query}&rdquo;
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-white transition"
          >
            <X size={18} />
          </button>
        </div>

        {/* Graph Body */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          
          {/* Left Pane: Interactive Graph Flow */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4 border-r border-line bg-zinc-900/20">
            <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
              <Layers size={13} />
              Reasoning Traversal Path ({reasoningSteps.length} Nodes)
            </div>

            <div className="space-y-3 pt-2">
              {reasoningSteps.map((step, idx) => {
                const isSelected = idx === selectedStepIdx;
                return (
                  <div key={step.step_index} className="relative">
                    {/* Directed Edge Line */}
                    {idx > 0 && (
                      <div className="flex items-center justify-center my-1.5 text-zinc-600">
                        <div className="flex items-center gap-1.5 px-3 py-0.5 rounded-full bg-zinc-800/80 border border-zinc-700/50 text-[10px] text-zinc-400">
                          <span>
                            {idx === 1 ? "supports mechanism" : "explains result"}
                          </span>
                          <ArrowRight size={10} className="text-purple-400" />
                        </div>
                      </div>
                    )}

                    {/* Node Card */}
                    <div
                      onClick={() => setSelectedStepIdx(idx)}
                      className={`cursor-pointer rounded-xl border p-4 transition-all ${
                        isSelected
                          ? "border-purple-500/60 bg-purple-950/30 shadow-lg shadow-purple-950/40"
                          : "border-line bg-panel hover:border-zinc-700 hover:bg-zinc-900/80"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-zinc-800">
                            {getModalityIcon(step.modality)}
                          </div>
                          <div>
                            <span className="text-xs font-semibold text-white">
                              {step.section || step.evidence_id}
                            </span>
                            <span className="ml-2 text-[11px] text-zinc-400">Page {step.page}</span>
                          </div>
                        </div>
                        {getMLRModeBadge(step.reasoning_mode, step.role)}
                      </div>

                      {step.subgoal && (
                        <p className="mt-1.5 text-[11px] font-medium text-purple-300 line-clamp-1">
                          <span className="text-zinc-400 font-normal">Subgoal: </span>
                          {step.subgoal}
                        </p>
                      )}

                      <p className="mt-1.5 text-xs text-zinc-300 leading-relaxed line-clamp-2">
                        {step.claim_contribution}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Pane: Selected Node Provenance & PDF Jump */}
          <div className="w-80 overflow-y-auto p-6 bg-zinc-950 flex flex-col justify-between">
            {currentStep ? (
              <div className="space-y-4">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-purple-400 mb-1">
                    Step {currentStep.step_index} of {reasoningSteps.length}
                  </div>
                  <h3 className="text-sm font-semibold text-white">
                    {currentStep.section || currentStep.evidence_id}
                  </h3>
                  <div className="mt-1 flex items-center gap-2 text-xs text-zinc-400">
                    <span>Modality: <strong className="text-zinc-200 capitalize">{currentStep.modality}</strong></span>
                    <span>·</span>
                    <span>Page {currentStep.page}</span>
                  </div>
                </div>

                {currentStep.subgoal && (
                  <div className="rounded-xl border border-purple-500/30 bg-purple-950/20 p-3 text-xs">
                    <div className="text-[11px] font-semibold text-purple-300 mb-1">MLR Step Subgoal</div>
                    <p className="text-zinc-200 leading-relaxed">{currentStep.subgoal}</p>
                    {currentStep.reasoning_mode && (
                      <div className="mt-2 pt-2 border-t border-purple-500/20 flex items-center justify-between text-[11px]">
                        <span className="text-zinc-400">Mode:</span>
                        {getMLRModeBadge(currentStep.reasoning_mode, currentStep.role)}
                      </div>
                    )}
                  </div>
                )}

                <div className="rounded-xl border border-line bg-zinc-900/60 p-3 text-xs text-zinc-300 leading-relaxed">
                  <div className="font-medium text-white mb-1">Semantic Contribution:</div>
                  <p>{currentStep.claim_contribution}</p>
                </div>

                <div className="rounded-xl border border-line bg-zinc-900/40 p-3 text-xs">
                  <div className="text-[11px] text-zinc-400 font-mono">Evidence ID: {currentStep.evidence_id}</div>
                  <div className="text-[11px] text-zinc-400 mt-1">Role: <span className="text-zinc-200">{currentStep.role}</span></div>
                </div>

                <button
                  onClick={() => {
                    onNodeClick({
                      page: currentStep.page,
                      chunk_id: currentStep.evidence_id,
                      section_title: currentStep.section,
                      chunk_type: currentStep.modality,
                      quote: currentStep.claim_contribution,
                    });
                    onClose();
                  }}
                  className="w-full flex items-center justify-center gap-2 rounded-xl bg-purple-600 hover:bg-purple-500 py-2.5 px-4 text-xs font-semibold text-white transition shadow-lg shadow-purple-600/30"
                >
                  <ExternalLink size={14} />
                  Jump to Page {currentStep.page} in PDF
                </button>
              </div>
            ) : (
              <div className="text-xs text-zinc-500 text-center py-8">Select a node to inspect provenance</div>
            )}

            <div className="pt-4 border-t border-line text-[11px] text-zinc-500 flex items-center gap-1.5">
              <CheckCircle2 size={12} className="text-emerald-400 shrink-0" />
              <span>Software-owned canonical provenance</span>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
