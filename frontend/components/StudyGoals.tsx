"use client";

import { useState } from "react";
import {
  CheckCircle2, Circle, Loader2, Sparkles, Zap,
  Clock, BookOpen, Layers, ChevronDown, ChevronUp, HelpCircle
} from "lucide-react";
import type { StudyGoal } from "../types/paper";

interface StudyGoalsProps {
  goals: StudyGoal[];
  loading: boolean;
  onGoalClick: (goal: StudyGoal) => void;
  onSubquestionClick?: (goal: StudyGoal, subquestion: string) => void;
}

type PhaseFilter = "All" | "Foundation" | "Architecture" | "Benchmarks" | "Implementation";

const PHASE_LABELS: Record<PhaseFilter, { label: string }> = {
  All: { label: "All Goals" },
  Foundation: { label: "Foundations" },
  Architecture: { label: "Architecture & Math" },
  Benchmarks: { label: "Benchmarks & Ablations" },
  Implementation: { label: "Build & Limits" },
};

function DifficultyBadge({ difficulty }: { difficulty?: string }) {
  if (!difficulty) return null;
  if (difficulty === "Foundational") {
    return (
      <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
        Foundational
      </span>
    );
  }
  if (difficulty === "Intermediate") {
    return (
      <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
        Intermediate
      </span>
    );
  }
  return (
    <span className="rounded-full bg-indigo-500/15 px-2 py-0.5 text-[10px] font-semibold text-indigo-300">
      Advanced
    </span>
  );
}

export function StudyGoals({ goals, loading, onGoalClick, onSubquestionClick }: StudyGoalsProps) {
  const [completedIds, setCompletedIds] = useState<Set<string>>(new Set());
  const [activePhase, setActivePhase] = useState<PhaseFilter>("All");
  const [expandedGoalId, setExpandedGoalId] = useState<string | null>(null);

  const toggleGoal = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setCompletedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const effectiveDoneCount = goals.filter((g) => g.status === "done" || completedIds.has(g.id)).length;
  const progressPct = goals.length > 0 ? Math.round((effectiveDoneCount / goals.length) * 100) : 0;

  const filteredGoals = activePhase === "All"
    ? goals
    : goals.filter((g) => (g.phase || "Foundation") === activePhase);

  if (loading && !goals.length) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-sm text-zinc-300">
          <Loader2 size={15} className="animate-spin text-acid" />
          Synthesizing 4-phase pedagogical curriculum for this paper…
        </div>
        <div className="space-y-3">
          {[...Array(8)].map((_, index) => (
            <div key={index} className="h-24 animate-pulse rounded-xl border border-line bg-panelSoft" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      {/* ── Mastery Progress Header ── */}
      {goals.length > 0 && (
        <div className="rounded-2xl border border-line bg-panel p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-acid/10 text-acid">
                <Sparkles size={16} />
              </div>
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">Curriculum Mastery</h3>
                <p className="text-xs text-zinc-500">
                  {effectiveDoneCount} of {goals.length} milestones explored
                </p>
              </div>
            </div>
            <span className="font-mono text-sm font-bold text-acid">{progressPct}%</span>
          </div>
          {/* Progress bar track */}
          <div className="mt-3.5 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-acid transition-all duration-500 ease-out"
              style={{ width: `${progressPct}%` }}
            />
          </div>

          {/* Phase Filter Tabs */}
          <div className="mt-4 flex flex-wrap items-center gap-1.5 border-t border-line/50 pt-3">
            {(Object.keys(PHASE_LABELS) as PhaseFilter[]).map((phase) => {
              const count = phase === "All" ? goals.length : goals.filter((g) => (g.phase || "Foundation") === phase).length;
              return (
                <button
                  key={phase}
                  type="button"
                  onClick={() => setActivePhase(phase)}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                    activePhase === phase
                      ? "bg-acid/20 text-acid ring-1 ring-acid/40"
                      : "bg-ink/60 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                  }`}
                >
                  <span>{PHASE_LABELS[phase].label}</span>
                  <span className="text-[10px] opacity-60">({count})</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Goals list ── */}
      <div className="space-y-3">
        {filteredGoals.map((goal, index) => {
          const isDone = goal.status === "done" || completedIds.has(goal.id);
          const isExpanded = expandedGoalId === goal.id;

          return (
            <div
              key={goal.id}
              className={`rounded-2xl border p-4 text-left transition-all ${
                isDone
                  ? "border-emerald-500/30 bg-emerald-950/10"
                  : "border-line bg-panel hover:border-zinc-500"
              }`}
            >
              <div className="flex items-start gap-3">
                <button
                  type="button"
                  onClick={(e) => toggleGoal(goal.id, e)}
                  className="mt-0.5 shrink-0 transition-transform active:scale-90"
                  title={isDone ? "Mark as in-progress" : "Mark as mastered"}
                >
                  {isDone ? (
                    <CheckCircle2 size={20} className="text-emerald-400" />
                  ) : (
                    <Circle size={20} className="text-zinc-600 hover:text-zinc-400" />
                  )}
                </button>

                <div className="min-w-0 flex-1">
                  {/* Top metadata tags */}
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-xs text-zinc-500">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <DifficultyBadge difficulty={goal.difficulty || "Foundational"} />
                      {goal.phase && (
                        <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] font-medium text-zinc-400">
                          {goal.phase}
                        </span>
                      )}
                      {goal.estimated_minutes && (
                        <span className="inline-flex items-center gap-1 text-[11px] text-zinc-500">
                          <Clock size={11} />
                          {goal.estimated_minutes} min
                        </span>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => onGoalClick(goal)}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-acid px-2.5 py-1 text-xs font-semibold text-black transition hover:bg-acid/80"
                    >
                      <Zap size={11} />
                      Ask ScholAR
                    </button>
                  </div>

                  {/* Goal title & description */}
                  <h4 className={`mt-2 text-sm font-semibold ${isDone ? "text-emerald-200 line-through opacity-80" : "text-white"}`}>
                    {goal.title}
                  </h4>
                  <p className="mt-1.5 text-xs leading-5 text-zinc-400">{goal.description}</p>

                  {/* High yield key takeaways */}
                  {goal.key_takeaways && goal.key_takeaways.length > 0 && (
                    <div className="mt-3 rounded-xl border border-line/60 bg-ink/50 p-2.5">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">Key Mastery Takeaways</p>
                      <div className="mt-1.5 space-y-1">
                        {goal.key_takeaways.map((takeaway, tIdx) => (
                          <div key={tIdx} className="flex items-start gap-1.5 text-xs text-zinc-300">
                            <span className="text-acid">•</span>
                            <span>{takeaway}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Target Evidence Tags */}
                  <div className="mt-3 flex flex-wrap items-center gap-1.5">
                    {goal.target_evidence?.map((ev, evIdx) => (
                      <span key={evIdx} className="inline-flex items-center gap-1 rounded-md border border-teal-500/30 bg-teal-950/30 px-2 py-0.5 text-[10px] font-medium text-teal-300">
                        <BookOpen size={10} />
                        {ev}
                      </span>
                    ))}
                    {goal.source_pages.map((p) => (
                      <span key={`${goal.id}-${p}`} className="rounded border border-line px-2 py-0.5 text-[10px] text-zinc-400">
                        p. {p}
                      </span>
                    ))}
                  </div>

                  {/* Subquestions accordion */}
                  {goal.subquestions && goal.subquestions.length > 0 && (
                    <div className="mt-3 border-t border-line/40 pt-2">
                      <button
                        type="button"
                        onClick={() => setExpandedGoalId(isExpanded ? null : goal.id)}
                        className="inline-flex items-center gap-1 text-xs font-medium text-zinc-400 transition hover:text-white"
                      >
                        <HelpCircle size={12} />
                        <span>{goal.subquestions.length} Research Subquestions</span>
                        {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                      </button>

                      {isExpanded && (
                        <div className="mt-2 space-y-2 rounded-xl border border-line bg-ink/70 p-3">
                          {goal.subquestions.map((subq) => (
                            <div key={subq.id} className="flex items-center justify-between gap-2 border-b border-line/30 pb-2 last:border-b-0 last:pb-0">
                              <span className="text-xs text-zinc-300 leading-relaxed">• {subq.question}</span>
                              <button
                                type="button"
                                onClick={() => onSubquestionClick ? onSubquestionClick(goal, subq.question) : onGoalClick(goal)}
                                className="shrink-0 rounded bg-acid/10 px-2 py-0.5 text-[10px] font-medium text-acid transition hover:bg-acid/20"
                                title="Ask this specific subquestion"
                              >
                                Ask
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                </div>
              </div>
            </div>
          );
        })}
      </div>

      {!goals.length ? (
        <div className="flex items-center gap-2 rounded-lg border border-line bg-panel p-4 text-sm text-zinc-400">
          <Loader2 size={16} />
          Curriculum will appear here after paper preparation.
        </div>
      ) : null}
    </div>
  );
}
