"use client";

import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import type { StudyGoal } from "../types/paper";

interface StudyGoalsProps {
  goals: StudyGoal[];
  loading: boolean;
  onGoalClick: (goal: StudyGoal) => void;
}

export function StudyGoals({ goals, loading, onGoalClick }: StudyGoalsProps) {
  if (loading && !goals.length) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-sm text-zinc-300">
          <Loader2 size={15} className="animate-spin text-acid" />
          Personalizing study goals for this paper…
        </div>
        <div className="space-y-3">
          {[...Array(8)].map((_, index) => (
            <div key={index} className="h-20 animate-pulse rounded-lg border border-line bg-panelSoft" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-3">
      {goals.map((goal, index) => (
        <button
          key={goal.id}
          onClick={() => onGoalClick(goal)}
          className="w-full rounded-lg border border-line bg-panel p-4 text-left transition hover:border-zinc-500 hover:bg-panelSoft"
        >
          <div className="flex items-start gap-3">
            {goal.status === "done" ? <CheckCircle2 size={18} className="mt-1 text-acid" /> : <Circle size={18} className="mt-1 text-zinc-500" />}
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-zinc-500">{String(index + 1).padStart(2, "0")}</span>
                <h3 className="text-sm font-semibold text-white">{goal.title}</h3>
              </div>
              <p className="mt-2 text-xs leading-5 text-zinc-400">{goal.description}</p>
              {goal.subquestions?.length ? (
                <div className="mt-3 space-y-2 rounded-md border border-line bg-ink/60 p-3">
                  {goal.subquestions.slice(0, 5).map((subquestion) => (
                    <div key={subquestion.id} className="text-xs leading-5 text-zinc-300">
                      <div>{subquestion.question}</div>
                      {subquestion.evidence_chunks?.length ? (
                        <div className="mt-1 flex flex-wrap gap-1.5">
                          {subquestion.evidence_chunks.slice(0, 3).map((evidence) => (
                            <span key={`${subquestion.id}-${evidence.chunk_id}`} className="rounded-md border border-line px-2 py-0.5 text-[11px] text-zinc-500">
                              p. {evidence.page}
                              {evidence.chunk_type ? ` · ${evidence.chunk_type}` : ""}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-1.5">
                {goal.source_pages.map((page) => (
                  <span key={`${goal.id}-${page}`} className="rounded-md border border-line px-2 py-0.5 text-xs text-zinc-400">
                    p. {page}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </button>
      ))}
      {!goals.length ? (
        <div className="flex items-center gap-2 rounded-lg border border-line bg-panel p-4 text-sm text-zinc-400">
          <Loader2 size={16} />
          Study goals will appear here after preparation.
        </div>
      ) : null}
    </div>
  );
}
