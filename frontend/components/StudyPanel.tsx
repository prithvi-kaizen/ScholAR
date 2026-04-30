"use client";

import { useEffect, useState } from "react";
import { BookOpen, Brain, Camera, Eye, Layers, MessageCircle, PanelRight, Settings2, ShieldAlert } from "lucide-react";
import type { StudyGoal } from "../types/paper";
import { ChatBox } from "./ChatBox";
import { StudyGoals } from "./StudyGoals";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

interface StudyPanelProps {
  paperId: string;
}

const quickStart = [
  { title: "Ask Questions", icon: MessageCircle },
  { title: "Interactive Citations", icon: Eye },
  { title: "Visual Explanations", icon: Layers },
  { title: "Screenshots", icon: Camera },
  { title: "Scope & Limitations", icon: ShieldAlert }
];

const defaultGoals: StudyGoal[] = [
  {
    id: "goal_1",
    title: "Define problem and motivation",
    description: "Clarify the research problem, why it matters, and what gap the paper is trying to close.",
    source_pages: [1],
    status: "not_started"
  },
  {
    id: "goal_2",
    title: "Summarize core idea",
    description: "Capture the central contribution in plain language before going into details.",
    source_pages: [1],
    status: "not_started"
  },
  {
    id: "goal_3",
    title: "Explain methodology",
    description: "Break down the method, assumptions, and the main technical workflow.",
    source_pages: [1],
    status: "not_started"
  },
  {
    id: "goal_4",
    title: "Identify algorithm or architecture",
    description: "Map the important model components, algorithms, or system design choices.",
    source_pages: [1],
    status: "not_started"
  },
  {
    id: "goal_5",
    title: "Understand experimental setup",
    description: "Review datasets, baselines, metrics, and implementation settings.",
    source_pages: [1],
    status: "not_started"
  },
  {
    id: "goal_6",
    title: "Report key results",
    description: "Extract the main quantitative and qualitative findings from the paper.",
    source_pages: [1],
    status: "not_started"
  },
  {
    id: "goal_7",
    title: "Discuss limitations",
    description: "Identify what the authors admit is limited, uncertain, or left for future work.",
    source_pages: [1],
    status: "not_started"
  },
  {
    id: "goal_8",
    title: "Convert paper into implementation plan",
    description: "Turn the paper into a practical build plan with steps, dependencies, and risks.",
    source_pages: [1],
    status: "not_started"
  }
];

export function StudyPanel({ paperId }: StudyPanelProps) {
  const [activeTab, setActiveTab] = useState<"goals" | "quick">("goals");
  const [goals, setGoals] = useState<StudyGoal[]>(defaultGoals);
  const [loadingGoals, setLoadingGoals] = useState(false);
  const [queuedPrompt, setQueuedPrompt] = useState<{ id: number; text: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadGoals() {
      setGoals(defaultGoals);
      setLoadingGoals(true);
      try {
        const response = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/study-goals`, {
          method: "POST"
        });
        if (!response.ok) throw new Error("Could not load study goals");
        const payload = await response.json();
        if (!cancelled && payload.goals?.length) setGoals(payload.goals);
      } finally {
        if (!cancelled) setLoadingGoals(false);
      }
    }
    void loadGoals();
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  function explainGoal(goal: StudyGoal) {
    setQueuedPrompt({
      id: Date.now(),
      text: `Explain this study goal in detail: ${goal.title}`
    });
  }

  return (
    <aside className="flex min-h-0 flex-col bg-ink">
      <div className="flex h-16 shrink-0 items-center justify-between border-b border-line bg-[#101114] px-4">
        <div className="flex items-center gap-2">
          <button className="rounded-md border border-line p-2 text-zinc-400 hover:text-white" aria-label="Toggle study sidebar">
            <PanelRight size={17} />
          </button>
          <button className="rounded-md border border-line bg-white/5 p-2 text-zinc-200" aria-label="Study controls">
            <Settings2 size={17} />
          </button>
        </div>
        <div className="flex items-center justify-between gap-3">
          <button className="inline-flex items-center gap-2 rounded-md border border-line bg-panel px-3 py-2 text-sm text-white">
            <BookOpen size={16} />
            Study Goals
            <span className="rounded-md bg-red-500 px-2 py-0.5 text-xs font-semibold text-white">0/8</span>
          </button>
          <button disabled className="inline-flex cursor-not-allowed items-center gap-2 rounded-md border border-line bg-panel px-3 py-2 text-sm text-zinc-500">
            <Brain size={16} />
            Evaluate Knowledge
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        <div className="mb-6 text-center">
          <p className="text-xs uppercase tracking-[0.2em] text-acid">Local Qwen session</p>
          <h1 className="mt-2 text-2xl font-semibold text-white">Let&apos;s study research!</h1>
        </div>

        <div className="mx-auto mb-5 grid max-w-2xl grid-cols-2 rounded-lg border border-line bg-panel p-1 text-sm">
          <button
            onClick={() => setActiveTab("goals")}
            className={`rounded-md px-3 py-2 ${activeTab === "goals" ? "bg-white text-black" : "text-zinc-400 hover:text-white"}`}
          >
            Study Goals
          </button>
          <button
            onClick={() => setActiveTab("quick")}
            className={`rounded-md px-3 py-2 ${activeTab === "quick" ? "bg-white text-black" : "text-zinc-400 hover:text-white"}`}
          >
            Quick Start
          </button>
        </div>

        {activeTab === "goals" ? (
          <StudyGoals goals={goals} loading={loadingGoals} onGoalClick={explainGoal} />
        ) : (
          <div className="mx-auto grid max-w-2xl gap-4">
            {quickStart.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.title}
                  onClick={() => setQueuedPrompt({ id: Date.now(), text: item.title })}
                  className="flex items-start gap-4 rounded-lg border border-line bg-panel p-5 text-left text-sm font-medium text-zinc-200 hover:border-zinc-500 hover:bg-panelSoft"
                >
                  <span className="grid h-11 w-11 shrink-0 place-items-center rounded-md bg-white/5 text-zinc-200">
                    <Icon size={18} />
                  </span>
                  <span>
                    <span className="block text-base text-white">{item.title}</span>
                    <span className="mt-1 block font-normal leading-6 text-zinc-400">
                      {item.title === "Ask Questions"
                        ? "Inquire about the methodology, key findings, or specific details within the text."
                        : item.title === "Interactive Citations"
                          ? "The AI cites relevant sections so you can connect answers back to the paper."
                          : item.title === "Visual Explanations"
                            ? "Ask for diagrams or structured explanations to understand complex concepts."
                            : item.title === "Screenshots"
                              ? "Use the visible paper page as your reading anchor while chatting."
                              : "Conversations are focused on this paper and its extracted context."}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      <ChatBox paperId={paperId} queuedPrompt={queuedPrompt} />
    </aside>
  );
}
