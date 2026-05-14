"use client";

import { useEffect, useState } from "react";
import { BookOpen } from "lucide-react";
import type { Citation, StudyGoal } from "../types/paper";
import { ChatBox } from "./ChatBox";
import { StudyGoals } from "./StudyGoals";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const providerStorageKey = "scholar-ai-provider-v2";

interface StudyPanelProps {
  paperId: string;
  onCitationClick: (citation: Citation) => void;
}

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

export function StudyPanel({ paperId, onCitationClick }: StudyPanelProps) {
  const [goals, setGoals] = useState<StudyGoal[]>(defaultGoals);
  const [loadingGoals, setLoadingGoals] = useState(false);
  const [queuedPrompt, setQueuedPrompt] = useState<{ id: number; text: string } | null>(null);
  const [provider, setProvider] = useState<"local" | "groq">("groq");
  const [providerNotice, setProviderNotice] = useState("");
  const [chatActive, setChatActive] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem(providerStorageKey);
    if (saved === "groq" || saved === "local") setProvider(saved);
  }, []);

  function changeProvider(nextProvider: "local" | "groq") {
    setProvider(nextProvider);
    window.localStorage.setItem(providerStorageKey, nextProvider);
  }

  useEffect(() => {
    let cancelled = false;
    async function loadGoals() {
      setGoals(defaultGoals);
      setLoadingGoals(true);
      try {
        const response = await fetch(`${backendUrl}/api/papers/${encodeURIComponent(paperId)}/study-goals`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider })
        });
        if (!response.ok) throw new Error("Could not load study goals");
        const payload = await response.json();
        if (!cancelled && payload.goals?.length) setGoals(payload.goals);
        if (!cancelled) {
          setProviderNotice(
            payload.requested_provider && payload.provider && payload.requested_provider !== payload.provider
              ? "Groq API key is not configured, so this response used local Qwen."
              : ""
          );
        }
      } finally {
        if (!cancelled) setLoadingGoals(false);
      }
    }
    void loadGoals();
    return () => {
      cancelled = true;
    };
  }, [paperId, provider]);

  function explainGoal(goal: StudyGoal) {
    const subquestions = goal.subquestions?.length
      ? ` Subquestions: ${goal.subquestions.map((item) => item.question).join(" ")}`
      : "";
    setChatActive(true);
    setQueuedPrompt({
      id: Date.now(),
      text: `Explain this study goal in detail for this paper: ${goal.title}. Goal details: ${goal.description}${subquestions} Start with pages ${goal.source_pages.join(", ")} if they are relevant.`
    });
  }

  return (
    <aside className="flex min-h-0 flex-col bg-ink">
      <div className="flex h-16 shrink-0 items-center justify-between border-b border-line bg-panel px-4">
        <div className="flex items-center gap-2">
          <div className="grid grid-cols-2 rounded-lg border border-line bg-ink p-1 text-xs">
            <button
              onClick={() => changeProvider("local")}
              className={`rounded-md px-3 py-1.5 transition ${provider === "local" ? "bg-white text-black" : "text-zinc-400 hover:text-white"}`}
            >
              Local
            </button>
            <button
              onClick={() => changeProvider("groq")}
              className={`rounded-md px-3 py-1.5 transition ${provider === "groq" ? "bg-acid text-black" : "text-zinc-400 hover:text-white"}`}
            >
              Groq
            </button>
          </div>
        </div>
        <div className="flex items-center justify-between gap-3">
          <button onClick={() => setChatActive(false)} className="inline-flex items-center gap-2 rounded-md border border-line bg-panel px-3 py-2 text-sm text-white">
            <BookOpen size={16} />
            ScholAR
            <span className="rounded-md bg-red-500 px-2 py-0.5 text-xs font-semibold text-white">0/8</span>
          </button>
        </div>
      </div>

      {!chatActive ? (
      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        <div className="mb-6 text-center">
          <p className="text-xs uppercase tracking-[0.2em] text-acid">
            {provider === "groq" ? "Groq API session" : "Local Qwen session"}
          </p>
          <h1 className="mt-2 text-2xl font-semibold text-white">Let&apos;s study research!</h1>
          {providerNotice ? <p className="mt-2 text-xs text-amber-300">{providerNotice}</p> : null}
        </div>

        <StudyGoals goals={goals} loading={loadingGoals} onGoalClick={explainGoal} />
      </div>
      ) : (
        <div className="shrink-0 border-b border-line px-5 py-3 text-sm text-zinc-400">
          Chat session active. Click ScholAR to return to the study plan.
        </div>
      )}

      <ChatBox
        paperId={paperId}
        queuedPrompt={queuedPrompt}
        provider={provider}
        onProviderChange={changeProvider}
        onCitationClick={onCitationClick}
        expanded={chatActive}
        onChatActivity={() => setChatActive(true)}
      />
    </aside>
  );
}
