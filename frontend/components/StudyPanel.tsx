"use client";

import { useEffect, useState } from "react";
import { MessageSquare, BookOpen, ListChecks } from "lucide-react";
import type { Citation, CustomSnippet, StudyGoal } from "../types/paper";
import { ChatBox } from "./ChatBox";
import { StudyGoals } from "./StudyGoals";
import { ReferencesPanel } from "./ReferencesPanel";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

type Tab = "chat" | "goals" | "references";

interface StudyPanelProps {
  paperId: string;
  onCitationClick: (citation: Citation) => void;
  activeSnippet?: CustomSnippet | null;
  onDismissSnippet?: () => void;
}

const defaultGoals: StudyGoal[] = [
  { id: "goal_1", title: "Define problem and motivation",       description: "Clarify the research problem, why it matters, and what gap the paper is trying to close.",              source_pages: [1], status: "not_started" },
  { id: "goal_2", title: "Summarize core idea",                description: "Capture the central contribution in plain language before going into details.",                         source_pages: [1], status: "not_started" },
  { id: "goal_3", title: "Explain methodology",                description: "Break down the method, assumptions, and the main technical workflow.",                                  source_pages: [1], status: "not_started" },
  { id: "goal_4", title: "Identify algorithm or architecture", description: "Map the important model components, algorithms, or system design choices.",                             source_pages: [1], status: "not_started" },
  { id: "goal_5", title: "Understand experimental setup",      description: "Review datasets, baselines, metrics, and implementation settings.",                                    source_pages: [1], status: "not_started" },
  { id: "goal_6", title: "Report key results",                 description: "Extract the main quantitative and qualitative findings from the paper.",                               source_pages: [1], status: "not_started" },
  { id: "goal_7", title: "Discuss limitations",                description: "Identify what the authors admit is limited, uncertain, or left for future work.",                       source_pages: [1], status: "not_started" },
  { id: "goal_8", title: "Convert paper into implementation plan", description: "Turn the paper into a practical build plan with steps, dependencies, and risks.",                  source_pages: [1], status: "not_started" },
];

export function StudyPanel({ paperId, onCitationClick, activeSnippet = null, onDismissSnippet }: StudyPanelProps) {
  const [activeTab,   setActiveTab]   = useState<Tab>("chat");
  const [goals,       setGoals]       = useState<StudyGoal[]>([]);
  const [loadingGoals, setLoadingGoals] = useState(false);
  const [queuedPrompt, setQueuedPrompt] = useState<{ id: number; text: string } | null>(null);
  const [secondaryPaperIds, setSecondaryPaperIds] = useState<Set<string>>(new Set());

  // Auto-switch to chat tab when snippet is captured
  useEffect(() => {
    if (activeSnippet) {
      setActiveTab("chat");
    }
  }, [activeSnippet]);

  // Tab switching shortcuts: 1 -> Chat, 2 -> Goals, 3 -> References
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName)) return;
      if (e.key === "1") {
        e.preventDefault();
        setActiveTab("chat");
      } else if (e.key === "2") {
        e.preventDefault();
        setActiveTab("goals");
      } else if (e.key === "3") {
        e.preventDefault();
        setActiveTab("references");
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // ---------- study goals ----------
  useEffect(() => {
    let cancelled = false;
    async function loadGoals() {
      setGoals([]);
      setLoadingGoals(true);
      try {
        const res = await fetch(
          `${backendUrl}/api/papers/${encodeURIComponent(paperId)}/study-goals`,
          { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) }
        );
        if (!res.ok) throw new Error("Could not load study goals");
        const payload = await res.json();
        if (!cancelled) setGoals(payload.goals?.length ? payload.goals : defaultGoals);
      } catch {
        if (!cancelled) setGoals(defaultGoals);
      } finally {
        if (!cancelled) setLoadingGoals(false);
      }
    }
    void loadGoals();
    return () => { cancelled = true; };
  }, [paperId]);

  function explainGoal(goal: StudyGoal) {
    const subqs = goal.subquestions?.length
      ? ` Subquestions: ${goal.subquestions.map((q) => q.question).join(" ")}`
      : "";
    setActiveTab("chat");
    setQueuedPrompt({
      id: Date.now(),
      text: `Explain this study goal in detail for this paper: ${goal.title}. Goal: ${goal.description}${subqs} Start with pages ${goal.source_pages.join(", ")} if relevant.`,
    });
  }

  function explainSubquestion(goal: StudyGoal, subquestion: string) {
    setActiveTab("chat");
    setQueuedPrompt({
      id: Date.now(),
      text: `Regarding study goal "${goal.title}": ${subquestion} (Refer to pages ${goal.source_pages.join(", ")})`,
    });
  }

  function handleSecondaryPaperIngested(secId: string) {
    setSecondaryPaperIds((prev) => new Set([...prev, secId]));
  }

  const doneCount = goals.filter((g) => g.status === "done").length;

  // ---------- tab config ----------
  const tabs: { id: Tab; label: string; icon: typeof MessageSquare; badge?: string }[] = [
    {
      id: "chat",
      label: "Chat",
      icon: MessageSquare,
    },
    {
      id: "goals",
      label: "Study Goals",
      icon: ListChecks,
      badge: doneCount > 0 ? `${doneCount}/${goals.length}` : undefined,
    },
    {
      id: "references",
      label: "References",
      icon: BookOpen,
      badge: secondaryPaperIds.size > 0 ? `${secondaryPaperIds.size} active` : undefined,
    },
  ];

  return (
    <aside className="flex min-h-0 h-full flex-col bg-ink border-l border-line">

      {/* ── Top bar: tabs ── */}
      <div className="shrink-0 border-b border-line bg-panel">

        {/* Tab row */}
        <div className="flex border-t border-line">
          {tabs.map(({ id, label, icon: Icon, badge }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`relative flex flex-1 items-center justify-center gap-1.5 py-2.5 text-xs font-medium transition ${
                activeTab === id
                  ? "border-b-2 border-acid text-white"
                  : "border-b-2 border-transparent text-zinc-500 hover:text-zinc-300"
              }`}
            >
              <Icon size={13} />
              {label}
              {badge && (
                <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${
                  activeTab === id ? "bg-acid/20 text-acid" : "bg-zinc-700 text-zinc-400"
                }`}>
                  {badge}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* ── Tab content ── */}

      {/* CHAT tab: always rendered so chat state is preserved, just hidden */}
      <div className={`min-h-0 flex-1 flex flex-col ${activeTab === "chat" ? "" : "hidden"}`}>
        <ChatBox
          paperId={paperId}
          queuedPrompt={queuedPrompt}
          onCitationClick={onCitationClick}
          expanded={true}
          onChatActivity={() => setActiveTab("chat")}
          secondaryPaperIds={[...secondaryPaperIds]}
          activeSnippet={activeSnippet}
          onDismissSnippet={onDismissSnippet}
        />
      </div>

      {/* GOALS tab */}
      {activeTab === "goals" && (
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <StudyGoals
            goals={goals}
            loading={loadingGoals}
            onGoalClick={explainGoal}
            onSubquestionClick={explainSubquestion}
          />
        </div>
      )}

      {/* REFERENCES tab */}
      {activeTab === "references" && (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <ReferencesPanel
            paperId={paperId}
            onSecondaryPaperIngested={handleSecondaryPaperIngested}
            activeSecondaryIds={secondaryPaperIds}
            inline
          />
        </div>
      )}
    </aside>
  );
}
