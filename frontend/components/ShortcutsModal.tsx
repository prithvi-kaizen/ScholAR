"use client";

import { useEffect } from "react";
import { X, Keyboard } from "lucide-react";

interface ShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SHORTCUT_GROUPS = [
  {
    title: "PDF Navigation",
    shortcuts: [
      { key: "J / ←", description: "Previous page" },
      { key: "K / →", description: "Next page" },
      { key: "+ / =", description: "Zoom in" },
      { key: "-", description: "Zoom out" },
      { key: "0", description: "Reset zoom" },
      { key: "S", description: "Toggle Snip Region (Screenshot tool)" },
    ],
  },
  {
    title: "Workspace & Chat",
    shortcuts: [
      { key: "1 / 2 / 3", description: "Switch tabs (Chat, Goals, References)" },
      { key: "Cmd + Enter", description: "Send message" },
      { key: "?", description: "Toggle keyboard shortcuts" },
      { key: "Esc", description: "Close modal / Clear focus" },
    ],
  },
];

export function ShortcutsModal({ isOpen, onClose }: ShortcutsModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-md rounded-2xl border border-zinc-700 bg-ink p-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/20 text-blue-400">
              <Keyboard size={18} />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Keyboard Shortcuts</h2>
              <p className="text-xs text-zinc-400">Quick actions for power users</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-zinc-400 transition hover:bg-zinc-800 hover:text-white"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        <div className="mt-5 space-y-6">
          {SHORTCUT_GROUPS.map((group, gIdx) => (
            <div key={gIdx} className="space-y-2.5">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
                {group.title}
              </h3>
              <div className="divide-y divide-zinc-800/60 rounded-xl border border-zinc-800 bg-zinc-900/40 px-3">
                {group.shortcuts.map((sc, scIdx) => (
                  <div key={scIdx} className="flex items-center justify-between py-2.5 text-xs">
                    <span className="text-zinc-300">{sc.description}</span>
                    <kbd className="inline-flex items-center gap-1 rounded border border-zinc-700 bg-zinc-800/80 px-2 py-0.5 font-mono text-[11px] font-medium text-zinc-200 shadow-sm">
                      {sc.key}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="rounded-lg bg-zinc-800 px-4 py-2 text-xs font-medium text-zinc-200 transition hover:bg-zinc-700 hover:text-white"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}
