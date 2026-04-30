"use client";

import { FileText } from "lucide-react";
import type { Paper } from "../types/paper";
import { Badge } from "./Badge";

interface PaperCardProps {
  paper: Paper;
  onSelect: (paper: Paper) => void;
}

export function PaperCard({ paper, onSelect }: PaperCardProps) {
  return (
    <button
      onClick={() => onSelect(paper)}
      className="group flex h-full flex-col rounded-lg border border-line bg-panel p-4 text-left transition hover:-translate-y-0.5 hover:border-zinc-500 hover:bg-panelSoft"
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {(paper.categories.length ? paper.categories : ["cs.AI"]).slice(0, 3).map((category) => (
            <Badge key={`${paper.id}-${category}`} label={category} />
          ))}
        </div>
        <FileText size={18} className="mt-0.5 shrink-0 text-zinc-600 transition group-hover:text-acid" />
      </div>
      <h3 className="line-clamp-3 text-sm font-semibold leading-6 text-white">{paper.title}</h3>
      <p className="mt-3 line-clamp-2 text-xs leading-5 text-zinc-400">{paper.authors.join(", ") || "Unknown authors"}</p>
      <div className="mt-auto pt-4 text-xs text-zinc-500">{paper.year || "Unknown year"}</div>
    </button>
  );
}
