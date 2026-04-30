const categoryColors: Record<string, string> = {
  NLP: "border-emerald-400/40 bg-emerald-400/12 text-emerald-200",
  LLM: "border-sky-400/40 bg-sky-400/12 text-sky-200",
  VLM: "border-purple-400/40 bg-purple-400/12 text-purple-200",
  DL: "border-orange-400/40 bg-orange-400/12 text-orange-200",
  RL: "border-violet-400/40 bg-violet-400/12 text-violet-200",
  GRAPH: "border-pink-400/40 bg-pink-400/12 text-pink-200"
};

function labelForCategory(category: string): string {
  const upper = category.toUpperCase();
  if (upper.includes("CL")) return "NLP";
  if (upper.includes("AI") || upper.includes("LG")) return "LLM";
  if (upper.includes("CV")) return "VLM";
  if (upper.includes("NE")) return "DL";
  if (upper.includes("RL")) return "RL";
  if (upper.includes("SI") || upper.includes("GRAPH")) return "GRAPH";
  return category;
}

interface BadgeProps {
  label: string;
  className?: string;
}

export function Badge({ label, className }: BadgeProps) {
  const displayLabel = labelForCategory(label);
  const colors = categoryColors[displayLabel] ?? "border-zinc-600 bg-zinc-800 text-zinc-300";
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${colors} ${
        className ?? ""
      }`}
    >
      {displayLabel}
    </span>
  );
}
