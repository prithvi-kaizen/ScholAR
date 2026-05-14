"use client";

import { useState } from "react";
import type { Citation } from "../types/paper";
import { PdfViewer } from "./PdfViewer";
import { StudyPanel } from "./StudyPanel";

interface StudyWorkspaceProps {
  paperId: string;
}

export function StudyWorkspace({ paperId }: StudyWorkspaceProps) {
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);

  return (
    <div className="grid min-h-0 flex-1 lg:grid-cols-[minmax(520px,1fr)_minmax(430px,1fr)]">
      <PdfViewer paperId={paperId} activeCitation={activeCitation} />
      <StudyPanel paperId={paperId} onCitationClick={setActiveCitation} />
    </div>
  );
}
