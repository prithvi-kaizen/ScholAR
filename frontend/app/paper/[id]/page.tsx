import { PdfViewer } from "../../../components/PdfViewer";
import { StudyPanel } from "../../../components/StudyPanel";
import { Bell, FlaskConical, Newspaper, Search } from "lucide-react";

interface PaperStudyPageProps {
  params: Promise<{ id: string }>;
}

export default async function PaperStudyPage({ params }: PaperStudyPageProps) {
  const { id } = await params;

  return (
    <main className="flex h-screen flex-col overflow-hidden bg-ink text-white">
      <header className="flex h-[72px] shrink-0 items-center justify-between border-b border-line bg-ink px-5">
        <div className="text-xl font-semibold">ScholAR</div>
        <label className="hidden w-full max-w-xl items-center gap-2 rounded-md border border-line bg-white/10 px-4 py-2.5 text-zinc-400 md:flex">
          <Search size={17} />
          <input className="w-full bg-transparent text-sm outline-none placeholder:text-zinc-500" placeholder="Search papers..." />
          <span className="rounded border border-zinc-600 px-1.5 py-0.5 text-xs text-zinc-400">⌘K</span>
        </label>
        <div className="flex items-center gap-4 text-zinc-300">
          <button className="rounded-md p-2 hover:bg-white/5 hover:text-white" aria-label="Labs">
            <FlaskConical size={20} />
          </button>
          <button className="rounded-md p-2 hover:bg-white/5 hover:text-white" aria-label="Notifications">
            <Bell size={20} />
          </button>
          <button className="rounded-md p-2 hover:bg-white/5 hover:text-white" aria-label="Library">
            <Newspaper size={20} />
          </button>
        </div>
      </header>
      <div className="grid min-h-0 flex-1 lg:grid-cols-[minmax(520px,1fr)_minmax(430px,1fr)]">
        <PdfViewer paperId={id} />
        <StudyPanel paperId={id} />
      </div>
    </main>
  );
}
