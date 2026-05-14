import Link from "next/link";
import { StudyWorkspace } from "../../../components/StudyWorkspace";
import { ThemeToggle } from "../../../components/ThemeToggle";

interface PaperStudyPageProps {
  params: Promise<{ id: string }>;
}

export default async function PaperStudyPage({ params }: PaperStudyPageProps) {
  const { id } = await params;

  return (
    <main className="flex h-screen flex-col overflow-hidden bg-ink text-white">
      <header className="flex h-[72px] shrink-0 items-center justify-between border-b border-line bg-ink px-5">
        <Link href="/" className="rounded-md text-xl font-semibold transition hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-acid">
          ScholAR
        </Link>
        <ThemeToggle />
      </header>
      <StudyWorkspace paperId={id} />
    </main>
  );
}
