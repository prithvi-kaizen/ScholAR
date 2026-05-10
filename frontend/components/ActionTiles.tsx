import { Bookmark, Compass, FileUp, Search, Settings, Sparkles } from "lucide-react";
import { useRef } from "react";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

const tiles = [
  { title: "Start Study Session", icon: Sparkles },
  { title: "Search", icon: Search },
  { title: "Upload PDF", icon: FileUp },
  { title: "Bookmarks", icon: Bookmark },
  { title: "Preferences", icon: Settings },
  { title: "Roadmap", icon: Compass },
];

interface UploadResult {
  paper_id: string;
  pages: number;
  chunks: number;
  goals: any[];
}

export function ActionTiles({ onUploadSuccess = (_data: UploadResult) => {} }: { onUploadSuccess?: (data: UploadResult) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    try {
      // Step 1: Upload the PDF
      const uploadRes = await fetch(`${backendUrl}/api/papers/upload`, { method: "POST", body: form });
      if (!uploadRes.ok) {
        const err = await uploadRes.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed (${uploadRes.status})`);
      }
      const uploadData = await uploadRes.json();

      // Step 2: Kick off the study goals pipeline
      const goalsRes = await fetch(`${backendUrl}/api/papers/${uploadData.paper_id}/study-goals`, {
        method: "POST",
      });
      if (!goalsRes.ok) {
        const err = await goalsRes.json().catch(() => ({}));
        throw new Error(err.detail || `Goals failed (${goalsRes.status})`);
      }
      const goalsData = await goalsRes.json();

      onUploadSuccess({ ...uploadData, goals: goalsData.goals });
    } catch (err) {
      console.error("Upload error:", (err as Error).message);
    }

    e.target.value = "";
  };

  const handleTileClick = (title: string) => {
    if (title === "Upload PDF") inputRef.current?.click();
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={handleUpload}
      />

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {tiles.map((tile) => (
          <button
            key={tile.title}
            onClick={() => handleTileClick(tile.title)}
            className="flex items-center gap-3 rounded-lg border border-line bg-panel px-4 py-4 text-left text-sm font-medium text-zinc-200 transition hover:border-zinc-500 hover:bg-panelSoft"
          >
            <span className="grid h-9 w-9 place-items-center rounded-md bg-white/5 text-acid">
              <tile.icon size={18} />
            </span>
            {tile.title}
          </button>
        ))}
      </section>
    </>
  );
}