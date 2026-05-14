import { Bookmark, FileUp } from "lucide-react";

const tiles = [
  { title: "Upload PDF", icon: FileUp },
  { title: "Bookmarks", icon: Bookmark }
];

interface ActionTilesProps {
  onUploadClick: () => void;
  onBookmarksClick: () => void;
}

export function ActionTiles({ onUploadClick, onBookmarksClick }: ActionTilesProps) {
  function handleTileClick(title: string) {
    if (title === "Upload PDF") onUploadClick();
    if (title === "Bookmarks") onBookmarksClick();
  }

  return (
    <section className="grid gap-3 sm:grid-cols-2">
      {tiles.map((tile) => {
        const Icon = tile.icon;
        return (
          <button
            key={tile.title}
            onClick={() => handleTileClick(tile.title)}
            className="flex items-center gap-3 rounded-lg border border-line bg-panel px-4 py-4 text-left text-sm font-medium text-zinc-200 transition hover:border-zinc-500 hover:bg-panelSoft"
          >
            <span className="grid h-9 w-9 place-items-center rounded-md bg-white/5 text-acid">
              <Icon size={18} />
            </span>
            {tile.title}
          </button>
        );
      })}
    </section>
  );
}
