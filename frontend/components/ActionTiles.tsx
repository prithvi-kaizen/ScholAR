import { Bookmark, Compass, FileUp, Map, Search, Settings, Sparkles } from "lucide-react";

const tiles = [
  { title: "Start Study Session", icon: Sparkles },
  { title: "Search", icon: Search },
  { title: "Upload PDF", icon: FileUp },
  { title: "Bookmarks", icon: Bookmark },
  { title: "Preferences", icon: Settings },
  { title: "Roadmap", icon: Map }
];

export function ActionTiles() {
  return (
    <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {tiles.map((tile) => {
        const Icon = tile.icon === Map ? Compass : tile.icon;
        return (
          <button
            key={tile.title}
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
