import { Bell, Bookmark, Settings } from "lucide-react";

export function Navbar() {
  return (
    <nav className="flex h-16 items-center justify-between border-b border-line bg-ink/95 px-5">
      <div className="flex items-center gap-3">
        <div className="grid h-8 w-8 place-items-center rounded-md border border-acid/40 bg-acid/10 text-sm font-bold text-acid">
          AR
        </div>
        <span className="text-base font-semibold text-white">ScholAR</span>
      </div>
      <div className="flex items-center gap-2 text-zinc-400">
        <button className="rounded-md border border-line p-2 transition hover:border-zinc-500 hover:text-white" aria-label="Bookmarks">
          <Bookmark size={16} />
        </button>
        <button className="rounded-md border border-line p-2 transition hover:border-zinc-500 hover:text-white" aria-label="Notifications">
          <Bell size={16} />
        </button>
        <button className="rounded-md border border-line p-2 transition hover:border-zinc-500 hover:text-white" aria-label="Settings">
          <Settings size={16} />
        </button>
      </div>
    </nav>
  );
}
