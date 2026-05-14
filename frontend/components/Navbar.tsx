import Link from "next/link";
import { ThemeToggle } from "./ThemeToggle";

export function Navbar() {
  return (
    <nav className="flex h-16 items-center justify-between border-b border-line bg-ink/95 px-5">
      <Link href="/" className="flex items-center gap-3 rounded-md outline-none transition hover:opacity-80 focus-visible:ring-2 focus-visible:ring-acid">
        <div className="grid h-8 w-8 place-items-center rounded-md border border-acid/40 bg-acid/10 text-sm font-bold text-acid">
          AR
        </div>
        <span className="text-base font-semibold text-white">ScholAR</span>
      </Link>
      <ThemeToggle />
    </nav>
  );
}
