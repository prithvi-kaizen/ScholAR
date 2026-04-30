"use client";

import { FormEvent, useState } from "react";
import { Search } from "lucide-react";

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading?: boolean;
}

export function SearchBar({ onSearch, loading = false }: SearchBarProps) {
  const [query, setQuery] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSearch(query);
  }

  return (
    <form onSubmit={handleSubmit} className="mx-auto flex w-full max-w-3xl items-center gap-3 rounded-lg border border-line bg-panel px-4 py-3 shadow-glow">
      <Search size={20} className="shrink-0 text-zinc-500" />
      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search arXiv papers..."
        className="min-w-0 flex-1 bg-transparent text-base text-white outline-none placeholder:text-zinc-500"
      />
      <button
        type="submit"
        disabled={loading}
        className="rounded-md bg-white px-4 py-2 text-sm font-semibold text-black transition hover:bg-acid disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Searching..." : "Search"}
      </button>
    </form>
  );
}
