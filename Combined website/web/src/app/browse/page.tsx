"use client";

import { Search } from "lucide-react";
import { useEffect, useState } from "react";

import { LibraryCard } from "@/components/library-card";
import { MovieGrid } from "@/components/movie-card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type Movie } from "@/lib/api";
import { useRatings, useSession } from "@/lib/use-ratings";

export default function BrowsePage() {
  const { session } = useSession();
  const { lists } = useRatings();
  const [starter, setStarter] = useState<Movie[] | null>(null);
  const [query, setQuery] = useState("");
  const [found, setFound] = useState<{ q: string; movies: Movie[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const q = query.trim();
  const active = q.length >= 2;
  const searching = active && found?.q !== q;

  useEffect(() => {
    api.starter(60).then(setStarter).catch((e) => setError(e.message));
  }, []);

  // Debounced title search.
  useEffect(() => {
    if (!active) return;
    const t = setTimeout(() => {
      api
        .search(q)
        .then((movies) => {
          setFound({ q, movies });
          setError(null);
        })
        .catch((e) => {
          setFound({ q, movies: [] });
          setError(e.message);
        });
    }, 300);
    return () => clearTimeout(t);
  }, [q, active]);

  const shown = !active ? starter : searching ? null : found!.movies;
  const finished = [...lists.values()].filter((l) => l === "finished").length;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Browse movies</h1>
        <p className="text-sm text-muted-foreground">
          {session
            ? finished === 0
              ? "Mark a movie as Finished (or give it stars) and your recommendations start right away."
              : `${finished} on Finished watching. Every movie you add sharpens your recommendations.`
            : "Search any movie. Log in to add movies to your lists."}
        </p>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
        <Input
          className="pl-9"
          placeholder="Search for a movie…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search movies"
          autoFocus
        />
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <h2 className="text-sm font-medium text-muted-foreground">
        {active ? (searching ? "Searching…" : `Results for “${q}”`) : "Popular picks across genres"}
      </h2>

      {shown === null ? (
        <MovieGrid>
          {Array.from({ length: 10 }, (_, i) => <Skeleton key={i} className="aspect-[2/3] w-full rounded-xl" />)}
        </MovieGrid>
      ) : shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">No movies match that title. Try fewer words.</p>
      ) : (
        <MovieGrid>
          {shown.map((m) => (
            <LibraryCard key={m.movie_id} movie={m} />
          ))}
        </MovieGrid>
      )}
    </div>
  );
}
