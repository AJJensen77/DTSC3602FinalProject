"use client";

import { Search } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { MovieCard, MovieGrid } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { api, MIN_RATINGS, type Movie } from "@/lib/api";
import { useRatings, useRequireSession } from "@/lib/use-ratings";

export default function RatePage() {
  const { session } = useRequireSession();
  const { ratings, setRating } = useRatings();
  const [starter, setStarter] = useState<Movie[] | null>(null);
  const [query, setQuery] = useState("");
  const [found, setFound] = useState<{ q: string; movies: Movie[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const q = query.trim();
  const active = q.length >= 2;
  const searching = active && found?.q !== q;

  useEffect(() => {
    api.starter().then(setStarter).catch((e) => setError(e.message));
  }, []);

  // Debounced title search.
  useEffect(() => {
    if (!active) return;
    const t = setTimeout(() => {
      api.search(q)
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

  if (!session) return null;
  const count = ratings.size;
  const ready = count >= MIN_RATINGS;
  const shown = !active ? starter : searching ? null : found!.movies;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3 rounded-xl border p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-1 flex-col gap-2">
          <h1 className="text-xl font-semibold">Rate movies you&apos;ve seen</h1>
          <p className="text-sm text-muted-foreground">
            {ready
              ? `You've rated ${count} movies. Rate more any time — your picks get sharper.`
              : `Rate at least ${MIN_RATINGS} to unlock your picks: ${count} / ${MIN_RATINGS}.`}
          </p>
          <Progress value={Math.min(100, (count / MIN_RATINGS) * 100)} className="max-w-md" aria-label="Rating progress" />
        </div>
        <Button disabled={!ready} nativeButton={!ready} render={ready ? <Link href="/for-you" /> : undefined}>
          See my picks
        </Button>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
        <Input
          className="pl-9"
          placeholder="Search for a movie you've seen…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search movies"
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
            <MovieCard key={m.movie_id} movie={m}>
              <StarRating
                size="sm"
                label={`Your rating for ${m.title}`}
                value={ratings.get(m.movie_id) ?? null}
                onChange={(v) => setRating(m.movie_id, v)}
              />
            </MovieCard>
          ))}
        </MovieGrid>
      )}
    </div>
  );
}
