"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Poster } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type Movie } from "@/lib/api";
import { useRatings, useRequireSession } from "@/lib/use-ratings";

export default function MyRatingsPage() {
  const { session } = useRequireSession();
  const { ratings, loading, error: loadError, setRating } = useRatings();
  const [movies, setMovies] = useState<Map<number, Movie>>(new Map());
  const [error, setError] = useState<string | null>(null);

  const ids = useMemo(() => [...ratings.keys()], [ratings]);
  const missing = ids.filter((id) => !movies.has(id));

  // Look up details only for films we haven't fetched yet.
  useEffect(() => {
    if (!missing.length) return;
    api.lookup(missing)
      .then((found) => setMovies((m) => new Map([...m, ...found.map((f) => [f.movie_id, f] as const)])))
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the ids string
  }, [missing.join(",")]);

  if (!session || loading) return null;

  const rows = ids
    .map((id) => ({ id, rating: ratings.get(id)!, movie: movies.get(id) }))
    .sort((a, b) => b.rating - a.rating || (a.movie?.title ?? "").localeCompare(b.movie?.title ?? ""));

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-4">
      <div>
        <h1 className="text-2xl font-semibold">My ratings</h1>
        <p className="text-sm text-muted-foreground">Change a rating any time; your picks update when you refresh them.</p>
      </div>
      {(error || loadError) && (
        <Alert variant="destructive">
          <AlertDescription>{error ?? loadError}</AlertDescription>
        </Alert>
      )}
      {rows.length === 0 ? (
        <Alert>
          <AlertDescription className="flex flex-col gap-3">
            <span>You haven&apos;t rated anything yet.</span>
            <Button className="self-start" nativeButton={false} render={<Link href="/rate" />}>
              Rate movies
            </Button>
          </AlertDescription>
        </Alert>
      ) : (
        <ul className="divide-y rounded-xl border">
          {rows.map(({ id, rating, movie }) => (
            <li key={id} className="flex items-center gap-4 p-3">
              <div className="w-12 shrink-0 overflow-hidden rounded-md">
                {movie ? <Poster movie={movie} compact /> : <Skeleton className="aspect-[2/3] w-full" />}
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">{movie?.title ?? "Loading…"}</p>
                <p className="truncate text-xs text-muted-foreground">
                  {[movie?.year, movie?.genres.slice(0, 3).join(" · ")].filter(Boolean).join(" · ")}
                </p>
              </div>
              <StarRating size="sm" label={`Your rating for ${movie?.title ?? "this movie"}`} value={rating}
                          onChange={(v) => setRating(id, v)} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
