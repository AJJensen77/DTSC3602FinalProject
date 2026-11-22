"use client";

import { Bookmark, Star, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { genreName, Poster } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type Movie } from "@/lib/api";
import { useLists, useRequireSession } from "@/lib/use-lists";
import { cn } from "@/lib/utils";

type Tab = "rated" | "want";

export default function MyListsPage() {
  const { session } = useRequireSession();
  const { entries, rated, want, loading, error: loadError, rate, toggleWant } = useLists();
  const [tab, setTab] = useState<Tab>("rated");
  const [movies, setMovies] = useState<Map<number, Movie>>(new Map());
  const [error, setError] = useState<string | null>(null);

  const ids = useMemo(() => [...entries.keys()], [entries]);
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
  const list = tab === "rated" ? rated : want;

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-4">
      <div>
        <h1 className="text-2xl font-semibold">My lists</h1>
        <p className="text-sm text-muted-foreground">
          Ratings count most toward your picks; Want to Watch nudges them. Rate a Want to Watch film once you&apos;ve seen
          it and it moves to My Ratings.
        </p>
      </div>

      <div className="flex gap-2" role="tablist" aria-label="Lists">
        {([["rated", "My Ratings", Star, rated.length], ["want", "Want to Watch", Bookmark, want.length]] as const).map(
          ([key, label, Icon, n]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={tab === key}
              onClick={() => setTab(key)}
              className={cn(
                "flex items-center gap-2 rounded-full border px-4 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground",
                tab === key && "border-primary bg-primary text-primary-foreground hover:text-primary-foreground",
              )}
            >
              <Icon className="size-4" aria-hidden />
              {label}
              <span className="tabular-nums">({n})</span>
            </button>
          ),
        )}
      </div>

      {(error || loadError) && (
        <Alert variant="destructive">
          <AlertDescription>{error ?? loadError}</AlertDescription>
        </Alert>
      )}

      {list.length === 0 ? (
        <Alert>
          <AlertDescription className="flex flex-col gap-3">
            <span>
              {tab === "rated"
                ? "You haven't rated anything yet. Rate films you've seen to get personal picks."
                : "Nothing saved yet. Hit Want to Watch on any film that catches your eye."}
            </span>
            <Button className="self-start" nativeButton={false} render={<Link href="/browse" />}>
              Browse movies
            </Button>
          </AlertDescription>
        </Alert>
      ) : (
        <ul className="divide-y rounded-xl border">
          {list.map((id) => {
            const movie = movies.get(id);
            const entry = entries.get(id);
            return (
              <li key={id} className="flex flex-wrap items-center gap-4 p-3">
                <Link href={`/movie/${id}`} className="w-12 shrink-0 overflow-hidden rounded-md">
                  {movie ? <Poster movie={movie} compact /> : <Skeleton className="aspect-[2/3] w-full" />}
                </Link>
                <div className="min-w-0 flex-1">
                  <Link href={`/movie/${id}`} className="block truncate font-medium hover:underline">
                    {movie?.title ?? "Loading…"}
                  </Link>
                  <p className="truncate text-xs text-muted-foreground">
                    {[movie?.year, movie?.genres.slice(0, 3).map(genreName).join(" · ")].filter(Boolean).join(" · ")}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {tab === "want" && <span className="text-xs text-muted-foreground">Watched it?</span>}
                  <StarRating
                    size="sm"
                    label={`Your rating for ${movie?.title ?? "this movie"}`}
                    value={entry?.stars ?? null}
                    onChange={(v) => rate(id, v)}
                  />
                  {tab === "want" && (
                    <button
                      type="button"
                      onClick={() => toggleWant(id)}
                      className="rounded-sm p-0.5 text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                      aria-label={`Remove ${movie?.title ?? "this movie"} from Want to Watch`}
                      title="Remove from Want to Watch"
                    >
                      <X className="size-4" />
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
