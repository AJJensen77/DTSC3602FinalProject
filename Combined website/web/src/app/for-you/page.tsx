"use client";

import { RefreshCw } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { LibraryCard } from "@/components/library-card";
import { MovieGrid } from "@/components/movie-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { useRatings, useRequireSession } from "@/lib/use-ratings";
import { useRecommendations } from "@/lib/use-recs";
import { cn } from "@/lib/utils";

export default function ForYouPage() {
  const { session } = useRequireSession();
  const { ratings, lists, loading: libraryLoading } = useRatings();
  const [genres, setGenres] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [nonce, setNonce] = useState(0);
  const { loading, slow, data, error } = useRecommendations(selected, { k: 20, nonce });

  useEffect(() => {
    api.genres().then(setGenres).catch(() => setGenres([]));
  }, []);

  const refresh = () => setNonce((n) => n + 1);
  const toggle = (g: string) => setSelected((s) => (s.includes(g) ? s.filter((x) => x !== g) : [...s, g]));

  if (!session || libraryLoading) return null;
  const finished = [...lists.values()].filter((l) => l === "finished").length;
  const empty = lists.size === 0 && ratings.size === 0;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Recommended for you</h1>
        <p className="text-sm text-muted-foreground">
          {empty
            ? "Popular, well-liked movies for now. Mark movies as Finished to personalize this."
            : `Based on ${finished} finished, ${lists.size - finished} planned and ${ratings.size} rated movies.` +
              (data?.used_gradient_boost ? " Stars show the score we predict you'd give." : "")}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Filter by genre">
        <span className="mr-1 text-sm font-medium">Genres:</span>
        <button
          type="button"
          onClick={() => setSelected([])}
          aria-pressed={selected.length === 0}
          className={cn(chip, selected.length === 0 && chipOn)}
        >
          All
        </button>
        {genres.map((g) => (
          <button
            key={g}
            type="button"
            onClick={() => toggle(g)}
            aria-pressed={selected.includes(g)}
            className={cn(chip, selected.includes(g) && chipOn)}
          >
            {g}
          </button>
        ))}
        <Button variant="ghost" size="sm" className="ml-auto" onClick={refresh} disabled={loading}>
          <RefreshCw className={cn(loading && "animate-spin")} aria-hidden /> Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Couldn&apos;t get your recommendations</AlertTitle>
          <AlertDescription className="flex flex-col gap-3">
            <span>{error}</span>
            <Button variant="outline" size="sm" className="self-start" onClick={refresh}>
              Try again
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {loading ? (
        <div className="flex flex-col gap-3">
          <p className="text-sm text-muted-foreground" aria-live="polite">
            {slow ? "Waking up the recommendation model — the first request can take about 20 seconds…" : "Finding movies for you…"}
          </p>
          <MovieGrid>
            {Array.from({ length: 10 }, (_, i) => <Skeleton key={i} className="aspect-[2/3] w-full rounded-xl" />)}
          </MovieGrid>
        </div>
      ) : (
        data && (
          <>
            {data.note && (
              <Alert>
                <AlertDescription className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <span>{data.note}</span>
                  {!data.used_gradient_boost && (
                    <Button size="sm" variant="outline" nativeButton={false} render={<Link href="/list" />}>
                      Score your movies
                    </Button>
                  )}
                </AlertDescription>
              </Alert>
            )}
            <MovieGrid>
              {data.recommendations.map((m) => (
                <LibraryCard key={m.movie_id} movie={m} />
              ))}
            </MovieGrid>
          </>
        )
      )}
    </div>
  );
}

const chip =
  "rounded-full border px-3 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring";
const chipOn = "border-foreground bg-foreground text-background hover:text-background";
