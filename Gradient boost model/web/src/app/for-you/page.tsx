"use client";

import { RefreshCw } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { MovieCard, MovieGrid } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, MIN_RATINGS, type RecommendResponse } from "@/lib/api";
import { useRatings, useRequireSession } from "@/lib/use-ratings";
import { cn } from "@/lib/utils";

export default function ForYouPage() {
  const { session } = useRequireSession();
  const { ratings, loading: ratingsLoading, setRating } = useRatings();
  const [genres, setGenres] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  // Each fetch is identified by the genre filter plus a refresh counter; the page is
  // loading until the latest result matches the latest request.
  const [nonce, setNonce] = useState(0);
  const requestKey = `${selected.join("|")}#${nonce}`;
  const [result, setResult] = useState<Result | null>(null);
  const [slowKey, setSlowKey] = useState<string | null>(null);
  // Read the latest ratings at fetch time without refetching on every star click.
  const ratingsRef = useRef(ratings);
  useEffect(() => {
    ratingsRef.current = ratings;
  }, [ratings]);

  const enough = ratings.size >= MIN_RATINGS;
  const loading = enough && result?.key !== requestKey;
  const slow = loading && slowKey === requestKey;
  const data = loading ? null : (result?.data ?? null);
  const error = loading ? null : (result?.error ?? null);

  useEffect(() => {
    api.genres().then(setGenres).catch(() => setGenres([]));
  }, []);

  // Fetch once ratings are loaded, whenever the genre filter changes, and on Refresh.
  useEffect(() => {
    if (ratingsLoading || !enough) return;
    let cancelled = false;
    const key = requestKey;
    const slowTimer = setTimeout(() => setSlowKey(key), 3000);
    api
      .recommend(ratingsRef.current, selected)
      .then((d) => !cancelled && setResult({ key, data: d, error: null }))
      .catch((e: Error) => !cancelled && setResult({ key, data: null, error: e.message }))
      .finally(() => clearTimeout(slowTimer));
    return () => {
      cancelled = true;
      clearTimeout(slowTimer);
    };
  }, [ratingsLoading, enough, selected, requestKey]);

  const refresh = () => setNonce((n) => n + 1);

  function toggle(g: string) {
    setSelected((s) => (s.includes(g) ? s.filter((x) => x !== g) : [...s, g]));
  }

  if (!session || ratingsLoading) return null;

  if (!enough) {
    return (
      <Alert className="mx-auto max-w-lg">
        <AlertTitle>Almost there</AlertTitle>
        <AlertDescription className="flex flex-col gap-3">
          <span>
            Rate at least {MIN_RATINGS} movies so we can learn your taste. You&apos;ve rated {ratings.size} so far.
          </span>
          <Button className="self-start" nativeButton={false} render={<Link href="/rate" />}>
            Rate movies
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Your picks</h1>
        <p className="text-sm text-muted-foreground">
          Based on your {ratings.size} ratings. Each is a movie we predict you&apos;d rate 3.5★ or higher.
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
          <AlertTitle>Couldn&apos;t get your picks</AlertTitle>
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
            {slow ? "Waking up the recommendation model — the first request can take about 15 seconds…" : "Finding your picks…"}
          </p>
          <MovieGrid>
            {Array.from({ length: 10 }, (_, i) => <Skeleton key={i} className="aspect-[2/3] w-full rounded-xl" />)}
          </MovieGrid>
        </div>
      ) : data && (
        <>
          {data.note && (
            <Alert>
              <AlertDescription>{data.note} Try adding another genre.</AlertDescription>
            </Alert>
          )}
          {data.recommendations.length > 0 && (
            <MovieGrid>
              {data.recommendations.map((m) => (
                <MovieCard key={m.movie_id} movie={m}>
                  <div className="flex flex-col gap-2">
                    <Badge variant="secondary" className="self-start">
                      Predicted for you: {m.predicted_rating.toFixed(1)}★
                    </Badge>
                    <div>
                      <p className="text-xs text-muted-foreground">Seen it? Rate it:</p>
                      <StarRating
                        size="sm"
                        label={`Your rating for ${m.title}`}
                        value={ratings.get(m.movie_id) ?? null}
                        onChange={(v) => setRating(m.movie_id, v)}
                      />
                    </div>
                  </div>
                </MovieCard>
              ))}
            </MovieGrid>
          )}
        </>
      )}
    </div>
  );
}

const chip =
  "rounded-full border px-3 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring";
const chipOn = "border-foreground bg-foreground text-background hover:text-background";

type Result = { key: string; data: RecommendResponse | null; error: string | null };
