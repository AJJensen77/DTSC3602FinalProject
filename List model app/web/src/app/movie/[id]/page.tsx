"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ListActions } from "@/components/list-actions";
import { genreName, MovieCard, MovieRow, Poster, RowItem } from "@/components/movie-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type Movie, type SimilarMovie } from "@/lib/api";
import { useLists } from "@/lib/use-lists";

// Fetch extra so there are still enough after hiding films already on your lists.
const SHOW = 12;
const FETCH = 30;

type State = { id: number; movie: Movie | null; similar: SimilarMovie[] | null; error: string | null };

export default function MoviePage() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  const router = useRouter();
  const [state, setState] = useState<State | null>(null);
  const [attempt, setAttempt] = useState(0);
  const valid = Number.isInteger(id) && id > 0;
  const { entries } = useLists();

  useEffect(() => {
    if (!valid) return;
    let cancelled = false;
    Promise.all([api.movie(id), api.similar(id, FETCH)])
      .then(([movie, similar]) => !cancelled && setState({ id, movie, similar, error: null }))
      .catch((e: Error) => !cancelled && setState({ id, movie: null, similar: null, error: e.message }));
    return () => {
      cancelled = true;
    };
  }, [id, valid, attempt]);

  const current = state?.id === id ? state : null;
  const back = (
    <Button variant="ghost" size="sm" className="self-start" onClick={() => router.back()}>
      <ArrowLeft aria-hidden /> Back
    </Button>
  );

  if (!valid || current?.error) {
    return (
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        {back}
        <Alert variant="destructive">
          <AlertTitle>Couldn&apos;t load this movie</AlertTitle>
          <AlertDescription className="flex flex-col gap-3">
            <span>{valid ? current?.error : "That isn't a valid movie link."}</span>
            <div className="flex gap-2">
              {valid && (
                <Button variant="outline" size="sm" onClick={() => setAttempt((a) => a + 1)}>
                  Try again
                </Button>
              )}
              <Button variant="outline" size="sm" nativeButton={false} render={<Link href="/browse" />}>
                Browse movies
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const movie = current?.movie;
  // Hide films already rated or on Want to Watch: this row is for discovering new ones.
  const similar = current?.similar?.filter((m) => !entries.has(m.movie_id)).slice(0, SHOW) ?? null;
  return (
    <div className="flex flex-col gap-8">
      {back}
      {!movie ? (
        <div className="flex flex-col gap-6 sm:flex-row">
          <Skeleton className="aspect-[2/3] w-48 shrink-0 rounded-xl" />
          <div className="flex flex-1 flex-col gap-3">
            <Skeleton className="h-8 w-2/3" />
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-24 w-full max-w-sm" />
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-6 sm:flex-row">
          <div className="w-48 shrink-0 overflow-hidden rounded-xl border">
            <Poster movie={movie} />
          </div>
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <h1 className="text-3xl font-semibold">{movie.title}</h1>
              <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                {movie.year && <span>{movie.year}</span>}
                {movie.genres.map((g) => (
                  <Badge key={g} variant="outline">{genreName(g)}</Badge>
                ))}
                {movie.is_new && <Badge>New release</Badge>}
              </div>
              {movie.imdb_rating !== null && (
                <p className="text-sm">
                  IMDb rating <span className="font-semibold tabular-nums">{movie.imdb_rating.toFixed(1)}</span>
                  {movie.imdb_id && (
                    <>
                      {" · "}
                      <a href={`https://www.imdb.com/title/${movie.imdb_id}/`} target="_blank" rel="noreferrer"
                         className="text-primary underline-offset-2 hover:underline">
                        View on IMDb
                      </a>
                    </>
                  )}
                </p>
              )}
            </div>
            <div className="max-w-xs rounded-xl border bg-card p-4">
              <ListActions movie={movie} size="md" />
            </div>
          </div>
        </div>
      )}

      <section className="flex flex-col gap-3">
        <div>
          <h2 className="text-lg font-semibold">More like this</h2>
          <p className="text-xs text-muted-foreground">
            {movie?.is_new
              ? "We have no viewing data for this film yet, so these are matched through older films with similar genres, director and cast."
              : "Films watched by the same kinds of people."}
          </p>
        </div>
        {!similar ? (
          <div className="flex gap-3 overflow-hidden">
            {Array.from({ length: 7 }, (_, j) => (
              <Skeleton key={j} className="aspect-[2/3] w-36 shrink-0 rounded-xl sm:w-44" />
            ))}
          </div>
        ) : similar.length === 0 ? (
          <p className="text-sm text-muted-foreground">You&apos;ve already got every close match on your lists.</p>
        ) : (
          <MovieRow label="More like this">
            {similar.map((m) => (
              <RowItem key={m.movie_id}>
                <MovieCard movie={m}>
                  <ListActions movie={m} />
                </MovieCard>
              </RowItem>
            ))}
          </MovieRow>
        )}
      </section>
    </div>
  );
}
