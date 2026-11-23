"use client";

import { ExternalLink, Play } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { LibraryCard } from "@/components/library-card";
import { ListButtons } from "@/components/list-buttons";
import { MovieGrid, Poster, trailerKey, trailerSrc } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, LIST_LABELS, type Movie, type MovieDetail } from "@/lib/api";
import { useRatings, useSession } from "@/lib/use-ratings";

function InfoRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 py-1.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right">{children}</span>
    </div>
  );
}

export default function MoviePage() {
  const { id } = useParams<{ id: string }>();
  const movieId = Number(id);
  const { session } = useSession();
  const { ratings, lists, setRating } = useRatings();
  const [movie, setMovie] = useState<{ id: number; data: MovieDetail | null; error: string | null } | null>(null);
  const [similar, setSimilar] = useState<{ id: number; movies: Movie[] } | null>(null);
  const [trailer, setTrailer] = useState<{ id: number; key: string | null } | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!Number.isInteger(movieId)) return;
    let cancelled = false;
    api
      .movie(movieId)
      .then((d) => !cancelled && setMovie({ id: movieId, data: d, error: null }))
      .catch((e: Error) => !cancelled && setMovie({ id: movieId, data: null, error: e.message }));
    api
      .similar(movieId, 15)
      .then((m) => !cancelled && setSimilar({ id: movieId, movies: m }))
      .catch(() => !cancelled && setSimilar({ id: movieId, movies: [] }));
    return () => {
      cancelled = true;
    };
  }, [movieId]);

  const m = movie?.id === movieId ? movie.data : null;
  const error = movie?.id === movieId ? movie.error : null;

  useEffect(() => {
    if (!m?.tmdb_id) return;
    let cancelled = false;
    trailerKey(m.tmdb_id).then((key) => !cancelled && setTrailer({ id: movieId, key }));
    return () => {
      cancelled = true;
    };
  }, [m?.tmdb_id, movieId]);

  if (error) {
    return (
      <Alert variant="destructive" className="mx-auto max-w-lg">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }
  if (!m) {
    return (
      <div className="flex gap-8">
        <Skeleton className="aspect-[2/3] w-56 shrink-0 rounded-xl" />
        <div className="flex flex-1 flex-col gap-3">
          <Skeleton className="h-8 w-2/3" />
          <Skeleton className="h-4 w-1/3" />
        </div>
      </div>
    );
  }

  const status = lists.get(movieId);
  const youtubeKey = trailer?.id === movieId ? trailer.key : null;
  const sims = similar?.id === movieId ? similar.movies : null;

  return (
    <div className="flex flex-col gap-10">
      <div className="flex flex-col gap-8 md:flex-row">
        {/* Left column: poster and the person's entry for this film. */}
        <aside className="flex w-full shrink-0 flex-col gap-4 md:w-60">
          <div className="mx-auto w-48 overflow-hidden rounded-xl border md:w-full">
            <Poster movie={m} />
          </div>
          <div className="flex flex-col gap-3 rounded-xl border p-4">
            <h2 className="text-sm font-semibold">Your entry</h2>
            {session ? (
              <>
                <p className="text-xs text-muted-foreground">
                  Status: <span className="text-foreground">{status ? LIST_LABELS[status] : "Not on your list"}</span>
                </p>
                <ListButtons movieId={movieId} title={m.title} />
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Your score</span>
                  <StarRating
                    value={ratings.get(movieId) ?? null}
                    onChange={(v) => setRating(movieId, v)}
                    label={`Your rating for ${m.title}`}
                  />
                </div>
              </>
            ) : (
              <Button size="sm" nativeButton={false} render={<Link href="/login" />}>
                Log in to add to your list
              </Button>
            )}
          </div>
          <div className="divide-y rounded-xl border px-4 py-2">
            <InfoRow label="Released">{m.year ?? "—"}</InfoRow>
            <InfoRow label="IMDb rating">
              {m.imdb_rating ? `${m.imdb_rating.toFixed(1)} / 10` : "—"}
            </InfoRow>
            <InfoRow label="IMDb votes">{m.imdb_votes ? m.imdb_votes.toLocaleString() : "—"}</InfoRow>
            <InfoRow label="MovieLens ratings">{m.movielens_ratings.toLocaleString()}</InfoRow>
          </div>
        </aside>

        {/* Right column: title, trailer, and links. */}
        <div className="flex min-w-0 flex-1 flex-col gap-5">
          <div className="flex flex-col gap-2">
            <h1 className="text-3xl font-semibold tracking-tight">{m.title}</h1>
            <div className="flex flex-wrap gap-1.5">
              {m.genres.map((g) => (
                <Badge key={g} variant="outline">
                  {g}
                </Badge>
              ))}
            </div>
          </div>

          {youtubeKey ? (
            <div className="overflow-hidden rounded-xl border bg-black">
              {playing ? (
                <iframe
                  src={trailerSrc(youtubeKey, { muted: false, controls: true, clip: false })}
                  title={`Trailer for ${m.title}`}
                  allow="autoplay; encrypted-media; fullscreen"
                  allowFullScreen
                  className="aspect-video w-full"
                />
              ) : (
                <button
                  type="button"
                  onClick={() => setPlaying(true)}
                  className="group relative flex aspect-video w-full items-center justify-center"
                  aria-label={`Play trailer for ${m.title}`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element -- YouTube thumbnail */}
                  <img
                    src={`https://i.ytimg.com/vi/${youtubeKey}/hqdefault.jpg`}
                    alt=""
                    className="absolute inset-0 size-full object-cover opacity-70 transition-opacity group-hover:opacity-90"
                  />
                  <span className="relative flex items-center gap-2 rounded-full bg-background/90 px-4 py-2 text-sm font-medium">
                    <Play className="size-4 fill-current" aria-hidden /> Play trailer
                  </span>
                </button>
              )}
            </div>
          ) : (
            <a
              className="inline-flex items-center gap-1.5 self-start text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
              href={`https://www.youtube.com/results?search_query=${encodeURIComponent(`${m.title} ${m.year ?? ""} trailer`)}`}
              target="_blank"
              rel="noreferrer"
            >
              <Play className="size-4" aria-hidden /> Find the trailer on YouTube
            </a>
          )}

          {m.tmdb_id && (
            <a
              className="inline-flex items-center gap-1.5 self-start text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
              href={`https://www.themoviedb.org/movie/${m.tmdb_id}`}
              target="_blank"
              rel="noreferrer"
            >
              Cast, plot and more on TMDB <ExternalLink className="size-3.5" aria-hidden />
            </a>
          )}
        </div>
      </div>

      <section className="flex flex-col gap-3">
        <div>
          <h2 className="text-lg font-semibold">More like this</h2>
          <p className="text-sm text-muted-foreground">Movies that the same kinds of viewers finished and loved.</p>
        </div>
        {sims === null ? (
          <MovieGrid>
            {Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="aspect-[2/3] rounded-xl" />)}
          </MovieGrid>
        ) : sims.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Not enough viewers have rated this one yet to find similar movies.
          </p>
        ) : (
          <MovieGrid>
            {sims.map((s) => (
              <LibraryCard key={s.movie_id} movie={s} showStars={false} />
            ))}
          </MovieGrid>
        )}
      </section>
    </div>
  );
}
