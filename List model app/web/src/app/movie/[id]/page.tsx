"use client";

import { ArrowLeft, ExternalLink, Play, User, X } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ListActions } from "@/components/list-actions";
import { genreName, MovieCard, MovieRow, Poster, posterKey, RowItem } from "@/components/movie-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, fetchDetails, type Movie, type MovieDetails, type SimilarMovie } from "@/lib/api";
import { useLists } from "@/lib/use-lists";

// Fetch extra so there are still enough after hiding films already on your lists.
const SHOW = 12;
const FETCH = 30;

type State = { id: number; movie: Movie | null; similar: SimilarMovie[] | null; error: string | null };
/** undefined = still loading; null = TMDB has nothing for this film. */
type Details = { key: string; data: MovieDetails | null };

export default function MoviePage() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  const router = useRouter();
  const [state, setState] = useState<State | null>(null);
  const [details, setDetails] = useState<Details | null>(null);
  const [playing, setPlaying] = useState<number | null>(null);
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
  const movie = current?.movie ?? null;
  const key = movie ? posterKey(movie) : null;

  // Description, cast, trailer... from TMDB, once we know the film.
  useEffect(() => {
    if (!key) return;
    let cancelled = false;
    fetchDetails(key).then((data) => !cancelled && setDetails({ key, data }));
    return () => {
      cancelled = true;
    };
  }, [key]);

  const info = key && details?.key === key ? details.data : undefined;
  const infoLoading = !!key && details?.key !== key;
  const showTrailer = playing === id && !!info?.trailer;

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

  // Hide films already rated or on Want to Watch: this row is for discovering new ones.
  const similar = current?.similar?.filter((m) => !entries.has(m.movie_id)).slice(0, SHOW) ?? null;

  return (
    <div className="flex flex-col gap-8">
      {/* Backdrop banner, full width behind the header area of the page. */}
      <div className="relative -mx-4 -mt-8">
        <div className="relative h-56 overflow-hidden sm:h-80 md:h-96">
          {info?.backdrop ? (
            // eslint-disable-next-line @next/next/no-img-element -- TMDB image CDN
            <img src={info.backdrop} alt="" className="size-full object-cover object-top opacity-60" />
          ) : (
            <div className="size-full bg-gradient-to-br from-primary/20 to-transparent" />
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/40 to-transparent" />
          {/* Fade the left/right edges into the page on wide screens. */}
          <div className="absolute inset-0 hidden bg-gradient-to-r from-background via-transparent to-background md:block" />
          <div className="absolute left-4 top-4">{back}</div>
        </div>

        <div className="relative -mt-28 flex flex-col gap-6 px-4 sm:-mt-40 sm:flex-row">
          <div className="w-36 shrink-0 self-start overflow-hidden rounded-xl border shadow-2xl sm:w-52">
            {movie ? <Poster movie={movie} /> : <Skeleton className="aspect-[2/3] w-full" />}
          </div>

          <div className="flex min-w-0 flex-1 flex-col gap-4 sm:pt-28">
            {!movie ? (
              <>
                <Skeleton className="h-9 w-2/3" />
                <Skeleton className="h-4 w-1/3" />
              </>
            ) : (
              <>
                <div className="flex flex-col gap-2">
                  <h1 className="text-3xl font-semibold leading-tight sm:text-4xl">{movie.title}</h1>
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
                    {movie.year && <span>{movie.year}</span>}
                    {info?.runtime && <span>· {formatRuntime(info.runtime)}</span>}
                    {info?.certification && <Badge variant="outline">{info.certification}</Badge>}
                    {movie.imdb_rating !== null && (
                      <span>
                        · IMDb <span className="font-semibold text-foreground tabular-nums">{movie.imdb_rating.toFixed(1)}</span>
                      </span>
                    )}
                    {movie.is_new && <Badge>New release</Badge>}
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {movie.genres.map((g) => (
                      <Badge key={g} variant="secondary">{genreName(g)}</Badge>
                    ))}
                  </div>
                </div>

                {infoLoading ? (
                  <div className="flex max-w-2xl flex-col gap-2">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-5/6" />
                    <Skeleton className="h-4 w-2/3" />
                  </div>
                ) : (
                  info && (
                    <div className="flex max-w-2xl flex-col gap-3">
                      {info.tagline && <p className="italic text-muted-foreground">{info.tagline}</p>}
                      {info.overview && <p className="leading-relaxed">{info.overview}</p>}
                      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                        {info.directors.length > 0 && (
                          <>
                            <dt className="text-muted-foreground">Director{info.directors.length > 1 ? "s" : ""}</dt>
                            <dd>{info.directors.join(", ")}</dd>
                          </>
                        )}
                        {info.writers.length > 0 && (
                          <>
                            <dt className="text-muted-foreground">Writers</dt>
                            <dd>{info.writers.join(", ")}</dd>
                          </>
                        )}
                        {info.release_date && (
                          <>
                            <dt className="text-muted-foreground">Released</dt>
                            <dd>{formatDate(info.release_date)}</dd>
                          </>
                        )}
                      </dl>
                    </div>
                  )
                )}

                <div className="flex flex-wrap gap-2">
                  {info?.trailer && (
                    <Button onClick={() => setPlaying(showTrailer ? null : id)}>
                      {showTrailer ? <X aria-hidden /> : <Play aria-hidden />}
                      {showTrailer ? "Close trailer" : "Watch trailer"}
                    </Button>
                  )}
                  {movie.imdb_id && (
                    <Button variant="outline" nativeButton={false}
                            render={<a href={`https://www.imdb.com/title/${movie.imdb_id}/`} target="_blank" rel="noreferrer" />}>
                      <ExternalLink aria-hidden /> IMDb
                    </Button>
                  )}
                </div>

                <div className="max-w-xs rounded-xl border bg-card p-4">
                  <ListActions movie={movie} size="md" />
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {showTrailer && (
        <section aria-label="Trailer" className="overflow-hidden rounded-xl border bg-black">
          <iframe
            className="aspect-video w-full"
            src={`https://www.youtube-nocookie.com/embed/${info!.trailer}?autoplay=1&rel=0`}
            title={`${movie?.title} trailer`}
            allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
            allowFullScreen
          />
        </section>
      )}

      {info && info.cast.length > 0 && (
        <section className="flex flex-col gap-3">
          <h2 className="text-lg font-semibold">Top cast</h2>
          <MovieRow label="Top cast">
            {info.cast.map((c) => (
              <div key={`${c.name}-${c.character}`} role="listitem" className="flex w-24 shrink-0 snap-start flex-col items-center gap-2 text-center sm:w-28">
                <div className="flex size-20 items-center justify-center overflow-hidden rounded-full border bg-muted sm:size-24">
                  {c.photo ? (
                    // eslint-disable-next-line @next/next/no-img-element -- TMDB image CDN
                    <img src={c.photo} alt={c.name} loading="lazy" className="size-full object-cover" />
                  ) : (
                    <User className="size-8 text-muted-foreground" aria-hidden />
                  )}
                </div>
                <div>
                  <p className="line-clamp-2 text-sm font-medium leading-tight">{c.name}</p>
                  {c.character && <p className="line-clamp-2 text-xs text-muted-foreground">{c.character}</p>}
                </div>
              </div>
            ))}
          </MovieRow>
        </section>
      )}

      <section className="flex flex-col gap-3">
        <div>
          <h2 className="text-lg font-semibold">More like this</h2>
          <p className="text-xs text-muted-foreground">
            {movie?.is_new
              ? "We have no viewing data for this film yet, so these are matched by genres, director and cast."
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

function formatRuntime(min: number) {
  const h = Math.floor(min / 60);
  return h ? `${h}h ${min % 60}m` : `${min}m`;
}

function formatDate(iso: string) {
  const d = new Date(`${iso}T00:00:00`);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
}
