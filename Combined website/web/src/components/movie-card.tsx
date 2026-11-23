"use client";

import { Film } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import type { Movie } from "@/lib/api";

export function Poster({ movie, compact = false }: { movie: Movie; compact?: boolean }) {
  const [failed, setFailed] = useState(false);
  if (!movie.tmdb_id || failed) {
    return (
      <div className="flex aspect-[2/3] w-full flex-col items-center justify-center gap-2 bg-muted p-3 text-center">
        <Film className={compact ? "size-5 text-muted-foreground" : "size-8 text-muted-foreground"} aria-hidden />
        {!compact && <span className="line-clamp-3 text-sm font-medium">{movie.title}</span>}
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element -- poster comes from a cached redirect
    <img
      src={`/api/poster/${movie.tmdb_id}`}
      alt={`Poster for ${movie.title}`}
      loading="lazy"
      onError={() => setFailed(true)}
      className="aspect-[2/3] w-full bg-muted object-cover"
    />
  );
}

// Trailer keys are looked up once per film per page load.
const trailerCache = new Map<number, Promise<string | null>>();

export function trailerKey(tmdbId: number): Promise<string | null> {
  if (!trailerCache.has(tmdbId)) {
    trailerCache.set(
      tmdbId,
      fetch(`/api/trailer/${tmdbId}`)
        .then((r) => (r.ok ? r.json() : { key: null }))
        .then((d: { key: string | null }) => d.key)
        .catch(() => null),
    );
  }
  return trailerCache.get(tmdbId)!;
}

/** A muted ~30 second stretch of the trailer (skipping the studio logos at the start). */
export function trailerSrc(key: string, { muted = true, controls = false, clip = true } = {}) {
  const p = new URLSearchParams({
    autoplay: "1",
    mute: muted ? "1" : "0",
    controls: controls ? "1" : "0",
    playsinline: "1",
    rel: "0",
    modestbranding: "1",
    ...(clip ? { start: "15", end: "45" } : {}),
  });
  return `https://www.youtube-nocookie.com/embed/${key}?${p}`;
}

/** Poster that plays a short muted trailer clip while hovered (on devices with a mouse). */
function HoverPoster({ movie }: { movie: Movie }) {
  const [clip, setClip] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hovering = useRef(false);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  function enter() {
    if (!movie.tmdb_id || !window.matchMedia("(hover: hover)").matches) return;
    hovering.current = true;
    const id = movie.tmdb_id;
    // Short delay so sweeping the mouse across a grid doesn't start a dozen videos.
    timer.current = setTimeout(async () => {
      const key = await trailerKey(id);
      if (key && hovering.current) setClip(key);
    }, 700);
  }

  function leave() {
    hovering.current = false;
    if (timer.current) clearTimeout(timer.current);
    setClip(null);
  }

  return (
    <div className="relative" onMouseEnter={enter} onMouseLeave={leave}>
      <Poster movie={movie} />
      {clip && (
        <div className="absolute inset-0 flex items-center bg-black">
          <iframe
            src={trailerSrc(clip)}
            title={`Trailer clip for ${movie.title}`}
            allow="autoplay; encrypted-media"
            className="pointer-events-none aspect-video w-full"
          />
        </div>
      )}
    </div>
  );
}

export function MovieCard({
  movie,
  children,
  eyebrow,
}: {
  movie: Movie;
  children?: React.ReactNode;
  eyebrow?: React.ReactNode;
}) {
  return (
    <div className="group flex flex-col overflow-hidden rounded-xl border bg-card transition-colors hover:border-foreground/30">
      <Link href={`/movie/${movie.movie_id}`} className="flex flex-col focus-visible:outline-2 focus-visible:outline-ring">
        <HoverPoster movie={movie} />
        <div className="flex flex-col gap-0.5 px-3 pt-3">
          {eyebrow}
          <h3 className="line-clamp-2 font-medium leading-snug group-hover:underline">{movie.title}</h3>
          <p className="text-xs text-muted-foreground">
            {[movie.year, movie.genres.slice(0, 3).join(" · ")].filter(Boolean).join(" · ")}
          </p>
        </div>
      </Link>
      <div className="mt-auto flex flex-col gap-2 p-3">{children}</div>
    </div>
  );
}

export function MovieGrid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">{children}</div>;
}

/** One horizontal scrolling row of cards, like a "You might like" shelf. */
export function MovieShelf({ children }: { children: React.ReactNode }) {
  return (
    <div className="-mx-4 flex snap-x gap-4 overflow-x-auto px-4 pb-2 [&>*]:w-40 [&>*]:shrink-0 [&>*]:snap-start sm:[&>*]:w-44">
      {children}
    </div>
  );
}
