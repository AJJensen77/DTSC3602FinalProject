"use client";

import { ChevronLeft, ChevronRight, Film } from "lucide-react";
import Link from "next/link";
import { useRef, useState } from "react";

import type { Movie } from "@/lib/api";
import { cn } from "@/lib/utils";

/** TMDB id when MovieLens links one; otherwise the IMDb id (newer films). */
export function posterKey(movie: Movie): string | null {
  return movie.tmdb_id ? String(movie.tmdb_id) : movie.imdb_id;
}

export function Poster({ movie, compact = false }: { movie: Movie; compact?: boolean }) {
  const [failed, setFailed] = useState(false);
  const key = posterKey(movie);
  if (!key || failed) {
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
      src={`/api/poster/${key}`}
      alt={`Poster for ${movie.title}`}
      loading="lazy"
      onError={() => setFailed(true)}
      className="aspect-[2/3] w-full bg-muted object-cover"
    />
  );
}

export function MovieCard({ movie, children }: { movie: Movie; children?: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border bg-card">
      <Link href={`/movie/${movie.movie_id}`} className="group/card relative block focus-visible:outline-2 focus-visible:outline-ring"
            aria-label={`${movie.title}: details and more like this`}>
        <Poster movie={movie} />
        {movie.is_new && (
          <span className="absolute left-2 top-2 rounded-md bg-primary px-1.5 py-0.5 text-xs font-semibold text-primary-foreground shadow">
            New
          </span>
        )}
        {movie.imdb_rating !== null && (
          <span className="absolute right-2 top-2 rounded-md bg-background/85 px-1.5 py-0.5 text-xs font-medium tabular-nums shadow backdrop-blur">
            IMDb {movie.imdb_rating.toFixed(1)}
          </span>
        )}
      </Link>
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div>
          <Link href={`/movie/${movie.movie_id}`} className="hover:underline">
            <h3 className="line-clamp-2 font-medium leading-snug">{movie.title}</h3>
          </Link>
          <p className="text-xs text-muted-foreground">
            {[movie.year, movie.genres.slice(0, 3).map(genreName).join(" · ")].filter(Boolean).join(" · ")}
          </p>
        </div>
        <div className="mt-auto">{children}</div>
      </div>
    </div>
  );
}

export function MovieGrid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">{children}</div>;
}

/** A Netflix-style strip: swipe on phones, arrow buttons on wider screens. */
export function MovieRow({ label, children }: { label: string; children: React.ReactNode }) {
  const strip = useRef<HTMLDivElement>(null);
  const scroll = (dir: 1 | -1) =>
    strip.current?.scrollBy({ left: dir * strip.current.clientWidth * 0.9, behavior: "smooth" });

  return (
    <div className="group/row relative">
      <div
        ref={strip}
        role="list"
        aria-label={label}
        className="-mx-4 flex snap-x snap-mandatory scroll-px-4 gap-3 overflow-x-auto px-4 pb-2 [scrollbar-width:thin]"
      >
        {children}
      </div>
      {[-1, 1].map((dir) => (
        <button
          key={dir}
          type="button"
          onClick={() => scroll(dir as 1 | -1)}
          aria-label={dir < 0 ? `Scroll ${label} left` : `Scroll ${label} right`}
          className={cn(
            "absolute top-1/3 hidden size-10 -translate-y-1/2 items-center justify-center rounded-full border bg-background/90 shadow-md backdrop-blur transition-opacity hover:bg-muted focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-ring sm:flex sm:opacity-0 sm:group-hover/row:opacity-100",
            dir < 0 ? "-left-3" : "-right-3",
          )}
        >
          {dir < 0 ? <ChevronLeft className="size-5" aria-hidden /> : <ChevronRight className="size-5" aria-hidden />}
        </button>
      ))}
    </div>
  );
}

export function RowItem({ children }: { children: React.ReactNode }) {
  return (
    <div role="listitem" className="w-36 shrink-0 snap-start sm:w-44">
      {children}
    </div>
  );
}

/** MovieLens calls family films "Children". */
export const genreName = (g: string) => (g === "Children" ? "Kids & family" : g);
