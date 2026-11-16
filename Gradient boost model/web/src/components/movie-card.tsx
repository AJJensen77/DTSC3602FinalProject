"use client";

import { Film } from "lucide-react";
import { useState } from "react";

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

export function MovieCard({ movie, children }: { movie: Movie; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col overflow-hidden rounded-xl border bg-card">
      <Poster movie={movie} />
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div>
          <h3 className="line-clamp-2 font-medium leading-snug">{movie.title}</h3>
          <p className="text-xs text-muted-foreground">
            {[movie.year, movie.genres.slice(0, 3).join(" · ")].filter(Boolean).join(" · ")}
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
