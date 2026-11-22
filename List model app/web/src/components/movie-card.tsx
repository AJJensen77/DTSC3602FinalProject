"use client";

import { ChevronLeft, ChevronRight, Film } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { fetchDetails, type Movie } from "@/lib/api";
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

const HOVER_DELAY_MS = 700;
// YouTube flashes its pause icon for about a second as playback starts; stay hidden
// until the player reports "playing", then wait this long before fading in.
const REVEAL_AFTER_PLAYING_MS = 1200;
const REVEAL_FALLBACK_MS = 4000; // in case the player never reports its state

/**
 * Plays a muted trailer over the poster after the mouse rests on it. Only on
 * devices with a real mouse (not phones/tablets) and when the system isn't set to
 * reduce motion. It stops as soon as the mouse leaves.
 */
function useHoverTrailer(movie: Movie) {
  const [trailer, setTrailer] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const frame = useRef<HTMLIFrameElement>(null);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const hovering = useRef(false);

  const clearTimers = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };
  const stop = () => {
    hovering.current = false;
    clearTimers();
    setTrailer(null);
    setVisible(false);
  };
  // On unmount: stop any pending timers (the iframe goes away with the component).
  useEffect(() => () => {
    hovering.current = false;
    timers.current.forEach(clearTimeout);
  }, []);

  // Listen for the YouTube player's state messages (IFrame API via postMessage).
  useEffect(() => {
    if (!trailer) return;
    let revealed = false;
    const reveal = (ms: number) => {
      if (revealed) return;
      revealed = true;
      timers.current.push(setTimeout(() => hovering.current && setVisible(true), ms));
    };
    const onMessage = (e: MessageEvent) => {
      if (e.source !== frame.current?.contentWindow || typeof e.data !== "string") return;
      try {
        const msg = JSON.parse(e.data);
        if (msg.event === "infoDelivery" && msg.info?.playerState === 1) reveal(REVEAL_AFTER_PLAYING_MS);
        if (msg.event === "onStateChange" && msg.info === 1) reveal(REVEAL_AFTER_PLAYING_MS);
      } catch {
        // not a YouTube message
      }
    };
    window.addEventListener("message", onMessage);
    timers.current.push(setTimeout(() => reveal(0), REVEAL_FALLBACK_MS));
    return () => window.removeEventListener("message", onMessage);
  }, [trailer]);

  const start = (e: React.PointerEvent) => {
    if (e.pointerType !== "mouse") return;
    if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const key = posterKey(movie);
    if (!key) return;
    hovering.current = true;
    timers.current.push(setTimeout(async () => {
      const details = await fetchDetails(key);
      if (hovering.current && details?.trailer) setTrailer(details.trailer);
    }, HOVER_DELAY_MS));
  };

  /** Ask the player to start sending state updates once it has loaded. */
  const onLoad = () =>
    frame.current?.contentWindow?.postMessage(JSON.stringify({ event: "listening", id: 1, channel: "widget" }), "*");

  return { trailer, visible, frame, onLoad, handlers: { onPointerEnter: start, onPointerLeave: stop } };
}

export function MovieCard({ movie, children }: { movie: Movie; children?: React.ReactNode }) {
  const { trailer, visible, frame, onLoad, handlers } = useHoverTrailer(movie);
  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border bg-card">
      <Link href={`/movie/${movie.movie_id}`} className="group/card relative block overflow-hidden focus-visible:outline-2 focus-visible:outline-ring"
            aria-label={`${movie.title}: details and more like this`} {...handlers}>
        <Poster movie={movie} />
        {trailer && (
          // A 16:9 trailer filling a 2:3 poster, zoomed 1.35x so YouTube's title bar
          // (top) and "More videos" panel (bottom) fall outside the poster: 135% tall,
          // 2.67 x 1.35 = 360% wide, centred both ways.
          <iframe
            aria-hidden
            tabIndex={-1}
            title={`${movie.title} trailer preview`}
            ref={frame}
            src={`https://www.youtube-nocookie.com/embed/${trailer}?enablejsapi=1&autoplay=1&mute=1&controls=0&playsinline=1&loop=1&playlist=${trailer}&rel=0&modestbranding=1&iv_load_policy=3&disablekb=1`}
            allow="autoplay; encrypted-media"
            onLoad={onLoad}
            className={cn(
              "pointer-events-none absolute left-1/2 top-1/2 h-[135%] w-[360%] max-w-none -translate-x-1/2 -translate-y-1/2 border-0 transition-opacity duration-500",
              visible ? "opacity-100" : "opacity-0",
            )}
          />
        )}
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
