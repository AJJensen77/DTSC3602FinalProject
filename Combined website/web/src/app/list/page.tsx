"use client";

import { Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";

import { Poster } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, LIST_LABELS, type ListName, type Movie } from "@/lib/api";
import { useRatings, useRequireSession } from "@/lib/use-ratings";
import { cn } from "@/lib/utils";

type Tab = "all" | ListName;
type Sort = "added" | "score" | "title";

const TABS: { id: Tab; label: string }[] = [
  { id: "all", label: "All movies" },
  { id: "finished", label: LIST_LABELS.finished },
  { id: "plan_to_watch", label: LIST_LABELS.plan_to_watch },
];

const STATUS_BAR: Record<ListName, string> = {
  finished: "bg-sky-500",
  plan_to_watch: "bg-zinc-500",
};

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col rounded-xl border px-4 py-3">
      <span className="text-2xl font-semibold tabular-nums">{value}</span>
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

function ListView() {
  const { session } = useRequireSession();
  const { ratings, lists, loading, error: loadError, setRating, setList } = useRatings();
  const params = useSearchParams();
  const router = useRouter();
  const tab = (TABS.some((t) => t.id === params.get("tab")) ? params.get("tab") : "all") as Tab;
  const [sort, setSort] = useState<Sort>("added");
  const [movies, setMovies] = useState<Map<number, Movie>>(new Map());
  const [error, setError] = useState<string | null>(null);

  // Everything on a list, plus anything rated but somehow off-list (counted as finished).
  const entries = useMemo(() => {
    const m = new Map(lists);
    for (const id of ratings.keys()) if (!m.has(id)) m.set(id, "finished");
    return [...m].map(([id, list], order) => ({ id, list, order }));
  }, [lists, ratings]);

  const missing = entries.map((e) => e.id).filter((id) => !movies.has(id));
  useEffect(() => {
    if (!missing.length) return;
    api
      .lookup(missing.slice(0, 500))
      .then((found) => setMovies((m) => new Map([...m, ...found.map((f) => [f.movie_id, f] as const)])))
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the ids string
  }, [missing.join(",")]);

  if (!session || loading) return null;

  const finished = entries.filter((e) => e.list === "finished");
  const scores = [...ratings.values()];
  const mean = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2) : "—";

  const rows = entries
    .filter((e) => tab === "all" || e.list === tab)
    .map((e) => ({ ...e, movie: movies.get(e.id), rating: ratings.get(e.id) ?? null }))
    .sort((a, b) =>
      sort === "score"
        ? (b.rating ?? -1) - (a.rating ?? -1)
        : sort === "title"
          ? (a.movie?.title ?? "").localeCompare(b.movie?.title ?? "")
          : b.order - a.order,
    );

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">My list</h1>
        <p className="text-sm text-muted-foreground">
          {session.user.email} · Your Finished watching list drives your recommendations.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Finished watching" value={finished.length} />
        <Stat label="Plan to watch" value={entries.length - finished.length} />
        <Stat label="Movies scored" value={scores.length} />
        <Stat label="Mean score" value={mean} />
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex gap-1 overflow-x-auto rounded-lg bg-muted p-1" role="tablist">
          {TABS.map((t) => {
            const n = t.id === "all" ? entries.length : entries.filter((e) => e.list === t.id).length;
            return (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={tab === t.id}
                onClick={() => router.replace(t.id === "all" ? "/list" : `/list?tab=${t.id}`, { scroll: false })}
                className={cn(
                  "whitespace-nowrap rounded-md px-3 py-1.5 text-sm text-muted-foreground",
                  tab === t.id && "bg-background font-medium text-foreground shadow-sm",
                )}
              >
                {t.label} <span className="tabular-nums text-xs">({n})</span>
              </button>
            );
          })}
        </div>
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          Sort by
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as Sort)}
            className="rounded-md border bg-background px-2 py-1 text-foreground"
          >
            <option value="added">Recently added</option>
            <option value="score">Your score</option>
            <option value="title">Title</option>
          </select>
        </label>
      </div>

      {(error || loadError) && (
        <Alert variant="destructive">
          <AlertDescription>{error ?? loadError}</AlertDescription>
        </Alert>
      )}

      {rows.length === 0 ? (
        <Alert>
          <AlertDescription className="flex flex-col gap-3">
            <span>
              {tab === "plan_to_watch"
                ? "Nothing on Plan to watch yet. Save movies you want to see from any movie page."
                : "No movies here yet. Find something you've watched and mark it Finished."}
            </span>
            <Button className="self-start" nativeButton={false} render={<Link href="/browse" />}>
              Browse movies
            </Button>
          </AlertDescription>
        </Alert>
      ) : (
        <div className="overflow-hidden rounded-xl border">
          <div className="hidden grid-cols-[2.5rem_3rem_1fr_11rem_10rem_2.5rem] items-center gap-3 border-b bg-muted/50 px-3 py-2 text-xs font-medium text-muted-foreground sm:grid">
            <span>#</span>
            <span />
            <span>Title</span>
            <span>Score</span>
            <span>Status</span>
            <span />
          </div>
          <ul className="divide-y">
            {rows.map(({ id, list, movie, rating }, i) => (
              <li
                key={id}
                className="relative grid grid-cols-[3rem_1fr_2.5rem] items-center gap-3 px-3 py-2.5 sm:grid-cols-[2.5rem_3rem_1fr_11rem_10rem_2.5rem]"
              >
                <span className={cn("absolute inset-y-0 left-0 w-1", STATUS_BAR[list])} aria-hidden />
                <span className="hidden text-sm tabular-nums text-muted-foreground sm:block">{i + 1}</span>
                <Link href={`/movie/${id}`} className="w-12 overflow-hidden rounded-md">
                  {movie ? <Poster movie={movie} compact /> : <Skeleton className="aspect-[2/3] w-full" />}
                </Link>
                <div className="min-w-0">
                  <Link href={`/movie/${id}`} className="block truncate font-medium hover:underline">
                    {movie?.title ?? "Loading…"}
                  </Link>
                  <p className="truncate text-xs text-muted-foreground">
                    {[movie?.year, movie?.genres.slice(0, 3).join(" · ")].filter(Boolean).join(" · ")}
                  </p>
                  {/* On phones, score and status sit under the title. */}
                  <div className="mt-2 flex flex-wrap items-center gap-3 sm:hidden">
                    <StarRating size="sm" value={rating} onChange={(v) => setRating(id, v)}
                                label={`Your rating for ${movie?.title ?? "this movie"}`} />
                    <StatusSelect value={list} onChange={(l) => setList(id, l)} />
                  </div>
                </div>
                <div className="hidden sm:block">
                  <StarRating size="sm" value={rating} onChange={(v) => setRating(id, v)}
                              label={`Your rating for ${movie?.title ?? "this movie"}`} />
                </div>
                <div className="hidden sm:block">
                  <StatusSelect value={list} onChange={(l) => setList(id, l)} />
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setList(id, null);
                    if (rating !== null) setRating(id, null);
                  }}
                  className="justify-self-end rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                  aria-label={`Remove ${movie?.title ?? "this movie"} from your list`}
                  title="Remove from list"
                >
                  <Trash2 className="size-4" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function StatusSelect({ value, onChange }: { value: ListName; onChange: (l: ListName) => void }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as ListName)}
      aria-label="Status"
      className="rounded-md border bg-background px-2 py-1 text-sm"
    >
      <option value="finished">{LIST_LABELS.finished}</option>
      <option value="plan_to_watch">{LIST_LABELS.plan_to_watch}</option>
    </select>
  );
}

export default function ListPage() {
  return (
    <Suspense fallback={null}>
      <ListView />
    </Suspense>
  );
}
