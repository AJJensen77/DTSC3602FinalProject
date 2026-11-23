"use client";

import { Loader2, Search } from "lucide-react";
import { useEffect, useState } from "react";

import { ListActions } from "@/components/list-actions";
import { genreName, MovieCard, MovieGrid } from "@/components/movie-card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type Movie } from "@/lib/api";
import { useLists } from "@/lib/use-lists";

const PAGE = 24;
const NEW = "__new__";

const fetchPage = (offset: number, t: string | null) =>
  t === NEW ? api.browse(offset, null, true, PAGE) : api.browse(offset, t, false, PAGE);

// Browsing state for one tab (null = all genres, NEW = new releases).
type Browse = { tab: string | null; movies: Movie[]; hasMore: boolean; more: boolean };

export default function BrowsePage() {
  const [genres, setGenres] = useState<string[]>([]);
  const [tab, setTab] = useState<string | null>(null);
  const [browse, setBrowse] = useState<Browse | null>(null);
  const [query, setQuery] = useState("");
  const [found, setFound] = useState<{ q: string; movies: Movie[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [showRated, setShowRated] = useState(false);
  const { entries, rated } = useLists();
  const q = query.trim();
  const active = q.length >= 2;
  const searching = active && found?.q !== q;
  const browsing = browse?.tab === tab ? browse : null;

  useEffect(() => {
    api.genres().then(setGenres).catch(() => setGenres([]));
  }, []);

  // First page whenever the tab changes (or on Try again).
  useEffect(() => {
    let cancelled = false;
    fetchPage(0, tab)
      .then((r) => {
        if (cancelled) return;
        setBrowse({ tab, movies: r.movies, hasMore: r.has_more, more: false });
        setError(null);
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [tab, attempt]);

  async function showMore() {
    if (!browsing) return;
    setBrowse({ ...browsing, more: true });
    try {
      const r = await fetchPage(browsing.movies.length, tab);
      const seen = new Set(browsing.movies.map((m) => m.movie_id));
      setBrowse({ tab, movies: [...browsing.movies, ...r.movies.filter((m) => !seen.has(m.movie_id))],
                  hasMore: r.has_more, more: false });
    } catch (e) {
      setBrowse({ ...browsing, more: false });
      setError((e as Error).message);
    }
  }

  // Debounced title search.
  useEffect(() => {
    if (!active) return;
    const t = setTimeout(() => {
      api.search(q)
        .then((movies) => {
          setFound({ q, movies });
          setError(null);
        })
        .catch((e) => {
          setFound({ q, movies: [] });
          setError(e.message);
        });
    }, 300);
    return () => clearTimeout(t);
  }, [q, active]);

  // Browsing hides films you've rated (you've seen them); search shows everything,
  // since you may be looking a film up to change its rating.
  const loaded = active ? (searching ? null : found!.movies) : (browsing?.movies ?? null);
  const hiding = !active && !showRated;
  const shown = hiding ? (loaded?.filter((m) => entries.get(m.movie_id)?.list !== "rated") ?? null) : loaded;
  const hiddenCount = loaded && shown ? loaded.length - shown.length : 0;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Browse movies</h1>
        <p className="text-sm text-muted-foreground">
          Rate the ones you&apos;ve seen, save the ones you want to watch. Click a poster for details and more like it.
        </p>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
        <Input
          className="pl-9"
          placeholder="Search by title, e.g. Dune or The Godfather"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search movies"
        />
      </div>

      {!active && (
        <div className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1 [scrollbar-width:none]" role="tablist" aria-label="Browse by genre">
          {[null, NEW, ...genres].map((g) => (
            <button
              key={g ?? "all"}
              type="button"
              role="tab"
              aria-selected={tab === g}
              onClick={() => setTab(g)}
              className={`${chip} ${tab === g ? chipOn : ""}`}
            >
              {g === null ? "All" : g === NEW ? "New releases" : genreName(g)}
            </button>
          ))}
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertDescription className="flex flex-col gap-3">
            <span>{error}</span>
            {!active && (
              <Button variant="outline" size="sm" className="self-start" onClick={() => setAttempt((a) => a + 1)}>
                Try again
              </Button>
            )}
          </AlertDescription>
        </Alert>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2">
      <h2 className="text-sm font-medium text-muted-foreground" aria-live="polite">
        {active
          ? searching ? "Searching…" : `Results for “${q}”`
          : tab === NEW ? "Released 2017 to today, most popular first"
            : tab ? `Well-known ${genreName(tab).toLowerCase()} movies` : "Well-known movies across genres"}
      </h2>
      {!active && rated.length > 0 && (
        <label className="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground">
          <input type="checkbox" className="size-4 accent-primary" checked={showRated}
                 onChange={(e) => setShowRated(e.target.checked)} />
          Show movies I&apos;ve rated{hiding && hiddenCount > 0 ? ` (${hiddenCount} hidden)` : ""}
        </label>
      )}
      </div>

      {shown === null ? (
        !error && (
          <MovieGrid>
            {Array.from({ length: 10 }, (_, i) => <Skeleton key={i} className="aspect-[2/3] w-full rounded-xl" />)}
          </MovieGrid>
        )
      ) : shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {active ? "No movies match that title. Try fewer words." : "You've rated everything shown so far. Load more below."}
        </p>
      ) : (
        <MovieGrid>
          {shown.map((m) => (
            <MovieCard key={m.movie_id} movie={m}>
              <ListActions movie={m} />
            </MovieCard>
          ))}
        </MovieGrid>
      )}

      {!active && browsing?.hasMore && (
        <Button variant="outline" className="self-center" onClick={showMore} disabled={browsing.more}>
          {browsing.more && <Loader2 className="animate-spin" aria-hidden />}
          Show more movies
        </Button>
      )}
    </div>
  );
}

const chip =
  "shrink-0 rounded-full border px-3 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring";
const chipOn = "border-primary bg-primary text-primary-foreground hover:text-primary-foreground";
