"use client";

import { RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { ListActions } from "@/components/list-actions";
import { genreName, MovieCard, MovieRow, Poster, RowItem } from "@/components/movie-card";
import Link from "next/link";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type ListsBody, type RowsResponse } from "@/lib/api";
import { listsSignature } from "@/lib/use-lists";
import { cn } from "@/lib/utils";

type Result = { key: number; sig: string; data: RowsResponse | null; error: string | null };

/**
 * Netflix-style rows from POST /recommend/rows. ``lists`` is read when fetching;
 * when it changes afterwards the Refresh button offers to update the picks
 * (so rows don't jump around while someone is rating).
 */
export function PicksRows({ lists, header }: { lists: ListsBody; header: (r: RowsResponse | null) => React.ReactNode }) {
  const [nonce, setNonce] = useState(0);
  const [result, setResult] = useState<Result | null>(null);
  const [slowKey, setSlowKey] = useState<number | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const listsRef = useRef(lists);
  useEffect(() => {
    listsRef.current = lists;
  }, [lists]);

  const loading = result?.key !== nonce;
  const slow = loading && slowKey === nonce;
  const data = loading ? null : (result?.data ?? null);
  const error = loading ? null : (result?.error ?? null);
  const stale = !loading && result?.sig !== listsSignature(lists);

  useEffect(() => {
    let cancelled = false;
    const key = nonce;
    const sent = listsRef.current;
    const sig = listsSignature(sent);
    const slowTimer = setTimeout(() => setSlowKey(key), 3000);
    api
      .rows(sent, key)
      .then((d) => !cancelled && setResult({ key, sig, data: d, error: null }))
      .catch((e: Error) => !cancelled && setResult({ key, sig, data: null, error: e.message }))
      .finally(() => clearTimeout(slowTimer));
    return () => {
      cancelled = true;
      clearTimeout(slowTimer);
    };
  }, [nonce]);

  // Highlight the chip of the row currently at the top of the screen.
  useEffect(() => {
    if (!data) return;
    const sections = data.rows.map((r) => document.getElementById(rowId(r.key))).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver(
      (entries) => {
        const top = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (top) setActive(top.target.id);
      },
      { rootMargin: "-130px 0px -55% 0px" },
    );
    sections.forEach((s) => io.observe(s));
    return () => io.disconnect();
  }, [data]);

  const refresh = () => setNonce((n) => n + 1);
  const personal = data?.personalized ?? false;

  function jump(key: string) {
    setActive(rowId(key));
    document.getElementById(rowId(key))?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">{header(data)}</div>
        <Button variant={stale ? "default" : "ghost"} size="sm" onClick={refresh} disabled={loading}>
          <RefreshCw className={cn(loading && "animate-spin")} aria-hidden />
          {stale ? "Update my picks" : "Refresh"}
        </Button>
      </div>

      {data && data.rows.length > 0 && (
        <nav
          aria-label="Jump to a row"
          className="sticky top-14 z-10 -mx-4 flex gap-2 overflow-x-auto border-b bg-background/90 px-4 py-2 backdrop-blur [scrollbar-width:none]"
        >
          {data.rows.map((r) => (
            <button
              key={r.key}
              type="button"
              onClick={() => jump(r.key)}
              aria-current={active === rowId(r.key) ? "true" : undefined}
              className={cn(chip, active === rowId(r.key) && chipOn)}
            >
              {chipLabel(r.key, r.title)}
            </button>
          ))}
        </nav>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Couldn&apos;t load picks</AlertTitle>
          <AlertDescription className="flex flex-col gap-3">
            <span>{error}</span>
            <Button variant="outline" size="sm" className="self-start" onClick={refresh}>
              Try again
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {loading ? (
        <div className="flex flex-col gap-8">
          <p className="text-sm text-muted-foreground" aria-live="polite">
            {slow ? "Waking up the recommendation model. The first visit can take about 20 seconds…" : "Finding picks…"}
          </p>
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex flex-col gap-3">
              <Skeleton className="h-6 w-40" />
              <div className="flex gap-3 overflow-hidden">
                {Array.from({ length: 7 }, (_, j) => (
                  <Skeleton key={j} className="aspect-[2/3] w-36 shrink-0 rounded-xl sm:w-44" />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        data && (
          <div className="flex flex-col gap-8">
            {data.rows.map((r) => (
              <section key={r.key} id={rowId(r.key)} aria-labelledby={`${rowId(r.key)}-h`} className="scroll-mt-28">
                <div className="flex items-center gap-3">
                  {r.seed && (
                    <Link href={`/movie/${r.seed.movie_id}`} className="w-8 shrink-0 overflow-hidden rounded"
                          aria-label={`${r.seed.title} details`}>
                      <Poster movie={r.seed} compact />
                    </Link>
                  )}
                  <div>
                    <h2 id={`${rowId(r.key)}-h`} className="text-lg font-semibold">
                      {rowTitle(r.key, r.title, personal)}
                    </h2>
                    {r.note && <p className="text-xs text-muted-foreground">{r.note}</p>}
                  </div>
                </div>
                {r.key === "new" && (
                  <p className="mb-3 text-xs text-muted-foreground">
                    Popular, well-reviewed films from 2017 on. Our model has no viewing data for these yet, so this row is
                    the same for everyone.
                  </p>
                )}
                <div className={r.key === "new" ? "" : "mt-3"}>
                  <MovieRow label={r.title}>
                    {r.movies.map((m) => (
                      <RowItem key={m.movie_id}>
                        <MovieCard movie={m}>
                          <ListActions movie={m} />
                        </MovieCard>
                      </RowItem>
                    ))}
                  </MovieRow>
                </div>
              </section>
            ))}
          </div>
        )
      )}
    </div>
  );
}

const rowId = (key: string) => `row-${key.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
const chipLabel = (key: string, title: string) =>
  key === "top" ? "Top picks" : key === "new" ? "New releases" : key.startsWith("because-") ? "Because you liked"
    : key.startsWith("want-") ? "Because you want" : genreName(title);
const rowTitle = (key: string, title: string, personal: boolean) =>
  // (Not key.includes("-"): genres like "Sci-Fi" and "Film-Noir" contain hyphens.)
  key === "top" || key === "new" || key.startsWith("because-") || key.startsWith("want-") ? title
    : personal ? `${genreName(title)} for you` : `Popular in ${genreName(title)}`;

const chip =
  "shrink-0 rounded-full border px-3 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring";
const chipOn = "border-primary bg-primary text-primary-foreground hover:text-primary-foreground";
