"use client";

import { Bookmark, CheckCircle2, Sparkles } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { LibraryCard } from "@/components/library-card";
import { MovieShelf } from "@/components/movie-card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type Movie } from "@/lib/api";
import { useRatings, useSession } from "@/lib/use-ratings";
import { useRecommendations } from "@/lib/use-recs";

const STEPS = [
  { icon: CheckCircle2, title: "Track what you've watched", text: "Add movies to Finished watching and give them a score." },
  { icon: Bookmark, title: "Keep a Plan to watch list", text: "Save anything that catches your eye for later." },
  { icon: Sparkles, title: "Get recommendations", text: "From your very first finished movie, we suggest what you'll like next." },
];

function Shelf({ title, href, movies, loading }: { title: string; href?: string; movies: Movie[] | null; loading?: boolean }) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-lg font-semibold">{title}</h2>
        {href && (
          <Link href={href} className="text-sm text-muted-foreground hover:text-foreground">
            See all →
          </Link>
        )}
      </div>
      <MovieShelf>
        {movies === null || loading
          ? Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="aspect-[2/3] rounded-xl" />)
          : movies.map((m) => <LibraryCard key={m.movie_id} movie={m} />)}
      </MovieShelf>
    </section>
  );
}

function Dashboard() {
  const { lists, ratings } = useRatings();
  const recs = useRecommendations([], { k: 20 });
  const [because, setBecause] = useState<{ seed: Movie; similar: Movie[] } | null>(null);
  const [plan, setPlan] = useState<Movie[] | null>(null);

  const finishedIds = useMemo(
    () => [...lists].filter(([, l]) => l === "finished").map(([id]) => id),
    [lists],
  );
  const planIds = useMemo(
    () => [...lists].filter(([, l]) => l === "plan_to_watch").map(([id]) => id),
    [lists],
  );
  // "Because you finished X": the most recently added film the person rated well (or any finished one).
  const seedId = useMemo(() => {
    const liked = finishedIds.filter((id) => (ratings.get(id) ?? 4) >= 3.5);
    return liked.at(-1) ?? null;
  }, [finishedIds, ratings]);

  useEffect(() => {
    if (seedId === null) return;
    Promise.all([api.movie(seedId), api.similar(seedId, 12)])
      .then(([seed, similar]) => setBecause({ seed, similar }))
      .catch(() => setBecause(null));
  }, [seedId]);

  useEffect(() => {
    api.lookup(planIds.slice(-20).reverse()).then(setPlan).catch(() => setPlan([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the ids string
  }, [planIds.join(",")]);

  const empty = lists.size === 0 && ratings.size === 0;

  return (
    <div className="flex flex-col gap-10">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Welcome back</h1>
        <p className="text-sm text-muted-foreground">
          {finishedIds.length} finished · {planIds.length} planned · {ratings.size} rated
        </p>
      </div>

      {empty && (
        <Alert>
          <AlertDescription className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <span>Add a movie you&apos;ve finished to start getting recommendations. Here are some popular ones meanwhile.</span>
            <Button nativeButton={false} render={<Link href="/browse" />}>
              Find movies
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {recs.error ? (
        <Alert variant="destructive">
          <AlertDescription>{recs.error}</AlertDescription>
        </Alert>
      ) : (
        <Shelf
          title={empty ? "Popular, well-liked movies" : "You might be interested in"}
          href="/for-you"
          movies={recs.data?.recommendations ?? null}
          loading={recs.loading}
        />
      )}
      {recs.slow && (
        <p className="-mt-8 text-sm text-muted-foreground" aria-live="polite">
          Waking up the recommendation model — the first request can take about 20 seconds…
        </p>
      )}

      {because && because.similar.length > 0 && (
        <Shelf title={`Because you finished ${because.seed.title}`} href={`/movie/${because.seed.movie_id}`} movies={because.similar} />
      )}

      {plan && plan.length > 0 && <Shelf title="On your Plan to watch list" href="/list?tab=plan_to_watch" movies={plan} />}
    </div>
  );
}

function Landing() {
  const [popular, setPopular] = useState<Movie[] | null>(null);
  useEffect(() => {
    api.starter(18).then(setPopular).catch(() => setPopular([]));
  }, []);

  return (
    <div className="flex flex-col gap-12">
      <div className="mx-auto flex max-w-3xl flex-col items-center gap-6 py-8 text-center">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">Your movie list, with picks that fit you.</h1>
        <p className="text-lg text-muted-foreground">
          Keep track of what you&apos;ve finished and what you plan to watch. Score what you&apos;ve seen, and we&apos;ll
          suggest movies you&apos;re likely to enjoy next.
        </p>
        <div className="flex gap-3">
          <Button size="lg" nativeButton={false} render={<Link href="/signup" />}>
            Create a free account
          </Button>
          <Button size="lg" variant="outline" nativeButton={false} render={<Link href="/browse" />}>
            Browse movies
          </Button>
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, text }) => (
          <div key={title} className="flex flex-col items-center gap-2 rounded-xl border p-5 text-center">
            <Icon className="size-6" aria-hidden />
            <h2 className="font-medium">{title}</h2>
            <p className="text-sm text-muted-foreground">{text}</p>
          </div>
        ))}
      </div>
      <Shelf title="Popular across genres" href="/browse" movies={popular} />
    </div>
  );
}

export default function Home() {
  const { session, loading } = useSession();
  if (loading) return null;
  return session ? <Dashboard /> : <Landing />;
}
