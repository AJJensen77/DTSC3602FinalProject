"use client";

import { Bookmark, Sparkles, Star } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { PicksRows } from "@/components/picks-rows";
import { Button } from "@/components/ui/button";
import { posterKey } from "@/components/movie-card";
import { api, type ListsBody, type Movie } from "@/lib/api";
import { useSession } from "@/lib/use-lists";

const STEPS = [
  { icon: Bookmark, title: "Save what you want to watch", text: "Bookmark anything that catches your eye. It nudges your picks a little." },
  { icon: Star, title: "Rate what you've seen", text: "Stars on films you finished count most. Loved it, it was fine, or not for you." },
  { icon: Sparkles, title: "Get picks that fit your palate", text: "Personal from your very first rating, plus new releases from 2017 to today." },
];

const NOTHING: ListsBody = { ratings: [], want_to_watch: [] };

export default function Home() {
  const { session, loading } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (session) router.replace("/for-you");
  }, [session, router]);

  if (loading || session) return null;

  return (
    <div className="flex flex-col gap-12">
      <section className="relative isolate">
      <PosterCollage />
      <div className="mx-auto flex max-w-3xl flex-col items-center gap-8 py-6 text-center">
        <div className="flex flex-col gap-4">
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">Movies matched to your palate.</h1>
          <p className="text-lg text-muted-foreground">
            Start with what everyone&apos;s loving below, no account needed. Sign up to rate what you&apos;ve seen and save
            what you want to watch, and the picks become yours.
          </p>
        </div>
        <div className="flex gap-3">
          <Button size="lg" nativeButton={false} render={<Link href="/signup" />}>
            Create a free account
          </Button>
          <Button size="lg" variant="outline" nativeButton={false} render={<Link href="/browse" />}>
            Browse movies
          </Button>
        </div>
        <div className="grid w-full gap-4 sm:grid-cols-3">
          {STEPS.map(({ icon: Icon, title, text }) => (
            <div key={title} className="flex flex-col items-center gap-2 rounded-xl border bg-card p-5">
              <Icon className="size-6 text-primary" aria-hidden />
              <h2 className="font-medium">{title}</h2>
              <p className="text-sm text-muted-foreground">{text}</p>
            </div>
          ))}
        </div>
      </div>
      </section>

      <PicksRows
        lists={NOTHING}
        header={() => (
          <>
            <h2 className="text-2xl font-semibold">Popular right now</h2>
            <p className="text-sm text-muted-foreground">Well-liked films and new releases, for anyone just arriving.</p>
          </>
        )}
      />
    </div>
  );
}

/** Faint, blurred wall of well-known posters behind the headline (decoration only). */
function PosterCollage() {
  const [movies, setMovies] = useState<Movie[]>([]);
  useEffect(() => {
    api.starter(24).then(setMovies).catch(() => setMovies([]));
  }, []);
  const keys = movies.map(posterKey).filter(Boolean) as string[];
  if (!keys.length) return null;
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-x-0 -top-8 bottom-0 -z-10 overflow-hidden [mask-image:radial-gradient(ellipse_at_center,black_20%,transparent_75%)]"
    >
      <div className="grid -rotate-6 scale-110 grid-cols-6 gap-3 opacity-25 blur-[3px] sm:grid-cols-8">
        {keys.map((k) => (
          // eslint-disable-next-line @next/next/no-img-element -- decorative, cached poster redirect
          <img key={k} src={`/api/poster/${k}`} alt="" loading="lazy" className="aspect-[2/3] w-full rounded-md object-cover"
               onError={(e) => (e.currentTarget.style.visibility = "hidden")} />
        ))}
      </div>
    </div>
  );
}
