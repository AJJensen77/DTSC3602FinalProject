"use client";

import { Sparkles, Star, Tags } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { Button } from "@/components/ui/button";
import { MIN_RATINGS } from "@/lib/api";
import { useRatings, useSession } from "@/lib/use-ratings";

const STEPS = [
  { icon: Star, title: `Rate ${MIN_RATINGS}+ movies`, text: "Tap the stars on films you've already seen." },
  { icon: Sparkles, title: "Get ten picks", text: "Our model predicts what you'd rate 3.5★ or higher." },
  { icon: Tags, title: "Filter by genre", text: "In the mood for comedy or horror? Narrow it down." },
];

export default function Home() {
  const { session, loading } = useSession();
  const { ratings, loading: ratingsLoading } = useRatings();
  const router = useRouter();

  useEffect(() => {
    if (session && !ratingsLoading) router.replace(ratings.size >= MIN_RATINGS ? "/for-you" : "/rate");
  }, [session, ratingsLoading, ratings.size, router]);

  if (loading || session) return null;

  return (
    <div className="mx-auto flex max-w-3xl flex-col items-center gap-10 py-10 text-center">
      <div className="flex flex-col gap-4">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">Stop scrolling. Start watching.</h1>
        <p className="text-lg text-muted-foreground">
          Tell us a few movies you&apos;ve seen and how you felt about them. We&apos;ll suggest ten more you&apos;re
          likely to enjoy.
        </p>
      </div>
      <div className="flex gap-3">
        <Button size="lg" nativeButton={false} render={<Link href="/signup" />}>
          Create a free account
        </Button>
        <Button size="lg" variant="outline" nativeButton={false} render={<Link href="/login" />}>
          Log in
        </Button>
      </div>
      <div className="grid w-full gap-4 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, text }) => (
          <div key={title} className="flex flex-col items-center gap-2 rounded-xl border p-5">
            <Icon className="size-6" aria-hidden />
            <h2 className="font-medium">{title}</h2>
            <p className="text-sm text-muted-foreground">{text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
