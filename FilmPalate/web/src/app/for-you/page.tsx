"use client";

import { Sparkles } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";

import { PicksRows } from "@/components/picks-rows";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { useLists, useRequireSession } from "@/lib/use-lists";

export default function ForYouPage() {
  const { session } = useRequireSession();
  const { loading, error, rated, want, body } = useLists();
  // Recomputed only when the lists change; PicksRows decides when to refetch.
  const lists = useMemo(() => body(), [body]);

  if (!session || loading) return null;

  return (
    <div className="flex flex-col gap-6">
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      <PicksRows
        lists={lists}
        header={(data) => {
          const personal = data?.personalized ?? false;
          return (
            <>
              <h1 className="text-2xl font-semibold">{data && !personal ? "Popular picks" : "Your picks"}</h1>
              <p className="text-sm text-muted-foreground">
                {data && personal
                  ? `Based on ${describe(rated.length, want.length)}.`
                  : "Movies lots of people loved. Rate one you enjoyed or save one you want to watch, and these become yours."}
              </p>
            </>
          );
        }}
      />
      {rated.length + want.length === 0 && (
        <Alert>
          <Sparkles aria-hidden />
          <AlertTitle>Make these picks yours</AlertTitle>
          <AlertDescription className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <span>Seen something above? Tap its stars. Interested? Hit Want to Watch. Then press Update.</span>
            <Button size="sm" variant="outline" className="self-start" nativeButton={false} render={<Link href="/browse" />}>
              Browse movies
            </Button>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}

function describe(nRated: number, nWant: number) {
  const parts = [];
  if (nRated) parts.push(`your ${nRated} rating${nRated === 1 ? "" : "s"}`);
  if (nWant) parts.push(`${nWant} film${nWant === 1 ? "" : "s"} you want to watch`);
  return parts.join(" and ") || "your lists";
}
