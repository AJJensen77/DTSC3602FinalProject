"use client";

import { Bookmark, BookmarkCheck } from "lucide-react";
import Link from "next/link";

import { StarRating } from "@/components/star-rating";
import { Button } from "@/components/ui/button";
import type { Movie } from "@/lib/api";
import { useLists } from "@/lib/use-lists";
import { cn } from "@/lib/utils";

/** Want to Watch toggle + "Seen it? Rate it" stars for one film. */
export function ListActions({ movie, size = "sm" }: { movie: Movie; size?: "sm" | "md" }) {
  const { entries, rate, toggleWant, signedIn } = useLists();
  const entry = entries.get(movie.movie_id);
  const wanted = entry?.list === "want";
  const rated = entry?.list === "rated";

  if (!signedIn) {
    return (
      <Link href="/signup" className="text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline">
        Sign up to rate or save this
      </Link>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {!rated && (
        <Button
          type="button"
          size="sm"
          variant={wanted ? "default" : "outline"}
          aria-pressed={wanted}
          onClick={() => toggleWant(movie.movie_id)}
          className={cn("justify-start", size === "sm" && "h-7 px-2 text-xs")}
        >
          {wanted ? <BookmarkCheck aria-hidden /> : <Bookmark aria-hidden />}
          {wanted ? "On Want to Watch" : "Want to Watch"}
        </Button>
      )}
      <div>
        <p className="text-xs text-muted-foreground">
          {rated ? "Your rating" : wanted ? "Watched it? Rate it:" : "Seen it? Rate it:"}
        </p>
        <StarRating
          size={size}
          label={`Your rating for ${movie.title}`}
          value={entry?.stars ?? null}
          onChange={(v) => rate(movie.movie_id, v)}
        />
      </div>
    </div>
  );
}
