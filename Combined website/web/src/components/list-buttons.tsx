"use client";

import { Bookmark, CheckCircle2 } from "lucide-react";

import { LIST_LABELS, type ListName } from "@/lib/api";
import { useRatings } from "@/lib/use-ratings";
import { cn } from "@/lib/utils";

const OPTIONS: { list: ListName; icon: typeof Bookmark; short: string }[] = [
  { list: "finished", icon: CheckCircle2, short: "Finished" },
  { list: "plan_to_watch", icon: Bookmark, short: "Plan to watch" },
];

/** Two toggles: put a movie on Finished watching or Plan to watch (clicking again removes it). */
export function ListButtons({ movieId, title, size = "sm" }: { movieId: number; title: string; size?: "sm" | "md" }) {
  const { lists, setList } = useRatings();
  const current = lists.get(movieId) ?? null;

  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label={`Lists for ${title}`}>
      {OPTIONS.map(({ list, icon: Icon, short }) => {
        const on = current === list;
        return (
          <button
            key={list}
            type="button"
            aria-pressed={on}
            title={on ? `Remove from ${LIST_LABELS[list]}` : `Add to ${LIST_LABELS[list]}`}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setList(movieId, on ? null : list);
            }}
            className={cn(
              "inline-flex items-center gap-1 rounded-full border px-2.5 py-1 font-medium transition-colors",
              "focus-visible:outline-2 focus-visible:outline-ring",
              size === "sm" ? "text-xs" : "text-sm",
              on
                ? "border-amber-400 bg-amber-400/15 text-amber-300"
                : "text-muted-foreground hover:border-foreground/40 hover:text-foreground",
            )}
          >
            <Icon className={cn(size === "sm" ? "size-3.5" : "size-4", on && "fill-amber-400/30")} aria-hidden />
            {short}
          </button>
        );
      })}
    </div>
  );
}
