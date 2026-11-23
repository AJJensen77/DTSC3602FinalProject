"use client";

import { Star, X } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/utils";

type Props = {
  value: number | null;
  onChange: (value: number | null) => void;
  size?: "sm" | "md";
  label?: string;
};

/** Five stars, half-star precision: the left half of a star is x.5, the right half is x.0. */
export function StarRating({ value, onChange, size = "md", label = "Your rating" }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const shown = hover ?? value ?? 0;
  const px = size === "sm" ? "size-5" : "size-6";

  return (
    <div className="flex items-center gap-1" role="group" aria-label={label}>
      <div className="flex" onMouseLeave={() => setHover(null)}>
        {[1, 2, 3, 4, 5].map((i) => {
          const fill = shown >= i ? 100 : shown >= i - 0.5 ? 50 : 0;
          return (
            <span key={i} className={cn("relative inline-block", px)}>
              <Star className={cn("absolute inset-0 text-muted-foreground/40", px)} aria-hidden />
              <span className="absolute inset-0 overflow-hidden" style={{ width: `${fill}%` }}>
                <Star className={cn("fill-amber-400 text-amber-400", px)} aria-hidden />
              </span>
              {[i - 0.5, i].map((v, half) => (
                <button
                  key={v}
                  type="button"
                  aria-label={`${v} star${v === 1 ? "" : "s"}`}
                  aria-pressed={value === v}
                  className={cn(
                    "absolute inset-y-0 w-1/2 cursor-pointer rounded-sm focus-visible:outline-2 focus-visible:outline-ring",
                    half ? "right-0" : "left-0",
                  )}
                  onMouseEnter={() => setHover(v)}
                  onFocus={() => setHover(v)}
                  onBlur={() => setHover(null)}
                  onClick={() => onChange(v)}
                />
              ))}
            </span>
          );
        })}
      </div>
      <span className="w-7 text-sm tabular-nums text-muted-foreground">{shown ? shown.toFixed(1) : ""}</span>
      {value !== null && (
        <button
          type="button"
          onClick={() => onChange(null)}
          className="rounded-sm p-0.5 text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
          aria-label="Remove rating"
          title="Remove rating"
        >
          <X className="size-4" />
        </button>
      )}
    </div>
  );
}
