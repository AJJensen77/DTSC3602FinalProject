"use client";

import { ListButtons } from "@/components/list-buttons";
import { MovieCard } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Badge } from "@/components/ui/badge";
import type { Movie, Recommendation } from "@/lib/api";
import { useRatings } from "@/lib/use-ratings";

/** A movie card with the person's list buttons and stars; shows why it was recommended. */
export function LibraryCard({ movie, showStars = true }: { movie: Movie | Recommendation; showStars?: boolean }) {
  const { ratings, setRating } = useRatings();
  const rec = "because" in movie ? movie : null;

  return (
    <MovieCard
      movie={movie}
      eyebrow={
        rec?.because ? (
          <p className="line-clamp-1 text-[11px] text-amber-300/90">Because of {rec.because.title} on your list</p>
        ) : undefined
      }
    >
      {rec?.predicted_rating != null && (
        <Badge variant="secondary" className="self-start">
          Predicted for you: {rec.predicted_rating.toFixed(1)}★
        </Badge>
      )}
      <ListButtons movieId={movie.movie_id} title={movie.title} />
      {showStars && (
        <StarRating
          size="sm"
          label={`Your rating for ${movie.title}`}
          value={ratings.get(movie.movie_id) ?? null}
          onChange={(v) => setRating(movie.movie_id, v)}
        />
      )}
    </MovieCard>
  );
}
