// Client for the combined recommender API on Modal (list taste model + gradient boost).
// Based on Yager's ReelPicks client; /recommend now also takes the watch lists.

export const API_URL = process.env.NEXT_PUBLIC_MOVIE_API_URL ?? "";

export type Movie = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  tmdb_id: number | null;
  imdb_rating: number | null;
};

export type MovieDetail = Movie & {
  imdb_votes: number | null;
  movielens_ratings: number;
  has_similar: boolean;
};

export type Recommendation = Movie & {
  predicted_rating: number | null;
  because: { movie_id: number; title: string } | null;
};

export type RecommendResponse = {
  recommendations: Recommendation[];
  n_ratings_used: number;
  n_list_items_used: number;
  used_gradient_boost: boolean;
  min_ratings_for_gradient_boost: number;
  genres: string[];
  note: string | null;
};

export type ListName = "finished" | "plan_to_watch";

export const LIST_LABELS: Record<ListName, string> = {
  finished: "Finished watching",
  plan_to_watch: "Plan to watch",
};

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

// The API scales to zero when idle; the first request can take ~20 s to wake it.
const TIMEOUT_MS = 60_000;

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, signal: AbortSignal.timeout(TIMEOUT_MS) });
  } catch (err) {
    const timedOut = err instanceof DOMException && err.name === "TimeoutError";
    throw new ApiError(
      timedOut
        ? "The recommendation service took too long to respond. Please try again."
        : "Couldn't reach the recommendation service. Check your connection and try again.",
      0,
    );
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : null;
    throw new ApiError(
      res.status === 503
        ? "The recommendation model is temporarily unavailable. Please try again shortly."
        : detail ?? `Request failed (${res.status}).`,
      res.status,
    );
  }
  return res.json() as Promise<T>;
}

export const api = {
  genres: () => call<string[]>("/genres"),
  starter: (n = 48) => call<Movie[]>(`/movies/starter?n=${n}`),
  search: (q: string) => call<Movie[]>(`/movies/search?q=${encodeURIComponent(q)}&limit=24`),
  lookup: (ids: number[]) =>
    ids.length ? call<Movie[]>(`/movies/lookup?ids=${ids.join(",")}`) : Promise.resolve([]),
  movie: (id: number) => call<MovieDetail>(`/movies/${id}`),
  similar: (id: number, k = 12) => call<Movie[]>(`/movies/${id}/similar?k=${k}`),
  recommend: (ratings: Map<number, number>, lists: Map<number, ListName>, genres: string[], k = 20) =>
    call<RecommendResponse>("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ratings: [...ratings].map(([movie_id, rating]) => ({ movie_id, rating })),
        finished: [...lists].filter(([, l]) => l === "finished").map(([id]) => id),
        plan_to_watch: [...lists].filter(([, l]) => l === "plan_to_watch").map(([id]) => id),
        genres,
        k,
      }),
    }),
};
