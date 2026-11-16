// Client for the movie recommender API on Modal. The site only relies on this
// contract (ratings in, movies out), so a different model can be swapped in behind it.

export const API_URL = process.env.NEXT_PUBLIC_MOVIE_API_URL ?? "";
export const MIN_RATINGS = 10;

export type Movie = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  tmdb_id: number | null;
  imdb_rating: number | null;
};

export type Recommendation = Movie & { predicted_rating: number };

export type RecommendResponse = {
  recommendations: Recommendation[];
  n_ratings_used: number;
  genres: string[];
  note: string | null;
};

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

// The API scales to zero when idle; the first request can take ~15 s to wake it.
const TIMEOUT_MS = 45_000;

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
  recommend: (ratings: Map<number, number>, genres: string[]) =>
    call<RecommendResponse>("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ratings: [...ratings].map(([movie_id, rating]) => ({ movie_id, rating })),
        genres,
        k: 10,
      }),
    }),
};
