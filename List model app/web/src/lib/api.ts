// Client for the FilmPalate API on Modal. The site only relies on this contract
// (My Ratings + Want to Watch in, movies out).

export const API_URL = process.env.NEXT_PUBLIC_MOVIE_API_URL ?? "";

export type Movie = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  tmdb_id: number | null;
  imdb_id: string | null;
  imdb_rating: number | null;
  /** Released 2017+ and added from IMDb (the model has no MovieLens ratings for it). */
  is_new: boolean;
};

export type SimilarMovie = Movie & { similarity: number };

/** ``seed`` is the film a "Because you…" row starts from; ``note`` explains weaker matches. */
export type Row = { key: string; title: string; movies: Movie[]; seed: Movie | null; note: string | null };
export type RowsResponse = { personalized: boolean; rows: Row[] };
export type BrowseResponse = { movies: Movie[]; has_more: boolean };

/** What the API needs from a user's lists, oldest first. */
export type ListsBody = {
  ratings: { movie_id: number; stars: number }[];
  want_to_watch: number[];
};

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

// The API scales to zero when idle; the first request can take ~20 s to wake it.
const TIMEOUT_MS = 45_000;

function describe(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  // FastAPI validation errors: [{ msg: "..." }, ...]
  if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return detail[0].msg.replace(/^Value error, /, "");
  return null;
}

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
    throw new ApiError(
      res.status === 503
        ? "The recommendation model is temporarily unavailable. Please try again shortly."
        : res.status === 404
          ? "We couldn't find that movie."
          : (describe(body?.detail) ?? `Request failed (${res.status}).`),
      res.status,
    );
  }
  return res.json() as Promise<T>;
}

const post = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  /** Wakes the API from scale-to-zero; fired once when the site opens. */
  warm: () => fetch(`${API_URL}/health`).catch(() => undefined),
  genres: () => call<string[]>("/genres"),
  browse: (offset: number, genre: string | null, newOnly = false, limit = 24) =>
    call<BrowseResponse>(
      `/movies/browse?offset=${offset}&limit=${limit}${genre ? `&genre=${encodeURIComponent(genre)}` : ""}${newOnly ? "&new_only=true" : ""}`,
    ),
  /** Well-known films across genres (used for the home page backdrop). */
  starter: (n = 24) => call<Movie[]>(`/movies/starter?n=${n}`),
  search: (q: string) => call<Movie[]>(`/movies/search?q=${encodeURIComponent(q)}&limit=24`),
  movie: (id: number) => call<Movie>(`/movies/${id}`),
  similar: (id: number, k = 12) => call<SimilarMovie[]>(`/movies/${id}/similar?k=${k}`),
  lookup: (ids: number[]) =>
    ids.length ? call<Movie[]>(`/movies/lookup?ids=${ids.join(",")}`) : Promise.resolve([]),
  /** ``rotate`` cycles the "Because you…" rows through your films; bump it on Refresh. */
  rows: (lists: ListsBody, rotate = 0) => call<RowsResponse>("/recommend/rows", post({ ...lists, rotate })),
};
