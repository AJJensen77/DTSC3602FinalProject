// Server-only TMDB helpers. The key comes from TMDB_API_KEY and never reaches the browser.

const WEEK = 60 * 60 * 24 * 7;
const IMG = "https://image.tmdb.org/t/p";

/** A TMDB id ("603") or an IMDb id ("tt0133093") for newer films MovieLens doesn't link. */
export function validKey(key: string) {
  return /^\d+$/.test(key) || /^tt\d+$/.test(key);
}

async function tmdb<T>(path: string): Promise<T | null> {
  const key = process.env.TMDB_API_KEY;
  if (!key) return null;
  // TMDB issues a short v3 "API key" and a long v4 "read access token"; accept either.
  const isToken = key.length > 60;
  const auth = isToken ? "" : `${path.includes("?") ? "&" : "?"}api_key=${key}`;
  const res = await fetch(`https://api.themoviedb.org/3${path}${auth}`, {
    headers: isToken ? { Authorization: `Bearer ${key}` } : undefined,
    next: { revalidate: WEEK },
  });
  return res.ok ? ((await res.json()) as T) : null;
}

/** IMDb ids are looked up through TMDB's /find endpoint. */
export async function tmdbId(key: string): Promise<number | null> {
  if (/^\d+$/.test(key)) return Number(key);
  const found = await tmdb<{ movie_results?: { id: number }[] }>(`/find/${key}?external_source=imdb_id`);
  return found?.movie_results?.[0]?.id ?? null;
}

type Raw = {
  id: number;
  overview: string | null;
  tagline: string | null;
  runtime: number | null;
  release_date: string | null;
  poster_path: string | null;
  backdrop_path: string | null;
  credits?: {
    cast: { name: string; character: string; profile_path: string | null; order: number }[];
    crew: { name: string; job: string }[];
  };
  videos?: { results: { key: string; site: string; type: string; official: boolean; iso_639_1: string }[] };
  release_dates?: { results: { iso_3166_1: string; release_dates: { certification: string }[] }[] };
};

export type MovieDetails = {
  tmdb_id: number;
  overview: string | null;
  tagline: string | null;
  runtime: number | null;
  release_date: string | null;
  certification: string | null;
  backdrop: string | null;
  directors: string[];
  writers: string[];
  cast: { name: string; character: string; photo: string | null }[];
  /** YouTube video id of the best trailer, if any. */
  trailer: string | null;
};

/** Pick the most "official trailer"-like YouTube video. */
function bestTrailer(videos: NonNullable<Raw["videos"]>["results"]): string | null {
  const yt = videos.filter((v) => v.site === "YouTube");
  const rank = (v: (typeof yt)[number]) =>
    (v.type === "Trailer" ? 0 : v.type === "Teaser" ? 2 : 4) + (v.official ? 0 : 1) + (v.iso_639_1 === "en" ? 0 : 0.5);
  return yt.sort((a, b) => rank(a) - rank(b))[0]?.key ?? null;
}

export async function movieDetails(key: string): Promise<MovieDetails | null> {
  const id = await tmdbId(key);
  if (id === null) return null;
  const m = await tmdb<Raw>(`/movie/${id}?append_to_response=credits,videos,release_dates`);
  if (!m) return null;
  const crew = m.credits?.crew ?? [];
  const us = m.release_dates?.results.find((r) => r.iso_3166_1 === "US");
  return {
    tmdb_id: m.id,
    overview: m.overview || null,
    tagline: m.tagline || null,
    runtime: m.runtime || null,
    release_date: m.release_date || null,
    certification: us?.release_dates.map((d) => d.certification).find(Boolean) ?? null,
    backdrop: m.backdrop_path ? `${IMG}/w1280${m.backdrop_path}` : null,
    directors: [...new Set(crew.filter((c) => c.job === "Director").map((c) => c.name))],
    writers: [...new Set(crew.filter((c) => c.job === "Screenplay" || c.job === "Writer").map((c) => c.name))].slice(0, 3),
    cast: (m.credits?.cast ?? [])
      .sort((a, b) => a.order - b.order)
      .slice(0, 15)
      .map((c) => ({ name: c.name, character: c.character, photo: c.profile_path ? `${IMG}/w185${c.profile_path}` : null })),
    trailer: bestTrailer(m.videos?.results ?? []),
  };
}

export const CACHE_HEADERS = { "Cache-Control": `public, max-age=${60 * 60}, s-maxage=${WEEK}` };
