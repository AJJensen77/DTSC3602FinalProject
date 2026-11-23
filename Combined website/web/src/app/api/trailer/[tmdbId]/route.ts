// Finds a film's YouTube trailer on TMDB and returns its video key. Runs on the server
// so the TMDB key never reaches the browser; responses are cached for a week.

const WEEK = 60 * 60 * 24 * 7;

type Video = { site: string; type: string; key: string; official?: boolean; size?: number; iso_639_1?: string };

export async function GET(_req: Request, ctx: RouteContext<"/api/trailer/[tmdbId]">) {
  const { tmdbId } = await ctx.params;
  const key = process.env.TMDB_API_KEY;
  const headers = { "Cache-Control": `public, max-age=${WEEK}, s-maxage=${WEEK}` };
  if (!key || !/^\d+$/.test(tmdbId)) return Response.json({ key: null });

  // TMDB issues a short v3 "API key" and a long v4 "read access token"; accept either.
  const isToken = key.length > 60;
  const url = `https://api.themoviedb.org/3/movie/${tmdbId}/videos${isToken ? "" : `?api_key=${key}`}`;
  const res = await fetch(url, {
    headers: isToken ? { Authorization: `Bearer ${key}` } : undefined,
    next: { revalidate: WEEK },
  });
  if (!res.ok) return Response.json({ key: null }, { headers });

  const { results = [] } = (await res.json()) as { results?: Video[] };
  const youtube = results.filter((v) => v.site === "YouTube" && (v.iso_639_1 ?? "en") === "en");
  // Prefer an official trailer, then any trailer, then a teaser or clip.
  const rank = (v: Video) =>
    (v.type === "Trailer" ? 0 : v.type === "Teaser" ? 2 : v.type === "Clip" ? 3 : 5) + (v.official ? 0 : 1);
  const best = youtube.sort((a, b) => rank(a) - rank(b) || (b.size ?? 0) - (a.size ?? 0))[0];
  return Response.json({ key: best?.key ?? null }, { headers });
}
