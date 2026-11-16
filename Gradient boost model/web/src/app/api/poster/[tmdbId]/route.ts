// Looks up a film's poster on TMDB and redirects to the image. Runs on the server
// so the TMDB key never reaches the browser; responses are cached for a week.

const WEEK = 60 * 60 * 24 * 7;

export async function GET(_req: Request, ctx: RouteContext<"/api/poster/[tmdbId]">) {
  const { tmdbId } = await ctx.params;
  const key = process.env.TMDB_API_KEY;
  if (!key || !/^\d+$/.test(tmdbId)) return new Response(null, { status: 404 });

  // TMDB issues a short v3 "API key" and a long v4 "read access token"; accept either.
  const isToken = key.length > 60;
  const url = `https://api.themoviedb.org/3/movie/${tmdbId}${isToken ? "" : `?api_key=${key}`}`;
  const res = await fetch(url, {
    headers: isToken ? { Authorization: `Bearer ${key}` } : undefined,
    next: { revalidate: WEEK },
  });
  if (!res.ok) return new Response(null, { status: 404 });

  const { poster_path } = (await res.json()) as { poster_path?: string | null };
  if (!poster_path) return new Response(null, { status: 404, headers: { "Cache-Control": `public, s-maxage=${WEEK}` } });

  return new Response(null, {
    status: 307,
    headers: {
      Location: `https://image.tmdb.org/t/p/w342${poster_path}`,
      "Cache-Control": `public, max-age=${WEEK}, s-maxage=${WEEK}`,
    },
  });
}
