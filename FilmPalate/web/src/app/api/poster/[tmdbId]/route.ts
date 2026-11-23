// Looks up a film's poster on TMDB and redirects to the image. Runs on the server
// so the TMDB key never reaches the browser; responses are cached for a week.
// The id is a TMDB id, or an IMDb id ("tt…") for newer films MovieLens doesn't link.

const WEEK = 60 * 60 * 24 * 7;

type TmdbMovie = { poster_path?: string | null };

export async function GET(_req: Request, ctx: RouteContext<"/api/poster/[tmdbId]">) {
  const { tmdbId: id } = await ctx.params;
  const key = process.env.TMDB_API_KEY;
  const isImdb = /^tt\d+$/.test(id);
  if (!key || !(isImdb || /^\d+$/.test(id))) return new Response(null, { status: 404 });

  // TMDB issues a short v3 "API key" and a long v4 "read access token"; accept either.
  const isToken = key.length > 60;
  const path = isImdb ? `/find/${id}?external_source=imdb_id` : `/movie/${id}`;
  const auth = isToken ? "" : `${path.includes("?") ? "&" : "?"}api_key=${key}`;
  const res = await fetch(`https://api.themoviedb.org/3${path}${auth}`, {
    headers: isToken ? { Authorization: `Bearer ${key}` } : undefined,
    next: { revalidate: WEEK },
  });
  if (!res.ok) return new Response(null, { status: 404 });

  const body = await res.json();
  const movie: TmdbMovie | undefined = isImdb ? (body as { movie_results?: TmdbMovie[] }).movie_results?.[0] : body;
  const poster = movie?.poster_path;
  if (!poster) return new Response(null, { status: 404, headers: { "Cache-Control": `public, s-maxage=${WEEK}` } });

  return new Response(null, {
    status: 307,
    headers: {
      Location: `https://image.tmdb.org/t/p/w342${poster}`,
      "Cache-Control": `public, max-age=${WEEK}, s-maxage=${WEEK}`,
    },
  });
}
