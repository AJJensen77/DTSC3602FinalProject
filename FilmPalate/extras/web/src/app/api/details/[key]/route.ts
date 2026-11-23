// Movie details from TMDB (description, cast, director, runtime, trailer) for the movie
// page and hover trailers. Runs on the server so the TMDB key stays private; cached.

import { CACHE_HEADERS, movieDetails, validKey } from "@/lib/tmdb";

export async function GET(_req: Request, ctx: RouteContext<"/api/details/[key]">) {
  const { key } = await ctx.params;
  if (!validKey(key)) return Response.json({ detail: "invalid id" }, { status: 400 });
  const details = await movieDetails(key);
  if (!details) return Response.json({ detail: "no details for this film" }, { status: 404, headers: CACHE_HEADERS });
  return Response.json(details, { headers: CACHE_HEADERS });
}
