"use client";

import type { Session } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import type { ListName } from "@/lib/api";
import { supabase } from "@/lib/supabase";

export function useSession() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setLoading(false);
    });
    const { data } = supabase.auth.onAuthStateChange((_event, s) => setSession(s));
    return () => data.subscription.unsubscribe();
  }, []);

  return { session, loading };
}

/** Redirects to /login when signed out; returns the session once known. */
export function useRequireSession() {
  const { session, loading } = useSession();
  const router = useRouter();
  useEffect(() => {
    if (!loading && !session) router.replace("/login");
  }, [loading, session, router]);
  return { session, loading };
}

/** One person's library: star ratings plus which list each movie is on. */
export type RatingsState = {
  ratings: Map<number, number>;
  lists: Map<number, ListName>;
  loading: boolean;
  error: string | null;
  /** Rating a movie also puts it on Finished watching. */
  setRating: (movieId: number, rating: number | null) => Promise<void>;
  /** null removes the movie from both lists (its rating, if any, is kept). */
  setList: (movieId: number, list: ListName | null) => Promise<void>;
  reload: () => Promise<void>;
};

export const RatingsContext = createContext<RatingsState | null>(null);

export function useRatings(): RatingsState {
  const ctx = useContext(RatingsContext);
  if (!ctx) throw new Error("useRatings must be used inside <RatingsProvider>");
  return ctx;
}

type Loaded = { ratings: Map<number, number>; lists: Map<number, ListName>; error: string | null };

async function fetchLibrary(userId: string | null): Promise<Loaded> {
  if (!userId) return { ratings: new Map(), lists: new Map(), error: null };
  const [r, l] = await Promise.all([
    supabase.from("ratings").select("movie_id, rating"),
    supabase.from("list_items").select("movie_id, list").order("added_at"),
  ]);
  if (r.error || l.error)
    return { ratings: new Map(), lists: new Map(), error: "Couldn't load your lists and ratings. Please refresh the page." };
  return {
    ratings: new Map(r.data.map((x) => [x.movie_id as number, Number(x.rating)])),
    lists: new Map(l.data.map((x) => [x.movie_id as number, x.list as ListName])),
    error: null,
  };
}

const EMPTY_R = new Map<number, number>();
const EMPTY_L = new Map<number, ListName>();

function withEntry<V>(m: Map<number, V>, key: number, value: V | null | undefined): Map<number, V> {
  const next = new Map(m);
  if (value === null || value === undefined) next.delete(key);
  else next.set(key, value);
  return next;
}

export function useRatingsStore(userId: string | null): RatingsState {
  const [ratings, setRatings] = useState<Map<number, number>>(EMPTY_R);
  const [lists, setLists] = useState<Map<number, ListName>>(EMPTY_L);
  // Which user the loaded data belongs to; loading until it matches the current one.
  const [loadedFor, setLoadedFor] = useState<string | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const loading = loadedFor !== userId;
  // Latest values for the async writers, without re-creating them on every change.
  const latest = useRef({ ratings, lists });
  useEffect(() => {
    latest.current = { ratings, lists };
  }, [ratings, lists]);

  const apply = useCallback((res: Loaded, forUser: string | null) => {
    setRatings(res.ratings);
    setLists(res.lists);
    setError(res.error);
    setLoadedFor(forUser);
  }, []);

  const reload = useCallback(async () => apply(await fetchLibrary(userId), userId), [apply, userId]);

  useEffect(() => {
    let cancelled = false;
    fetchLibrary(userId).then((res) => !cancelled && apply(res, userId));
    return () => {
      cancelled = true;
    };
  }, [apply, userId]);

  const writeList = useCallback(
    async (movieId: number, list: ListName | null) => {
      const previous = latest.current.lists.get(movieId);
      if (previous === (list ?? undefined)) return true;
      setLists((m) => withEntry(m, movieId, list));
      const { error } =
        list === null
          ? await supabase.from("list_items").delete().eq("movie_id", movieId)
          : await supabase
              .from("list_items")
              .upsert({ user_id: userId, movie_id: movieId, list, added_at: new Date().toISOString() });
      if (error) {
        setLists((m) => withEntry(m, movieId, previous));
        return false;
      }
      return true;
    },
    [userId],
  );

  const setList = useCallback(
    async (movieId: number, list: ListName | null) => {
      if (!userId) return void toast("Log in or sign up to keep your own lists.");
      if (!(await writeList(movieId, list))) toast.error("Couldn't update your list. Please try again.");
    },
    [userId, writeList],
  );

  const setRating = useCallback(
    async (movieId: number, rating: number | null) => {
      if (!userId) return void toast("Log in or sign up to rate movies.");
      const previous = latest.current.ratings.get(movieId);
      // Optimistic update so the stars respond instantly; rolled back on failure.
      setRatings((m) => withEntry(m, movieId, rating));
      const { error } =
        rating === null
          ? await supabase.from("ratings").delete().eq("movie_id", movieId)
          : await supabase
              .from("ratings")
              .upsert({ user_id: userId, movie_id: movieId, rating, updated_at: new Date().toISOString() });
      if (error) {
        toast.error("Couldn't save that rating. Please try again.");
        setRatings((m) => withEntry(m, movieId, previous));
        return;
      }
      // You can only rate what you've watched: move it to Finished watching.
      if (rating !== null) await writeList(movieId, "finished");
    },
    [userId, writeList],
  );

  return {
    ratings: loading ? EMPTY_R : ratings,
    lists: loading ? EMPTY_L : lists,
    loading,
    error,
    setRating,
    setList,
    reload,
  };
}
