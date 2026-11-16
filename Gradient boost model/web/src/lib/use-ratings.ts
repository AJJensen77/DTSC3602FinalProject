"use client";

import type { Session } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { toast } from "sonner";

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

export type RatingsState = {
  ratings: Map<number, number>;
  loading: boolean;
  error: string | null;
  setRating: (movieId: number, rating: number | null) => Promise<void>;
  reload: () => Promise<void>;
};

export const RatingsContext = createContext<RatingsState | null>(null);

export function useRatings(): RatingsState {
  const ctx = useContext(RatingsContext);
  if (!ctx) throw new Error("useRatings must be used inside <RatingsProvider>");
  return ctx;
}

/** The signed-in user's ratings, kept in sync with Supabase. */
type Loaded = { ratings: Map<number, number>; error: string | null };

async function fetchRatings(userId: string | null): Promise<Loaded> {
  if (!userId) return { ratings: new Map(), error: null };
  const { data, error } = await supabase.from("ratings").select("movie_id, rating");
  if (error) return { ratings: new Map(), error: "Couldn't load your ratings. Please refresh the page." };
  return { ratings: new Map(data.map((r) => [r.movie_id as number, Number(r.rating)])), error: null };
}

const EMPTY = new Map<number, number>();

export function useRatingsStore(userId: string | null): RatingsState {
  const [ratings, setRatings] = useState<Map<number, number>>(EMPTY);
  // Which user the loaded ratings belong to; loading until it matches the current one.
  const [loadedFor, setLoadedFor] = useState<string | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const loading = loadedFor !== userId;

  const apply = useCallback((res: Loaded, forUser: string | null) => {
    setRatings(res.ratings);
    setError(res.error);
    setLoadedFor(forUser);
  }, []);

  const reload = useCallback(async () => apply(await fetchRatings(userId), userId), [apply, userId]);

  useEffect(() => {
    let cancelled = false;
    fetchRatings(userId).then((res) => !cancelled && apply(res, userId));
    return () => {
      cancelled = true;
    };
  }, [apply, userId]);

  const setRating = useCallback(
    async (movieId: number, rating: number | null) => {
      if (!userId) return;
      const previous = ratings.get(movieId);
      // Optimistic update so the stars respond instantly; rolled back on failure.
      setRatings((m) => {
        const next = new Map(m);
        if (rating === null) next.delete(movieId);
        else next.set(movieId, rating);
        return next;
      });
      const { error } =
        rating === null
          ? await supabase.from("ratings").delete().eq("movie_id", movieId)
          : await supabase
              .from("ratings")
              .upsert({ user_id: userId, movie_id: movieId, rating, updated_at: new Date().toISOString() });
      if (error) {
        toast.error("Couldn't save that rating. Please try again.");
        setRatings((m) => {
          const next = new Map(m);
          if (previous === undefined) next.delete(movieId);
          else next.set(movieId, previous);
          return next;
        });
      }
    },
    [userId, ratings],
  );

  return { ratings: loading ? EMPTY : ratings, loading, error, setRating, reload };
}
