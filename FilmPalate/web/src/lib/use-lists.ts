"use client";

import type { Session } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import type { ListsBody } from "@/lib/api";
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

/** One film on one of the two lists. A film is on at most one list. */
export type Entry = { list: "rated" | "want"; stars: number | null; addedAt: string };

export type ListsState = {
  entries: Map<number, Entry>;
  /** Ids on My Ratings / Want to Watch, newest first. */
  rated: number[];
  want: number[];
  loading: boolean;
  error: string | null;
  signedIn: boolean;
  /** Stars on a film (moves it off Want to Watch); null removes the rating. */
  rate: (movieId: number, stars: number | null) => Promise<void>;
  /** Adds to / removes from Want to Watch. Rated films can't be added. */
  toggleWant: (movieId: number) => Promise<void>;
  /** The lists in the API's format, oldest first (recency matters for some rows). */
  body: () => ListsBody;
  reload: () => Promise<void>;
};

export const ListsContext = createContext<ListsState | null>(null);

export function useLists(): ListsState {
  const ctx = useContext(ListsContext);
  if (!ctx) throw new Error("useLists must be used inside <ListsContext.Provider>");
  return ctx;
}

type Loaded = { entries: Map<number, Entry>; error: string | null };

async function fetchLists(userId: string | null): Promise<Loaded> {
  if (!userId) return { entries: new Map(), error: null };
  const { data, error } = await supabase.from("fp_lists").select("movie_id, list, stars, added_at");
  if (error) return { entries: new Map(), error: "Couldn't load your lists. Please refresh the page." };
  return {
    entries: new Map(
      data.map((r) => [
        Number(r.movie_id),
        { list: r.list as Entry["list"], stars: r.stars === null ? null : Number(r.stars), addedAt: r.added_at as string },
      ]),
    ),
    error: null,
  };
}

const EMPTY = new Map<number, Entry>();

export function useListsStore(userId: string | null): ListsState {
  const [entries, setEntries] = useState<Map<number, Entry>>(EMPTY);
  // Which user the loaded lists belong to; loading until it matches the current one.
  const [loadedFor, setLoadedFor] = useState<string | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const loading = loadedFor !== userId;

  const apply = useCallback((res: Loaded, forUser: string | null) => {
    setEntries(res.entries);
    setError(res.error);
    setLoadedFor(forUser);
  }, []);

  const reload = useCallback(async () => apply(await fetchLists(userId), userId), [apply, userId]);

  useEffect(() => {
    let cancelled = false;
    fetchLists(userId).then((res) => !cancelled && apply(res, userId));
    return () => {
      cancelled = true;
    };
  }, [apply, userId]);

  /** Optimistic local change, then the database write; rolled back if it fails. */
  const change = useCallback(
    async (movieId: number, next: Entry | null, write: () => PromiseLike<{ error: unknown }>, failMsg: string) => {
      let previous: Entry | undefined;
      setEntries((m) => {
        previous = m.get(movieId);
        const copy = new Map(m);
        if (next) copy.set(movieId, next);
        else copy.delete(movieId);
        return copy;
      });
      const { error } = await write();
      if (error) {
        toast.error(failMsg);
        setEntries((m) => {
          const copy = new Map(m);
          if (previous) copy.set(movieId, previous);
          else copy.delete(movieId);
          return copy;
        });
      }
    },
    [],
  );

  const rate = useCallback(
    async (movieId: number, stars: number | null) => {
      if (!userId) return;
      const now = new Date().toISOString();
      const existing = entries.get(movieId);
      if (stars === null) {
        if (existing?.list !== "rated") return;
        await change(movieId, null, () => supabase.from("fp_lists").delete().eq("movie_id", movieId),
          "Couldn't remove that rating. Please try again.");
        return;
      }
      const addedAt = existing?.list === "rated" ? existing.addedAt : now;
      await change(
        movieId,
        { list: "rated", stars, addedAt },
        () => supabase.from("fp_lists").upsert({
          user_id: userId, movie_id: movieId, list: "rated", stars, added_at: addedAt, updated_at: now,
        }),
        "Couldn't save that rating. Please try again.",
      );
      if (existing?.list === "want") toast.success("Moved from Want to Watch to My Ratings");
    },
    [userId, entries, change],
  );

  const toggleWant = useCallback(
    async (movieId: number) => {
      if (!userId) return;
      const existing = entries.get(movieId);
      if (existing?.list === "rated") return;
      if (existing?.list === "want") {
        await change(movieId, null, () => supabase.from("fp_lists").delete().eq("movie_id", movieId),
          "Couldn't update Want to Watch. Please try again.");
        return;
      }
      const now = new Date().toISOString();
      await change(
        movieId,
        { list: "want", stars: null, addedAt: now },
        () => supabase.from("fp_lists").insert({ user_id: userId, movie_id: movieId, list: "want", added_at: now }),
        "Couldn't add that to Want to Watch. Please try again.",
      );
    },
    [userId, entries, change],
  );

  const shown = loading ? EMPTY : entries;
  const { rated, want } = useMemo(() => {
    const byNewest = [...shown].sort((a, b) => b[1].addedAt.localeCompare(a[1].addedAt));
    return {
      rated: byNewest.filter(([, e]) => e.list === "rated").map(([id]) => id),
      want: byNewest.filter(([, e]) => e.list === "want").map(([id]) => id),
    };
  }, [shown]);

  const body = useCallback(
    (): ListsBody => ({
      ratings: [...rated].reverse().map((id) => ({ movie_id: id, stars: shown.get(id)!.stars! })),
      want_to_watch: [...want].reverse(),
    }),
    [rated, want, shown],
  );

  return { entries: shown, rated, want, loading, error, signedIn: !!userId, rate, toggleWant, body, reload };
}

/** Changes whenever the lists change; used to tell if shown picks are out of date. */
export const listsSignature = (b: ListsBody) =>
  `${b.ratings.map((r) => `${r.movie_id}:${r.stars}`).join(",")}|${b.want_to_watch.join(",")}`;
