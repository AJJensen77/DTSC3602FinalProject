"use client";

import { useEffect, useRef, useState } from "react";

import { api, type RecommendResponse } from "@/lib/api";
import { useRatings } from "@/lib/use-ratings";

type Result = { key: string; data: RecommendResponse | null; error: string | null };

/**
 * Recommendations from the person's lists + ratings. Refetches when the genre filter or
 * `nonce` changes, and when the library changes if `live` is set; otherwise the latest
 * library is read at fetch time so star clicks don't reshuffle the page under the cursor.
 */
export function useRecommendations(genres: string[], { k = 20, nonce = 0, live = false } = {}) {
  const { ratings, lists, loading: libraryLoading } = useRatings();
  const library = useRef({ ratings, lists });
  useEffect(() => {
    library.current = { ratings, lists };
  }, [ratings, lists]);

  const libKey = live ? `${[...ratings].join()}|${[...lists].join()}` : "";
  const key = `${genres.join("|")}#${nonce}#${k}#${libKey}`;
  const [result, setResult] = useState<Result | null>(null);
  const [slowKey, setSlowKey] = useState<string | null>(null);
  const loading = !libraryLoading && result?.key !== key;

  useEffect(() => {
    if (libraryLoading) return;
    let cancelled = false;
    const slowTimer = setTimeout(() => setSlowKey(key), 3000);
    api
      .recommend(library.current.ratings, library.current.lists, genres, k)
      .then((d) => !cancelled && setResult({ key, data: d, error: null }))
      .catch((e: Error) => !cancelled && setResult({ key, data: null, error: e.message }))
      .finally(() => clearTimeout(slowTimer));
    return () => {
      cancelled = true;
      clearTimeout(slowTimer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `key` captures genres, k, nonce and library
  }, [libraryLoading, key]);

  return {
    loading: libraryLoading || loading,
    slow: loading && slowKey === key,
    data: loading ? null : (result?.data ?? null),
    error: loading ? null : (result?.error ?? null),
  };
}
