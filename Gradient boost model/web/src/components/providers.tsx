"use client";

import { RatingsContext, useRatingsStore, useSession } from "@/lib/use-ratings";

export function Providers({ children }: { children: React.ReactNode }) {
  const { session } = useSession();
  const store = useRatingsStore(session?.user.id ?? null);
  return <RatingsContext.Provider value={store}>{children}</RatingsContext.Provider>;
}
