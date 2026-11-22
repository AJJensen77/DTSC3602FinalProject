"use client";

import { useEffect } from "react";

import { api } from "@/lib/api";
import { ListsContext, useListsStore, useSession } from "@/lib/use-lists";

export function Providers({ children }: { children: React.ReactNode }) {
  const { session } = useSession();
  const store = useListsStore(session?.user.id ?? null);

  // The model API sleeps when idle and takes a while to wake. Nudge it as soon as
  // anyone opens the site, so it's ready by the time they need it.
  useEffect(() => {
    api.warm();
  }, []);

  return <ListsContext.Provider value={store}>{children}</ListsContext.Provider>;
}
