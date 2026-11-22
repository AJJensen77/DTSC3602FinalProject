import { createBrowserClient } from "@supabase/ssr";

// One browser client for the whole app. Row Level Security on the `ratings`
// table means each signed-in person can only ever read or change their own rows.
export const supabase = createBrowserClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
);
