-- ReelPicks (combined model) database. Run once in the Supabase SQL editor. Safe to re-run.
-- Accounts live in Supabase Auth (auth.users). We store each person's star ratings and
-- their two watch lists: Finished watching and Plan to watch.

-- Star ratings (same table as Yager's ReelPicks site).
create table if not exists public.ratings (
  user_id    uuid         not null default auth.uid() references auth.users (id) on delete cascade,
  movie_id   integer      not null,
  rating     numeric(2,1) not null check (rating between 0.5 and 5.0 and rating * 2 = floor(rating * 2)),
  updated_at timestamptz  not null default now(),
  primary key (user_id, movie_id)
);

-- Watch lists. A movie is on at most one list per person (moving it replaces the row).
create table if not exists public.list_items (
  user_id  uuid        not null default auth.uid() references auth.users (id) on delete cascade,
  movie_id integer     not null,
  list     text        not null check (list in ('finished', 'plan_to_watch')),
  added_at timestamptz not null default now(),
  primary key (user_id, movie_id)
);

-- Signed-in users may use these tables at all (newer Supabase projects don't grant this by
-- default); the row-level security policies below then limit them to their own rows.
grant select, insert, update, delete on public.ratings, public.list_items to authenticated;
revoke all on public.ratings, public.list_items from anon;

-- Row Level Security: each signed-in person can only see and change their own rows.
alter table public.ratings enable row level security;
alter table public.list_items enable row level security;

do $$
declare t text;
begin
  foreach t in array array['ratings', 'list_items'] loop
    execute format('drop policy if exists "read own" on public.%I', t);
    execute format('create policy "read own" on public.%I for select to authenticated using ((select auth.uid()) = user_id)', t);
    execute format('drop policy if exists "add own" on public.%I', t);
    execute format('create policy "add own" on public.%I for insert to authenticated with check ((select auth.uid()) = user_id)', t);
    execute format('drop policy if exists "change own" on public.%I', t);
    execute format('create policy "change own" on public.%I for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id)', t);
    execute format('drop policy if exists "remove own" on public.%I', t);
    execute format('create policy "remove own" on public.%I for delete to authenticated using ((select auth.uid()) = user_id)', t);
  end loop;
end $$;
