-- ReelPicks database: one table of each person's movie ratings.
-- Accounts themselves live in Supabase Auth (auth.users); we store nothing else.
-- Run once in the Supabase SQL editor (or `supabase db query`). Safe to re-run.

create table if not exists public.ratings (
  user_id    uuid        not null default auth.uid() references auth.users (id) on delete cascade,
  movie_id   integer     not null,
  rating     numeric(2,1) not null check (rating between 0.5 and 5.0 and rating * 2 = floor(rating * 2)),
  updated_at timestamptz not null default now(),
  primary key (user_id, movie_id)
);

-- Row Level Security: each signed-in person can only see and change their own ratings.
alter table public.ratings enable row level security;

drop policy if exists "read own ratings" on public.ratings;
create policy "read own ratings" on public.ratings
  for select to authenticated using ((select auth.uid()) = user_id);

drop policy if exists "add own ratings" on public.ratings;
create policy "add own ratings" on public.ratings
  for insert to authenticated with check ((select auth.uid()) = user_id);

drop policy if exists "change own ratings" on public.ratings;
create policy "change own ratings" on public.ratings
  for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);

drop policy if exists "remove own ratings" on public.ratings;
create policy "remove own ratings" on public.ratings
  for delete to authenticated using ((select auth.uid()) = user_id);
