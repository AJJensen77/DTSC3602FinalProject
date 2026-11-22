-- FilmPalate database: each person's two movie lists in one table.
--   list = 'rated' -> My Ratings (stars required, 0.5-5 in half steps)
--   list = 'want'  -> Want to Watch (no stars)
-- The primary key keeps a film on at most one list per person, so rating a film
-- that was on Want to Watch simply moves it to My Ratings.
-- Accounts live in Supabase Auth (auth.users). This is a separate table from
-- ReelPicks' `ratings`, so the two sites never touch each other's data.
-- Run once in the Supabase SQL editor (or `supabase db query`). Safe to re-run.

create table if not exists public.fp_lists (
  user_id    uuid         not null default auth.uid() references auth.users (id) on delete cascade,
  movie_id   bigint       not null,
  list       text         not null check (list in ('rated', 'want')),
  stars      numeric(2,1) check (stars between 0.5 and 5.0 and stars * 2 = floor(stars * 2)),
  added_at   timestamptz  not null default now(),
  updated_at timestamptz  not null default now(),
  primary key (user_id, movie_id),
  constraint stars_only_when_rated check ((list = 'rated') = (stars is not null))
);

-- Row Level Security: each signed-in person can only see and change their own rows.
alter table public.fp_lists enable row level security;

drop policy if exists "read own lists" on public.fp_lists;
create policy "read own lists" on public.fp_lists
  for select to authenticated using ((select auth.uid()) = user_id);

drop policy if exists "add to own lists" on public.fp_lists;
create policy "add to own lists" on public.fp_lists
  for insert to authenticated with check ((select auth.uid()) = user_id);

drop policy if exists "change own lists" on public.fp_lists;
create policy "change own lists" on public.fp_lists
  for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);

drop policy if exists "remove from own lists" on public.fp_lists;
create policy "remove from own lists" on public.fp_lists
  for delete to authenticated using ((select auth.uid()) = user_id);
