# DTSC 3602 Final Project — Movie Recommender

Can a personalized model built from a user's earlier ratings recommend movies they go
on to rate highly, better than recommending the most popular movies?

**Team.** Ishika Bhattarai, A.J. Jensen, Srijith G Menon, Yashvi Patel, Yager Range,
Max Valentine

## Current deliverable — D2: Data preparation and EDA

| Item | Location |
|---|---|
| Data note (2–3 pages) | [`docs/D2_data_note.md`](docs/D2_data_note.md) |
| Data dictionary (72 columns, 14 tables) | [`docs/data_dictionary.md`](docs/data_dictionary.md) |
| Figures (5, interpreted in the note) | [`figures/`](figures/) |
| Raw audit output | `docs/audit_report.json` |
| Preparation output | `docs/prep_report.json` |
| Preparation code | `audit_raw.py`, `prepare_data.py`, `make_figures.py`, `make_data_dictionary.py` |
| Streamlit dashboard | `app.py` (Supabase-backed), deployed via `modal_app.py` |

**Live dashboard:** _(added once deployed — see "Deploying" below)_

### Headline findings

- **Television removed.** MovieLens has no title-type field, so the catalog was joined
  to IMDb's `title.basics.tsv` through `links.csv`. 2,505 episodic TV titles were
  dropped (0.33% of ratings); made-for-TV and direct-to-video *films* were kept.
- **The rating scale changed mid-dataset.** Half-stars did not exist before Feb 2003
  and are 42% of ratings after.
- **Popularity is extremely concentrated.** The top 1% of films hold 39% of training
  ratings; the top 10% hold 90%.
- **Most of the catalog is unlearnable.** 49,419 of 85,080 films have zero training
  ratings.
- **Most later ratings cannot be personalized.** 84% of validation and 90% of test
  ratings come from users with no training history, so evaluation runs on an explicit
  scorable cohort — which rates 0.25 stars lower than the users it excludes.

## Setup

```bash
uv sync
```

### Getting the data

Raw data is gitignored (877 MB of ratings). Recreate `data/raw/`:

1. Download [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) and unzip it
   so the files land at `data/raw/ml-32m/{ratings,movies,links,tags}.csv`.
2. Download IMDb title basics to `data/raw/title.basics.tsv.gz`:
   ```bash
   curl -o data/raw/title.basics.tsv.gz https://datasets.imdbws.com/title.basics.tsv.gz
   ```

### Running the pipeline

```bash
uv run python audit_raw.py             # audit the raw files    -> docs/audit_report.json
uv run python prepare_data.py          # clean, drop TV, split  -> data/processed/, data/reserved_test/
uv run python make_figures.py          # five figures           -> figures/
uv run python make_data_dictionary.py  # dictionary             -> docs/data_dictionary.md
```

`prepare_data.py` needs about 4 GB of RAM and takes a few minutes on 32M rows.

### Database and dashboard

Copy `.env.example` to `.env` and fill in your Supabase connection string
(Supabase → Project Settings → Database → Connection string → URI), then:

```bash
uv run python upload_to_supabase.py    # creates the d2_* tables
uv run streamlit run app.py            # http://localhost:8501
```

Only the small aggregate tables are uploaded — the 22M-row ratings splits stay local
as parquet. Tables are namespaced `d2_` so they cannot collide with other coursework.

To work on the layout without a database, `USE_LOCAL_PARQUET=1 uv run streamlit run app.py`
reads the local parquet files instead. This is a development convenience; the graded
path is the Supabase one.

### Deploying

```bash
uv run modal secret create supabase-db-url DATABASE_URL=postgresql://...
uv run modal deploy modal_app.py
```

## Reserved test data

`data/reserved_test/ratings_test.parquet` holds every rating after **2019-11-02**
(4,783,987 rows). It is sealed until D6: model selection, hyperparameters, and
thresholds in D3–D5 use `data/processed/ratings_validation.parquet` only. See
`data/reserved_test/SEALED.md`.

## Repository layout

```
audit_raw.py              raw-file audit, before any cleaning
prepare_data.py           TV removal, cleaning, chronological split, aggregates
make_figures.py           the five figures in the data note
make_data_dictionary.py   dictionary generated from the written parquet schema
upload_to_supabase.py     pushes aggregate tables to Supabase
common.py                 Supabase connection + table names
app.py                    Streamlit dashboard (reads Supabase)
modal_app.py              Modal deployment of the dashboard
src/dtsc_3602_final_project/
    paths.py              canonical paths
    catalog.py            IMDb join + the film/TV decision
    viz.py                shared chart styling
docs/                     data note, data dictionary, audit reports
figures/                  rendered figures
data/                     gitignored (raw, processed, reserved test)
```

## Data sources and terms

- **MovieLens 32M** — GroupLens Research, University of Minnesota. Free for
  non-commercial research with attribution; does not imply GroupLens endorsement;
  commercial use requires permission.
  Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context.
  *ACM TiiS*, 5(4), 19:1–19:19. https://doi.org/10.1145/2827872
- **IMDb non-commercial datasets** — free for personal and non-commercial use.
  https://developer.imdb.com/non-commercial-datasets/

Credentials live in `.env`, which is gitignored and never committed.
