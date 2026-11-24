# PostgreSQL and SQL analytics (milestone 2)

## What is implemented

The loader reads the raw CSV and calls the same validation/cleaning function as
the file pipeline. It loads PostgreSQL directly from that validated result, so
manually edited intermediate CSVs cannot bypass the contract.

| Relation | Grain and purpose |
|---|---|
| `spotify.tracks` | One unique track ID, canonical popularity and audio attributes |
| `spotify.genres` | One source genre label |
| `spotify.track_genres` | One unique track/genre pair, linked by foreign keys |
| `spotify.dataset_snapshot` | Current input fingerprint, quality report and load time |

Audio features are columns of tracks: a separate one-to-one table adds no useful
relationship here. Artist strings are retained as supplied; name strings are not
verified Spotify artist IDs. No artist entity or artist-lift SQL claims are
introduced until reliable identity rules and out-of-fold residuals are available.

Primary keys prevent duplicates. Foreign keys prevent orphan genre memberships.
CHECK constraints reject invalid ranges. The genre-first index supports genre
filters; the membership primary key already supports track-first joins.

## Run locally

Use an existing PostgreSQL database dedicated to this project. Create it with
your normal PostgreSQL tools; the loader does not create users or databases.
The role needs permission to create the `spotify` schema and its tables/views.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-db.txt
$env:SPOTIFY_DATABASE_URL = 'postgresql://USER:PASSWORD@127.0.0.1:5432/spotify'
.\.venv\Scripts\python.exe src\spotify_database.py
.\.venv\Scripts\python.exe src\spotify_sql_analytics.py
```

Replace the sample credentials; URL-encode special characters in the URL.
Connection strings are read from the process environment and are not logged.
`.env` files are ignored by Git but are **not** automatically loaded.

An optional disposable database for experimentation, if Docker Desktop is running:

```powershell
docker run --detach --rm --name spotify-local -e POSTGRES_USER=spotify -e POSTGRES_PASSWORD=local-only -e POSTGRES_DB=spotify -p 127.0.0.1:55439:5432 postgres:17
# Wait until this reports accepting connections:
docker exec spotify-local pg_isready -U spotify -d spotify
$env:SPOTIFY_DATABASE_URL = 'postgresql://spotify:local-only@127.0.0.1:55439/spotify'
```

This optional container has no persistent volume; stopping it removes its data.
It is a local verification convenience, not a deployment requirement.

The loader accepts `--input`; the exporter accepts `--output-dir`. Default exports
are ignored, rebuildable CSV files in `reports/sql/`, with a source-fingerprint
manifest. The exporter uses a single repeatable-read, read-only database snapshot
so the four query results describe the same catalog.

## Repeatability and failure behavior

Each load replaces the **entire current snapshot** in the dedicated `spotify`
schema. Records absent from the new snapshot are removed. This is snapshot
replacement, not incremental ingestion or append-only history.

Validation runs before writing. Inserts use COPY for efficient bulk transfer.
Deletes, inserts, metadata and views are in one transaction: an error restores
the previous snapshot. An advisory lock serializes these loaders; table locks
protect writes while ordinary SELECT readers can still use the committed data.
An empty validated snapshot is refused. Repeated loads preserve catalog content
and counts; `loaded_at` updates to reflect the new load.

Use only the project's schema: this is not a generic loader for shared tables.
`CREATE TABLE IF NOT EXISTS` initializes this schema; future structural changes
will require explicit migrations rather than relying on it to alter old tables.

## Analyses and interpretation

`sql/views.sql` defines catalog overview, genre summary and energy-band summary.
`sql/analytics.sql` contains four analyses using CTEs, joins, window functions,
percentiles and conditional aggregation:

1. Genre popularity ranking: median, interquartile range and high-popularity
   share, with at least 30 tracks. Top ten **dense ranks** can include more than
   ten genres when medians tie.
2. Five highest-popularity tracks per genre, with deviation from the genre mean.
   Ties are resolved deterministically by track ID. This is descriptive genre
   comparison, not model residual lift.
3. Popularity across low (<0.33), medium (<0.67) and high energy bands.
   Cutoffs are descriptive choices, not discovered significance thresholds.
4. Mean audio profiles for all genres.

Global statistics read `tracks`, not a joined track-genre table. Each genre
contains unique tracks, but genres overlap, so genre totals cannot be summed
to infer unique catalog size. These associations are observational, not causal.
Canonical popularity follows the distinct-value median rule in the data contract.

## Verification

```powershell
# Use an empty disposable database ending in _test, with no spotify schema.
$env:SPOTIFY_TEST_DATABASE_URL = 'postgresql://USER:PASSWORD@127.0.0.1:5432/spotify_test'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Integration tests roll back their changes and refuse an existing spotify schema.
Without the test URL or driver, PostgreSQL tests are explicitly skipped; data
contract tests still run. GitHub Actions now provisions PostgreSQL 17 to run both.

Locally verified against PostgreSQL 17.12: all 13 tests passed, including repeat
loading, stale-row removal, constraints, forced-write rollback, empty-input
protection and all four SQL queries. The full 114,000-row source was loaded twice:
89,583 tracks, 114 genres and 113,392 memberships each time. SQL exports contained
18 genre-ranking rows (ties), 570 top-track rows, 3 energy bands and 114 genre
profiles. Remote Actions execution is pending a push.

The dashboard and ML pipeline still use their legacy artifacts. This milestone
does not change model metrics or implement the future statistical investigation.
