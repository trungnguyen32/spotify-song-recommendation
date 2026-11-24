# Data preparation contract (milestone 1)

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe src\spotify_data_pipeline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The CLI accepts `--input` and `--output-dir`. Default input is
`data/raw/dataset.csv`; output is `data/processed/canonical/`. Generated outputs
are ignored by Git and can be rebuilt. The pipeline uses existing pandas and
NumPy dependencies; tests use Python's built-in unittest.

## Outputs and grain

| Output | One row represents | Purpose |
|---|---|---|
| `tracks.csv` | One track ID | Global counts and future track-level analysis |
| `track_genres.csv` | One unique track ID and genre pair | Preserve all genre memberships for joins |
| `observations.csv` | One distinct valid source observation | Retain source popularity and genre evidence |
| `rejected_rows.csv` | One invalid source row | Explain exclusion using rule names |
| `quality_report.json` | One run summary | Counts, rule failures, input SHA-256 and dependency versions |

`source_row` in the rejection file is the zero-based data-row position, excluding
the CSV header. Rule counts can overlap; add them only after accounting for this.

## Cleaning decisions

- Require all expected fields. Missing columns fail immediately.
- Trim text. Reject missing/blank track ID, artist, title or genre. Missing albums
  become `Unknown`. Artist strings remain source labels, not verified artist IDs.
- Convert numerical fields; reject missing, nonnumeric and infinite values.
- Enforce audio proportions in [0, 1], popularity in [0, 100], positive tempo and
  duration, key in {-1, 0, ..., 11}, mode in {0, 1}, and integer meter in [0, 7].
  Zero meter is retained as an unknown source value. Explicit accepts true/false
  or 1/0. Positive tempo is an eligibility rule, not proof that zero means corrupt
  data: some unmetered recordings may be excluded.
- Remove exact duplicates after dropping the raw CSV's incidental index column.
- Keep every track-genre membership. A multi-genre song appears once in global
  track statistics and once in each applicable genre. Genre counts overlap and
  must not be summed to claim a unique catalog size.
- For conflicting popularity values, use the median of **distinct** observed
  values per ID. This avoids overweighting repeated genre rows and does not claim
  to reconstruct the latest Spotify score (no timestamps are available). The
  original values remain in observations. Fractional medians are allowed.
- Fail on conflicting non-popularity attributes for an ID, including artist,
  album, title and audio attributes. Review these conflicts rather than silently
  selecting a version. Conflicts among rejected rows are outside this check.
- Sort derived tables deterministically and overwrite outputs on rerun. A source
  SHA-256 fingerprint identifies the exact input bytes. This is local file ETL;
  database loading and transactional publishing are later milestones.

## Verified snapshot

The included raw snapshot produced 114,000 input rows, 158 rejected rows, 450
removed exact duplicates, 113,392 distinct valid observations, 89,583 tracks,
113,392 memberships and all 114 genres. 720 tracks have conflicting popularity.
The source fingerprint is recorded in the generated report.

## Integration boundary and next milestone

The existing dashboard, saved estimators, `spotify_cleaned.csv` and
`spotify_cleaned_eda.csv` still use the legacy path. Their reported model scores
have not been remeasured for the canonical data. The new tables are a validated
foundation, not a claim that the model migration is complete. Existing local
notebook edits are preserved.

PostgreSQL tracks and track-genre tables, repeatable loading and SQL analytics
are now implemented; see [database documentation](postgresql.md). Model/EDA migration needs explicit choices
about multi-genre features and track weighting; retain original evaluation
metrics until a fresh grouped evaluation is run. Later steps cover grouped CV,
out-of-fold artist residuals, explanations, recommender evaluation and the four
dashboard sections.
