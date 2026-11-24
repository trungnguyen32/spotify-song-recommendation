# spotify-song-recommendation

Composer-assistant dashboard for Spotify song analysis, similarity search, fame-free popularity prediction, and artist-fit recommendations.

## Development status: reproducible data preparation

The first upgrade adds a validated raw-CSV pipeline, canonical track and genre
membership tables, a quality report, automated data-contract tests, and a GitHub
Actions test workflow. See [the data contract](docs/data_contract.md) for cleaning
rules, measured counts and the next milestone. The current app and saved models
continue to use the existing datasets; model migration is planned.

## PostgreSQL and SQL analytics

A transactional snapshot loader, keyed track/genre tables, analytical views,
four SQL investigations, CSV exports and PostgreSQL integration tests are now
implemented. See [database setup and methodology](docs/postgresql.md).

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-db.txt
# Set SPOTIFY_DATABASE_URL to your dedicated project database first.
.\.venv\Scripts\python.exe src\spotify_database.py
.\.venv\Scripts\python.exe src\spotify_sql_analytics.py
```

```powershell
.\.venv\Scripts\python.exe src\spotify_data_pipeline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Project Structure

```text
spotify-song-recommendation/
  data/
    raw/                # original Spotify dataset
    processed/          # cleaned datasets for EDA and modeling
  models/               # trained model artifacts and model metadata
  notebooks/            # EDA and modeling notebooks
  reports/
    figures/             # generated EDA charts
    tables/              # generated EDA summary tables
  src/
    app.py                       # Streamlit dashboard
    spotify_eda_analysis.py      # reusable popularity-focused EDA
    spotify_train_models.py      # training and recommendation functions
  requirements.txt
```

## Local Setup In VS Code

Open PowerShell:

```powershell
git clone https://github.com/trungnguyen32/spotify-song-recommendation.git
cd spotify-song-recommendation
code .
```

Create a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

In VS Code, select the interpreter:

1. Press `Ctrl+Shift+P`
2. Choose `Python: Select Interpreter`
3. Pick `.venv\Scripts\python.exe`

Run the dashboard:

```powershell
.\.venv\Scripts\python -m streamlit run src\app.py
```

Generate the popularity-focused EDA report:

```powershell
.\.venv\Scripts\python src\spotify_eda_analysis.py
```

Retrain model artifacts:

```powershell
.\.venv\Scripts\python src\spotify_train_models.py
```

## Model Notes

- Similar songs use scaled audio features and cosine similarity.
- Genre is applied as a strict filter before returning similar songs.
- The base popularity model excludes artist, album, track name, and track ID to reduce artist-fame bias.
- Artist-fit recommendations combine style similarity with historical artist residual lift.
- Popularity predictions are estimates, not guarantees of commercial performance or virality.

## EDA Focus

The EDA notebook emphasizes which composition features are associated with popularity. It keeps one row per track ID, compares Pearson and Spearman correlations, repeats the analysis within genre, examines genre-adjusted feature trends, and reports eta-squared effect sizes for discrete attributes such as key, mode, time signature, and explicit status.

In the current dataset, every individual within-genre feature correlation is very weak: the largest absolute Spearman correlation is about `0.03`. Categorical effect sizes are also small. This supports treating the popularity model as a composition-only baseline rather than a prediction of commercial success. These results describe association rather than causation.
