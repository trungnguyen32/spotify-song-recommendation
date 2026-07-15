# spotify-song-recommendation

Composer-assistant dashboard for Spotify song analysis, similarity search, fame-free popularity prediction, and artist-fit recommendations.

## Project Structure

```text
spotify-song-recommendation/
  data/
    raw/                # original Spotify dataset
    processed/          # cleaned datasets for EDA and modeling
  models/               # trained model artifacts and model metadata
  notebooks/            # EDA and modeling notebooks
  src/
    app.py              # Streamlit dashboard
    spotify_train_models.py
  requirements.txt
```

## Local Setup In VS Code

Open PowerShell:

```powershell
cd D:\DA\spotify-song-recommendation
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
