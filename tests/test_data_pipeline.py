"""Contracts for record grain, rejection, and reproducibility."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spotify_data_pipeline import BOUNDED, prepare_tables, run_pipeline


def track(**updates):
    row = dict(track_id="a", artists="Artist", album_name="Album", track_name="Song",
               track_genre="pop", popularity=40, duration_ms=180000, explicit=False,
               key=1, loudness=-6, mode=1, tempo=120, time_signature=4)
    row.update({c: 0.5 for c in BOUNDED})
    row.update(updates)
    return row


class DataPipelineTests(unittest.TestCase):
    def test_genres_and_conflicting_popularity_are_preserved(self):
        raw = pd.DataFrame([track(), track(), track(track_genre="rock", popularity=60)])
        tables, report = prepare_tables(raw)
        self.assertEqual(len(tables["tracks"]), 1)
        self.assertEqual(tables["tracks"].iloc[0].popularity, 50)
        self.assertEqual(set(tables["track_genres"].track_genre), {"pop", "rock"})
        self.assertEqual(set(tables["observations"].popularity), {40, 60})
        self.assertEqual(report["exact_duplicates_removed"], 1)
        self.assertEqual(report["popularity_conflict_tracks"], 1)

    def test_invalid_rows_have_reasons(self):
        rows = [track(), track(track_id=" "), track(tempo=0), track(energy=2),
                track(loudness=float("inf")), track(explicit="maybe"), track(key=1.5)]
        tables, report = prepare_tables(pd.DataFrame(rows))
        self.assertEqual(report["rejected_rows"], 6)
        self.assertEqual(len(tables["tracks"]), 1)
        self.assertIn("invalid_tempo", tables["rejected_rows"].iloc[1].reasons)

    def test_missing_columns_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            prepare_tables(pd.DataFrame({"track_id": ["a"]}))

    def test_all_rejected_and_empty_inputs_have_valid_output_schemas(self):
        for raw in [pd.DataFrame([track(tempo=0)]), pd.DataFrame([track()]).iloc[:0]]:
            tables, report = prepare_tables(raw)
            self.assertEqual(report["unique_tracks"], 0)
            self.assertIn("track_id", tables["tracks"].columns)
            self.assertIn("reasons", tables["rejected_rows"].columns)

    def test_conflicting_audio_is_not_silently_selected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting track attributes"):
            prepare_tables(pd.DataFrame([track(), track(energy=0.9)]))

    def test_canonical_tables_ignore_input_order(self):
        raw = pd.DataFrame([track(), track(track_genre="rock", popularity=60)])
        first, _ = prepare_tables(raw)
        second, _ = prepare_tables(raw.iloc[::-1])
        for name in ["tracks", "track_genres", "observations"]:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_rerunning_overwrites_instead_of_appending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            pd.DataFrame([track()]).to_csv(source, index=False)
            output = root / "derived"
            run_pipeline(source, output)
            before = {f.name: f.read_bytes() for f in output.iterdir()}
            run_pipeline(source, output)
            self.assertEqual(before, {f.name: f.read_bytes() for f in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
