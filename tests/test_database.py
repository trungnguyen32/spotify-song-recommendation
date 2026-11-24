"""PostgreSQL integration contracts; use a disposable database ending in _test."""
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from test_data_pipeline import track

try:
    import psycopg
    from spotify_database import ROOT, load_snapshot
except ImportError:
    psycopg = None

DSN = os.environ.get("SPOTIFY_TEST_DATABASE_URL")


@unittest.skipUnless(DSN and psycopg, "Set SPOTIFY_TEST_DATABASE_URL and install psycopg for PostgreSQL integration tests")
class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connection = psycopg.connect(DSN, autocommit=True, connect_timeout=10)
        database = cls.connection.execute("SELECT current_database()").fetchone()[0]
        if not database.endswith("_test"):
            cls.connection.close()
            raise RuntimeError("Integration tests require a disposable database ending in _test")
        if cls.connection.execute("SELECT to_regnamespace('spotify')").fetchone()[0]:
            cls.connection.close()
            raise RuntimeError("Integration tests require a database without a spotify schema")

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "input.csv"
        pd.DataFrame([track(), track(track_genre="rock", popularity=60),
                      track(track_id="b", track_name="Other", popularity=80)]).to_csv(self.source, index=False)
        transaction = self.connection.transaction(force_rollback=True)
        transaction.__enter__()
        self.addCleanup(transaction.__exit__, None, None, None)

    def test_repeat_load_and_global_genre_grains(self):
        load_snapshot(self.connection, self.source)
        load_snapshot(self.connection, self.source)
        self.assertEqual(self.connection.execute("SELECT unique_tracks, genre_count, mean_popularity FROM spotify.catalog_overview").fetchone(), (2, 2, 65))
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM spotify.track_genres").fetchone()[0], 3)
        self.assertEqual(self.connection.execute("SELECT median_popularity FROM spotify.genre_summary WHERE genre_name='pop'").fetchone()[0], 65)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM spotify.dataset_snapshot").fetchone()[0], 1)

    def test_stale_records_removed_on_replacement(self):
        load_snapshot(self.connection, self.source)
        pd.DataFrame([track()]).to_csv(self.source, index=False)
        load_snapshot(self.connection, self.source)
        self.assertEqual(self.connection.execute("SELECT track_id FROM spotify.tracks").fetchall(), [("a",)])
        self.assertEqual(self.connection.execute("SELECT genre_name FROM spotify.genres").fetchall(), [("pop",)])

    def test_failed_database_write_restores_previous_snapshot(self):
        load_snapshot(self.connection, self.source)
        self.connection.execute("""
            CREATE FUNCTION spotify.reject_insert() RETURNS trigger LANGUAGE plpgsql AS
            $$ BEGIN RAISE EXCEPTION 'test failure'; END $$;
            CREATE TRIGGER reject_insert BEFORE INSERT ON spotify.tracks
            FOR EACH ROW EXECUTE FUNCTION spotify.reject_insert();
        """)
        with self.assertRaises(psycopg.Error):
            load_snapshot(self.connection, self.source)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM spotify.tracks").fetchone()[0], 2)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM spotify.track_genres").fetchone()[0], 3)

    def test_constraints_reject_orphans_and_invalid_popularity(self):
        load_snapshot(self.connection, self.source)
        for sql in ["INSERT INTO spotify.track_genres VALUES ('absent', 'pop')",
                    "UPDATE spotify.tracks SET popularity=101 WHERE track_id='a'"]:
            with self.assertRaises(psycopg.IntegrityError):
                with self.connection.transaction():
                    self.connection.execute(sql)

    def test_empty_snapshot_rejected_before_writes(self):
        load_snapshot(self.connection, self.source)
        pd.DataFrame([track(tempo=0)]).to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "empty"):
            load_snapshot(self.connection, self.source)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM spotify.tracks").fetchone()[0], 2)

    def test_all_analytical_queries_execute(self):
        load_snapshot(self.connection, self.source)
        with self.connection.cursor() as cursor:
            cursor.execute((ROOT / "sql/analytics.sql").read_text())
            result_sets = 0
            while True:
                if cursor.description:
                    cursor.fetchall()
                    result_sets += 1
                if not cursor.nextset():
                    break
        self.assertEqual(result_sets, 4)


if __name__ == "__main__":
    unittest.main()
