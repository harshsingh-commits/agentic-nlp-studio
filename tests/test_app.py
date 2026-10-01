import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
DATASET_PATH = APP_PATH.parent / "movie_recommendation_nlp_dataset.csv"


class MovieDashboardTests(unittest.TestCase):
    def test_dashboard_renders_and_genre_filter_updates_review_count(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Overview", "Discover", "Review NLP", "Review records"],
        )

        genre_widget = next(widget for widget in app.multiselect if widget.label == "Review genre")
        genre_widget.select("Comedy").run()
        expected_count = len(pd.read_csv(DATASET_PATH).query("genre == 'Comedy'"))

        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, str(expected_count))


if __name__ == "__main__":
    unittest.main()