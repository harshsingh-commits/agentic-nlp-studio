import unittest

import pandas as pd

from movie_recommender import build_movie_profiles, load_movie_data, recommend_similar_movies


class MovieRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.reviews = pd.DataFrame(
            {
                "movie_title": ["North Star", "North Star", "Quiet River", "Red Planet"],
                "genre": ["Drama", "Comedy", "Drama", "Sci-Fi"],
                "language": ["English", "English", "French", "English"],
                "release_year": ["2020", "2021", "2019", "2022"],
                "director": ["A", "B", "C", "D"],
                "user_review": [
                    "A quiet, thoughtful story with a moving ending",
                    "Thoughtful story and beautiful ending",
                    "A quiet story with a thoughtful ending",
                    "A spaceship explores a distant planet",
                ],
                "rating": [5, 4, 4, 3],
                "sentiment": ["Positive", "Positive", "Positive", "Neutral"],
                "watch_time_min": [90, 88, 100, 110],
                "recommend": [1, 1, 1, 0],
            }
        )

    def test_profiles_aggregate_reviews_not_metadata_as_unique_films(self):
        profiles = build_movie_profiles(self.reviews)
        north_star = profiles.set_index("movie_title").loc["North Star"]
        self.assertEqual(len(profiles), 3)
        self.assertEqual(north_star["review_count"], 2)
        self.assertEqual(north_star["average_rating"], 4.5)
        self.assertIn("Drama", north_star["genre_mix"])
        self.assertIn("Comedy", north_star["genre_mix"])

    def test_review_similarity_excludes_selected_movie_and_ranks_related_text(self):
        recommendations = recommend_similar_movies(self.reviews, "North Star", limit=2)
        self.assertNotIn("North Star", recommendations["movie_title"].tolist())
        self.assertEqual(recommendations.iloc[0]["movie_title"], "Quiet River")

    def test_csv_loader_validates_required_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            load_movie_data(b"movie_title\nNorth Star\n")


if __name__ == "__main__":
    unittest.main()