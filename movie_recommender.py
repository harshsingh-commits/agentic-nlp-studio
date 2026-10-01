"""Dataset preparation and review-text recommendation helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {
    "movie_title",
    "genre",
    "language",
    "release_year",
    "director",
    "user_review",
    "rating",
    "sentiment",
    "watch_time_min",
    "recommend",
}


def load_movie_data(source: Path | bytes) -> pd.DataFrame:
    if isinstance(source, bytes):
        reviews = pd.read_csv(BytesIO(source))
    else:
        reviews = pd.read_csv(source)

    missing = sorted(REQUIRED_COLUMNS.difference(reviews.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    reviews = reviews.copy()
    for column in REQUIRED_COLUMNS.difference({"rating", "watch_time_min", "recommend"}):
        reviews[column] = reviews[column].fillna("").astype(str).str.strip()
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews["watch_time_min"] = pd.to_numeric(reviews["watch_time_min"], errors="coerce")
    recommend_numeric = pd.to_numeric(reviews["recommend"], errors="coerce")
    if recommend_numeric.notna().any():
        reviews["recommend"] = recommend_numeric.fillna(0).clip(0, 1)
    else:
        reviews["recommend"] = (
            reviews["recommend"].astype(str).str.casefold().isin({"yes", "true", "recommended"}).astype(int)
        )
    reviews = reviews[reviews["movie_title"].ne("")].reset_index(drop=True)
    if reviews.empty:
        raise ValueError("The CSV does not contain any rows with a movie title.")
    return reviews


def _category_mix(values: pd.Series, limit: int = 3) -> str:
    shares = values[values.ne("")].value_counts(normalize=True).head(limit)
    return "  ·  ".join(f"{label} {share:.0%}" for label, share in shares.items()) or "Not provided"


def build_movie_profiles(reviews: pd.DataFrame) -> pd.DataFrame:
    profiles = []
    for title, group in reviews.groupby("movie_title", sort=True):
        ratings = group["rating"].dropna()
        watch_times = group["watch_time_min"].dropna()
        profiles.append(
            {
                "movie_title": title,
                "review_count": len(group),
                "average_rating": float(ratings.mean()) if not ratings.empty else 0.0,
                "recommend_rate": float(group["recommend"].mean() * 100),
                "positive_rate": float(group["sentiment"].str.casefold().eq("positive").mean() * 100),
                "average_watch_time": float(watch_times.mean()) if not watch_times.empty else 0.0,
                "genre_mix": _category_mix(group["genre"]),
                "language_mix": _category_mix(group["language"]),
                "director_mix": _category_mix(group["director"]),
                "year_mix": _category_mix(group["release_year"]),
                "review_profile": " ".join(group["user_review"].dropna().astype(str)),
            }
        )
    return pd.DataFrame(profiles)


def recommend_similar_movies(
    reviews: pd.DataFrame,
    movie_title: str,
    limit: int = 3,
) -> pd.DataFrame:
    profiles = build_movie_profiles(reviews)
    if movie_title not in set(profiles["movie_title"]):
        raise ValueError(f"Unknown movie title: {movie_title}")
    if len(profiles) < 2:
        return profiles.iloc[0:0].assign(similarity_percent=pd.Series(dtype=float))

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    vectors = vectorizer.fit_transform(
        profiles["movie_title"].str.cat(profiles["review_profile"], sep=" ")
    )
    selected_index = profiles.index[profiles["movie_title"].eq(movie_title)][0]
    scores = cosine_similarity(vectors[selected_index], vectors).ravel()
    scored = profiles.copy()
    scored["similarity_percent"] = scores * 100
    recommendations = scored[scored["movie_title"].ne(movie_title)]
    return recommendations.sort_values(
        ["similarity_percent", "recommend_rate", "average_rating"], ascending=False
    ).head(limit).reset_index(drop=True)