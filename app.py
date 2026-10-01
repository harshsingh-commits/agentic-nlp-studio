from pathlib import Path

import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from movie_recommender import build_movie_profiles, load_movie_data, recommend_similar_movies
from nlp_mini_project import analyze_text, ensure_nltk_resources


DATASET_PATH = Path(__file__).parent / "movie_recommendation_nlp_dataset.csv"

st.set_page_config(
    page_title="Movie signals",
    page_icon=":material/movie_filter:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None) -> pd.DataFrame:
    source = uploaded_bytes if uploaded_bytes is not None else DATASET_PATH
    return load_movie_data(source)


with st.sidebar:
    st.title("Movie signals", icon=":material/movie_filter:")
    st.caption("Review-led discovery from the project dataset")
    uploaded_file = st.file_uploader("Use another CSV", type=["csv"])

try:
    dataset_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    reviews = load_data(dataset_bytes)
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load this movie dataset: {error}")
    st.stop()

st.sidebar.multiselect.options if False else None

genre_options = sorted(value for value in reviews["genre"].unique() if value)
language_options = sorted(value for value in reviews["language"].unique() if value)

with st.sidebar:
    selected_genres = st.multiselect(
        "Review genre", genre_options, placeholder="All genres"
    )
    selected_languages = st.multiselect(
        "Review language", language_options, placeholder="All languages"
    )
    selected_sentiment = st.selectbox("Review sentiment", ["All", "Positive", "Neutral", "Negative"])
    minimum_rating = st.slider("Minimum rating", min_value=1, max_value=5, value=1)
    search_term = st.text_input("Find a title or review", placeholder="Search the dataset")

filtered = reviews
if selected_genres:
    filtered = filtered[filtered["genre"].isin(selected_genres)]
if selected_languages:
    filtered = filtered[filtered["language"].isin(selected_languages)]
if selected_sentiment != "All":
    filtered = filtered[filtered["sentiment"].str.casefold().eq(selected_sentiment.casefold())]
filtered = filtered[filtered["rating"].fillna(0).ge(minimum_rating)]
if search_term.strip():
    query = search_term.strip()
    searchable = filtered["movie_title"].str.contains(query, case=False, na=False, regex=False)
    searchable |= filtered["user_review"].str.contains(query, case=False, na=False, regex=False)
    filtered = filtered[searchable]

if filtered.empty:
    st.title("Movie signals", icon=":material/movie_filter:")
    st.warning("No reviews match these filters. Broaden the sidebar filters to continue.")
    st.stop()

profiles = build_movie_profiles(filtered)
recommend_share = filtered["recommend"].mean() * 100
average_rating = filtered["rating"].mean()
positive_share = filtered["sentiment"].str.casefold().eq("positive").mean() * 100

st.title("Movie signals", icon=":material/movie_filter:")
st.caption(
    f"{len(filtered):,} review records · {filtered['movie_title'].nunique()} film titles · "
    "recommendations ranked from audience review language"
)

with st.container(horizontal=True):
    st.metric("Reviews in view", f"{len(filtered):,}", border=True)
    st.metric("Titles in view", f"{filtered['movie_title'].nunique()}", border=True)
    st.metric("Average rating", f"{average_rating:.2f} / 5", border=True)
    st.metric("Recommended", f"{recommend_share:.1f}%", border=True)
    st.metric("Positive sentiment", f"{positive_share:.1f}%", border=True)

overview_tab, discover_tab, review_tab, records_tab = st.tabs(
    ["Overview", "Discover", "Review NLP", "Review records"]
)

with overview_tab:
    st.header("What viewers are saying", icon=":material/query_stats:")
    chart_genre, chart_sentiment = st.columns(2)
    with chart_genre:
        st.subheader("Review tags by genre")
        genre_counts = (
            filtered["genre"].value_counts().rename_axis("Genre").reset_index(name="Reviews")
        )
        st.bar_chart(genre_counts, x="Genre", y="Reviews")
    with chart_sentiment:
        st.subheader("Sentiment mix")
        sentiment_counts = (
            filtered["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
        )
        st.bar_chart(sentiment_counts, x="Sentiment", y="Reviews")

    st.subheader("Movie performance")
    display_profiles = profiles.sort_values(
        ["recommend_rate", "average_rating"], ascending=False
    )[
        ["movie_title", "review_count", "average_rating", "recommend_rate", "positive_rate", "genre_mix"]
    ].rename(
        columns={
            "movie_title": "Movie",
            "review_count": "Reviews",
            "average_rating": "Average rating",
            "recommend_rate": "Recommend rate (%)",
            "positive_rate": "Positive sentiment (%)",
            "genre_mix": "Most common review tags",
        }
    )
    st.dataframe(display_profiles, hide_index=True)
    st.caption(
        "Genre, director, language, and year are reported as distributions across review rows. "
        "The source data assigns multiple values to the same title."
    )

with discover_tab:
    st.header("Find a nearby watch", icon=":material/movie_filter:")
    titles = profiles["movie_title"].tolist()
    selected_title = st.selectbox("Start from a movie", titles, key="recommendation_title")
    selected_profile = profiles.loc[profiles["movie_title"].eq(selected_title)].iloc[0]
    st.write(
        f"**{selected_title}** · {selected_profile['review_count']} matching reviews · "
        f"{selected_profile['average_rating']:.2f} average rating · "
        f"{selected_profile['recommend_rate']:.1f}% recommend"
    )
    st.caption(
        "Matches use TF-IDF cosine similarity over review text and title. Current sidebar filters "
        "change the review corpus used for ranking. This is a lexical baseline, not a verdict on taste."
    )
    recommendations = recommend_similar_movies(filtered, selected_title, limit=3)
    if recommendations.empty:
        st.info("At least two titles are needed to suggest similar movies.")
    else:
        recommendation_columns = st.columns(len(recommendations))
        for column, (_, recommendation) in zip(recommendation_columns, recommendations.iterrows()):
            with column:
                with st.container(border=True):
                    st.caption(f"REVIEW SIMILARITY · {recommendation['similarity_percent']:.1f}%")
                    st.subheader(recommendation["movie_title"])
                    st.write(f"{recommendation['average_rating']:.2f} average rating")
                    st.write(f"{recommendation['recommend_rate']:.1f}% recommend")
                    st.caption(f"{recommendation['review_count']} reviews in this view")

    top_recommended = profiles.nlargest(5, "recommend_rate")[["movie_title", "recommend_rate"]]
    top_recommended = top_recommended.rename(
        columns={"movie_title": "Movie", "recommend_rate": "Recommend rate (%)"}
    )
    st.subheader("Most recommended in this view")
    st.bar_chart(top_recommended, x="Movie", y="Recommend rate (%)")

with review_tab:
    st.header("Review language lab", icon=":material/biotech:")
    review_title = st.selectbox("Choose a movie", profiles["movie_title"].tolist(), key="nlp_movie")
    title_reviews = filtered.loc[filtered["movie_title"].eq(review_title)].reset_index(drop=True)
    word_column, review_column = st.columns([1, 1.35])
    with word_column:
        st.subheader("Frequent review words")
        review_corpus = " ".join(title_reviews["user_review"].dropna().astype(str))
        if review_corpus.strip():
            cloud = WordCloud(
                width=900,
                height=520,
                background_color="#f5f6f1",
                collocations=False,
                max_words=80,
                colormap="Dark2",
            ).generate(review_corpus)
            st.image(cloud.to_array(), caption=f"{review_title}: word cloud")
        else:
            st.info("No review text is available for this movie in the current view.")
    with review_column:
        st.subheader("Analyze one review")
        if "review_analysis" not in st.session_state:
            st.session_state.review_analysis = None
        with st.form("review_analysis_form"):
            review_index = st.selectbox(
                "Review sample",
                options=title_reviews.index.tolist(),
                format_func=lambda index: (
                    f"{title_reviews.loc[index, 'sentiment']} · "
                    f"{title_reviews.loc[index, 'user_review'][:76]}"
                ),
            )
            analyze_clicked = st.form_submit_button("Run NLP analysis", icon=":material/search:")
        if analyze_clicked:
            try:
                ensure_nltk_resources()
                review_text = title_reviews.loc[review_index, "user_review"]
                st.session_state.review_analysis = {
                    "movie": review_title,
                    "text": review_text,
                    "result": analyze_text(review_text),
                }
            except (LookupError, RuntimeError) as error:
                st.warning(str(error))

        saved_analysis = st.session_state.review_analysis
        if saved_analysis and saved_analysis["movie"] == review_title:
            result = saved_analysis["result"]
            st.write(f"> {saved_analysis['text']}")
            st.badge(
                result["script_validation"]["message"],
                icon=":material/check_circle:" if result["script_validation"]["valid"] else ":material/warning:",
                color="green" if result["script_validation"]["valid"] else "orange",
            )
            st.caption("Tokens: " + " · ".join(result["filtered_tokens"]))
            st.caption("After stop-word removal: " + " · ".join(result["without_stopwords"]))
            entity_text = ", ".join(
                f"{entity['text']} ({entity['label']})" for entity in result["named_entities"]
            ) or "No named entities detected"
            st.write("**Named entities:** " + entity_text)
            with st.expander("POS tags, chunks, morphology, and n-grams"):
                st.dataframe(pd.DataFrame(result["pos_tags"]), hide_index=True)
                st.write("**Noun phrases:** " + (", ".join(result["noun_phrases"]) or "None detected"))
                st.write("**N-grams**")
                st.json(result["ngrams"])
                st.write("**Morphological forms**")
                st.dataframe(pd.DataFrame(result["morphology"]), hide_index=True)

with records_tab:
    st.header("Filtered review records", icon=":material/table_view:")
    visible_columns = [
        "movie_title",
        "genre",
        "language",
        "rating",
        "sentiment",
        "recommend",
        "user_review",
    ]
    st.dataframe(
        filtered[visible_columns].rename(
            columns={
                "movie_title": "Movie",
                "genre": "Review genre",
                "language": "Review language",
                "rating": "Rating",
                "sentiment": "Sentiment",
                "recommend": "Recommend",
                "user_review": "Review text",
            }
        ),
        hide_index=True,
    )
    st.download_button(
        "Download filtered CSV",
        data=filtered[visible_columns].to_csv(index=False).encode("utf-8"),
        file_name="filtered_movie_reviews.csv",
        mime="text/csv",
        icon=":material/download:",
    )