"""LangGraph workflow connecting the project's NLP modules."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from wordcloud import WordCloud

from agents.insight_agent import InsightAgent
from agents.supervisor_agent import SupervisorAgent, extract_text_payload
from movie_recommender import build_movie_profiles, load_movie_data, recommend_similar_movies
from nlp_mini_project import (
    analyze_text,
    compare_documents,
    ensure_nltk_resources,
    extract_named_entities,
    filter_surface_tokens,
    tag_and_chunk,
    tokenize,
)


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "movie_recommendation_nlp_dataset.csv"
AGENT_NODES = (
    "nlp_analysis",
    "sentiment",
    "ner",
    "similarity",
    "movie_similarity",
    "word_cloud",
)


class AnalystState(TypedDict, total=False):
    query: str
    text: str
    movie_title: str | None
    known_titles: list[str]
    selected_agents: list[str]
    completed_agents: list[str]
    routing_mode: str
    routing_note: str
    analysis: dict[str, Any]
    sentiment_result: dict[str, Any]
    ner_result: list[dict[str, str]]
    similarity_result: list[dict[str, Any]]
    movie_similarity_result: list[dict[str, Any]]
    word_cloud_png: bytes
    agent_errors: dict[str, str]
    agent_status: dict[str, str]
    insights: str
    insight_mode: str
    keywords: list[str]


@lru_cache(maxsize=4)
def _load_reviews(dataset_path: str) -> pd.DataFrame:
    return load_movie_data(Path(dataset_path))


@lru_cache(maxsize=4)
def _load_sentiment_classifier(dataset_path: str) -> Any:
    reviews = _load_reviews(dataset_path)
    training_rows = reviews.loc[
        reviews["user_review"].str.strip().ne("") & reviews["sentiment"].str.strip().ne("")
    ]
    if training_rows["sentiment"].nunique() < 2:
        raise ValueError("At least two labeled sentiment classes are required in the review dataset.")
    classifier = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7),
    )
    classifier.fit(training_rows["user_review"], training_rows["sentiment"])
    return classifier


def _append_completion(state: AnalystState, agent_name: str, updates: dict[str, Any]) -> dict[str, Any]:
    completed = list(state.get("completed_agents", []))
    if agent_name not in completed:
        completed.append(agent_name)
    status = dict(state.get("agent_status", {}))
    status[agent_name] = "complete"
    return {**updates, "completed_agents": completed, "agent_status": status}


def _record_error(state: AnalystState, agent_name: str, error: Exception) -> dict[str, Any]:
    errors = dict(state.get("agent_errors", {}))
    errors[agent_name] = f"{type(error).__name__}: {str(error)[:240]}"
    status = dict(state.get("agent_status", {}))
    status[agent_name] = "failed"
    return {"agent_errors": errors, "agent_status": status}


def _require_text(state: AnalystState, agent_name: str) -> str | None:
    text = str(state.get("text", "")).strip()
    if text:
        return text
    errors = dict(state.get("agent_errors", {}))
    errors[agent_name] = "Add review text in the sidebar to run this module."
    state["agent_errors"] = errors
    return None


def build_workflow(
    llm: Any | None = None,
    dataset_path: str | Path = DEFAULT_DATASET,
):
    """Compile the supervisor -> selected agents -> insight LangGraph."""
    resolved_dataset = str(Path(dataset_path).resolve())
    reviews = _load_reviews(resolved_dataset)
    known_titles = sorted(reviews["movie_title"].dropna().unique().tolist())
    supervisor = SupervisorAgent(llm)
    insight_agent = InsightAgent(llm)

    def supervisor_node(state: AnalystState) -> dict[str, Any]:
        text = str(state.get("text", "")).strip() or extract_text_payload(state.get("query", ""))
        plan = supervisor.plan(state.get("query", ""), known_titles)
        return {
            **plan,
            "known_titles": known_titles,
            "text": text,
            "completed_agents": [],
            "agent_errors": {},
            "agent_status": {},
        }

    def nlp_analysis_node(state: AnalystState) -> dict[str, Any]:
        agent_name = "nlp_analysis"
        text = _require_text(state, agent_name)
        if text is None:
            return _append_completion(state, agent_name, {})
        try:
            ensure_nltk_resources()
            analysis = analyze_text(text)
            return _append_completion(state, agent_name, {"analysis": analysis})
        except Exception as error:
            return {
                **_record_error(state, agent_name, error),
                **_append_completion(state, agent_name, {}),
            }

    def sentiment_node(state: AnalystState) -> dict[str, Any]:
        agent_name = "sentiment"
        text = _require_text(state, agent_name)
        if text is None:
            return _append_completion(state, agent_name, {})
        try:
            classifier = _load_sentiment_classifier(resolved_dataset)
            probabilities = classifier.predict_proba([text])[0]
            classes = classifier.classes_
            best_index = int(probabilities.argmax())
            return _append_completion(
                state,
                agent_name,
                {
                    "sentiment_result": {
                        "label": str(classes[best_index]),
                        "confidence": float(probabilities[best_index]),
                        "source": "TF-IDF classifier trained on the dataset's sentiment labels",
                    }
                },
            )
        except Exception as error:
            return {
                **_record_error(state, agent_name, error),
                **_append_completion(state, agent_name, {}),
            }

    def ner_node(state: AnalystState) -> dict[str, Any]:
        agent_name = "ner"
        text = _require_text(state, agent_name)
        if text is None:
            return _append_completion(state, agent_name, {})
        try:
            ensure_nltk_resources()
            if state.get("analysis"):
                entities = state["analysis"].get("named_entities", [])
            else:
                tagged, _chunks = tag_and_chunk(filter_surface_tokens(tokenize(text)))
                entities = extract_named_entities(tagged)
            return _append_completion(state, agent_name, {"ner_result": entities})
        except Exception as error:
            return {
                **_record_error(state, agent_name, error),
                **_append_completion(state, agent_name, {}),
            }

    def similarity_node(state: AnalystState) -> dict[str, Any]:
        agent_name = "similarity"
        text = _require_text(state, agent_name)
        if text is None:
            return _append_completion(state, agent_name, {})
        try:
            profiles = build_movie_profiles(reviews)
            documents = {"query_text": text}
            documents.update(
                {
                    f"movie_review_profile: {row.movie_title}": row.review_profile
                    for row in profiles.itertuples(index=False)
                }
            )
            compared = compare_documents(documents)
            matches = [
                {
                    "document_b": pair["document_b"].split(": ", maxsplit=1)[-1],
                    "similarity_percent": pair["similarity_percent"],
                }
                for pair in compared["pairs"]
                if pair["document_a"] == "query_text"
            ]
            return _append_completion(state, agent_name, {"similarity_result": matches[:5]})
        except Exception as error:
            return {
                **_record_error(state, agent_name, error),
                **_append_completion(state, agent_name, {}),
            }

    def movie_similarity_node(state: AnalystState) -> dict[str, Any]:
        agent_name = "movie_similarity"
        title = state.get("movie_title")
        if not title:
            errors = dict(state.get("agent_errors", {}))
            errors[agent_name] = "Mention a movie title from the catalog, such as Interstellar."
            state["agent_errors"] = errors
            return _append_completion(state, agent_name, {})
        try:
            recommendations = recommend_similar_movies(reviews, title, limit=5)
            result = recommendations[
                ["movie_title", "similarity_percent", "review_count", "average_rating", "recommend_rate"]
            ].to_dict(orient="records")
            return _append_completion(state, agent_name, {"movie_similarity_result": result})
        except Exception as error:
            return {
                **_record_error(state, agent_name, error),
                **_append_completion(state, agent_name, {}),
            }

    def word_cloud_node(state: AnalystState) -> dict[str, Any]:
        agent_name = "word_cloud"
        text = _require_text(state, agent_name)
        if text is None:
            return _append_completion(state, agent_name, {})
        try:
            image = WordCloud(width=900, height=500, background_color="white", collocations=False)
            image.generate(text)
            buffer = BytesIO()
            image.to_image().save(buffer, format="PNG")
            return _append_completion(state, agent_name, {"word_cloud_png": buffer.getvalue()})
        except Exception as error:
            return {
                **_record_error(state, agent_name, error),
                **_append_completion(state, agent_name, {}),
            }

    def insight_node(state: AnalystState) -> dict[str, Any]:
        result = insight_agent.generate(state.get("query", ""), state)
        analysis = state.get("analysis", {})
        result["keywords"] = [
            token
            for token, _count in Counter(analysis.get("without_stopwords", [])).most_common(8)
        ]
        return result

    def route_next(state: AnalystState) -> str:
        completed = set(state.get("completed_agents", []))
        for agent_name in state.get("selected_agents", []):
            if agent_name in AGENT_NODES and agent_name not in completed:
                return agent_name
        return "insight_agent"

    builder = StateGraph(AnalystState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("nlp_analysis", nlp_analysis_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("ner", ner_node)
    builder.add_node("similarity", similarity_node)
    builder.add_node("movie_similarity", movie_similarity_node)
    builder.add_node("word_cloud", word_cloud_node)
    builder.add_node("insight_agent", insight_node)
    builder.add_edge(START, "supervisor")

    route_map = {agent_name: agent_name for agent_name in AGENT_NODES}
    route_map["insight_agent"] = "insight_agent"
    builder.add_conditional_edges("supervisor", route_next, route_map)
    for agent_name in AGENT_NODES:
        builder.add_conditional_edges(agent_name, route_next, route_map)
    builder.add_edge("insight_agent", END)
    return builder.compile()
