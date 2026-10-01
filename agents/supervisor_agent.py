"""Supervisor routing for the NLP analyst agent."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


AGENT_NAMES = [
    "nlp_analysis",
    "sentiment",
    "ner",
    "similarity",
    "movie_similarity",
    "word_cloud",
]


class RoutingDecision(BaseModel):
    """Constrained task plan returned by the optional chat model."""

    agents: list[str] = Field(description="Ordered list of agent names to run")
    movie_title: str | None = Field(default=None, description="A mentioned title from the provided catalog")
    reason: str = Field(default="", description="Brief routing summary")


def create_chat_model(provider: str, model_name: str, api_key: str) -> Any:
    """Create the configured LangChain chat model without requiring provider imports otherwise."""
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0)
    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, api_key=api_key, temperature=0)
    raise ValueError(f"Unsupported model provider: {provider}")


def find_movie_title(query: str, known_titles: list[str]) -> str | None:
    query_folded = query.casefold()
    candidates = sorted(known_titles, key=len, reverse=True)
    for title in candidates:
        pattern = rf"(?<!\w){re.escape(title)}(?!\w)"
        if re.search(pattern, query_folded, flags=re.IGNORECASE):
            return title
    return None


def extract_text_payload(query: str) -> str:
    """Extract text explicitly introduced with review/text/document punctuation."""
    match = re.search(
        r"\b(?:review|text|document)\s*[:=\-]\s*[\"'“]?(.+?)[\"'”]?\s*$",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip().strip("\"'“”")
    quoted = re.search(r"[\"“](.+?)[\"”]", query, flags=re.DOTALL)
    return quoted.group(1).strip() if quoted else ""


def _fallback_plan(query: str, known_titles: list[str]) -> RoutingDecision:
    normalized = query.casefold()
    title = find_movie_title(query, known_titles)
    full_report = any(
        phrase in normalized
        for phrase in ("complete", "full report", "all modules", "everything", "comprehensive")
    )
    movie_request = any(
        phrase in normalized
        for phrase in (
            "similar movie",
            "similar movies",
            "movies similar",
            "movie similar",
            "movies like",
            "movie similarity",
            "recommend movie",
        )
    )

    if full_report:
        agents = ["nlp_analysis", "sentiment", "ner", "similarity", "word_cloud"]
        if title:
            agents.append("movie_similarity")
        return RoutingDecision(agents=agents, movie_title=title, reason="Complete NLP report requested")

    if movie_request:
        task = "movie_similarity" if title else "similarity"
        return RoutingDecision(agents=[task], movie_title=title, reason="Movie similarity request")

    agents = []
    if any(word in normalized for word in ("analy", "review", "token", "keyword", "nlp", "report")):
        agents.append("nlp_analysis")
    if any(word in normalized for word in ("sentiment", "opinion", "positive", "negative")):
        agents.append("sentiment")
    if any(word in normalized for word in ("entity", "entities", "ner", "person", "place", "organization")):
        agents.append("ner")
    if any(word in normalized for word in ("similar", "compare", "plagiar", "overlap")):
        agents.append("similarity")
    if "word cloud" in normalized or "wordcloud" in normalized:
        agents.append("word_cloud")

    if "analyze" in normalized or "review" in normalized:
        agents.extend(["sentiment", "ner"])
        if title:
            agents.append("movie_similarity")
        agents.append("word_cloud")

    unique_agents = list(dict.fromkeys(agent for agent in agents if agent in AGENT_NAMES))
    if not unique_agents:
        unique_agents = ["nlp_analysis", "sentiment", "ner"]
    return RoutingDecision(agents=unique_agents, movie_title=title, reason="Keyword-based local routing")


class SupervisorAgent:
    """Plan a small, validated agent workflow with local fallback routing."""

    def __init__(self, llm: Any | None = None):
        self.llm = llm

    def plan(self, query: str, known_titles: list[str]) -> dict[str, Any]:
        fallback = _fallback_plan(query, known_titles)
        if self.llm is None:
            return {
                "selected_agents": fallback.agents,
                "movie_title": fallback.movie_title,
                "routing_mode": "local rules",
                "routing_note": fallback.reason,
            }

        prompt = (
            "Route the user's NLP request to the smallest sufficient ordered set of agents. "
            "Allowed agents: " + ", ".join(AGENT_NAMES) + ". "
            "For 'analyze this review', include nlp_analysis, sentiment, ner, and word_cloud. "
            "For a complete NLP report, include nlp_analysis, sentiment, ner, similarity, and word_cloud. "
            "For a movie recommendation request, use movie_similarity and select a title only from the catalog. "
            "Use similarity for text-to-review comparison. Do not invent titles."
        )
        try:
            structured_model = self.llm.with_structured_output(RoutingDecision)
            decision = structured_model.invoke(
                [
                    ("system", prompt),
                    (
                        "human",
                        f"Catalog: {known_titles}\nUser query: {query}",
                    ),
                ]
            )
            selected_agents = list(
                dict.fromkeys(agent for agent in decision.agents if agent in AGENT_NAMES)
            )
            if not selected_agents:
                selected_agents = fallback.agents
            selected_title = decision.movie_title
            if selected_title not in known_titles:
                selected_title = fallback.movie_title
            return {
                "selected_agents": selected_agents,
                "movie_title": selected_title,
                "routing_mode": "LangChain model",
                "routing_note": decision.reason or fallback.reason,
            }
        except Exception as error:
            return {
                "selected_agents": fallback.agents,
                "movie_title": fallback.movie_title,
                "routing_mode": "local rules (model fallback)",
                "routing_note": f"{fallback.reason}. Model routing unavailable: {error}",
            }
