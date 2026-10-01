"""Chat page for the LangGraph-backed NLP analyst."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from graph.workflow import DEFAULT_DATASET, build_workflow
from movie_recommender import load_movie_data
from agents.supervisor_agent import create_chat_model


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon=":material/psychology:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: str) -> pd.DataFrame:
    return load_movie_data(Path(dataset_path))


def configured_secret(provider: str) -> str:
    key_name = "GOOGLE_API_KEY" if provider == "Gemini" else "GROQ_API_KEY"
    environment_value = os.getenv(key_name, "")
    if environment_value:
        return environment_value
    try:
        return str(st.secrets.get(key_name, ""))
    except Exception:
        return ""


def render_agent_result(result: dict[str, Any]) -> None:
    if result.get("sentiment_result"):
        sentiment = result["sentiment_result"]
        st.badge(
            f"{sentiment['label']} · {sentiment['confidence']:.0%}",
            icon=":material/psychology:",
            color="green" if sentiment["label"].casefold() == "positive" else "orange",
        )
        st.caption(sentiment["source"])

    if result.get("keywords"):
        st.write("**Keywords:** " + " · ".join(result["keywords"]))

    if result.get("ner_result"):
        st.write("**Named entities**")
        st.dataframe(pd.DataFrame(result["ner_result"]), hide_index=True)
    elif "ner" in result.get("completed_agents", []):
        st.caption("No named entities detected.")

    if result.get("movie_similarity_result"):
        st.write("**Similar movies**")
        movie_rows = pd.DataFrame(result["movie_similarity_result"]).rename(
            columns={
                "movie_title": "Movie",
                "similarity_percent": "Review similarity (%)",
                "review_count": "Reviews",
                "average_rating": "Average rating",
                "recommend_rate": "Recommend rate (%)",
            }
        )
        st.dataframe(movie_rows, hide_index=True)

    if result.get("similarity_result"):
        st.write("**Closest review profiles**")
        st.dataframe(
            pd.DataFrame(result["similarity_result"]).rename(
                columns={
                    "document_b": "Movie review profile",
                    "similarity_percent": "Text similarity (%)",
                }
            ),
            hide_index=True,
        )

    if result.get("word_cloud_png"):
        st.image(result["word_cloud_png"], caption="Word cloud from the input text")

    if result.get("analysis"):
        analysis = result["analysis"]
        with st.expander("Preprocessing and linguistic analysis"):
            st.caption("Tokens: " + " · ".join(analysis["tokens"]))
            st.caption("Filtered: " + " · ".join(analysis["filtered_tokens"]))
            st.caption("Without stop words: " + " · ".join(analysis["without_stopwords"]))
            st.write("**Script validation:** " + analysis["script_validation"]["message"])
            st.write("**Noun phrases:** " + (", ".join(analysis["noun_phrases"]) or "None detected"))
            st.write("**N-grams**")
            st.json(analysis["ngrams"])
            st.write("**POS tags**")
            st.dataframe(pd.DataFrame(analysis["pos_tags"]), hide_index=True)
            st.write("**Morphology**")
            st.dataframe(pd.DataFrame(analysis["morphology"]), hide_index=True)

    errors = result.get("agent_errors", {})
    if errors:
        with st.expander("Modules needing attention"):
            for module_name, message in errors.items():
                st.warning(f"{module_name}: {message}")

    completed = result.get("completed_agents", [])
    if completed:
        st.caption("Modules run: " + " → ".join(completed) + " → insights")
    st.caption(f"Supervisor: {result.get('routing_mode', 'local')} · Insight: {result.get('insight_mode', 'local')}")


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("result"):
            render_agent_result(message["result"])


try:
    reviews = load_dataset(str(DEFAULT_DATASET))
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load the movie review dataset: {error}")
    st.stop()

with st.sidebar:
    st.title("AI Agent Assistant", icon=":material/psychology:")
    st.caption("A LangGraph supervisor routes requests to the NLP modules.")
    source_mode = st.selectbox("Text source", ["Paste text", "Choose a dataset review"])
    if source_mode == "Paste text":
        input_text = st.text_area(
            "Review or document",
            placeholder="Paste a review here for NLP analysis and text similarity.",
            height=150,
        )
    else:
        title_options = sorted(reviews["movie_title"].unique().tolist())
        source_title = st.selectbox("Movie title", title_options)
        title_rows = reviews.loc[reviews["movie_title"].eq(source_title)].reset_index(drop=True)
        source_review_index = st.selectbox(
            "Review sample",
            options=title_rows.index.tolist(),
            format_func=lambda index: (
                f"{title_rows.loc[index, 'sentiment']} · "
                f"{title_rows.loc[index, 'user_review'][:66]}"
            ),
        )
        input_text = str(title_rows.loc[source_review_index, "user_review"])

    provider = st.selectbox("Chat model", ["Gemini", "Groq"])
    default_models = {"Gemini": "gemini-2.5-flash", "Groq": "llama-3.3-70b-versatile"}
    model_name = st.text_input("Model name", value=default_models[provider])
    api_key = st.text_input(
        "Provider API key",
        value=configured_secret(provider),
        type="password",
        help="Read from the project .env file, matching environment variable, or Streamlit secrets.",
    )
    if not api_key:
        st.caption("No API key: local routing, dataset sentiment, and local insights remain available.")
    st.caption("Dataset-trained sentiment is a baseline, not a definitive judgment.")
    if st.button("Clear chat", icon=":material/delete:"):
        st.session_state.agent_assistant_messages = []
        st.rerun()

st.title("AI Agent Assistant", icon=":material/psychology:")
st.caption("Ask for a review analysis, a complete NLP report, or similar movies from the catalog.")
st.caption("Example: “Analyze this review” · “Find movies similar to Interstellar” · “Generate complete NLP report”")

if "agent_assistant_messages" not in st.session_state:
    st.session_state.agent_assistant_messages = []

for saved_message in st.session_state.agent_assistant_messages:
    render_message(saved_message)

prompt = st.chat_input("Ask the NLP analyst…", submit_mode="disable")
if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.agent_assistant_messages.append(user_message)
    render_message(user_message)

    llm = None
    provider_note = ""
    if api_key.strip():
        try:
            llm = create_chat_model(provider, model_name.strip(), api_key.strip())
        except Exception as error:
            provider_note = f"Could not configure {provider}: {type(error).__name__}. Using local fallbacks."

    try:
        workflow = build_workflow(llm=llm)
        initial_state = {"query": prompt, "text": input_text.strip()}
        with st.chat_message("assistant"):
            with st.status("Supervisor is planning the analysis", expanded=True) as status:
                result = workflow.invoke(initial_state)
                status.write("Completed: " + (", ".join(result.get("completed_agents", [])) or "no analysis modules"))
                status.update(label="Analysis complete", state="complete")
            answer = result.get("insights", "The workflow completed without a summary.")
            if provider_note:
                st.caption(provider_note)
            st.markdown(answer)
            render_agent_result(result)
        st.session_state.agent_assistant_messages.append(
            {"role": "assistant", "content": answer, "result": result}
        )
    except Exception as error:
        safe_message = f"The analyst workflow could not complete: {type(error).__name__}: {error}"
        with st.chat_message("assistant"):
            st.error(safe_message)
        st.session_state.agent_assistant_messages.append(
            {"role": "assistant", "content": safe_message}
        )
