# NLP Mini Project

This project explores movie recommendations from the supplied review dataset. Its Streamlit frontend summarizes review ratings and sentiment, recommends similar titles from review language, and exposes NLP analysis for an individual review. The standalone NLP workbench also demonstrates the assignment's preprocessing and text-similarity techniques.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python nlp_mini_project.py --download-data
```

The NLTK command downloads the POS tagger, named-entity chunker, English stop-word list, and WordNet data needed by the linguistic analysis. It also runs the sample text workbench once.

## Frontend

Start the movie dashboard from the project directory:

```powershell
streamlit run app.py
```

The dashboard loads `movie_recommendation_nlp_dataset.csv` by default. Use the sidebar filters to explore review-level genre, language, sentiment, rating, and text. The Discover view ranks titles by TF-IDF cosine similarity over review text. Review NLP includes a word cloud and token, POS, chunk, named-entity, morphology, and n-gram analysis. An alternate CSV can be uploaded if it has the same required columns.

The source data contains multiple genre, director, language, and year values for the same title. The dashboard therefore shows these as distributions across review records rather than treating them as definitive movie metadata.

## AI Agent Assistant

The LangGraph chat page is available from Streamlit's page navigation after starting the app. It routes review-analysis, full-report, text-similarity, and movie-similarity requests to the existing NLP and recommendation functions. Sentiment predictions use a TF-IDF classifier trained from the dataset's existing sentiment labels.

For model-based supervision and insight generation, set `GOOGLE_API_KEY` or `GROQ_API_KEY` before starting Streamlit, or enter a key in the page sidebar. Without a key, local query routing and result synthesis remain available.

The project-root `.env` file is ready for a key: set `GOOGLE_API_KEY=your_key` or `GROQ_API_KEY=your_key`. The assistant loads this file automatically; shell environment variables take precedence. `.env.example` is a template, and the real `.env` is excluded from Git.

## Run

Run the built-in example:

```powershell
python nlp_mini_project.py
```

Analyze a sentence or UTF-8 file and compare your own documents:

```powershell
python nlp_mini_project.py --text "Ada Lovelace wrote about a computing machine in London."
python nlp_mini_project.py --text-file input.txt --documents documents
```

The documents directory must contain at least two `.txt` files. The program writes `analysis.json` and `wordcloud.png` by default. Use `--output` and `--wordcloud` to change those paths. JSON includes tokenization, script validation, filtering, stop-word removal, stemming/lemmatization, heuristic regular-word forms, 1- to 3-grams, POS tags, noun phrases, named entities, and a pairwise similarity matrix.

## Test

```powershell
python -m unittest discover -s tests -v
```

See [MINI_PROJECT_REPORT.md](MINI_PROJECT_REPORT.md) for the application study, problem statement, methodology, results interpretation, and limitations.