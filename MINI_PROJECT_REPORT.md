# NLP Mini Project Report

## Title

Movie Recommendation from Audience Reviews Using Natural Language Processing

## 1. Study of NLP Applications

Natural Language Processing (NLP) enables software to process human language. Common applications include machine translation, text categorization, summarization, chatbots, plagiarism screening, spelling and grammar correction, sentiment analysis, question answering, personal assistants, and tutoring systems. These applications commonly combine text normalization, linguistic analysis, and statistical or machine-learning methods.

This project chooses **movie recommendation** as its real-world application. Recommendation systems help viewers discover films from previous audience feedback. The supplied dataset contains 2,000 review records for 10 movie titles, with review text, rating, sentiment, and a recommend label.

## 2. Problem Statement

Given movie-review records, build a small recommendation system that summarizes audience response and suggests titles with similar review language. The system should also expose standard NLP analysis of a selected review. It must distinguish the review-level observations in the data from reliable movie facts: the supplied rows assign multiple genres, directors, languages, and release years to the same title.

## 3. Objectives and Scope

- Tokenize input and filter non-word tokens; report the Unicode writing scripts found.
- Remove English stop words and show stemming and lemmatization results.
- Demonstrate morphological analysis and heuristic generation of regular English forms.
- Build unigram, bigram, and trigram frequency tables.
- Perform English POS tagging, noun-phrase chunking, and named-entity recognition.
- Recommend similar movie titles by comparing their aggregated review text with TF-IDF and cosine similarity.
- Generate a word cloud for exploratory data analysis.
- Export machine-readable results to JSON for inspection and reuse.

## 4. Methodology

1. Load the CSV and validate the required movie-review fields. Coerce numeric ratings, watch times, and recommendation labels into usable values.
2. Aggregate review counts, average ratings, recommendation rates, positive-sentiment rates, and review-tag distributions by movie title.
3. Build one review-text profile per title. Use TF-IDF over words and adjacent word pairs, then cosine similarity to rank other titles by review-language overlap.
4. Read a selected review and tokenize it with NLTK's word-punctuation tokenizer.
5. Validate script consistency by counting Unicode letter-name script prefixes. Filter to alphabetic tokens and lowercase them for normalized analysis.
6. Remove English stop words. Apply the Porter stemmer and WordNet lemmatizer; generate common regular forms with transparent spelling heuristics.
7. Count 1-, 2-, and 3-grams. Apply NLTK's English POS tagger, a noun-phrase grammar, and the NLTK named-entity chunker. Generate a word cloud from review text.

## 5. Implementation and Tools

The dashboard is implemented in `app.py`, with data aggregation and review-based ranking in `movie_recommender.py`. `nlp_mini_project.py` supplies the NLP exercise pipeline. Streamlit provides the UI; pandas loads and aggregates the dataset; scikit-learn supplies TF-IDF and cosine similarity; NLTK supplies English linguistic analysis; WordCloud creates the EDA image. Install dependencies with `pip install -r requirements.txt`, then run `python nlp_mini_project.py --download-data` once to fetch NLTK data.

## 6. Evaluation

The supplied file contains 2,000 rows and 10 unique titles. Its label distribution is 1,314 recommended and 686 not recommended; sentiment is 1,011 positive, 595 neutral, and 394 negative. Average rating is 4.50 for positive rows, 2.99 for neutral rows, and 1.49 for negative rows. The dashboard provides live filtered summaries and recommendation ranks. Unit tests check title aggregation, noisy metadata distributions, CSV validation, and similarity ranking; NLP tests check preprocessing, script detection, n-grams, and text similarity.

For a meaningful evaluation on a real corpus, prepare labeled pairs (overlapping / not overlapping), select a threshold on validation data, and report precision, recall, and F1. A fixed threshold should not be treated as universal: corpus, document length, citation practices, and paraphrasing all affect scores.

## 7. Limitations and Ethics

- TF-IDF cosine similarity is a lexical baseline; it can miss semantic relationships and can over-score shared template language.
- The NLTK models are general English models and may make mistakes on domain-specific, short, or multilingual text.
- Script detection uses Unicode character names, not language identification. It flags mixed scripts but does not establish that input is valid English.
- Generated inflections are simple regular-form heuristics, not a full morphological generator; irregular forms are not covered.
- Review text and recommendation labels may be synthetic or biased. Similarity is not a measure of objective film quality or an individual's taste.
- Genre, director, year, and language values vary within each title in the supplied dataset; they are displayed only as review-row distributions.
- N-gram output is capped at the 20 most frequent entries per size in the JSON report.

## 8. Conclusion

The project demonstrates an end-to-end movie discovery workflow, from review preprocessing and sentiment summaries to TF-IDF similarity, linguistic analysis, and visualization. It is an educational prototype; recommendations reflect wording patterns in this dataset, not a guarantee that a viewer will enjoy a title.