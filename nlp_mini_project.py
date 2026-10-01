"""NLP analysis workbench for movie-review text and document similarity."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import nltk
from nltk import RegexpParser, ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud


NLTK_RESOURCES = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet.zip": "wordnet",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "chunkers/maxent_ne_chunker_tab": "maxent_ne_chunker_tab",
    "corpora/words": "words",
}

DEMO_DOCUMENTS = {
    "document_a": (
        "The university uses natural language processing to compare student essays. "
        "The system highlights matching passages for an instructor to review."
    ),
    "document_b": (
        "Natural language processing helps the university compare student essays. "
        "Matching passages are highlighted so an instructor can review them."
    ),
    "document_c": (
        "Ada Lovelace visited London in 1843 and wrote about an early computing machine."
    ),
}


def ensure_nltk_resources(download: bool = False) -> None:
    """Check for the NLTK data packages used by tagging and lemmatization."""
    missing = []
    for resource, package in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(package)

    if download:
        for package in missing:
            nltk.download(package, quiet=True)
        missing = []
        for resource, package in NLTK_RESOURCES.items():
            try:
                nltk.data.find(resource)
            except LookupError:
                missing.append(package)

    if missing:
        package_list = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Required NLTK data is missing: "
            f"{package_list}. Run `python nlp_mini_project.py --download-data`."
        )


def tokenize(text: str) -> list[str]:
    return wordpunct_tokenize(text)


def validate_script(text: str) -> dict[str, Any]:
    script_counts: Counter[str] = Counter()
    for character in text:
        if not character.isalpha():
            continue
        unicode_name = unicodedata.name(character, "UNKNOWN")
        script = unicode_name.split()[0]
        script_counts[script] += 1

    scripts = dict(sorted(script_counts.items()))
    return {
        "valid": len(scripts) <= 1,
        "scripts": scripts,
        "message": "Single writing script detected"
        if len(scripts) <= 1
        else "Mixed writing scripts detected",
    }


def filter_tokens(tokens: list[str]) -> list[str]:
    return [token.casefold() for token in filter_surface_tokens(tokens)]


def filter_surface_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token.isalpha()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    english_stopwords = set(stopwords.words("english"))
    return [token for token in tokens if token.casefold() not in english_stopwords]


def build_ngrams(tokens: list[str], maximum_n: int = 3) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for n in range(1, maximum_n + 1):
        counts = Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
        result[str(n)] = {" ".join(gram): count for gram, count in counts.most_common(20)}
    return result


def _pluralize(word: str) -> str:
    if re.search(r"[^aeiou]y$", word):
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def generate_regular_forms(lemma: str) -> list[str]:
    """Generate common regular English forms using simple spelling rules."""
    if not lemma.isalpha():
        return []
    if lemma.endswith("ie"):
        gerund = lemma[:-2] + "ying"
    elif lemma.endswith("e") and not lemma.endswith("ee"):
        gerund = lemma[:-1] + "ing"
    else:
        gerund = lemma + "ing"
    if re.search(r"[^aeiou]y$", lemma):
        past = lemma[:-1] + "ied"
    elif lemma.endswith("e"):
        past = lemma + "d"
    else:
        past = lemma + "ed"
    return sorted({lemma, _pluralize(lemma), past, gerund})


def analyze_morphology(tokens: list[str]) -> list[dict[str, str | list[str]]]:
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    analysis = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, pos="v")
        analysis.append(
            {
                "word": token,
                "stem": stemmer.stem(token),
                "lemma": lemma,
                "generated_regular_forms": generate_regular_forms(lemma),
            }
        )
    return analysis


def tag_and_chunk(tokens: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    tagged = pos_tag(tokens)
    grammar = r"NP: {<DT>?<JJ.*>*<NN.*>+}"
    tree = RegexpParser(grammar).parse(tagged)
    chunks = []
    for subtree in tree.subtrees(lambda node: node.label() == "NP"):
        chunks.append(" ".join(word for word, _tag in subtree.leaves()))
    return tagged, chunks


def extract_named_entities(tagged: list[tuple[str, str]]) -> list[dict[str, str]]:
    tree = ne_chunk(tagged, binary=False)
    entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entities.append(
                {
                    "text": " ".join(word for word, _tag in subtree.leaves()),
                    "label": subtree.label(),
                }
            )
    return entities


def compare_documents(documents: dict[str, str]) -> dict[str, Any]:
    names = list(documents)
    if len(names) < 2:
        return {"documents": names, "matrix": [], "pairs": []}
    vectors = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(documents.values())
    matrix = cosine_similarity(vectors)
    pairs = [
        {
            "document_a": names[left],
            "document_b": names[right],
            "similarity_percent": round(float(matrix[left, right]) * 100, 2),
        }
        for left, right in combinations(range(len(names)), 2)
    ]
    return {
        "documents": names,
        "matrix": [[round(float(value), 4) for value in row] for row in matrix],
        "pairs": sorted(pairs, key=lambda pair: pair["similarity_percent"], reverse=True),
    }


def read_documents(directory: Path | None) -> dict[str, str]:
    if directory is None:
        return DEMO_DOCUMENTS.copy()
    paths = sorted(directory.glob("*.txt"))
    if len(paths) < 2:
        raise ValueError("The documents directory must contain at least two .txt files.")
    return {path.stem: path.read_text(encoding="utf-8") for path in paths}


def analyze_text(text: str) -> dict[str, Any]:
    raw_tokens = tokenize(text)
    surface_tokens = filter_surface_tokens(raw_tokens)
    filtered_tokens = filter_tokens(raw_tokens)
    tokens_without_stopwords = remove_stopwords(filtered_tokens)
    tagged, chunks = tag_and_chunk(surface_tokens)
    return {
        "script_validation": validate_script(text),
        "tokens": raw_tokens,
        "filtered_tokens": filtered_tokens,
        "without_stopwords": tokens_without_stopwords,
        "morphology": analyze_morphology(tokens_without_stopwords),
        "ngrams": build_ngrams(tokens_without_stopwords),
        "pos_tags": [{"word": word, "tag": tag} for word, tag in tagged],
        "noun_phrases": chunks,
        "named_entities": extract_named_entities(tagged),
    }


def save_wordcloud(tokens: list[str], output_path: Path) -> None:
    if not tokens:
        raise ValueError("Cannot create a word cloud because no alphabetic tokens were found.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud = WordCloud(width=1200, height=700, background_color="white", collocations=False)
    cloud.generate(" ".join(tokens)).to_file(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Text to analyze; use --text-file for longer input.")
    parser.add_argument("--text-file", type=Path, help="UTF-8 text file to analyze.")
    parser.add_argument("--documents", type=Path, help="Directory containing at least two .txt files.")
    parser.add_argument("--output", type=Path, default=Path("analysis.json"), help="JSON report path.")
    parser.add_argument("--wordcloud", type=Path, default=Path("wordcloud.png"), help="Word cloud image path.")
    parser.add_argument("--download-data", action="store_true", help="Download required NLTK data before analysis.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.text and args.text_file:
        raise SystemExit("Choose either --text or --text-file, not both.")
    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        text = args.text or "Ada Lovelace wrote about an early computing machine in London."

    try:
        ensure_nltk_resources(download=args.download_data)
        analysis = analyze_text(text)
        documents = read_documents(args.documents)
        result = {
            "application": "Movie recommendation from audience reviews",
            "analysis": analysis,
            "document_similarity": compare_documents(documents),
        }
        save_wordcloud(analysis["without_stopwords"], args.wordcloud)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except (LookupError, RuntimeError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from error

    print(f"Analysis saved to {args.output}")
    print(f"Word cloud saved to {args.wordcloud}")
    print("Document similarity (highest first):")
    for pair in result["document_similarity"]["pairs"]:
        print(
            f"  {pair['document_a']} <> {pair['document_b']}: "
            f"{pair['similarity_percent']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())