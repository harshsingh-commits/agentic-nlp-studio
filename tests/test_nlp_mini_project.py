import unittest

from nlp_mini_project import (
    build_ngrams,
    compare_documents,
    filter_tokens,
    filter_surface_tokens,
    generate_regular_forms,
    tokenize,
    validate_script,
)


class PreprocessingTests(unittest.TestCase):
    def test_tokenization_and_filtering(self):
        tokens = tokenize("Hello, NLP 2026!")
        self.assertEqual(filter_tokens(tokens), ["hello", "nlp"])
        self.assertEqual(filter_surface_tokens(tokens), ["Hello", "NLP"])

    def test_script_validation_detects_mixed_scripts(self):
        self.assertTrue(validate_script("Hello world")['valid'])
        self.assertFalse(validate_script("Hello Привет")['valid'])

    def test_ngram_counts(self):
        ngrams = build_ngrams(["a", "b", "a"], maximum_n=2)
        self.assertEqual(ngrams["1"]["a"], 2)
        self.assertEqual(ngrams["2"]["a b"], 1)

    def test_regular_form_generation(self):
        self.assertCountEqual(generate_regular_forms("study"), ["studies", "studied", "studying", "study"])

    def test_document_similarity_ranks_related_pair_first(self):
        documents = {
            "first": "The cat sat on the mat.",
            "related": "The cat sat on a mat.",
            "unrelated": "Space telescopes observe distant galaxies.",
        }
        result = compare_documents(documents)
        self.assertEqual((result["pairs"][0]["document_a"], result["pairs"][0]["document_b"]), ("first", "related"))


if __name__ == "__main__":
    unittest.main()