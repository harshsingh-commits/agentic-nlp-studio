import unittest

from agents.supervisor_agent import SupervisorAgent, extract_text_payload
from graph.workflow import build_workflow


class AgentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = SupervisorAgent()
        self.known_titles = ["Interstellar", "The Dark Knight"]

    def test_supervisor_routes_movie_query_to_movie_similarity(self):
        plan = self.supervisor.plan("Find movies similar to Interstellar", self.known_titles)
        self.assertEqual(plan["selected_agents"], ["movie_similarity"])
        self.assertEqual(plan["movie_title"], "Interstellar")

    def test_review_query_selects_review_analysis_agents(self):
        plan = self.supervisor.plan("Analyze this review", self.known_titles)
        self.assertEqual(
            set(plan["selected_agents"]),
            {"nlp_analysis", "sentiment", "ner", "word_cloud"},
        )
        self.assertEqual(
            extract_text_payload("Analyze this review: A beautiful, moving story."),
            "A beautiful, moving story.",
        )

    def test_complete_report_runs_existing_modules(self):
        result = build_workflow().invoke(
            {
                "query": "Generate complete NLP report",
                "text": "Ada Lovelace wrote an insightful story about computing.",
            }
        )
        self.assertEqual(
            set(result["completed_agents"]),
            {"nlp_analysis", "sentiment", "ner", "similarity", "word_cloud"},
        )
        self.assertIn("sentiment_result", result)
        self.assertIn("analysis", result)
        self.assertIn("ner_result", result)
        self.assertIn("similarity_result", result)
        self.assertTrue(result["word_cloud_png"])
        self.assertTrue(result["insights"])

    def test_movie_similarity_graph_returns_existing_recommender_results(self):
        result = build_workflow().invoke({"query": "Find movies similar to Interstellar"})
        self.assertEqual(result["completed_agents"], ["movie_similarity"])
        self.assertTrue(result["movie_similarity_result"])
        self.assertNotIn(
            "Interstellar",
            [item["movie_title"] for item in result["movie_similarity_result"]],
        )


if __name__ == "__main__":
    unittest.main()
