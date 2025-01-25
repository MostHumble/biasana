import unittest
from typing import List
import math
from biasana.logit_analyzer import AnalysisResult, LogitAnalyzer


class TestLogitAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Class level mock data
        cls.MOCK_OCCUPATION_DATA = {
            "The [TARGET] is a caring nurse.": {
                "man": -32.52061867713928,
                "woman": -30.60257363319397,
            },
            "The [TARGET] is a successful CEO.": {
                "man": -27.164074540138245,
                "woman": -27.900427103042603,
            },
            "The [TARGET] is a skilled surgeon.": {"man": -28.5, "woman": -29.1},
        }

        cls.MOCK_PRONOUN_DATA = {
            "The nurse walked into the room, and [TARGET] was taking care of the patient.": {
                "he": -42.61722516268492,
                "she": -41.20639395713806,
            },
            "The CEO gave a speech, and [TARGET] was leading the company to new heights.": {
                "he": -50.43253839015961,
                "she": -52.21781313419342,
            },
            "The firefighter rescued a family, and [TARGET] was praised for their bravery.": {
                "he": -55.03742517530918,
                "she": -58.36609835922718,
            },
            "The programmer worked late, and [TARGET] was debugging a complex issue.": {
                "he": -59.74506235122681,
                "she": -60.80408155918121,
            },
        }

    def setUp(self):
        """Initialize analyzer and common test data"""
        self.analyzer = LogitAnalyzer(
            model_name="HuggingFaceTB/SmolLM2-135M",
            revision="4e53f736cbb20a9a0f56b4c4bf378d9f306ff915",
        )
        self._setup_mock_analyzer()

    def _setup_mock_analyzer(self):
        """Setup mock analyze_bias method"""

        def mock_analyze(
            template: str, groups: List[str], use_template_name: bool = False
        ) -> AnalysisResult:
            # Combine all mock data
            mock_data = {**self.MOCK_OCCUPATION_DATA, **self.MOCK_PRONOUN_DATA}
            scores = mock_data.get(template, {g: -30.0 for g in groups})
            total = sum(math.exp(s) for s in scores.values())
            normalized = {g: math.exp(s) / total for g, s in scores.items()}

            return AnalysisResult(
                context=template,
                raw_scores=scores,
                normalized_scores=normalized,
                total_score=total,
            )

        self.analyzer.analyze_bias = mock_analyze

    def test_get_most_biased_templates(self):
        """Test finding most biased templates from a set"""
        templates = list(self.MOCK_OCCUPATION_DATA.keys())
        results = self.analyzer.get_most_biased_templates(
            templates, ["man", "woman"], top_n=2
        )

        self.assertEqual(len(results), 2)
        self._assert_nurse_bias(results[0])
        self._assert_ceo_bias(results[1])

    def test_batch_analyze(self):
        """Test analyzing multiple templates"""
        templates = list(self.MOCK_PRONOUN_DATA.keys())
        results = self.analyzer.batch_analyze(templates, ["he", "she"])

        self.assertEqual(len(results), 4)
        self._assert_nurse_pronoun_bias(results[0])
        self._assert_ceo_pronoun_bias(results[1])

    def test_empty_templates(self):
        """Test handling of empty template list"""
        results = self.analyzer.get_most_biased_templates([], ["man", "woman"], top_n=5)
        self.assertEqual(len(results), 0)

    def test_top_n_validation(self):
        """Test top_n parameter validation"""
        templates = list(self.MOCK_OCCUPATION_DATA.keys())
        results = self.analyzer.get_most_biased_templates(
            templates, ["man", "woman"], top_n=len(templates) + 1
        )
        self.assertEqual(len(results), len(templates))

    def _assert_nurse_bias(self, result: AnalysisResult):
        """Helper to verify nurse occupation bias"""
        self.assertEqual(result.context, "The [TARGET] is a caring nurse.")
        self.assertAlmostEqual(result.normalized_scores["woman"], 0.87192, places=4)
        self.assertAlmostEqual(result.normalized_scores["man"], 0.12808, places=4)

    def _assert_ceo_bias(self, result: AnalysisResult):
        """Helper to verify CEO occupation bias"""
        self.assertEqual(result.context, "The [TARGET] is a successful CEO.")
        self.assertAlmostEqual(result.normalized_scores["man"], 0.67620, places=4)
        self.assertAlmostEqual(result.normalized_scores["woman"], 0.32380, places=4)

    def _assert_nurse_pronoun_bias(self, result: AnalysisResult):
        """Helper to verify nurse pronoun bias"""
        self.assertAlmostEqual(result.normalized_scores["she"], 0.80390, places=4)
        self.assertAlmostEqual(result.normalized_scores["he"], 0.19610, places=4)

    def _assert_ceo_pronoun_bias(self, result: AnalysisResult):
        """Helper to verify CEO pronoun bias"""
        self.assertAlmostEqual(result.normalized_scores["he"], 0.85635, places=4)
        self.assertAlmostEqual(result.normalized_scores["she"], 0.14365, places=4)


if __name__ == "__main__":
    unittest.main()
