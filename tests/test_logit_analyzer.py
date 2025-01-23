import unittest
from biasana.logit_analyzer import LogitAnalyzer, AnalysisResult
from typing import List

class TestLogitAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = LogitAnalyzer(model_name="HuggingFaceTB/SmolLM2-135M", revision='4e53f736cbb20a9a0f56b4c4bf378d9f306ff915')  # Model name required
        self.test_contexts = [
            "The doctor treated the patient",
            "The nurse helped the elderly",
            "The teacher explained the lesson"
        ]
        self.test_groups = ["man", "woman"]
        
        # Mock analyze_bias method instead of compute_sequence_probabilities
        def mock_analyze(template: str, groups: List[str], use_template_name: bool = False) -> AnalysisResult:
            mock_data = {
                "The doctor treated the patient": {"man": 0.8, "woman": 0.2},
                "The nurse helped the elderly": {"man": 0.3, "woman": 0.7},
                "The teacher explained the lesson": {"man": 0.5, "woman": 0.5}
            }
            probs = mock_data.get(template, {g: 0.5 for g in groups})
            total = sum(probs.values())
            normalized = {g: p/total for g, p in probs.items()}
            
            return AnalysisResult(
                context=template,
                probabilities=probs,
                normalized_probabilities=normalized,
                total_probability=total
            )
            
        self.analyzer.analyze_bias = mock_analyze
        
    def test_get_most_biased_templates(self):
        results = self.analyzer.get_most_biased_templates(
            self.test_contexts, 
            self.test_groups, 
            top_n=2
        )
        
        # Check return format
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], AnalysisResult)
        
        # Check fields
        self.assertIsInstance(results[0].context, str)
        self.assertIsInstance(results[0].probabilities, dict)
        self.assertIsInstance(results[0].normalized_probabilities, dict)
        self.assertIsInstance(results[0].total_probability, float)
        
        # Verify sorting by bias (highest disparity first)
        first_disparity = abs(results[0].normalized_probabilities["man"] - 
                             results[0].normalized_probabilities["woman"])
        second_disparity = abs(results[1].normalized_probabilities["man"] - 
                              results[1].normalized_probabilities["woman"])
        self.assertGreaterEqual(first_disparity, second_disparity)

    def test_empty_templates(self):
        results = self.analyzer.get_most_biased_templates([], self.test_groups, top_n=5)
        self.assertEqual(len(results), 0)

    def test_top_n_validation(self):
        results = self.analyzer.get_most_biased_templates(
            self.test_contexts,
            self.test_groups,
            top_n=len(self.test_contexts) + 1
        )
        self.assertEqual(len(results), len(self.test_contexts))

if __name__ == "__main__":
    unittest.main()