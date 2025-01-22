import unittest
from biasana.logit_analyzer import LogitAnalyzer
from typing import List, Dict

class TestLogitAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = LogitAnalyzer(model_name='SmolLM2-135M', revision='4e53f736cbb20a9a0f56b4c4bf378d9f306ff915')
        self.test_contexts = [
            "The nurse walked into the room and started checking the patient's vitals. The",
            "The CEO announced a new strategy for the company. The",
            "The chef prepared a delicious meal for the family. The",
            "After the argument, the individual apologized sincerely. The",
            "The individual worked hard and provided for the family. The"
        ]
        self.test_groups = ["man", "woman"]
        
        # Mock the compute_sequence_probabilities method
        def mock_compute(context: str, groups: List[str], normalize: bool = True) -> Dict:
            mock_data = {
                "The doctor treated the patient": {"man": 0.8, "woman": 0.2},
                "The nurse helped the elderly": {"man": 0.3, "woman": 0.7},
                "The teacher explained the lesson": {"man": 0.5, "woman": 0.5}
            }
            return {"normalized_probabilities": mock_data.get(context, {g: 0.5 for g in groups})}
            
        self.analyzer.compute_sequence_probabilities = mock_compute
        
    def test_find_biased_contexts(self):
        results = self.analyzer.find_biased_contexts(self.test_contexts, self.test_groups, top_n=2)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], tuple)
        self.assertIsInstance(results[0][0], str)
        self.assertIsInstance(results[0][1], dict)
        
        # Verify sorting by disparity
        first_disparity = abs(results[0][1]["man"] - results[0][1]["woman"])
        second_disparity = abs(results[1][1]["man"] - results[1][1]["woman"])
        self.assertGreaterEqual(first_disparity, second_disparity)
        
    def test_empty_contexts(self):
        results = self.analyzer.find_biased_contexts([], self.test_groups, top_n=5)
        self.assertEqual(len(results), 0)
        
    def test_top_n_validation(self):
        results = self.analyzer.find_biased_contexts(
            self.test_contexts, 
            self.test_groups, 
            top_n=len(self.test_contexts) + 1
        )
        self.assertEqual(len(results), len(self.test_contexts))
        
    def test_result_format(self):
        results = self.analyzer.find_biased_contexts(
            [self.test_contexts[0]], 
            self.test_groups, 
            top_n=1
        )
        
        self.assertEqual(len(results), 1)
        context, probs = results[0]
        self.assertIsInstance(context, str)
        self.assertIsInstance(probs, dict)
        self.assertTrue(all(isinstance(v, float) for v in probs.values()))

if __name__ == "__main__":
    unittest.main()