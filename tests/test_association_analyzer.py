import unittest
from biasana.association_analyzer import AssociationAnalyzer
import numpy as np
import logging

class TestAssociationAnalyzer(unittest.TestCase):
    
    def setUp(self):
        # Initialize the analyzer and test data
        self.analyzer = AssociationAnalyzer(min_df=1)
        self.documents = [
            "The woman worked hard as a scientist and discovered a groundbreaking cure for a disease.",
            "A young woman worked as an artist and painted a beautiful sunset on the canvas.",
            "The woman entrepreneur launched a successful tech startup.",
            "The man firefighter saved a family from the burning building but lost the dog.",
            "A man carpenter built a wooden table using traditional tools but lost his fingers.",
            "The man athlete won the gold medal in the 100m sprint.",
            "A team of engineers, both men and women, collaborated on designing the new software.",
            "The woman teacher inspired her students to pursue science and mathematics.",
            "A man and a woman co-authored a best-selling novel.",
            "The man and woman worked together to organize the community event.",
        ]
        self.subgroup_terms = ["woman", "man"]

    def test_preprocess_text(self):
        # Test text preprocessing
        processed = self.analyzer.preprocess_text(self.documents)
        self.assertEqual(len(processed), len(self.documents))
        self.assertTrue(all(isinstance(doc, str) for doc in processed))
        self.assertTrue('woman work hard scientist discover groundbreaking cure disease .' == processed[0])

    def test_fit_transform(self):
        # Test TF-IDF fitting and transformation
        tfidf_matrix, feature_names, processed_docs = self.analyzer.fit_transform(self.documents)
        self.assertEqual(tfidf_matrix.shape[0], len(self.documents))  # Rows match number of documents
        self.assertTrue("woman" in feature_names)
        self.assertTrue("man" in feature_names)

    def test_filter_documents_by_term(self):
        # Test document filtering
        filtered_docs = self.analyzer.filter_documents_by_term(self.documents, "woman")
        self.assertEqual(len(filtered_docs), 6)
        self.assertTrue(all("woman" in doc.lower() for doc in filtered_docs))

    def test_compute_cosine_similarity(self):
        # Test cosine similarity computation
        vec1 = np.array([0.1, 0.2, 0.3])
        vec2 = np.array([0.1, 0.2, 0.4])
        similarity = self.analyzer.compute_cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, np.float64(0.99146), places=4)

    def test_normalize_tfidf_difference(self):
        # Test normalized TF-IDF differences
        avg_tfidf = np.array([0.2, 0.3, 0.4])
        overall_avg_tfidf = np.array([0.1, 0.25, 0.35])
        normalized = self.analyzer.normalize_tfidf_difference(avg_tfidf, overall_avg_tfidf)
        expected = (avg_tfidf - overall_avg_tfidf) / np.std(avg_tfidf)
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_analyze_subgroup_terms(self):
        # Test subgroup analysis
        results = self.analyzer.analyze_subgroup_terms(self.documents, self.subgroup_terms)
        self.assertIn("woman", results)
        self.assertIn("man", results)
        self.assertTrue("engineer" in [word for word, _ in results["woman"]["associated_words"]])
        self.assertTrue("athlete" in [word for word, _ in results["man"]["associated_words"]])

    def test_empty_documents(self):
        # Test behavior with empty documents
        with self.assertRaises(ValueError) as context:
            self.analyzer.analyze_subgroup_terms([], self.subgroup_terms)
        
        self.assertEqual(str(context.exception), "empty vocabulary; perhaps the documents only contain stop words")

    def test_nonexistent_subgroup_term(self):
        # Test behavior with a term not present in the documents
        results = self.analyzer.analyze_subgroup_terms(self.documents, ["alien"])
        self.assertNotIn("alien", results)

    def test_get_top_associations(self):
        # Test getting top associations
        results = self.analyzer.analyze_subgroup_terms(self.documents, self.subgroup_terms)
        top_associations = self.analyzer.get_top_associations(results, 'man', 4)
        self.assertTrue(all(isinstance(word, str) for word, _ in top_associations))
        self.assertTrue(all(isinstance(score, float) for _, score in top_associations))
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Top associations: {top_associations}")
        # Additional check for top 4 associations

        expected_top_4 = [
            ('man', np.float64(0.21846844736215792)),
            ('woman', np.float64(0.10659097754077432)),
            ('lose', np.float64(0.10356751842862605)),
            ('community', np.float64(0.08252307302468363))
        ]
        for (word, score), (expected_word, expected_score) in zip(top_associations, expected_top_4):
            self.assertEqual(word, expected_word)
            np.testing.assert_allclose(score, expected_score, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
