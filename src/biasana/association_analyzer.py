from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import spacy

class AssociationAnalyzer:
    def __init__(self, min_df=2, preprocess=True, spacy_model="en_core_web_sm"):
        """
        Initialize the analyzer with a minimum document frequency for vocabulary.
        
        Args:
            min_df (int): Minimum number of times a word must appear to be included.
        """
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(min_df=min_df)
        self.nlp = spacy.load(spacy_model)
        self.results = None
    
    def preprocess_text(self, documents):
        """
        Tokenize, lemmatize, and preprocess documents for analysis.
        Args:
            documents (list): List of raw text documents.
        Returns:
            list: List of preprocessed, lemmatized documents.
        """
        processed = []
        for doc in documents:
            tokens = self.nlp(doc.lower())
            lemmatized = " ".join([token.lemma_ for token in tokens if not token.is_stop])
            processed.append(lemmatized)
        return processed
    
    def fit_transform(self, documents, preprocess=True):
        """
        Fit and transform the TF-IDF vectorizer on the documents.
        Args:
            documents (list): List of raw text documents.
        Returns:
            sparse matrix: TF-IDF matrix for the documents.
        """
        if preprocess:
            documents = self.preprocess_text(documents)
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        return tfidf_matrix, feature_names, documents

    def filter_documents_by_term(slef, documents, term):
        """
        Filter documents containing the exact term.
        Args:
            documents (list): List of text documents.
            term (str): Subgroup term to match.
        Returns:
            list: Filtered documents containing the term.
        """
        term_pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)  # Match as a whole word
        return [doc for doc in documents if term_pattern.search(doc)]

    def compute_cosine_similarity(self, term_vector, global_avg_vector):
        """
        Compute cosine similarity between two vectors.
        Args:
            term_vector (np.array): TF-IDF vector for subgroup term.
            global_avg_vector (np.array): Global average TF-IDF vector.
        Returns:
            float: Cosine similarity score.
        """
        term_vector = term_vector.reshape(1, -1)
        global_avg_vector = global_avg_vector.reshape(1, -1)
        return cosine_similarity(term_vector, global_avg_vector)[0][0]


    def normalize_tfidf_difference(self, avg_tfidf, overall_avg_tfidf):
        """
        Compute normalized TF-IDF differences.
        Args:
            avg_tfidf (np.array): Average TF-IDF scores for subgroup documents.
            overall_avg_tfidf (np.array): Average TF-IDF scores across all documents.
        Returns:
            np.array: Normalized TF-IDF differences.
        """
        std_dev = np.std(avg_tfidf)
        return (avg_tfidf - overall_avg_tfidf) / (std_dev if std_dev > 0 else 1)

    def analyze_subgroup_terms(self, documents, subgroup_terms, preprecess=True):
        """
        Analyze associations between subgroup terms and co-occurring words.
        Args:
            documents (list): Preprocessed text documents.
            subgroup_terms (list): Subgroup terms to analyze.
            vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
            tfidf_matrix (sparse matrix): TF-IDF matrix for the documents.
        Returns:
            dict: Analysis results for each subgroup term.
        """
        tfidf_matrix, feature_names, documents= self.fit_transform(documents, preprecess)
        results = {}
        global_avg_tfidf = np.ravel(tfidf_matrix.mean(axis=0))

        for term in subgroup_terms:
            term_docs = self.filter_documents_by_term(documents, term)
            if not term_docs:
                continue
            
            term_tfidf = self.vectorizer.transform(term_docs)
            avg_tfidf = np.ravel(term_tfidf.mean(axis=0))
            normalized_differences = self.normalize_tfidf_difference(avg_tfidf, global_avg_tfidf)
            
            # Cosine similarity
            cosine_sim = self.compute_cosine_similarity(avg_tfidf, global_avg_tfidf)
            
            # Filter positive scores
            positive_mask = avg_tfidf > 0
            associated_words = feature_names[positive_mask]
            associated_scores = avg_tfidf[positive_mask]
            
            results[term] = {
                'associated_words': list(zip(associated_words, associated_scores)),
                'tfidf_differences': list(zip(feature_names[positive_mask], normalized_differences[positive_mask])),
                'cosine_similarity': cosine_sim,
                'num_documents': len(term_docs)
            }
        
        self.results = results

        return results

    def get_top_associations(self, results, term, n=5):
        """
        Get the top n associations for a given term.
        
        :param results: Dictionary containing terms and their associated words with scores.
        :param term: The term for which to get the top associations.
        :param n: The number of top associations to return.
        :return: List of tuples containing associated words and their scores.
        """
        if term not in results:
            return []

        # Sort the associations by score in descending order and return the top n
        sorted_associations = sorted(results[term]['associated_words'], key=lambda item: item[1], reverse=True)
        return sorted_associations[:n]