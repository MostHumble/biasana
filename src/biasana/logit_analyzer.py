from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Optional, Tuple, Union

class LogitAnalyzer:
    """
    Analyzer for computing and comparing probabilities of different terms appearing in
    specific contexts using language model logits.
    """
    
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: Optional[str] = None
    ):
        """
        Initialize the LogitAnalyzer with a specific language model.
        
        Args:
            model_name (str): Name of the pretrained model to use 
            revision (str, optional): Revision of the model to load
            device (str, optional): Device to run the model on ("cuda" or "cpu")
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(self.device)
        self.model.eval()
        
    def compute_sequence_probabilities(
        self,
        context: str,
        target_terms: List[str],
        normalize: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute the probability of each target term appearing after the given context.
        
        Args:
            context (str): The context string before the target terms
            target_terms (List[str]): List of terms to analyze
            normalize (bool): Whether to normalize the probabilities
            
        Returns:
            Dict containing both raw and normalized probabilities for each term
        """
        # Tokenize the input context
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        # Tokenize target terms
        term_tokens = {
            term: self.tokenizer(term, add_special_tokens=False)["input_ids"]
            for term in target_terms
        }
        
        # Compute probabilities for each term
        probabilities = {}
        
        with torch.no_grad():
            for term, tokens in term_tokens.items():
                prob = self._compute_term_probability(inputs["input_ids"], tokens)
                probabilities[term] = prob
                
        # Create results dictionary
        results = {
            "raw_probabilities": probabilities
        }
        
        if normalize:
            total_prob = sum(probabilities.values())
            results["normalized_probabilities"] = {
                term: prob / total_prob 
                for term, prob in probabilities.items()
            }
            
        return results
    
    def _compute_term_probability(
        self,
        context_ids: torch.Tensor,
        term_tokens: List[int]
    ) -> float:
        """
        Compute the probability of a specific term given the context.
        
        Args:
            context_ids (torch.Tensor): Tokenized context
            term_tokens (List[int]): Tokens of the target term
            
        Returns:
            float: Probability of the term
        """
        prob = 1.0
        current_ids = context_ids
        
        for token in term_tokens:
            outputs = self.model(input_ids=current_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            token_prob = probs[0, token].item()
            prob *= token_prob
            
            # Update context for next token prediction
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token]], device=self.device)
            ], dim=1)
            
        return prob
    
    def analyze_bias(
        self,
        contexts: List[str],
        target_groups: List[str],
        aggregate: str = "mean"
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Analyze potential biases across multiple contexts for different target groups.
        
        Args:
            contexts (List[str]): List of context strings to analyze
            target_groups (List[str]): List of target groups to compare
            aggregate (str): Aggregation method ("mean" or "raw")
            
        Returns:
            Dict containing analysis results including raw probabilities and summary statistics
        """
        all_results = []
        
        for context in contexts:
            result = self.compute_sequence_probabilities(
                context,
                target_groups,
                normalize=True
            )
            all_results.append(result)
            
        # Aggregate results
        aggregated = {
            "contexts": contexts,
            "groups": target_groups,
            "raw_results": all_results,
        }
        
        if aggregate == "mean":
            # Compute mean probabilities across contexts
            mean_probs = {}
            for group in target_groups:
                probs = [
                    r["normalized_probabilities"][group]
                    for r in all_results
                ]
                mean_probs[group] = sum(probs) / len(probs)
            aggregated["mean_probabilities"] = mean_probs
            
        return aggregated

    def get_top_biased_contexts(
        self,
        contexts: List[str],
        target_groups: List[str],
        top_n: int = 5
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Find the contexts with the highest disparity between group probabilities.
        
        Args:
            contexts (List[str]): List of contexts to analyze
            target_groups (List[str]): Groups to compare
            top_n (int): Number of top biased contexts to return
            
        Returns:
            List of tuples containing contexts and their probability distributions
        """
        context_scores = []
        
        for context in contexts:
            result = self.compute_sequence_probabilities(
                context,
                target_groups,
                normalize=True
            )
            probs = result["normalized_probabilities"]
            
            # Compute disparity score (max difference between any two groups)
            max_diff = max(
                abs(probs[g1] - probs[g2])
                for i, g1 in enumerate(target_groups)
                for g2 in target_groups[i+1:]
            )
            
            context_scores.append((context, probs, max_diff))
            
        # Sort by disparity score and return top_n
        sorted_contexts = sorted(
            context_scores,
            key=lambda x: x[2],
            reverse=True
        )
        
        return [
            (context, probs)
            for context, probs, _ in sorted_contexts[:top_n]
        ]