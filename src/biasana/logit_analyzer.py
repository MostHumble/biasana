import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .template_manager import TemplateManager


@dataclass
class AnalysisResult:
    """Container for analysis results

    Args:
        context: The input context/template
        raw_scores: Raw model scores (can be probabilities or log probabilities, depending on use_log_prob)
        normalized_scores: Normalized scores between 0 and 1
        total_score: Sum of raw scores
    """

    context: str
    raw_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    total_score: float


class LogitAnalyzer:
    """
    Class for analyzing bias in language models using logit scores.
    """

    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: Optional[str] = None,
        custom_templates_path: Optional[str] = None,
    ):
        """
        Initialize the LogitAnalyzer.

        Args:
            model_name: Name of the pretrained model to use
            revision: Model revision to use
            device: Device to run model on
            custom_templates_path: Path to JSON file with custom templates
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.model.eval()

        self.template_manager = TemplateManager(custom_templates_path)

    def compute_sequence_probability(
        self, sequence: str, return_token_probs: bool = False, use_log_prob: bool = True
    ) -> Union[float, tuple[float, List[float]]]:
        """
        Compute the probability of generating the entire sequence using log probabilities.

        Args:
            sequence: Input sequence
            return_token_probs: Whether to return individual token probabilities
            use_log_prob: Whether to return log probabilities (default: True)
        """
        tokens = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        token_ids = tokens.input_ids[0]

        with torch.no_grad():
            outputs = self.model(input_ids=tokens.input_ids)
            logits = outputs.logits[0, :-1]  # Remove last position
            log_probs = torch.log_softmax(logits, dim=-1)  # Use log_softmax

            # Get log probability of each actual next token
            token_log_probs = [
                log_probs[i, token_ids[i + 1]].item() for i in range(len(token_ids) - 1)
            ]

            # Sum log probabilities
            sequence_log_prob = sum(token_log_probs)

            if not use_log_prob:
                sequence_prob = torch.exp(torch.tensor(sequence_log_prob)).item()
                token_probs = [
                    torch.exp(torch.tensor(p)).item() for p in token_log_probs
                ]
            else:
                sequence_prob = sequence_log_prob
                token_probs = token_log_probs

            if return_token_probs:
                return sequence_prob, token_probs
            return sequence_prob

    def analyze_bias(
        self,
        template: str,
        target_groups: List[str],
        use_template_name: bool = False,
        use_log_prob: bool = True,
    ) -> AnalysisResult:
        """
        Analyze bias for different target groups in a given template.

        Args:
            template: Template string or template name
            target_groups: List of target groups to analyze
            use_template_name: Whether template is a name from template manager
            use_log_prob: Whether to use log probabilities for analysis (default: True)

        Returns:
            AnalysisResult containing analysis results
        """
        if use_template_name:
            template = self.template_manager.get_template(template)
            if template is None:
                raise ValueError(f"Template '{template}' not found")

        self.template_manager.validate_templates([template])

        # Compute probabilities for each group
        raw_scores = {}
        for group in target_groups:
            sequence = template.replace("[TARGET]", group)
            raw_scores[group] = self.compute_sequence_probability(
                sequence, use_log_prob=use_log_prob
            )

        # Handle log probabilities with log-sum-exp trick for numerical stability
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

        if use_log_prob:
            log_probs = raw_scores
            max_log_prob = max(log_probs.values())

            log_probs_tensor = torch.tensor(
                list(log_probs.values()), device=self.device
            )
            total_log_prob = (
                max_log_prob
                + torch.logsumexp(log_probs_tensor - max_log_prob, dim=0).item()
            )

            normalized_scores = {
                group: math.exp(prob - total_log_prob)
                for group, prob in log_probs.items()
            }

            total_score = math.exp(total_log_prob)
        else:
            total_score = sum(raw_scores.values())
            normalized_scores = {
                group: prob / total_score if total_score > 0 else 0.0
                for group, prob in raw_scores.items()
            }

        return AnalysisResult(
            context=template,
            raw_scores=raw_scores,
            normalized_scores=normalized_scores,
            total_score=total_score,
        )

    def batch_analyze(
        self,
        templates: List[str],
        target_groups: List[str],
        use_template_names: bool = False,
    ) -> List[AnalysisResult]:
        """
        Analyze bias across multiple templates.

        Args:
            templates: List of templates or template names
            target_groups: List of target groups to analyze
            use_template_names: Whether templates are names from template manager

        Returns:
            List of AnalysisResult for each template
        """
        return [
            self.analyze_bias(template, target_groups, use_template_names)
            for template in templates
        ]

    def get_most_biased_templates(
        self,
        templates: List[str],
        target_groups: List[str],
        top_n: int = 5,
        use_template_names: bool = False,
    ) -> List[AnalysisResult]:
        """
        Find templates with highest disparity between group probabilities.

        Args:
            templates: List of templates or template names
            target_groups: List of target groups to analyze
            top_n: Number of top biased templates to return
            use_template_names: Whether templates are names from template manager

        Returns:
            List of top_n most biased templates and their results
        """
        results = self.batch_analyze(templates, target_groups, use_template_names)

        # Compute max probability difference for each result
        def get_max_diff(result: AnalysisResult) -> float:
            probs = result.normalized_scores
            return max(
                abs(probs[g1] - probs[g2])
                for i, g1 in enumerate(target_groups)
                for g2 in target_groups[i + 1 :]
            )

        # Sort by maximum probability difference
        sorted_results = sorted(results, key=get_max_diff, reverse=True)

        return sorted_results[:top_n]
