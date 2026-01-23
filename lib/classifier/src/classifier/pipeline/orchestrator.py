"""
Classification Orchestrator

Two-stage classification pipeline that combines:
    - Stage 1: Category detection
    - Stage 2: Element (and optionally attribute) extraction

TODO: Implement orchestrator after Stage 2 prompts are complete.

Future usage:
    from classifier.pipeline import ClassificationOrchestrator
    from classifier.prompts import build_stage1_prompt_function
    from classifier.taxonomy import load_condensed, load_examples

    # Load artifacts
    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    # Build prompt functions
    stage1_prompt = build_stage1_prompt_function(condensed, examples)
    # stage2_prompts = build_stage2_prompt_functions(condensed, examples)  # future

    # Create orchestrator
    orchestrator = ClassificationOrchestrator(
        stage1_prompt=stage1_prompt,
        # stage2_prompts=stage2_prompts,
        processor=processor,
        category_to_schema=models["category_to_schema"],
    )

    # Classify comments
    results = orchestrator.classify_comments(comments)
    df = orchestrator.results_to_dataframe(results)
"""

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel


class ClassificationOrchestrator:
    """
    Two-stage classification orchestrator.

    This class orchestrates the full classification pipeline:
    1. Stage 1: Detect which categories are present in each comment
    2. Stage 2: Extract specific elements (and optionally attributes) per category
    3. Combine results into unified output

    TODO: Implement this class after Stage 2 prompts are complete.
    """

    def __init__(
        self,
        stage1_prompt: Callable[[str], str],
        processor: Any,
        category_to_schema: Optional[Dict[str, type]] = None,
        stage2_prompts: Optional[Dict[str, Callable[[str], str]]] = None,
        guided_config: Optional[dict] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            stage1_prompt: Function that generates Stage 1 prompt from a comment
            processor: LLM processor with process_with_schema method
            category_to_schema: Maps category name to Pydantic schema for Stage 2
            stage2_prompts: Maps category name to Stage 2 prompt function
            guided_config: Optional sampling parameters
        """
        raise NotImplementedError(
            "Orchestrator not yet implemented. Use Stage 1 classification directly for now."
        )

    def classify_comments(self, comments: List[str]) -> List[Any]:
        """
        Run two-stage classification on a list of comments.

        Args:
            comments: List of comment strings

        Returns:
            List of FinalClassificationOutput objects
        """
        raise NotImplementedError("Orchestrator not yet implemented")

    def classify_single(self, comment: str) -> Any:
        """
        Classify a single comment.

        Args:
            comment: Comment string

        Returns:
            FinalClassificationOutput object
        """
        raise NotImplementedError("Orchestrator not yet implemented")

    def results_to_dataframe(self, results: List[Any]):
        """
        Convert results to a pandas DataFrame for analysis.

        Args:
            results: List of FinalClassificationOutput objects

        Returns:
            pandas DataFrame
        """
        raise NotImplementedError("Orchestrator not yet implemented")
