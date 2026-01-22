"""
Classification Orchestrator

Two-stage classification pipeline that combines:
    - Stage 1: Category detection
    - Stage 2: Element (and optionally attribute) extraction

TODO: Implement orchestrator.

Future usage:
    from classifier.pipeline import ClassificationOrchestrator

    orchestrator = ClassificationOrchestrator(
        taxonomy=taxonomy,
        processor=llm_processor,
        include_attributes=True,
    )

    results = orchestrator.classify_comments(comments)
"""

from typing import Any, Dict, List


class ClassificationOrchestrator:
    """
    Two-stage classification orchestrator.

    TODO: Implement this class.
    """

    def __init__(
        self,
        taxonomy: dict,
        processor: Any,
        include_attributes: bool = False,
    ):
        raise NotImplementedError("Orchestrator not yet implemented")

    def classify_comments(self, comments: List[str]) -> List[Any]:
        raise NotImplementedError("Orchestrator not yet implemented")

    def results_to_dataframe(self, results: List[Any]):
        raise NotImplementedError("Orchestrator not yet implemented")
