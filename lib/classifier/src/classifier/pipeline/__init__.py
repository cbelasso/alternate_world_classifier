"""
Pipeline Module

Components for running classification pipelines.

Submodules:
    - orchestrator: Two-stage classification orchestrator

Usage:
    from classifier.pipeline import ClassificationOrchestrator, create_orchestrator

    # Create with explicit components
    orchestrator = ClassificationOrchestrator(
        stage1_prompt=stage1_prompt_fn,
        stage2_prompts=stage2_prompt_fns,
        category_to_schema=models["category_to_schema"],
    )

    # Or use convenience function
    orchestrator = create_orchestrator(condensed, examples, taxonomy)

    # Run classification
    with NewProcessor(...) as processor:
        results = orchestrator.classify_comments(comments, processor)
        df = orchestrator.results_to_dataframe(results)
"""

from .orchestrator import ClassificationOrchestrator, create_orchestrator

__all__ = [
    "ClassificationOrchestrator",
    "create_orchestrator",
]
