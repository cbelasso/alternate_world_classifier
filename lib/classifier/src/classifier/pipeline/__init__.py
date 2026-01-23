"""
Pipeline Module

Components for running classification pipelines.

Submodules:
    - orchestrator: Two-stage classification orchestrator (future)

Future usage:
    from classifier.pipeline import ClassificationOrchestrator

    orchestrator = ClassificationOrchestrator(
        stage1_prompt=stage1_prompt_fn,
        stage2_prompts=stage2_prompt_fns,
        processor=processor,
    )

    results = orchestrator.classify_comments(comments)
    df = orchestrator.results_to_dataframe(results)
"""

# TODO: Export orchestrator when implemented
# from .orchestrator import ClassificationOrchestrator

__all__ = []
