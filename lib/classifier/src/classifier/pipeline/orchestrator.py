"""
Classification Orchestrator

Two-stage classification pipeline that combines:
    - Stage 1: Category detection
    - Stage 2: Element extraction (per detected category)

The orchestrator handles the full classification workflow:
    1. Run Stage 1 to detect categories in each comment
    2. For each detected category, run the corresponding Stage 2 prompt
    3. Combine results into unified FinalClassificationOutput

Usage:
    from classifier.pipeline import ClassificationOrchestrator
    from classifier import (
        load_condensed, load_examples,
        build_stage1_prompt_function, build_stage2_prompt_functions,
        build_models_from_taxonomy,
    )

    # Load artifacts
    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    # Build prompt functions
    stage1_prompt = build_stage1_prompt_function(condensed, examples)
    stage2_prompts = build_stage2_prompt_functions(condensed, examples)

    # Build dynamic schemas
    models = build_models_from_taxonomy(taxonomy)

    # Create orchestrator
    orchestrator = ClassificationOrchestrator(
        stage1_prompt=stage1_prompt,
        stage2_prompts=stage2_prompts,
        category_to_schema=models["category_to_schema"],
    )

    # Classify comments
    with NewProcessor(...) as processor:
        results = orchestrator.classify_comments(comments, processor)
        df = orchestrator.results_to_dataframe(results)
"""

from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from ..taxonomy.schemas import (
    CategoryDetectionOutput,
    ClassificationSpan,
    ElementExtractionOutput,
    FinalClassificationOutput,
)


class ClassificationOrchestrator:
    """
    Two-stage classification orchestrator.

    This class orchestrates the full classification pipeline:
    1. Stage 1: Detect which categories are present in each comment
    2. Stage 2: Extract specific elements per detected category
    3. Combine results into unified output

    Attributes:
        stage1_prompt: Function that generates Stage 1 prompt from a comment
        stage2_prompts: Dict mapping category name to Stage 2 prompt function
        category_to_schema: Dict mapping category name to Pydantic schema for Stage 2
        guided_config: Sampling parameters for LLM
    """

    def __init__(
        self,
        stage1_prompt: Callable[[str], str],
        stage2_prompts: Dict[str, Callable[[str], str]],
        category_to_schema: Optional[Dict[str, Type[BaseModel]]] = None,
        guided_config: Optional[dict] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            stage1_prompt: Function that generates Stage 1 prompt from a comment
            stage2_prompts: Dict mapping category name to Stage 2 prompt function
            category_to_schema: Optional dict mapping category name to Pydantic schema.
                               If not provided, uses generic ElementExtractionOutput.
            guided_config: Optional sampling parameters
        """
        self.stage1_prompt = stage1_prompt
        self.stage2_prompts = stage2_prompts
        self.category_to_schema = category_to_schema or {}
        self.guided_config = guided_config or {
            "temperature": 0.1,
            "top_k": 50,
            "top_p": 0.95,
            "max_tokens": 1500,
        }

        # Categories we can process (those with prompts)
        self.available_categories = set(stage2_prompts.keys())

    def _get_schema_for_category(self, category: str) -> Type[BaseModel]:
        """Get the Pydantic schema for a category."""
        if category in self.category_to_schema:
            return self.category_to_schema[category]
        # Fall back to generic schema
        return ElementExtractionOutput

    def classify_comments(
        self,
        comments: List[str],
        processor: Any,
        batch_size: int = 25,
        verbose: bool = True,
    ) -> List[FinalClassificationOutput]:
        """
        Run two-stage classification on a list of comments.

        Args:
            comments: List of comment strings
            processor: LLM processor with process_with_schema method
            batch_size: Batch size for processing
            verbose: Print progress information

        Returns:
            List of FinalClassificationOutput objects
        """
        # =====================================================================
        # Stage 1: Category Detection
        # =====================================================================

        if verbose:
            print("=" * 60)
            print("STAGE 1: Category Detection")
            print("=" * 60)

        stage1_prompts = [self.stage1_prompt(c) for c in comments]

        stage1_responses = processor.process_with_schema(
            prompts=stage1_prompts,
            schema=CategoryDetectionOutput,
            batch_size=batch_size,
            guided_config=self.guided_config,
        )

        stage1_results = processor.parse_results_with_schema(
            schema=CategoryDetectionOutput,
            responses=stage1_responses,
            validate=True,
        )

        if verbose:
            successful = sum(1 for r in stage1_results if r is not None)
            print(f"✓ Stage 1 complete: {successful}/{len(comments)} parsed successfully")

        # =====================================================================
        # Build Stage 2 Tasks
        # =====================================================================

        # List of (comment_idx, category, prompt)
        stage2_tasks: List[tuple] = []

        for idx, (comment, result) in enumerate(zip(comments, stage1_results)):
            if result is None or not result.has_classifiable_content:
                continue

            for category in result.categories_present:
                if category not in self.available_categories:
                    if verbose:
                        print(f"  ⚠ Unknown category '{category}', skipping")
                    continue

                prompt_fn = self.stage2_prompts[category]
                stage2_tasks.append((idx, category, prompt_fn(comment)))

        if verbose:
            print(f"\nStage 2 tasks: {len(stage2_tasks)} total")

        # =====================================================================
        # Stage 2: Element Extraction (grouped by category)
        # =====================================================================

        if verbose:
            print("\n" + "=" * 60)
            print(f"STAGE 2: Element Extraction ({len(stage2_tasks)} tasks)")
            print("=" * 60)

        # Results map: comment_idx -> [(category, result), ...]
        stage2_results_map: Dict[int, List[tuple]] = {i: [] for i in range(len(comments))}

        # Process each category separately (each has its own schema)
        for category in self.available_categories:
            category_tasks = [
                (idx, prompt) for idx, cat, prompt in stage2_tasks if cat == category
            ]

            if not category_tasks:
                continue

            if verbose:
                print(f"  Processing: {category} ({len(category_tasks)} comments)")

            task_indices = [t[0] for t in category_tasks]
            task_prompts = [t[1] for t in category_tasks]

            schema = self._get_schema_for_category(category)

            responses = processor.process_with_schema(
                prompts=task_prompts,
                schema=schema,
                batch_size=batch_size,
                guided_config=self.guided_config,
            )

            parsed = processor.parse_results_with_schema(
                schema=schema,
                responses=responses,
                validate=True,
            )

            for idx, result in zip(task_indices, parsed):
                if result is not None:
                    stage2_results_map[idx].append((category, result))

        # =====================================================================
        # Combine Results
        # =====================================================================

        if verbose:
            print("\n" + "=" * 60)
            print("Combining Results")
            print("=" * 60)

        final_results = []

        for idx, comment in enumerate(comments):
            stage1_result = stage1_results[idx]

            # Build classifications list
            classifications = []

            for category, stage2_result in stage2_results_map[idx]:
                for span in stage2_result.classifications:
                    classifications.append(
                        ClassificationSpan(
                            excerpt=span.excerpt,
                            category=category,
                            element=span.element,
                            sentiment=span.sentiment,
                            reasoning=span.reasoning,
                        )
                    )

            final_results.append(
                FinalClassificationOutput(
                    original_comment=comment,
                    has_classifiable_content=(
                        stage1_result.has_classifiable_content if stage1_result else False
                    ),
                    category_reasoning=(
                        stage1_result.reasoning if stage1_result else "Failed to parse"
                    ),
                    classifications=classifications,
                )
            )

        if verbose:
            total_classifications = sum(len(r.classifications) for r in final_results)
            print(
                f"✓ Combined {total_classifications} classifications across {len(comments)} comments"
            )

        return final_results

    def classify_single(
        self,
        comment: str,
        processor: Any,
        verbose: bool = False,
    ) -> FinalClassificationOutput:
        """
        Classify a single comment.

        Args:
            comment: Comment string
            processor: LLM processor
            verbose: Print progress

        Returns:
            FinalClassificationOutput object
        """
        results = self.classify_comments([comment], processor, batch_size=1, verbose=verbose)
        return results[0]

    def results_to_dataframe(
        self,
        results: List[FinalClassificationOutput],
        include_empty: bool = True,
    ):
        """
        Convert results to a pandas DataFrame for analysis.

        Creates one row per classification span. Comments with multiple
        classifications will have multiple rows.

        Args:
            results: List of FinalClassificationOutput objects
            include_empty: If True, include rows for comments with no classifications

        Returns:
            pandas DataFrame with columns:
                - original_comment
                - has_classifiable_content
                - category_reasoning
                - excerpt
                - category
                - element
                - sentiment
                - reasoning
        """
        import pandas as pd

        rows = []

        for result in results:
            if not result.classifications:
                if include_empty:
                    rows.append(
                        {
                            "original_comment": result.original_comment,
                            "has_classifiable_content": result.has_classifiable_content,
                            "category_reasoning": result.category_reasoning,
                            "excerpt": None,
                            "category": None,
                            "element": None,
                            "sentiment": None,
                            "reasoning": None,
                        }
                    )
            else:
                for classification in result.classifications:
                    rows.append(
                        {
                            "original_comment": result.original_comment,
                            "has_classifiable_content": result.has_classifiable_content,
                            "category_reasoning": result.category_reasoning,
                            "excerpt": classification.excerpt,
                            "category": classification.category,
                            "element": classification.element,
                            "sentiment": classification.sentiment,
                            "reasoning": classification.reasoning,
                        }
                    )

        return pd.DataFrame(rows)

    def results_to_json(
        self,
        results: List[FinalClassificationOutput],
    ) -> List[dict]:
        """
        Convert results to JSON-serializable list of dicts.

        Args:
            results: List of FinalClassificationOutput objects

        Returns:
            List of dictionaries
        """
        return [r.model_dump() for r in results]

    def get_classification_stats(
        self,
        results: List[FinalClassificationOutput],
    ) -> dict:
        """
        Get statistics about classification results.

        Args:
            results: List of FinalClassificationOutput objects

        Returns:
            Dict with statistics
        """
        from collections import Counter

        total = len(results)
        classifiable = sum(1 for r in results if r.has_classifiable_content)
        non_classifiable = total - classifiable

        total_classifications = sum(len(r.classifications) for r in results)

        # Category distribution
        category_counts = Counter()
        element_counts = Counter()
        sentiment_counts = Counter()

        for result in results:
            for c in result.classifications:
                category_counts[c.category] += 1
                element_counts[(c.category, c.element)] += 1
                sentiment_counts[c.sentiment] += 1

        return {
            "total_comments": total,
            "classifiable_comments": classifiable,
            "non_classifiable_comments": non_classifiable,
            "total_classifications": total_classifications,
            "avg_classifications_per_comment": (
                total_classifications / classifiable if classifiable > 0 else 0
            ),
            "category_distribution": dict(category_counts),
            "element_distribution": {
                f"{cat} > {elem}": count for (cat, elem), count in element_counts.most_common()
            },
            "sentiment_distribution": dict(sentiment_counts),
        }

    def print_results_summary(
        self,
        results: List[FinalClassificationOutput],
        max_display: int = 10,
    ) -> None:
        """
        Print a summary of classification results.

        Args:
            results: List of FinalClassificationOutput objects
            max_display: Maximum number of comments to display in detail
        """
        stats = self.get_classification_stats(results)

        print("\n" + "=" * 70)
        print("CLASSIFICATION SUMMARY")
        print("=" * 70)

        print(f"\nComments: {stats['total_comments']}")
        print(f"  Classifiable: {stats['classifiable_comments']}")
        print(f"  Non-classifiable: {stats['non_classifiable_comments']}")
        print(f"\nTotal classifications: {stats['total_classifications']}")
        print(f"Avg per comment: {stats['avg_classifications_per_comment']:.2f}")

        print("\nCategory distribution:")
        for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
            pct = (
                100 * count / stats["total_classifications"]
                if stats["total_classifications"] > 0
                else 0
            )
            print(f"  {cat}: {count} ({pct:.1f}%)")

        print("\nSentiment distribution:")
        for sent, count in sorted(stats["sentiment_distribution"].items(), key=lambda x: -x[1]):
            pct = (
                100 * count / stats["total_classifications"]
                if stats["total_classifications"] > 0
                else 0
            )
            print(f"  {sent}: {count} ({pct:.1f}%)")

        print("\n" + "-" * 70)
        print(f"SAMPLE RESULTS (first {min(len(results), max_display)})")
        print("-" * 70)

        for i, result in enumerate(results[:max_display], 1):
            preview = (
                result.original_comment[:60] + "..."
                if len(result.original_comment) > 60
                else result.original_comment
            )
            print(f"\n{i}. {preview}")

            if not result.has_classifiable_content:
                print("   ⚠️  No classifiable content")
                continue

            if not result.classifications:
                print("   (no element extractions)")
                continue

            for c in result.classifications:
                excerpt_preview = c.excerpt[:40] + "..." if len(c.excerpt) > 40 else c.excerpt
                print(f"   • [{c.sentiment}] {c.category} > {c.element}")
                print(f'     "{excerpt_preview}"')


# =============================================================================
# Convenience Functions
# =============================================================================


def create_orchestrator(
    condensed,
    examples,
    taxonomy: Optional[dict] = None,
    guided_config: Optional[dict] = None,
):
    """
    Convenience function to create a fully configured orchestrator.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet
        taxonomy: Optional raw taxonomy for building dynamic schemas
        guided_config: Optional sampling config

    Returns:
        Configured ClassificationOrchestrator
    """
    from ..prompts import build_stage1_prompt_function, build_stage2_prompt_functions

    stage1_prompt = build_stage1_prompt_function(condensed, examples)
    stage2_prompts = build_stage2_prompt_functions(condensed, examples)

    category_to_schema = None
    if taxonomy:
        from ..taxonomy.model_builder import build_models_from_taxonomy

        models = build_models_from_taxonomy(taxonomy)
        category_to_schema = models["category_to_schema"]

    return ClassificationOrchestrator(
        stage1_prompt=stage1_prompt,
        stage2_prompts=stage2_prompts,
        category_to_schema=category_to_schema,
        guided_config=guided_config,
    )
