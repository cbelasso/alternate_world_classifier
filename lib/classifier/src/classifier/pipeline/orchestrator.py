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

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel

from ..taxonomy.schemas import (
    AttributeExtractionOutput,
    CategoryDetectionOutput,
    ClassificationSpan,
    ClassificationSpanWithAttribute,
    ElementExtractionOutput,
    FinalClassificationOutput,
    FinalClassificationOutputWithAttributes,
)


class ClassificationOrchestrator:
    """
    Multi-stage classification orchestrator (2 or 3 stages).

    This class orchestrates the full classification pipeline:
    1. Stage 1: Detect which categories are present in each comment
    2. Stage 2: Extract specific elements per detected category (+ element sentiment)
    3. Stage 3 (optional): Extract specific attributes per element (+ attribute sentiment)

    Supports both 2-stage (stops at element) and 3-stage (continues to attribute) modes.

    Attributes:
        stage1_prompt: Function that generates Stage 1 prompt from a comment
        stage2_prompts: Dict mapping category name to Stage 2 prompt function
        stage3_prompts: Optional nested dict {category: {element: prompt_fn}}
        category_to_schema: Dict mapping category name to Pydantic schema for Stage 2
        element_to_schema: Optional nested dict {category: {element: schema}} for Stage 3
        guided_config: Sampling parameters for LLM
    """

    def __init__(
        self,
        stage1_prompt: Callable[[str], str],
        stage2_prompts: Dict[str, Callable[[str], str]],
        stage3_prompts: Optional[Dict[str, Dict[str, Callable[[str], str]]]] = None,
        category_to_schema: Optional[Dict[str, Type[BaseModel]]] = None,
        element_to_schema: Optional[Dict[str, Dict[str, Type[BaseModel]]]] = None,
        guided_config: Optional[dict] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            stage1_prompt: Function that generates Stage 1 prompt from a comment
            stage2_prompts: Dict mapping category name to Stage 2 prompt function
            stage3_prompts: Optional nested dict {category: {element: prompt_fn}} for Stage 3
            category_to_schema: Optional dict mapping category name to Pydantic schema.
                               If not provided, uses generic ElementExtractionOutput.
            element_to_schema: Optional nested dict {category: {element: schema}} for Stage 3.
                              If not provided, uses generic AttributeExtractionOutput.
            guided_config: Optional sampling parameters
        """
        self.stage1_prompt = stage1_prompt
        self.stage2_prompts = stage2_prompts
        self.stage3_prompts = stage3_prompts or {}
        self.category_to_schema = category_to_schema or {}
        self.element_to_schema = element_to_schema or {}
        self.guided_config = guided_config or {
            "temperature": 0.1,
            "top_k": 50,
            "top_p": 0.95,
            "max_tokens": 1500,
        }

        # Categories we can process (those with prompts)
        self.available_categories = set(stage2_prompts.keys())

        # Elements we can do Stage 3 on
        self.stage3_available = {}
        for cat, elements in self.stage3_prompts.items():
            self.stage3_available[cat] = set(elements.keys())

    def _get_schema_for_category(self, category: str) -> Type[BaseModel]:
        """Get the Pydantic schema for a category (Stage 2)."""
        if category in self.category_to_schema:
            return self.category_to_schema[category]
        # Fall back to generic schema
        return ElementExtractionOutput

    def _get_schema_for_element(self, category: str, element: str) -> Type[BaseModel]:
        """Get the Pydantic schema for an element (Stage 3)."""
        if category in self.element_to_schema:
            if element in self.element_to_schema[category]:
                return self.element_to_schema[category][element]
        # Fall back to generic schema
        return AttributeExtractionOutput

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

    def classify_comments_3stage(
        self,
        comments: List[str],
        processor: Any,
        batch_size: int = 25,
        verbose: bool = True,
    ) -> List[FinalClassificationOutputWithAttributes]:
        """
        Run three-stage classification on a list of comments.

        Stage 1: Category detection
        Stage 2: Element extraction (with element_sentiment)
        Stage 3: Attribute extraction (with attribute_sentiment)

        Args:
            comments: List of comment strings
            processor: LLM processor with process_with_schema method
            batch_size: Batch size for processing
            verbose: Print progress information

        Returns:
            List of FinalClassificationOutputWithAttributes objects
        """
        if not self.stage3_prompts:
            raise ValueError(
                "No Stage 3 prompts configured. Use classify_comments() for 2-stage."
            )

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

        stage2_tasks: List[Tuple[int, str, str]] = []

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
        # Stage 2: Element Extraction
        # =====================================================================

        if verbose:
            print("\n" + "=" * 60)
            print(f"STAGE 2: Element Extraction ({len(stage2_tasks)} tasks)")
            print("=" * 60)

        # Results: comment_idx -> [(category, element, sentiment, excerpt, reasoning), ...]
        stage2_results_map: Dict[int, List[tuple]] = {i: [] for i in range(len(comments))}

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
                    for span in result.classifications:
                        stage2_results_map[idx].append(
                            (
                                category,
                                span.element,
                                span.sentiment,
                                span.excerpt,
                                span.reasoning,
                            )
                        )

        # =====================================================================
        # Build Stage 3 Tasks
        # =====================================================================

        # List of (comment_idx, category, element, element_sentiment, element_excerpt, element_reasoning, prompt)
        stage3_tasks: List[Tuple[int, str, str, str, str, str, str]] = []

        for idx in range(len(comments)):
            comment = comments[idx]
            for (
                category,
                element,
                element_sentiment,
                element_excerpt,
                element_reasoning,
            ) in stage2_results_map[idx]:
                # Check if we have Stage 3 prompt for this category/element
                if category in self.stage3_prompts:
                    if element in self.stage3_prompts[category]:
                        prompt_fn = self.stage3_prompts[category][element]
                        # Use the excerpt as input to Stage 3
                        prompt = prompt_fn(element_excerpt)
                        stage3_tasks.append(
                            (
                                idx,
                                category,
                                element,
                                element_sentiment,
                                element_excerpt,
                                element_reasoning,
                                prompt,
                            )
                        )

        if verbose:
            print(f"\nStage 3 tasks: {len(stage3_tasks)} total")

        # =====================================================================
        # Stage 3: Attribute Extraction (grouped by category+element)
        # =====================================================================

        if verbose:
            print("\n" + "=" * 60)
            print(f"STAGE 3: Attribute Extraction ({len(stage3_tasks)} tasks)")
            print("=" * 60)

        # Results: comment_idx -> list of ClassificationSpanWithAttribute
        stage3_results_map: Dict[int, List[ClassificationSpanWithAttribute]] = {
            i: [] for i in range(len(comments))
        }

        if stage3_tasks:
            # Group tasks by (category, element) for element-specific schemas
            from collections import defaultdict

            tasks_by_element: Dict[Tuple[str, str], List[tuple]] = defaultdict(list)

            for task in stage3_tasks:
                idx, category, element = task[0], task[1], task[2]
                tasks_by_element[(category, element)].append(task)

            # Process each (category, element) group with its specific schema
            for (category, element), element_tasks in tasks_by_element.items():
                if verbose:
                    print(f"  Processing: {category} > {element} ({len(element_tasks)} tasks)")

                task_prompts = [t[6] for t in element_tasks]  # prompt is at index 6
                task_meta = [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in element_tasks]

                # Get element-specific schema (with Literal attribute types)
                schema = self._get_schema_for_element(category, element)

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

                for (idx, cat, elem, elem_sent, elem_excerpt, elem_reasoning), result in zip(
                    task_meta, parsed
                ):
                    if result is None or not result.classifications:
                        # No attributes extracted, keep element-level result
                        stage3_results_map[idx].append(
                            ClassificationSpanWithAttribute(
                                category=cat,
                                element=elem,
                                element_excerpt=elem_excerpt,
                                element_sentiment=elem_sent,
                                element_reasoning=elem_reasoning,
                                attribute="(no attribute)",
                                attribute_excerpt=elem_excerpt,  # Same as element
                                attribute_sentiment=elem_sent,  # Same as element
                                attribute_reasoning="No specific attribute identified",
                            )
                        )
                    else:
                        for attr_span in result.classifications:
                            stage3_results_map[idx].append(
                                ClassificationSpanWithAttribute(
                                    category=cat,
                                    element=elem,
                                    element_excerpt=elem_excerpt,
                                    element_sentiment=elem_sent,
                                    element_reasoning=elem_reasoning,
                                    attribute=attr_span.attribute,
                                    attribute_excerpt=attr_span.excerpt,
                                    attribute_sentiment=attr_span.sentiment,
                                    attribute_reasoning=attr_span.reasoning,
                                )
                            )

        # Also add Stage 2 results that don't have Stage 3 prompts
        for idx in range(len(comments)):
            for category, element, element_sentiment, excerpt, reasoning in stage2_results_map[
                idx
            ]:
                # Check if this was NOT processed in Stage 3
                has_stage3 = (
                    category in self.stage3_prompts and element in self.stage3_prompts[category]
                )
                if not has_stage3:
                    stage3_results_map[idx].append(
                        ClassificationSpanWithAttribute(
                            category=category,
                            element=element,
                            element_excerpt=excerpt,
                            element_sentiment=element_sentiment,
                            element_reasoning=reasoning,
                            attribute="(not applicable)",
                            attribute_excerpt=excerpt,  # Same as element
                            attribute_sentiment=element_sentiment,
                            attribute_reasoning=reasoning,  # Same as element
                        )
                    )

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

            final_results.append(
                FinalClassificationOutputWithAttributes(
                    original_comment=comment,
                    has_classifiable_content=(
                        stage1_result.has_classifiable_content if stage1_result else False
                    ),
                    category_reasoning=(
                        stage1_result.reasoning if stage1_result else "Failed to parse"
                    ),
                    classifications=stage3_results_map[idx],
                )
            )

        if verbose:
            total_classifications = sum(len(r.classifications) for r in final_results)
            with_attrs = sum(
                1
                for r in final_results
                for c in r.classifications
                if c.attribute not in ("(no attribute)", "(not applicable)")
            )
            print(
                f"✓ Combined {total_classifications} classifications ({with_attrs} with attributes)"
            )

            # Consensus stats
            all_spans = [c for r in final_results for c in r.classifications]
            if all_spans:
                consensus = sum(1 for c in all_spans if c.sentiment_consensus)
                print(
                    f"✓ Sentiment consensus: {consensus}/{len(all_spans)} ({100 * consensus / len(all_spans):.1f}%)"
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

    def classify_comments_stage1_only(
        self,
        comments: List[str],
        processor: Any,
        batch_size: int = 25,
        verbose: bool = True,
    ) -> List[CategoryDetectionOutput]:
        """
        Run Stage 1 (category detection) only.

        This is useful for:
        - Quick category-level analysis
        - Testing Stage 1 prompts before running full pipeline
        - Understanding category distribution in a dataset

        Args:
            comments: List of comment strings
            processor: LLM processor with process_with_schema method
            batch_size: Batch size for processing
            verbose: Print progress information

        Returns:
            List of CategoryDetectionOutput objects (one per comment)
        """
        if verbose:
            print("=" * 60)
            print("STAGE 1: Category Detection (Only)")
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
            classifiable = sum(
                1 for r in stage1_results if r is not None and r.has_classifiable_content
            )
            print(f"✓ Stage 1 complete: {successful}/{len(comments)} parsed successfully")
            print(f"✓ Classifiable: {classifiable}/{successful} comments")

        return stage1_results

    def results_to_dataframe_stage1(
        self,
        results: List[CategoryDetectionOutput],
        comments: List[str],
    ):
        """
        Convert Stage 1-only results to a pandas DataFrame.

        Creates one row per detected category. Comments with multiple
        categories will have multiple rows (consistent with Stage 2/3 behavior).

        Args:
            results: List of CategoryDetectionOutput objects
            comments: Original comment strings (must match results order)

        Returns:
            pandas DataFrame with columns:
                - comment: Original comment text
                - has_classifiable_content: Whether comment is classifiable
                - category: Single detected category (one row per category)
                - reasoning: Stage 1 reasoning
        """

        import pandas as pd

        rows = []
        for comment, result in zip(comments, results):
            if result is None:
                # Parse failure
                rows.append(
                    {
                        "comment": comment,
                        "has_classifiable_content": False,
                        "category": None,
                        "reasoning": "Failed to parse",
                    }
                )
            elif not result.categories_present:
                # No categories detected (but parsed successfully)
                rows.append(
                    {
                        "comment": comment,
                        "has_classifiable_content": result.has_classifiable_content,
                        "category": None,
                        "reasoning": result.reasoning,
                    }
                )
            else:
                # One row per detected category
                for category in result.categories_present:
                    rows.append(
                        {
                            "comment": comment,
                            "has_classifiable_content": result.has_classifiable_content,
                            "category": category,
                            "reasoning": result.reasoning,
                        }
                    )

        return pd.DataFrame(rows)

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

    def results_to_dataframe_3stage(
        self,
        results: List[FinalClassificationOutputWithAttributes],
        include_empty: bool = True,
    ):
        """
        Convert 3-stage results to a pandas DataFrame for analysis.

        Creates one row per classification span. Comments with multiple
        classifications will have multiple rows.

        Args:
            results: List of FinalClassificationOutputWithAttributes objects
            include_empty: If True, include rows for comments with no classifications

        Returns:
            pandas DataFrame with columns for each stage's reasoning and excerpts
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
                            "category": None,
                            "element": None,
                            "element_excerpt": None,
                            "element_sentiment": None,
                            "element_reasoning": None,
                            "attribute": None,
                            "attribute_excerpt": None,
                            "attribute_sentiment": None,
                            "attribute_reasoning": None,
                            "sentiment_consensus": None,
                        }
                    )
            else:
                for classification in result.classifications:
                    rows.append(
                        {
                            "original_comment": result.original_comment,
                            "has_classifiable_content": result.has_classifiable_content,
                            "category_reasoning": result.category_reasoning,
                            "category": classification.category,
                            "element": classification.element,
                            "element_excerpt": classification.element_excerpt,
                            "element_sentiment": classification.element_sentiment,
                            "element_reasoning": classification.element_reasoning,
                            "attribute": classification.attribute,
                            "attribute_excerpt": classification.attribute_excerpt,
                            "attribute_sentiment": classification.attribute_sentiment,
                            "attribute_reasoning": classification.attribute_reasoning,
                            "sentiment_consensus": classification.sentiment_consensus,
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
        results,
    ) -> dict:
        """
        Get statistics about classification results.

        Works with both 2-stage (FinalClassificationOutput) and
        3-stage (FinalClassificationOutputWithAttributes) results.

        Args:
            results: List of classification output objects

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
        attribute_counts = Counter()

        for result in results:
            for c in result.classifications:
                category_counts[c.category] += 1
                element_counts[(c.category, c.element)] += 1

                # Handle both 2-stage (c.sentiment) and 3-stage (c.attribute_sentiment)
                if hasattr(c, "sentiment"):
                    sentiment_counts[c.sentiment] += 1
                elif hasattr(c, "attribute_sentiment"):
                    sentiment_counts[c.attribute_sentiment] += 1

                # Track attributes for 3-stage
                if hasattr(c, "attribute"):
                    attribute_counts[(c.category, c.element, c.attribute)] += 1

        stats = {
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

        # Add attribute distribution for 3-stage
        if attribute_counts:
            stats["attribute_distribution"] = {
                f"{cat} > {elem} > {attr}": count
                for (cat, elem, attr), count in attribute_counts.most_common()
            }

        return stats

    def print_results_summary(
        self,
        results,
        max_display: int = 10,
    ) -> None:
        """
        Print a summary of classification results.

        Works with both 2-stage and 3-stage results.

        Args:
            results: List of classification output objects
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

                # Handle both 2-stage and 3-stage results
                if hasattr(c, "attribute") and c.attribute not in (
                    "(no attribute)",
                    "(not applicable)",
                ):
                    # 3-stage with attribute
                    consensus = "✓" if c.sentiment_consensus else "✗"
                    print(
                        f"   • [{c.attribute_sentiment}] {c.category} > {c.element} > {c.attribute} [{consensus}]"
                    )
                elif hasattr(c, "sentiment"):
                    # 2-stage
                    print(f"   • [{c.sentiment}] {c.category} > {c.element}")
                else:
                    # 3-stage without specific attribute
                    print(f"   • [{c.element_sentiment}] {c.category} > {c.element}")
                print(f'     "{excerpt_preview}"')


# =============================================================================
# Convenience Functions
# =============================================================================


def create_orchestrator(
    condensed,
    examples,
    taxonomy: Optional[dict] = None,
    include_stage3: bool = False,
    guided_config: Optional[dict] = None,
):
    """
    Convenience function to create a fully configured orchestrator.

    Args:
        condensed: CondensedTaxonomy (with attributes if include_stage3=True)
        examples: ExampleSet
        taxonomy: Optional raw taxonomy for building dynamic schemas
        include_stage3: If True, also build Stage 3 prompts
        guided_config: Optional sampling config

    Returns:
        Configured ClassificationOrchestrator
    """
    from ..prompts import build_stage1_prompt_function, build_stage2_prompt_functions

    stage1_prompt = build_stage1_prompt_function(condensed, examples)
    stage2_prompts = build_stage2_prompt_functions(condensed, examples)

    stage3_prompts = None
    if include_stage3:
        from ..prompts.stage3 import build_stage3_prompt_functions

        stage3_prompts = build_stage3_prompt_functions(condensed, examples)

    category_to_schema = None
    if taxonomy:
        from ..taxonomy.model_builder import build_models_from_taxonomy

        models = build_models_from_taxonomy(taxonomy)
        category_to_schema = models["category_to_schema"]

    return ClassificationOrchestrator(
        stage1_prompt=stage1_prompt,
        stage2_prompts=stage2_prompts,
        stage3_prompts=stage3_prompts,
        category_to_schema=category_to_schema,
        guided_config=guided_config,
    )
