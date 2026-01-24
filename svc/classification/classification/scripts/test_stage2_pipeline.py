"""
Test Stage 2 Pipeline (Element Extraction)

This script tests Stage 2 in isolation - useful for:
    - Debugging Stage 2 prompts
    - Testing element extraction without running Stage 1
    - Quick iteration on Stage 2 prompt improvements

Usage:
    # Test Stage 2 for a specific category
    python test_stage2_pipeline.py \\
        --condensed artifacts/condensed_taxonomy.json \\
        --examples artifacts/examples.json \\
        --category "People"

    # Test all categories
    python test_stage2_pipeline.py \\
        --condensed artifacts/condensed_taxonomy.json \\
        --examples artifacts/examples.json \\
        --all-categories

    # Export Stage 2 prompts only
    python test_stage2_pipeline.py \\
        --condensed artifacts/condensed_taxonomy.json \\
        --examples artifacts/examples.json \\
        --export-prompts prompts/stage2/
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

# =============================================================================
# Test Comments by Category
# =============================================================================

CATEGORY_TEST_COMMENTS = {
    "Attendee Engagement & Interaction": [
        "The community is so supportive and open to sharing their knowledge.",
        "The networking sessions were fantastic - I made so many new connections!",
        "The gala dinner was beautiful but I wish there was more time to mingle.",
        "Loved connecting with other Blue users and learning from their experiences.",
    ],
    "Event Logistics & Infrastructure": [
        "The WiFi kept dropping during the keynote which was frustrating.",
        "Beautiful venue but the rooms were too cold.",
        "Having the Bluepulse app earlier would have been helpful.",
        "Great food at the conference, loved the vegetarian options.",
    ],
    "Event Operations & Management": [
        "An excellent, well-organised event from start to finish.",
        "The registration process was slow and confusing.",
        "Sessions ran over time which caused conflicts with my schedule.",
        "Better signage would have helped us find the right rooms.",
    ],
    "Learning & Content Delivery": [
        "The presentations were excellent and very insightful.",
        "I wish there were more hands-on workshops.",
        "The panel discussions needed better moderation.",
        "I came away with a significant to-do list of actionable items.",
    ],
    "People": [
        "The Explorance staff are so genuine and knowledgeable.",
        "Some speakers didn't stick to their abstracts which was disappointing.",
        "The ability to talk to Explorance experts made this valuable.",
        "Fellow attendees were smart, creative, and willing to share.",
    ],
}


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test Stage 2 Pipeline (Element Extraction)")

    # Input sources
    parser.add_argument(
        "--condensed",
        type=str,
        required=True,
        help="Path to condensed taxonomy JSON",
    )
    parser.add_argument(
        "--examples",
        type=str,
        required=True,
        help="Path to examples JSON",
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        help="Path to raw taxonomy (for dynamic schema validation)",
    )

    # Category selection
    parser.add_argument(
        "--category",
        type=str,
        help="Specific category to test",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Test all categories",
    )

    # Custom comments
    parser.add_argument(
        "--comments",
        type=str,
        nargs="+",
        help="Custom comments to test (instead of built-in)",
    )

    # Output options
    parser.add_argument(
        "--export-prompts",
        type=str,
        help="Directory to export Stage 2 prompts",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Directory to save results",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Display the full prompt for the category",
    )

    # Processing options
    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0],
        help="GPU(s) to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="casperhansen/mistral-nemo-instruct-2407-awq",
        help="LLM model to use",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip classification (just build/export prompts)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.category and not args.all_categories and not args.export_prompts:
        parser.error("Specify --category, --all-categories, or --export-prompts")

    # Create output directories
    if args.export_prompts:
        Path(args.export_prompts).mkdir(parents=True, exist_ok=True)
    if args.save_results:
        Path(args.save_results).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Import and Load
    # =========================================================================

    print("=" * 70)
    print("STAGE 2 PIPELINE TEST")
    print("=" * 70)

    from classifier import (
        # Schemas
        ElementExtractionOutput,
        # Model building
        build_models_from_taxonomy,
        # Stage 2 Prompts
        build_stage2_prompt_functions,
        build_stage2_prompt_string,
        export_stage2_prompt_module,
        export_stage2_prompt_modules,
        get_stage2_examples_for_category,
        # Load
        load_condensed,
        load_examples,
        print_stage2_prompts_preview,
    )

    print("✓ Imports successful")

    # Load artifacts
    print(f"\nLoading condensed taxonomy from: {args.condensed}")
    condensed = load_condensed(args.condensed)
    print(f"✓ Loaded {len(condensed.categories)} categories")

    print(f"Loading examples from: {args.examples}")
    examples = load_examples(args.examples)
    print(f"✓ Loaded {examples.get_stats()['total']} examples")

    # Load taxonomy for dynamic schemas
    category_to_schema = {}
    if args.taxonomy:
        print(f"Loading taxonomy for schema building: {args.taxonomy}")
        with open(args.taxonomy, "r") as f:
            taxonomy = json.load(f)
        models = build_models_from_taxonomy(taxonomy)
        category_to_schema = models["category_to_schema"]
        print(f"✓ Built schemas for {len(category_to_schema)} categories")

    # =========================================================================
    # Build Stage 2 Prompts
    # =========================================================================

    print("\n" + "=" * 70)
    print("BUILDING STAGE 2 PROMPTS")
    print("=" * 70)

    stage2_prompts = build_stage2_prompt_functions(condensed, examples)
    print(f"✓ Built {len(stage2_prompts)} Stage 2 prompt functions")

    available_categories = list(stage2_prompts.keys())
    print("\nAvailable categories:")
    for cat in available_categories:
        cat_examples = get_stage2_examples_for_category(examples, cat)
        print(f"  • {cat} ({len(cat_examples)} examples)")

    # Preview
    print_stage2_prompts_preview(condensed, examples)

    # =========================================================================
    # Export Prompts (if requested)
    # =========================================================================

    if args.export_prompts:
        print("\n" + "=" * 70)
        print("EXPORTING PROMPTS")
        print("=" * 70)

        export_dir = Path(args.export_prompts)

        # Export as single module
        single_path = export_dir / "stage2_prompts.py"
        export_stage2_prompt_module(condensed, examples, single_path)
        print(f"✓ Exported single module: {single_path}")

        # Export as separate modules
        multi_dir = export_dir / "stage2"
        paths = export_stage2_prompt_modules(condensed, examples, multi_dir)
        print(f"✓ Exported {len(paths)} separate modules to: {multi_dir}")

    # =========================================================================
    # Determine Categories to Test
    # =========================================================================

    if args.skip_classification:
        print("\n🔥 PROMPTS READY! (skipping classification)")
        return

    categories_to_test = []
    if args.all_categories:
        categories_to_test = available_categories
    elif args.category:
        if args.category not in available_categories:
            print(f"❌ Unknown category: {args.category}")
            print(f"Available: {available_categories}")
            return
        categories_to_test = [args.category]

    if not categories_to_test:
        print("\n⚠ No categories to test. Use --category or --all-categories")
        return

    # =========================================================================
    # Run Stage 2 Classification
    # =========================================================================

    print("\n" + "=" * 70)
    print("RUNNING STAGE 2 CLASSIFICATION")
    print("=" * 70)

    from llm_parallelization.new_processor import NewProcessor

    all_results = {}

    with NewProcessor(
        gpu_list=args.gpu,
        llm=args.model,
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.9,
    ) as processor:
        for category in categories_to_test:
            print(f"\n--- {category} ---")

            # Get test comments
            if args.comments:
                test_comments = args.comments
            else:
                test_comments = CATEGORY_TEST_COMMENTS.get(
                    category, [f"This is a test comment for {category}."]
                )

            print(f"Testing with {len(test_comments)} comments")

            # Show prompt if requested
            if args.show_prompt:
                prompt_fn = stage2_prompts[category]
                sample_prompt = prompt_fn(test_comments[0])
                print(f"\n[PROMPT PREVIEW]\n{sample_prompt[:1500]}...")
                print(f"\n(Total: {len(sample_prompt):,} chars)")

            # Generate prompts
            prompt_fn = stage2_prompts[category]
            prompts = [prompt_fn(c) for c in test_comments]

            # Get schema
            schema = category_to_schema.get(category, ElementExtractionOutput)

            # Run classification
            responses = processor.process_with_schema(
                prompts=prompts,
                schema=schema,
                batch_size=len(prompts),
                guided_config={
                    "temperature": 0.1,
                    "top_k": 50,
                    "top_p": 0.95,
                    "max_tokens": 1500,
                },
            )

            results = processor.parse_results_with_schema(
                schema=schema,
                responses=responses,
                validate=True,
            )

            # Display results
            category_results = []
            for comment, result in zip(test_comments, results):
                preview = comment[:60] + "..." if len(comment) > 60 else comment
                print(f'\n  "{preview}"')

                if result is None:
                    print("    ❌ Failed to parse")
                    category_results.append({"comment": comment, "classifications": None})
                    continue

                if not result.classifications:
                    print("    (no elements extracted)")
                    category_results.append({"comment": comment, "classifications": []})
                    continue

                for c in result.classifications:
                    excerpt_preview = (
                        c.excerpt[:40] + "..." if len(c.excerpt) > 40 else c.excerpt
                    )
                    print(f'    • [{c.sentiment}] {c.element}: "{excerpt_preview}"')

                category_results.append(
                    {
                        "comment": comment,
                        "classifications": [
                            {
                                "excerpt": c.excerpt,
                                "element": c.element,
                                "sentiment": c.sentiment,
                                "reasoning": c.reasoning,
                            }
                            for c in result.classifications
                        ],
                    }
                )

            all_results[category] = category_results

    # =========================================================================
    # Save Results
    # =========================================================================

    if args.save_results:
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        results_path = Path(args.save_results) / "stage2_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Saved to: {results_path}")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_comments = 0
    total_extractions = 0

    for category, results in all_results.items():
        cat_comments = len(results)
        cat_extractions = sum(
            len(r["classifications"]) for r in results if r["classifications"]
        )
        print(f"  {category}: {cat_comments} comments, {cat_extractions} extractions")
        total_comments += cat_comments
        total_extractions += cat_extractions

    print(f"\nTotal: {total_comments} comments, {total_extractions} extractions")
    print(f"Avg extractions per comment: {total_extractions / total_comments:.2f}")

    print("\n🔥 STAGE 2 TEST COMPLETE! 🔥")


if __name__ == "__main__":
    main()
