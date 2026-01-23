"""
Test Stage 1 Classification Pipeline

This script demonstrates the complete Stage 1 workflow using the refactored
modular classifier library.

WORKFLOW:
    1. Load raw taxonomy from JSON
    2. Condense taxonomy (with LLM) OR load pre-condensed
    3. Generate examples (with LLM) OR load pre-generated
    4. Build Stage 1 prompt function (pure Python)
    5. Run classification on test comments
    6. Display results

Usage:
    # Full pipeline with built-in test comments
    python test_stage1_pipeline.py --taxonomy /path/to/taxonomy.json

    # Classify your own data from a file
    python test_stage1_pipeline.py \\
        --condensed artifacts/condensed.json \\
        --examples artifacts/examples.json \\
        --input-file my_comments.csv \\
        --comment-column "feedback_text"

    # Classify Excel file with specific sheet
    python test_stage1_pipeline.py \\
        --condensed artifacts/condensed.json \\
        --examples artifacts/examples.json \\
        --input-file survey_data.xlsx \\
        --sheet-name "Comments" \\
        --comment-column "response"

    # Save artifacts after generation
    python test_stage1_pipeline.py \\
        --taxonomy /path/to/taxonomy.json \\
        --save-artifacts artifacts/

    # Limit number of comments to classify
    python test_stage1_pipeline.py \\
        --condensed artifacts/condensed.json \\
        --examples artifacts/examples.json \\
        --input-file large_dataset.csv \\
        --max-comments 100

Supported input file formats:
    - CSV (.csv): Requires --comment-column
    - Excel (.xlsx, .xls): Requires --comment-column, optional --sheet-name
    - JSON (.json): List of strings, or list of objects with --comment-column key
    - Text (.txt): One comment per line
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import List, Optional

# =============================================================================
# Helper Functions
# =============================================================================


def load_comments_from_file(
    filepath: str,
    comment_column: str = "comment",
    sheet_name: Optional[str] = None,
    max_comments: Optional[int] = None,
) -> List[str]:
    """
    Load comments from a file (CSV, Excel, or JSON).

    Args:
        filepath: Path to the input file
        comment_column: Column name for CSV/Excel files
        sheet_name: Sheet name for Excel files
        max_comments: Maximum number of comments to load

    Returns:
        List of comment strings
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in [".csv"]:
        import pandas as pd

        df = pd.read_csv(filepath)
        if comment_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Column '{comment_column}' not found. Available columns: {available}"
            )
        comments = df[comment_column].dropna().astype(str).tolist()

    elif suffix in [".xlsx", ".xls"]:
        import pandas as pd

        df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
        if comment_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Column '{comment_column}' not found. Available columns: {available}"
            )
        comments = df[comment_column].dropna().astype(str).tolist()

    elif suffix == ".json":
        with open(filepath, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                # List of strings
                comments = data
            elif all(isinstance(item, dict) for item in data):
                # List of dicts - look for comment_column key
                if data and comment_column in data[0]:
                    comments = [
                        item[comment_column] for item in data if item.get(comment_column)
                    ]
                else:
                    available = ", ".join(data[0].keys()) if data else "none"
                    raise ValueError(
                        f"Key '{comment_column}' not found in JSON objects. Available keys: {available}"
                    )
            else:
                raise ValueError("JSON must be a list of strings or list of objects")
        elif isinstance(data, dict) and comment_column in data:
            # Dict with comment_column as key containing list
            comments = data[comment_column]
        else:
            raise ValueError(
                f"JSON structure not recognized. Expected list of strings, "
                f"list of objects with '{comment_column}' key, or dict with '{comment_column}' key"
            )

    elif suffix == ".txt":
        # One comment per line
        with open(filepath, "r") as f:
            comments = [line.strip() for line in f if line.strip()]

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .csv, .xlsx, .xls, .json, or .txt"
        )

    # Limit if requested
    if max_comments:
        comments = comments[:max_comments]

    return comments


# =============================================================================
# Test Comments
# =============================================================================

TEST_COMMENTS = [
    # Single category examples
    "The networking sessions were fantastic and I made great connections with peers from other institutions.",
    "The WiFi kept dropping during sessions and the room was too cold.",
    "The keynote speaker was brilliant and really knew their stuff.",
    "Registration was smooth and the staff were very helpful.",
    "I came away with so many actionable insights to implement.",
    # Multi-category examples
    "The keynote speaker was brilliant and the presentation on machine learning was very insightful.",
    "The conference was well organized but I wish there were more hands-on workshops. The Explorance team was very helpful.",
    "WiFi was terrible but the venue was beautiful and the food was great.",
    # Edge cases
    "Everything was perfect!",
    "Seeing is believing!",
    "Data integrity never goes out of style.",  # Should be non-classifiable
    "The only constant in life is change.",  # Should be non-classifiable
]


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test Stage 1 Classification Pipeline")

    # Input sources (mutually exclusive paths)
    parser.add_argument(
        "--taxonomy",
        type=str,
        help="Path to raw taxonomy JSON (will condense and generate examples)",
    )
    parser.add_argument(
        "--condensed",
        type=str,
        help="Path to pre-condensed taxonomy JSON",
    )
    parser.add_argument(
        "--examples",
        type=str,
        help="Path to pre-generated examples JSON",
    )

    # Output options
    parser.add_argument(
        "--save-artifacts",
        type=str,
        help="Directory to save generated artifacts",
    )
    parser.add_argument(
        "--export-prompt",
        type=str,
        help="Path to export Stage 1 prompt as Python module",
    )

    # Input data options
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to file with comments to classify (CSV, Excel, or JSON). "
        "For CSV/Excel, specify column with --comment-column",
    )
    parser.add_argument(
        "--comment-column",
        type=str,
        default="comment",
        help="Column name containing comments (for CSV/Excel files). Default: 'comment'",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default=None,
        help="Sheet name for Excel files (default: first sheet)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=None,
        help="Maximum number of comments to classify (default: all)",
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
        help="Skip running classification (just build prompts)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Batch size for classification (default: 25)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.taxonomy and not (args.condensed and args.examples):
        parser.error("Either --taxonomy OR both --condensed and --examples must be provided")

    # Create save directory if needed
    if args.save_artifacts:
        Path(args.save_artifacts).mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Import classifier components
    # ==========================================================================

    print("=" * 70)
    print("IMPORTING CLASSIFIER LIBRARY")
    print("=" * 70)

    from classifier import (
        # Schemas
        CategoryDetectionOutput,
        # Prompts
        build_stage1_prompt_function,
        # Condensation
        condense_taxonomy,
        export_stage1_prompt_module,
        # Examples
        generate_all_examples,
        # Utilities
        get_taxonomy_stats,
        load_condensed,
        load_examples,
        print_condensed_preview,
        print_examples_preview,
        save_condensed,
        save_examples,
        validate_examples,
    )

    print("✓ Imports successful")

    # ==========================================================================
    # Determine what needs LLM
    # ==========================================================================

    need_llm_for_condensation = not args.condensed
    need_llm_for_examples = not args.examples
    need_llm_for_setup = need_llm_for_condensation or need_llm_for_examples

    # Variables to hold our artifacts
    condensed = None
    examples = None

    # ==========================================================================
    # SETUP PHASE: Condensation and Example Generation
    # ==========================================================================

    if need_llm_for_setup:
        # We need LLM for at least one setup step
        # Use a SINGLE processor context for all LLM setup operations

        from llm_parallelization.new_processor import NewProcessor

        with NewProcessor(
            gpu_list=args.gpu,
            llm=args.model,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
        ) as processor:
            # ------------------------------------------------------------------
            # Step 1: Condensed Taxonomy
            # ------------------------------------------------------------------
            print("\n" + "=" * 70)
            print("STEP 1: CONDENSED TAXONOMY")
            print("=" * 70)

            if need_llm_for_condensation:
                print(f"Loading raw taxonomy from: {args.taxonomy}")
                with open(args.taxonomy, "r") as f:
                    taxonomy = json.load(f)

                stats = get_taxonomy_stats(taxonomy)
                print(f"  Categories: {stats['categories']}")
                print(f"  Elements: {stats['elements']}")
                print(f"  Attributes: {stats['attributes']}")

                print("\nCondensing taxonomy (this requires LLM)...")
                condensed = condense_taxonomy(taxonomy, processor, verbose=True)
                print("✓ Condensation complete")

                # Save if requested
                if args.save_artifacts:
                    save_path = Path(args.save_artifacts) / "condensed_taxonomy.json"
                    save_condensed(condensed, save_path)
                    print(f"✓ Saved to: {save_path}")
            else:
                print(f"Loading pre-condensed taxonomy from: {args.condensed}")
                condensed = load_condensed(args.condensed)
                print(f"✓ Loaded {len(condensed.categories)} categories")

            # Preview condensed taxonomy
            print_condensed_preview(condensed)

            # ------------------------------------------------------------------
            # Step 2: Training Examples
            # ------------------------------------------------------------------
            print("\n" + "=" * 70)
            print("STEP 2: TRAINING EXAMPLES")
            print("=" * 70)

            if need_llm_for_examples:
                print("Generating examples (this requires LLM)...")
                examples = generate_all_examples(condensed, processor, verbose=True)

                # Save if requested
                if args.save_artifacts:
                    save_path = Path(args.save_artifacts) / "examples.json"
                    save_examples(examples, save_path)
                    print(f"✓ Saved to: {save_path}")
            else:
                print(f"Loading pre-generated examples from: {args.examples}")
                examples = load_examples(args.examples)
                stats = examples.get_stats()
                print(f"✓ Loaded {stats['total']} examples")
                print(f"  Simple: {stats['simple']}")
                print(f"  Multi-category: {stats['multi_category']}")
                print(f"  Negative: {stats['negative']}")

        # Processor is now released after the with block

    else:
        # No LLM needed for setup - just load artifacts

        # ------------------------------------------------------------------
        # Step 1: Load Condensed Taxonomy
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("STEP 1: CONDENSED TAXONOMY")
        print("=" * 70)
        print(f"Loading pre-condensed taxonomy from: {args.condensed}")
        condensed = load_condensed(args.condensed)
        print(f"✓ Loaded {len(condensed.categories)} categories")
        print_condensed_preview(condensed)

        # ------------------------------------------------------------------
        # Step 2: Load Training Examples
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("STEP 2: TRAINING EXAMPLES")
        print("=" * 70)
        print(f"Loading pre-generated examples from: {args.examples}")
        examples = load_examples(args.examples)
        stats = examples.get_stats()
        print(f"✓ Loaded {stats['total']} examples")
        print(f"  Simple: {stats['simple']}")
        print(f"  Multi-category: {stats['multi_category']}")
        print(f"  Negative: {stats['negative']}")

    # ==========================================================================
    # Validate Examples
    # ==========================================================================

    validation = validate_examples(examples, condensed)
    if validation["is_valid"]:
        print("✓ All examples valid")
    else:
        print("⚠ Validation issues:")
        for err in validation["errors"][:5]:
            print(f"  - {err}")
        if len(validation["errors"]) > 5:
            print(f"  ... and {len(validation['errors']) - 5} more")

    # Preview examples
    print_examples_preview(examples)

    # ==========================================================================
    # Step 3: Build Stage 1 Prompt Function (Pure Python - No LLM)
    # ==========================================================================

    print("\n" + "=" * 70)
    print("STEP 3: BUILD STAGE 1 PROMPT (Pure Python)")
    print("=" * 70)

    stage1_prompt = build_stage1_prompt_function(condensed, examples)
    print("✓ Stage 1 prompt function built")

    # Show sample prompt stats
    sample_prompt = stage1_prompt("The WiFi was terrible but the speakers were great!")
    print(
        f"\nSample prompt length: {len(sample_prompt):,} chars (~{len(sample_prompt) // 4:,} tokens)"
    )

    # Export if requested
    if args.export_prompt:
        export_stage1_prompt_module(condensed, examples, args.export_prompt)
        print(f"✓ Exported prompt module to: {args.export_prompt}")

    # ==========================================================================
    # Step 4: Run Classification (Requires LLM)
    # ==========================================================================

    if args.skip_classification:
        print("\n" + "=" * 70)
        print("SKIPPING CLASSIFICATION (--skip-classification)")
        print("=" * 70)
        print("\n🔥 SETUP COMPLETE! Artifacts are ready for use. 🔥")
        return

    print("\n" + "=" * 70)
    print("STEP 4: RUN STAGE 1 CLASSIFICATION")
    print("=" * 70)

    # Determine which comments to classify
    if args.input_file:
        print(f"Loading comments from: {args.input_file}")
        comments_to_classify = load_comments_from_file(
            args.input_file,
            comment_column=args.comment_column,
            sheet_name=args.sheet_name,
            max_comments=args.max_comments,
        )
        print(f"✓ Loaded {len(comments_to_classify)} comments")
        if args.max_comments and len(comments_to_classify) == args.max_comments:
            print(f"  (limited to {args.max_comments} by --max-comments)")
    else:
        comments_to_classify = TEST_COMMENTS
        print(f"Using {len(TEST_COMMENTS)} built-in test comments")
        print("  (use --input-file to classify your own data)")

    print(f"\nClassifying {len(comments_to_classify)} comments...")

    # Generate prompts
    prompts = [stage1_prompt(c) for c in comments_to_classify]

    # Run classification with a fresh processor
    from llm_parallelization.new_processor import NewProcessor

    with NewProcessor(
        gpu_list=args.gpu,
        llm=args.model,
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.9,
    ) as processor:
        responses = processor.process_with_schema(
            prompts=prompts,
            schema=CategoryDetectionOutput,
            batch_size=args.batch_size,
            guided_config={
                "temperature": 0.1,
                "top_k": 50,
                "top_p": 0.95,
                "max_tokens": 500,
            },
        )

        results = processor.parse_results_with_schema(
            schema=CategoryDetectionOutput,
            responses=responses,
            validate=True,
        )

    # ==========================================================================
    # Step 5: Display Results
    # ==========================================================================

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    # Show first N results (don't spam terminal with hundreds)
    max_display = 20
    display_count = min(len(comments_to_classify), max_display)

    for i, (comment, result) in enumerate(
        zip(comments_to_classify[:display_count], results[:display_count]), 1
    ):
        preview = comment[:70] + "..." if len(comment) > 70 else comment
        print(f"\n{i}. {preview}")

        if result is None:
            print("   ❌ FAILED TO PARSE")
            continue

        if not result.has_classifiable_content:
            print("   ⚠️  No classifiable content")
            print(f"   Reasoning: {result.reasoning}")
        else:
            print(f"   ✓ Categories: {result.categories_present}")
            print(f"   Reasoning: {result.reasoning}")

    if len(comments_to_classify) > max_display:
        print(f"\n... and {len(comments_to_classify) - max_display} more (see saved results)")

    # ==========================================================================
    # Summary Statistics
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r is not None)
    classifiable = sum(1 for r in results if r and r.has_classifiable_content)
    non_classifiable = sum(1 for r in results if r and not r.has_classifiable_content)
    failed = len(results) - successful

    print(f"Total comments: {len(results)}")
    print(f"Successfully parsed: {successful}")
    print(f"Classifiable: {classifiable}")
    print(f"Non-classifiable: {non_classifiable}")
    print(f"Failed to parse: {failed}")

    # Category distribution
    category_counts = Counter()
    for result in results:
        if result and result.has_classifiable_content:
            for cat in result.categories_present:
                category_counts[cat] += 1

    if category_counts:
        print("\nCategory distribution:")
        for cat, count in category_counts.most_common():
            pct = 100 * count / classifiable if classifiable > 0 else 0
            print(f"  {cat}: {count} ({pct:.1f}%)")

    # ==========================================================================
    # Save Results (Optional)
    # ==========================================================================

    if args.save_artifacts:
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        # Create results data
        results_data = []
        for comment, result in zip(comments_to_classify, results):
            if result is None:
                results_data.append(
                    {
                        "comment": comment,
                        "has_classifiable_content": None,
                        "categories_present": None,
                        "reasoning": "FAILED TO PARSE",
                    }
                )
            else:
                results_data.append(
                    {
                        "comment": comment,
                        "has_classifiable_content": result.has_classifiable_content,
                        "categories_present": result.categories_present,
                        "reasoning": result.reasoning,
                    }
                )

        # Save as JSON
        results_path = Path(args.save_artifacts) / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"✓ Saved results to: {results_path}")

        # Also save as CSV if pandas available
        try:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "comment": r["comment"],
                        "has_classifiable_content": r["has_classifiable_content"],
                        "categories_present": ", ".join(r["categories_present"])
                        if r["categories_present"]
                        else "",
                        "reasoning": r["reasoning"],
                    }
                    for r in results_data
                ]
            )
            csv_path = Path(args.save_artifacts) / "classification_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved CSV to: {csv_path}")
        except ImportError:
            print("  (pandas not available, skipping CSV export)")

    print("\n" + "=" * 70)
    print("🔥 STAGE 1 PIPELINE TEST COMPLETE! 🔥")
    print("=" * 70)
    print("\nYour dynamically generated Stage 1 prompt is working!")
    print("\nNext steps:")
    print("  1. Review the classification results")
    print("  2. Tweak condensed descriptions or examples if needed")
    print("  3. Build Stage 2 prompts for element extraction!")


if __name__ == "__main__":
    main()
