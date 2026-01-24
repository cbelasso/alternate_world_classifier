"""
Test Full Classification Pipeline (Stage 1 + Stage 2)

This script demonstrates the complete two-stage classification workflow:
    1. Load/condense taxonomy
    2. Load/generate examples
    3. Build Stage 1 and Stage 2 prompts
    4. Run full classification pipeline
    5. Export results as DataFrame/CSV/JSON

Usage:
    # Full pipeline from taxonomy (generates everything fresh)
    python test_full_pipeline.py --taxonomy /path/to/taxonomy.json

    # Use pre-built artifacts (recommended for production)
    python test_full_pipeline.py \\
        --condensed artifacts/condensed_taxonomy.json \\
        --examples artifacts/examples.json \\
        --input-file comments.xlsx \\
        --comment-column "feedback"

    # Save all artifacts and prompts
    python test_full_pipeline.py \\
        --taxonomy /path/to/taxonomy.json \\
        --save-artifacts artifacts/ \\
        --export-prompts prompts/

    # Quick test with limited comments
    python test_full_pipeline.py \\
        --condensed artifacts/condensed_taxonomy.json \\
        --examples artifacts/examples.json \\
        --input-file comments.xlsx \\
        --max-comments 20

Supported input file formats:
    - CSV (.csv): Requires --comment-column
    - Excel (.xlsx, .xls): Requires --comment-column, optional --sheet-name
    - JSON (.json): List of strings, or list of objects with --comment-column key
    - Text (.txt): One comment per line
"""

import argparse
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
    """Load comments from a file (CSV, Excel, JSON, or TXT)."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(filepath)
        if comment_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(f"Column '{comment_column}' not found. Available: {available}")
        comments = df[comment_column].dropna().astype(str).tolist()

    elif suffix in [".xlsx", ".xls"]:
        import pandas as pd

        df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
        if comment_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(f"Column '{comment_column}' not found. Available: {available}")
        comments = df[comment_column].dropna().astype(str).tolist()

    elif suffix == ".json":
        with open(filepath, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                comments = data
            elif all(isinstance(item, dict) for item in data):
                if data and comment_column in data[0]:
                    comments = [
                        item[comment_column] for item in data if item.get(comment_column)
                    ]
                else:
                    available = ", ".join(data[0].keys()) if data else "none"
                    raise ValueError(
                        f"Key '{comment_column}' not found. Available: {available}"
                    )
            else:
                raise ValueError("JSON must be list of strings or list of objects")
        elif isinstance(data, dict) and comment_column in data:
            comments = data[comment_column]
        else:
            raise ValueError("JSON structure not recognized")

    elif suffix == ".txt":
        with open(filepath, "r") as f:
            comments = [line.strip() for line in f if line.strip()]

    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .csv, .xlsx, .xls, .json, or .txt")

    if max_comments:
        comments = comments[:max_comments]

    return comments


# =============================================================================
# Test Comments (fallback when no input file provided)
# =============================================================================

TEST_COMMENTS = [
    # Single category - simple
    "The networking sessions were fantastic and I made great connections with peers.",
    "The WiFi kept dropping during sessions and the room was too cold.",
    "The keynote speaker was brilliant and really knew their stuff.",
    "Registration was smooth and the staff were very helpful.",
    "I came away with so many actionable insights to implement.",
    # Multi-category
    "The keynote speaker was brilliant and the presentation on ML was very insightful.",
    "Great conference overall but the WiFi was terrible. The Explorance team was helpful though.",
    "Loved the workshops but wish there were more networking opportunities.",
    "The venue was beautiful but the sessions ran over time. Speakers were excellent!",
    # Edge cases
    "Everything was perfect!",
    "Data integrity never goes out of style.",  # Non-classifiable
    "The only constant in life is change.",  # Non-classifiable
]


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test Full Classification Pipeline (Stage 1 + Stage 2)"
    )

    # Input sources
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

    # Input data
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to file with comments to classify",
    )
    parser.add_argument(
        "--comment-column",
        type=str,
        default="comment",
        help="Column name containing comments (default: 'comment')",
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
        help="Maximum number of comments to classify",
    )

    # Output options
    parser.add_argument(
        "--save-artifacts",
        type=str,
        help="Directory to save generated artifacts and results",
    )
    parser.add_argument(
        "--export-prompts",
        type=str,
        help="Directory to export prompt modules",
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
        "--batch-size",
        type=int,
        default=25,
        help="Batch size for classification",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip classification (just build artifacts and prompts)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.taxonomy and not (args.condensed and args.examples):
        parser.error("Either --taxonomy OR both --condensed and --examples required")

    # Create output directories
    if args.save_artifacts:
        Path(args.save_artifacts).mkdir(parents=True, exist_ok=True)
    if args.export_prompts:
        Path(args.export_prompts).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Import classifier components
    # =========================================================================

    print("=" * 70)
    print("FULL CLASSIFICATION PIPELINE")
    print("=" * 70)

    from classifier import (
        # Schemas
        CategoryDetectionOutput,
        # Orchestrator
        ClassificationOrchestrator,
        FinalClassificationOutput,
        # Model building
        build_models_from_taxonomy,
        # Stage 1 Prompts
        build_stage1_prompt_function,
        # Stage 2 Prompts
        build_stage2_prompt_functions,
        # Condensation
        condense_taxonomy,
        export_stage1_prompt_module,
        export_stage2_prompt_module,
        # Examples
        generate_all_examples,
        # Utilities
        get_taxonomy_stats,
        load_condensed,
        load_examples,
        print_condensed_preview,
        print_examples_preview,
        print_stage2_prompts_preview,
        save_condensed,
        save_examples,
        validate_examples,
    )

    print("✓ Imports successful")

    # =========================================================================
    # Determine what needs LLM
    # =========================================================================

    need_llm_for_condensation = not args.condensed
    need_llm_for_examples = not args.examples
    need_llm_for_setup = need_llm_for_condensation or need_llm_for_examples

    # Variables
    condensed = None
    examples = None
    taxonomy = None

    # =========================================================================
    # PHASE 1: Build/Load Artifacts
    # =========================================================================

    if need_llm_for_setup:
        from llm_parallelization.new_processor import NewProcessor

        with NewProcessor(
            gpu_list=args.gpu,
            llm=args.model,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
        ) as processor:
            # -----------------------------------------------------------------
            # Step 1: Condensed Taxonomy
            # -----------------------------------------------------------------
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

                print("\nCondensing taxonomy...")
                condensed = condense_taxonomy(taxonomy, processor, verbose=True)

                if args.save_artifacts:
                    save_path = Path(args.save_artifacts) / "condensed_taxonomy.json"
                    save_condensed(condensed, save_path)
                    print(f"✓ Saved to: {save_path}")
            else:
                print(f"Loading pre-condensed from: {args.condensed}")
                condensed = load_condensed(args.condensed)
                print(f"✓ Loaded {len(condensed.categories)} categories")

            print_condensed_preview(condensed)

            # -----------------------------------------------------------------
            # Step 2: Training Examples
            # -----------------------------------------------------------------
            print("\n" + "=" * 70)
            print("STEP 2: TRAINING EXAMPLES")
            print("=" * 70)

            if need_llm_for_examples:
                print("Generating examples...")
                examples = generate_all_examples(condensed, processor, verbose=True)

                if args.save_artifacts:
                    save_path = Path(args.save_artifacts) / "examples.json"
                    save_examples(examples, save_path)
                    print(f"✓ Saved to: {save_path}")
            else:
                print(f"Loading pre-generated from: {args.examples}")
                examples = load_examples(args.examples)
                stats = examples.get_stats()
                print(f"✓ Loaded {stats['total']} examples")

    else:
        # No LLM needed - just load
        print("\n" + "=" * 70)
        print("STEP 1: LOAD CONDENSED TAXONOMY")
        print("=" * 70)
        condensed = load_condensed(args.condensed)
        print(f"✓ Loaded {len(condensed.categories)} categories")
        print_condensed_preview(condensed)

        print("\n" + "=" * 70)
        print("STEP 2: LOAD TRAINING EXAMPLES")
        print("=" * 70)
        examples = load_examples(args.examples)
        stats = examples.get_stats()
        print(f"✓ Loaded {stats['total']} examples")

    # Load taxonomy for model building if needed
    if args.taxonomy and taxonomy is None:
        with open(args.taxonomy, "r") as f:
            taxonomy = json.load(f)

    # Validate examples
    validation = validate_examples(examples, condensed)
    if validation["is_valid"]:
        print("✓ All examples valid")
    else:
        print(f"⚠ Validation issues: {len(validation['errors'])} errors")

    print_examples_preview(examples)

    # =========================================================================
    # PHASE 2: Build Prompts
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 3: BUILD PROMPTS (Pure Python)")
    print("=" * 70)

    # Stage 1
    stage1_prompt = build_stage1_prompt_function(condensed, examples)
    print("✓ Stage 1 prompt function built")

    # Stage 2
    stage2_prompts = build_stage2_prompt_functions(condensed, examples)
    print(f"✓ Stage 2 prompt functions built ({len(stage2_prompts)} categories)")

    # Preview
    print_stage2_prompts_preview(condensed, examples)

    # Export prompts if requested
    if args.export_prompts:
        export_dir = Path(args.export_prompts)

        # Stage 1
        stage1_path = export_dir / "stage1_prompt.py"
        export_stage1_prompt_module(condensed, examples, stage1_path)
        print(f"✓ Exported Stage 1: {stage1_path}")

        # Stage 2
        stage2_path = export_dir / "stage2_prompts.py"
        export_stage2_prompt_module(condensed, examples, stage2_path)
        print(f"✓ Exported Stage 2: {stage2_path}")

    # =========================================================================
    # PHASE 3: Build Orchestrator
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 4: BUILD ORCHESTRATOR")
    print("=" * 70)

    # Build dynamic schemas if we have taxonomy
    category_to_schema = None
    if taxonomy:
        models = build_models_from_taxonomy(taxonomy)
        category_to_schema = models["category_to_schema"]
        print(f"✓ Built dynamic schemas for {len(category_to_schema)} categories")

    orchestrator = ClassificationOrchestrator(
        stage1_prompt=stage1_prompt,
        stage2_prompts=stage2_prompts,
        category_to_schema=category_to_schema,
    )
    print("✓ Orchestrator ready")

    # =========================================================================
    # PHASE 4: Run Classification
    # =========================================================================

    if args.skip_classification:
        print("\n" + "=" * 70)
        print("SKIPPING CLASSIFICATION (--skip-classification)")
        print("=" * 70)
        print("\n🔥 ARTIFACTS AND PROMPTS READY! 🔥")
        return

    print("\n" + "=" * 70)
    print("STEP 5: RUN CLASSIFICATION")
    print("=" * 70)

    # Load comments
    if args.input_file:
        print(f"Loading comments from: {args.input_file}")
        comments = load_comments_from_file(
            args.input_file,
            comment_column=args.comment_column,
            sheet_name=args.sheet_name,
            max_comments=args.max_comments,
        )
        print(f"✓ Loaded {len(comments)} comments")
    else:
        comments = TEST_COMMENTS
        print(f"Using {len(TEST_COMMENTS)} built-in test comments")

    # Run classification
    from llm_parallelization.new_processor import NewProcessor

    with NewProcessor(
        gpu_list=args.gpu,
        llm=args.model,
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.9,
    ) as processor:
        results = orchestrator.classify_comments(
            comments,
            processor,
            batch_size=args.batch_size,
            verbose=True,
        )

    # =========================================================================
    # PHASE 5: Display and Save Results
    # =========================================================================

    orchestrator.print_results_summary(results, max_display=15)

    if args.save_artifacts:
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        # Save as DataFrame/CSV
        df = orchestrator.results_to_dataframe(results)
        csv_path = Path(args.save_artifacts) / "classification_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")

        # Save as JSON
        json_data = orchestrator.results_to_json(results)
        json_path = Path(args.save_artifacts) / "classification_results.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")

        # Save stats
        stats = orchestrator.get_classification_stats(results)
        stats_path = Path(args.save_artifacts) / "classification_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved stats: {stats_path}")

    print("\n" + "=" * 70)
    print("🔥 FULL PIPELINE COMPLETE! 🔥")
    print("=" * 70)


if __name__ == "__main__":
    main()
