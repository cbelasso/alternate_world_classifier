#!/usr/bin/env python
"""
Full Classification Pipeline with YAML Artifacts

End-to-end pipeline using the YAML-based artifact system:
1. Scaffold YAML folder structure from schema
2. Generate content (descriptions, rules, examples) via LLM
3. Build prompts from YAML artifacts
4. Run multi-stage classification
5. Save results

Usage:
    python test_full_pipeline_yaml.py \
        --taxonomy schema.json \
        --gpu 0 1 \
        --artifacts output/artifacts/ \
        --export-prompts output/prompts/ \
        --input-file comments.xlsx \
        --comment-column "comment" \
        --stages 3

Workflow:
    schema.json
        → scaffold_artifacts()           # Create folder structure
        → generate_artifact_content()    # LLM fills descriptions/rules/examples
        → build_classifier_objects()     # Load YAML → Python
        → build prompts (Stage 1, 2, 3)
        → export_prompts_hierarchical()  # Save Python prompt files
        → run classification
        → save results
"""

import argparse
import json
import os
from pathlib import Path
import sys


def load_comments_from_file(
    filepath: str,
    comment_column: str = "comment",
    sheet_name: str = None,
    max_comments: int = None,
) -> list:
    """Load comments from Excel or CSV file."""
    filepath = Path(filepath)

    if filepath.suffix in (".xlsx", ".xls"):
        import pandas as pd

        df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
    elif filepath.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")

    if comment_column not in df.columns:
        available = list(df.columns)
        raise ValueError(f"Column '{comment_column}' not found. Available: {available}")

    comments = df[comment_column].dropna().astype(str).tolist()

    if max_comments:
        comments = comments[:max_comments]

    return comments


def main():
    parser = argparse.ArgumentParser(
        description="Full Classification Pipeline with YAML Artifacts"
    )

    # Required
    parser.add_argument(
        "--taxonomy",
        "-t",
        type=str,
        required=True,
        help="Path to taxonomy schema JSON file",
    )

    # GPU
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device(s) to use",
    )

    # Artifacts
    parser.add_argument(
        "--artifacts",
        "-a",
        type=str,
        default="artifacts/",
        help="Directory for YAML artifacts",
    )

    # Prompts export
    parser.add_argument(
        "--export-prompts",
        type=str,
        help="Directory to export Python prompt files",
    )

    # Input
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        help="Input file with comments (Excel or CSV)",
    )
    parser.add_argument(
        "--comment-column",
        type=str,
        default="comment",
        help="Column name containing comments",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        help="Sheet name for Excel files",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        help="Maximum comments to process",
    )

    # Pipeline options
    parser.add_argument(
        "--stages",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of classification stages (2=element, 3=attribute)",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for results",
    )

    # Skip options
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip content generation (use existing artifacts)",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip classification (artifacts and prompts only)",
    )
    parser.add_argument(
        "--skip-attribute-content",
        action="store_true",
        help="Skip generating content for attribute-level YAML files (can save time with many attributes)",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="casperhansen/mistral-nemo-instruct-2407-awq",
        help="Model to use for LLM processing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Batch size for classification",
    )

    args = parser.parse_args()

    # Validate paths
    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.exists():
        print(f"ERROR: Taxonomy file not found: {taxonomy_path}")
        sys.exit(1)

    artifacts_dir = Path(args.artifacts)
    output_dir = Path(args.output) if args.output else artifacts_dir.parent / "output"

    print("\n" + "=" * 70)
    print("FULL CLASSIFICATION PIPELINE (YAML ARTIFACTS)")
    print("=" * 70)
    print(f"Taxonomy: {taxonomy_path}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Output: {output_dir}")
    print(f"Stages: {args.stages}")
    print(f"GPU(s): {args.gpu}")

    # Load taxonomy
    with open(taxonomy_path, "r") as f:
        taxonomy = json.load(f)

    # Import classifier modules
    from classifier import (
        # Orchestrator
        ClassificationOrchestrator,
        build_classifier_objects,
        # Model building
        build_models_from_taxonomy,
        # Prompts
        build_stage1_prompt_function,
        build_stage2_prompt_functions,
        build_stage3_schemas_from_taxonomy,
        generate_artifact_content,
        # Artifacts
        scaffold_artifacts,
        validate_artifacts,
    )
    from classifier.prompts.exporter import export_prompts_hierarchical_with_stage3
    from classifier.prompts.stage3 import build_stage3_prompt_functions

    # =========================================================================
    # PHASE 1: Initialize Processor
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 1: INITIALIZE PROCESSOR")
    print("=" * 70)

    from llm_parallelization.new_processor import NewProcessor

    # We'll use context manager for the processor
    # But we need it for both generation and classification phases

    processor_config = {
        "gpu_list": args.gpu,
        "llm": args.model,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
    }
    print(f"✓ Processor config ready: {args.model} on GPU {args.gpu}")

    # =========================================================================
    # PHASE 2: Scaffold Artifacts
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: SCAFFOLD ARTIFACTS")
    print("=" * 70)

    if not artifacts_dir.exists():
        result = scaffold_artifacts(
            schema_path=taxonomy_path,
            output_dir=artifacts_dir,
            verbose=True,
        )
        print(f"✓ Created {result['files_created']} YAML files")
    else:
        print(f"✓ Artifacts directory exists: {artifacts_dir}")
        # Validate existing artifacts
        validation = validate_artifacts(
            artifacts_dir=artifacts_dir,
            schema_path=taxonomy_path,
            verbose=False,
        )
        if not validation["is_valid"]:
            print(f"⚠ Validation issues: {len(validation['errors'])} errors")
            for err in validation["errors"][:5]:
                print(f"  - {err}")

    # =========================================================================
    # PHASE 3: Generate Content
    # =========================================================================

    if not args.skip_generation:
        print("\n" + "=" * 70)
        print("STEP 3: GENERATE CONTENT")
        print("=" * 70)

        with NewProcessor(**processor_config) as processor:
            stats = generate_artifact_content(
                artifacts_dir=artifacts_dir,
                schema_path=taxonomy_path,
                processor=processor,
                generate_attribute_content=not args.skip_attribute_content,
                verbose=True,
            )
        print(
            f"✓ Generated: {stats['descriptions']} descriptions, {stats['rules']} rules, {stats['examples']} examples"
        )
    else:
        print("\n" + "=" * 70)
        print("STEP 3: SKIPPING GENERATION (--skip-generation)")
        print("=" * 70)

    # =========================================================================
    # PHASE 4: Build Classifier Objects from YAML
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 4: BUILD CLASSIFIER OBJECTS")
    print("=" * 70)

    condensed, examples, rules = build_classifier_objects(
        artifacts_dir=artifacts_dir,
        verbose=True,
    )

    # =========================================================================
    # PHASE 5: Build Prompts
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 5: BUILD PROMPTS")
    print("=" * 70)

    # Stage 1
    stage1_prompt = build_stage1_prompt_function(condensed, examples, rules)
    print("✓ Stage 1 prompt ready")

    # Stage 2
    stage2_prompts = build_stage2_prompt_functions(condensed, examples, rules)
    print(f"✓ Stage 2 prompts ready ({len(stage2_prompts)} categories)")

    # Stage 3 (if enabled)
    stage3_prompts = {}
    if args.stages == 3:
        stage3_prompts = build_stage3_prompt_functions(condensed, examples, taxonomy, rules)
        total_s3 = sum(len(elems) for elems in stage3_prompts.values())
        print(f"✓ Stage 3 prompts ready ({total_s3} elements)")

    # =========================================================================
    # PHASE 6: Export Prompts (optional)
    # =========================================================================

    if args.export_prompts:
        print("\n" + "=" * 70)
        print("STEP 6: EXPORT PROMPTS")
        print("=" * 70)

        prompts_dir = Path(args.export_prompts)

        if args.stages == 3:
            result = export_prompts_hierarchical_with_stage3(
                condensed=condensed,
                examples=examples,
                rules=rules,
                taxonomy=taxonomy,
                output_dir=prompts_dir,
                force_overwrite=False,
                verbose=True,
            )
        else:
            from classifier import export_prompts_hierarchical

            result = export_prompts_hierarchical(
                condensed=condensed,
                examples=examples,
                rules=rules,
                output_dir=prompts_dir,
                verbose=True,
            )

        print(f"✓ Exported {result['total_files']} prompt files")

    # =========================================================================
    # PHASE 7: Build Orchestrator
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 7: BUILD ORCHESTRATOR")
    print("=" * 70)

    # Build schemas
    category_to_schema = None
    element_to_schema = None

    models = build_models_from_taxonomy(taxonomy)
    category_to_schema = models["category_to_schema"]
    print(f"✓ Stage 2 schemas: {len(category_to_schema)} categories")

    if args.stages == 3:
        element_to_schema = build_stage3_schemas_from_taxonomy(taxonomy)
        total_elem = sum(len(e) for e in element_to_schema.values())
        print(f"✓ Stage 3 schemas: {total_elem} elements")

    orchestrator = ClassificationOrchestrator(
        stage1_prompt=stage1_prompt,
        stage2_prompts=stage2_prompts,
        stage3_prompts=stage3_prompts if args.stages == 3 else None,
        category_to_schema=category_to_schema,
        element_to_schema=element_to_schema,
    )
    print(f"✓ Orchestrator ready ({args.stages}-stage)")

    # =========================================================================
    # PHASE 8: Run Classification
    # =========================================================================

    if args.skip_classification:
        print("\n" + "=" * 70)
        print("SKIPPING CLASSIFICATION (--skip-classification)")
        print("=" * 70)
        print("✓ Artifacts and prompts ready!")
        return

    if not args.input_file:
        print("\n" + "=" * 70)
        print("NO INPUT FILE - PIPELINE COMPLETE")
        print("=" * 70)
        print("✓ Artifacts and prompts ready!")
        print("Run with --input-file to classify comments")
        return

    print("\n" + "=" * 70)
    print(f"STEP 8: CLASSIFY COMMENTS ({args.stages}-STAGE)")
    print("=" * 70)

    # Load comments
    comments = load_comments_from_file(
        args.input_file,
        comment_column=args.comment_column,
        sheet_name=args.sheet_name,
        max_comments=args.max_comments,
    )
    print(f"✓ Loaded {len(comments)} comments")

    # Run classification with processor
    with NewProcessor(
        gpu_list=args.gpu,
        llm=args.model,
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.9,
    ) as processor:
        if args.stages == 3:
            results = orchestrator.classify_comments_3stage(
                comments=comments,
                processor=processor,
                batch_size=args.batch_size,
                verbose=True,
            )
        else:
            results = orchestrator.classify_comments(
                comments=comments,
                processor=processor,
                batch_size=args.batch_size,
                verbose=True,
            )

    # =========================================================================
    # PHASE 9: Save Results
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 9: SAVE RESULTS")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    if args.stages == 3:
        df = orchestrator.results_to_dataframe_3stage(results)
    else:
        df = orchestrator.results_to_dataframe(results)

    # Save CSV
    csv_path = output_dir / f"classification_results_{args.stages}stage.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

    # Save JSON
    json_path = output_dir / f"classification_results_{args.stages}stage.json"
    json_results = [r.model_dump() for r in results]
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved: {json_path}")

    # Stats
    stats = orchestrator.get_classification_stats(results)
    stats_path = output_dir / f"classification_stats_{args.stages}stage.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {stats_path}")

    # Consensus analysis (3-stage only)
    if args.stages == 3:
        consensus_data = {
            "total": len([c for r in results for c in r.classifications]),
            "consensus": len(
                [c for r in results for c in r.classifications if c.sentiment_consensus]
            ),
            "mismatches": [
                {
                    "comment": r.original_comment[:100],
                    "category": c.category,
                    "element": c.element,
                    "element_sentiment": c.element_sentiment,
                    "attribute": c.attribute,
                    "attribute_sentiment": c.attribute_sentiment,
                }
                for r in results
                for c in r.classifications
                if not c.sentiment_consensus
            ],
        }
        consensus_path = output_dir / "sentiment_consensus.json"
        with open(consensus_path, "w") as f:
            json.dump(consensus_data, f, indent=2)
        print(f"✓ Saved: {consensus_path}")

        rate = consensus_data["consensus"] / max(consensus_data["total"], 1) * 100
        print(f"\n📊 Sentiment Consensus: {rate:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("🔥 PIPELINE COMPLETE! 🔥")
    print("=" * 70)
    print(f"Artifacts: {artifacts_dir}")
    if args.export_prompts:
        print(f"Prompts: {args.export_prompts}")
    print(f"Results: {output_dir}")
    orchestrator.print_results_summary(results)


if __name__ == "__main__":
    main()
