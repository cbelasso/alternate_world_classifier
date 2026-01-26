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

OR load prompts directly from Python files:
    python test_full_pipeline_yaml.py \
        --taxonomy schema.json \
        --load-python-prompts-directly output/prompts/ \
        --input-file comments.xlsx \
        --stages 1
"""

import argparse
import importlib.util
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

        df = pd.read_csv(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")

    if comment_column not in df.columns:
        available = list(df.columns)
        raise ValueError(f"Column '{comment_column}' not found. Available: {available}")

    comments = df[comment_column].dropna().astype(str).tolist()

    if max_comments:
        comments = comments[:max_comments]

    return comments


def load_prompts_from_python_files(
    prompts_dir: Path, taxonomy: dict, stages: int, verbose: bool = True
):
    """
    Load prompt functions directly from exported Python files.

    Args:
        prompts_dir: Directory containing exported prompt files
        taxonomy: Taxonomy dict (needed for building schemas)
        stages: Number of stages (1, 2, or 3)
        verbose: Print loading information

    Returns:
        Tuple of (stage1_prompt, stage2_prompts, stage3_prompts)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("LOADING PROMPTS FROM PYTHON FILES")
        print("=" * 70)
        print(f"Prompts directory: {prompts_dir}")

    # Stage 1: Find and load the stage1 prompt file
    stage1_dir = prompts_dir / "stage1"
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Stage 1 directory not found: {stage1_dir}")

    # Find the first .py file that's not __init__.py
    stage1_files = [f for f in stage1_dir.glob("*.py") if f.name != "__init__.py"]
    if not stage1_files:
        raise FileNotFoundError(f"No Stage 1 prompt file found in {stage1_dir}")

    stage1_path = stage1_files[0]  # Use the first one found

    spec = importlib.util.spec_from_file_location("stage1_module", stage1_path)
    stage1_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1_module)

    # Try to find the prompt function (could be named stage1_prompt or similar)
    if hasattr(stage1_module, "stage1_prompt"):
        stage1_prompt = stage1_module.stage1_prompt
    elif hasattr(stage1_module, "PROMPT"):
        stage1_prompt = stage1_module.PROMPT
    else:
        # Try to find any function that looks like a prompt function
        prompt_funcs = [
            name
            for name in dir(stage1_module)
            if "prompt" in name.lower() and callable(getattr(stage1_module, name))
        ]
        if prompt_funcs:
            stage1_prompt = getattr(stage1_module, prompt_funcs[0])
        else:
            raise AttributeError(f"No prompt function found in {stage1_path}")

    if verbose:
        print(f"✓ Loaded Stage 1 prompt from {stage1_path.name}")

    # Stage 2: Load stage2/<category>.py files
    stage2_prompts = {}
    if stages >= 2:
        stage2_dir = prompts_dir / "stage2"
        if not stage2_dir.exists():
            raise FileNotFoundError(f"Stage 2 directory not found: {stage2_dir}")

        for category_file in stage2_dir.glob("*.py"):
            category_name = category_file.stem

            spec = importlib.util.spec_from_file_location(
                f"stage2_{category_name}", category_file
            )
            category_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(category_module)

            # The function name is stage2_prompt_{category_name}
            func_name = f"stage2_prompt_{category_name}"
            if hasattr(category_module, func_name):
                stage2_prompts[category_name] = getattr(category_module, func_name)
            else:
                raise AttributeError(f"Function {func_name} not found in {category_file}")

        if verbose:
            print(f"✓ Loaded {len(stage2_prompts)} Stage 2 prompts from {stage2_dir}")

    # Stage 3: Load stage3/<category>/<element>.py files
    stage3_prompts = {}
    if stages >= 3:
        stage3_dir = prompts_dir / "stage3"
        if not stage3_dir.exists():
            raise FileNotFoundError(f"Stage 3 directory not found: {stage3_dir}")

        for category_dir in stage3_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name
            stage3_prompts[category_name] = {}

            for element_file in category_dir.glob("*.py"):
                element_name = element_file.stem

                spec = importlib.util.spec_from_file_location(
                    f"stage3_{category_name}_{element_name}", element_file
                )
                element_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(element_module)

                # The function name is stage3_prompt_{category}_{element}
                func_name = f"stage3_prompt_{category_name}_{element_name}"
                if hasattr(element_module, func_name):
                    stage3_prompts[category_name][element_name] = getattr(
                        element_module, func_name
                    )
                else:
                    raise AttributeError(f"Function {func_name} not found in {element_file}")

        total_s3 = sum(len(elems) for elems in stage3_prompts.values())
        if verbose:
            print(f"✓ Loaded {total_s3} Stage 3 prompts from {stage3_dir}")

    return stage1_prompt, stage2_prompts, stage3_prompts


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

    # NEW: Load prompts directly from Python files
    parser.add_argument(
        "--load-python-prompts-directly",
        type=str,
        help="Load prompts from existing Python files (skips YAML generation). Provide path to prompts directory.",
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
        choices=[1, 2, 3],
        default=2,
        help="Number of classification stages (1=category only, 2=element, 3=attribute)",
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

    # Check if loading prompts directly
    load_from_python = args.load_python_prompts_directly is not None
    prompts_dir = Path(args.load_python_prompts_directly) if load_from_python else None

    print("\n" + "=" * 70)
    print("FULL CLASSIFICATION PIPELINE (YAML ARTIFACTS)")
    print("=" * 70)
    print(f"Taxonomy: {taxonomy_path}")
    if load_from_python:
        print("Mode: LOAD PROMPTS FROM PYTHON FILES")
        print(f"Prompts: {prompts_dir}")
    else:
        print("Mode: BUILD FROM YAML ARTIFACTS")
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
    # PHASE 1: Initialize Processor (if needed)
    # =========================================================================

    processor_config = {
        "gpu_list": args.gpu,
        "llm": args.model,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
    }

    # =========================================================================
    # BRANCHING: Load from Python OR Build from YAML
    # =========================================================================

    if load_from_python:
        # =====================================================================
        # PATH A: LOAD PROMPTS FROM PYTHON FILES
        # =====================================================================

        if not prompts_dir.exists():
            print(f"ERROR: Prompts directory not found: {prompts_dir}")
            sys.exit(1)

        # Load prompts from Python files
        stage1_prompt, stage2_prompts, stage3_prompts = load_prompts_from_python_files(
            prompts_dir=prompts_dir,
            taxonomy=taxonomy,
            stages=args.stages,
            verbose=True,
        )

        # We still need to build schemas from taxonomy
        print("\n" + "=" * 70)
        print("BUILD SCHEMAS FROM TAXONOMY")
        print("=" * 70)

        models = build_models_from_taxonomy(taxonomy)
        category_to_schema = models["category_to_schema"]
        print(f"✓ Stage 2 schemas: {len(category_to_schema)} categories")

        element_to_schema = None
        if args.stages == 3:
            element_to_schema = build_stage3_schemas_from_taxonomy(taxonomy)
            total_elem = sum(len(e) for e in element_to_schema.values())
            print(f"✓ Stage 3 schemas: {total_elem} elements")

    else:
        # =====================================================================
        # PATH B: BUILD FROM YAML ARTIFACTS (ORIGINAL WORKFLOW)
        # =====================================================================

        print("\n" + "=" * 70)
        print("STEP 1: INITIALIZE PROCESSOR")
        print("=" * 70)
        print(f"✓ Processor config ready: {args.model} on GPU {args.gpu}")

        # =====================================================================
        # PHASE 2: Scaffold Artifacts
        # =====================================================================

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

        # =====================================================================
        # PHASE 3: Generate Content
        # =====================================================================

        if not args.skip_generation:
            print("\n" + "=" * 70)
            print("STEP 3: GENERATE CONTENT")
            print("=" * 70)

            from llm_parallelization.new_processor import NewProcessor

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

        # =====================================================================
        # PHASE 4: Build Classifier Objects from YAML
        # =====================================================================

        print("\n" + "=" * 70)
        print("STEP 4: BUILD CLASSIFIER OBJECTS")
        print("=" * 70)

        condensed, examples, rules = build_classifier_objects(
            artifacts_dir=artifacts_dir,
            verbose=True,
        )

        # =====================================================================
        # PHASE 5: Build Prompts
        # =====================================================================

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

        # =====================================================================
        # PHASE 6: Export Prompts (optional)
        # =====================================================================

        if args.export_prompts:
            print("\n" + "=" * 70)
            print("STEP 6: EXPORT PROMPTS")
            print("=" * 70)

            prompts_export_dir = Path(args.export_prompts)

            if args.stages == 3:
                result = export_prompts_hierarchical_with_stage3(
                    condensed=condensed,
                    examples=examples,
                    rules=rules,
                    taxonomy=taxonomy,
                    output_dir=prompts_export_dir,
                    force_overwrite=False,
                    verbose=True,
                )
            else:
                from classifier import export_prompts_hierarchical

                result = export_prompts_hierarchical(
                    condensed=condensed,
                    examples=examples,
                    rules=rules,
                    output_dir=prompts_export_dir,
                    verbose=True,
                )

            print(f"✓ Exported {result['total_files']} prompt files")

        # =====================================================================
        # PHASE 7: Build Schemas
        # =====================================================================

        print("\n" + "=" * 70)
        print("STEP 7: BUILD SCHEMAS")
        print("=" * 70)

        models = build_models_from_taxonomy(taxonomy)
        category_to_schema = models["category_to_schema"]
        print(f"✓ Stage 2 schemas: {len(category_to_schema)} categories")

        element_to_schema = None
        if args.stages == 3:
            element_to_schema = build_stage3_schemas_from_taxonomy(taxonomy)
            total_elem = sum(len(e) for e in element_to_schema.values())
            print(f"✓ Stage 3 schemas: {total_elem} elements")

    # =========================================================================
    # PHASE 8: Build Orchestrator (COMMON PATH)
    # =========================================================================

    print("\n" + "=" * 70)
    print("BUILD ORCHESTRATOR")
    print("=" * 70)

    orchestrator = ClassificationOrchestrator(
        stage1_prompt=stage1_prompt,
        stage2_prompts=stage2_prompts,
        stage3_prompts=stage3_prompts if args.stages == 3 else None,
        category_to_schema=category_to_schema,
        element_to_schema=element_to_schema,
    )
    print(f"✓ Orchestrator ready ({args.stages}-stage)")

    # =========================================================================
    # PHASE 9: Run Classification
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
    print(f"RUN CLASSIFICATION ({args.stages}-STAGE)")
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
    from llm_parallelization.new_processor import NewProcessor

    with NewProcessor(
        gpu_list=args.gpu,
        llm=args.model,
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.9,
    ) as processor:
        if args.stages == 1:
            results = orchestrator.classify_comments_stage1_only(
                comments=comments,
                processor=processor,
                batch_size=args.batch_size,
                verbose=True,
            )
        elif args.stages == 3:
            results = orchestrator.classify_comments_3stage(
                comments=comments,
                processor=processor,
                batch_size=args.batch_size,
                verbose=True,
            )
        else:  # stages == 2
            results = orchestrator.classify_comments(
                comments=comments,
                processor=processor,
                batch_size=args.batch_size,
                verbose=True,
            )

    # =========================================================================
    # PHASE 10: Save Results
    # =========================================================================

    print("\n" + "=" * 70)
    print("SAVE RESULTS")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame (different methods for different stages)
    if args.stages == 1:
        df = orchestrator.results_to_dataframe_stage1(results, comments)
    elif args.stages == 3:
        df = orchestrator.results_to_dataframe_3stage(results)
    else:
        df = orchestrator.results_to_dataframe(results)

    # Save CSV
    csv_path = output_dir / f"classification_results_{args.stages}stage.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {csv_path}")

    # Save JSON
    json_path = output_dir / f"classification_results_{args.stages}stage.json"
    if args.stages == 1:
        json_results = [r.model_dump() if r else None for r in results]
    else:
        json_results = [r.model_dump() for r in results]
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved: {json_path}")

    # Stats (different for Stage 1 vs Stage 2/3)
    if args.stages == 1:
        # Stage 1 stats
        stats = {
            "total_comments": len(comments),
            "classifiable": sum(1 for r in results if r and r.has_classifiable_content),
            "non_classifiable": sum(1 for r in results if r and not r.has_classifiable_content),
            "parse_failures": sum(1 for r in results if r is None),
            "category_distribution": {},
        }
        for r in results:
            if r and r.categories_present:
                for cat in r.categories_present:
                    stats["category_distribution"][cat] = (
                        stats["category_distribution"].get(cat, 0) + 1
                    )
        stats_path = output_dir / "classification_stats_1stage.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved: {stats_path}")

        # Stage 1 summary
        print("\n" + "=" * 70)
        print("🔥 STAGE 1 CLASSIFICATION COMPLETE! 🔥")
        print("=" * 70)
        print(f"Total comments: {stats['total_comments']}")
        print(f"Classifiable: {stats['classifiable']}")
        print(f"Non-classifiable: {stats['non_classifiable']}")
        print("\nCategory distribution:")
        for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
            pct = 100 * count / stats["classifiable"] if stats["classifiable"] > 0 else 0
            print(f"  {cat}: {count} ({pct:.1f}%)")

    else:
        # Stage 2/3 stats
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
        orchestrator.print_results_summary(results)

    if load_from_python:
        print(f"Prompts loaded from: {prompts_dir}")
    else:
        print(f"Artifacts: {artifacts_dir}")
        if args.export_prompts:
            print(f"Prompts: {args.export_prompts}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
