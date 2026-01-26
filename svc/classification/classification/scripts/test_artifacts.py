#!/usr/bin/env python
"""
Test Artifacts System

Demonstrates the YAML-based artifact system:
1. Scaffold from schema
2. Migrate from existing JSON
3. Load artifacts
4. Validate
5. Sync with schema changes

Usage:
    # Scaffold fresh from schema
    python test_artifacts.py --scaffold --schema schema.json --output artifacts/

    # Migrate from existing JSON
    python test_artifacts.py --migrate \\
        --schema schema.json \\
        --condensed condensed.json \\
        --examples examples.json \\
        --rules rules.json \\
        --output artifacts/

    # Load and validate
    python test_artifacts.py --load --artifacts artifacts/

    # Sync with updated schema
    python test_artifacts.py --sync --artifacts artifacts/ --schema schema_v2.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test Artifacts System")

    # Actions
    parser.add_argument(
        "--scaffold", action="store_true", help="Scaffold new artifacts from schema"
    )
    parser.add_argument("--migrate", action="store_true", help="Migrate from existing JSON")
    parser.add_argument("--load", action="store_true", help="Load and display artifacts")
    parser.add_argument("--validate", action="store_true", help="Validate artifacts")
    parser.add_argument("--sync", action="store_true", help="Sync with updated schema")
    parser.add_argument(
        "--build", action="store_true", help="Build classifier objects from artifacts"
    )

    # Paths
    parser.add_argument("--schema", type=str, help="Path to schema.json")
    parser.add_argument("--condensed", type=str, help="Path to condensed_taxonomy.json")
    parser.add_argument("--examples", type=str, help="Path to examples.json")
    parser.add_argument("--rules", type=str, help="Path to rules.json")
    parser.add_argument(
        "--artifacts", type=str, default="artifacts/", help="Artifacts directory"
    )
    parser.add_argument("--output", type=str, help="Output directory (for scaffold/migrate)")

    # Options
    parser.add_argument("--dry-run", action="store_true", help="Don't make changes (for sync)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    # Import artifacts module
    from classifier.artifacts import (
        build_classifier_objects,
        build_from_folder,
        scaffold_artifacts,
        scaffold_from_existing,
        sync_artifacts,
        validate_artifacts,
    )

    # =========================================================================
    # Scaffold
    # =========================================================================
    if args.scaffold:
        if not args.schema:
            parser.error("--scaffold requires --schema")

        output_dir = args.output or args.artifacts

        print("\n" + "=" * 70)
        print("SCAFFOLD ARTIFACTS FROM SCHEMA")
        print("=" * 70)

        result = scaffold_artifacts(
            schema_path=args.schema,
            output_dir=output_dir,
            overwrite=args.overwrite,
            verbose=True,
        )

        print(f"\n✓ Created {result['files_created']} files in {result['output_dir']}")

    # =========================================================================
    # Migrate
    # =========================================================================
    if args.migrate:
        if not all([args.schema, args.condensed, args.examples, args.rules]):
            parser.error("--migrate requires --schema, --condensed, --examples, --rules")

        output_dir = args.output or args.artifacts

        print("\n" + "=" * 70)
        print("MIGRATE FROM JSON TO YAML")
        print("=" * 70)

        result = scaffold_from_existing(
            schema_path=args.schema,
            condensed_path=args.condensed,
            examples_path=args.examples,
            rules_path=args.rules,
            output_dir=output_dir,
            verbose=True,
        )

        print(f"\n✓ Migrated to {result['files_created']} YAML files")

    # =========================================================================
    # Load
    # =========================================================================
    if args.load:
        print("\n" + "=" * 70)
        print("LOAD ARTIFACTS")
        print("=" * 70)

        artifacts = build_from_folder(args.artifacts, verbose=True)

        print("\n" + "-" * 70)
        print("LOADED STRUCTURE")
        print("-" * 70)

        for cat_name, cat in artifacts.categories.items():
            print(f"\n📁 {cat_name}")
            print(f"   Description: {cat.description[:60]}...")
            print(f"   Rules: {len(cat.rules)}")
            print(f"   Examples: {len(cat.examples)}")

            for elem_name, elem in cat.elements.items():
                print(f"   📁 {elem_name}")
                print(f"      Attributes: {list(elem.attributes.keys())}")
                print(f"      Examples: {len(elem.examples)}")

    # =========================================================================
    # Validate
    # =========================================================================
    if args.validate:
        print("\n" + "=" * 70)
        print("VALIDATE ARTIFACTS")
        print("=" * 70)

        result = validate_artifacts(
            artifacts_dir=args.artifacts,
            schema_path=args.schema,
            verbose=True,
        )

        if result["is_valid"]:
            print("\n✓ Artifacts are valid!")
        else:
            print(f"\n✗ Validation failed with {len(result['errors'])} errors")

    # =========================================================================
    # Sync
    # =========================================================================
    if args.sync:
        if not args.schema:
            parser.error("--sync requires --schema")

        print("\n" + "=" * 70)
        print("SYNC WITH SCHEMA")
        print("=" * 70)

        result = sync_artifacts(
            artifacts_dir=args.artifacts,
            schema_path=args.schema,
            dry_run=args.dry_run,
            verbose=True,
        )

        print(f"\n✓ Added: {len(result['added'])}, Orphaned: {len(result['removed'])}")

    # =========================================================================
    # Build
    # =========================================================================
    if args.build:
        print("\n" + "=" * 70)
        print("BUILD CLASSIFIER OBJECTS")
        print("=" * 70)

        condensed, examples, rules = build_classifier_objects(args.artifacts, verbose=True)

        print("\n" + "-" * 70)
        print("BUILT OBJECTS")
        print("-" * 70)
        print(f"CondensedTaxonomy: {len(condensed.categories)} categories")
        print(f"ExampleSet: {len(examples.examples)} examples")
        print(f"ClassificationRules: {len(rules.stage2_category_rules)} category rules")

        # Show sample
        if condensed.categories:
            sample_cat = condensed.categories[0]
            print(f"\nSample category: {sample_cat.name}")
            print(f"  Description: {sample_cat.short_description[:60]}...")
            if sample_cat.elements:
                print(f"  Elements: {[e.name for e in sample_cat.elements[:3]]}...")


if __name__ == "__main__":
    main()
