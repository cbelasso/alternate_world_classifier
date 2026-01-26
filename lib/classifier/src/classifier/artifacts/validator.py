"""
Artifact Validator

Validates YAML artifact folder structure against the source schema.

Usage:
    from classifier.artifacts import validate_artifacts

    # Validate folder structure
    result = validate_artifacts(
        artifacts_dir="artifacts/",
        schema_path="schema.json",
        strict=True,
    )

    if result["is_valid"]:
        print("✓ All good!")
    else:
        for error in result["errors"]:
            print(f"✗ {error}")
"""

import json
from pathlib import Path
import re
from typing import List, Optional

import yaml


def sanitize_folder_name(name: str) -> str:
    """Convert display name to folder name."""
    result = name.lower()
    result = result.replace("&", "and")
    result = result.replace("/", "_")
    result = result.replace(" ", "_")
    result = result.replace("-", "_")
    result = re.sub(r"[^a-z0-9_]", "", result)
    result = re.sub(r"_+", "_", result)
    result = result.strip("_")
    return result


def load_yaml(filepath: Path) -> dict:
    """Load YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_artifacts(
    artifacts_dir: str | Path,
    schema_path: Optional[str | Path] = None,
    strict: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Validate YAML artifacts against a schema.

    Checks:
    - Folder structure matches schema categories/elements/attributes
    - Required files exist (_category.yaml, _element.yaml)
    - YAML files are valid
    - Names in YAML match expected names
    - Examples have required fields

    Args:
        artifacts_dir: Path to artifacts directory
        schema_path: Optional path to schema.json (uses _schema_ref if not provided)
        strict: If True, missing items are errors. If False, just warnings.
        verbose: Print progress

    Returns:
        Dict with:
            - is_valid: bool
            - errors: List[str]
            - warnings: List[str]
            - stats: dict
    """
    artifacts_dir = Path(artifacts_dir)
    errors: List[str] = []
    warnings: List[str] = []

    if not artifacts_dir.exists():
        return {
            "is_valid": False,
            "errors": [f"Artifacts directory not found: {artifacts_dir}"],
            "warnings": [],
            "stats": {},
        }

    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATING ARTIFACTS")
        print("=" * 70)
        print(f"Directory: {artifacts_dir}")

    # Load schema
    if schema_path:
        schema_path = Path(schema_path)
        if not schema_path.exists():
            return {
                "is_valid": False,
                "errors": [f"Schema file not found: {schema_path}"],
                "warnings": [],
                "stats": {},
            }
        with open(schema_path, "r") as f:
            schema = json.load(f)
    else:
        # Try to get from _schema_ref.yaml
        schema_ref_path = artifacts_dir / "_schema_ref.yaml"
        if not schema_ref_path.exists():
            errors.append("Missing _schema_ref.yaml and no schema_path provided")
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "stats": {},
            }

        schema_ref = load_yaml(schema_ref_path)
        schema_source = schema_ref.get("schema_source", "")

        if schema_source and Path(schema_source).exists():
            with open(schema_source, "r") as f:
                schema = json.load(f)
        else:
            warnings.append(f"Could not load schema from {schema_source}")
            schema = None

    # Stats tracking
    stats = {
        "categories_expected": 0,
        "categories_found": 0,
        "elements_expected": 0,
        "elements_found": 0,
        "attributes_expected": 0,
        "attributes_found": 0,
        "examples_found": 0,
        "rules_found": 0,
    }

    # If we have schema, validate against it
    if schema:
        categories = schema.get("children", [])
        stats["categories_expected"] = len(categories)

        for category in categories:
            cat_name = category.get("name", "")
            cat_folder = sanitize_folder_name(cat_name)
            cat_dir = artifacts_dir / cat_folder

            # Check category directory exists
            if not cat_dir.exists():
                msg = f"Missing category directory: {cat_folder}/ ({cat_name})"
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)
                continue

            stats["categories_found"] += 1

            if verbose:
                print(f"\n📁 {cat_name}/")

            # Check _category.yaml
            cat_yaml_path = cat_dir / "_category.yaml"
            if not cat_yaml_path.exists():
                errors.append(f"Missing _category.yaml in {cat_folder}/")
            else:
                try:
                    cat_data = load_yaml(cat_yaml_path)
                    yaml_name = cat_data.get("name", "")
                    if yaml_name != cat_name:
                        warnings.append(
                            f"Name mismatch in {cat_folder}/_category.yaml: "
                            f"expected '{cat_name}', got '{yaml_name}'"
                        )

                    # Count examples (non-placeholder)
                    examples = [
                        e
                        for e in cat_data.get("examples", [])
                        if isinstance(e, dict) and not e.get("comment", "").startswith("#")
                    ]
                    stats["examples_found"] += len(examples)

                    # Count rules (non-placeholder)
                    rules = [
                        r
                        for r in cat_data.get("rules", [])
                        if isinstance(r, str) and not r.startswith("#")
                    ]
                    stats["rules_found"] += len(rules)

                    if verbose:
                        print(
                            f"   ✓ _category.yaml ({len(examples)} examples, {len(rules)} rules)"
                        )

                except Exception as e:
                    errors.append(f"Invalid YAML in {cat_folder}/_category.yaml: {e}")

            # Check elements
            elements = category.get("children", [])
            stats["elements_expected"] += len(elements)

            for element in elements:
                elem_name = element.get("name", "")
                elem_folder = sanitize_folder_name(elem_name)
                elem_dir = cat_dir / elem_folder

                if not elem_dir.exists():
                    msg = (
                        f"Missing element directory: {cat_folder}/{elem_folder}/ ({elem_name})"
                    )
                    if strict:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
                    continue

                stats["elements_found"] += 1

                if verbose:
                    print(f"   📁 {elem_name}/")

                # Check _element.yaml
                elem_yaml_path = elem_dir / "_element.yaml"
                if not elem_yaml_path.exists():
                    errors.append(f"Missing _element.yaml in {cat_folder}/{elem_folder}/")
                else:
                    try:
                        elem_data = load_yaml(elem_yaml_path)
                        yaml_name = elem_data.get("name", "")
                        if yaml_name != elem_name:
                            warnings.append(
                                f"Name mismatch in {elem_folder}/_element.yaml: "
                                f"expected '{elem_name}', got '{yaml_name}'"
                            )

                        # Count examples
                        examples = [
                            e
                            for e in elem_data.get("examples", [])
                            if isinstance(e, dict)
                            and not e.get("comment", "").startswith("#")
                            and not e.get("excerpt", "").startswith("#")
                        ]
                        stats["examples_found"] += len(examples)

                        rules = [
                            r
                            for r in elem_data.get("rules", [])
                            if isinstance(r, str) and not r.startswith("#")
                        ]
                        stats["rules_found"] += len(rules)

                        if verbose:
                            print(f"      ✓ _element.yaml ({len(examples)} examples)")

                    except Exception as e:
                        errors.append(
                            f"Invalid YAML in {cat_folder}/{elem_folder}/_element.yaml: {e}"
                        )

                # Check attributes
                attributes = element.get("children", [])
                stats["attributes_expected"] += len(attributes)

                for attribute in attributes:
                    attr_name = attribute.get("name", "")
                    attr_filename = sanitize_folder_name(attr_name) + ".yaml"
                    attr_path = elem_dir / attr_filename

                    if not attr_path.exists():
                        msg = f"Missing attribute file: {cat_folder}/{elem_folder}/{attr_filename} ({attr_name})"
                        if strict:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                        continue

                    stats["attributes_found"] += 1

                    try:
                        attr_data = load_yaml(attr_path)
                        yaml_name = attr_data.get("name", "")
                        if yaml_name != attr_name:
                            warnings.append(
                                f"Name mismatch in {attr_filename}: "
                                f"expected '{attr_name}', got '{yaml_name}'"
                            )

                        examples = [
                            e
                            for e in attr_data.get("examples", [])
                            if isinstance(e, dict) and not e.get("excerpt", "").startswith("#")
                        ]
                        stats["examples_found"] += len(examples)

                        rules = [
                            r
                            for r in attr_data.get("rules", [])
                            if isinstance(r, str) and not r.startswith("#")
                        ]
                        stats["rules_found"] += len(rules)

                        if verbose:
                            print(f"      ✓ {attr_filename} ({len(examples)} examples)")

                    except Exception as e:
                        errors.append(f"Invalid YAML in {attr_filename}: {e}")

    else:
        # No schema - just validate YAML syntax
        if verbose:
            print("No schema available - validating YAML syntax only")

        for yaml_file in artifacts_dir.rglob("*.yaml"):
            try:
                load_yaml(yaml_file)
                stats["categories_found"] += 1  # Rough count
            except Exception as e:
                errors.append(f"Invalid YAML in {yaml_file}: {e}")

    # Summary
    is_valid = len(errors) == 0

    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION RESULT")
        print("=" * 70)

        if is_valid:
            print("✓ VALID")
        else:
            print("✗ INVALID")

        print("\nStats:")
        print(f"  Categories: {stats['categories_found']}/{stats['categories_expected']}")
        print(f"  Elements: {stats['elements_found']}/{stats['elements_expected']}")
        print(f"  Attributes: {stats['attributes_found']}/{stats['attributes_expected']}")
        print(f"  Examples: {stats['examples_found']}")
        print(f"  Rules: {stats['rules_found']}")

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for e in errors[:10]:
                print(f"  ✗ {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for w in warnings[:10]:
                print(f"  ⚠ {w}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def check_examples_quality(
    artifacts_dir: str | Path,
    min_examples_per_element: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Check quality of examples in artifacts.

    Checks:
    - Minimum examples per element/attribute
    - Examples have non-empty excerpts
    - Sentiment values are valid

    Args:
        artifacts_dir: Path to artifacts directory
        min_examples_per_element: Minimum examples required
        verbose: Print progress

    Returns:
        Dict with quality metrics
    """
    artifacts_dir = Path(artifacts_dir)
    issues: List[str] = []
    stats = {
        "total_examples": 0,
        "elements_with_few_examples": 0,
        "empty_excerpts": 0,
        "invalid_sentiments": 0,
    }

    valid_sentiments = {"positive", "negative", "neutral", "mixed"}

    for yaml_file in artifacts_dir.rglob("*.yaml"):
        if yaml_file.name.startswith("_"):
            continue

        try:
            data = load_yaml(yaml_file)
            examples = data.get("examples", [])

            # Filter out placeholders
            real_examples = [
                e
                for e in examples
                if isinstance(e, dict)
                and not str(e.get("excerpt", e.get("comment", ""))).startswith("#")
            ]

            stats["total_examples"] += len(real_examples)

            if len(real_examples) < min_examples_per_element:
                stats["elements_with_few_examples"] += 1
                issues.append(
                    f"{yaml_file.relative_to(artifacts_dir)}: "
                    f"only {len(real_examples)} examples (min: {min_examples_per_element})"
                )

            for ex in real_examples:
                excerpt = ex.get("excerpt", ex.get("comment", ""))
                if not excerpt or not excerpt.strip():
                    stats["empty_excerpts"] += 1

                sentiment = ex.get("sentiment", "")
                if sentiment and sentiment not in valid_sentiments:
                    stats["invalid_sentiments"] += 1
                    issues.append(f"Invalid sentiment '{sentiment}' in {yaml_file.name}")

        except Exception as e:
            issues.append(f"Error reading {yaml_file}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("EXAMPLES QUALITY CHECK")
        print("=" * 70)
        print(f"Total examples: {stats['total_examples']}")
        print(f"Elements with few examples: {stats['elements_with_few_examples']}")
        print(f"Empty excerpts: {stats['empty_excerpts']}")
        print(f"Invalid sentiments: {stats['invalid_sentiments']}")

        if issues:
            print(f"\nIssues ({len(issues)}):")
            for issue in issues[:10]:
                print(f"  ⚠ {issue}")

    return {
        "stats": stats,
        "issues": issues,
        "quality_score": max(0, 1 - (len(issues) / max(stats["total_examples"], 1))),
    }
