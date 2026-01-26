"""
Artifact Builder

Loads YAML artifact files into Python objects for use in prompt generation.

Usage:
    from classifier.artifacts import build_from_folder

    # Load all artifacts
    artifacts = build_from_folder("artifacts/")

    # Access data
    community = artifacts.get_element("Attendee Engagement & Interaction", "Community")
    print(community.description)
    print(community.rules)
    print(community.examples)

    # Convert to CondensedTaxonomy + ExampleSet + Rules
    condensed, examples, rules = artifacts.to_classifier_objects()
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .schemas import (
    AttributeExample,
    AttributeFile,
    CategoryExample,
    CategoryFile,
    ElementExample,
    ElementFile,
    LoadedArtifacts,
    LoadedAttribute,
    LoadedCategory,
    LoadedElement,
    SchemaRefFile,
)

# =============================================================================
# YAML Loading Utilities
# =============================================================================


def load_yaml(filepath: Path) -> dict:
    """Load a YAML file, returning empty dict if file doesn't exist."""
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sanitize_folder_name(name: str) -> str:
    """Convert display name to folder name (same as scaffold.py)."""
    import re

    result = name.lower()
    result = result.replace("&", "and")
    result = result.replace("/", "_")
    result = result.replace(" ", "_")
    result = result.replace("-", "_")
    result = re.sub(r"[^a-z0-9_]", "", result)
    result = re.sub(r"_+", "_", result)
    result = result.strip("_")
    return result


# =============================================================================
# Main Builder
# =============================================================================


def build_from_folder(
    artifacts_dir: str | Path,
    verbose: bool = True,
) -> LoadedArtifacts:
    """
    Load all YAML artifacts from a folder structure.

    Args:
        artifacts_dir: Path to artifacts directory
        verbose: Print progress

    Returns:
        LoadedArtifacts object with all categories, elements, attributes

    Raises:
        FileNotFoundError: If artifacts_dir doesn't exist
        ValueError: If structure is invalid
    """
    artifacts_dir = Path(artifacts_dir)

    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    if verbose:
        print("\n" + "=" * 70)
        print("LOADING ARTIFACTS")
        print("=" * 70)
        print(f"Directory: {artifacts_dir}")

    # Load schema reference
    schema_ref_path = artifacts_dir / "_schema_ref.yaml"
    if not schema_ref_path.exists():
        raise ValueError(f"Missing _schema_ref.yaml in {artifacts_dir}")

    schema_ref_data = load_yaml(schema_ref_path)
    schema_ref = SchemaRefFile(**schema_ref_data)

    if verbose:
        print("✓ Loaded _schema_ref.yaml")
        print(f"  Structure: {schema_ref.structure}")

    # Find category directories
    category_dirs = [
        d for d in artifacts_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    ]

    if verbose:
        print(f"\nFound {len(category_dirs)} category directories")

    categories: Dict[str, LoadedCategory] = {}
    total_elements = 0
    total_attributes = 0
    total_examples = 0

    for cat_dir in sorted(category_dirs):
        # Load _category.yaml
        cat_file_path = cat_dir / "_category.yaml"
        if not cat_file_path.exists():
            if verbose:
                print(f"  ⚠ Skipping {cat_dir.name}/ (no _category.yaml)")
            continue

        cat_data = load_yaml(cat_file_path)
        cat_name = cat_data.get("name", cat_dir.name)

        if verbose:
            print(f"\n📁 {cat_name}/")

        # Parse examples (filter out placeholder comments)
        cat_examples = []
        for ex in cat_data.get("examples", []):
            if isinstance(ex, dict):
                comment = ex.get("comment", "")
                reasoning = ex.get("reasoning", "")
                # Skip placeholder examples
                if not comment.startswith("#"):
                    cat_examples.append(
                        CategoryExample(
                            comment=comment,
                            reasoning=reasoning,
                        )
                    )

        total_examples += len(cat_examples)

        # Parse rules (filter out placeholder comments)
        cat_rules = [
            r for r in cat_data.get("rules", []) if isinstance(r, str) and not r.startswith("#")
        ]

        # Find element directories
        element_dirs = [
            d for d in cat_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
        ]

        elements: Dict[str, LoadedElement] = {}

        for elem_dir in sorted(element_dirs):
            # Load _element.yaml
            elem_file_path = elem_dir / "_element.yaml"
            if not elem_file_path.exists():
                if verbose:
                    print(f"     ⚠ Skipping {elem_dir.name}/ (no _element.yaml)")
                continue

            elem_data = load_yaml(elem_file_path)
            elem_name = elem_data.get("name", elem_dir.name)

            if verbose:
                print(f"   📁 {elem_name}/")

            # Parse element examples
            elem_examples = []
            for ex in elem_data.get("examples", []):
                if isinstance(ex, dict):
                    comment = ex.get("comment", "")
                    excerpt = ex.get("excerpt", "")
                    if not comment.startswith("#") and not excerpt.startswith("#"):
                        elem_examples.append(
                            ElementExample(
                                comment=comment,
                                excerpt=excerpt,
                                sentiment=ex.get("sentiment", "positive"),
                                reasoning=ex.get("reasoning", ""),
                            )
                        )

            total_examples += len(elem_examples)

            # Parse element rules
            elem_rules = [
                r
                for r in elem_data.get("rules", [])
                if isinstance(r, str) and not r.startswith("#")
            ]

            # Find attribute files
            attr_files = [
                f
                for f in elem_dir.iterdir()
                if f.is_file() and f.suffix == ".yaml" and not f.name.startswith("_")
            ]

            attributes: Dict[str, LoadedAttribute] = {}

            for attr_file in sorted(attr_files):
                attr_data = load_yaml(attr_file)
                attr_name = attr_data.get("name", attr_file.stem)

                # Parse attribute examples
                attr_examples = []
                for ex in attr_data.get("examples", []):
                    if isinstance(ex, dict):
                        excerpt = ex.get("excerpt", "")
                        if not excerpt.startswith("#"):
                            attr_examples.append(
                                AttributeExample(
                                    excerpt=excerpt,
                                    sentiment=ex.get("sentiment", "positive"),
                                    reasoning=ex.get("reasoning", ""),
                                )
                            )

                total_examples += len(attr_examples)

                # Parse attribute rules
                attr_rules = [
                    r
                    for r in attr_data.get("rules", [])
                    if isinstance(r, str) and not r.startswith("#")
                ]

                attributes[attr_name] = LoadedAttribute(
                    name=attr_name,
                    category=cat_name,
                    element=elem_name,
                    description=attr_data.get("description", ""),
                    rules=attr_rules,
                    examples=attr_examples,
                )

                total_attributes += 1

                if verbose:
                    print(f"      ✓ {attr_file.name} ({len(attr_examples)} examples)")

            elements[elem_name] = LoadedElement(
                name=elem_name,
                category=cat_name,
                description=elem_data.get("description", ""),
                rules=elem_rules,
                examples=elem_examples,
                attributes=attributes,
            )

            total_elements += 1

        categories[cat_name] = LoadedCategory(
            name=cat_name,
            description=cat_data.get("description", ""),
            rules=cat_rules,
            examples=cat_examples,
            elements=elements,
        )

        if verbose:
            print(f"   ✓ {len(elements)} elements loaded")

    if verbose:
        print("\n" + "=" * 70)
        print(
            f"✓ Loaded {len(categories)} categories, {total_elements} elements, {total_attributes} attributes"
        )
        print(f"✓ Total examples: {total_examples}")
        print("=" * 70)

    return LoadedArtifacts(
        schema_ref=schema_ref,
        categories=categories,
    )


# =============================================================================
# Conversion to Classifier Objects
# =============================================================================


def artifacts_to_condensed(artifacts: LoadedArtifacts) -> dict:
    """
    Convert LoadedArtifacts to CondensedTaxonomy format.

    Returns a dict that can be passed to CondensedTaxonomy(**result).
    """
    categories = []

    for cat_name, cat in artifacts.categories.items():
        elements = []

        for elem_name, elem in cat.elements.items():
            attributes = []

            for attr_name, attr in elem.attributes.items():
                attributes.append(
                    {
                        "name": attr_name,
                        "short_description": attr.description,
                    }
                )

            elements.append(
                {
                    "name": elem_name,
                    "short_description": elem.description,
                    "attributes": attributes if attributes else None,
                }
            )

        categories.append(
            {
                "name": cat_name,
                "short_description": cat.description,
                "elements": elements,
            }
        )

    return {"categories": categories}


def artifacts_to_examples(artifacts: LoadedArtifacts) -> dict:
    """
    Convert LoadedArtifacts to ExampleSet format.

    Returns a dict that can be passed to ExampleSet(**result).
    """
    # Collect all unique comments with their full annotations
    comment_annotations: Dict[str, dict] = {}

    # From category examples
    for cat_name, cat in artifacts.categories.items():
        for ex in cat.examples:
            if ex.comment not in comment_annotations:
                comment_annotations[ex.comment] = {
                    "comment": ex.comment,
                    "has_classifiable_content": True,  # Has category = classifiable
                    "categories_present": [],
                    "stage1_reasoning": ex.reasoning,  # Use category reasoning
                    "element_details": [],
                }
            if cat_name not in comment_annotations[ex.comment]["categories_present"]:
                comment_annotations[ex.comment]["categories_present"].append(cat_name)
            # Update stage1_reasoning if we have better reasoning
            if ex.reasoning and not comment_annotations[ex.comment]["stage1_reasoning"]:
                comment_annotations[ex.comment]["stage1_reasoning"] = ex.reasoning

    # From element examples
    for cat_name, cat in artifacts.categories.items():
        for elem_name, elem in cat.elements.items():
            for ex in elem.examples:
                if ex.comment not in comment_annotations:
                    comment_annotations[ex.comment] = {
                        "comment": ex.comment,
                        "has_classifiable_content": True,  # Has element = classifiable
                        "categories_present": [],
                        "stage1_reasoning": f"Contains feedback about {elem_name}",
                        "element_details": [],
                    }
                if cat_name not in comment_annotations[ex.comment]["categories_present"]:
                    comment_annotations[ex.comment]["categories_present"].append(cat_name)

                comment_annotations[ex.comment]["element_details"].append(
                    {
                        "category": cat_name,
                        "element": elem_name,
                        "excerpt": ex.excerpt,
                        "sentiment": ex.sentiment,
                        "reasoning": ex.reasoning,
                    }
                )

    examples = list(comment_annotations.values())

    return {"examples": examples}


def artifacts_to_rules(artifacts: LoadedArtifacts) -> dict:
    """
    Convert LoadedArtifacts to ClassificationRules format.

    Returns a dict that can be passed to ClassificationRules(**result).
    """
    # Stage 2 category rules
    stage2_category_rules = []
    for cat_name, cat in artifacts.categories.items():
        if cat.rules:
            stage2_category_rules.append(
                {
                    "category": cat_name,
                    "rules": cat.rules,
                }
            )

    # Stage 3 element rules
    stage3_element_rules = []
    for cat_name, cat in artifacts.categories.items():
        for elem_name, elem in cat.elements.items():
            if elem.rules:
                stage3_element_rules.append(
                    {
                        "category": cat_name,
                        "element": elem_name,
                        "rules": elem.rules,
                    }
                )

    return {
        "stage1_base_rules": [],  # Usually defaults
        "stage2_base_rules": [],
        "stage2_category_rules": stage2_category_rules,
        "stage3_base_rules": [],
        "stage3_element_rules": stage3_element_rules,
    }


def build_classifier_objects(
    artifacts_dir: str | Path,
    verbose: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    Load artifacts and convert to classifier objects.

    This is the main entry point for using YAML artifacts with the
    existing classifier pipeline.

    Args:
        artifacts_dir: Path to artifacts directory
        verbose: Print progress

    Returns:
        Tuple of (CondensedTaxonomy, ExampleSet, ClassificationRules)

    Example:
        >>> condensed, examples, rules = build_classifier_objects("artifacts/")
        >>> stage1_prompt = build_stage1_prompt_function(condensed, examples)
    """
    from ..taxonomy.condenser import CondensedTaxonomy
    from ..taxonomy.example_generator import ExampleSet
    from ..taxonomy.rule_generator import ClassificationRules

    # Load artifacts
    artifacts = build_from_folder(artifacts_dir, verbose=verbose)

    # Convert
    condensed_dict = artifacts_to_condensed(artifacts)
    examples_dict = artifacts_to_examples(artifacts)
    rules_dict = artifacts_to_rules(artifacts)

    # Create objects
    condensed = CondensedTaxonomy(**condensed_dict)
    examples = ExampleSet(**examples_dict)
    rules = ClassificationRules(**rules_dict)

    if verbose:
        print("\n✓ Converted to classifier objects:")
        print(f"  - CondensedTaxonomy: {len(condensed.categories)} categories")
        print(f"  - ExampleSet: {len(examples.examples)} examples")
        print(
            f"  - ClassificationRules: {len(rules.stage2_category_rules)} category rules, {len(rules.stage3_element_rules)} element rules"
        )

    return condensed, examples, rules
