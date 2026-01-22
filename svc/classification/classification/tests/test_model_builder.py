"""
Test Model Builder

Validates dynamic Pydantic model generation from a taxonomy JSON file.
Tests both 2-level (element + sentiment) and 3-level (element + attribute + sentiment) modes.

Usage:
    python -m classification.tests.test_model_builder --taxonomy /path/to/taxonomy.json

    # Or with defaults (uses TAXONOMY_PATH constant)
    python -m classification.tests.test_model_builder

    # Show all valid combinations
    python -m classification.tests.test_model_builder --show-combinations
"""

import argparse
import sys
from typing import Any, Dict

from classifier import build_models_from_taxonomy
from classifier.taxonomy import (
    extract_valid_combinations,
    get_taxonomy_stats,
    print_valid_combinations,
)
from utils.data_io import load_json

# =============================================================================
# Configuration
# =============================================================================

# Default taxonomy path - update this for your environment
TAXONOMY_PATH = "/data-fast/data3/clyde/projects/world/documents/schemas/schema_v1.json"


# =============================================================================
# Test Functions
# =============================================================================


def test_2level_models(taxonomy: dict) -> Dict[str, Any]:
    """Test 2-level model generation (element + sentiment)."""

    print("=" * 70)
    print("TEST: 2-Level Models (element + sentiment)")
    print("=" * 70)

    models = build_models_from_taxonomy(taxonomy, include_attributes=False)

    print("\n>>> Categories loaded:")
    for cat in models["category_to_elements"].keys():
        print(f"    - {cat}")

    print(
        f"\n>>> Total elements: {sum(len(v) for v in models['category_to_elements'].values())}"
    )

    print("\n>>> category_to_schema:")
    for cat, schema in models["category_to_schema"].items():
        print(f"    {cat}: {schema.__name__}")

    # Create sample output
    FinalOutput = models["final_output_model"]
    ClassSpan = models["classification_span_model"]

    sample = FinalOutput(
        original_comment="The keynote speaker was brilliant and the WiFi was terrible.",
        has_classifiable_content=True,
        category_reasoning="Discusses speaker (People) and WiFi (Event Logistics)",
        classifications=[
            ClassSpan(
                excerpt="The keynote speaker was brilliant",
                reasoning="Positive feedback about speaker",
                category="People",
                element="Speakers/Presenters",
                sentiment="positive",
            ),
            ClassSpan(
                excerpt="the WiFi was terrible",
                reasoning="Negative feedback about connectivity",
                category="Event Logistics & Infrastructure",
                element="Wi-Fi",
                sentiment="negative",
            ),
        ],
    )

    print("\n>>> Sample FinalOutput (2-level):")
    print(sample.model_dump_json(indent=2))

    return models


def test_3level_models(taxonomy: dict) -> Dict[str, Any]:
    """Test 3-level model generation (element + attribute + sentiment)."""

    print("\n")
    print("=" * 70)
    print("TEST: 3-Level Models (element + attribute + sentiment)")
    print("=" * 70)

    models = build_models_from_taxonomy(taxonomy, include_attributes=True)

    unique_attrs = set(
        attr for attrs in models["element_to_attributes"].values() for attr in attrs
    )
    print(f"\n>>> Total unique attributes: {len(unique_attrs)}")

    # Create sample output with attributes
    FinalOutput = models["final_output_model"]
    ClassSpan = models["classification_span_model"]

    sample = FinalOutput(
        original_comment="The keynote speaker was brilliant and the WiFi was terrible.",
        has_classifiable_content=True,
        category_reasoning="Discusses speaker (People) and WiFi (Event Logistics)",
        classifications=[
            ClassSpan(
                excerpt="The keynote speaker was brilliant",
                reasoning="Positive feedback about speaker's expertise",
                category="People",
                element="Speakers/Presenters",
                attribute="Expertise",
                sentiment="positive",
            ),
            ClassSpan(
                excerpt="the WiFi was terrible",
                reasoning="Negative feedback about WiFi functionality",
                category="Event Logistics & Infrastructure",
                element="Wi-Fi",
                attribute="Functionality",
                sentiment="negative",
            ),
        ],
    )

    print("\n>>> Sample FinalOutput (3-level):")
    print(sample.model_dump_json(indent=2))

    return models


def test_attribute_validation(models: Dict[str, Any]) -> None:
    """Test that attribute validation works correctly."""

    print("\n")
    print("=" * 70)
    print("TEST: Attribute Validation")
    print("=" * 70)

    # Get the People schema for testing
    PeopleSchema = models["category_to_schema"]["People"]
    PeopleSpan = PeopleSchema.model_fields["classifications"].annotation.__args__[0]

    # Test valid combinations
    print("\n>>> Valid combinations:")

    valid_tests = [
        ("Speakers/Presenters", "Expertise"),
        ("Speakers/Presenters", "Clarity"),
        ("Conference Staff", "Professionalism"),
        ("Conference Staff", "Support"),
        ("Participants/Attendees", "Demeanor"),
    ]

    for element, attribute in valid_tests:
        try:
            span = PeopleSpan(
                excerpt="Test",
                reasoning="Test",
                element=element,
                attribute=attribute,
                sentiment="positive",
            )
            print(f"    ✓ {element} + {attribute}")
        except Exception as e:
            print(f"    ✗ {element} + {attribute}: UNEXPECTED ERROR - {e}")

    # Test invalid combinations
    print("\n>>> Invalid combinations (should fail):")

    invalid_tests = [
        (
            "Speakers/Presenters",
            "Professionalism",
        ),  # Professionalism is for Staff, not Speakers
        (
            "Participants/Attendees",
            "Expertise",
        ),  # Expertise is for Speakers/Experts, not Attendees
        ("Conference Staff", "Expertise"),  # Expertise is for Speakers/Experts, not Staff
    ]

    for element, attribute in invalid_tests:
        try:
            span = PeopleSpan(
                excerpt="Test",
                reasoning="Test",
                element=element,
                attribute=attribute,
                sentiment="positive",
            )
            print(f"    ✗ {element} + {attribute}: SHOULD HAVE FAILED")
        except ValueError as e:
            print(f"    ✓ {element} + {attribute}: Correctly rejected")


def test_element_attribute_mappings(models: Dict[str, Any]) -> None:
    """Display element-to-attribute mappings for inspection."""

    print("\n")
    print("=" * 70)
    print("TEST: Element-Attribute Mappings")
    print("=" * 70)

    for category in models["category_to_elements"].keys():
        print(f"\n{category}")
        print("-" * len(category))

        for element in models["category_to_elements"][category]:
            attrs = models["element_to_attributes"].get(element, [])
            if attrs:
                print(f"  {element}:")
                print(f"      {', '.join(attrs)}")
            else:
                print(f"  {element}: (no attributes)")


def test_taxonomy_stats(taxonomy: dict) -> None:
    """Display taxonomy statistics."""

    print("\n")
    print("=" * 70)
    print("TEST: Taxonomy Statistics")
    print("=" * 70)

    stats = get_taxonomy_stats(taxonomy)

    print("\n>>> Taxonomy structure:")
    print(f"    Categories: {stats['categories']}")
    print(f"    Elements: {stats['elements']}")
    print(f"    Attributes: {stats['attributes']}")
    print(f"    Total nodes: {stats['total_nodes']}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test dynamic Pydantic model generation from taxonomy"
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=TAXONOMY_PATH,
        help=f"Path to taxonomy JSON file (default: {TAXONOMY_PATH})",
    )
    parser.add_argument(
        "--show-combinations",
        action="store_true",
        help="Print all valid category/element/attribute combinations",
    )

    args = parser.parse_args()

    # Load taxonomy
    print(f"Loading taxonomy from: {args.taxonomy}")
    try:
        taxonomy = load_json(args.taxonomy)
    except FileNotFoundError:
        print(f"ERROR: Taxonomy file not found: {args.taxonomy}")
        sys.exit(1)

    # Show combinations if requested
    if args.show_combinations:
        print("\n")
        print("=" * 70)
        print("TAXONOMY: Valid Combinations")
        print("=" * 70)
        print_valid_combinations(taxonomy)
        print("\n")

    # Run tests
    test_taxonomy_stats(taxonomy)
    models_2level = test_2level_models(taxonomy)
    models_3level = test_3level_models(taxonomy)
    test_attribute_validation(models_3level)
    test_element_attribute_mappings(models_3level)

    print("\n")
    print("=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
