"""
Combine and Curate Examples (Step 2c)

Combines simple and complex examples, provides review interface,
and creates the final curated example set for Stage 1 prompt.

Usage:
    python combine_examples.py
    python combine_examples.py --auto  # Skip manual review, use all
"""

import argparse
import json

from utils.data_io import load_json

# =============================================================================
# Configuration
# =============================================================================

SIMPLE_EXAMPLES_PATH = "./examples_simple_stage1.json"
COMPLEX_EXAMPLES_PATH = "./examples_complex_stage1.json"
DEFAULT_OUTPUT = "./examples_curated_stage1.json"


# =============================================================================
# Curation Functions
# =============================================================================


def load_all_examples(simple_path: str, complex_path: str):
    """Load examples from both sources."""
    simple_data = load_json(simple_path)
    complex_data = load_json(complex_path)

    simple_examples = simple_data.get("examples", [])
    multi_examples = complex_data.get("multi_category_examples", [])
    negative_examples = complex_data.get("negative_examples", [])

    return simple_examples, multi_examples, negative_examples


def display_example(example: dict, index: int, total: int):
    """Display a single example for review."""
    print(f"\n{'=' * 70}")
    print(f"Example {index + 1}/{total}")
    print(f"{'=' * 70}")
    print(f'Comment: "{example["comment"]}"')
    print(f"Stage 1 - Categories: {example['categories_present']}")
    print(f"Stage 1 - Classifiable: {example['has_classifiable_content']}")
    print(
        f"Stage 1 - Reasoning: {example.get('stage1_reasoning', example.get('reasoning', 'N/A'))}"
    )

    # Show element details if present
    element_details = example.get("element_details", [])
    if element_details:
        print(f"Stage 2 - Elements ({len(element_details)}):")
        for ed in element_details:
            cat = ed.get("category", "")
            cat_prefix = f"[{cat}] " if cat else ""
            print(f'  - {cat_prefix}{ed["element"]}: "{ed["excerpt"]}" [{ed["sentiment"]}]')
    else:
        print("Stage 2 - Elements: None")

    if "category" in example:
        print(f"Source category: {example['category']}")
    if "example_type" in example:
        print(f"Type: {example['example_type']}")


def auto_select_examples(simple_examples, multi_examples, negative_examples):
    """
    Automatically select a balanced set of examples.

    Strategy:
    - Pick 1 example per category from simple examples (5 total)
    - Use all multi-category examples (3-5)
    - Use all negative examples (2-3)
    Total: ~10-13 examples
    """
    selected = []

    # Select one example per category
    categories_seen = set()
    for ex in simple_examples:
        cat = ex.get("category")
        if cat and cat not in categories_seen:
            categories_seen.add(cat)
            selected.append(ex)

    # Add all multi-category examples
    selected.extend(multi_examples)

    # Add all negative examples
    selected.extend(negative_examples)

    return selected


def manual_select_examples(simple_examples, multi_examples, negative_examples):
    """
    Manually review and select examples.
    """
    print("\n" + "=" * 70)
    print("MANUAL CURATION MODE")
    print("=" * 70)
    print("Review each example and decide whether to include it.")
    print("Commands: y (yes), n (no), q (quit and use all so far)")

    selected = []

    # Review simple examples
    print("\n" + "=" * 70)
    print("SIMPLE CATEGORY EXAMPLES")
    print("=" * 70)

    for i, ex in enumerate(simple_examples):
        display_example(ex, i, len(simple_examples))

        while True:
            choice = input("\nInclude this example? (y/n/q): ").strip().lower()
            if choice in ["y", "n", "q"]:
                break
            print("Invalid choice. Use y/n/q")

        if choice == "q":
            print("Stopping review. Using selected examples so far.")
            break
        elif choice == "y":
            selected.append(ex)
            print("✓ Added")

    # Review multi-category examples
    print("\n" + "=" * 70)
    print("MULTI-CATEGORY EXAMPLES")
    print("=" * 70)

    for i, ex in enumerate(multi_examples):
        display_example(ex, i, len(multi_examples))

        while True:
            choice = input("\nInclude this example? (y/n/q): ").strip().lower()
            if choice in ["y", "n", "q"]:
                break
            print("Invalid choice. Use y/n/q")

        if choice == "q":
            print("Stopping review. Using selected examples so far.")
            break
        elif choice == "y":
            selected.append(ex)
            print("✓ Added")

    # Review negative examples
    print("\n" + "=" * 70)
    print("NEGATIVE EXAMPLES")
    print("=" * 70)

    for i, ex in enumerate(negative_examples):
        display_example(ex, i, len(negative_examples))

        while True:
            choice = input("\nInclude this example? (y/n/q): ").strip().lower()
            if choice in ["y", "n", "q"]:
                break
            print("Invalid choice. Use y/n/q")

        if choice == "q":
            print("Stopping review. Using selected examples so far.")
            break
        elif choice == "y":
            selected.append(ex)
            print("✓ Added")

    return selected


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Combine and curate examples")
    parser.add_argument(
        "--simple",
        type=str,
        default=SIMPLE_EXAMPLES_PATH,
        help="Path to simple examples JSON",
    )
    parser.add_argument(
        "--complex",
        type=str,
        default=COMPLEX_EXAMPLES_PATH,
        help="Path to complex examples JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to save curated examples JSON",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-select examples without manual review",
    )

    args = parser.parse_args()

    # Load examples
    print("Loading examples...")
    simple_examples, multi_examples, negative_examples = load_all_examples(
        args.simple, args.complex
    )

    print(f"  Simple examples: {len(simple_examples)}")
    print(f"  Multi-category examples: {len(multi_examples)}")
    print(f"  Negative examples: {len(negative_examples)}")
    print(
        f"  Total available: {len(simple_examples) + len(multi_examples) + len(negative_examples)}"
    )

    # Select examples
    if args.auto:
        print("\nAuto-selecting examples...")
        selected = auto_select_examples(simple_examples, multi_examples, negative_examples)
    else:
        selected = manual_select_examples(simple_examples, multi_examples, negative_examples)

    # Summary
    print("\n" + "=" * 70)
    print("SELECTION SUMMARY:")
    print("=" * 70)

    # Count by type
    simple_count = sum(1 for ex in selected if "category" in ex)
    multi_count = sum(1 for ex in selected if ex.get("example_type") == "multi_category")
    neg_count = sum(1 for ex in selected if ex.get("example_type") == "negative")

    print(f"Simple examples: {simple_count}")
    print(f"Multi-category examples: {multi_count}")
    print(f"Negative examples: {neg_count}")
    print(f"Total selected: {len(selected)}")

    # Check coverage
    categories_covered = set()
    for ex in selected:
        if "category" in ex:
            categories_covered.add(ex["category"])

    print(f"\nCategories represented: {len(categories_covered)}")
    if len(categories_covered) < 5:
        print("  ⚠ Not all categories represented")
        print(f"  Covered: {categories_covered}")

    # Preview selected examples
    print("\n" + "=" * 70)
    print("SELECTED EXAMPLES PREVIEW:")
    print("=" * 70)

    for i, ex in enumerate(selected[:5], 1):  # Show first 5
        print(f'\n{i}. "{ex["comment"]}"')
        print(f"   → {ex['categories_present']}")

    if len(selected) > 5:
        print(f"\n... and {len(selected) - 5} more")

    # Save to file
    print(f"\nSaving curated examples to: {args.output}")

    output_data = {
        "description": "Curated examples for Stage 1 classification prompt",
        "total_examples": len(selected),
        "breakdown": {
            "simple": simple_count,
            "multi_category": multi_count,
            "negative": neg_count,
        },
        "examples": [
            {
                "comment": ex["comment"],
                "categories_present": ex["categories_present"],
                "has_classifiable_content": ex["has_classifiable_content"],
                "stage1_reasoning": ex.get("stage1_reasoning", ex.get("reasoning", "")),
                "element_details": ex.get("element_details", []),
            }
            for ex in selected
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved {len(selected)} curated examples")


if __name__ == "__main__":
    main()
