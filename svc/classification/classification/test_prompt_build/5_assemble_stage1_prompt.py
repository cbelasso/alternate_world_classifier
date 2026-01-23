"""
Assemble Stage 1 Prompt (Step 3)

Pure Python templating - takes condensed taxonomy and curated examples
and assembles the final Stage 1 classification prompt.

NO LLM needed - just string formatting!

Usage:
    python assemble_stage1_prompt.py
    python assemble_stage1_prompt.py --output stage1_prompt.py
"""

import argparse
from typing import Any, Dict, List

from utils.data_io import load_json

# =============================================================================
# Configuration
# =============================================================================

CONDENSED_TAXONOMY_PATH = "./condensed_taxonomy_stage1.json"
CURATED_EXAMPLES_PATH = "./examples_curated_stage1.json"
DEFAULT_OUTPUT = "./stage1_prompt_generated.py"

# Static classification rules (could be made dynamic later)
CLASSIFICATION_RULES = [
    "A comment can belong to MULTIPLE categories if it discusses multiple aspects.",
    "Focus on what the comment is ABOUT, not just words mentioned.",
    '"Community" refers to the feeling of belonging; "Networking" refers to the act of meeting people.',
    '"Presentations" = talk quality/content; "Speakers/Presenters" = the people themselves.',
    'General praise like "great conference" without specifics → Event Operations & Management > Conference.',
    "If a comment mentions both the content AND the presenter, include BOTH categories.",
]


# =============================================================================
# Formatting Functions
# =============================================================================


def format_categories(condensed_taxonomy: dict) -> str:
    """
    Format the categories and elements section.

    Output format:
        **Category Name**
        Category description.
        - Element: Element description
        - Element: Element description
    """
    lines = []

    for category in condensed_taxonomy.get("categories", []):
        cat_name = category["name"]
        cat_desc = category["short_description"]

        # Category header
        lines.append(f"**{cat_name}**")
        lines.append(cat_desc)

        # Elements
        for element in category.get("elements", []):
            elem_name = element["name"]
            elem_desc = element["short_description"]
            lines.append(f"- {elem_name}: {elem_desc}")

        lines.append("")  # Blank line between categories

    return "\n".join(lines).strip()


def format_rules(rules: List[str]) -> str:
    """
    Format the classification rules section.

    Output format:
        1. Rule one
        2. Rule two
    """
    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))


def format_examples(examples: List[Dict[str, Any]]) -> str:
    """
    Format the examples section.

    Output format:
        Comment: "The actual comment text"
        {{"categories_present": [...], "has_classifiable_content": true, "reasoning": "..."}}

    Note: Uses double braces {{}} because this will be embedded in an f-string template.
    """
    example_lines = []

    for example in examples:
        comment = example["comment"]
        categories = example["categories_present"]
        has_classifiable = example["has_classifiable_content"]
        reasoning = example["stage1_reasoning"]

        # Convert to JSON string representation
        import json

        categories_str = json.dumps(categories)
        has_classifiable_str = str(has_classifiable).lower()

        # Format with DOUBLE braces {{}} because this goes into an f-string template
        example_lines.append(f'Comment: "{comment}"')
        example_lines.append(
            f'{{{{"categories_present": {categories_str}, '
            f'"has_classifiable_content": {has_classifiable_str}, '
            f'"reasoning": "{reasoning}"}}}}'
        )
        example_lines.append("")  # Blank line between examples

    return "\n".join(example_lines).strip()


# =============================================================================
# Prompt Assembly
# =============================================================================


def assemble_stage1_prompt_template(
    condensed_taxonomy: dict,
    examples: List[Dict[str, Any]],
    rules: List[str] = CLASSIFICATION_RULES,
) -> str:
    """
    Assemble the complete Stage 1 prompt template.

    Returns a string that is a Python function definition.
    """
    categories_section = format_categories(condensed_taxonomy)
    rules_section = format_rules(rules)
    examples_section = format_examples(examples)

    # Build the prompt template as a Python function
    prompt_template = f'''"""
Stage 1: Category Detection Prompt

Auto-generated from condensed taxonomy and curated examples.
"""


def stage1_category_detection_prompt(comment: str) -> str:
    """
    Generate Stage 1 classification prompt for category detection.
    
    Args:
        comment: The conference feedback comment to analyze
        
    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in a conference attendee comment.

COMMENT TO ANALYZE:
{{comment}}

---

CATEGORIES AND THEIR SCOPE:

{categories_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Analyze the comment and return ONLY valid JSON."""
'''

    return prompt_template


def assemble_stage1_prompt_string(
    comment: str,
    condensed_taxonomy: dict,
    examples: List[Dict[str, Any]],
    rules: List[str] = CLASSIFICATION_RULES,
) -> str:
    """
    Assemble the complete Stage 1 prompt for a specific comment.

    This is useful for one-off testing without saving to a file.

    Args:
        comment: The conference feedback comment to analyze
        condensed_taxonomy: The condensed taxonomy dict
        examples: List of curated examples
        rules: Classification rules

    Returns:
        Complete prompt string ready for LLM
    """
    categories_section = format_categories(condensed_taxonomy)
    rules_section = format_rules(rules)
    examples_section = format_examples(examples)

    return f"""You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in a conference attendee comment.

COMMENT TO ANALYZE:
{comment}

---

CATEGORIES AND THEIR SCOPE:

{categories_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Analyze the comment and return ONLY valid JSON."""


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Assemble Stage 1 prompt from taxonomy and examples"
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=CONDENSED_TAXONOMY_PATH,
        help="Path to condensed taxonomy JSON",
    )
    parser.add_argument(
        "--examples",
        type=str,
        default=CURATED_EXAMPLES_PATH,
        help="Path to curated examples JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to save generated prompt module",
    )
    parser.add_argument(
        "--test-comment",
        type=str,
        default=None,
        help="Optional: Test with a specific comment and display the prompt",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading condensed taxonomy from: {args.taxonomy}")
    condensed_taxonomy = load_json(args.taxonomy)

    print(f"Loading curated examples from: {args.examples}")
    examples_data = load_json(args.examples)
    examples = examples_data.get("examples", [])

    print(f"\nFound {len(condensed_taxonomy.get('categories', []))} categories")
    print(f"Found {len(examples)} examples")

    # Generate prompt template
    print("\n" + "=" * 70)
    print("GENERATING STAGE 1 PROMPT TEMPLATE")
    print("=" * 70)

    prompt_module = assemble_stage1_prompt_template(
        condensed_taxonomy,
        examples,
        CLASSIFICATION_RULES,
    )

    # Save to file
    print(f"\nSaving prompt template to: {args.output}")
    with open(args.output, "w") as f:
        f.write(prompt_module)

    print("✓ Saved successfully")

    # Show stats
    print("\n" + "=" * 70)
    print("PROMPT STATISTICS:")
    print("=" * 70)
    print(f"Total prompt length: {len(prompt_module):,} characters")
    print(f"Categories: {len(condensed_taxonomy.get('categories', []))}")
    print(
        f"Total elements: {sum(len(cat.get('elements', [])) for cat in condensed_taxonomy.get('categories', []))}"
    )
    print(f"Examples: {len(examples)}")
    print(f"Rules: {len(CLASSIFICATION_RULES)}")

    # Test with comment if provided
    if args.test_comment:
        print("\n" + "=" * 70)
        print("TEST PROMPT:")
        print("=" * 70)

        test_prompt = assemble_stage1_prompt_string(
            args.test_comment,
            condensed_taxonomy,
            examples,
            CLASSIFICATION_RULES,
        )

        print(test_prompt)
        print("\n" + "=" * 70)
        print(f"Prompt length: {len(test_prompt):,} characters")
        print(f"Estimated tokens: ~{len(test_prompt) // 4}")

    # Preview first 1000 chars
    print("\n" + "=" * 70)
    print("PREVIEW (first 1000 chars):")
    print("=" * 70)
    print(prompt_module[:1000])
    print("\n... (truncated)")

    print("\n" + "=" * 70)
    print("ASSEMBLY COMPLETE!")
    print("=" * 70)
    print("\nTo use the generated prompt:")
    print("  from stage1_prompt_generated import stage1_category_detection_prompt")
    print("  prompt = stage1_category_detection_prompt('Your comment here')")


if __name__ == "__main__":
    main()
