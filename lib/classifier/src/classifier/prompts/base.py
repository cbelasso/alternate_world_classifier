"""
Prompt Building Utilities

Shared formatting functions used across prompt builders.
"""

import json
from typing import Any, Dict, List

from ..taxonomy.condenser import CondensedTaxonomy
from ..taxonomy.example_generator import ClassificationExample, ExampleSet


def format_categories_section(condensed: CondensedTaxonomy) -> str:
    """
    Format categories and elements for use in prompts.

    Output format:
        **Category Name**
        Category description.
        - Element: Element description
        - Element: Element description

    Args:
        condensed: CondensedTaxonomy instance

    Returns:
        Formatted string
    """
    lines = []

    for category in condensed.categories:
        # Category header
        lines.append(f"**{category.name}**")
        lines.append(category.short_description)

        # Elements
        for element in category.elements:
            lines.append(f"- {element.name}: {element.short_description}")

        lines.append("")  # Blank line between categories

    return "\n".join(lines).strip()


def format_rules_section(rules: List[str]) -> str:
    """
    Format classification rules as numbered list.

    Args:
        rules: List of rule strings

    Returns:
        Formatted numbered list
    """
    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))


def format_stage1_example(example: ClassificationExample) -> str:
    """
    Format a single example for Stage 1 prompt.

    Uses double braces for JSON since this may be embedded in f-string templates.

    Args:
        example: ClassificationExample instance

    Returns:
        Formatted example string
    """
    categories_str = json.dumps(example.categories_present)
    has_classifiable_str = str(example.has_classifiable_content).lower()

    return (
        f'Comment: "{example.comment}"\n'
        f'{{{{"categories_present": {categories_str}, '
        f'"has_classifiable_content": {has_classifiable_str}, '
        f'"reasoning": "{example.stage1_reasoning}"}}}}'
    )


def format_stage1_examples_section(examples: List[ClassificationExample]) -> str:
    """
    Format all examples for Stage 1 prompt.

    Args:
        examples: List of ClassificationExample

    Returns:
        Formatted examples section
    """
    formatted = [format_stage1_example(ex) for ex in examples]
    return "\n\n".join(formatted)


def format_stage2_example(example: ClassificationExample, category: str) -> str:
    """
    Format a single example for Stage 2 prompt (category-specific).

    Args:
        example: ClassificationExample instance
        category: The category this prompt is for

    Returns:
        Formatted example string, or None if no relevant elements
    """
    # Filter element details to only those matching this category
    relevant_details = [
        detail
        for detail in example.element_details
        if detail.category == category or example.source_category == category
    ]

    if not relevant_details:
        return None

    # Format the classifications array
    classifications = []
    for detail in relevant_details:
        classifications.append(
            {
                "excerpt": detail.excerpt,
                "reasoning": detail.reasoning,
                "element": detail.element,
                "sentiment": detail.sentiment,
            }
        )

    classifications_str = json.dumps({"classifications": classifications}, indent=2)

    return f'Comment: "{example.comment}"\n{classifications_str}'


def escape_for_fstring(text: str) -> str:
    """
    Escape text for embedding in an f-string.

    Doubles up braces so they don't get interpreted as format specifiers.

    Args:
        text: Raw text

    Returns:
        Text safe for f-string embedding
    """
    return text.replace("{", "{{").replace("}", "}}")


def get_valid_categories_list(condensed: CondensedTaxonomy) -> str:
    """
    Get a formatted list of valid category names.

    Args:
        condensed: CondensedTaxonomy instance

    Returns:
        Comma-separated list of category names
    """
    return ", ".join(f'"{cat.name}"' for cat in condensed.categories)


def get_valid_elements_for_category(condensed: CondensedTaxonomy, category: str) -> str:
    """
    Get a formatted list of valid element names for a category.

    Args:
        condensed: CondensedTaxonomy instance
        category: Category name

    Returns:
        Comma-separated list of element names
    """
    cat = condensed.get_category(category)
    if not cat:
        return ""
    return ", ".join(f'"{e.name}"' for e in cat.elements)
