"""
Stage 3 Prompt Builder

Assembles Stage 3 (attribute extraction) prompts from condensed taxonomy and examples.

Stage 3 prompts are ELEMENT-SPECIFIC - each element within each category gets its
own prompt function that extracts attributes relevant to that element.

This is pure Python templating - NO LLM required!

Usage:
    from classifier.prompts.stage3 import build_stage3_prompt_functions
    from classifier.taxonomy.condenser import load_condensed
    from classifier.taxonomy.example_generator import load_examples

    # Load artifacts
    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    # Build prompt functions (nested: category -> element -> prompt_fn)
    stage3_prompts = build_stage3_prompt_functions(condensed, examples)

    # Use at runtime
    community_prompt = stage3_prompts["Attendee Engagement & Interaction"]["Community"]
    prompt = community_prompt("The community was very welcoming!")
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..taxonomy.condenser import CondensedCategory, CondensedElement, CondensedTaxonomy
from ..taxonomy.example_generator import ClassificationExample, ExampleSet
from ..taxonomy.rule_generator import ClassificationRules

# =============================================================================
# Default Classification Rules
# =============================================================================

DEFAULT_STAGE3_BASE_RULES = [
    "Extract the EXACT excerpt from the comment that relates to each attribute.",
    "Each excerpt should be classified to ONE attribute only.",
    "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
    "If multiple distinct excerpts relate to the same attribute, create separate entries.",
    "Only extract attributes that are clearly present - do not infer or assume.",
    "Focus on the specific aspect (attribute) being discussed, not just the general element.",
]


# =============================================================================
# Prompt Template
# =============================================================================

STAGE3_TEMPLATE = """You are an expert conference feedback analyzer. Extract specific feedback related to attributes of {element_name} (within {category_name}) from this comment.

COMMENT TO ANALYZE:
{comment}

---

CONTEXT:
This comment has been identified as discussing "{element_name}" within the "{category_name}" category.
Your task is to identify which specific ATTRIBUTES of {element_name} are being discussed.

---

ATTRIBUTES TO IDENTIFY:

{attributes_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to specific attributes of {element_name}, return {{"classifications": []}}."""


# =============================================================================
# Formatting Helpers
# =============================================================================


def format_attributes_section(
    element: CondensedElement,
    taxonomy: dict,
    category_name: str,
) -> str:
    """
    Format attributes for a specific element.

    Args:
        element: CondensedElement instance
        taxonomy: Raw taxonomy dict to get attribute details
        category_name: Name of the parent category

    Returns:
        Formatted string with attribute names and descriptions
    """
    # Try to get attributes from taxonomy
    attributes = get_attributes_for_element(taxonomy, category_name, element.name)

    if not attributes:
        # Fallback: use element description
        return f"**{element.name}**: {element.short_description}\n(No specific attributes defined - classify general aspects)"

    lines = []
    for attr in attributes:
        name = attr.get("name", "Unknown")
        description = attr.get("description", attr.get("definition", ""))
        if description:
            lines.append(f"**{name}**: {description[:200]}")
        else:
            lines.append(f"**{name}**")

    return "\n".join(lines)


def get_attributes_for_element(
    taxonomy: dict,
    category_name: str,
    element_name: str,
) -> List[dict]:
    """
    Get attributes for a specific element from the raw taxonomy.

    Args:
        taxonomy: Raw taxonomy dict
        category_name: Category name
        element_name: Element name

    Returns:
        List of attribute dicts with 'name', 'description', etc.
    """
    if not taxonomy:
        return []

    # Navigate: root -> category -> element -> children (attributes)
    for category in taxonomy.get("children", []):
        if category.get("name") == category_name:
            for element in category.get("children", []):
                if element.get("name") == element_name:
                    return element.get("children", [])

    return []


def format_stage3_rules(
    category_name: str,
    element_name: str,
    rules: Optional[ClassificationRules] = None,
    base_rules: Optional[List[str]] = None,
) -> str:
    """
    Format rules for Stage 3.

    Args:
        category_name: Name of the category
        element_name: Name of the element
        rules: Optional ClassificationRules with element-specific rules
        base_rules: Base rules (uses DEFAULT_STAGE3_BASE_RULES if not provided)

    Returns:
        Formatted numbered rules
    """
    all_rules = list(base_rules or DEFAULT_STAGE3_BASE_RULES)

    # Add element-specific rules if available
    if rules:
        element_rules = rules.get_stage3_element_rules(category_name, element_name)
        all_rules.extend(element_rules)

    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(all_rules, 1))


def get_stage3_examples_for_element(
    examples: ExampleSet | List[ClassificationExample],
    category_name: str,
    element_name: str,
) -> List[ClassificationExample]:
    """
    Filter examples to only those relevant to a specific element.

    An example is relevant if it has element_details for this category+element.

    Args:
        examples: ExampleSet or list of ClassificationExample
        category_name: Category to filter for
        element_name: Element to filter for

    Returns:
        List of relevant examples
    """
    if isinstance(examples, ExampleSet):
        example_list = examples.examples
    else:
        example_list = examples

    relevant = []
    for ex in example_list:
        # Check if any element_details match this category+element
        has_relevant_details = any(
            detail.category == category_name and detail.element == element_name
            for detail in ex.element_details
        )

        if has_relevant_details:
            relevant.append(ex)

    return relevant


def format_stage3_example(
    example: ClassificationExample,
    category_name: str,
    element_name: str,
    taxonomy: Optional[dict] = None,
) -> Optional[str]:
    """
    Format a single example for Stage 3 prompt.

    For now, we'll format element-level examples and indicate they should
    be further classified into attributes.

    Args:
        example: ClassificationExample instance
        category_name: Category to filter for
        element_name: Element to filter for
        taxonomy: Optional taxonomy for attribute names

    Returns:
        Formatted example string, or None if not relevant
    """
    # Filter element_details to only this category+element
    relevant_details = [
        detail
        for detail in example.element_details
        if detail.category == category_name and detail.element == element_name
    ]

    if not relevant_details:
        return None

    # For Stage 3, we show the excerpt and indicate what attribute it maps to
    # Since we don't have attribute-level examples yet, we'll create a template
    classifications = []
    for detail in relevant_details:
        # Use the element detail as a placeholder - in production,
        # you'd have attribute-specific examples
        classifications.append(
            {
                "excerpt": detail.excerpt,
                "reasoning": f"Relates to specific aspect of {element_name}",
                "attribute": "[AttributeName]",  # Placeholder
                "sentiment": detail.sentiment,
            }
        )

    output = {"classifications": classifications}
    output_str = json.dumps(output, indent=2)

    return f'Comment: "{example.comment}"\n{output_str}'


def format_stage3_examples_section(
    examples: List[ClassificationExample],
    category_name: str,
    element_name: str,
    taxonomy: Optional[dict] = None,
) -> str:
    """
    Format all examples for a Stage 3 element prompt.

    Args:
        examples: List of ClassificationExample
        category_name: Category to filter for
        element_name: Element to filter for
        taxonomy: Optional taxonomy for attribute info

    Returns:
        Formatted examples section
    """
    formatted = []
    for ex in examples:
        formatted_ex = format_stage3_example(ex, category_name, element_name, taxonomy)
        if formatted_ex:
            formatted.append(formatted_ex)

    if not formatted:
        # Provide a generic example structure
        return f"""Comment: "Example feedback about {element_name}."
{{"classifications": [{{"excerpt": "Example feedback", "reasoning": "Relates to specific attribute", "attribute": "AttributeName", "sentiment": "positive"}}]}}"""

    return "\n\n".join(formatted[:3])  # Limit to 3 examples


# =============================================================================
# Main Builder Functions
# =============================================================================


def build_stage3_prompt_function(
    category: CondensedCategory,
    element: CondensedElement,
    examples: ExampleSet | List[ClassificationExample],
    taxonomy: Optional[dict] = None,
    rules: Optional[ClassificationRules] = None,
) -> Callable[[str], str]:
    """
    Build a reusable Stage 3 prompt function for a single element.

    This creates a closure that "bakes in" the element's attributes and examples,
    returning a simple function that takes a comment and returns a prompt.

    Args:
        category: CondensedCategory containing this element
        element: CondensedElement to build prompt for
        examples: ExampleSet or list of ClassificationExample
        taxonomy: Optional raw taxonomy for attribute details
        rules: Optional ClassificationRules

    Returns:
        Function that takes a comment string and returns a complete prompt
    """
    # Filter examples for this element
    element_examples = get_stage3_examples_for_element(examples, category.name, element.name)

    # Pre-compute static parts
    attributes_section = format_attributes_section(element, taxonomy, category.name)
    rules_section = format_stage3_rules(category.name, element.name, rules)
    examples_section = format_stage3_examples_section(
        element_examples, category.name, element.name, taxonomy
    )

    # Store in dict for closure
    static_parts = {
        "category_name": category.name,
        "element_name": element.name,
        "attributes_section": attributes_section,
        "rules_section": rules_section,
        "examples_section": examples_section,
    }

    def prompt_function(comment: str) -> str:
        """Generate Stage 3 attribute extraction prompt for a comment."""
        return f"""You are an expert conference feedback analyzer. Extract specific feedback related to attributes of {static_parts["element_name"]} (within {static_parts["category_name"]}) from this comment.

COMMENT TO ANALYZE:
{comment}

---

CONTEXT:
This comment has been identified as discussing "{static_parts["element_name"]}" within the "{static_parts["category_name"]}" category.
Your task is to identify which specific ATTRIBUTES of {static_parts["element_name"]} are being discussed.

---

ATTRIBUTES TO IDENTIFY:

{static_parts["attributes_section"]}

---

CLASSIFICATION RULES:

{static_parts["rules_section"]}

---

EXAMPLES:

{static_parts["examples_section"]}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to specific attributes of {static_parts["element_name"]}, return {{"classifications": []}}."""

    return prompt_function


def build_stage3_prompt_functions(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    taxonomy: Optional[dict] = None,
    rules: Optional[ClassificationRules] = None,
) -> Dict[str, Dict[str, Callable[[str], str]]]:
    """
    Build Stage 3 prompt functions for ALL elements in ALL categories.

    This is the main entry point for Stage 3 prompt generation.

    Args:
        condensed: CondensedTaxonomy with all categories and elements
        examples: ExampleSet or list of ClassificationExample
        taxonomy: Optional raw taxonomy for attribute details
        rules: Optional ClassificationRules

    Returns:
        Nested dict: category_name -> element_name -> prompt_function

    Example:
        >>> stage3_prompts = build_stage3_prompt_functions(condensed, examples, taxonomy)
        >>> community_prompt = stage3_prompts["Attendee Engagement & Interaction"]["Community"]
        >>> prompt = community_prompt("The community was welcoming!")
    """
    prompt_functions: Dict[str, Dict[str, Callable[[str], str]]] = {}

    for category in condensed.categories:
        prompt_functions[category.name] = {}

        for element in category.elements:
            prompt_functions[category.name][element.name] = build_stage3_prompt_function(
                category, element, examples, taxonomy, rules
            )

    return prompt_functions


# =============================================================================
# Statistics and Preview
# =============================================================================


def get_stage3_prompt_stats(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    taxonomy: Optional[dict] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Get statistics about Stage 3 prompts for each element.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        taxonomy: Optional raw taxonomy

    Returns:
        Nested dict: category -> element -> stats
    """
    stats = {}

    for category in condensed.categories:
        stats[category.name] = {}

        for element in category.elements:
            element_examples = get_stage3_examples_for_element(
                examples, category.name, element.name
            )

            # Get attribute count from taxonomy
            attributes = (
                get_attributes_for_element(taxonomy, category.name, element.name)
                if taxonomy
                else []
            )

            # Build a sample prompt for size estimation
            prompt_fn = build_stage3_prompt_function(category, element, examples, taxonomy)
            sample_prompt = prompt_fn("Sample comment for length estimation.")

            stats[category.name][element.name] = {
                "num_attributes": len(attributes),
                "num_examples": len(element_examples),
                "prompt_chars": len(sample_prompt),
                "estimated_tokens": len(sample_prompt) // 4,
            }

    return stats


def print_stage3_prompts_preview(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    taxonomy: Optional[dict] = None,
    show_full_prompt: Optional[Tuple[str, str]] = None,
) -> None:
    """
    Print a preview of Stage 3 prompts.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        taxonomy: Optional raw taxonomy
        show_full_prompt: Optional (category, element) tuple to show full prompt
    """
    stats = get_stage3_prompt_stats(condensed, examples, taxonomy)

    print("\n" + "=" * 70)
    print("STAGE 3 PROMPTS PREVIEW")
    print("=" * 70)

    total_elements = 0
    total_attributes = 0

    for category_name, elements in stats.items():
        print(f"\n**{category_name}**")

        for element_name, element_stats in elements.items():
            total_elements += 1
            total_attributes += element_stats["num_attributes"]

            print(f"  • {element_name}:")
            print(f"      Attributes: {element_stats['num_attributes']}")
            print(f"      Examples: {element_stats['num_examples']}")
            print(f"      Prompt: ~{element_stats['estimated_tokens']} tokens")

    print("\n--- TOTALS ---")
    print(f"Total elements: {total_elements}")
    print(f"Total attributes: {total_attributes}")
    print(f"Total Stage 3 prompts: {total_elements}")

    if show_full_prompt:
        cat_name, elem_name = show_full_prompt
        category = condensed.get_category(cat_name)
        if category:
            element = next((e for e in category.elements if e.name == elem_name), None)
            if element:
                prompt_fn = build_stage3_prompt_function(category, element, examples, taxonomy)
                prompt = prompt_fn("Sample feedback about this element.")

                print("\n" + "-" * 70)
                print(f"FULL PROMPT FOR: {cat_name} > {elem_name}")
                print("-" * 70)
                print(prompt)


# =============================================================================
# Export Functions
# =============================================================================


def get_all_category_element_pairs(
    condensed: CondensedTaxonomy,
) -> List[Tuple[str, str]]:
    """
    Get all (category, element) pairs from the taxonomy.

    Returns:
        List of (category_name, element_name) tuples
    """
    pairs = []
    for category in condensed.categories:
        for element in category.elements:
            pairs.append((category.name, element.name))
    return pairs
