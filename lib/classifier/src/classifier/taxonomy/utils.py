"""
Taxonomy Utilities

Helper functions for taxonomy inspection, display, and string manipulation.
"""

from typing import Dict, List


def sanitize_model_name(name: str) -> str:
    """
    Convert a taxonomy node name to a valid Python class name.

    Args:
        name: The original name (e.g., "Attendee Engagement & Interaction")

    Returns:
        A valid Python identifier (e.g., "AttendeeEngagementInteraction")

    Example:
        >>> sanitize_model_name("Event Logistics & Infrastructure")
        'EventLogisticsInfrastructure'
    """
    return "".join(c if c.isalnum() else "" for c in name)


def print_taxonomy_hierarchy(taxonomy: dict, indent: int = 0) -> None:
    """
    Print the taxonomy hierarchy in a readable format.

    Args:
        taxonomy: The taxonomy dict (or any node within it)
        indent: Current indentation level (used recursively)

    Example output:
        [ROOT] World
          [CATEGORY] Attendee Engagement & Interaction
            [ELEMENT] Community
              [ATTRIBUTE] Comfort Level
              [ATTRIBUTE] Engagement
    """
    name = taxonomy.get("name", "Unknown")
    children = taxonomy.get("children", [])

    prefix = "  " * indent
    if indent == 0:
        print(f"{prefix}[ROOT] {name}")
    elif indent == 1:
        print(f"{prefix}[CATEGORY] {name}")
    elif indent == 2:
        print(f"{prefix}[ELEMENT] {name}")
    elif indent == 3:
        print(f"{prefix}[ATTRIBUTE] {name}")
    else:
        print(f"{prefix}[LEVEL-{indent}] {name}")

    for child in children:
        print_taxonomy_hierarchy(child, indent + 1)


def extract_valid_combinations(taxonomy: dict) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract all valid category → element → attribute combinations.

    Args:
        taxonomy: The taxonomy dict

    Returns:
        Nested dict: {category: {element: [attributes]}}

    Example:
        >>> combos = extract_valid_combinations(taxonomy)
        >>> combos["People"]["Speakers/Presenters"]
        ['Accessibility', 'Availability', 'Clarity', 'Demeanor', 'Expertise', ...]
    """
    combinations = {}

    for category in taxonomy.get("children", []):
        cat_name = category["name"]
        combinations[cat_name] = {}

        for element in category.get("children", []):
            elem_name = element["name"]
            attributes = [attr["name"] for attr in element.get("children", [])]
            combinations[cat_name][elem_name] = attributes

    return combinations


def print_valid_combinations(taxonomy: dict) -> None:
    """
    Print valid category/element/attribute combinations in a compact format.

    Args:
        taxonomy: The taxonomy dict

    Example output:
        Attendee Engagement & Interaction
        ---------------------------------
          Community: Comfort Level, Engagement, Support, Value
          Knowledge Exchange: Effectiveness, Engagement, Impact Level, ...
    """
    combos = extract_valid_combinations(taxonomy)

    for category, elements in combos.items():
        print(f"\n{category}")
        print("-" * len(category))
        for element, attributes in elements.items():
            if attributes:
                print(f"  {element}: {', '.join(attributes)}")
            else:
                print(f"  {element}: (no attributes)")


def get_taxonomy_stats(taxonomy: dict) -> Dict[str, int]:
    """
    Get statistics about the taxonomy structure.

    Args:
        taxonomy: The taxonomy dict

    Returns:
        Dict with counts: categories, elements, attributes, total_nodes
    """
    categories = taxonomy.get("children", [])
    num_categories = len(categories)

    num_elements = 0
    num_attributes = 0

    for category in categories:
        elements = category.get("children", [])
        num_elements += len(elements)

        for element in elements:
            attributes = element.get("children", [])
            num_attributes += len(attributes)

    return {
        "categories": num_categories,
        "elements": num_elements,
        "attributes": num_attributes,
        "total_nodes": num_categories + num_elements + num_attributes,
    }


def get_category_names(taxonomy: dict) -> List[str]:
    """
    Get list of all category names from taxonomy.

    Args:
        taxonomy: The taxonomy dict

    Returns:
        List of category names
    """
    return [cat["name"] for cat in taxonomy.get("children", [])]


def get_elements_for_category(taxonomy: dict, category_name: str) -> List[str]:
    """
    Get list of element names for a specific category.

    Args:
        taxonomy: The taxonomy dict
        category_name: Name of the category

    Returns:
        List of element names, or empty list if category not found
    """
    for category in taxonomy.get("children", []):
        if category["name"] == category_name:
            return [el["name"] for el in category.get("children", [])]
    return []


def get_attributes_for_element(
    taxonomy: dict, category_name: str, element_name: str
) -> List[str]:
    """
    Get list of attribute names for a specific element.

    Args:
        taxonomy: The taxonomy dict
        category_name: Name of the category
        element_name: Name of the element

    Returns:
        List of attribute names, or empty list if not found
    """
    for category in taxonomy.get("children", []):
        if category["name"] == category_name:
            for element in category.get("children", []):
                if element["name"] == element_name:
                    return [attr["name"] for attr in element.get("children", [])]
    return []
