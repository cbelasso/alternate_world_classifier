"""
Artifacts Utilities

Shared utilities for the YAML-based artifacts system.
"""

from pathlib import Path
import re
from typing import List, Optional


def sanitize_folder_name(name: str) -> str:
    """
    Convert a display name to a valid folder/file name.

    Examples:
        "Attendee Engagement & Interaction" → "attendee_engagement_and_interaction"
        "Speakers/Presenters" → "speakers_presenters"
        "Q&A Sessions" → "qanda_sessions"

    Args:
        name: Display name (e.g., category or element name)

    Returns:
        Sanitized name suitable for folder/file names
    """
    # Lowercase
    result = name.lower()

    # Replace & with "and"
    result = result.replace("&", "and")
    result = result.replace("&", "and")

    # Replace / with underscore
    result = result.replace("/", "_")

    # Replace spaces and hyphens with underscores
    result = result.replace(" ", "_")
    result = result.replace("-", "_")

    # Remove any other special characters
    result = re.sub(r"[^a-z0-9_]", "", result)

    # Collapse multiple underscores
    result = re.sub(r"_+", "_", result)

    # Strip leading/trailing underscores
    result = result.strip("_")

    return result


def display_name_from_folder(folder_name: str) -> str:
    """
    Attempt to convert folder name back to display format.

    Note: This is lossy - we can't perfectly recover original names.
    Use _category.yaml or _element.yaml for canonical names.

    Args:
        folder_name: Sanitized folder name

    Returns:
        Human-readable version (title case)
    """
    # Replace underscores with spaces
    result = folder_name.replace("_", " ")

    # Title case
    result = result.title()

    # Common replacements
    result = result.replace(" And ", " & ")
    result = result.replace("Qanda", "Q&A")

    return result


def get_yaml_header(
    title: str,
    description: Optional[str] = None,
    editable: bool = True,
) -> str:
    """
    Generate a YAML file header with documentation.

    Args:
        title: Title for the file
        description: Optional description
        editable: Whether this file is meant to be edited

    Returns:
        YAML comment header string
    """
    lines = [
        "# " + "=" * 68,
        f"# {title}",
        "# " + "=" * 68,
    ]

    if description:
        lines.append("#")
        # Wrap description
        for line in description.split("\n"):
            lines.append(f"# {line}")

    lines.append("#")

    if editable:
        lines.append("# This file is meant to be edited by annotators.")
        lines.append("# Changes here will be reflected in generated prompts.")
    else:
        lines.append("# AUTO-GENERATED - Do not edit directly.")

    lines.append("# " + "=" * 68)
    lines.append("")

    return "\n".join(lines)


def load_yaml_file(filepath: Path) -> dict:
    """
    Load a YAML file safely.

    Args:
        filepath: Path to YAML file

    Returns:
        Parsed YAML content as dict

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    import yaml

    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml_file(
    data: dict,
    filepath: Path,
    header: Optional[str] = None,
) -> Path:
    """
    Save data to a YAML file.

    Args:
        data: Data to save
        filepath: Output path
        header: Optional header comment to prepend

    Returns:
        Path to saved file
    """
    import yaml

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    content = ""
    if header:
        content = header

    # Use default_flow_style=False for readable output
    content += yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def get_schema_structure(schema: dict) -> dict:
    """
    Extract the hierarchical structure from a schema.

    Args:
        schema: Raw taxonomy schema dict

    Returns:
        Dict with structure:
        {
            "Category Name": {
                "elements": {
                    "Element Name": {
                        "attributes": ["Attr1", "Attr2", ...]
                    }
                }
            }
        }
    """
    structure = {}

    for category in schema.get("children", []):
        cat_name = category.get("name", "")
        structure[cat_name] = {
            "definition": category.get("definition", ""),
            "elements": {},
        }

        for element in category.get("children", []):
            elem_name = element.get("name", "")
            structure[cat_name]["elements"][elem_name] = {
                "definition": element.get("definition", ""),
                "attributes": [],
            }

            for attribute in element.get("children", []):
                attr_name = attribute.get("name", "")
                structure[cat_name]["elements"][elem_name]["attributes"].append(
                    {
                        "name": attr_name,
                        "definition": attribute.get("definition", ""),
                    }
                )

    return structure


def count_taxonomy_items(schema: dict) -> dict:
    """
    Count categories, elements, and attributes in a schema.

    Args:
        schema: Raw taxonomy schema dict

    Returns:
        Dict with counts
    """
    categories = schema.get("children", [])
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
    }
