"""
Artifact Scaffolding

Creates the initial YAML folder structure from a schema.json file.

Usage:
    from classifier.artifacts import scaffold_artifacts

    # Create folder structure
    scaffold_artifacts(
        schema_path="schema.json",
        output_dir="artifacts/",
        include_placeholders=True,
    )

This generates:
    artifacts/
    ├── _schema_ref.yaml
    ├── attendee_engagement_and_interaction/
    │   ├── _category.yaml
    │   ├── community/
    │   │   ├── _element.yaml
    │   │   ├── comfort_level.yaml
    │   │   └── ...
    │   └── ...
    └── ...
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Optional

from .utils import (
    count_taxonomy_items,
    get_schema_structure,
    get_yaml_header,
    sanitize_folder_name,
    save_yaml_file,
)


def _create_schema_ref(
    schema_path: Path,
    output_dir: Path,
    schema: dict,
) -> Path:
    """Create the _schema_ref.yaml file."""
    counts = count_taxonomy_items(schema)

    data = {
        "schema_source": str(schema_path.absolute()),
        "created_at": datetime.now().isoformat(),
        "structure": {
            "categories": counts["categories"],
            "elements": counts["elements"],
            "attributes": counts["attributes"],
        },
    }

    header = get_yaml_header(
        title="Schema Reference",
        description="Reference to the source schema file.\nUsed for validation and sync operations.",
        editable=False,
    )

    return save_yaml_file(data, output_dir / "_schema_ref.yaml", header)


def _create_category_yaml(
    category: dict,
    category_dir: Path,
) -> Path:
    """Create the _category.yaml file for a category."""
    cat_name = category.get("name", "")
    cat_def = category.get("definition", "")

    # Truncate definition for short_description placeholder
    short_desc = cat_def[:200] + "..." if len(cat_def) > 200 else cat_def

    data = {
        "name": cat_name,
        "description": short_desc,
        "rules": [
            "# Add disambiguation rules for this category",
            "# Example: 'This category covers X, not Y'",
        ],
        "examples": [
            {
                "comment": "# Add example comments that belong to this category",
                "reasoning": "# Explain why this comment belongs here",
            }
        ],
    }

    header = get_yaml_header(
        title=f"Category: {cat_name}",
        description=f"Stage 1 configuration for '{cat_name}'.\n\nEdit this file to:\n- Refine the category description\n- Add disambiguation rules\n- Add example comments",
        editable=True,
    )

    return save_yaml_file(data, category_dir / "_category.yaml", header)


def _create_element_yaml(
    element: dict,
    category_name: str,
    element_dir: Path,
) -> Path:
    """Create the _element.yaml file for an element."""
    elem_name = element.get("name", "")
    elem_def = element.get("definition", "")

    # Truncate definition for short_description placeholder
    short_desc = elem_def[:200] + "..." if len(elem_def) > 200 else elem_def

    data = {
        "name": elem_name,
        "category": category_name,
        "description": short_desc,
        "rules": [
            "# Add disambiguation rules for this element",
            "# Example: 'Look for keywords: X, Y, Z'",
        ],
        "examples": [
            {
                "comment": "# Add example comment",
                "excerpt": "# The specific part of the comment",
                "sentiment": "positive",  # positive, negative, neutral, mixed
                "reasoning": "# Why this excerpt maps to this element",
            }
        ],
    }

    header = get_yaml_header(
        title=f"Element: {elem_name}",
        description=f"Stage 2 configuration for '{category_name} > {elem_name}'.\n\nEdit this file to:\n- Refine the element description\n- Add disambiguation rules\n- Add example excerpts",
        editable=True,
    )

    return save_yaml_file(data, element_dir / "_element.yaml", header)


def _create_attribute_yaml(
    attribute: dict,
    category_name: str,
    element_name: str,
    element_dir: Path,
) -> Path:
    """Create an attribute YAML file."""
    attr_name = attribute.get("name", "")
    attr_def = attribute.get("definition", "")

    # Truncate definition for short_description placeholder
    short_desc = attr_def[:200] + "..." if len(attr_def) > 200 else attr_def

    # Sanitize for filename
    filename = sanitize_folder_name(attr_name) + ".yaml"

    data = {
        "name": attr_name,
        "category": category_name,
        "element": element_name,
        "description": short_desc,
        "rules": [
            "# Add disambiguation rules for this attribute",
            "# Example: 'Distinguish from [other attribute] by looking for...'",
        ],
        "examples": [
            {
                "excerpt": "# The specific text that maps to this attribute",
                "sentiment": "positive",  # positive, negative, neutral, mixed
                "reasoning": "# Why this excerpt maps to this attribute",
            }
        ],
    }

    header = get_yaml_header(
        title=f"Attribute: {attr_name}",
        description=f"Stage 3 configuration for '{category_name} > {element_name} > {attr_name}'.\n\nEdit this file to:\n- Refine the attribute description\n- Add disambiguation rules\n- Add example excerpts",
        editable=True,
    )

    return save_yaml_file(data, element_dir / filename, header)


def scaffold_artifacts(
    schema_path: str | Path,
    output_dir: str | Path,
    include_placeholders: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Create the YAML folder structure from a schema.json file.

    Args:
        schema_path: Path to the source schema.json
        output_dir: Output directory for artifacts
        include_placeholders: If True, include placeholder comments in YAML
        overwrite: If True, overwrite existing files
        verbose: Print progress

    Returns:
        Dict with creation statistics

    Raises:
        FileNotFoundError: If schema file doesn't exist
        FileExistsError: If output_dir exists and overwrite=False
    """
    schema_path = Path(schema_path)
    output_dir = Path(output_dir)

    # Load schema
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Check output directory
    if output_dir.exists() and not overwrite:
        # Check if it's non-empty
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Use overwrite=True to overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        counts = count_taxonomy_items(schema)
        print("\n" + "=" * 70)
        print("SCAFFOLDING ARTIFACTS")
        print("=" * 70)
        print(f"Schema: {schema_path}")
        print(f"Output: {output_dir}")
        print(f"Categories: {counts['categories']}")
        print(f"Elements: {counts['elements']}")
        print(f"Attributes: {counts['attributes']}")

    # Track created files
    created_files = []
    created_dirs = []

    # Create schema reference
    ref_file = _create_schema_ref(schema_path, output_dir, schema)
    created_files.append(ref_file)
    if verbose:
        print("\n✓ Created: _schema_ref.yaml")

    # Process categories
    for category in schema.get("children", []):
        cat_name = category.get("name", "")
        cat_folder = sanitize_folder_name(cat_name)
        cat_dir = output_dir / cat_folder

        cat_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.append(cat_dir)

        if verbose:
            print(f"\n📁 {cat_name}/")

        # Create category YAML
        cat_file = _create_category_yaml(category, cat_dir)
        created_files.append(cat_file)
        if verbose:
            print("   ✓ _category.yaml")

        # Process elements
        for element in category.get("children", []):
            elem_name = element.get("name", "")
            elem_folder = sanitize_folder_name(elem_name)
            elem_dir = cat_dir / elem_folder

            elem_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(elem_dir)

            if verbose:
                print(f"   📁 {elem_name}/")

            # Create element YAML
            elem_file = _create_element_yaml(element, cat_name, elem_dir)
            created_files.append(elem_file)
            if verbose:
                print("      ✓ _element.yaml")

            # Process attributes
            for attribute in element.get("children", []):
                attr_name = attribute.get("name", "")
                attr_file = _create_attribute_yaml(attribute, cat_name, elem_name, elem_dir)
                created_files.append(attr_file)
                if verbose:
                    attr_filename = sanitize_folder_name(attr_name) + ".yaml"
                    print(f"      ✓ {attr_filename}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"✓ Created {len(created_files)} YAML files in {len(created_dirs)} directories")
        print("=" * 70)

    return {
        "output_dir": str(output_dir),
        "files_created": len(created_files),
        "directories_created": len(created_dirs),
        "file_paths": [str(f) for f in created_files],
    }


def scaffold_from_existing(
    condensed_path: str | Path,
    examples_path: str | Path,
    rules_path: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
    verbose: bool = True,
) -> dict:
    """
    Create YAML folder structure from existing JSON artifacts.

    This migrates from the old JSON format to the new YAML folder structure.

    Args:
        condensed_path: Path to condensed_taxonomy.json
        examples_path: Path to examples.json
        rules_path: Path to rules.json
        schema_path: Path to original schema.json
        output_dir: Output directory for artifacts
        verbose: Print progress

    Returns:
        Dict with migration statistics
    """
    import json
    from pathlib import Path

    condensed_path = Path(condensed_path)
    examples_path = Path(examples_path)
    rules_path = Path(rules_path)
    schema_path = Path(schema_path)
    output_dir = Path(output_dir)

    # Load existing files
    with open(schema_path, "r") as f:
        schema = json.load(f)

    with open(condensed_path, "r") as f:
        condensed = json.load(f)

    with open(examples_path, "r") as f:
        examples_data = json.load(f)

    with open(rules_path, "r") as f:
        rules = json.load(f)

    # Build lookup dicts
    condensed_lookup = {}
    for cat in condensed.get("categories", []):
        condensed_lookup[cat["name"]] = {
            "description": cat.get("short_description", ""),
            "elements": {
                el["name"]: {
                    "description": el.get("short_description", ""),
                    "attributes": {
                        attr["name"]: attr.get("short_description", "")
                        for attr in el.get("attributes", [])
                    }
                    if el.get("attributes")
                    else {},
                }
                for el in cat.get("elements", [])
            },
        }

    # Build rules lookup
    rules_lookup = {
        "stage1": rules.get("stage1_base_rules", []),
        "stage2_base": rules.get("stage2_base_rules", []),
        "stage2_categories": {
            cr["category"]: cr.get("rules", []) for cr in rules.get("stage2_category_rules", [])
        },
        "stage3_base": rules.get("stage3_base_rules", []),
        "stage3_elements": {
            (er["category"], er["element"]): er.get("rules", [])
            for er in rules.get("stage3_element_rules", [])
        },
    }

    # Build examples lookup by category and element
    examples_by_category = {}
    examples_by_element = {}

    for ex in examples_data.get("examples", []):
        comment = ex.get("comment", "")
        categories = ex.get("categories_present", [])

        for cat in categories:
            if cat not in examples_by_category:
                examples_by_category[cat] = []
            examples_by_category[cat].append(
                {
                    "comment": comment,
                    "reasoning": ex.get("category_reasoning", ""),
                }
            )

        for detail in ex.get("element_details", []):
            key = (detail.get("category"), detail.get("element"))
            if key not in examples_by_element:
                examples_by_element[key] = []
            examples_by_element[key].append(
                {
                    "comment": comment,
                    "excerpt": detail.get("excerpt", ""),
                    "sentiment": detail.get("sentiment", ""),
                    "reasoning": detail.get("reasoning", ""),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("MIGRATING TO YAML FOLDER STRUCTURE")
        print("=" * 70)

    created_files = []

    # Create schema reference
    counts = count_taxonomy_items(schema)
    ref_data = {
        "schema_source": str(schema_path.absolute()),
        "migrated_from": {
            "condensed": str(condensed_path),
            "examples": str(examples_path),
            "rules": str(rules_path),
        },
        "created_at": datetime.now().isoformat(),
        "structure": counts,
    }
    header = get_yaml_header("Schema Reference", editable=False)
    save_yaml_file(ref_data, output_dir / "_schema_ref.yaml", header)
    created_files.append(output_dir / "_schema_ref.yaml")

    # Process each category
    for category in schema.get("children", []):
        cat_name = category.get("name", "")
        cat_folder = sanitize_folder_name(cat_name)
        cat_dir = output_dir / cat_folder
        cat_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n📁 {cat_name}/")

        # Get condensed description
        cat_desc = condensed_lookup.get(cat_name, {}).get("description", "")
        cat_rules = rules_lookup["stage2_categories"].get(cat_name, [])
        cat_examples = examples_by_category.get(cat_name, [])

        # Create category YAML
        cat_data = {
            "name": cat_name,
            "description": cat_desc or category.get("definition", "")[:200],
            "rules": cat_rules if cat_rules else ["# Add rules here"],
            "examples": cat_examples[:5]
            if cat_examples
            else [{"comment": "# Add example", "reasoning": "# Add reasoning"}],
        }
        header = get_yaml_header(f"Category: {cat_name}", editable=True)
        save_yaml_file(cat_data, cat_dir / "_category.yaml", header)
        created_files.append(cat_dir / "_category.yaml")

        if verbose:
            print(f"   ✓ _category.yaml ({len(cat_examples)} examples)")

        # Process elements
        for element in category.get("children", []):
            elem_name = element.get("name", "")
            elem_folder = sanitize_folder_name(elem_name)
            elem_dir = cat_dir / elem_folder
            elem_dir.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"   📁 {elem_name}/")

            # Get condensed description
            elem_desc = (
                condensed_lookup.get(cat_name, {})
                .get("elements", {})
                .get(elem_name, {})
                .get("description", "")
            )
            elem_rules = rules_lookup["stage3_elements"].get((cat_name, elem_name), [])
            elem_examples = examples_by_element.get((cat_name, elem_name), [])

            # Create element YAML
            elem_data = {
                "name": elem_name,
                "category": cat_name,
                "description": elem_desc or element.get("definition", "")[:200],
                "rules": elem_rules if elem_rules else ["# Add rules here"],
                "examples": elem_examples[:5]
                if elem_examples
                else [
                    {
                        "comment": "# Add example",
                        "excerpt": "# excerpt",
                        "sentiment": "positive",
                        "reasoning": "# reasoning",
                    }
                ],
            }
            header = get_yaml_header(f"Element: {elem_name}", editable=True)
            save_yaml_file(elem_data, elem_dir / "_element.yaml", header)
            created_files.append(elem_dir / "_element.yaml")

            if verbose:
                print(f"      ✓ _element.yaml ({len(elem_examples)} examples)")

            # Process attributes
            for attribute in element.get("children", []):
                attr_name = attribute.get("name", "")
                attr_filename = sanitize_folder_name(attr_name) + ".yaml"

                # Get condensed description
                attr_desc = (
                    condensed_lookup.get(cat_name, {})
                    .get("elements", {})
                    .get(elem_name, {})
                    .get("attributes", {})
                    .get(attr_name, "")
                )

                # Create attribute YAML
                attr_data = {
                    "name": attr_name,
                    "category": cat_name,
                    "element": elem_name,
                    "description": attr_desc or attribute.get("definition", "")[:200],
                    "rules": ["# Add attribute-specific rules"],
                    "examples": [
                        {
                            "excerpt": "# Add example excerpt",
                            "sentiment": "positive",
                            "reasoning": "# reasoning",
                        }
                    ],
                }
                header = get_yaml_header(f"Attribute: {attr_name}", editable=True)
                save_yaml_file(attr_data, elem_dir / attr_filename, header)
                created_files.append(elem_dir / attr_filename)

                if verbose:
                    print(f"      ✓ {attr_filename}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"✓ Migrated to {len(created_files)} YAML files")
        print("=" * 70)

    return {
        "output_dir": str(output_dir),
        "files_created": len(created_files),
    }
