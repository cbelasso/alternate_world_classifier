"""
Artifact Sync

Updates the YAML artifact folder when the schema changes.

Handles:
- New categories/elements/attributes in schema → creates placeholder files
- Removed items → warns (doesn't delete to preserve manual edits)
- Renamed items → warns (user must handle manually)

Usage:
    from classifier.artifacts import sync_artifacts

    # Update folder when schema changes
    result = sync_artifacts(
        artifacts_dir="artifacts/",
        schema_path="schema_v2.json",
    )

    print(f"Added: {result['added']}")
    print(f"Removed: {result['removed']}")
"""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import List, Set, Tuple

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


def get_yaml_header(title: str, editable: bool = True) -> str:
    """Generate a YAML file header."""
    lines = [
        "# " + "=" * 68,
        f"# {title}",
        "# " + "=" * 68,
        "#",
    ]
    if editable:
        lines.append("# This file is meant to be edited by annotators.")
    else:
        lines.append("# AUTO-GENERATED")
    lines.append("# " + "=" * 68)
    lines.append("")
    return "\n".join(lines)


def save_yaml(data: dict, filepath: Path, header: str = None) -> Path:
    """Save YAML file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    content = header or ""
    content += yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


def sync_artifacts(
    artifacts_dir: str | Path,
    schema_path: str | Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Sync artifact folder with updated schema.

    This function:
    1. Identifies new items in schema → creates placeholder YAML files
    2. Identifies removed items → warns but doesn't delete
    3. Updates _schema_ref.yaml with new schema path

    Args:
        artifacts_dir: Path to artifacts directory
        schema_path: Path to updated schema.json
        dry_run: If True, don't actually make changes, just report
        verbose: Print progress

    Returns:
        Dict with:
            - added: List of added file paths
            - removed: List of items in artifacts but not in schema (warnings)
            - unchanged: Count of unchanged items
    """
    artifacts_dir = Path(artifacts_dir)
    schema_path = Path(schema_path)

    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    if verbose:
        print("\n" + "=" * 70)
        print("SYNCING ARTIFACTS WITH SCHEMA")
        print("=" * 70)
        print(f"Artifacts: {artifacts_dir}")
        print(f"Schema: {schema_path}")
        if dry_run:
            print("(DRY RUN - no changes will be made)")

    added: List[str] = []
    removed: List[str] = []
    unchanged = 0

    # Get items from schema
    schema_items: Set[Tuple[str, ...]] = (
        set()
    )  # (category,), (category, element), (category, element, attr)

    categories = schema.get("children", [])
    for category in categories:
        cat_name = category.get("name", "")
        schema_items.add((cat_name,))

        for element in category.get("children", []):
            elem_name = element.get("name", "")
            schema_items.add((cat_name, elem_name))

            for attribute in element.get("children", []):
                attr_name = attribute.get("name", "")
                schema_items.add((cat_name, elem_name, attr_name))

    # Get items from artifacts
    artifact_items: Set[Tuple[str, ...]] = set()

    for cat_dir in artifacts_dir.iterdir():
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue

        cat_yaml = cat_dir / "_category.yaml"
        if cat_yaml.exists():
            with open(cat_yaml, "r") as f:
                cat_data = yaml.safe_load(f) or {}
            cat_name = cat_data.get("name", cat_dir.name)
            artifact_items.add((cat_name,))

            for elem_dir in cat_dir.iterdir():
                if not elem_dir.is_dir() or elem_dir.name.startswith("_"):
                    continue

                elem_yaml = elem_dir / "_element.yaml"
                if elem_yaml.exists():
                    with open(elem_yaml, "r") as f:
                        elem_data = yaml.safe_load(f) or {}
                    elem_name = elem_data.get("name", elem_dir.name)
                    artifact_items.add((cat_name, elem_name))

                    for attr_file in elem_dir.iterdir():
                        if (
                            attr_file.is_file()
                            and attr_file.suffix == ".yaml"
                            and not attr_file.name.startswith("_")
                        ):
                            with open(attr_file, "r") as f:
                                attr_data = yaml.safe_load(f) or {}
                            attr_name = attr_data.get("name", attr_file.stem)
                            artifact_items.add((cat_name, elem_name, attr_name))

    # Find what's new in schema
    new_items = schema_items - artifact_items

    # Find what's in artifacts but not in schema
    orphaned_items = artifact_items - schema_items

    if verbose:
        print(f"\nSchema items: {len(schema_items)}")
        print(f"Artifact items: {len(artifact_items)}")
        print(f"New in schema: {len(new_items)}")
        print(f"Orphaned in artifacts: {len(orphaned_items)}")

    # Create new items
    for item in sorted(new_items, key=lambda x: (len(x), x)):
        if len(item) == 1:
            # New category
            cat_name = item[0]
            cat_folder = sanitize_folder_name(cat_name)
            cat_dir = artifacts_dir / cat_folder

            if verbose:
                print(f"\n+ Creating category: {cat_name}/")

            # Find category data in schema
            cat_data = next((c for c in categories if c.get("name") == cat_name), {})

            if not dry_run:
                cat_dir.mkdir(parents=True, exist_ok=True)

                yaml_data = {
                    "name": cat_name,
                    "description": cat_data.get("definition", "")[:300],
                    "rules": ["# Add disambiguation rules"],
                    "examples": [{"comment": "# Add example", "reasoning": "# reasoning"}],
                }
                header = get_yaml_header(f"Category: {cat_name}")
                save_yaml(yaml_data, cat_dir / "_category.yaml", header)

            added.append(f"{cat_folder}/_category.yaml")

        elif len(item) == 2:
            # New element
            cat_name, elem_name = item
            cat_folder = sanitize_folder_name(cat_name)
            elem_folder = sanitize_folder_name(elem_name)
            elem_dir = artifacts_dir / cat_folder / elem_folder

            if verbose:
                print(f"  + Creating element: {elem_name}/")

            # Find element data in schema
            cat_data = next((c for c in categories if c.get("name") == cat_name), {})
            elem_data = next(
                (e for e in cat_data.get("children", []) if e.get("name") == elem_name), {}
            )

            if not dry_run:
                elem_dir.mkdir(parents=True, exist_ok=True)

                yaml_data = {
                    "name": elem_name,
                    "category": cat_name,
                    "description": elem_data.get("definition", "")[:300],
                    "rules": ["# Add disambiguation rules"],
                    "examples": [
                        {
                            "comment": "# Add example",
                            "excerpt": "# excerpt",
                            "sentiment": "positive",
                            "reasoning": "# reason",
                        }
                    ],
                }
                header = get_yaml_header(f"Element: {elem_name}")
                save_yaml(yaml_data, elem_dir / "_element.yaml", header)

            added.append(f"{cat_folder}/{elem_folder}/_element.yaml")

        elif len(item) == 3:
            # New attribute
            cat_name, elem_name, attr_name = item
            cat_folder = sanitize_folder_name(cat_name)
            elem_folder = sanitize_folder_name(elem_name)
            attr_filename = sanitize_folder_name(attr_name) + ".yaml"
            attr_path = artifacts_dir / cat_folder / elem_folder / attr_filename

            if verbose:
                print(f"    + Creating attribute: {attr_filename}")

            # Find attribute data in schema
            cat_data = next((c for c in categories if c.get("name") == cat_name), {})
            elem_data = next(
                (e for e in cat_data.get("children", []) if e.get("name") == elem_name), {}
            )
            attr_data = next(
                (a for a in elem_data.get("children", []) if a.get("name") == attr_name), {}
            )

            if not dry_run:
                yaml_data = {
                    "name": attr_name,
                    "category": cat_name,
                    "element": elem_name,
                    "description": attr_data.get("definition", "")[:300],
                    "rules": ["# Add disambiguation rules"],
                    "examples": [
                        {
                            "excerpt": "# Example excerpt",
                            "sentiment": "positive",
                            "reasoning": "# reason",
                        }
                    ],
                }
                header = get_yaml_header(f"Attribute: {attr_name}")
                save_yaml(yaml_data, attr_path, header)

            added.append(f"{cat_folder}/{elem_folder}/{attr_filename}")

    # Report orphaned items (don't delete)
    for item in sorted(orphaned_items, key=lambda x: (len(x), x)):
        if len(item) == 1:
            removed.append(f"Category '{item[0]}' not in schema (folder preserved)")
        elif len(item) == 2:
            removed.append(f"Element '{item[0]} > {item[1]}' not in schema (folder preserved)")
        elif len(item) == 3:
            removed.append(
                f"Attribute '{item[0]} > {item[1]} > {item[2]}' not in schema (file preserved)"
            )

    # Update _schema_ref.yaml
    if not dry_run:
        counts = {
            "categories": len([i for i in schema_items if len(i) == 1]),
            "elements": len([i for i in schema_items if len(i) == 2]),
            "attributes": len([i for i in schema_items if len(i) == 3]),
        }

        ref_data = {
            "schema_source": str(schema_path.absolute()),
            "synced_at": datetime.now().isoformat(),
            "structure": counts,
        }
        header = get_yaml_header("Schema Reference", editable=False)
        save_yaml(ref_data, artifacts_dir / "_schema_ref.yaml", header)

        if verbose:
            print("\n✓ Updated _schema_ref.yaml")

    # Calculate unchanged
    unchanged = len(artifact_items - new_items - orphaned_items)

    if verbose:
        print("\n" + "=" * 70)
        print("SYNC RESULT")
        print("=" * 70)
        print(f"Added: {len(added)}")
        print(f"Orphaned (preserved): {len(removed)}")
        print(f"Unchanged: {unchanged}")

        if removed:
            print("\n⚠ Orphaned items (files preserved, may need manual review):")
            for item in removed[:10]:
                print(f"  - {item}")
            if len(removed) > 10:
                print(f"  ... and {len(removed) - 10} more")

    return {
        "added": added,
        "removed": removed,
        "unchanged": unchanged,
        "dry_run": dry_run,
    }
