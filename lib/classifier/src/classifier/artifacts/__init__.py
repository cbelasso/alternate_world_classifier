"""
Artifacts Module

YAML-based artifact management for the classifier system.

This module provides a hierarchical folder structure for managing
taxonomy descriptions, rules, and examples. Each category, element,
and attribute has its own YAML file, making it easy to:
- Find and edit specific items
- Track changes with git
- Work in parallel without conflicts

Folder Structure:
    artifacts/
    ├── _schema_ref.yaml              # Reference to source schema
    ├── attendee_engagement/          # Category folder
    │   ├── _category.yaml            # Category config + Stage 1 examples
    │   ├── community/                # Element folder
    │   │   ├── _element.yaml         # Element config + Stage 2 examples
    │   │   ├── comfort_level.yaml    # Attribute config + Stage 3 examples
    │   │   └── engagement.yaml
    │   └── networking/
    │       └── ...
    └── ...

Usage:
    # Create initial folder structure from schema
    from classifier.artifacts import scaffold_artifacts
    scaffold_artifacts("schema.json", "artifacts/")

    # Or migrate from existing JSON artifacts
    from classifier.artifacts import scaffold_from_existing
    scaffold_from_existing(
        schema_path="schema.json",
        condensed_path="condensed.json",
        examples_path="examples.json",
        rules_path="rules.json",
        output_dir="artifacts/",
    )

    # Load artifacts for use with classifier
    from classifier.artifacts import build_classifier_objects
    condensed, examples, rules = build_classifier_objects("artifacts/")

    # Validate artifacts
    from classifier.artifacts import validate_artifacts
    result = validate_artifacts("artifacts/", "schema.json")

    # Sync with updated schema
    from classifier.artifacts import sync_artifacts
    sync_artifacts("artifacts/", "schema_v2.json")
"""

# Schemas
# Building
from .builder import (
    artifacts_to_condensed,
    artifacts_to_examples,
    artifacts_to_rules,
    build_classifier_objects,
    build_from_folder,
)

# Generation (requires LLM)
from .generator import (
    generate_artifact_content,
    generate_descriptions,
    generate_examples,
    generate_rules,
)

# Scaffolding
from .scaffold import (
    scaffold_artifacts,
    scaffold_from_existing,
)
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

# Sync
from .sync import (
    sync_artifacts,
)

# Validation
from .validator import (
    check_examples_quality,
    validate_artifacts,
)

__all__ = [
    # Schemas
    "AttributeExample",
    "AttributeFile",
    "CategoryExample",
    "CategoryFile",
    "ElementExample",
    "ElementFile",
    "LoadedArtifacts",
    "LoadedAttribute",
    "LoadedCategory",
    "LoadedElement",
    "SchemaRefFile",
    # Scaffolding
    "scaffold_artifacts",
    "scaffold_from_existing",
    # Building
    "build_from_folder",
    "build_classifier_objects",
    "artifacts_to_condensed",
    "artifacts_to_examples",
    "artifacts_to_rules",
    # Validation
    "validate_artifacts",
    "check_examples_quality",
    # Sync
    "sync_artifacts",
    # Generation
    "generate_artifact_content",
    "generate_descriptions",
    "generate_examples",
    "generate_rules",
]
