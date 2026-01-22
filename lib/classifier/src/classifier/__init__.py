"""
Classifier Library

Provides tools for building dynamic classification pipelines from taxonomy definitions.

Subpackages:
    - taxonomy: Build classification artifacts (models, prompts) from JSON taxonomies
    - pipeline: Run classification pipelines (future)

Quick access to commonly used items:
    from classifier import build_models_from_taxonomy, SentimentType

For full access:
    from classifier.taxonomy import (
        build_models_from_taxonomy,
        load_taxonomy_models,
        print_taxonomy_hierarchy,
        extract_valid_combinations,
        SentimentType,
        CategoryDetectionOutput,
    )
"""

# Re-export commonly used items at package level for convenience
from .taxonomy import (
    CategoryDetectionOutput,
    # Schemas
    SentimentType,
    # Model building
    build_models_from_taxonomy,
    extract_valid_combinations,
    load_taxonomy_models,
    # Utilities
    print_taxonomy_hierarchy,
    print_valid_combinations,
)

__all__ = [
    # Model building
    "build_models_from_taxonomy",
    "load_taxonomy_models",
    # Schemas
    "SentimentType",
    "CategoryDetectionOutput",
    # Utilities
    "print_taxonomy_hierarchy",
    "extract_valid_combinations",
    "print_valid_combinations",
]
