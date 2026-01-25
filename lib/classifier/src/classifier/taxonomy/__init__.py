"""
Taxonomy Module

Tools for working with classification taxonomies:
    - Building dynamic Pydantic models from JSON taxonomies
    - Condensing verbose definitions for classification prompts
    - Generating training examples

Submodules:
    - schemas: Static Pydantic schemas (SentimentType, CategoryDetectionOutput)
    - model_builder: Dynamic model generation from taxonomy
    - condenser: Condense verbose taxonomy definitions
    - example_generator: Generate training examples
    - utils: Taxonomy inspection and manipulation utilities

Usage:
    # Static schemas
    from classifier.taxonomy import SentimentType, CategoryDetectionOutput

    # Dynamic model building
    from classifier.taxonomy import build_models_from_taxonomy, load_taxonomy_models

    # Condensation (requires LLM)
    from classifier.taxonomy import condense_taxonomy, CondensedTaxonomy
    from classifier.taxonomy import save_condensed, load_condensed

    # Example generation (requires LLM)
    from classifier.taxonomy import generate_all_examples, ExampleSet
    from classifier.taxonomy import save_examples, load_examples

    # Utilities
    from classifier.taxonomy import print_taxonomy_hierarchy, get_taxonomy_stats
"""

# Static schemas
# Condenser
from .condenser import (
    CondensedCategory,
    CondensedElement,
    CondensedTaxonomy,
    check_condensation_quality,
    condense_taxonomy,
    load_condensed,
    print_condensed_preview,
    save_condensed,
)

# Example generator
from .example_generator import (
    ClassificationExample,
    ElementDetail,
    ExampleSet,
    combine_examples,
    generate_all_examples,
    generate_complex_examples,
    generate_simple_examples,
    load_examples,
    print_examples_preview,
    save_examples,
    validate_examples,
)

# Model building
from .model_builder import build_models_from_taxonomy, load_taxonomy_models

# Rule generator
from .rule_generator import (
    DEFAULT_STAGE1_BASE_RULES,
    DEFAULT_STAGE2_BASE_RULES,
    CategoryRules,
    ClassificationRules,
    create_default_rules,
    generate_all_rules,
    generate_category_rules,
    load_rules,
    merge_rules,
    print_rules_preview,
    save_rules,
)
from .schemas import (
    CategoryDetectionOutput,
    ClassificationSpan,
    ElementExtractionOutput,
    ElementExtractionSpan,
    FinalClassificationOutput,
    SentimentType,
)

# Utilities
from .utils import (
    extract_valid_combinations,
    get_attributes_for_element,
    get_category_names,
    get_elements_for_category,
    get_taxonomy_stats,
    print_taxonomy_hierarchy,
    print_valid_combinations,
    sanitize_model_name,
)

__all__ = [
    # Static schemas
    "SentimentType",
    "CategoryDetectionOutput",
    "ElementExtractionSpan",
    "ElementExtractionOutput",
    "ClassificationSpan",
    "FinalClassificationOutput",
    # Model building
    "build_models_from_taxonomy",
    "load_taxonomy_models",
    # Condenser types
    "CondensedTaxonomy",
    "CondensedCategory",
    "CondensedElement",
    # Condenser functions
    "condense_taxonomy",
    "save_condensed",
    "load_condensed",
    "check_condensation_quality",
    "print_condensed_preview",
    # Example types
    "ExampleSet",
    "ClassificationExample",
    "ElementDetail",
    # Example functions
    "generate_simple_examples",
    "generate_complex_examples",
    "generate_all_examples",
    "combine_examples",
    "save_examples",
    "load_examples",
    "validate_examples",
    "print_examples_preview",
    # Rule types
    "ClassificationRules",
    "CategoryRules",
    # Rule functions
    "generate_category_rules",
    "generate_all_rules",
    "create_default_rules",
    "save_rules",
    "load_rules",
    "merge_rules",
    "print_rules_preview",
    "DEFAULT_STAGE1_BASE_RULES",
    "DEFAULT_STAGE2_BASE_RULES",
    # Utilities
    "sanitize_model_name",
    "print_taxonomy_hierarchy",
    "extract_valid_combinations",
    "print_valid_combinations",
    "get_taxonomy_stats",
    "get_category_names",
    "get_elements_for_category",
    "get_attributes_for_element",
]
