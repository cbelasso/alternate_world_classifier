"""
Classifier Library

Provides tools for building dynamic classification pipelines from taxonomy definitions.

Architecture:
    1. PREPARATION PHASE (done once, with LLM help)
       - Condense taxonomy definitions
       - Generate training examples
       - Assemble prompts

    2. RUNTIME PHASE (production inference)
       - Load pre-built artifacts
       - Run classification with orchestrator

Subpackages:
    - taxonomy: Build classification artifacts (models, prompts) from JSON taxonomies
    - prompts: Assemble classification prompts (pure Python)
    - pipeline: Run classification pipelines (future)

Quick Start:
    # Load and condense taxonomy
    from classifier.taxonomy import condense_taxonomy, save_condensed
    from classifier.taxonomy import generate_all_examples, save_examples
    from classifier.prompts import build_stage1_prompt_function

    condensed = condense_taxonomy(taxonomy, processor)
    examples = generate_all_examples(condensed, processor)
    stage1_prompt = build_stage1_prompt_function(condensed, examples)

    # Use for classification
    prompts = [stage1_prompt(c) for c in comments]
    results = processor.process_with_schema(prompts, CategoryDetectionOutput)

For production (with pre-built artifacts):
    from classifier.taxonomy import load_condensed, load_examples
    from classifier.prompts import build_stage1_prompt_function

    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")
    stage1_prompt = build_stage1_prompt_function(condensed, examples)
"""

# =============================================================================
# Core Schemas (always needed)
# =============================================================================

# =============================================================================
# Prompt Building (pure Python)
# =============================================================================
from .prompts import (
    DEFAULT_STAGE1_RULES,
    build_stage1_prompt_function,
    build_stage1_prompt_string,
    export_stage1_prompt_module,
)

# =============================================================================
# Taxonomy Types (for type hints)
# =============================================================================
# =============================================================================
# Condensation (requires LLM for generation, pure Python for loading)
# =============================================================================
from .taxonomy.condenser import (
    CondensedCategory,
    CondensedElement,
    CondensedTaxonomy,
    check_condensation_quality,
    condense_taxonomy,
    load_condensed,
    print_condensed_preview,
    save_condensed,
)

# =============================================================================
# Example Generation (requires LLM for generation, pure Python for loading)
# =============================================================================
from .taxonomy.example_generator import (
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

# =============================================================================
# Model Building
# =============================================================================
from .taxonomy.model_builder import build_models_from_taxonomy, load_taxonomy_models
from .taxonomy.schemas import CategoryDetectionOutput, SentimentType

# =============================================================================
# Utilities
# =============================================================================
from .taxonomy.utils import (
    get_taxonomy_stats,
    print_taxonomy_hierarchy,
    print_valid_combinations,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core schemas
    "SentimentType",
    "CategoryDetectionOutput",
    # Taxonomy types
    "CondensedTaxonomy",
    "CondensedCategory",
    "CondensedElement",
    "ExampleSet",
    "ClassificationExample",
    "ElementDetail",
    # Model building
    "build_models_from_taxonomy",
    "load_taxonomy_models",
    # Condensation
    "condense_taxonomy",
    "save_condensed",
    "load_condensed",
    "check_condensation_quality",
    "print_condensed_preview",
    # Examples
    "generate_simple_examples",
    "generate_complex_examples",
    "generate_all_examples",
    "combine_examples",
    "save_examples",
    "load_examples",
    "validate_examples",
    "print_examples_preview",
    # Prompts
    "build_stage1_prompt_function",
    "build_stage1_prompt_string",
    "export_stage1_prompt_module",
    "DEFAULT_STAGE1_RULES",
    # Utilities
    "get_taxonomy_stats",
    "print_taxonomy_hierarchy",
    "print_valid_combinations",
]
