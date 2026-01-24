"""
Classifier Library

Provides tools for building dynamic classification pipelines from taxonomy definitions.

Architecture:
    1. PREPARATION PHASE (done once, with LLM help)
       - Condense taxonomy definitions
       - Generate training examples
       - Assemble prompts (Stage 1 and Stage 2)

    2. RUNTIME PHASE (production inference)
       - Load pre-built artifacts
       - Run classification with orchestrator

Subpackages:
    - taxonomy: Build classification artifacts (models, prompts) from JSON taxonomies
    - prompts: Assemble classification prompts (pure Python)
    - pipeline: Run classification pipelines

Quick Start:
    # Load and condense taxonomy
    from classifier import condense_taxonomy, generate_all_examples
    from classifier import build_stage1_prompt_function, build_stage2_prompt_functions

    condensed = condense_taxonomy(taxonomy, processor)
    examples = generate_all_examples(condensed, processor)

    # Stage 1: Category detection
    stage1_prompt = build_stage1_prompt_function(condensed, examples)

    # Stage 2: Element extraction (one per category)
    stage2_prompts = build_stage2_prompt_functions(condensed, examples)

For production (with pre-built artifacts):
    from classifier import load_condensed, load_examples
    from classifier import build_stage1_prompt_function, build_stage2_prompt_functions

    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    stage1_prompt = build_stage1_prompt_function(condensed, examples)
    stage2_prompts = build_stage2_prompt_functions(condensed, examples)
"""

# =============================================================================
# Core Schemas (always needed)
# =============================================================================

# =============================================================================
# Pipeline (Orchestrator)
# =============================================================================
from .pipeline import ClassificationOrchestrator, create_orchestrator

# =============================================================================
# Prompt Building - Stage 1 (pure Python)
# =============================================================================
# =============================================================================
# Prompt Building - Stage 2 (pure Python)
# =============================================================================
from .prompts import (
    CATEGORY_SPECIFIC_RULES,
    DEFAULT_STAGE1_RULES,
    DEFAULT_STAGE2_RULES,
    build_stage1_prompt_function,
    build_stage1_prompt_string,
    build_stage2_prompt_function,
    build_stage2_prompt_functions,
    build_stage2_prompt_string,
    export_stage1_prompt_module,
    export_stage2_prompt_module,
    export_stage2_prompt_modules,
    get_stage2_examples_for_category,
    get_stage2_prompt_stats,
    print_stage2_prompts_preview,
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
from .taxonomy.schemas import (
    CategoryDetectionOutput,
    ClassificationSpan,
    ElementExtractionOutput,
    ElementExtractionSpan,
    FinalClassificationOutput,
    SentimentType,
)

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
    "ElementExtractionSpan",
    "ElementExtractionOutput",
    "ClassificationSpan",
    "FinalClassificationOutput",
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
    # Stage 1 Prompts
    "build_stage1_prompt_function",
    "build_stage1_prompt_string",
    "export_stage1_prompt_module",
    "DEFAULT_STAGE1_RULES",
    # Stage 2 Prompts
    "build_stage2_prompt_function",
    "build_stage2_prompt_functions",
    "build_stage2_prompt_string",
    "export_stage2_prompt_module",
    "export_stage2_prompt_modules",
    "get_stage2_examples_for_category",
    "get_stage2_prompt_stats",
    "print_stage2_prompts_preview",
    "DEFAULT_STAGE2_RULES",
    "CATEGORY_SPECIFIC_RULES",
    # Pipeline / Orchestrator
    "ClassificationOrchestrator",
    "create_orchestrator",
    # Utilities
    "get_taxonomy_stats",
    "print_taxonomy_hierarchy",
    "print_valid_combinations",
]
