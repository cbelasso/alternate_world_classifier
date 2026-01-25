"""
Prompts Module

Pure Python templating for assembling classification prompts.

This module takes condensed taxonomy and curated examples to produce
prompt functions that can be used at runtime for classification.

NO LLM required - just string formatting!

Submodules:
    - base: Shared formatting utilities
    - stage1: Stage 1 category detection prompts
    - stage2: Stage 2 element extraction prompts

Usage:
    from classifier.prompts import (
        build_stage1_prompt_function,
        build_stage2_prompt_functions,
    )
    from classifier.taxonomy.condenser import load_condensed
    from classifier.taxonomy.example_generator import load_examples

    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    # Stage 1: Category detection
    stage1_prompt = build_stage1_prompt_function(condensed, examples)
    prompt = stage1_prompt("The WiFi was terrible!")

    # Stage 2: Element extraction (one prompt per category)
    stage2_prompts = build_stage2_prompt_functions(condensed, examples)
    people_prompt = stage2_prompts["People"]
    prompt = people_prompt("The speaker was brilliant!")
"""

# Hierarchical exporter
from .exporter import (
    export_prompts_hierarchical,
    export_prompts_hierarchical_with_stage3,
    export_stage1_prompt,
    export_stage2_prompts,
    export_stage3_prompts,
    is_manually_edited,
    sanitize_name,
)
from .stage1 import (
    DEFAULT_STAGE1_RULES,
    build_stage1_prompt_function,
    build_stage1_prompt_string,
    export_stage1_prompt_module,
)
from .stage2 import (
    CATEGORY_SPECIFIC_RULES,
    DEFAULT_STAGE2_RULES,
    build_stage2_prompt_function,
    build_stage2_prompt_functions,
    build_stage2_prompt_string,
    export_stage2_prompt_module,
    export_stage2_prompt_modules,
    get_stage2_examples_for_category,
    get_stage2_prompt_stats,
    print_stage2_prompts_preview,
)

# Stage 3
from .stage3 import (
    DEFAULT_STAGE3_BASE_RULES,
    build_stage3_prompt_function,
    build_stage3_prompt_functions,
    get_all_category_element_pairs,
    get_stage3_examples_for_element,
    get_stage3_prompt_stats,
    print_stage3_prompts_preview,
)

__all__ = [
    # Stage 1
    "build_stage1_prompt_function",
    "build_stage1_prompt_string",
    "export_stage1_prompt_module",
    "DEFAULT_STAGE1_RULES",
    # Stage 2
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
    # Stage 3
    "build_stage3_prompt_function",
    "build_stage3_prompt_functions",
    "get_stage3_examples_for_element",
    "get_stage3_prompt_stats",
    "print_stage3_prompts_preview",
    "get_all_category_element_pairs",
    "DEFAULT_STAGE3_BASE_RULES",
    # Hierarchical exporter
    "export_prompts_hierarchical",
    "export_prompts_hierarchical_with_stage3",
    "export_stage1_prompt",
    "export_stage2_prompts",
    "export_stage3_prompts",
    "is_manually_edited",
    "sanitize_name",
]
