"""
Prompts Module

Pure Python templating for assembling classification prompts.

This module takes condensed taxonomy and curated examples to produce
prompt functions that can be used at runtime for classification.

NO LLM required - just string formatting!

Submodules:
    - base: Shared formatting utilities
    - stage1: Stage 1 category detection prompts
    - stage2: Stage 2 element extraction prompts (future)

Usage:
    from classifier.prompts import build_stage1_prompt_function
    from classifier.taxonomy.condenser import load_condensed
    from classifier.taxonomy.example_generator import load_examples

    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    stage1_prompt = build_stage1_prompt_function(condensed, examples)

    # Use at runtime
    prompt = stage1_prompt("The WiFi was terrible but the speakers were great!")
"""

from .stage1 import (
    DEFAULT_STAGE1_RULES,
    build_stage1_prompt_function,
    build_stage1_prompt_string,
    export_stage1_prompt_module,
)

__all__ = [
    "build_stage1_prompt_function",
    "build_stage1_prompt_string",
    "export_stage1_prompt_module",
    "DEFAULT_STAGE1_RULES",
]
