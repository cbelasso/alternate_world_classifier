"""
Stage 1 Prompt Builder

Assembles Stage 1 (category detection) prompts from condensed taxonomy and examples.

This is pure Python templating - NO LLM required!

Usage:
    from classifier.prompts import build_stage1_prompt_function
    from classifier.taxonomy.condenser import load_condensed
    from classifier.taxonomy.example_generator import load_examples

    # Load artifacts
    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    # Build prompt function (pure Python)
    stage1_prompt = build_stage1_prompt_function(condensed, examples)

    # Use at runtime
    prompt = stage1_prompt("The WiFi was terrible!")

    # Or export as a standalone Python module
    export_stage1_prompt_module(condensed, examples, "prompts/stage1_prompt.py")
"""

from pathlib import Path
from typing import Callable, List, Optional

from ..taxonomy.condenser import CondensedTaxonomy
from ..taxonomy.example_generator import ClassificationExample, ExampleSet
from .base import (
    format_categories_section,
    format_rules_section,
    format_stage1_examples_section,
)

# =============================================================================
# Default Classification Rules
# =============================================================================

DEFAULT_STAGE1_RULES = [
    "A comment can belong to MULTIPLE categories if it discusses multiple aspects.",
    "Focus on what the comment is ABOUT, not just words mentioned.",
    '"Community" refers to the feeling of belonging; "Networking" refers to the act of meeting people.',
    '"Presentations" = talk quality/content; "Speakers/Presenters" = the people themselves.',
    'General praise like "great conference" without specifics → Event Operations & Management > Conference.',
    "If a comment mentions both the content AND the presenter, include BOTH categories.",
]


# =============================================================================
# Prompt Template
# =============================================================================

STAGE1_TEMPLATE = """You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in a conference attendee comment.

COMMENT TO ANALYZE:
{comment}

---

CATEGORIES AND THEIR SCOPE:

{categories_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Analyze the comment and return ONLY valid JSON."""


# =============================================================================
# Main Builder Functions
# =============================================================================


def build_stage1_prompt_string(
    comment: str,
    condensed: CondensedTaxonomy,
    examples: List[ClassificationExample],
    rules: Optional[List[str]] = None,
) -> str:
    """
    Build a complete Stage 1 prompt for a specific comment.

    This is useful for one-off testing or when you need direct control.

    Args:
        comment: The conference feedback comment to analyze
        condensed: CondensedTaxonomy with category/element descriptions
        examples: List of ClassificationExample for few-shot learning
        rules: Optional custom rules (uses DEFAULT_STAGE1_RULES if not provided)

    Returns:
        Complete prompt string ready for LLM

    Example:
        >>> prompt = build_stage1_prompt_string(
        ...     "The WiFi was terrible!",
        ...     condensed,
        ...     examples.examples,
        ... )
        >>> result = processor.process_with_schema([prompt], CategoryDetectionOutput)
    """
    rules = rules or DEFAULT_STAGE1_RULES

    categories_section = format_categories_section(condensed)
    rules_section = format_rules_section(rules)
    examples_section = format_stage1_examples_section(examples)

    return STAGE1_TEMPLATE.format(
        comment=comment,
        categories_section=categories_section,
        rules_section=rules_section,
        examples_section=examples_section,
    )


def build_stage1_prompt_function(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: Optional[List[str]] = None,
) -> Callable[[str], str]:
    """
    Build a reusable Stage 1 prompt function.

    This creates a closure that "bakes in" the taxonomy and examples,
    returning a simple function that takes a comment and returns a prompt.

    Args:
        condensed: CondensedTaxonomy with category/element descriptions
        examples: ExampleSet or list of ClassificationExample
        rules: Optional custom rules (uses DEFAULT_STAGE1_RULES if not provided)

    Returns:
        Function that takes a comment string and returns a complete prompt

    Example:
        >>> stage1_prompt = build_stage1_prompt_function(condensed, examples)
        >>> prompts = [stage1_prompt(c) for c in comments]
        >>> results = processor.process_with_schema(prompts, CategoryDetectionOutput)
    """
    rules = rules or DEFAULT_STAGE1_RULES

    # Handle both ExampleSet and raw list
    if isinstance(examples, ExampleSet):
        example_list = examples.examples
    else:
        example_list = examples

    # Pre-compute the static parts (these don't change per-comment)
    categories_section = format_categories_section(condensed)
    rules_section = format_rules_section(rules)
    examples_section = format_stage1_examples_section(example_list)

    # Build the template with static parts filled in
    static_template = STAGE1_TEMPLATE.format(
        comment="{comment}",  # Leave this as placeholder
        categories_section=categories_section,
        rules_section=rules_section,
        examples_section=examples_section,
    )

    def prompt_function(comment: str) -> str:
        """Generate Stage 1 classification prompt for a comment."""
        return static_template.format(comment=comment)

    return prompt_function


# =============================================================================
# Export as Python Module
# =============================================================================


def export_stage1_prompt_module(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    filepath: str | Path,
    rules: Optional[List[str]] = None,
    function_name: str = "stage1_category_detection_prompt",
) -> Path:
    """
    Export the Stage 1 prompt as a standalone Python module.

    This creates a .py file that can be imported and used without
    needing the original condensed taxonomy or examples files.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list of ClassificationExample
        filepath: Output path for the .py file
        rules: Optional custom rules
        function_name: Name of the generated function

    Returns:
        Path to the generated module

    Example:
        >>> export_stage1_prompt_module(condensed, examples, "prompts/stage1.py")
        >>> # Later, in production:
        >>> from prompts.stage1 import stage1_category_detection_prompt
        >>> prompt = stage1_category_detection_prompt("Great conference!")
    """
    rules = rules or DEFAULT_STAGE1_RULES

    # Handle both ExampleSet and raw list
    if isinstance(examples, ExampleSet):
        example_list = examples.examples
    else:
        example_list = examples

    # Pre-compute sections
    categories_section = format_categories_section(condensed)
    rules_section = format_rules_section(rules)
    examples_section = format_stage1_examples_section(example_list)

    # Escape for f-string embedding (double the braces in JSON examples)
    # The examples_section already has doubled braces from format_stage1_example

    # Build the module content
    module_content = f'''"""
Stage 1: Category Detection Prompt

Auto-generated from condensed taxonomy and curated examples.

Usage:
    from {Path(filepath).stem} import {function_name}

    prompt = {function_name}("Your comment here")
"""


def {function_name}(comment: str) -> str:
    """
    Generate Stage 1 classification prompt for category detection.

    Args:
        comment: The conference feedback comment to analyze

    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in a conference attendee comment.

COMMENT TO ANALYZE:
{{comment}}

---

CATEGORIES AND THEIR SCOPE:

{categories_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Analyze the comment and return ONLY valid JSON."""
'''

    # Write to file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(module_content)

    return filepath


# =============================================================================
# Prompt Statistics
# =============================================================================


def get_stage1_prompt_stats(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: Optional[List[str]] = None,
) -> dict:
    """
    Get statistics about the Stage 1 prompt.

    Useful for understanding token usage and prompt composition.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        rules: Optional custom rules

    Returns:
        Dict with statistics
    """
    rules = rules or DEFAULT_STAGE1_RULES

    if isinstance(examples, ExampleSet):
        example_list = examples.examples
    else:
        example_list = examples

    # Build a sample prompt
    sample_prompt = build_stage1_prompt_string(
        "Sample comment for length estimation.",
        condensed,
        example_list,
        rules,
    )

    # Estimate tokens (rough: ~4 chars per token)
    estimated_tokens = len(sample_prompt) // 4

    return {
        "total_chars": len(sample_prompt),
        "estimated_tokens": estimated_tokens,
        "num_categories": len(condensed.categories),
        "num_elements": sum(len(cat.elements) for cat in condensed.categories),
        "num_examples": len(example_list),
        "num_rules": len(rules),
    }


def print_stage1_prompt_preview(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    sample_comment: str = "The networking sessions were great but the WiFi was terrible.",
    max_chars: int = 2000,
) -> None:
    """
    Print a preview of the Stage 1 prompt.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        sample_comment: Comment to use in preview
        max_chars: Maximum characters to display
    """
    if isinstance(examples, ExampleSet):
        example_list = examples.examples
    else:
        example_list = examples

    prompt = build_stage1_prompt_string(sample_comment, condensed, example_list)
    stats = get_stage1_prompt_stats(condensed, examples)

    print("\n" + "=" * 70)
    print("STAGE 1 PROMPT PREVIEW")
    print("=" * 70)
    print(
        f"Total length: {stats['total_chars']:,} chars (~{stats['estimated_tokens']:,} tokens)"
    )
    print(f"Categories: {stats['num_categories']}")
    print(f"Elements: {stats['num_elements']}")
    print(f"Examples: {stats['num_examples']}")
    print(f"Rules: {stats['num_rules']}")
    print("-" * 70)

    if len(prompt) > max_chars:
        print(prompt[:max_chars])
        print(f"\n... (truncated, {len(prompt) - max_chars:,} more chars)")
    else:
        print(prompt)
