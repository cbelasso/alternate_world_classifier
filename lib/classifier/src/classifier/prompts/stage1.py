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

    # Build prompt function (pure Python) - with handcrafted examples
    stage1_prompt = build_stage1_prompt_function(condensed, use_handcrafted=True)

    # Or with generated examples (automatically curated)
    stage1_prompt = build_stage1_prompt_function(condensed, examples, max_examples_per_category=1)

    # Use at runtime
    prompt = stage1_prompt("The WiFi was terrible!")

    # Or export as a standalone Python module
    export_stage1_prompt_module(condensed, "prompts/stage1_prompt.py", use_handcrafted=True)
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
# Handcrafted Examples (High-Quality Category-Level Examples)
# =============================================================================

HANDCRAFTED_STAGE1_EXAMPLES = [
    # Single-category examples
    ClassificationExample(
        comment="The networking sessions were fantastic and I made great connections with peers from other institutions.",
        categories_present=["Attendee Engagement & Interaction"],
        has_classifiable_content=True,
        stage1_reasoning="Discusses networking and peer connections",
        element_details=[],
        example_type="simple",
        source_category="Attendee Engagement & Interaction",
    ),
    ClassificationExample(
        comment="The WiFi kept dropping during sessions and the room was too cold.",
        categories_present=["Event Logistics & Infrastructure"],
        has_classifiable_content=True,
        stage1_reasoning="Mentions WiFi connectivity and venue temperature issues",
        element_details=[],
        example_type="simple",
        source_category="Event Logistics & Infrastructure",
    ),
    ClassificationExample(
        comment="The registration process was slow and confusing.",
        categories_present=["Event Operations & Management"],
        has_classifiable_content=True,
        stage1_reasoning="Feedback about registration process",
        element_details=[],
        example_type="simple",
        source_category="Event Operations & Management",
    ),
    ClassificationExample(
        comment="The hands-on workshops were excellent and I learned so much from the practical exercises.",
        categories_present=["Learning & Content Delivery"],
        has_classifiable_content=True,
        stage1_reasoning="Feedback about workshop content and learning experience",
        element_details=[],
        example_type="simple",
        source_category="Learning & Content Delivery",
    ),
    ClassificationExample(
        comment="The conference staff was incredibly helpful and always available to answer questions.",
        categories_present=["People"],
        has_classifiable_content=True,
        stage1_reasoning="Praise for conference staff",
        element_details=[],
        example_type="simple",
        source_category="People",
    ),
    # Multi-category examples
    ClassificationExample(
        comment="The keynote speaker was brilliant and the presentation on machine learning was very insightful.",
        categories_present=["Learning & Content Delivery", "People"],
        has_classifiable_content=True,
        stage1_reasoning="Discusses both the presentation content and the speaker",
        element_details=[],
        example_type="multi_category",
    ),
    ClassificationExample(
        comment="The conference was well organized but I wish there were more hands-on workshops. The Explorance team was very helpful.",
        categories_present=[
            "Event Operations & Management",
            "Learning & Content Delivery",
            "People",
        ],
        has_classifiable_content=True,
        stage1_reasoning="Covers organization quality, workshop content request, and staff praise",
        element_details=[],
        example_type="multi_category",
    ),
    ClassificationExample(
        comment="I loved connecting with the Blue community and sharing knowledge with other users.",
        categories_present=["Attendee Engagement & Interaction"],
        has_classifiable_content=True,
        stage1_reasoning="Discusses community connection and knowledge sharing among attendees",
        element_details=[],
        example_type="simple",
        source_category="Attendee Engagement & Interaction",
    ),
    # Negative examples
    ClassificationExample(
        comment="Seeing is believing!",
        categories_present=[],
        has_classifiable_content=False,
        stage1_reasoning="Generic phrase without specific conference feedback context",
        element_details=[],
        example_type="negative",
    ),
    ClassificationExample(
        comment="Data integrity never goes out of style.",
        categories_present=[],
        has_classifiable_content=False,
        stage1_reasoning="General statement not specifically about conference feedback",
        element_details=[],
        example_type="negative",
    ),
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
# Example Curation Helper
# =============================================================================


def _curate_generated_examples(
    examples: ExampleSet | List[ClassificationExample],
    max_examples_per_category: int = 1,
) -> List[ClassificationExample]:
    """
    Curate generated examples to avoid element-level bloat.

    Takes at most max_examples_per_category simple examples per category,
    plus all multi-category and negative examples.

    Args:
        examples: ExampleSet or list of examples
        max_examples_per_category: Max simple examples per category

    Returns:
        Curated list of examples
    """
    if isinstance(examples, ExampleSet):
        simple = examples.get_simple_examples()
        multi = examples.get_multi_category_examples()
        negative = examples.get_negative_examples()
    else:
        # Separate by type
        simple = [ex for ex in examples if ex.example_type == "simple"]
        multi = [ex for ex in examples if ex.example_type == "multi_category"]
        negative = [ex for ex in examples if ex.example_type == "negative"]

    # Group simple examples by category and take max N per category
    from collections import defaultdict

    by_category = defaultdict(list)
    for ex in simple:
        if ex.source_category:
            by_category[ex.source_category].append(ex)

    curated_simple = []
    for cat_examples in by_category.values():
        curated_simple.extend(cat_examples[:max_examples_per_category])

    # Combine: curated simple + all multi + all negative
    return curated_simple + multi + negative


# =============================================================================
# Main Builder Functions
# =============================================================================


def build_stage1_prompt_string(
    comment: str,
    condensed: CondensedTaxonomy,
    examples: Optional[List[ClassificationExample]] = None,
    rules: Optional[List[str]] = None,
    use_handcrafted: bool = True,
) -> str:
    """
    Build a complete Stage 1 prompt for a specific comment.

    This is useful for one-off testing or when you need direct control.

    Args:
        comment: The conference feedback comment to analyze
        condensed: CondensedTaxonomy with category/element descriptions
        examples: List of ClassificationExample for few-shot learning (optional if use_handcrafted=True)
        rules: Optional custom rules (uses DEFAULT_STAGE1_RULES if not provided)
        use_handcrafted: If True, use HANDCRAFTED_STAGE1_EXAMPLES instead of provided examples

    Returns:
        Complete prompt string ready for LLM

    Example:
        >>> prompt = build_stage1_prompt_string(
        ...     "The WiFi was terrible!",
        ...     condensed,
        ...     use_handcrafted=True,
        ... )
        >>> result = processor.process_with_schema([prompt], CategoryDetectionOutput)
    """
    rules = rules or DEFAULT_STAGE1_RULES

    if use_handcrafted:
        example_list = HANDCRAFTED_STAGE1_EXAMPLES
    elif examples is None:
        raise ValueError("Must provide examples or set use_handcrafted=True")
    else:
        example_list = examples

    categories_section = format_categories_section(condensed)
    rules_section = format_rules_section(rules)
    examples_section = format_stage1_examples_section(example_list)

    return STAGE1_TEMPLATE.format(
        comment=comment,
        categories_section=categories_section,
        rules_section=rules_section,
        examples_section=examples_section,
    )


def build_stage1_prompt_function(
    condensed: CondensedTaxonomy,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    rules: Optional[List[str]] = None,
    use_handcrafted: bool = True,
    max_examples_per_category: int = 1,
) -> Callable[[str], str]:
    """
    Build a reusable Stage 1 prompt function.

    This creates a closure that "bakes in" the taxonomy and examples,
    returning a simple function that takes a comment and returns a prompt.

    Args:
        condensed: CondensedTaxonomy with category/element descriptions
        examples: ExampleSet or list of ClassificationExample (optional if use_handcrafted=True)
        rules: Optional custom rules (uses DEFAULT_STAGE1_RULES if not provided)
        use_handcrafted: If True, use HANDCRAFTED_STAGE1_EXAMPLES (default: True)
        max_examples_per_category: If using generated examples, max simple examples per category

    Returns:
        Function that takes a comment string and returns a complete prompt

    Example:
        >>> # Use handcrafted examples (recommended)
        >>> stage1_prompt = build_stage1_prompt_function(condensed, use_handcrafted=True)
        >>>
        >>> # Or use generated examples (auto-curated)
        >>> stage1_prompt = build_stage1_prompt_function(condensed, examples, use_handcrafted=False, max_examples_per_category=1)
        >>>
        >>> prompts = [stage1_prompt(c) for c in comments]
        >>> results = processor.process_with_schema(prompts, CategoryDetectionOutput)
    """
    rules = rules or DEFAULT_STAGE1_RULES

    # Select examples
    if use_handcrafted:
        example_list = HANDCRAFTED_STAGE1_EXAMPLES
    elif examples is None:
        raise ValueError("Must provide examples or set use_handcrafted=True")
    else:
        # Auto-curate generated examples to avoid bloat
        example_list = _curate_generated_examples(examples, max_examples_per_category)

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
    filepath: str | Path,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    rules: Optional[List[str]] = None,
    use_handcrafted: bool = True,
    max_examples_per_category: int = 1,
    function_name: str = "stage1_category_detection_prompt",
) -> Path:
    """
    Export the Stage 1 prompt as a standalone Python module.

    This creates a .py file that can be imported and used without
    needing the original condensed taxonomy or examples files.

    Args:
        condensed: CondensedTaxonomy
        filepath: Output path for the .py file
        examples: ExampleSet or list of ClassificationExample (optional if use_handcrafted=True)
        rules: Optional custom rules
        use_handcrafted: If True, use HANDCRAFTED_STAGE1_EXAMPLES (default: True)
        max_examples_per_category: If using generated examples, max simple examples per category
        function_name: Name of the generated function

    Returns:
        Path to the generated module

    Example:
        >>> export_stage1_prompt_module(condensed, "prompts/stage1.py", use_handcrafted=True)
        >>> # Later, in production:
        >>> from prompts.stage1 import stage1_category_detection_prompt
        >>> prompt = stage1_category_detection_prompt("Great conference!")
    """
    rules = rules or DEFAULT_STAGE1_RULES

    # Select examples
    if use_handcrafted:
        example_list = HANDCRAFTED_STAGE1_EXAMPLES
    elif examples is None:
        raise ValueError("Must provide examples or set use_handcrafted=True")
    else:
        # Auto-curate generated examples
        example_list = _curate_generated_examples(examples, max_examples_per_category)

    # Pre-compute sections
    categories_section = format_categories_section(condensed)
    rules_section = format_rules_section(rules)
    examples_section = format_stage1_examples_section(example_list)

    # Build the module content
    module_content = f'''# =============================================================================
# Stage 1: Category Detection Prompt
# =============================================================================
#
# Detects which categories are present in conference feedback comments.
#
# AUTO-GENERATED from {"handcrafted examples" if use_handcrafted else "curated generated examples"}
#
# Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# ⚠️  WARNING: Direct edits to this file will be LOST if regenerated!
#     To make permanent changes, edit the source and rebuild.
#
#     If you intentionally edited this file, change the flag below to True
#     to protect it from being overwritten during regeneration.
#
# MANUALLY_EDITED: False
# =============================================================================


def {function_name}(comment: str) -> str:
    """
    Generate Stage 1 category detection prompt.

    Args:
        comment: The conference feedback comment to analyze

    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in the following comment.

COMMENT TO ANALYZE:
{{comment}}

---

CATEGORIES TO CONSIDER:

{categories_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Analyze the comment and return ONLY valid JSON with:
- categories_present: list of category names that apply
- has_classifiable_content: true/false
- reasoning: brief explanation"""


# Convenience alias
STAGE1_PROMPT = {function_name}
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
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    rules: Optional[List[str]] = None,
    use_handcrafted: bool = True,
    max_examples_per_category: int = 1,
) -> dict:
    """
    Get statistics about the Stage 1 prompt.

    Useful for understanding token usage and prompt composition.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list (optional if use_handcrafted=True)
        rules: Optional custom rules
        use_handcrafted: If True, use handcrafted examples
        max_examples_per_category: If using generated, max per category

    Returns:
        Dict with statistics
    """
    rules = rules or DEFAULT_STAGE1_RULES

    if use_handcrafted:
        example_list = HANDCRAFTED_STAGE1_EXAMPLES
    elif examples is None:
        raise ValueError("Must provide examples or set use_handcrafted=True")
    else:
        example_list = _curate_generated_examples(examples, max_examples_per_category)

    # Build a sample prompt
    sample_prompt = build_stage1_prompt_string(
        "Sample comment for length estimation.",
        condensed,
        example_list,
        rules,
        use_handcrafted=False,  # Already have example_list
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
        "using_handcrafted": use_handcrafted,
    }


def print_stage1_prompt_preview(
    condensed: CondensedTaxonomy,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    sample_comment: str = "The networking sessions were great but the WiFi was terrible.",
    max_chars: int = 2000,
    use_handcrafted: bool = True,
    max_examples_per_category: int = 1,
) -> None:
    """
    Print a preview of the Stage 1 prompt.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list (optional if use_handcrafted=True)
        sample_comment: Comment to use in preview
        max_chars: Maximum characters to display
        use_handcrafted: If True, use handcrafted examples
        max_examples_per_category: If using generated, max per category
    """
    if use_handcrafted:
        example_list = HANDCRAFTED_STAGE1_EXAMPLES
    elif examples is None:
        raise ValueError("Must provide examples or set use_handcrafted=True")
    else:
        example_list = _curate_generated_examples(examples, max_examples_per_category)

    prompt = build_stage1_prompt_string(
        sample_comment, condensed, example_list, use_handcrafted=False
    )
    stats = get_stage1_prompt_stats(
        condensed,
        examples,
        use_handcrafted=use_handcrafted,
        max_examples_per_category=max_examples_per_category,
    )

    print("\n" + "=" * 70)
    print("STAGE 1 PROMPT PREVIEW")
    print("=" * 70)
    print(
        f"Total length: {stats['total_chars']:,} chars (~{stats['estimated_tokens']:,} tokens)"
    )
    print(f"Categories: {stats['num_categories']}")
    print(f"Elements: {stats['num_elements']}")
    print(
        f"Examples: {stats['num_examples']} ({'handcrafted' if stats['using_handcrafted'] else 'generated (curated)'})"
    )
    print(f"Rules: {stats['num_rules']}")
    print("-" * 70)

    if len(prompt) > max_chars:
        print(prompt[:max_chars])
        print(f"\n... (truncated, {len(prompt) - max_chars:,} more chars)")
    else:
        print(prompt)
