"""
Stage 2 Prompt Builder

Assembles Stage 2 (element extraction) prompts from condensed taxonomy and examples.

Stage 2 prompts are CATEGORY-SPECIFIC - each category gets its own prompt function
that extracts elements relevant to that category.

This is pure Python templating - NO LLM required!

Usage:
    from classifier.prompts import build_stage2_prompt_functions
    from classifier.taxonomy.condenser import load_condensed
    from classifier.taxonomy.example_generator import load_examples

    # Load artifacts
    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    # Build prompt functions (one per category)
    stage2_prompts = build_stage2_prompt_functions(condensed, examples)

    # Use at runtime
    people_prompt = stage2_prompts["People"]
    prompt = people_prompt("The speaker was brilliant!")

    # Or export as standalone Python module
    export_stage2_prompt_module(condensed, examples, "prompts/stage2_prompts.py")
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..taxonomy.condenser import CondensedCategory, CondensedTaxonomy
from ..taxonomy.example_generator import ClassificationExample, ExampleSet

# =============================================================================
# Default Classification Rules (Category-Specific)
# =============================================================================

DEFAULT_STAGE2_RULES = [
    "Extract the EXACT excerpt from the comment that relates to each element.",
    "Each excerpt should be classified to ONE element only.",
    "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
    "If multiple distinct excerpts relate to the same element, create separate entries.",
    "Only extract elements that are clearly present - do not infer or assume.",
]

# Category-specific disambiguation rules
CATEGORY_SPECIFIC_RULES: Dict[str, List[str]] = {
    "Attendee Engagement & Interaction": [
        '"Community" = feeling of belonging; "Networking" = act of meeting/connecting.',
        '"Knowledge Exchange" = peer-to-peer learning; different from formal presentations.',
    ],
    "Event Logistics & Infrastructure": [
        '"Conference Application/Software" = apps for attendees; "Technological Tools" = A/V equipment for sessions.',
        '"Conference Venue" = physical space; "Hotel" = accommodation.',
    ],
    "Event Operations & Management": [
        'General praise like "great conference" or "well organized" → Conference element.',
        "Comments about session timing or agenda structure → Conference Scheduling.",
    ],
    "Learning & Content Delivery": [
        '"Presentations" = quality of talks; "Topics" = what subjects were covered.',
        '"Session/Workshop" = format of learning; "Gained Knowledge" = what was learned.',
        'Requests for "more workshops" or "hands-on sessions" → Session/Workshop.',
    ],
    "People": [
        '"Explorance team" as hosts/organizers → Conference Staff.',
        '"Explorance experts" for knowledge/consulting → Experts/Consultants.',
        '"Blue users" or "community members" → Participants/Attendees.',
        'Named speakers or "the presenter" → Speakers/Presenters.',
    ],
}


# =============================================================================
# Prompt Template
# =============================================================================

STAGE2_TEMPLATE = """You are an expert conference feedback analyzer. Extract specific feedback related to {category_name} from this comment.

COMMENT TO ANALYZE:
{comment}

---

ELEMENTS TO IDENTIFY:

{elements_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to {category_name}, return {{"classifications": []}}."""


# =============================================================================
# Formatting Helpers
# =============================================================================


def format_elements_section(category: CondensedCategory) -> str:
    """
    Format elements for a specific category.

    Output format:
        **Element Name**: Element description

    Args:
        category: CondensedCategory instance

    Returns:
        Formatted string
    """
    lines = []
    for element in category.elements:
        lines.append(f"**{element.name}**: {element.short_description}")
    return "\n".join(lines)


def format_stage2_rules(
    category_name: str,
    base_rules: Optional[List[str]] = None,
) -> str:
    """
    Format rules for Stage 2, including category-specific rules.

    Args:
        category_name: Name of the category
        base_rules: Base rules (uses DEFAULT_STAGE2_RULES if not provided)

    Returns:
        Formatted numbered rules
    """
    rules = list(base_rules or DEFAULT_STAGE2_RULES)

    # Add category-specific rules
    specific_rules = CATEGORY_SPECIFIC_RULES.get(category_name, [])
    rules.extend(specific_rules)

    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))


def get_stage2_examples_for_category(
    examples: ExampleSet | List[ClassificationExample],
    category_name: str,
) -> List[ClassificationExample]:
    """
    Filter examples to only those relevant to a specific category.

    An example is relevant if:
    - It has element_details for this category, OR
    - Its source_category matches this category

    Args:
        examples: ExampleSet or list of ClassificationExample
        category_name: Category to filter for

    Returns:
        List of relevant examples
    """
    if isinstance(examples, ExampleSet):
        example_list = examples.examples
    else:
        example_list = examples

    relevant = []
    for ex in example_list:
        # Check if any element_details match this category
        has_relevant_details = any(
            detail.category == category_name for detail in ex.element_details
        )

        # Check if source_category matches (for simple examples)
        is_source_category = ex.source_category == category_name

        # Check if category is in categories_present
        in_categories = category_name in ex.categories_present

        if has_relevant_details or (is_source_category and in_categories):
            relevant.append(ex)

    return relevant


def format_stage2_example(example: ClassificationExample, category_name: str) -> str:
    """
    Format a single example for Stage 2 prompt.

    Only includes element_details that match the category.

    Args:
        example: ClassificationExample instance
        category_name: Category to filter for

    Returns:
        Formatted example string
    """
    import json

    # Filter element_details to only this category
    relevant_details = [
        detail
        for detail in example.element_details
        if detail.category == category_name or example.source_category == category_name
    ]

    if not relevant_details:
        return None

    # Format classifications array
    classifications = []
    for detail in relevant_details:
        classifications.append(
            {
                "excerpt": detail.excerpt,
                "reasoning": detail.reasoning,
                "element": detail.element,
                "sentiment": detail.sentiment,
            }
        )

    output = {"classifications": classifications}
    output_str = json.dumps(output, indent=2)

    return f'Comment: "{example.comment}"\n{output_str}'


def format_stage2_examples_section(
    examples: List[ClassificationExample],
    category_name: str,
) -> str:
    """
    Format all examples for a Stage 2 category prompt.

    Args:
        examples: List of ClassificationExample
        category_name: Category to filter for

    Returns:
        Formatted examples section
    """
    formatted = []
    for ex in examples:
        formatted_ex = format_stage2_example(ex, category_name)
        if formatted_ex:
            formatted.append(formatted_ex)

    if not formatted:
        # Provide a generic example structure
        return f"""Comment: "Example feedback about {category_name}."
{{"classifications": [{{"excerpt": "Example feedback", "reasoning": "Relates to element", "element": "ElementName", "sentiment": "positive"}}]}}"""

    return "\n\n".join(formatted)


# =============================================================================
# Main Builder Functions
# =============================================================================


def build_stage2_prompt_string(
    comment: str,
    category: CondensedCategory,
    examples: List[ClassificationExample],
    rules: Optional[List[str]] = None,
) -> str:
    """
    Build a complete Stage 2 prompt for a specific comment and category.

    Args:
        comment: The conference feedback comment to analyze
        category: CondensedCategory with element descriptions
        examples: List of ClassificationExample for few-shot learning
        rules: Optional custom base rules

    Returns:
        Complete prompt string ready for LLM

    Example:
        >>> prompt = build_stage2_prompt_string(
        ...     "The speaker was great!",
        ...     condensed.get_category("People"),
        ...     category_examples,
        ... )
    """
    elements_section = format_elements_section(category)
    rules_section = format_stage2_rules(category.name, rules)
    examples_section = format_stage2_examples_section(examples, category.name)

    return STAGE2_TEMPLATE.format(
        category_name=category.name,
        comment=comment,
        elements_section=elements_section,
        rules_section=rules_section,
        examples_section=examples_section,
    )


def build_stage2_prompt_function(
    category: CondensedCategory,
    examples: ExampleSet | List[ClassificationExample],
    rules: Optional[List[str]] = None,
) -> Callable[[str], str]:
    """
    Build a reusable Stage 2 prompt function for a single category.

    This creates a closure that "bakes in" the category's elements and examples,
    returning a simple function that takes a comment and returns a prompt.

    Args:
        category: CondensedCategory with element descriptions
        examples: ExampleSet or list of ClassificationExample
        rules: Optional custom base rules

    Returns:
        Function that takes a comment string and returns a complete prompt
    """
    # Filter examples for this category
    category_examples = get_stage2_examples_for_category(examples, category.name)

    # Pre-compute static parts
    elements_section = format_elements_section(category)
    rules_section = format_stage2_rules(category.name, rules)
    examples_section = format_stage2_examples_section(category_examples, category.name)

    # Build the static parts of the prompt (everything except the comment)
    # We use a placeholder that won't conflict with JSON braces
    static_parts = {
        "category_name": category.name,
        "elements_section": elements_section,
        "rules_section": rules_section,
        "examples_section": examples_section,
    }

    def prompt_function(comment: str) -> str:
        """Generate Stage 2 element extraction prompt for a comment."""
        return f"""You are an expert conference feedback analyzer. Extract specific feedback related to {static_parts["category_name"]} from this comment.

COMMENT TO ANALYZE:
{comment}

---

ELEMENTS TO IDENTIFY:

{static_parts["elements_section"]}

---

CLASSIFICATION RULES:

{static_parts["rules_section"]}

---

EXAMPLES:

{static_parts["examples_section"]}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to {static_parts["category_name"]}, return {{"classifications": []}}."""

    return prompt_function


def build_stage2_prompt_functions(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: Optional[List[str]] = None,
) -> Dict[str, Callable[[str], str]]:
    """
    Build Stage 2 prompt functions for ALL categories.

    This is the main entry point for Stage 2 prompt generation.

    Args:
        condensed: CondensedTaxonomy with all categories
        examples: ExampleSet or list of ClassificationExample
        rules: Optional custom base rules

    Returns:
        Dict mapping category name to prompt function

    Example:
        >>> stage2_prompts = build_stage2_prompt_functions(condensed, examples)
        >>> people_prompt = stage2_prompts["People"]
        >>> prompt = people_prompt("The speaker was brilliant!")
    """
    prompt_functions = {}

    for category in condensed.categories:
        prompt_functions[category.name] = build_stage2_prompt_function(
            category, examples, rules
        )

    return prompt_functions


# =============================================================================
# Export as Python Module
# =============================================================================


def export_stage2_prompt_module(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    filepath: str | Path,
    rules: Optional[List[str]] = None,
) -> Path:
    """
    Export Stage 2 prompts as a standalone Python module.

    Creates a .py file with one prompt function per category, plus a
    STAGE2_PROMPTS dict mapping category names to functions.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list of ClassificationExample
        filepath: Output path for the .py file
        rules: Optional custom base rules

    Returns:
        Path to the generated module

    Example:
        >>> export_stage2_prompt_module(condensed, examples, "prompts/stage2.py")
        >>> # Later, in production:
        >>> from prompts.stage2 import STAGE2_PROMPTS, stage2_people_prompt
        >>> prompt = stage2_people_prompt("Great speaker!")
        >>> # Or use the dict:
        >>> prompt = STAGE2_PROMPTS["People"]("Great speaker!")
    """
    from ..taxonomy.utils import sanitize_model_name

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Build all the prompt function code
    function_codes = []
    function_names = []

    for category in condensed.categories:
        # Get examples for this category
        category_examples = get_stage2_examples_for_category(examples, category.name)

        # Pre-compute sections
        elements_section = format_elements_section(category)
        rules_section = format_stage2_rules(category.name, rules)
        examples_section = format_stage2_examples_section(category_examples, category.name)

        # Create function name
        sanitized_name = sanitize_model_name(category.name).lower()
        func_name = f"stage2_{sanitized_name}_prompt"
        function_names.append((category.name, func_name))

        # Build the function code
        func_code = f'''
def {func_name}(comment: str) -> str:
    """
    Generate Stage 2 prompt for {category.name}.

    Args:
        comment: The conference feedback comment to analyze

    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Extract specific feedback related to {category.name} from this comment.

COMMENT TO ANALYZE:
{{comment}}

---

ELEMENTS TO IDENTIFY:

{elements_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to {category.name}, return {{{{"classifications": []}}}}."""
'''
        function_codes.append(func_code)

    # Build the STAGE2_PROMPTS dict
    dict_entries = ",\n    ".join(
        f'"{cat_name}": {func_name}' for cat_name, func_name in function_names
    )

    # Assemble the module
    module_content = f'''"""
Stage 2: Element Extraction Prompts

Auto-generated from condensed taxonomy and curated examples.

Usage:
    from {Path(filepath).stem} import STAGE2_PROMPTS, stage2_people_prompt

    # Use specific function
    prompt = stage2_people_prompt("The speaker was great!")

    # Or use the dict
    prompt = STAGE2_PROMPTS["People"]("The speaker was great!")

Available functions:
{chr(10).join(f"    - {func_name}" for _, func_name in function_names)}
"""

from typing import Callable, Dict

{"".join(function_codes)}

# =============================================================================
# Category to Prompt Function Mapping
# =============================================================================

STAGE2_PROMPTS: Dict[str, Callable[[str], str]] = {{
    {dict_entries}
}}


def get_stage2_prompt(category: str) -> Callable[[str], str]:
    """
    Get the Stage 2 prompt function for a category.

    Args:
        category: Category name

    Returns:
        Prompt function for that category

    Raises:
        KeyError: If category not found
    """
    if category not in STAGE2_PROMPTS:
        available = ", ".join(STAGE2_PROMPTS.keys())
        raise KeyError(f"Unknown category '{{category}}'. Available: {{available}}")
    return STAGE2_PROMPTS[category]
'''

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(module_content)

    return filepath


def export_stage2_prompt_modules(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    directory: str | Path,
    rules: Optional[List[str]] = None,
) -> List[Path]:
    """
    Export Stage 2 prompts as separate modules (one per category).

    Creates a directory with one .py file per category plus an __init__.py.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list of ClassificationExample
        directory: Output directory
        rules: Optional custom base rules

    Returns:
        List of paths to generated modules

    Example:
        >>> export_stage2_prompt_modules(condensed, examples, "prompts/stage2/")
        >>> # Creates:
        >>> #   prompts/stage2/__init__.py
        >>> #   prompts/stage2/people.py
        >>> #   prompts/stage2/learning_content_delivery.py
        >>> #   ...
    """
    from ..taxonomy.utils import sanitize_model_name

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    created_paths = []
    module_imports = []

    for category in condensed.categories:
        # Get examples for this category
        category_examples = get_stage2_examples_for_category(examples, category.name)

        # Pre-compute sections
        elements_section = format_elements_section(category)
        rules_section = format_stage2_rules(category.name, rules)
        examples_section = format_stage2_examples_section(category_examples, category.name)

        # Create filename
        sanitized_name = sanitize_model_name(category.name).lower()
        filename = f"{sanitized_name}.py"
        func_name = f"stage2_{sanitized_name}_prompt"

        module_content = f'''"""
Stage 2 Prompt: {category.name}

Auto-generated element extraction prompt.

Usage:
    from {sanitized_name} import {func_name}

    prompt = {func_name}("Your comment here")
"""


def {func_name}(comment: str) -> str:
    """
    Generate Stage 2 prompt for {category.name}.

    Args:
        comment: The conference feedback comment to analyze

    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Extract specific feedback related to {category.name} from this comment.

COMMENT TO ANALYZE:
{{comment}}

---

ELEMENTS TO IDENTIFY:

{elements_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to {category.name}, return {{{{"classifications": []}}}}."""


# Convenience alias
CATEGORY_NAME = "{category.name}"
'''

        filepath = directory / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(module_content)

        created_paths.append(filepath)
        module_imports.append((sanitized_name, func_name, category.name))

    # Create __init__.py
    init_imports = "\n".join(f"from .{mod} import {func}" for mod, func, _ in module_imports)
    dict_entries = ",\n    ".join(f'"{cat}": {func}' for mod, func, cat in module_imports)

    init_content = f'''"""
Stage 2 Element Extraction Prompts

Auto-generated prompt modules for each category.

Usage:
    from stage2 import STAGE2_PROMPTS
    from stage2 import stage2_people_prompt

    # Use the dict
    prompt = STAGE2_PROMPTS["People"]("Great speaker!")

    # Or use directly
    prompt = stage2_people_prompt("Great speaker!")
"""

from typing import Callable, Dict

{init_imports}

STAGE2_PROMPTS: Dict[str, Callable[[str], str]] = {{
    {dict_entries}
}}

__all__ = [
    "STAGE2_PROMPTS",
{chr(10).join(f'    "{func}",' for _, func, _ in module_imports)}
]
'''

    init_path = directory / "__init__.py"
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(init_content)

    created_paths.append(init_path)

    return created_paths


# =============================================================================
# Prompt Statistics
# =============================================================================


def get_stage2_prompt_stats(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Get statistics about Stage 2 prompts for each category.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        rules: Optional custom rules

    Returns:
        Dict mapping category name to stats dict
    """
    stats = {}

    for category in condensed.categories:
        category_examples = get_stage2_examples_for_category(examples, category.name)

        # Build a sample prompt
        sample_prompt = build_stage2_prompt_string(
            "Sample comment for length estimation.",
            category,
            category_examples,
            rules,
        )

        stats[category.name] = {
            "total_chars": len(sample_prompt),
            "estimated_tokens": len(sample_prompt) // 4,
            "num_elements": len(category.elements),
            "num_examples": len(category_examples),
            "num_rules": len(rules or DEFAULT_STAGE2_RULES)
            + len(CATEGORY_SPECIFIC_RULES.get(category.name, [])),
        }

    return stats


def print_stage2_prompts_preview(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    sample_comment: str = "The speaker was great and very knowledgeable.",
    show_full_prompt: Optional[str] = None,
) -> None:
    """
    Print a preview of Stage 2 prompts.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        sample_comment: Comment to use in preview
        show_full_prompt: If provided, show full prompt for this category
    """
    stats = get_stage2_prompt_stats(condensed, examples)

    print("\n" + "=" * 70)
    print("STAGE 2 PROMPTS PREVIEW")
    print("=" * 70)

    for category in condensed.categories:
        cat_stats = stats[category.name]
        print(f"\n**{category.name}**")
        print(f"  Elements: {cat_stats['num_elements']}")
        print(f"  Examples: {cat_stats['num_examples']}")
        print(f"  Rules: {cat_stats['num_rules']}")
        print(
            f"  Prompt size: {cat_stats['total_chars']:,} chars (~{cat_stats['estimated_tokens']:,} tokens)"
        )

    if show_full_prompt:
        category = condensed.get_category(show_full_prompt)
        if category:
            category_examples = get_stage2_examples_for_category(examples, show_full_prompt)
            prompt = build_stage2_prompt_string(sample_comment, category, category_examples)

            print("\n" + "-" * 70)
            print(f"FULL PROMPT FOR: {show_full_prompt}")
            print("-" * 70)
            print(prompt)
