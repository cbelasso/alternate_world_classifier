"""
Example Generator

Generates training examples for classification prompts from a condensed taxonomy.

Supports three types of examples:
    1. Simple: Single-category examples (one per category)
    2. Complex: Multi-category examples
    3. Negative: Non-classifiable examples

This module supports two modes:
    1. GENERATE mode: Use an LLM to generate examples (requires processor)
    2. LOAD mode: Load previously generated examples from JSON

Usage:
    # Generate mode (with LLM)
    from classifier.taxonomy.example_generator import (
        generate_simple_examples,
        generate_complex_examples,
        combine_examples,
        save_examples,
    )

    simple = generate_simple_examples(condensed, processor)
    complex_ex = generate_complex_examples(condensed, processor)
    all_examples = combine_examples(simple, complex_ex)
    save_examples(all_examples, "artifacts/examples.json")

    # Load mode (no LLM needed)
    from classifier.taxonomy.example_generator import load_examples

    examples = load_examples("artifacts/examples.json")
"""

from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .condenser import CondensedTaxonomy, ProcessorProtocol

# =============================================================================
# Pydantic Schemas for Examples
# =============================================================================


class ElementDetail(BaseModel):
    """Stage 2 element extraction detail within an example."""

    category: Optional[str] = Field(
        default=None,
        description="Parent category name (required for multi-category examples)",
    )
    element: str = Field(description="Element name - MUST match exact taxonomy name")
    excerpt: str = Field(description="Exact text excerpt from comment relating to this element")
    sentiment: str = Field(description="Sentiment: positive, negative, neutral, or mixed")
    reasoning: str = Field(
        description="Brief explanation of why this excerpt maps to this element"
    )


class ClassificationExample(BaseModel):
    """A dual-purpose example for both Stage 1 and Stage 2 classification."""

    comment: str = Field(description="Realistic conference feedback comment")

    # Stage 1: Category detection
    categories_present: List[str] = Field(description="List of top-level category names")
    has_classifiable_content: bool = Field(
        description="Whether this contains classifiable conference feedback"
    )
    stage1_reasoning: str = Field(description="Brief explanation of why these categories apply")

    # Stage 2: Element extraction
    element_details: List[ElementDetail] = Field(
        default_factory=list,
        description="Detailed element-level extractions for Stage 2",
    )

    # Metadata
    example_type: Optional[str] = Field(
        default=None,
        description="Type: 'simple', 'multi_category', or 'negative'",
    )
    source_category: Optional[str] = Field(
        default=None,
        description="For simple examples, the category this was generated for",
    )


class ExampleSet(BaseModel):
    """Complete set of examples for prompt generation."""

    examples: List[ClassificationExample] = Field(description="All curated examples")
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional metadata about generation",
    )

    def get_by_type(self, example_type: str) -> List[ClassificationExample]:
        """Get examples of a specific type."""
        return [ex for ex in self.examples if ex.example_type == example_type]

    def get_simple_examples(self) -> List[ClassificationExample]:
        """Get single-category examples."""
        return self.get_by_type("simple")

    def get_multi_category_examples(self) -> List[ClassificationExample]:
        """Get multi-category examples."""
        return self.get_by_type("multi_category")

    def get_negative_examples(self) -> List[ClassificationExample]:
        """Get negative (non-classifiable) examples."""
        return self.get_by_type("negative")

    def get_stats(self) -> dict:
        """Get statistics about the example set."""
        return {
            "total": len(self.examples),
            "simple": len(self.get_simple_examples()),
            "multi_category": len(self.get_multi_category_examples()),
            "negative": len(self.get_negative_examples()),
        }


# =============================================================================
# Internal Schemas for LLM Responses
# =============================================================================


class _CategoryExamplesResponse(BaseModel):
    """LLM response schema for simple category examples."""

    category_name: str
    examples: List[ClassificationExample]


class _ComplexExamplesResponse(BaseModel):
    """LLM response schema for complex examples."""

    multi_category_examples: List[ClassificationExample]
    negative_examples: List[ClassificationExample]


# =============================================================================
# Prompt Builders
# =============================================================================


def _build_simple_example_prompt(category: dict, all_element_names: List[str]) -> str:
    """
    Build prompt to generate simple examples for a single category.

    Args:
        category: Category dict with name, short_description, and elements
        all_element_names: List of exact element names for validation emphasis
    """
    cat_name = category["name"]
    cat_desc = category["short_description"]

    # Format elements
    elements_text = "\n".join(
        f"- {elem['name']}: {elem['short_description']}"
        for elem in category.get("elements", [])
    )

    element_names = [elem["name"] for elem in category.get("elements", [])]
    element_names_str = ", ".join(f'"{name}"' for name in element_names)

    return f"""You are an expert at creating training examples for conference feedback classification systems.

TASK: Generate 1-2 realistic conference attendee comments for the category below, with BOTH Stage 1 (category detection) and Stage 2 (element extraction) labels.

---

CATEGORY: {cat_name}

DESCRIPTION: {cat_desc}

VALID ELEMENTS (use these EXACT names - do NOT paraphrase):
{elements_text}

**CRITICAL: When specifying elements in element_details, you MUST use these exact element names:**
{element_names_str}

Do NOT create variations like "Event Schedule", "Panels/Discussions", "Venue/Facilities", etc.
Use the EXACT names listed above.

---

GUIDELINES FOR CREATING DUAL-PURPOSE EXAMPLES:

1. **Write realistic conference feedback** (1-2 sentences)
   - Should sound like something an actual attendee would write
   - Can be positive, negative, or neutral
   - Should clearly relate to this category's scope

2. **Stage 1 labels** (Category detection)
   - categories_present: ["{cat_name}"] (just the top-level category name)
   - stage1_reasoning: Why this comment belongs to this category

3. **Stage 2 labels** (Element extraction)
   - For each element mentioned in the comment, provide:
     - element: The EXACT element name from the list above
     - excerpt: The EXACT text from the comment that relates to this element
     - sentiment: positive, negative, neutral, or mixed
     - reasoning: Why this excerpt maps to this element

4. **Vary the elements**
   - Example 1: Focus on one specific element
   - Example 2: Focus on a different element (can include multiple elements if natural)

---

EXAMPLE FORMAT:

{{
  "comment": "The networking sessions were fantastic and I made great connections with peers from other institutions.",
  "categories_present": ["Attendee Engagement & Interaction"],
  "has_classifiable_content": true,
  "stage1_reasoning": "Discusses networking and peer connections",
  "element_details": [
    {{
      "element": "Networking",
      "excerpt": "The networking sessions were fantastic and I made great connections with peers",
      "sentiment": "positive",
      "reasoning": "Praises networking sessions and making professional connections"
    }}
  ],
  "example_type": "simple",
  "source_category": "Attendee Engagement & Interaction"
}}

---

Generate 1-2 dual-purpose examples for the **{cat_name}** category. Return as JSON.

REMEMBER: Use ONLY the exact element names listed above."""


def _build_complex_example_prompt(condensed: CondensedTaxonomy) -> str:
    """
    Build prompt to generate multi-category and negative examples.
    """
    # Format all categories concisely
    categories_text = []
    elements_by_category = []

    for cat in condensed.categories:
        categories_text.append(f"**{cat.name}**: {cat.short_description}")

        elem_names = [f'"{e.name}"' for e in cat.elements]
        elements_by_category.append(f"**{cat.name}**: {', '.join(elem_names)}")

    categories_summary = "\n".join(categories_text)
    elements_reference = "\n".join(elements_by_category)

    return f"""You are an expert at creating training examples for conference feedback classification systems.

TASK: Generate complex and edge-case examples with BOTH Stage 1 (category detection) and Stage 2 (element extraction) labels.

---

AVAILABLE CATEGORIES:

{categories_summary}

---

VALID ELEMENT NAMES BY CATEGORY (use EXACT names - do NOT paraphrase):

{elements_reference}

**CRITICAL**: When specifying elements in element_details, you MUST:
1. Include the "category" field for each element_detail
2. Use the EXACT element names listed above

---

PART 1: MULTI-CATEGORY EXAMPLES

Generate 3-5 realistic conference feedback comments that span MULTIPLE categories.

These should:
- Sound like real attendee feedback (1-3 sentences)
- Naturally discuss 2-3 different aspects of the conference
- Demonstrate how categories can co-occur
- Vary in sentiment (positive, negative, mixed)

For each, set example_type to "multi_category".

---

PART 2: NEGATIVE EXAMPLES

Generate 2-3 comments that look like feedback but are NOT about the conference.

These should:
- Sound similar to feedback (opinions, observations)
- But are NOT about conference experience
- Return empty categories: []
- Return empty element_details: []

For each, set example_type to "negative".

---

EXAMPLE OUTPUT FORMAT:

Multi-category example:
{{
  "comment": "The keynote speaker was brilliant and the presentation on ML was very insightful.",
  "categories_present": ["Learning & Content Delivery", "People"],
  "has_classifiable_content": true,
  "stage1_reasoning": "Discusses both presentation content and the speaker",
  "element_details": [
    {{
      "category": "Learning & Content Delivery",
      "element": "Presentations",
      "excerpt": "the presentation on ML was very insightful",
      "sentiment": "positive",
      "reasoning": "Praises the presentation content"
    }},
    {{
      "category": "People",
      "element": "Speakers/Presenters",
      "excerpt": "The keynote speaker was brilliant",
      "sentiment": "positive",
      "reasoning": "Praises the keynote speaker"
    }}
  ],
  "example_type": "multi_category"
}}

Negative example:
{{
  "comment": "The only constant in life is change.",
  "categories_present": [],
  "has_classifiable_content": false,
  "stage1_reasoning": "Philosophical statement, not conference feedback",
  "element_details": [],
  "example_type": "negative"
}}

---

Generate the examples and return as JSON with multi_category_examples and negative_examples arrays."""


# =============================================================================
# Generation Functions
# =============================================================================


def generate_simple_examples(
    condensed: CondensedTaxonomy,
    processor: ProcessorProtocol,
    guided_config: Optional[dict] = None,
    verbose: bool = True,
) -> List[ClassificationExample]:
    """
    Generate simple single-category examples for each category.

    Args:
        condensed: CondensedTaxonomy to generate examples for
        processor: LLM processor
        guided_config: Optional sampling parameters
        verbose: Print progress

    Returns:
        List of ClassificationExample (1-2 per category)
    """
    config = guided_config or {
        "temperature": 0.8,
        "max_tokens": 1500,
    }

    if verbose:
        print(f"Generating simple examples for {len(condensed.categories)} categories...")

    # Build prompts
    prompts = []
    category_names = []

    # Collect all element names for reference
    all_elements = []
    for cat in condensed.categories:
        all_elements.extend(e.name for e in cat.elements)

    for cat in condensed.categories:
        cat_dict = {
            "name": cat.name,
            "short_description": cat.short_description,
            "elements": [
                {"name": e.name, "short_description": e.short_description} for e in cat.elements
            ],
        }
        prompts.append(_build_simple_example_prompt(cat_dict, all_elements))
        category_names.append(cat.name)

        if verbose:
            print(f"  • {cat.name}")

    # Process
    if verbose:
        print("\nRunning LLM generation...")

    responses = processor.process_with_schema(
        prompts=prompts,
        schema=_CategoryExamplesResponse,
        batch_size=len(prompts),
        guided_config=config,
    )

    results = processor.parse_results_with_schema(
        schema=_CategoryExamplesResponse,
        responses=responses,
        validate=True,
    )

    # Collect all examples
    all_examples = []
    for cat_name, result in zip(category_names, results):
        if result is None:
            if verbose:
                print(f"  ⚠ Failed to generate examples for {cat_name}")
            continue

        for ex in result.examples:
            # Ensure metadata is set
            ex.example_type = "simple"
            ex.source_category = cat_name
            all_examples.append(ex)

    if verbose:
        print(f"✓ Generated {len(all_examples)} simple examples")

    return all_examples


def generate_complex_examples(
    condensed: CondensedTaxonomy,
    processor: ProcessorProtocol,
    guided_config: Optional[dict] = None,
    verbose: bool = True,
) -> tuple[List[ClassificationExample], List[ClassificationExample]]:
    """
    Generate multi-category and negative examples.

    Args:
        condensed: CondensedTaxonomy to generate examples for
        processor: LLM processor
        guided_config: Optional sampling parameters
        verbose: Print progress

    Returns:
        Tuple of (multi_category_examples, negative_examples)
    """
    config = guided_config or {
        "temperature": 0.85,
        "max_tokens": 2500,
    }

    if verbose:
        print("Generating complex and negative examples...")

    prompt = _build_complex_example_prompt(condensed)

    responses = processor.process_with_schema(
        prompts=[prompt],
        schema=_ComplexExamplesResponse,
        batch_size=1,
        guided_config=config,
    )

    results = processor.parse_results_with_schema(
        schema=_ComplexExamplesResponse,
        responses=responses,
        validate=True,
    )

    if not results or results[0] is None:
        raise ValueError("Failed to generate complex examples")

    result = results[0]

    # Ensure example types are set
    for ex in result.multi_category_examples:
        ex.example_type = "multi_category"
    for ex in result.negative_examples:
        ex.example_type = "negative"

    if verbose:
        print(f"✓ Generated {len(result.multi_category_examples)} multi-category examples")
        print(f"✓ Generated {len(result.negative_examples)} negative examples")

    return result.multi_category_examples, result.negative_examples


def generate_all_examples(
    condensed: CondensedTaxonomy,
    processor: ProcessorProtocol,
    guided_config: Optional[dict] = None,
    verbose: bool = True,
) -> ExampleSet:
    """
    Generate all example types in one call.

    Args:
        condensed: CondensedTaxonomy
        processor: LLM processor
        guided_config: Optional sampling parameters
        verbose: Print progress

    Returns:
        ExampleSet with all examples
    """
    simple = generate_simple_examples(condensed, processor, guided_config, verbose)
    multi, negative = generate_complex_examples(condensed, processor, guided_config, verbose)

    all_examples = simple + multi + negative

    return ExampleSet(
        examples=all_examples,
        metadata={
            "simple_count": len(simple),
            "multi_category_count": len(multi),
            "negative_count": len(negative),
        },
    )


# =============================================================================
# Combination and Curation
# =============================================================================


def combine_examples(
    simple: List[ClassificationExample],
    multi_category: List[ClassificationExample],
    negative: List[ClassificationExample],
    select_one_per_category: bool = True,
) -> ExampleSet:
    """
    Combine examples from different sources into a curated set.

    Args:
        simple: Simple single-category examples
        multi_category: Multi-category examples
        negative: Negative examples
        select_one_per_category: If True, select only one simple example per category

    Returns:
        ExampleSet with combined examples
    """
    selected_simple = simple

    if select_one_per_category:
        # Select one per category
        categories_seen = set()
        selected_simple = []
        for ex in simple:
            cat = ex.source_category
            if cat and cat not in categories_seen:
                categories_seen.add(cat)
                selected_simple.append(ex)

    all_examples = selected_simple + multi_category + negative

    return ExampleSet(
        examples=all_examples,
        metadata={
            "simple_count": len(selected_simple),
            "multi_category_count": len(multi_category),
            "negative_count": len(negative),
            "one_per_category": select_one_per_category,
        },
    )


# =============================================================================
# Save / Load Functions
# =============================================================================


def save_examples(
    examples: ExampleSet,
    filepath: str | Path,
    indent: int = 2,
) -> Path:
    """
    Save examples to JSON file.

    Args:
        examples: ExampleSet to save
        filepath: Output path
        indent: JSON indentation

    Returns:
        Path to saved file
    """
    import json

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(examples.model_dump(), f, indent=indent, ensure_ascii=False)

    return filepath


def load_examples(filepath: str | Path) -> ExampleSet:
    """
    Load examples from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        ExampleSet instance

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    import json

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Examples file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ExampleSet.model_validate(data)


# =============================================================================
# Quality Checking
# =============================================================================


def validate_examples(
    examples: ExampleSet,
    condensed: CondensedTaxonomy,
) -> dict:
    """
    Validate examples against the taxonomy.

    Checks:
    - Category names match taxonomy
    - Element names match taxonomy
    - Multi-category examples have 2+ categories
    - Negative examples have no categories

    Args:
        examples: ExampleSet to validate
        condensed: CondensedTaxonomy to validate against

    Returns:
        Dict with validation results and warnings
    """
    valid_categories = set(condensed.get_category_names())
    valid_elements = condensed.get_all_elements()

    warnings = []
    errors = []

    for i, ex in enumerate(examples.examples):
        # Check categories
        for cat in ex.categories_present:
            if cat not in valid_categories:
                errors.append(f"Example {i}: Invalid category '{cat}'")

        # Check elements
        for detail in ex.element_details:
            cat = detail.category or ex.source_category
            if cat and cat in valid_elements:
                if detail.element not in valid_elements[cat]:
                    errors.append(
                        f"Example {i}: Invalid element '{detail.element}' for category '{cat}'"
                    )

        # Type-specific checks
        if ex.example_type == "multi_category":
            if len(ex.categories_present) < 2:
                warnings.append(
                    f"Example {i}: Multi-category example has only {len(ex.categories_present)} categories"
                )

        if ex.example_type == "negative":
            if ex.categories_present:
                errors.append(
                    f"Example {i}: Negative example has categories: {ex.categories_present}"
                )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_checked": len(examples.examples),
    }


def print_examples_preview(examples: ExampleSet) -> None:
    """
    Print a preview of the example set.
    """
    stats = examples.get_stats()

    print("\n" + "=" * 70)
    print("EXAMPLES PREVIEW")
    print("=" * 70)
    print(f"Total: {stats['total']}")
    print(f"  Simple: {stats['simple']}")
    print(f"  Multi-category: {stats['multi_category']}")
    print(f"  Negative: {stats['negative']}")

    print("\n--- Sample Examples ---")

    for ex_type in ["simple", "multi_category", "negative"]:
        type_examples = examples.get_by_type(ex_type)
        if type_examples:
            ex = type_examples[0]
            print(f"\n[{ex_type.upper()}]")
            print(f'  "{ex.comment}"')
            print(f"  → Categories: {ex.categories_present}")
