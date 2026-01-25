"""
Rule Generator

Generates classification rules for Stage 1 and Stage 2 prompts from a condensed taxonomy.

Rules fall into two categories:
    1. Base rules: Universal rules that apply to all classifications
    2. Category-specific rules: Disambiguation rules for specific categories

This module supports two modes:
    1. GENERATE mode: Use an LLM to generate rules (requires processor)
    2. LOAD mode: Load previously generated rules from JSON

Usage:
    # Generate mode (with LLM)
    from classifier.taxonomy.rule_generator import generate_all_rules, save_rules

    rules = generate_all_rules(condensed, processor)
    save_rules(rules, "artifacts/rules.json")

    # Load mode (no LLM needed)
    from classifier.taxonomy.rule_generator import load_rules

    rules = load_rules("artifacts/rules.json")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .condenser import CondensedTaxonomy, ProcessorProtocol

# =============================================================================
# Pydantic Schemas for Rules
# =============================================================================


class CategoryRules(BaseModel):
    """Rules specific to a single category."""

    category: str = Field(description="Category name")
    rules: List[str] = Field(description="List of disambiguation rules for this category")
    element_guidance: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional per-element guidance (element_name -> guidance)",
    )


class ClassificationRules(BaseModel):
    """Complete set of classification rules."""

    # Stage 1 rules
    stage1_base_rules: List[str] = Field(
        description="Base rules for Stage 1 category detection"
    )

    # Stage 2 rules
    stage2_base_rules: List[str] = Field(
        description="Base rules for Stage 2 element extraction"
    )
    stage2_category_rules: List[CategoryRules] = Field(
        description="Category-specific rules for Stage 2"
    )

    # Metadata
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional metadata (generation date, source, etc.)",
    )

    def get_stage1_rules(self) -> List[str]:
        """Get all Stage 1 rules."""
        return self.stage1_base_rules

    def get_stage2_base_rules(self) -> List[str]:
        """Get Stage 2 base rules."""
        return self.stage2_base_rules

    def get_stage2_category_rules(self, category: str) -> List[str]:
        """Get category-specific rules for Stage 2."""
        for cat_rules in self.stage2_category_rules:
            if cat_rules.category == category:
                return cat_rules.rules
        return []

    def get_all_stage2_rules(self, category: str) -> List[str]:
        """Get combined base + category rules for Stage 2."""
        return self.stage2_base_rules + self.get_stage2_category_rules(category)


# =============================================================================
# Default Base Rules
# =============================================================================

DEFAULT_STAGE1_BASE_RULES = [
    "A comment can belong to MULTIPLE categories if it discusses multiple aspects.",
    "Focus on what the comment is ABOUT, not just words mentioned.",
    'General praise like "great conference" without specifics → Event Operations & Management > Conference.',
    "If a comment mentions both the content AND the presenter, include BOTH categories.",
]

DEFAULT_STAGE2_BASE_RULES = [
    "Extract the EXACT excerpt from the comment that relates to each element.",
    "Each excerpt should be classified to ONE element only.",
    "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
    "If multiple distinct excerpts relate to the same element, create separate entries.",
    "Only extract elements that are clearly present - do not infer or assume.",
]


# =============================================================================
# LLM Response Schema
# =============================================================================


class _CategoryRulesResponse(BaseModel):
    """LLM response for generating category-specific rules."""

    category: str
    disambiguation_rules: List[str] = Field(
        description="Rules to help distinguish between similar elements"
    )
    common_mistakes: List[str] = Field(description="Common classification mistakes to avoid")


# =============================================================================
# Prompt Builders
# =============================================================================


def _build_category_rules_prompt(category_name: str, elements: List[dict]) -> str:
    """
    Build prompt to generate disambiguation rules for a category.

    Args:
        category_name: Name of the category
        elements: List of element dicts with 'name' and 'short_description'
    """
    elements_text = "\n".join(f"- **{e['name']}**: {e['short_description']}" for e in elements)

    element_names = [e["name"] for e in elements]

    return f"""You are an expert at creating classification guidelines. Your task is to generate disambiguation rules for a conference feedback classification system.

CONTEXT: Annotators and LLMs will use these rules to correctly classify feedback excerpts into elements. Good rules help distinguish between SIMILAR elements that could be confused.

---

CATEGORY: {category_name}

ELEMENTS IN THIS CATEGORY:
{elements_text}

---

YOUR TASK:

Generate rules that help classifiers distinguish between these elements. Focus on:

1. **Disambiguation rules**: How to tell similar elements apart
   - What makes Element A different from Element B?
   - What keywords or phrases indicate each element?
   - What's the key distinguishing feature?

2. **Common mistakes to avoid**: Pitfalls classifiers might fall into
   - Elements that are often confused
   - Subtle distinctions that matter

---

EXAMPLES OF GOOD DISAMBIGUATION RULES:

For "Attendee Engagement & Interaction":
- "Community" = feeling of belonging, being welcomed; "Networking" = act of meeting new people, making connections
- "Knowledge Exchange" = peer-to-peer learning among attendees; different from formal "Presentations" by speakers

For "People":
- "Explorance team" when mentioned as hosts/organizers → Conference Staff
- "Explorance experts" when mentioned for their knowledge/consulting → Experts/Consultants
- Named speakers or "the presenter" → Speakers/Presenters
- "Blue users" or "fellow attendees" → Participants/Attendees

For "Event Logistics & Infrastructure":
- "Conference Application/Software" = apps/platforms for attendees to use; "Technological Tools" = A/V equipment for presentations
- "Conference Venue" = physical space, rooms, temperature; "Hotel" = accommodation, lodging

---

GUIDELINES:

1. Write rules in the format: "X" = description; "Y" = description (for comparisons)
2. Or use arrow notation: keyword/phrase → Element Name
3. Be specific and actionable - a classifier should be able to apply the rule directly
4. Focus on elements that could genuinely be confused
5. 3-6 rules is typically sufficient
6. Don't just repeat the element descriptions - add NEW disambiguation insight

---

Generate disambiguation rules and common mistakes for the **{category_name}** category with elements: {", ".join(element_names)}.

Return as JSON."""


# =============================================================================
# Generation Functions
# =============================================================================


def generate_category_rules(
    condensed: CondensedTaxonomy,
    processor: ProcessorProtocol,
    guided_config: Optional[dict] = None,
    verbose: bool = True,
) -> List[CategoryRules]:
    """
    Generate category-specific disambiguation rules.

    Args:
        condensed: CondensedTaxonomy to generate rules for
        processor: LLM processor
        guided_config: Optional sampling parameters
        verbose: Print progress

    Returns:
        List of CategoryRules (one per category)
    """
    config = guided_config or {
        "temperature": 0.7,
        "max_tokens": 1500,
    }

    if verbose:
        print(f"Generating rules for {len(condensed.categories)} categories...")

    # Build prompts for all categories
    prompts = []
    category_names = []

    for cat in condensed.categories:
        elements = [
            {"name": e.name, "short_description": e.short_description} for e in cat.elements
        ]
        prompts.append(_build_category_rules_prompt(cat.name, elements))
        category_names.append(cat.name)

        if verbose:
            print(f"  • {cat.name} ({len(cat.elements)} elements)")

    # Process all categories
    if verbose:
        print("\nRunning LLM generation...")

    responses = processor.process_with_schema(
        prompts=prompts,
        schema=_CategoryRulesResponse,
        batch_size=len(prompts),
        guided_config=config,
    )

    results = processor.parse_results_with_schema(
        schema=_CategoryRulesResponse,
        responses=responses,
        validate=True,
    )

    # Convert to CategoryRules
    category_rules = []
    for cat_name, result in zip(category_names, results):
        if result is None:
            if verbose:
                print(f"  ⚠ Failed to generate rules for {cat_name}, using empty")
            category_rules.append(CategoryRules(category=cat_name, rules=[]))
            continue

        # Combine disambiguation rules and common mistakes
        all_rules = result.disambiguation_rules + [
            f"Avoid: {mistake}" for mistake in result.common_mistakes
        ]

        category_rules.append(CategoryRules(category=cat_name, rules=all_rules))

    if verbose:
        total_rules = sum(len(cr.rules) for cr in category_rules)
        print(f"✓ Generated {total_rules} rules across {len(category_rules)} categories")

    return category_rules


def generate_all_rules(
    condensed: CondensedTaxonomy,
    processor: ProcessorProtocol,
    stage1_base_rules: Optional[List[str]] = None,
    stage2_base_rules: Optional[List[str]] = None,
    guided_config: Optional[dict] = None,
    verbose: bool = True,
) -> ClassificationRules:
    """
    Generate complete classification rules.

    Args:
        condensed: CondensedTaxonomy
        processor: LLM processor
        stage1_base_rules: Custom Stage 1 base rules (uses defaults if not provided)
        stage2_base_rules: Custom Stage 2 base rules (uses defaults if not provided)
        guided_config: Optional sampling parameters
        verbose: Print progress

    Returns:
        Complete ClassificationRules
    """
    # Use defaults if not provided
    s1_rules = stage1_base_rules or DEFAULT_STAGE1_BASE_RULES
    s2_rules = stage2_base_rules or DEFAULT_STAGE2_BASE_RULES

    # Generate category-specific rules
    category_rules = generate_category_rules(condensed, processor, guided_config, verbose)

    return ClassificationRules(
        stage1_base_rules=s1_rules,
        stage2_base_rules=s2_rules,
        stage2_category_rules=category_rules,
        metadata={
            "generated_from": "condensed_taxonomy",
            "num_categories": len(condensed.categories),
        },
    )


# =============================================================================
# Save / Load Functions
# =============================================================================


def save_rules(
    rules: ClassificationRules,
    filepath: str | Path,
    indent: int = 2,
) -> Path:
    """
    Save rules to JSON file.

    Args:
        rules: ClassificationRules to save
        filepath: Output path
        indent: JSON indentation

    Returns:
        Path to saved file
    """
    import json

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(rules.model_dump(), f, indent=indent, ensure_ascii=False)

    return filepath


def load_rules(filepath: str | Path) -> ClassificationRules:
    """
    Load rules from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        ClassificationRules instance

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    import json

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Rules file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ClassificationRules.model_validate(data)


def create_default_rules(condensed: CondensedTaxonomy) -> ClassificationRules:
    """
    Create rules with defaults (no LLM).

    Useful for quick testing or when LLM-generated rules aren't needed.

    Args:
        condensed: CondensedTaxonomy for category names

    Returns:
        ClassificationRules with default base rules and empty category rules
    """
    category_rules = [
        CategoryRules(category=cat.name, rules=[]) for cat in condensed.categories
    ]

    return ClassificationRules(
        stage1_base_rules=DEFAULT_STAGE1_BASE_RULES,
        stage2_base_rules=DEFAULT_STAGE2_BASE_RULES,
        stage2_category_rules=category_rules,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def print_rules_preview(rules: ClassificationRules) -> None:
    """Print a preview of the rules."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION RULES PREVIEW")
    print("=" * 70)

    print(f"\nStage 1 Base Rules ({len(rules.stage1_base_rules)}):")
    for i, rule in enumerate(rules.stage1_base_rules, 1):
        print(f"  {i}. {rule}")

    print(f"\nStage 2 Base Rules ({len(rules.stage2_base_rules)}):")
    for i, rule in enumerate(rules.stage2_base_rules, 1):
        print(f"  {i}. {rule}")

    print("\nStage 2 Category-Specific Rules:")
    for cat_rules in rules.stage2_category_rules:
        print(f"\n  **{cat_rules.category}** ({len(cat_rules.rules)} rules)")
        for rule in cat_rules.rules[:3]:  # Show first 3
            print(f"    • {rule}")
        if len(cat_rules.rules) > 3:
            print(f"    ... and {len(cat_rules.rules) - 3} more")


def merge_rules(
    base: ClassificationRules,
    overrides: ClassificationRules,
) -> ClassificationRules:
    """
    Merge two rule sets, with overrides taking precedence.

    Useful for combining auto-generated rules with manual additions.

    Args:
        base: Base rules
        overrides: Rules that override/extend base

    Returns:
        Merged ClassificationRules
    """
    # Merge category rules
    merged_category_rules = {cr.category: cr for cr in base.stage2_category_rules}

    for override_cr in overrides.stage2_category_rules:
        if override_cr.category in merged_category_rules:
            # Extend existing rules
            existing = merged_category_rules[override_cr.category]
            merged_rules = list(set(existing.rules + override_cr.rules))
            merged_category_rules[override_cr.category] = CategoryRules(
                category=override_cr.category,
                rules=merged_rules,
            )
        else:
            # Add new category
            merged_category_rules[override_cr.category] = override_cr

    return ClassificationRules(
        stage1_base_rules=overrides.stage1_base_rules or base.stage1_base_rules,
        stage2_base_rules=overrides.stage2_base_rules or base.stage2_base_rules,
        stage2_category_rules=list(merged_category_rules.values()),
    )
