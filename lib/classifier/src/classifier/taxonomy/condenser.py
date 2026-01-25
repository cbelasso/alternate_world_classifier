"""
Taxonomy Condenser

Transforms verbose taxonomy definitions into concise, classification-friendly
descriptions suitable for use in classification prompts.

This module supports two modes:
    1. GENERATE mode: Use an LLM to condense definitions (requires processor)
    2. LOAD mode: Load previously condensed taxonomy from JSON

Usage:
    # Generate mode (with LLM)
    from classifier.taxonomy.condenser import condense_taxonomy, save_condensed

    condensed = condense_taxonomy(taxonomy, processor)
    save_condensed(condensed, "artifacts/condensed_taxonomy.json")

    # Load mode (no LLM needed)
    from classifier.taxonomy.condenser import load_condensed

    condensed = load_condensed("artifacts/condensed_taxonomy.json")
"""

from pathlib import Path
from typing import Any, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# =============================================================================
# Pydantic Schemas for Condensed Output
# =============================================================================


class CondensedAttribute(BaseModel):
    """Concise attribute description for use in classification prompts."""

    name: str = Field(description="Attribute name (unchanged from taxonomy)")
    short_description: str = Field(
        description="Concise description focusing on key distinguishing features"
    )


class CondensedElement(BaseModel):
    """Concise element description for use in classification prompts."""

    name: str = Field(description="Element name (unchanged from taxonomy)")
    short_description: str = Field(
        description="Concise description focusing on key distinguishing features and examples"
    )
    attributes: Optional[List[CondensedAttribute]] = Field(
        default=None, description="Optional list of attributes for this element (for Stage 3)"
    )


class CondensedCategory(BaseModel):
    """Condensed category with all its elements."""

    name: str = Field(description="Category name (unchanged from taxonomy)")
    short_description: str = Field(description="One sentence description of category scope")
    elements: List[CondensedElement] = Field(
        description="List of condensed element descriptions"
    )


class CondensedTaxonomy(BaseModel):
    """Complete condensed taxonomy ready for prompt generation."""

    categories: List[CondensedCategory] = Field(description="List of condensed categories")
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional metadata (source taxonomy path, generation date, etc.)",
    )

    def get_category(self, name: str) -> Optional[CondensedCategory]:
        """Get a category by name."""
        for cat in self.categories:
            if cat.name == name:
                return cat
        return None

    def get_category_names(self) -> List[str]:
        """Get list of all category names."""
        return [cat.name for cat in self.categories]

    def get_elements_for_category(self, category_name: str) -> List[str]:
        """Get element names for a specific category."""
        cat = self.get_category(category_name)
        if cat:
            return [elem.name for elem in cat.elements]
        return []

    def get_all_elements(self) -> dict[str, List[str]]:
        """Get all elements organized by category."""
        return {cat.name: [e.name for e in cat.elements] for cat in self.categories}


# =============================================================================
# Processor Protocol (for type hints without hard dependency)
# =============================================================================


@runtime_checkable
class ProcessorProtocol(Protocol):
    """Protocol for LLM processors that support schema-guided generation."""

    def process_with_schema(
        self,
        prompts: List[str],
        schema: type,
        batch_size: int,
        guided_config: dict,
    ) -> List[str]: ...

    def parse_results_with_schema(
        self,
        schema: type,
        responses: List[str],
        validate: bool,
    ) -> List[Any]: ...


# =============================================================================
# Condensation Prompt
# =============================================================================


def _build_condensation_prompt(category_data: dict) -> str:
    """
    Build prompt to condense verbose taxonomy definitions for a single category.

    Args:
        category_data: Category dict from the raw taxonomy with structure:
            {
                "name": "Category Name",
                "definition": "...",
                "description": "...",
                "children": [
                    {"name": "Element", "definition": "...", ...},
                    ...
                ]
            }

    Returns:
        Prompt string for the LLM
    """
    cat_name = category_data["name"]
    cat_verbose = category_data.get("definition", category_data.get("description", ""))

    # Extract elements with their verbose definitions
    elements_info = []
    for elem in category_data.get("children", []):
        elem_name = elem["name"]
        elem_verbose = elem.get("definition", elem.get("description", ""))
        elements_info.append(f"**{elem_name}**\n{elem_verbose}")

    elements_text = "\n\n".join(elements_info)

    return f"""You are a prompt engineering expert. Your task is to condense verbose taxonomy definitions into concise, classification-friendly descriptions.

CONTEXT: These descriptions will be used in a conference feedback classification system. An LLM will read attendee comments and determine which categories/elements apply. Your condensed descriptions must help the LLM make accurate classifications.

---

EXAMPLE OF HOW CONDENSED DESCRIPTIONS ARE USED:

Here's how descriptions appear in the classification prompt that processes user comments:

```
**Attendee Engagement & Interaction**
Feedback about connecting with others, community building, and social aspects.
- Community: Sense of belonging, community spirit, feeling welcomed
- Knowledge Exchange: Sharing experiences, learning from peers, collaborative learning
- Networking: Meeting new people, professional connections, peer discussions
- Social Events: Gala dinners, receptions, informal gatherings, social activities

**Event Logistics & Infrastructure**
Feedback about physical/technical infrastructure and venue-related services.
- Conference Application/Software: Mobile apps, event platforms, digital tools for attendees
- Conference Venue: Location, rooms, facilities, accessibility, seating
- Food/Beverages: Meals, snacks, drinks, catering quality, dietary options
```

Notice how descriptions:
- Include concrete examples (e.g., "gala dinners, receptions")
- Focus on distinguishing features (Community vs Networking)
- Use accessible language, not technical jargon
- Help classifiers understand WHEN to use each label

---

CATEGORY TO CONDENSE: {cat_name}

VERBOSE DEFINITION:
{cat_verbose}

---

ELEMENTS WITH VERBOSE DEFINITIONS:

{elements_text}

---

YOUR TASK:

Create concise descriptions that match the style of the examples above:

1. **Category short_description**: 
   - One clear sentence describing what type of feedback this category covers
   - Should help classifiers know when to select this category
   - Example: "Feedback about connecting with others, community building, and social aspects"

2. **Element short_descriptions** (for each element):
   - Be concise but CONCRETE - prioritize examples over generic descriptions
   - List specific things, activities, or aspects that define this element
   - Example: "Live demos, product showcases, hands-on examples"
   - NOT: "Overall impressions or availability of demonstrations"

CRITICAL GUIDELINES:

✓ DO:
- List concrete examples: "Gala dinners, receptions, informal gatherings"
- Use specific keywords: "Q&A sessions, audience participation, interactive discussions"
- Think: "What would appear in a comment about this element?"

✗ DON'T:
- Use vague meta-language: "Overall impressions", "General remarks", "Broad feedback"
- Use abstract descriptions: "Availability or impact"
- Over-compress at the expense of clarity
- Be too brief if it sacrifices concreteness

Remember: A slightly longer description with concrete examples is MUCH better than a short vague one. The goal is to help classifiers recognize this element in real comments.

Return JSON with the condensed category and all its elements."""


# =============================================================================
# Main Condensation Function
# =============================================================================


def condense_taxonomy(
    taxonomy: dict,
    processor: ProcessorProtocol,
    guided_config: Optional[dict] = None,
    verbose: bool = True,
) -> CondensedTaxonomy:
    """
    Condense a verbose taxonomy into classification-friendly descriptions.

    This function uses an LLM to transform verbose, detailed taxonomy definitions
    into concise descriptions suitable for classification prompts.

    Args:
        taxonomy: Raw taxonomy dict with structure:
            {
                "name": "Root",
                "children": [
                    {"name": "Category", "definition": "...", "children": [...]},
                    ...
                ]
            }
        processor: LLM processor with process_with_schema and parse_results_with_schema
        guided_config: Optional sampling parameters override
        verbose: Print progress information

    Returns:
        CondensedTaxonomy with all categories condensed

    Raises:
        ValueError: If any category fails to condense

    Example:
        >>> from llm_parallelization.new_processor import NewProcessor
        >>> processor = NewProcessor(gpu_list=[0], llm="mistral-nemo")
        >>> condensed = condense_taxonomy(taxonomy, processor)
        >>> save_condensed(condensed, "condensed.json")
    """
    config = guided_config or {
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    categories = taxonomy.get("children", [])

    if verbose:
        print(f"Condensing {len(categories)} categories...")

    # Build prompts for all categories
    prompts = []
    category_names = []

    for cat in categories:
        cat_name = cat["name"]
        category_names.append(cat_name)
        prompt = _build_condensation_prompt(cat)
        prompts.append(prompt)

        if verbose:
            num_elements = len(cat.get("children", []))
            print(f"  • {cat_name} ({num_elements} elements)")

    # Process all categories
    if verbose:
        print("\nRunning LLM condensation...")

    responses = processor.process_with_schema(
        prompts=prompts,
        schema=CondensedCategory,
        batch_size=len(prompts),
        guided_config=config,
    )

    results = processor.parse_results_with_schema(
        schema=CondensedCategory,
        responses=responses,
        validate=True,
    )

    # Check for failures
    failed = [
        (i, name)
        for i, (result, name) in enumerate(zip(results, category_names))
        if result is None
    ]

    if failed:
        failed_names = [name for _, name in failed]
        raise ValueError(f"Failed to condense categories: {failed_names}")

    if verbose:
        print(f"✓ Successfully condensed all {len(results)} categories")

    return CondensedTaxonomy(categories=results)


# =============================================================================
# Save / Load Functions
# =============================================================================


def save_condensed(
    condensed: CondensedTaxonomy,
    filepath: str | Path,
    indent: int = 2,
) -> Path:
    """
    Save condensed taxonomy to JSON file.

    Args:
        condensed: CondensedTaxonomy to save
        filepath: Output path
        indent: JSON indentation

    Returns:
        Path to saved file
    """
    import json

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(condensed.model_dump(), f, indent=indent, ensure_ascii=False)

    return filepath


def load_condensed(filepath: str | Path) -> CondensedTaxonomy:
    """
    Load condensed taxonomy from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        CondensedTaxonomy instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If JSON doesn't match schema
    """
    import json

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Condensed taxonomy not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return CondensedTaxonomy.model_validate(data)


# =============================================================================
# Quality Checking Utilities
# =============================================================================


VAGUE_WORDS = [
    "overall",
    "general",
    "broad",
    "various",
    "impact",
    "availability",
    "impressions",
    "remarks",
]


def check_condensation_quality(condensed: CondensedTaxonomy) -> dict:
    """
    Check the quality of condensed descriptions.

    Looks for vague language that might hurt classification accuracy.

    Args:
        condensed: CondensedTaxonomy to check

    Returns:
        Dict with quality metrics and warnings
    """
    warnings = []
    total_elements = 0
    vague_elements = 0

    for cat in condensed.categories:
        # Check category description
        if any(word in cat.short_description.lower() for word in VAGUE_WORDS):
            warnings.append(f"Category '{cat.name}' has vague language in description")

        # Check element descriptions
        for elem in cat.elements:
            total_elements += 1
            if any(word in elem.short_description.lower() for word in VAGUE_WORDS):
                vague_elements += 1
                warnings.append(f"Element '{elem.name}' in '{cat.name}' has vague language")

    return {
        "total_categories": len(condensed.categories),
        "total_elements": total_elements,
        "vague_elements": vague_elements,
        "vague_percentage": (
            100 * vague_elements / total_elements if total_elements > 0 else 0
        ),
        "warnings": warnings,
        "is_good_quality": vague_elements / total_elements < 0.1
        if total_elements > 0
        else True,
    }


def print_condensed_preview(condensed: CondensedTaxonomy) -> None:
    """
    Print a preview of the condensed taxonomy.

    Useful for reviewing condensation results.
    """
    print("\n" + "=" * 70)
    print("CONDENSED TAXONOMY PREVIEW")
    print("=" * 70)

    for cat in condensed.categories:
        print(f"\n**{cat.name}**")
        print(f"{cat.short_description}")
        for elem in cat.elements:
            if elem.attributes:
                print(f"  - {elem.name}: {elem.short_description}")
                for attr in elem.attributes[:3]:  # Show first 3
                    print(f"      • {attr.name}")
                if len(elem.attributes) > 3:
                    print(f"      ... and {len(elem.attributes) - 3} more attributes")
            else:
                print(f"  - {elem.name}: {elem.short_description}")

    # Quality check
    quality = check_condensation_quality(condensed)
    print("\n" + "-" * 70)
    print("QUALITY CHECK:")
    print(f"  Total elements: {quality['total_elements']}")
    print(f"  Vague elements: {quality['vague_elements']} ({quality['vague_percentage']:.1f}%)")

    # Attribute stats
    attr_stats = get_attribute_stats(condensed)
    if attr_stats["total_attributes"] > 0:
        print(f"  Elements with attributes: {attr_stats['elements_with_attributes']}")
        print(f"  Total attributes: {attr_stats['total_attributes']}")

    if quality["warnings"]:
        print("\n  Warnings:")
        for w in quality["warnings"][:5]:  # Show first 5
            print(f"    ⚠ {w}")
        if len(quality["warnings"]) > 5:
            print(f"    ... and {len(quality['warnings']) - 5} more")


# =============================================================================
# Attribute Enrichment (for Stage 3)
# =============================================================================


def enrich_with_attributes(
    condensed: CondensedTaxonomy,
    taxonomy: dict,
    max_description_length: int = 150,
    verbose: bool = True,
) -> CondensedTaxonomy:
    """
    Enrich a condensed taxonomy with attribute information from the raw taxonomy.

    This function extracts attribute definitions from the raw taxonomy and adds
    them to the corresponding elements in the condensed taxonomy. Attributes are
    used for Stage 3 classification.

    Args:
        condensed: CondensedTaxonomy to enrich
        taxonomy: Raw taxonomy dict with attribute children
        max_description_length: Maximum length for attribute descriptions
        verbose: Print progress

    Returns:
        New CondensedTaxonomy with attributes added to elements

    Example:
        >>> condensed = load_condensed("condensed.json")
        >>> with open("taxonomy.json") as f:
        ...     taxonomy = json.load(f)
        >>> enriched = enrich_with_attributes(condensed, taxonomy)
        >>> save_condensed(enriched, "condensed_with_attrs.json")
    """
    if verbose:
        print("\nEnriching condensed taxonomy with attributes...")

    total_attributes = 0
    enriched_categories = []

    for cat in condensed.categories:
        # Find matching category in taxonomy
        raw_cat = None
        for c in taxonomy.get("children", []):
            if c.get("name") == cat.name:
                raw_cat = c
                break

        if not raw_cat:
            # No match, keep as-is
            enriched_categories.append(cat)
            continue

        enriched_elements = []

        for elem in cat.elements:
            # Find matching element in taxonomy
            raw_elem = None
            for e in raw_cat.get("children", []):
                if e.get("name") == elem.name:
                    raw_elem = e
                    break

            if not raw_elem or not raw_elem.get("children"):
                # No attributes, keep as-is
                enriched_elements.append(elem)
                continue

            # Extract attributes
            attributes = []
            for attr in raw_elem.get("children", []):
                attr_name = attr.get("name", "")
                # Use description or definition, truncated if needed
                description = attr.get("description", attr.get("definition", ""))
                if len(description) > max_description_length:
                    description = description[: max_description_length - 3] + "..."

                attributes.append(
                    CondensedAttribute(
                        name=attr_name,
                        short_description=description,
                    )
                )
                total_attributes += 1

            # Create enriched element
            enriched_elements.append(
                CondensedElement(
                    name=elem.name,
                    short_description=elem.short_description,
                    attributes=attributes if attributes else None,
                )
            )

            if verbose and attributes:
                print(f"  • {cat.name} > {elem.name}: {len(attributes)} attributes")

        enriched_categories.append(
            CondensedCategory(
                name=cat.name,
                short_description=cat.short_description,
                elements=enriched_elements,
            )
        )

    if verbose:
        print(f"✓ Added {total_attributes} attributes")

    return CondensedTaxonomy(
        categories=enriched_categories,
        metadata=condensed.metadata,
    )


def get_attribute_stats(condensed: CondensedTaxonomy) -> dict:
    """
    Get statistics about attributes in a condensed taxonomy.

    Args:
        condensed: CondensedTaxonomy

    Returns:
        Dict with attribute statistics
    """
    total_elements = 0
    elements_with_attrs = 0
    total_attrs = 0

    for cat in condensed.categories:
        for elem in cat.elements:
            total_elements += 1
            if elem.attributes:
                elements_with_attrs += 1
                total_attrs += len(elem.attributes)

    return {
        "total_elements": total_elements,
        "elements_with_attributes": elements_with_attrs,
        "total_attributes": total_attrs,
        "avg_attributes_per_element": (
            total_attrs / elements_with_attrs if elements_with_attrs > 0 else 0
        ),
    }
