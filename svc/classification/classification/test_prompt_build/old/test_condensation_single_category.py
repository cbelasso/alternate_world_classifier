"""
Condensation Test - Single Category

Tests whether an LLM can effectively condense verbose taxonomy definitions
into concise, prompt-friendly descriptions.

Tests on ONE category at a time to validate approach before scaling.

Usage:
    python test_condensation_single_category.py --category "Attendee Engagement & Interaction"
    python test_condensation_single_category.py --category "Event Logistics & Infrastructure"
"""

import argparse
from typing import List, Optional

from llm_parallelization.new_processor import NewProcessor
from pydantic import BaseModel, Field
from utils.data_io import load_json

# =============================================================================
# Configuration
# =============================================================================

TAXONOMY_PATH = "/data-fast/data3/clyde/projects/world/documents/schemas/schema_v1.json"
MODEL = "casperhansen/mistral-nemo-instruct-2407-awq"


# =============================================================================
# Condensation Schemas
# =============================================================================


class CondensedElement(BaseModel):
    """Concise element description for use in classification prompts."""

    name: str = Field(description="Element name (unchanged from taxonomy)")
    short_description: str = Field(
        description="Concise description in 8-15 words, focusing on key distinguishing features"
    )


class CondensedCategory(BaseModel):
    """Condensed category with all elements."""

    name: str = Field(description="Category name (unchanged from taxonomy)")
    short_description: str = Field(
        description="One sentence description of category scope (max 20 words)"
    )
    elements: List[CondensedElement] = Field(
        description="List of condensed element descriptions"
    )


# =============================================================================
# Condensation Prompt
# =============================================================================


def create_condensation_prompt(category_data: dict) -> str:
    """
    Create a prompt that asks the LLM to condense verbose taxonomy definitions
    into concise, classifier-friendly descriptions.
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

    # Note: The prompt instructs to return in the CondensedCategory format


# =============================================================================
# Helper: Find category in taxonomy
# =============================================================================


def find_category(taxonomy: dict, category_name: str) -> Optional[dict]:
    """Find a category by name in the taxonomy."""
    for cat in taxonomy.get("children", []):
        if cat["name"] == category_name:
            return cat
    return None


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test condensation on a single category")
    parser.add_argument(
        "--taxonomy", type=str, default=TAXONOMY_PATH, help="Path to taxonomy JSON"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Attendee Engagement & Interaction",
        help="Category name to test",
    )
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="GPU(s) to use")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Display the full condensation prompt",
    )

    args = parser.parse_args()

    # Load taxonomy
    print(f"Loading taxonomy from: {args.taxonomy}")
    taxonomy = load_json(args.taxonomy)

    # Find the requested category
    print(f"\nSearching for category: {args.category}")
    category_data = find_category(taxonomy, args.category)

    if not category_data:
        print(f"ERROR: Category '{args.category}' not found in taxonomy")
        print("\nAvailable categories:")
        for cat in taxonomy.get("children", []):
            print(f"  - {cat['name']}")
        return

    print(f"✓ Found category with {len(category_data.get('children', []))} elements")

    # Create condensation prompt
    condensation_prompt = create_condensation_prompt(category_data)

    if args.show_prompt:
        print("\n" + "=" * 70)
        print("CONDENSATION PROMPT:")
        print("=" * 70)
        print(condensation_prompt)

    # Show BEFORE state
    print("\n" + "=" * 70)
    print("BEFORE - Verbose Definitions:")
    print("=" * 70)
    print(f"\nCategory: {category_data['name']}")
    print(f"Definition: {category_data.get('definition', 'N/A')[:200]}...")
    print(f"\nElements ({len(category_data.get('children', []))}):")
    for elem in category_data.get("children", []):
        elem_def = elem.get("definition", elem.get("description", "N/A"))
        print(f"\n  {elem['name']}:")
        print(f"    {elem_def[:150]}...")

    # Initialize processor and generate
    print("\n" + "=" * 70)
    print("Initializing processor and generating condensed version...")
    print("=" * 70)

    with NewProcessor(
        gpu_list=args.gpu,
        llm=MODEL,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    ) as processor:
        responses = processor.process_with_schema(
            prompts=[condensation_prompt],
            schema=CondensedCategory,
            batch_size=1,
            guided_config={
                "temperature": 0.7,  # Some creativity but not too much
                "max_tokens": 2000,
            },
        )

        results = processor.parse_results_with_schema(
            schema=CondensedCategory,
            responses=responses,
            validate=True,
        )

        if results and results[0]:
            condensed = results[0]

            # Show AFTER state
            print("\n" + "=" * 70)
            print("AFTER - Condensed Descriptions:")
            print("=" * 70)
            print(f"\nCategory: {condensed.name}")
            print(f"Short Description: {condensed.short_description}")
            print(f"\nElements ({len(condensed.elements)}):")
            for elem in condensed.elements:
                print(f"  - {elem.name}: {elem.short_description}")

            # Quality checks
            print("\n" + "=" * 70)
            print("QUALITY CHECKS:")
            print("=" * 70)

            # Check category description
            cat_desc = condensed.short_description
            print(f"✓ Category description: {cat_desc}")

            # Check for concreteness markers
            vague_words = ["overall", "general", "broad", "various", "impact", "availability"]
            has_vague = any(word in cat_desc.lower() for word in vague_words)
            if has_vague:
                print("  ⚠ Contains vague language")
            else:
                print("  ✓ Concrete description")

            print("\nElement descriptions:")
            for elem in condensed.elements:
                desc = elem.short_description

                # Check for vague language
                has_vague = any(word in desc.lower() for word in vague_words)

                # Check for concrete examples (commas often indicate lists of examples)
                has_examples = "," in desc or " and " in desc

                if has_vague:
                    status = "⚠"
                    note = " (vague language)"
                elif not has_examples:
                    status = "⚠"
                    note = " (no concrete examples)"
                else:
                    status = "✓"
                    note = ""

                print(f"  {status} {elem.name}: {desc}{note}")

            # Show how it would look in a prompt
            print("\n" + "=" * 70)
            print("PREVIEW - How this would appear in Stage 1 prompt:")
            print("=" * 70)
            print(f"\n**{condensed.name}**")
            print(f"{condensed.short_description}")
            for elem in condensed.elements:
                print(f"- {elem.name}: {elem.short_description}")

            print("\n" + "=" * 70)
            print("COMPARISON:")
            print("=" * 70)
            original_total = sum(
                len(e.get("definition", e.get("description", "")))
                for e in category_data.get("children", [])
            )
            condensed_total = sum(len(e.short_description) for e in condensed.elements)

            print(f"Original total chars: {original_total}")
            print(f"Condensed total chars: {condensed_total}")
            print(f"Reduction: {100 * (1 - condensed_total / original_total):.1f}%")

        else:
            print("\nERROR: Failed to generate condensed version")
            print(f"Raw responses: {responses}")


if __name__ == "__main__":
    main()
