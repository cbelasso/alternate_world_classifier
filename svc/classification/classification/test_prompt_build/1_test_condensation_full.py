"""
Full Taxonomy Condensation

Condenses all categories in the taxonomy to create concise, classification-friendly
descriptions suitable for use in Stage 1 prompts.

Saves the condensed taxonomy as JSON for use in prompt generation.

Usage:
    python test_condensation_full.py
    python test_condensation_full.py --output condensed_taxonomy.json
"""

import argparse
import json
from typing import List

from llm_parallelization.new_processor import NewProcessor
from pydantic import BaseModel, Field
from utils.data_io import load_json

# =============================================================================
# Configuration
# =============================================================================

TAXONOMY_PATH = "/data-fast/data3/clyde/projects/world/documents/schemas/schema_v1.json"
MODEL = "casperhansen/mistral-nemo-instruct-2407-awq"
DEFAULT_OUTPUT = "./condensed_taxonomy_stage1.json"


# =============================================================================
# Condensation Schemas
# =============================================================================


class CondensedElement(BaseModel):
    """Concise element description for use in classification prompts."""

    name: str = Field(description="Element name (unchanged from taxonomy)")
    short_description: str = Field(
        description="Concise description focusing on key distinguishing features and examples"
    )


class CondensedCategory(BaseModel):
    """Condensed category with all elements."""

    name: str = Field(description="Category name (unchanged from taxonomy)")
    short_description: str = Field(description="One sentence description of category scope")
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


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Condense all categories in taxonomy for Stage 1"
    )
    parser.add_argument(
        "--taxonomy", type=str, default=TAXONOMY_PATH, help="Path to taxonomy JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to save condensed taxonomy JSON",
    )
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="GPU(s) to use")
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Display the condensation prompts",
    )

    args = parser.parse_args()

    # Load taxonomy
    print(f"Loading taxonomy from: {args.taxonomy}")
    taxonomy = load_json(args.taxonomy)

    categories = taxonomy.get("children", [])
    print(f"Found {len(categories)} categories to condense")

    # Create condensation prompts for all categories
    print("\nPreparing condensation prompts...")
    prompts = []
    category_names = []

    for cat in categories:
        cat_name = cat["name"]
        category_names.append(cat_name)
        prompt = create_condensation_prompt(cat)
        prompts.append(prompt)
        print(f"  ✓ {cat_name} ({len(cat.get('children', []))} elements)")

    if args.show_prompts:
        print("\n" + "=" * 70)
        print("SAMPLE CONDENSATION PROMPT (first category):")
        print("=" * 70)
        print(prompts[0][:1000] + "...\n")

    # Initialize processor and generate condensed descriptions
    print("\n" + "=" * 70)
    print("Initializing processor and condensing all categories...")
    print("=" * 70)

    with NewProcessor(
        gpu_list=args.gpu,
        llm=MODEL,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    ) as processor:
        responses = processor.process_with_schema(
            prompts=prompts,
            schema=CondensedCategory,
            batch_size=len(prompts),  # Process all at once
            guided_config={
                "temperature": 0.7,
                "max_tokens": 2000,
            },
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
            if not result
        ]

        if failed:
            print("\n⚠ WARNING: Some categories failed to condense:")
            for i, name in failed:
                print(f"  - {name}")
                print(f"    Raw response: {responses[i][:200]}...")
            return

        print(f"\n✓ Successfully condensed all {len(results)} categories")

        # Display results
        print("\n" + "=" * 70)
        print("CONDENSED TAXONOMY PREVIEW:")
        print("=" * 70)

        condensed_data = {"categories": []}

        for cat_result in results:
            print(f"\n**{cat_result.name}**")
            print(f"{cat_result.short_description}")

            for elem in cat_result.elements:
                print(f"- {elem.name}: {elem.short_description}")

            # Build output structure
            condensed_data["categories"].append(
                {
                    "name": cat_result.name,
                    "short_description": cat_result.short_description,
                    "elements": [
                        {"name": e.name, "short_description": e.short_description}
                        for e in cat_result.elements
                    ],
                }
            )

        # Quality summary
        print("\n" + "=" * 70)
        print("QUALITY SUMMARY:")
        print("=" * 70)

        vague_words = ["overall", "general", "broad", "various", "impact", "availability"]
        total_elements = 0
        vague_count = 0

        for cat_result in results:
            for elem in cat_result.elements:
                total_elements += 1
                if any(word in elem.short_description.lower() for word in vague_words):
                    vague_count += 1

        print(f"Total elements: {total_elements}")
        print(
            f"Elements with vague language: {vague_count} ({100 * vague_count / total_elements:.1f}%)"
        )

        # Category description quality
        cat_vague = sum(
            1
            for cat in results
            if any(word in cat.short_description.lower() for word in vague_words)
        )
        print(f"Categories with vague language: {cat_vague}/{len(results)}")

        # Save to file
        print(f"\nSaving condensed taxonomy to: {args.output}")
        with open(args.output, "w") as f:
            json.dump(condensed_data, f, indent=2)

        print("✓ Saved successfully")

        # Show comparison stats
        print("\n" + "=" * 70)
        print("SIZE COMPARISON:")
        print("=" * 70)

        original_chars = sum(
            len(elem.get("definition", elem.get("description", "")))
            for cat in categories
            for elem in cat.get("children", [])
        )

        condensed_chars = sum(
            len(elem.short_description)
            for cat_result in results
            for elem in cat_result.elements
        )

        print(f"Original total chars: {original_chars:,}")
        print(f"Condensed total chars: {condensed_chars:,}")
        reduction = 100 * (1 - condensed_chars / original_chars)
        print(f"Size change: {reduction:+.1f}%")


if __name__ == "__main__":
    main()
