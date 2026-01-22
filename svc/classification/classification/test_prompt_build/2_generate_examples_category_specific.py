"""
Generate Category-Specific Examples (Step 2a)

Generates 1-2 simple, realistic conference feedback comments for each category
that clearly demonstrate when that category should be detected.

Usage:
    python generate_examples_category_specific.py
    python generate_examples_category_specific.py --output examples_simple.json
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

CONDENSED_TAXONOMY_PATH = "./condensed_taxonomy_stage1.json"
MODEL = "casperhansen/mistral-nemo-instruct-2407-awq"
DEFAULT_OUTPUT = "./examples_simple_stage1.json"


# =============================================================================
# Example Schemas
# =============================================================================


class ElementDetail(BaseModel):
    """Stage 2 element extraction details."""

    element: str = Field(description="Specific element name within the category")
    excerpt: str = Field(description="Exact text excerpt from comment relating to this element")
    sentiment: str = Field(description="Sentiment: positive, negative, neutral, or mixed")
    reasoning: str = Field(
        description="Brief explanation of why this excerpt maps to this element"
    )


class DualPurposeExample(BaseModel):
    """Example that works for both Stage 1 (category detection) and Stage 2 (element extraction)."""

    comment: str = Field(description="A realistic conference feedback comment (1-2 sentences)")

    # Stage 1: Category detection
    categories_present: List[str] = Field(
        description="Top-level category names ONLY (e.g., 'Attendee Engagement & Interaction', 'Event Logistics & Infrastructure')"
    )
    has_classifiable_content: bool = Field(
        description="Whether this contains classifiable conference feedback"
    )
    stage1_reasoning: str = Field(description="Brief explanation of why these categories apply")

    # Stage 2: Element extraction
    element_details: List[ElementDetail] = Field(
        description="Detailed element-level extractions for Stage 2"
    )


class CategoryExamples(BaseModel):
    """Examples for a single category."""

    category_name: str = Field(description="Name of the category")
    examples: List[DualPurposeExample] = Field(
        description="1-2 dual-purpose examples that demonstrate this category"
    )


# =============================================================================
# Example Generation Prompt
# =============================================================================


def create_category_example_prompt(category_data: dict) -> str:
    """
    Create a prompt to generate dual-purpose examples for a specific category.
    """
    cat_name = category_data["name"]
    cat_desc = category_data["short_description"]

    # Format elements
    elements_text = "\n".join(
        f"- {elem['name']}: {elem['short_description']}" for elem in category_data["elements"]
    )

    return f"""You are an expert at creating training examples for conference feedback classification systems.

TASK: Generate 1-2 realistic conference attendee comments for the category below, with BOTH Stage 1 (category detection) and Stage 2 (element extraction) labels.

---

CATEGORY: {cat_name}

DESCRIPTION: {cat_desc}

ELEMENTS IN THIS CATEGORY:
{elements_text}

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
     - element: The specific element name (e.g., "Community", "Networking")
     - excerpt: The EXACT text from the comment that relates to this element
     - sentiment: positive, negative, neutral, or mixed
     - reasoning: Why this excerpt maps to this element

4. **Vary the elements**
   - Example 1: Focus on one specific element
   - Example 2: Focus on a different element (can include multiple elements if natural)

---

EXAMPLE FORMAT (from a different category):

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
  ]
}}

Another example with multiple elements:

{{
  "comment": "The community is so supportive and open to sharing their knowledge.",
  "categories_present": ["Attendee Engagement & Interaction"],
  "has_classifiable_content": true,
  "stage1_reasoning": "Discusses community feeling and knowledge sharing",
  "element_details": [
    {{
      "element": "Community",
      "excerpt": "The community is so supportive",
      "sentiment": "positive",
      "reasoning": "Expresses feeling of supportive community environment"
    }},
    {{
      "element": "Knowledge Exchange",
      "excerpt": "open to sharing their knowledge",
      "sentiment": "positive",
      "reasoning": "References peer knowledge sharing"
    }}
  ]
}}

---

Generate 1-2 dual-purpose examples for the **{cat_name}** category. Return as JSON."""


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate simple category-specific examples for Stage 1"
    )
    parser.add_argument(
        "--condensed-taxonomy",
        type=str,
        default=CONDENSED_TAXONOMY_PATH,
        help="Path to condensed taxonomy JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to save examples JSON",
    )
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="GPU(s) to use")
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Display sample generation prompts",
    )

    args = parser.parse_args()

    # Load condensed taxonomy
    print(f"Loading condensed taxonomy from: {args.condensed_taxonomy}")
    condensed_taxonomy = load_json(args.condensed_taxonomy)

    categories = condensed_taxonomy.get("categories", [])
    print(f"Found {len(categories)} categories")

    # Create prompts for each category
    print("\nPreparing example generation prompts...")
    prompts = []
    category_names = []

    for cat in categories:
        cat_name = cat["name"]
        category_names.append(cat_name)
        prompt = create_category_example_prompt(cat)
        prompts.append(prompt)
        print(f"  ✓ {cat_name}")

    if args.show_prompts:
        print("\n" + "=" * 70)
        print("SAMPLE PROMPT (first category):")
        print("=" * 70)
        print(prompts[0])

    # Generate examples
    print("\n" + "=" * 70)
    print("Generating examples for all categories...")
    print("=" * 70)

    with NewProcessor(
        gpu_list=args.gpu,
        llm=MODEL,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    ) as processor:
        responses = processor.process_with_schema(
            prompts=prompts,
            schema=CategoryExamples,
            batch_size=len(prompts),
            guided_config={
                "temperature": 0.8,  # More creative for diverse examples
                "max_tokens": 1000,
            },
        )

        results = processor.parse_results_with_schema(
            schema=CategoryExamples,
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
            print("\n⚠ WARNING: Some categories failed:")
            for i, name in failed:
                print(f"  - {name}")
                print(f"    Raw: {responses[i][:200]}...")
            return

        print(f"\n✓ Generated examples for all {len(results)} categories")

        # Display examples
        print("\n" + "=" * 70)
        print("GENERATED EXAMPLES:")
        print("=" * 70)

        all_examples = []

        for cat_examples in results:
            print(f"\n**{cat_examples.category_name}**")
            for i, example in enumerate(cat_examples.examples, 1):
                print(f"\nExample {i}:")
                print(f'  Comment: "{example.comment}"')
                print(f"  Stage 1 - Categories: {example.categories_present}")
                print(f"  Stage 1 - Reasoning: {example.stage1_reasoning}")
                print(f"  Stage 2 - Elements ({len(example.element_details)}):")
                for elem_detail in example.element_details:
                    print(
                        f'    - {elem_detail.element}: "{elem_detail.excerpt}" [{elem_detail.sentiment}]'
                    )
                    print(f"      Reasoning: {elem_detail.reasoning}")

                # Collect for output
                all_examples.append(
                    {
                        "category": cat_examples.category_name,
                        "comment": example.comment,
                        "categories_present": example.categories_present,
                        "has_classifiable_content": example.has_classifiable_content,
                        "stage1_reasoning": example.stage1_reasoning,
                        "element_details": [
                            {
                                "element": ed.element,
                                "excerpt": ed.excerpt,
                                "sentiment": ed.sentiment,
                                "reasoning": ed.reasoning,
                            }
                            for ed in example.element_details
                        ],
                    }
                )

        # Quality checks
        print("\n" + "=" * 70)
        print("QUALITY CHECKS:")
        print("=" * 70)

        total_examples = sum(len(cat.examples) for cat in results)
        print(f"Total examples generated: {total_examples}")

        # Check that each example only mentions its own category
        single_category_count = sum(
            1 for cat_ex in results for ex in cat_ex.examples if len(ex.categories_present) == 1
        )

        print(f"Single-category examples: {single_category_count}/{total_examples}")
        if single_category_count < total_examples:
            print("  ⚠ Some examples reference multiple categories (review these)")

        # Check comment lengths (should be concise)
        avg_length = (
            sum(len(ex.comment.split()) for cat_ex in results for ex in cat_ex.examples)
            / total_examples
        )

        print(f"Average comment length: {avg_length:.1f} words")
        if avg_length > 30:
            print("  ⚠ Comments might be too long (aim for 10-25 words)")

        # Save to file
        print(f"\nSaving examples to: {args.output}")
        output_data = {
            "description": "Simple category-specific examples for Stage 1 classification",
            "total_examples": total_examples,
            "examples": all_examples,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Saved {total_examples} examples")


if __name__ == "__main__":
    main()
