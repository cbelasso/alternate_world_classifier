"""
Generate Complex & Negative Examples (Step 2b)

Generates multi-category examples that test the classifier's ability to handle
complex scenarios, plus negative examples that should not be classified.

Usage:
    python generate_examples_complex.py
    python generate_examples_complex.py --output examples_complex.json
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
DEFAULT_OUTPUT = "./examples_complex_stage1.json"


# =============================================================================
# Example Schemas
# =============================================================================


class ElementDetail(BaseModel):
    """Stage 2 element extraction details."""

    category: str = Field(description="Parent category name for this element")
    element: str = Field(description="Specific element name within the category")
    excerpt: str = Field(description="Exact text excerpt from comment relating to this element")
    sentiment: str = Field(description="Sentiment: positive, negative, neutral, or mixed")
    reasoning: str = Field(
        description="Brief explanation of why this excerpt maps to this element"
    )


class ComplexExample(BaseModel):
    """A complex dual-purpose example for category detection and element extraction."""

    comment: str = Field(description="A realistic conference feedback comment (1-3 sentences)")

    # Stage 1: Category detection
    categories_present: List[str] = Field(
        description="List of top-level category names that should be detected"
    )
    has_classifiable_content: bool = Field(
        description="Whether this contains classifiable conference feedback"
    )
    stage1_reasoning: str = Field(description="Brief explanation of why these categories apply")

    # Stage 2: Element extraction (can span multiple categories)
    element_details: List[ElementDetail] = Field(
        description="Detailed element-level extractions for Stage 2"
    )

    example_type: str = Field(description="Type of example: 'multi_category' or 'negative'")


class ComplexExamples(BaseModel):
    """Collection of complex and negative examples."""

    multi_category_examples: List[ComplexExample] = Field(
        description="3-5 examples that span multiple categories"
    )
    negative_examples: List[ComplexExample] = Field(
        description="2-3 examples that should NOT be classified (not conference feedback)"
    )


# =============================================================================
# Complex Example Generation Prompt
# =============================================================================


def create_complex_example_prompt(condensed_taxonomy: dict) -> str:
    """
    Create a prompt to generate multi-category and negative examples.
    """
    # Format all categories concisely
    categories_text = []
    all_valid_elements = {}  # category -> [element names]

    for cat in condensed_taxonomy.get("categories", []):
        cat_name = cat["name"]
        cat_desc = cat["short_description"]
        categories_text.append(f"**{cat_name}**: {cat_desc}")

        # Collect valid element names
        all_valid_elements[cat_name] = [elem["name"] for elem in cat.get("elements", [])]

    categories_summary = "\n".join(categories_text)

    # Format valid elements per category
    elements_by_category = []
    for cat_name, elem_names in all_valid_elements.items():
        elements_by_category.append(
            f"**{cat_name}**: {', '.join(f'\\"{e}\\"' for e in elem_names)}"
        )

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
2. Use the EXACT element names listed above (not variations like "Event Schedule", "Panels/Discussions", etc.)

---

PART 1: MULTI-CATEGORY EXAMPLES

Generate 3-5 realistic conference feedback comments that span MULTIPLE categories.

These should:
- Sound like real attendee feedback (1-3 sentences)
- Naturally discuss 2-3 different aspects of the conference
- Demonstrate how categories can co-occur
- Show realistic combinations (e.g., praising both content AND speakers)
- Vary in sentiment (positive, negative, mixed)

For each example, provide:
- categories_present: List of 2-3 top-level categories
- stage1_reasoning: Why these categories apply
- element_details: For each element mentioned, specify the category, element (EXACT name), excerpt, sentiment, and reasoning

Examples of good multi-category combinations:
- Content + Speaker: "The keynote was brilliant and the speaker was engaging"
- Logistics + Operations: "WiFi was poor but registration was smooth"
- Engagement + Content: "Loved networking at the workshop sessions"

---

PART 2: NEGATIVE EXAMPLES

Generate 2-3 comments that look like feedback but are NOT about the conference.

These should:
- Sound similar to feedback (opinions, observations)
- But are NOT about conference experience
- Should return empty categories: []
- Should return empty element_details: []
- Examples: General wisdom, off-topic statements, philosophical musings

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
      "reasoning": "Praises the presentation content on machine learning"
    }},
    {{
      "category": "People",
      "element": "Speakers/Presenters",
      "excerpt": "The keynote speaker was brilliant",
      "sentiment": "positive",
      "reasoning": "Praises the keynote speaker specifically"
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

Generate the examples and return as JSON with two arrays: multi_category_examples and negative_examples.

REMEMBER: Use ONLY the exact element names listed above and ALWAYS include the "category" field in element_details."""


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate complex multi-category and negative examples"
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
        "--show-prompt",
        action="store_true",
        help="Display the generation prompt",
    )

    args = parser.parse_args()

    # Load condensed taxonomy
    print(f"Loading condensed taxonomy from: {args.condensed_taxonomy}")
    condensed_taxonomy = load_json(args.condensed_taxonomy)

    # Create prompt
    print("\nPreparing complex example generation prompt...")
    prompt = create_complex_example_prompt(condensed_taxonomy)

    if args.show_prompt:
        print("\n" + "=" * 70)
        print("GENERATION PROMPT:")
        print("=" * 70)
        print(prompt)

    # Generate examples
    print("\n" + "=" * 70)
    print("Generating complex and negative examples...")
    print("=" * 70)

    with NewProcessor(
        gpu_list=args.gpu,
        llm=MODEL,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    ) as processor:
        responses = processor.process_with_schema(
            prompts=[prompt],
            schema=ComplexExamples,
            batch_size=1,
            guided_config={
                "temperature": 0.85,  # Creative but controlled
                "max_tokens": 2000,
            },
        )

        results = processor.parse_results_with_schema(
            schema=ComplexExamples,
            responses=responses,
            validate=True,
        )

        if not results or not results[0]:
            print("\n⚠ ERROR: Failed to generate examples")
            print(f"Raw response: {responses[0] if responses else 'No response'}")
            return

        examples = results[0]

        # Display results
        print("\n" + "=" * 70)
        print("MULTI-CATEGORY EXAMPLES:")
        print("=" * 70)

        for i, ex in enumerate(examples.multi_category_examples, 1):
            print(f"\nExample {i}:")
            print(f'  Comment: "{ex.comment}"')
            print(f"  Stage 1 - Categories: {ex.categories_present}")
            print(f"  Stage 1 - Reasoning: {ex.stage1_reasoning}")
            print(f"  Stage 2 - Elements ({len(ex.element_details)}):")
            for elem_detail in ex.element_details:
                print(
                    f'    - [{elem_detail.category}] {elem_detail.element}: "{elem_detail.excerpt}" [{elem_detail.sentiment}]'
                )
                print(f"      Reasoning: {elem_detail.reasoning}")

        print("\n" + "=" * 70)
        print("NEGATIVE EXAMPLES:")
        print("=" * 70)

        for i, ex in enumerate(examples.negative_examples, 1):
            print(f"\nExample {i}:")
            print(f'  Comment: "{ex.comment}"')
            print(f"  Stage 1 - Categories: {ex.categories_present}")
            print(f"  Stage 1 - Reasoning: {ex.stage1_reasoning}")
            print(f"  Stage 2 - Elements: {len(ex.element_details)} (should be 0)")

        # Quality checks
        print("\n" + "=" * 70)
        print("QUALITY CHECKS:")
        print("=" * 70)

        multi_count = len(examples.multi_category_examples)
        neg_count = len(examples.negative_examples)

        print(f"Multi-category examples: {multi_count} (target: 3-5)")
        print(f"Negative examples: {neg_count} (target: 2-3)")

        # Check multi-category examples actually have multiple categories
        truly_multi = sum(
            1 for ex in examples.multi_category_examples if len(ex.categories_present) >= 2
        )
        print(f"\nExamples with 2+ categories: {truly_multi}/{multi_count}")
        if truly_multi < multi_count:
            print("  ⚠ Some 'multi-category' examples only have 1 category")

        # Check negative examples are actually negative
        truly_negative = sum(
            1 for ex in examples.negative_examples if len(ex.categories_present) == 0
        )
        print(f"Negative examples with 0 categories: {truly_negative}/{neg_count}")
        if truly_negative < neg_count:
            print("  ⚠ Some 'negative' examples have categories assigned")

        # Comment length distribution
        multi_lengths = [len(ex.comment.split()) for ex in examples.multi_category_examples]
        avg_multi = sum(multi_lengths) / len(multi_lengths) if multi_lengths else 0

        print(f"\nAverage multi-category comment length: {avg_multi:.1f} words")

        # Save to file
        print(f"\nSaving examples to: {args.output}")
        output_data = {
            "description": "Complex multi-category and negative examples for Stage 1",
            "multi_category_examples": [
                {
                    "comment": ex.comment,
                    "categories_present": ex.categories_present,
                    "has_classifiable_content": ex.has_classifiable_content,
                    "stage1_reasoning": ex.stage1_reasoning,
                    "element_details": [
                        {
                            "category": ed.category,
                            "element": ed.element,
                            "excerpt": ed.excerpt,
                            "sentiment": ed.sentiment,
                            "reasoning": ed.reasoning,
                        }
                        for ed in ex.element_details
                    ],
                    "example_type": "multi_category",
                }
                for ex in examples.multi_category_examples
            ],
            "negative_examples": [
                {
                    "comment": ex.comment,
                    "categories_present": ex.categories_present,
                    "has_classifiable_content": ex.has_classifiable_content,
                    "stage1_reasoning": ex.stage1_reasoning,
                    "element_details": [],
                    "example_type": "negative",
                }
                for ex in examples.negative_examples
            ],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Saved {multi_count + neg_count} examples")


if __name__ == "__main__":
    main()
