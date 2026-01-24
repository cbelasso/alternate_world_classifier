"""
Stage 1 Prompt Generation Experiment

Tests whether an LLM can generate a Stage 1 classification prompt
from taxonomy metadata, using a handcrafted example as reference.

Usage:
    python test_prompt_generation.py --taxonomy /path/to/schema_v1.json
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
# Output Schema
# =============================================================================


class GeneratedStage1Prompt(BaseModel):
    """Schema for the generated Stage 1 prompt."""

    prompt_template: str = Field(
        description="The complete Stage 1 prompt template with {comment} placeholder"
    )


# =============================================================================
# Reference: Handcrafted Stage 1 Prompt (truncated for example)
# =============================================================================

HANDCRAFTED_EXAMPLE = """You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in a conference attendee comment.

COMMENT TO ANALYZE:
{comment}

---

CATEGORIES AND THEIR SCOPE:

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
- Hotel: Accommodation, lodging arrangements
- Technological Tools: A/V equipment, microphones, projectors, tech setup
- Transportation: Getting to/from venue, shuttles, parking, travel arrangements
- Wi-Fi: Internet connectivity, network access

**Event Operations & Management**
Feedback about how the conference was organized and run.
- Conference: General event organization, overall management, event quality
- Conference Registration: Sign-up process, check-in, badge pickup
- Conference Scheduling: Session timing, agenda, time management, scheduling conflicts
- Messaging & Awareness: Communication, announcements, information clarity, signage

**Learning & Content Delivery**
Feedback about educational content and how it was delivered.
- Demonstration: Live demos, product showcases, hands-on examples
- Gained Knowledge: What attendees learned, takeaways, actionable insights
- Open Discussion: Q&A sessions, audience participation, interactive discussions
- Panel Discussions: Panel format sessions, multi-speaker discussions
- Presentations: Individual talks, keynotes, speaker presentations
- Resources/Materials: Handouts, slides, documentation, learning materials
- Session/Workshop: Breakout sessions, workshops, training sessions
- Topics: Subject matter, themes, content relevance, topic selection

**People**
Feedback about specific people or groups at the conference.
- Conference Staff: Organizers, volunteers, support staff, event team
- Experts/Consultants: Industry experts, product specialists, consultants
- Participants/Attendees: Fellow attendees, other conference-goers
- Speakers/Presenters: Keynote speakers, session presenters, panelists
- Unspecified Person: References to people without clear role identification

---

CLASSIFICATION RULES:

1. A comment can belong to MULTIPLE categories if it discusses multiple aspects.
2. Focus on what the comment is ABOUT, not just words mentioned.
3. "Community" refers to the feeling of belonging; "Networking" refers to the act of meeting people.
4. "Presentations" = talk quality/content; "Speakers/Presenters" = the people themselves.
5. General praise like "great conference" without specifics → Event Operations & Management > Conference.
6. If a comment mentions both the content AND the presenter, include BOTH categories.

---

EXAMPLES:

Comment: "The networking sessions were fantastic and I made great connections with peers from other institutions."
{{"categories_present": ["Attendee Engagement & Interaction"], "has_classifiable_content": true, "reasoning": "Discusses networking and peer connections"}}

Comment: "The WiFi kept dropping during sessions and the room was too cold."
{{"categories_present": ["Event Logistics & Infrastructure"], "has_classifiable_content": true, "reasoning": "Mentions WiFi connectivity and venue temperature issues"}}

Comment: "Data integrity never goes out of style."
{{"categories_present": [], "has_classifiable_content": false, "reasoning": "General statement not specifically about conference feedback"}}

---

Analyze the comment and return ONLY valid JSON."""


# =============================================================================
# Helper: Extract taxonomy info for the meta-prompt
# =============================================================================


def extract_taxonomy_for_prompt(taxonomy: dict) -> str:
    """
    Extract category and element information from taxonomy
    in a format suitable for the meta-prompt.
    """
    lines = []

    for category in taxonomy.get("children", []):
        cat_name = category["name"]
        cat_description = category.get("description", "")
        cat_definition = category.get("definition", "")

        lines.append(f"CATEGORY: {cat_name}")
        if cat_description:
            lines.append(f"  Description: {cat_description}")
        if cat_definition:
            lines.append(f"  Definition: {cat_definition}")

        lines.append("  Elements:")
        for element in category.get("children", []):
            elem_name = element["name"]
            elem_description = element.get("description", "")
            elem_definition = element.get("definition", "")

            if elem_description:
                lines.append(f"    - {elem_name}: {elem_description}")
            elif elem_definition:
                lines.append(f"    - {elem_name}: {elem_definition}")
            else:
                lines.append(f"    - {elem_name}")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Meta-Prompt: Ask LLM to generate Stage 1 prompt
# =============================================================================


def create_meta_prompt(taxonomy: dict) -> str:
    """
    Create the meta-prompt that asks the LLM to generate a Stage 1 classification prompt.
    """
    taxonomy_info = extract_taxonomy_for_prompt(taxonomy)

    return f"""You are a prompt engineering expert. Your task is to generate a Stage 1 classification prompt for a conference feedback classification system.

The Stage 1 prompt should:
1. Identify which CATEGORIES are present in a user comment
2. Include a {{comment}} placeholder where the actual comment will be inserted
3. List all categories with their elements and brief descriptions
4. Include classification rules to help disambiguate between categories
5. Include 3-5 examples showing input comments and expected JSON output
6. End with an instruction to return ONLY valid JSON

The expected JSON output format is:
{{"categories_present": ["Category1", "Category2"], "has_classifiable_content": true, "reasoning": "Brief explanation"}}

---

TAXONOMY DATA:

{taxonomy_info}

---

EXAMPLE OF A WELL-STRUCTURED STAGE 1 PROMPT:
```
{HANDCRAFTED_EXAMPLE}
```

---

Now generate a Stage 1 classification prompt using the taxonomy data provided above. 
The prompt should follow the same structure as the example but use the actual taxonomy categories and elements.
Make sure to include the {{comment}} placeholder.

Return your response as JSON with a single field "prompt_template" containing the full prompt text."""


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test LLM-based Stage 1 prompt generation")
    parser.add_argument(
        "--taxonomy", type=str, default=TAXONOMY_PATH, help="Path to taxonomy JSON file"
    )
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="GPU(s) to use")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional: Save generated prompt to file"
    )

    args = parser.parse_args()

    # Load taxonomy
    print(f"Loading taxonomy from: {args.taxonomy}")
    taxonomy = load_json(args.taxonomy)

    # Create meta-prompt
    meta_prompt = create_meta_prompt(taxonomy)

    print("\n" + "=" * 70)
    print("META-PROMPT (first 500 chars):")
    print("=" * 70)
    print(meta_prompt[:500] + "...")

    # Initialize processor
    print("\n" + "=" * 70)
    print("Initializing processor...")
    print("=" * 70)

    with NewProcessor(
        gpu_list=args.gpu,
        llm=MODEL,
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.9,
    ) as processor:
        # Generate
        print("\nGenerating Stage 1 prompt...")

        responses = processor.process_with_schema(
            prompts=[meta_prompt],
            schema=GeneratedStage1Prompt,
            batch_size=1,
            guided_config={
                "temperature": 0.9,
                "max_tokens": 10000,
            },
        )

        # Parse
        results = processor.parse_results_with_schema(
            schema=GeneratedStage1Prompt,
            responses=responses,
            validate=True,
        )

        __import__("ipdb").set_trace()
        if results and results[0]:
            generated = results[0]

            print("\n" + "=" * 70)
            print("GENERATED STAGE 1 PROMPT:")
            print("=" * 70)
            print(generated.prompt_template)

            # Save if requested
            if args.output:
                with open(args.output, "w") as f:
                    f.write(generated.prompt_template)
                print(f"\nSaved to: {args.output}")

            # Quick comparison
            print("\n" + "=" * 70)
            print("COMPARISON:")
            print("=" * 70)
            print(f"Handcrafted prompt length: {len(HANDCRAFTED_EXAMPLE)} chars")
            print(f"Generated prompt length: {len(generated.prompt_template)} chars")

            # Check for key components
            checks = [
                ("{comment}", "Has {comment} placeholder"),
                ("CATEGORIES", "Has CATEGORIES section"),
                ("RULES", "Has RULES section"),
                ("EXAMPLES", "Has EXAMPLES section"),
                ("categories_present", "Has expected JSON format"),
            ]

            print("\nComponent checks:")
            for pattern, description in checks:
                found = pattern in generated.prompt_template
                status = "✓" if found else "✗"
                print(f"  {status} {description}")

        else:
            print("ERROR: Failed to generate prompt")
            print(f"Raw responses: {responses}")


if __name__ == "__main__":
    main()
