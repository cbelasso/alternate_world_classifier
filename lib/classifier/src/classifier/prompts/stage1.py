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
    stage1_prompt = build_stage1_prompt_function(condensed, examples, use_handcrafted=False)

    # Use at runtime
    prompt = stage1_prompt("The keynote was amazing!")
"""

from pathlib import Path
from typing import Callable, List, Optional

from ..taxonomy.condenser import CondensedTaxonomy
from ..taxonomy.example_generator import ClassificationExample, ExampleSet
from .base import format_stage1_example

# =============================================================================
# Handcrafted Stage 1 Content (Extracted from Proven Standalone Script)
# =============================================================================

HANDCRAFTED_STAGE1_CATEGORIES = """**Attendee Engagement & Interaction**
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
- Unspecified Person: References to people without clear role identification"""

HANDCRAFTED_STAGE1_RULES = [
    "A comment can belong to MULTIPLE categories if it discusses multiple aspects.",
    "Focus on what the comment is ABOUT, not just words mentioned.",
    '"Community" refers to the feeling of belonging; "Networking" refers to the act of meeting people.',
    '"Presentations" = talk quality/content; "Speakers/Presenters" = the people themselves.',
    'General praise like "great conference" without specifics → Event Operations & Management > Conference.',
    "If a comment mentions both the content AND the presenter, include BOTH categories.",
]

HANDCRAFTED_STAGE1_EXAMPLES = [
    {
        "comment": "The networking sessions were fantastic and I made great connections with peers from other institutions.",
        "output": '{"categories_present": ["Attendee Engagement & Interaction"], "has_classifiable_content": true, "reasoning": "Discusses networking and peer connections"}',
    },
    {
        "comment": "The WiFi kept dropping during sessions and the room was too cold.",
        "output": '{"categories_present": ["Event Logistics & Infrastructure"], "has_classifiable_content": true, "reasoning": "Mentions WiFi connectivity and venue temperature issues"}',
    },
    {
        "comment": "The keynote speaker was brilliant and the presentation on machine learning was very insightful.",
        "output": '{"categories_present": ["Learning & Content Delivery", "People"], "has_classifiable_content": true, "reasoning": "Discusses both the presentation content and the speaker"}',
    },
    {
        "comment": "The conference was well organized but I wish there were more hands-on workshops. The Explorance team was very helpful.",
        "output": '{"categories_present": ["Event Operations & Management", "Learning & Content Delivery", "People"], "has_classifiable_content": true, "reasoning": "Covers organization quality, workshop content request, and staff praise"}',
    },
    {
        "comment": "I loved connecting with the Blue community and sharing knowledge with other users.",
        "output": '{"categories_present": ["Attendee Engagement & Interaction"], "has_classifiable_content": true, "reasoning": "Discusses community connection and knowledge sharing among attendees"}',
    },
    {
        "comment": "The registration process was slow and confusing.",
        "output": '{"categories_present": ["Event Operations & Management"], "has_classifiable_content": true, "reasoning": "Feedback about registration process"}',
    },
    {
        "comment": "Seeing is believing!",
        "output": '{"categories_present": ["Event Operations & Management"], "has_classifiable_content": true, "reasoning": "Vague positive sentiment about the conference overall"}',
    },
    {
        "comment": "Data integrity never goes out of style.",
        "output": '{"categories_present": [], "has_classifiable_content": false, "reasoning": "General statement not specifically about conference feedback"}',
    },
    {
        "comment": "The gala dinner was amazing and the hotel was conveniently located near the venue.",
        "output": '{"categories_present": ["Attendee Engagement & Interaction", "Event Logistics & Infrastructure"], "has_classifiable_content": true, "reasoning": "Discusses social event (gala dinner) and accommodation/venue location"}',
    },
    {
        "comment": "I learned so much from the panel discussions and really appreciated the open Q&A format.",
        "output": '{"categories_present": ["Learning & Content Delivery"], "has_classifiable_content": true, "reasoning": "Discusses learning from panel discussions and Q&A sessions"}',
    },
]


# =============================================================================
# Prompt Template
# =============================================================================

STAGE1_TEMPLATE = """You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in a conference attendee comment.

COMMENT TO ANALYZE:
{comment}

---

CATEGORIES AND THEIR SCOPE:

{categories}

---

CLASSIFICATION RULES:

{rules}

---

EXAMPLES:

{examples}

---

Analyze the comment and return ONLY valid JSON."""


# =============================================================================
# Example Curation Helper
# =============================================================================


def _curate_generated_examples(
    examples: List[ClassificationExample],
    max_per_category: int = 1,
) -> List[ClassificationExample]:
    """
    Curate generated examples to avoid element-level bloat.

    Strategy:
    - Group examples by source_category (or first category in categories_present)
    - Take max N examples per category
    - Always include multi-category examples
    - Always include negative examples (has_classifiable_content=False)

    Args:
        examples: List of all simple examples
        max_per_category: Max examples per single category (default: 1)

    Returns:
        Curated list of examples suitable for Stage 1
    """
    from collections import defaultdict

    # Group by category
    by_category = defaultdict(list)
    multi_category = []
    negative = []

    for ex in examples:
        # Negative examples (always include)
        if not ex.has_classifiable_content:
            negative.append(ex)
            continue

        # Multi-category examples (always include)
        if len(ex.categories_present) > 1:
            multi_category.append(ex)
            continue

        # Single category examples (limit per category)
        if ex.categories_present:
            cat = ex.categories_present[0]
            by_category[cat].append(ex)
        elif ex.source_category:
            by_category[ex.source_category].append(ex)

    # Take max N per category
    curated = []
    for cat, cat_examples in by_category.items():
        curated.extend(cat_examples[:max_per_category])

    # Add all multi-category and negative
    curated.extend(multi_category)
    curated.extend(negative)

    return curated


# =============================================================================
# Formatting Helpers
# =============================================================================


def _format_stage1_examples_handcrafted() -> str:
    """Format handcrafted Stage 1 examples."""
    formatted = []
    for ex in HANDCRAFTED_STAGE1_EXAMPLES:
        formatted.append(f'Comment: "{ex["comment"]}"\n{ex["output"]}')
    return "\n\n".join(formatted)


def _format_stage1_examples_generated(examples: List[ClassificationExample]) -> str:
    """Format generated Stage 1 examples."""
    if not examples:
        return "(No examples available)"

    formatted = []
    for ex in examples:
        formatted.append(format_stage1_example(ex))

    return "\n\n".join(formatted)


def _format_rules(rules: List[str]) -> str:
    """Format rules as numbered list."""
    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))


# =============================================================================
# Main Builder Functions
# =============================================================================


def build_stage1_prompt_string(
    comment: str,
    condensed: Optional[CondensedTaxonomy] = None,
    examples: Optional[List[ClassificationExample]] = None,
    rules: Optional[List[str]] = None,
    use_handcrafted: bool = True,
) -> str:
    """
    Build a complete Stage 1 prompt for a specific comment.

    Args:
        comment: The conference feedback comment
        condensed: CondensedTaxonomy (optional, for generated content)
        examples: List of examples (optional, for generated content)
        rules: Optional rules (optional, for generated content)
        use_handcrafted: If True, use handcrafted sections (default: True)

    Returns:
        Complete prompt string
    """
    if use_handcrafted:
        categories_text = HANDCRAFTED_STAGE1_CATEGORIES
        rules_text = _format_rules(HANDCRAFTED_STAGE1_RULES)
        examples_text = _format_stage1_examples_handcrafted()
    else:
        # Use generated content
        if not condensed or not examples:
            raise ValueError("Must provide condensed and examples when use_handcrafted=False")

        # Format categories
        categories_lines = []
        for cat in condensed.categories:
            categories_lines.append(f"**{cat.name}**")
            categories_lines.append(cat.short_description)
            for elem in cat.elements:
                categories_lines.append(f"- {elem.name}: {elem.short_description}")
            categories_lines.append("")  # blank line
        categories_text = "\n".join(categories_lines)

        # Rules
        default_rules = rules or HANDCRAFTED_STAGE1_RULES
        rules_text = _format_rules(default_rules)

        # Curate and format examples
        curated_examples = _curate_generated_examples(examples, max_per_category=1)
        examples_text = _format_stage1_examples_generated(curated_examples)

    # Build full prompt
    return STAGE1_TEMPLATE.format(
        comment=comment,
        categories=categories_text,
        rules=rules_text,
        examples=examples_text,
    )


def build_stage1_prompt_function(
    condensed: CondensedTaxonomy,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    rules: Optional[any] = None,  # For compatibility
    use_handcrafted: bool = True,
) -> Callable[[str], str]:
    """
    Build a reusable Stage 1 prompt function.

    This creates a function that takes a comment and returns a prompt.
    All static content is pre-formatted for efficiency.

    Args:
        condensed: CondensedTaxonomy
        examples: Optional ExampleSet or list (needed if use_handcrafted=False)
        rules: Optional rules (for compatibility, not used with handcrafted)
        use_handcrafted: If True, use handcrafted sections (default: True)

    Returns:
        Function that takes a comment string and returns a prompt string

    Example:
        >>> # Use handcrafted (recommended)
        >>> stage1_prompt = build_stage1_prompt_function(condensed, use_handcrafted=True)
        >>>
        >>> # Or use generated (auto-curated)
        >>> stage1_prompt = build_stage1_prompt_function(condensed, examples, use_handcrafted=False)
        >>>
        >>> prompt = stage1_prompt("The conference was great!")
    """
    if use_handcrafted:
        # Pre-format all static sections
        categories_text = HANDCRAFTED_STAGE1_CATEGORIES
        rules_text = _format_rules(HANDCRAFTED_STAGE1_RULES)
        examples_text = _format_stage1_examples_handcrafted()
    else:
        # Use generated content
        if not examples:
            raise ValueError("Must provide examples when use_handcrafted=False")

        # Convert ExampleSet to list if needed
        if isinstance(examples, ExampleSet):
            example_list = examples.examples
        else:
            example_list = examples

        # Format categories
        categories_lines = []
        for cat in condensed.categories:
            categories_lines.append(f"**{cat.name}**")
            categories_lines.append(cat.short_description)
            for elem in cat.elements:
                categories_lines.append(f"- {elem.name}: {elem.short_description}")
            categories_lines.append("")
        categories_text = "\n".join(categories_lines)

        # Rules (use handcrafted as default)
        rules_text = _format_rules(HANDCRAFTED_STAGE1_RULES)

        # Curate and format examples
        curated_examples = _curate_generated_examples(example_list, max_per_category=1)
        examples_text = _format_stage1_examples_generated(curated_examples)

    # Build static template (everything except {comment})
    static_template = STAGE1_TEMPLATE.format(
        comment="{comment}",  # Keep as placeholder
        categories=categories_text,
        rules=rules_text,
        examples=examples_text,
    )

    # Return closure with pre-formatted template
    def stage1_prompt(comment: str) -> str:
        return static_template.format(comment=comment)

    return stage1_prompt


# =============================================================================
# Export Functions
# =============================================================================


def export_stage1_prompt_module(
    condensed: CondensedTaxonomy,
    filepath: str | Path,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    rules: Optional[List[str]] = None,
    use_handcrafted: bool = True,
    function_name: str = "stage1_category_detection_prompt",
) -> Path:
    """
    Export Stage 1 prompt as a standalone Python module.

    Args:
        condensed: CondensedTaxonomy
        filepath: Output path for the .py file
        examples: Optional examples (needed if use_handcrafted=False)
        rules: Optional rules
        use_handcrafted: If True, use handcrafted sections (default: True)
        function_name: Name for the prompt function

    Returns:
        Path to the generated module
    """
    # Build prompt components
    if use_handcrafted:
        categories_text = HANDCRAFTED_STAGE1_CATEGORIES
        rules_text = _format_rules(HANDCRAFTED_STAGE1_RULES)
        examples_text = _format_stage1_examples_handcrafted()
    else:
        if not examples:
            raise ValueError("Must provide examples when use_handcrafted=False")

        # Convert ExampleSet to list if needed
        if isinstance(examples, ExampleSet):
            example_list = examples.examples
        else:
            example_list = examples

        # Format categories
        categories_lines = []
        for cat in condensed.categories:
            categories_lines.append(f"**{cat.name}**")
            categories_lines.append(cat.short_description)
            for elem in cat.elements:
                categories_lines.append(f"- {elem.name}: {elem.short_description}")
            categories_lines.append("")
        categories_text = "\n".join(categories_lines)

        # Rules
        default_rules = rules or HANDCRAFTED_STAGE1_RULES
        rules_text = _format_rules(default_rules)

        # Curate and format examples
        curated_examples = _curate_generated_examples(example_list, max_per_category=1)
        examples_text = _format_stage1_examples_generated(curated_examples)

    # Get category names for metadata
    category_names = [cat.name for cat in condensed.categories]

    # Build module content
    source_desc = "handcrafted examples" if use_handcrafted else "curated generated examples"

    module_content = f'''# =============================================================================
# Stage 1: Category Detection
# =============================================================================
#
# Identifies which categories of feedback are present in a conference comment.
#
# AUTO-GENERATED from {source_desc}
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

CATEGORIES = [
    {", ".join(f'"{name}"' for name in category_names)}
]


def {function_name}(comment: str) -> str:
    """
    Generate Stage 1 category detection prompt.
    
    Args:
        comment: The conference feedback comment to analyze
        
    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in a conference attendee comment.

COMMENT TO ANALYZE:
{{comment}}

---

CATEGORIES AND THEIR SCOPE:

{categories_text}

---

CLASSIFICATION RULES:

{rules_text}

---

EXAMPLES:

{examples_text}

---

Analyze the comment and return ONLY valid JSON."""


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
# Stats and Preview Functions
# =============================================================================


def get_stage1_prompt_stats(
    condensed: CondensedTaxonomy,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    use_handcrafted: bool = True,
) -> dict:
    """Get statistics about the Stage 1 prompt."""
    prompt = build_stage1_prompt_string(
        comment="[SAMPLE COMMENT]",
        condensed=condensed,
        examples=examples,
        use_handcrafted=use_handcrafted,
    )

    return {
        "total_length": len(prompt),
        "num_categories": len(condensed.categories),
        "num_examples": len(HANDCRAFTED_STAGE1_EXAMPLES) if use_handcrafted else "varies",
        "source": "handcrafted" if use_handcrafted else "generated (auto-curated)",
    }


def print_stage1_prompt_preview(
    condensed: CondensedTaxonomy,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    use_handcrafted: bool = True,
    max_chars: int = 500,
) -> None:
    """Print a preview of the Stage 1 prompt."""
    prompt = build_stage1_prompt_string(
        comment="The conference was amazing!",
        condensed=condensed,
        examples=examples,
        use_handcrafted=use_handcrafted,
    )

    stats = get_stage1_prompt_stats(condensed, examples, use_handcrafted)

    print("=" * 70)
    print("STAGE 1 PROMPT PREVIEW")
    print("=" * 70)
    print(f"Source: {stats['source']}")
    print(f"Length: {stats['total_length']:,} chars")
    print(f"Categories: {stats['num_categories']}")
    print(f"Examples: {stats['num_examples']}")
    print("\nFirst {max_chars} characters:")
    print("-" * 70)
    print(prompt[:max_chars] + "...")
