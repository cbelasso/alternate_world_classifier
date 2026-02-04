"""
Stage 2 Prompt Builder

Assembles Stage 2 (element extraction) prompts from condensed taxonomy and examples.

This is pure Python templating - NO LLM required!

Usage:
    from classifier.prompts import build_stage2_prompt_functions
    from classifier.taxonomy.condenser import load_condensed
    from classifier.taxonomy.example_generator import load_examples

    # Load artifacts
    condensed = load_condensed("artifacts/condensed.json")
    examples = load_examples("artifacts/examples.json")

    # Build prompt functions (pure Python) - with handcrafted examples
    stage2_prompts = build_stage2_prompt_functions(condensed, use_handcrafted=True)

    # Or with generated examples (automatically curated)
    stage2_prompts = build_stage2_prompt_functions(condensed, examples, use_handcrafted=False)

    # Use at runtime
    prompt = stage2_prompts["People"]("The staff was helpful!")
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..taxonomy.condenser import CondensedCategory, CondensedTaxonomy
from ..taxonomy.example_generator import ClassificationExample, ExampleSet
from .base import format_stage2_example

# =============================================================================
# Handcrafted Stage 2 Content (Extracted from Proven Standalone Script)
# =============================================================================

HANDCRAFTED_STAGE2_SECTIONS = {
    "Attendee Engagement & Interaction": {
        "elements": """**Community**: Sense of belonging, community spirit, feeling welcomed, being part of a group, supportive environment
**Knowledge Exchange**: Sharing experiences with peers, learning from others' implementations, collaborative problem-solving, best practice sharing
**Networking**: Meeting new people, professional connections, peer discussions, contact exchange, relationship building
**Social Events**: Gala dinners, receptions, evening events, informal gatherings, social activities outside sessions""",
        "rules": [
            "Extract the EXACT excerpt from the comment that relates to each element.",
            "Each excerpt should be classified to ONE element only.",
            "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
            '"Community" = feeling of belonging; "Networking" = act of meeting/connecting.',
            '"Knowledge Exchange" = peer-to-peer learning; different from formal presentations.',
        ],
        "examples": [
            {
                "comment": "The community is so supportive and open to sharing their knowledge.",
                "output": '{{"classifications": [{{"excerpt": "The community is so supportive", "reasoning": "Expresses feeling of supportive community environment", "element": "Community", "sentiment": "positive"}}, {{"excerpt": "open to sharing their knowledge", "reasoning": "References peer knowledge sharing", "element": "Knowledge Exchange", "sentiment": "positive"}}]}}',
            },
            {
                "comment": "Change the networking session. Nobody from Explorance showed up at my table. Sitting there with 1 other person was awkward.",
                "output": '{{"classifications": [{{"excerpt": "Change the networking session. Nobody from Explorance showed up at my table. Sitting there with 1 other person was awkward", "reasoning": "Negative experience with networking session format", "element": "Networking", "sentiment": "negative"}}]}}',
            },
            {
                "comment": "I would have liked to go to the Marina for the gala dinner but I understand that would have been challenging.",
                "output": '{{"classifications": [{{"excerpt": "I would have liked to go to the Marina for the gala dinner", "reasoning": "Suggestion about gala dinner venue", "element": "Social Events", "sentiment": "mixed"}}]}}',
            },
            {
                "comment": "It only took an hour to go from brand new user to genuinely feeling like a part of the Blue community.",
                "output": '{{"classifications": [{{"excerpt": "genuinely feeling like a part of the Blue community", "reasoning": "Expresses sense of belonging to community", "element": "Community", "sentiment": "positive"}}]}}',
            },
        ],
    },
    "Event Logistics & Infrastructure": {
        "elements": """**Conference Application/Software**: Mobile apps, event platforms, Bluepulse app, digital tools for attendees, software for participation
**Conference Venue**: Location, rooms, facilities, seating, temperature, accessibility, physical space
**Food/Beverages**: Meals, snacks, drinks, catering quality, dietary options, refreshments
**Hotel**: Accommodation, lodging, hotel arrangements, room quality
**Technological Tools**: A/V equipment, microphones, projectors, screens, tech setup for presentations
**Transportation**: Getting to/from venue, shuttles, parking, travel arrangements, logistics of movement
**Wi-Fi**: Internet connectivity, network access, connection quality""",
        "rules": [
            "Extract the EXACT excerpt from the comment that relates to each element.",
            "Each excerpt should be classified to ONE element only.",
            "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
            '"Conference Application/Software" = apps for attendees; "Technological Tools" = A/V equipment for sessions.',
        ],
        "examples": [
            {
                "comment": "Having the Bluepulse app information earlier. Those without a Smart phone were not able to evaluate the sessions.",
                "output": '{{"classifications": [{{"excerpt": "Having the Bluepulse app information earlier. Those without a Smart phone were not able to evaluate the sessions", "reasoning": "Feedback about conference app accessibility and communication", "element": "Conference Application/Software", "sentiment": "negative"}}]}}',
            },
            {
                "comment": "Wish there was hand sanitizer more available around the conference.",
                "output": '{{"classifications": [{{"excerpt": "Wish there was hand sanitizer more available around the conference", "reasoning": "Request for venue amenity", "element": "Conference Venue", "sentiment": "negative"}}]}}',
            },
            {
                "comment": "have drinks available, even if a cash bar, at events.",
                "output": '{{"classifications": [{{"excerpt": "have drinks available, even if a cash bar, at events", "reasoning": "Request for beverage availability", "element": "Food/Beverages", "sentiment": "negative"}}]}}',
            },
            {
                "comment": "The WiFi kept dropping and made it hard to follow along.",
                "output": '{{"classifications": [{{"excerpt": "The WiFi kept dropping and made it hard to follow along", "reasoning": "Complaint about internet connectivity", "element": "Wi-Fi", "sentiment": "negative"}}]}}',
            },
        ],
    },
    "Event Operations & Management": {
        "elements": """**Conference**: General event organization, overall management, event quality, hospitality, general conference experience
**Conference Registration**: Sign-up process, check-in, badge pickup, registration system, enrollment
**Conference Scheduling**: Session timing, agenda, time management, scheduling conflicts, program structure
**Messaging & Awareness**: Communication, announcements, information clarity, signage, pre-event information""",
        "rules": [
            "Extract the EXACT excerpt from the comment that relates to each element.",
            "Each excerpt should be classified to ONE element only.",
            "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
            'General praise like "great conference" or "well organized" → Conference element.',
            "Comments about session timing or agenda structure → Conference Scheduling.",
        ],
        "examples": [
            {
                "comment": "An excellent, informative, and well-organised event.",
                "output": '{{"classifications": [{{"excerpt": "An excellent, informative, and well-organised event", "reasoning": "General positive feedback about event organization", "element": "Conference", "sentiment": "positive"}}]}}',
            },
            {
                "comment": "The registration process was rather slow, I did sign up quite close to the date but I never even got an invoice.",
                "output": '{{"classifications": [{{"excerpt": "The registration process was rather slow, I did sign up quite close to the date but I never even got an invoice", "reasoning": "Complaint about registration process and invoicing", "element": "Conference Registration", "sentiment": "negative"}}]}}',
            },
            {
                "comment": "More repeated sessions which, while not bad, the time could have been spent doing more hands-on activities.",
                "output": '{{"classifications": [{{"excerpt": "More repeated sessions which, while not bad, the time could have been spent doing more hands-on activities", "reasoning": "Feedback about session scheduling and program structure", "element": "Conference Scheduling", "sentiment": "mixed"}}]}}',
            },
            {
                "comment": "BNG 2022 has set the bar high in terms of hospitality. They go beyond their call for duty.",
                "output": '{{"classifications": [{{"excerpt": "BNG 2022 has set the bar high in terms of hospitality. They go beyond their call for duty", "reasoning": "Praise for conference hospitality and management", "element": "Conference", "sentiment": "positive"}}]}}',
            },
        ],
    },
    "Learning & Content Delivery": {
        "elements": """**Demonstration**: Live demos, product showcases, hands-on examples, showing how things work
**Gained Knowledge**: What attendees learned, takeaways, actionable insights, things to implement
**Open Discussion**: Q&A sessions, audience participation, interactive discussions, roundtables
**Panel Discussions**: Panel format sessions, multi-speaker discussions, panel quality
**Presentations**: Individual talks, keynotes, speaker presentations, talk quality and content
**Resources/Materials**: Handouts, slides, documentation, learning materials, presentation copies
**Session/Workshop**: Breakout sessions, workshops, training sessions, hands-on learning
**Topics**: Subject matter, themes, content relevance, topic selection, what was covered""",
        "rules": [
            "Extract the EXACT excerpt from the comment that relates to each element.",
            "Each excerpt should be classified to ONE element only.",
            "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
            '"Presentations" = quality of talks; "Topics" = what subjects were covered.',
            '"Session/Workshop" = format of learning; "Gained Knowledge" = what was learned.',
            'Requests for "more workshops" or "hands-on sessions" → Session/Workshop.',
        ],
        "examples": [
            {
                "comment": "The presentations were better than the panel discussion. Better panelists would be preferred.",
                "output": '{{"classifications": [{{"excerpt": "The presentations were better than the panel discussion", "reasoning": "Comparison of presentation vs panel format quality", "element": "Presentations", "sentiment": "positive"}}, {{"excerpt": "Better panelists would be preferred", "reasoning": "Criticism of panel discussion quality", "element": "Panel Discussions", "sentiment": "negative"}}]}}',
            },
            {
                "comment": "Provide presentation materials hard or soft copies.",
                "output": '{{"classifications": [{{"excerpt": "Provide presentation materials hard or soft copies", "reasoning": "Request for presentation materials/handouts", "element": "Resources/Materials", "sentiment": "negative"}}]}}',
            },
            {
                "comment": "More technical workshops would be great. The presentations were good.",
                "output": '{{"classifications": [{{"excerpt": "More technical workshops would be great", "reasoning": "Request for more workshop sessions", "element": "Session/Workshop", "sentiment": "mixed"}}, {{"excerpt": "The presentations were good", "reasoning": "Positive feedback on presentations", "element": "Presentations", "sentiment": "positive"}}]}}',
            },
            {
                "comment": "I came away with a significant 'to do' list which will help us leverage insights collected from our students.",
                "output": '{{"classifications": [{{"excerpt": "I came away with a significant \'to do\' list which will help us leverage insights collected from our students", "reasoning": "Actionable takeaways gained from conference", "element": "Gained Knowledge", "sentiment": "positive"}}]}}',
            },
            {
                "comment": "I wonder if some round table discussions would help.",
                "output": '{{"classifications": [{{"excerpt": "I wonder if some round table discussions would help", "reasoning": "Suggestion for open discussion format", "element": "Open Discussion", "sentiment": "mixed"}}]}}',
            },
            {
                "comment": "high quality papers and mix of topics from increasing response rates through to machine learning applied to text analytics",
                "output": '{{"classifications": [{{"excerpt": "high quality papers and mix of topics from increasing response rates through to machine learning applied to text analytics", "reasoning": "Praise for topic variety and quality", "element": "Topics", "sentiment": "positive"}}]}}',
            },
        ],
    },
    "People": {
        "elements": """**Conference Staff**: Organizers, volunteers, support staff, event team, Explorance team (when mentioned as organizers/hosts)
**Experts/Consultants**: Industry experts, product specialists, consultants, Explorance experts (when mentioned for their expertise)
**Participants/Attendees**: Fellow attendees, other conference-goers, peers at the conference
**Speakers/Presenters**: Keynote speakers, session presenters, panelists, people giving talks
**Unspecified Person**: References to people without clear role identification""",
        "rules": [
            "Extract the EXACT excerpt from the comment that relates to each element.",
            "Each excerpt should be classified to ONE element only.",
            "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
            '"Explorance team" as hosts/organizers → Conference Staff.',
            '"Explorance experts" for knowledge/consulting → Experts/Consultants.',
            '"Blue users" or "community members" → Participants/Attendees.',
            'Named speakers or "the presenter" → Speakers/Presenters.',
        ],
        "examples": [
            {
                "comment": "The Explorance staff are so genuine, knowledgeable, and accessible.",
                "output": '{{"classifications": [{{"excerpt": "The Explorance staff are so genuine, knowledgeable, and accessible", "reasoning": "Praise for conference staff qualities", "element": "Conference Staff", "sentiment": "positive"}}]}}',
            },
            {
                "comment": "The ability to talk not only to Explorance Experts but to network with community members made this valuable.",
                "output": '{{"classifications": [{{"excerpt": "talk not only to Explorance Experts", "reasoning": "Reference to product experts", "element": "Experts/Consultants", "sentiment": "positive"}}, {{"excerpt": "network with community members", "reasoning": "Reference to fellow attendees", "element": "Participants/Attendees", "sentiment": "positive"}}]}}',
            },
            {
                "comment": "The Blue users you'll meet are smart, creative, and always willing to share and collaborate.",
                "output": '{{"classifications": [{{"excerpt": "The Blue users you\'ll meet are smart, creative, and always willing to share and collaborate", "reasoning": "Praise for fellow attendees", "element": "Participants/Attendees", "sentiment": "positive"}}]}}',
            },
            {
                "comment": "some speakers appeared to have changed or modified the presentation from the original abstract",
                "output": '{{"classifications": [{{"excerpt": "some speakers appeared to have changed or modified the presentation from the original abstract", "reasoning": "Criticism of speakers deviating from abstract", "element": "Speakers/Presenters", "sentiment": "negative"}}]}}',
            },
            {
                "comment": "Nobody from Explorance showed up at my table.",
                "output": '{{"classifications": [{{"excerpt": "Nobody from Explorance showed up at my table", "reasoning": "Complaint about staff absence at networking", "element": "Conference Staff", "sentiment": "negative"}}]}}',
            },
        ],
    },
}


# =============================================================================
# Prompt Template
# =============================================================================

STAGE2_TEMPLATE = """You are an expert conference feedback analyzer. Extract specific feedback related to {category} from this comment.

COMMENT TO ANALYZE:
{{comment}}

---

ELEMENTS TO IDENTIFY:

{elements}

---

CLASSIFICATION RULES:

{rules}

---

EXAMPLES:

{examples}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to {category}, return {{"classifications": []}}."""


# =============================================================================
# Example Curation Helper
# =============================================================================


def _get_stage2_examples_for_category(
    examples: ExampleSet | List[ClassificationExample],
    category_name: str,
) -> List[ClassificationExample]:
    """
    Get examples relevant to a specific category.

    Filters to examples where:
    1. The category appears in categories_present, OR
    2. Element details contain elements from this category

    Args:
        examples: ExampleSet or list of examples
        category_name: Name of the category

    Returns:
        Filtered list of examples
    """
    if isinstance(examples, ExampleSet):
        example_list = examples.examples
    else:
        example_list = examples

    relevant = []
    for ex in example_list:
        # Check if category is in categories_present
        if category_name in ex.categories_present:
            relevant.append(ex)
            continue

        # Check if any element_details match this category
        for detail in ex.element_details:
            if detail.category == category_name or ex.source_category == category_name:
                relevant.append(ex)
                break

    return relevant


def _curate_stage2_examples(
    examples: List[ClassificationExample],
    category_name: str,
    max_examples: int = 6,
) -> List[ClassificationExample]:
    """
    Curate Stage 2 examples to avoid bloat.

    Strategy:
    - Prioritize examples with multiple elements (show diversity)
    - Limit to max_examples total
    - Ensure variety of sentiments

    Args:
        examples: List of examples for this category
        category_name: Category name
        max_examples: Maximum examples to include

    Returns:
        Curated list
    """
    # Sort by number of element_details (prefer multi-element examples)
    sorted_examples = sorted(
        examples,
        key=lambda ex: len(
            [
                d
                for d in ex.element_details
                if d.category == category_name or ex.source_category == category_name
            ]
        ),
        reverse=True,
    )

    return sorted_examples[:max_examples]


# =============================================================================
# Formatting Helpers
# =============================================================================


def _format_stage2_examples_handcrafted(category: str) -> str:
    """Format handcrafted examples for a category."""
    examples = HANDCRAFTED_STAGE2_SECTIONS[category]["examples"]

    formatted = []
    for ex in examples:
        formatted.append(f'Comment: "{ex["comment"]}"\n{ex["output"]}')

    return "\n\n".join(formatted)


def _format_stage2_examples_generated(
    examples: List[ClassificationExample],
    category: str,
) -> str:
    """Format generated examples for a category."""
    if not examples:
        return "(No examples available)"

    formatted = []
    for ex in examples:
        # Filter element_details to this category
        relevant_details = [
            d
            for d in ex.element_details
            if d.category == category or ex.source_category == category
        ]

        if not relevant_details:
            continue

        # Build classifications array
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

        import json

        output_json = json.dumps({"classifications": classifications})
        formatted.append(f'Comment: "{ex.comment}"\n{output_json}')

    return "\n\n".join(formatted) if formatted else "(No examples available)"


def _format_rules(rules: List[str]) -> str:
    """Format rules as numbered list."""
    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))


# =============================================================================
# Main Builder Functions
# =============================================================================


def build_stage2_prompt_string(
    comment: str,
    category: str,
    condensed: Optional[CondensedTaxonomy] = None,
    examples: Optional[List[ClassificationExample]] = None,
    use_handcrafted: bool = True,
) -> str:
    """
    Build a complete Stage 2 prompt for a specific comment and category.

    Args:
        comment: The conference feedback comment
        category: Category name
        condensed: CondensedTaxonomy (optional, for generated content)
        examples: List of examples (optional, for generated content)
        use_handcrafted: If True, use handcrafted sections (default)

    Returns:
        Complete prompt string
    """
    if use_handcrafted:
        if category not in HANDCRAFTED_STAGE2_SECTIONS:
            raise ValueError(f"No handcrafted content for category: {category}")

        sections = HANDCRAFTED_STAGE2_SECTIONS[category]
        elements_text = sections["elements"]
        rules_text = _format_rules(sections["rules"])
        examples_text = _format_stage2_examples_handcrafted(category)
    else:
        # Use generated content
        if not condensed or not examples:
            raise ValueError("Must provide condensed and examples when use_handcrafted=False")

        # Get category from condensed
        cat_obj = condensed.get_category(category)
        if not cat_obj:
            raise ValueError(f"Category not found: {category}")

        # Format elements
        elements_text = "\n".join(
            f"**{elem.name}**: {elem.short_description}" for elem in cat_obj.elements
        )

        # Use default rules (could be loaded from rules.json)
        rules_text = _format_rules(
            [
                "Extract the EXACT excerpt from the comment that relates to each element.",
                "Each excerpt should be classified to ONE element only.",
                "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
            ]
        )

        # Format examples
        examples_text = _format_stage2_examples_generated(examples, category)

    # Build full prompt
    template = STAGE2_TEMPLATE.format(
        category=category,
        elements=elements_text,
        rules=rules_text,
        examples=examples_text,
    )

    return template.format(comment=comment)


def build_stage2_prompt_functions(
    condensed: CondensedTaxonomy,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    rules: Optional[any] = None,  # For compatibility with existing code
    use_handcrafted: bool = True,
    max_examples_per_category: int = 6,
) -> Dict[str, Callable[[str], str]]:
    """
    Build reusable Stage 2 prompt functions for all categories.

    This creates a dict of functions that take a comment and return a prompt.

    Args:
        condensed: CondensedTaxonomy
        examples: Optional ExampleSet or list (needed if use_handcrafted=False)
        rules: Optional rules (for compatibility, not used with handcrafted)
        use_handcrafted: If True, use handcrafted sections (default: True)
        max_examples_per_category: Max examples per category if using generated

    Returns:
        Dict mapping category name to prompt function

    Example:
        >>> # Use handcrafted (recommended)
        >>> stage2_prompts = build_stage2_prompt_functions(condensed, use_handcrafted=True)
        >>>
        >>> # Or use generated (auto-curated)
        >>> stage2_prompts = build_stage2_prompt_functions(
        ...     condensed, examples, use_handcrafted=False, max_examples_per_category=6
        ... )
        >>>
        >>> prompt = stage2_prompts["People"]("The staff was helpful!")
    """
    prompt_functions = {}

    for category in condensed.categories:
        cat_name = category.name

        if use_handcrafted:
            # Pre-build static sections
            if cat_name not in HANDCRAFTED_STAGE2_SECTIONS:
                print(f"⚠️  No handcrafted content for '{cat_name}', skipping")
                continue

            sections = HANDCRAFTED_STAGE2_SECTIONS[cat_name]
            elements_text = sections["elements"]
            rules_text = _format_rules(sections["rules"])
            examples_text = _format_stage2_examples_handcrafted(cat_name)
        else:
            # Use generated content
            if not examples:
                raise ValueError("Must provide examples when use_handcrafted=False")

            # Get and curate examples
            relevant_examples = _get_stage2_examples_for_category(examples, cat_name)
            curated_examples = _curate_stage2_examples(
                relevant_examples, cat_name, max_examples_per_category
            )

            # Format elements
            elements_text = "\n".join(
                f"**{elem.name}**: {elem.short_description}" for elem in category.elements
            )

            # Default rules
            default_rules = [
                "Extract the EXACT excerpt from the comment that relates to each element.",
                "Each excerpt should be classified to ONE element only.",
                "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
            ]
            rules_text = _format_rules(default_rules)

            # Format examples
            examples_text = _format_stage2_examples_generated(curated_examples, cat_name)

        # Build static template
        static_template = STAGE2_TEMPLATE.format(
            category=cat_name,
            elements=elements_text,
            rules=rules_text,
            examples=examples_text,
        )

        # Create closure
        def make_prompt_fn(template):
            def prompt_fn(comment: str) -> str:
                return template.format(comment=comment)

            return prompt_fn

        prompt_functions[cat_name] = make_prompt_fn(static_template)

    return prompt_functions


# =============================================================================
# Export Functions
# =============================================================================


def export_stage2_prompt_module(
    category: str,
    filepath: str | Path,
    condensed: CondensedTaxonomy,
    examples: Optional[ExampleSet | List[ClassificationExample]] = None,
    rules: Optional[List[str]] = None,
    use_handcrafted: bool = True,
    max_examples: int = 6,
) -> Path:
    """
    Export a Stage 2 prompt as a standalone Python module.

    Args:
        category: Category name
        filepath: Output path for the .py file
        condensed: CondensedTaxonomy
        examples: Optional examples (needed if use_handcrafted=False)
        rules: Optional rules
        use_handcrafted: If True, use handcrafted sections (default: True)
        max_examples: Max examples if using generated

    Returns:
        Path to the generated module
    """
    # Build prompt components
    if use_handcrafted:
        if category not in HANDCRAFTED_STAGE2_SECTIONS:
            raise ValueError(f"No handcrafted content for category: {category}")

        sections = HANDCRAFTED_STAGE2_SECTIONS[category]
        elements_text = sections["elements"]
        rules_text = _format_rules(sections["rules"])
        examples_text = _format_stage2_examples_handcrafted(category)
    else:
        if not examples:
            raise ValueError("Must provide examples when use_handcrafted=False")

        # Get category object
        cat_obj = condensed.get_category(category)
        if not cat_obj:
            raise ValueError(f"Category not found: {category}")

        # Get and curate examples
        relevant_examples = _get_stage2_examples_for_category(examples, category)
        curated_examples = _curate_stage2_examples(relevant_examples, category, max_examples)

        # Format elements
        elements_text = "\n".join(
            f"**{elem.name}**: {elem.short_description}" for elem in cat_obj.elements
        )

        # Rules
        default_rules = rules or [
            "Extract the EXACT excerpt from the comment that relates to each element.",
            "Each excerpt should be classified to ONE element only.",
            "Sentiment: positive (praise), negative (criticism), neutral (observation), mixed (both positive and negative).",
        ]
        rules_text = _format_rules(default_rules)

        # Examples
        examples_text = _format_stage2_examples_generated(curated_examples, category)

    # Sanitize function name
    import re

    func_name_base = re.sub(r"[^a-zA-Z0-9_]", "_", category.lower()).strip("_")
    func_name = f"stage2_{func_name_base}_prompt"

    # Get element names
    cat_obj = condensed.get_category(category)
    element_names = [e.name for e in cat_obj.elements] if cat_obj else []

    # Build module content
    source_desc = (
        "handcrafted examples"
        if use_handcrafted
        else f"curated generated examples (max {max_examples})"
    )

    module_content = f'''# =============================================================================
# Stage 2: {category}
# =============================================================================
#
# Extracts feedback elements related to {category}.
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

CATEGORY_NAME = "{category}"

ELEMENTS = [
    {", ".join(f'"{name}"' for name in element_names)}
]


def {func_name}(comment: str) -> str:
    """
    Generate Stage 2 element extraction prompt for {category}.
    
    Args:
        comment: The conference feedback comment to analyze
        
    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Extract specific feedback related to {category} from this comment.

COMMENT TO ANALYZE:
{{comment}}

---

ELEMENTS TO IDENTIFY:

{elements_text}

---

CLASSIFICATION RULES:

{rules_text}

---

EXAMPLES:

{examples_text}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to {category}, return {{"classifications": []}}."""


# Convenience alias
PROMPT = {func_name}
'''

    # Write to file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(module_content)

    return filepath


# =============================================================================
# Utility Functions
# =============================================================================


def format_elements_section(category: CondensedCategory) -> str:
    """Format elements section for a category (for generated content)."""
    return "\n".join(f"**{elem.name}**: {elem.short_description}" for elem in category.elements)


def format_stage2_examples_section(
    examples: List[ClassificationExample],
    category: str,
) -> str:
    """Format examples section for Stage 2 (wrapper for compatibility)."""
    return _format_stage2_examples_generated(examples, category)


def get_stage2_examples_for_category(
    examples: ExampleSet | List[ClassificationExample],
    category: str,
) -> List[ClassificationExample]:
    """Get examples for a category (wrapper for compatibility)."""
    return _get_stage2_examples_for_category(examples, category)
