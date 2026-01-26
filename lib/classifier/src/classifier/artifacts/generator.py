"""
Artifact Generator

Generates content for YAML artifact files using LLM.

This fills in the scaffolded YAML files with:
- Condensed descriptions (from verbose definitions)
- Disambiguation rules
- Classification examples

Usage:
    from classifier.artifacts import generate_artifact_content

    # Generate all content for artifacts
    generate_artifact_content(
        artifacts_dir="artifacts/",
        schema_path="schema.json",
        processor=processor,
    )
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
import yaml

# =============================================================================
# Response Schemas for LLM
# =============================================================================


class DescriptionResponse(BaseModel):
    """Schema for description generation."""

    description: str


class RulesResponse(BaseModel):
    """Schema for rules generation."""

    rules: List[str]


class CategoryExampleItem(BaseModel):
    """Single category example."""

    comment: str
    reasoning: str


class CategoryExamplesResponse(BaseModel):
    """Schema for category examples generation."""

    examples: List[CategoryExampleItem]


class ElementExampleItem(BaseModel):
    """Single element example."""

    comment: str
    excerpt: str
    sentiment: str
    reasoning: str


class ElementExamplesResponse(BaseModel):
    """Schema for element examples generation."""

    examples: List[ElementExampleItem]


class AttributeExampleItem(BaseModel):
    """Single attribute example."""

    excerpt: str
    sentiment: str
    reasoning: str


class AttributeExamplesResponse(BaseModel):
    """Schema for attribute examples generation."""

    examples: List[AttributeExampleItem]


# =============================================================================
# Utilities
# =============================================================================


def sanitize_folder_name(name: str) -> str:
    """Convert display name to folder name."""
    result = name.lower()
    result = result.replace("&", "and")
    result = result.replace("/", "_")
    result = result.replace(" ", "_")
    result = result.replace("-", "_")
    result = re.sub(r"[^a-z0-9_]", "", result)
    result = re.sub(r"_+", "_", result)
    result = result.strip("_")
    return result


def load_yaml(filepath: Path) -> dict:
    """Load YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, filepath: Path, preserve_header: bool = True) -> None:
    """Save YAML file, preserving header comments if present."""
    header = ""
    if preserve_header and filepath.exists():
        with open(filepath, "r") as f:
            lines = f.readlines()
            header_lines = []
            for line in lines:
                if line.startswith("#"):
                    header_lines.append(line)
                else:
                    break
            if header_lines:
                header = "".join(header_lines)

    content = header
    content += yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# Description Generation
# =============================================================================


CONDENSE_PROMPT = """You are helping create a classification system. Given a verbose definition, create a SHORT, CLEAR description suitable for an LLM classification prompt.

Requirements:
- Maximum 2 sentences
- Focus on WHAT to look for in text
- Use concrete keywords/phrases when possible
- Avoid jargon and abstract language

Verbose definition:
{definition}

Respond with a JSON object containing the description."""


def generate_descriptions(
    artifacts_dir: Path,
    schema: dict,
    processor: Any,
    verbose: bool = True,
) -> int:
    """
    Generate condensed descriptions for all YAML files.

    Args:
        artifacts_dir: Path to artifacts directory
        schema: Loaded schema dict
        processor: LLM processor for generation
        verbose: Print progress

    Returns:
        Number of descriptions generated
    """
    count = 0
    prompts = []
    targets = []  # (filepath, yaml_data, label)

    # Collect all items needing descriptions
    for category in schema.get("children", []):
        cat_name = category.get("name", "")
        cat_def = category.get("definition", "")
        cat_folder = sanitize_folder_name(cat_name)
        cat_yaml_path = artifacts_dir / cat_folder / "_category.yaml"

        if cat_yaml_path.exists() and cat_def:
            cat_data = load_yaml(cat_yaml_path)
            # Only generate if it's still a placeholder or truncated
            if cat_data.get("description", "").endswith("...") or cat_data.get(
                "description", ""
            ).startswith("#"):
                prompts.append(CONDENSE_PROMPT.format(definition=cat_def))
                targets.append((cat_yaml_path, cat_data, cat_name))

        for element in category.get("children", []):
            elem_name = element.get("name", "")
            elem_def = element.get("definition", "")
            elem_folder = sanitize_folder_name(elem_name)
            elem_yaml_path = artifacts_dir / cat_folder / elem_folder / "_element.yaml"

            if elem_yaml_path.exists() and elem_def:
                elem_data = load_yaml(elem_yaml_path)
                if elem_data.get("description", "").endswith("...") or elem_data.get(
                    "description", ""
                ).startswith("#"):
                    prompts.append(CONDENSE_PROMPT.format(definition=elem_def))
                    targets.append((elem_yaml_path, elem_data, f"{cat_name} > {elem_name}"))

            for attribute in element.get("children", []):
                attr_name = attribute.get("name", "")
                attr_def = attribute.get("definition", "")
                attr_filename = sanitize_folder_name(attr_name) + ".yaml"
                attr_yaml_path = artifacts_dir / cat_folder / elem_folder / attr_filename

                if attr_yaml_path.exists() and attr_def:
                    attr_data = load_yaml(attr_yaml_path)
                    if attr_data.get("description", "").endswith("...") or attr_data.get(
                        "description", ""
                    ).startswith("#"):
                        prompts.append(CONDENSE_PROMPT.format(definition=attr_def))
                        targets.append(
                            (
                                attr_yaml_path,
                                attr_data,
                                f"{cat_name} > {elem_name} > {attr_name}",
                            )
                        )

    if not prompts:
        if verbose:
            print("  No descriptions need generation")
        return 0

    if verbose:
        print(f"  Generating {len(prompts)} descriptions...")

    # Process with schema
    responses = processor.process_with_schema(
        prompts=prompts,
        schema=DescriptionResponse,
        guided_config={"temperature": 0.3, "max_tokens": 200},
    )

    results = processor.parse_results_with_schema(
        schema=DescriptionResponse,
        responses=responses,
        validate=True,
    )

    # Update YAML files
    for (filepath, yaml_data, label), result in zip(targets, results):
        if result and result.description:
            yaml_data["description"] = result.description.strip()
            save_yaml(yaml_data, filepath)
            count += 1
            if verbose:
                print(f"    ✓ {label}")

    return count


# =============================================================================
# Rules Generation
# =============================================================================


CATEGORY_RULES_PROMPT = """You are creating disambiguation rules for a text classification system.

Category: {category_name}
Description: {category_description}

Other categories in the system:
{other_categories}

Generate 2-4 SHORT rules that help distinguish this category from others.
Focus on:
- Keywords/phrases that indicate this category
- What NOT to confuse with similar categories
- Edge cases

Respond with a JSON object containing a "rules" array of strings."""


ELEMENT_RULES_PROMPT = """You are creating disambiguation rules for a text classification system.

Category: {category_name}
Element: {element_name}
Description: {element_description}

Other elements in this category:
{other_elements}

Attributes under this element:
{attributes}

Generate 2-4 SHORT rules that help:
1. Distinguish this element from other elements in the category
2. Guide classification to the right attributes

Respond with a JSON object containing a "rules" array of strings."""


ATTRIBUTE_RULES_PROMPT = """You are creating disambiguation rules for a text classification system.

Category: {category_name}
Element: {element_name}
Attribute: {attribute_name}
Description: {attribute_description}

Other attributes under this element:
{other_attributes}

Generate 2-3 SHORT rules that help distinguish this attribute from the other attributes listed.
Focus on specific keywords, phrases, or patterns.

Respond with a JSON object containing a "rules" array of strings."""


def generate_rules(
    artifacts_dir: Path,
    schema: dict,
    processor: Any,
    verbose: bool = True,
    include_attributes: bool = True,
) -> int:
    """
    Generate disambiguation rules for categories, elements, and optionally attributes.

    Args:
        artifacts_dir: Path to artifacts directory
        schema: Loaded schema dict
        processor: LLM processor
        verbose: Print progress
        include_attributes: Whether to generate attribute-level rules (can be slow)

    Returns:
        Number of rule sets generated
    """
    count = 0
    prompts = []
    targets = []

    categories = schema.get("children", [])
    cat_names = [c.get("name", "") for c in categories]

    # Category rules
    for category in categories:
        cat_name = category.get("name", "")
        cat_folder = sanitize_folder_name(cat_name)
        cat_yaml_path = artifacts_dir / cat_folder / "_category.yaml"

        if cat_yaml_path.exists():
            cat_data = load_yaml(cat_yaml_path)

            # Check if rules are still placeholders (all start with #)
            rules = cat_data.get("rules", [])
            is_placeholder = not rules or all(str(r).startswith("#") for r in rules)

            if is_placeholder:
                other_cats = [c for c in cat_names if c != cat_name]
                prompt = CATEGORY_RULES_PROMPT.format(
                    category_name=cat_name,
                    category_description=cat_data.get("description", ""),
                    other_categories="\n".join(f"- {c}" for c in other_cats[:10]),
                )
                prompts.append(prompt)
                targets.append((cat_yaml_path, cat_data, f"Category: {cat_name}"))

    # Element rules
    for category in categories:
        cat_name = category.get("name", "")
        cat_folder = sanitize_folder_name(cat_name)
        elements = category.get("children", [])
        elem_names = [e.get("name", "") for e in elements]

        for element in elements:
            elem_name = element.get("name", "")
            elem_folder = sanitize_folder_name(elem_name)
            elem_yaml_path = artifacts_dir / cat_folder / elem_folder / "_element.yaml"

            if elem_yaml_path.exists():
                elem_data = load_yaml(elem_yaml_path)

                # Check if rules are still placeholders (all start with #)
                rules = elem_data.get("rules", [])
                is_placeholder = not rules or all(str(r).startswith("#") for r in rules)

                if is_placeholder:
                    other_elems = [e for e in elem_names if e != elem_name]
                    attrs = [a.get("name", "") for a in element.get("children", [])]

                    prompt = ELEMENT_RULES_PROMPT.format(
                        category_name=cat_name,
                        element_name=elem_name,
                        element_description=elem_data.get("description", ""),
                        other_elements="\n".join(f"- {e}" for e in other_elems[:10]),
                        attributes="\n".join(f"- {a}" for a in attrs) if attrs else "(none)",
                    )
                    prompts.append(prompt)
                    targets.append(
                        (elem_yaml_path, elem_data, f"Element: {cat_name} > {elem_name}")
                    )

            # Attribute rules (only if include_attributes is True)
            if include_attributes:
                attributes = element.get("children", [])
                attr_names = [a.get("name", "") for a in attributes]

                for attribute in attributes:
                    attr_name = attribute.get("name", "")
                    attr_filename = sanitize_folder_name(attr_name) + ".yaml"
                    attr_yaml_path = artifacts_dir / cat_folder / elem_folder / attr_filename

                    if attr_yaml_path.exists():
                        attr_data = load_yaml(attr_yaml_path)

                        rules = attr_data.get("rules", [])
                        is_placeholder = not rules or all(str(r).startswith("#") for r in rules)

                        if is_placeholder:
                            other_attrs = [a for a in attr_names if a != attr_name]
                            prompt = ATTRIBUTE_RULES_PROMPT.format(
                                category_name=cat_name,
                                element_name=elem_name,
                                attribute_name=attr_name,
                                attribute_description=attr_data.get("description", ""),
                                other_attributes="\n".join(f"- {a}" for a in other_attrs[:10])
                                if other_attrs
                                else "(none)",
                            )
                            prompts.append(prompt)
                            targets.append(
                                (
                                    attr_yaml_path,
                                    attr_data,
                                    f"Attribute: {cat_name} > {elem_name} > {attr_name}",
                                )
                            )

    if not prompts:
        if verbose:
            print("  No rules need generation")
        return 0

    if verbose:
        print(f"  Generating {len(prompts)} rule sets...")

    # Process with schema
    responses = processor.process_with_schema(
        prompts=prompts,
        schema=RulesResponse,
        guided_config={"temperature": 0.5, "max_tokens": 500},
    )

    results = processor.parse_results_with_schema(
        schema=RulesResponse,
        responses=responses,
        validate=True,
    )

    # Update YAML files
    for (filepath, yaml_data, label), result in zip(targets, results):
        if result and result.rules:
            yaml_data["rules"] = result.rules[:5]  # Limit to 5 rules
            save_yaml(yaml_data, filepath)
            count += 1
            if verbose:
                print(f"    ✓ {label} ({len(result.rules)} rules)")

    return count


# =============================================================================
# Examples Generation
# =============================================================================


CATEGORY_EXAMPLES_PROMPT = """You are creating training examples for a text classification system.

Category: {category_name}
Description: {category_description}

Generate 3 realistic example comments that would belong to this category.
For each example, explain briefly why it fits this category.

Respond with a JSON object containing an "examples" array where each item has "comment" and "reasoning" fields."""


ELEMENT_EXAMPLES_PROMPT = """You are creating training examples for a text classification system.

Category: {category_name}
Element: {element_name}
Description: {element_description}

Generate 3 realistic example comments with specific excerpts that map to this element.
Include sentiment (positive/negative/neutral/mixed) for each.

Respond with a JSON object containing an "examples" array where each item has:
- "comment": the full comment text
- "excerpt": the specific part about this element
- "sentiment": one of "positive", "negative", "neutral", "mixed"
- "reasoning": brief explanation"""


ATTRIBUTE_EXAMPLES_PROMPT = """You are creating training examples for a text classification system.

Category: {category_name}
Element: {element_name}
Attribute: {attribute_name}
Description: {attribute_description}

Generate 2 realistic text excerpts that would map to this specific attribute.
Include sentiment (positive/negative/neutral/mixed) for each.

Respond with a JSON object containing an "examples" array where each item has:
- "excerpt": a specific text excerpt (just a few sentences)
- "sentiment": one of "positive", "negative", "neutral", "mixed"
- "reasoning": brief explanation of why this maps to this attribute"""


def generate_examples(
    artifacts_dir: Path,
    schema: dict,
    processor: Any,
    verbose: bool = True,
    include_attributes: bool = True,
) -> int:
    """
    Generate classification examples for categories, elements, and optionally attributes.

    Args:
        artifacts_dir: Path to artifacts directory
        schema: Loaded schema dict
        processor: LLM processor
        verbose: Print progress
        include_attributes: Whether to generate attribute-level examples (can be slow)

    Returns:
        Number of example sets generated
    """
    count = 0

    categories = schema.get("children", [])

    # -------------------------------------------------------------------------
    # Category examples
    # -------------------------------------------------------------------------
    cat_prompts = []
    cat_targets = []

    for category in categories:
        cat_name = category.get("name", "")
        cat_folder = sanitize_folder_name(cat_name)
        cat_yaml_path = artifacts_dir / cat_folder / "_category.yaml"

        if cat_yaml_path.exists():
            cat_data = load_yaml(cat_yaml_path)

            # Check if examples are still placeholders
            examples = cat_data.get("examples", [])
            is_placeholder = not examples or (
                len(examples) == 1 and str(examples[0].get("comment", "")).startswith("#")
            )

            if is_placeholder:
                prompt = CATEGORY_EXAMPLES_PROMPT.format(
                    category_name=cat_name,
                    category_description=cat_data.get("description", ""),
                )
                cat_prompts.append(prompt)
                cat_targets.append((cat_yaml_path, cat_data, f"Category: {cat_name}"))

    if cat_prompts:
        if verbose:
            print(f"  Generating {len(cat_prompts)} category example sets...")

        responses = processor.process_with_schema(
            prompts=cat_prompts,
            schema=CategoryExamplesResponse,
            guided_config={"temperature": 0.7, "max_tokens": 1000},
        )

        results = processor.parse_results_with_schema(
            schema=CategoryExamplesResponse,
            responses=responses,
            validate=True,
        )

        for (filepath, yaml_data, label), result in zip(cat_targets, results):
            if result and result.examples:
                yaml_data["examples"] = [
                    {"comment": ex.comment, "reasoning": ex.reasoning} for ex in result.examples
                ]
                save_yaml(yaml_data, filepath)
                count += 1
                if verbose:
                    print(f"    ✓ {label} ({len(result.examples)} examples)")

    # -------------------------------------------------------------------------
    # Element examples
    # -------------------------------------------------------------------------
    elem_prompts = []
    elem_targets = []

    for category in categories:
        cat_name = category.get("name", "")
        cat_folder = sanitize_folder_name(cat_name)

        for element in category.get("children", []):
            elem_name = element.get("name", "")
            elem_folder = sanitize_folder_name(elem_name)
            elem_yaml_path = artifacts_dir / cat_folder / elem_folder / "_element.yaml"

            if elem_yaml_path.exists():
                elem_data = load_yaml(elem_yaml_path)

                examples = elem_data.get("examples", [])
                is_placeholder = not examples or (
                    len(examples) == 1 and str(examples[0].get("comment", "")).startswith("#")
                )

                if is_placeholder:
                    prompt = ELEMENT_EXAMPLES_PROMPT.format(
                        category_name=cat_name,
                        element_name=elem_name,
                        element_description=elem_data.get("description", ""),
                    )
                    elem_prompts.append(prompt)
                    elem_targets.append(
                        (elem_yaml_path, elem_data, f"Element: {cat_name} > {elem_name}")
                    )

    if elem_prompts:
        if verbose:
            print(f"  Generating {len(elem_prompts)} element example sets...")

        responses = processor.process_with_schema(
            prompts=elem_prompts,
            schema=ElementExamplesResponse,
            guided_config={"temperature": 0.7, "max_tokens": 1500},
        )

        results = processor.parse_results_with_schema(
            schema=ElementExamplesResponse,
            responses=responses,
            validate=True,
        )

        for (filepath, yaml_data, label), result in zip(elem_targets, results):
            if result and result.examples:
                yaml_data["examples"] = [
                    {
                        "comment": ex.comment,
                        "excerpt": ex.excerpt,
                        "sentiment": ex.sentiment
                        if ex.sentiment in ("positive", "negative", "neutral", "mixed")
                        else "positive",
                        "reasoning": ex.reasoning,
                    }
                    for ex in result.examples
                ]
                save_yaml(yaml_data, filepath)
                count += 1
                if verbose:
                    print(f"    ✓ {label} ({len(result.examples)} examples)")

    # -------------------------------------------------------------------------
    # Attribute examples (only if include_attributes is True)
    # -------------------------------------------------------------------------
    if include_attributes:
        attr_prompts = []
        attr_targets = []

        for category in categories:
            cat_name = category.get("name", "")
            cat_folder = sanitize_folder_name(cat_name)

            for element in category.get("children", []):
                elem_name = element.get("name", "")
                elem_folder = sanitize_folder_name(elem_name)

                for attribute in element.get("children", []):
                    attr_name = attribute.get("name", "")
                    attr_filename = sanitize_folder_name(attr_name) + ".yaml"
                    attr_yaml_path = artifacts_dir / cat_folder / elem_folder / attr_filename

                    if attr_yaml_path.exists():
                        attr_data = load_yaml(attr_yaml_path)

                        examples = attr_data.get("examples", [])
                        is_placeholder = not examples or (
                            len(examples) == 1
                            and str(examples[0].get("excerpt", "")).startswith("#")
                        )

                        if is_placeholder:
                            prompt = ATTRIBUTE_EXAMPLES_PROMPT.format(
                                category_name=cat_name,
                                element_name=elem_name,
                                attribute_name=attr_name,
                                attribute_description=attr_data.get("description", ""),
                            )
                            attr_prompts.append(prompt)
                            attr_targets.append(
                                (
                                    attr_yaml_path,
                                    attr_data,
                                    f"Attribute: {cat_name} > {elem_name} > {attr_name}",
                                )
                            )

        if attr_prompts:
            if verbose:
                print(f"  Generating {len(attr_prompts)} attribute example sets...")

            responses = processor.process_with_schema(
                prompts=attr_prompts,
                schema=AttributeExamplesResponse,
                guided_config={"temperature": 0.7, "max_tokens": 800},
            )

            results = processor.parse_results_with_schema(
                schema=AttributeExamplesResponse,
                responses=responses,
                validate=True,
            )

            for (filepath, yaml_data, label), result in zip(attr_targets, results):
                if result and result.examples:
                    yaml_data["examples"] = [
                        {
                            "excerpt": ex.excerpt,
                            "sentiment": ex.sentiment
                            if ex.sentiment in ("positive", "negative", "neutral", "mixed")
                            else "positive",
                            "reasoning": ex.reasoning,
                        }
                        for ex in result.examples
                    ]
                    save_yaml(yaml_data, filepath)
                    count += 1
                    if verbose:
                        print(f"    ✓ {label} ({len(result.examples)} examples)")

    return count


# =============================================================================
# Main Generator Function
# =============================================================================


def generate_artifact_content(
    artifacts_dir: str | Path,
    schema_path: str | Path,
    processor: Any,
    generate_descriptions_flag: bool = True,
    generate_rules_flag: bool = True,
    generate_examples_flag: bool = True,
    generate_attribute_content: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Generate all content for YAML artifacts using LLM.

    This fills in scaffolded YAML files with:
    - Condensed descriptions
    - Disambiguation rules
    - Classification examples

    Args:
        artifacts_dir: Path to artifacts directory
        schema_path: Path to schema.json
        processor: LLM processor for generation
        generate_descriptions_flag: Generate descriptions
        generate_rules_flag: Generate rules
        generate_examples_flag: Generate examples
        generate_attribute_content: Generate content for attribute-level files (can be slow with many attributes)
        verbose: Print progress

    Returns:
        Dict with generation statistics
    """
    artifacts_dir = Path(artifacts_dir)
    schema_path = Path(schema_path)

    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    if verbose:
        print("\n" + "=" * 70)
        print("GENERATING ARTIFACT CONTENT")
        print("=" * 70)
        print(f"Artifacts: {artifacts_dir}")
        print(f"Schema: {schema_path}")

    stats = {
        "descriptions": 0,
        "rules": 0,
        "examples": 0,
    }

    # Generate descriptions
    if generate_descriptions_flag:
        if verbose:
            print("\n📝 DESCRIPTIONS")
        stats["descriptions"] = generate_descriptions(artifacts_dir, schema, processor, verbose)

    # Generate rules
    if generate_rules_flag:
        if verbose:
            print("\n📋 RULES")
        stats["rules"] = generate_rules(
            artifacts_dir,
            schema,
            processor,
            verbose,
            include_attributes=generate_attribute_content,
        )

    # Generate examples
    if generate_examples_flag:
        if verbose:
            print("\n💡 EXAMPLES")
        stats["examples"] = generate_examples(
            artifacts_dir,
            schema,
            processor,
            verbose,
            include_attributes=generate_attribute_content,
        )

    if verbose:
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Descriptions: {stats['descriptions']}")
        print(f"Rules: {stats['rules']}")
        print(f"Examples: {stats['examples']}")

    return stats
