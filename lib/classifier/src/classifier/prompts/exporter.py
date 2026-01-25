"""
Prompt Exporter

Exports classification prompts to a hierarchical folder structure with:
    - Organized folders for Stage 1, Stage 2, and future stages
    - Auto-generated headers with MANUALLY_EDITED flag
    - Sanitized file/folder names
    - __init__.py files for easy imports

Folder Structure:
    prompts/
    ├── __init__.py
    ├── stage1/
    │   ├── __init__.py
    │   └── category_detection.py
    └── stage2/
        ├── __init__.py
        ├── attendee_engagement_and_interaction.py
        ├── event_logistics_and_infrastructure.py
        ├── event_operations_and_management.py
        ├── learning_and_content_delivery.py
        └── people.py

Usage:
    from classifier.prompts.exporter import export_prompts_hierarchical

    export_prompts_hierarchical(
        condensed=condensed,
        examples=examples,
        rules=rules,
        output_dir="prompts/",
    )
"""

from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional

from ..taxonomy.condenser import CondensedTaxonomy
from ..taxonomy.example_generator import ClassificationExample, ExampleSet
from ..taxonomy.rule_generator import ClassificationRules

# =============================================================================
# Name Sanitization
# =============================================================================


def sanitize_name(name: str) -> str:
    """
    Sanitize a category/element name for use as a filename.

    Transformations:
        - Replace '&' with 'and'
        - Replace '/' with '_'
        - Replace spaces with '_'
        - Remove other special characters
        - Convert to lowercase
        - Collapse multiple underscores

    Examples:
        "Attendee Engagement & Interaction" -> "attendee_engagement_and_interaction"
        "Speakers/Presenters" -> "speakers_presenters"
        "Wi-Fi" -> "wi_fi"
        "Q&A Sessions" -> "q_and_a_sessions"

    Args:
        name: Original name

    Returns:
        Sanitized name suitable for filenames
    """
    # Replace & with 'and'
    result = name.replace("&", "and")

    # Replace / with _
    result = result.replace("/", "_")

    # Replace - with _
    result = result.replace("-", "_")

    # Replace spaces with _
    result = result.replace(" ", "_")

    # Remove any other special characters (keep alphanumeric and _)
    result = re.sub(r"[^a-zA-Z0-9_]", "", result)

    # Convert to lowercase
    result = result.lower()

    # Collapse multiple underscores
    result = re.sub(r"_+", "_", result)

    # Strip leading/trailing underscores
    result = result.strip("_")

    return result


# =============================================================================
# Header Generation
# =============================================================================


def generate_header(
    title: str,
    description: str,
    source_files: List[str],
    additional_notes: Optional[List[str]] = None,
) -> str:
    """
    Generate a header comment for auto-generated prompt files.

    Includes:
        - Title and description
        - Generation timestamp
        - Source files used
        - MANUALLY_EDITED flag
        - Warning about regeneration

    Args:
        title: Title for the file
        description: Brief description
        source_files: List of source JSON files
        additional_notes: Optional additional notes

    Returns:
        Header comment string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sources = "\n#   - ".join(source_files)
    notes = ""
    if additional_notes:
        notes = "\n#\n# Notes:\n" + "\n".join(f"#   - {n}" for n in additional_notes)

    return f"""# =============================================================================
# {title}
# =============================================================================
#
# {description}
#
# AUTO-GENERATED from JSON artifacts:
#   - {sources}
#
# Generated: {timestamp}
#
# ⚠️  WARNING: Direct edits to this file will be LOST if regenerated!
#     To make permanent changes, edit the source JSON files and rebuild.
#
#     If you intentionally edited this file, change the flag below to True
#     to protect it from being overwritten during regeneration.
#
# MANUALLY_EDITED: False
# ============================================================================={notes}
"""


def is_manually_edited(filepath: Path) -> bool:
    """
    Check if a file has been marked as manually edited.

    Args:
        filepath: Path to check

    Returns:
        True if MANUALLY_EDITED: True is found in file
    """
    if not filepath.exists():
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read(2000)  # Check first 2000 chars (header area)

    return "MANUALLY_EDITED: True" in content


# =============================================================================
# Stage 1 Export
# =============================================================================


def export_stage1_prompt(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: ClassificationRules,
    output_dir: str | Path,
    force_overwrite: bool = False,
) -> Path:
    """
    Export Stage 1 category detection prompt.

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        rules: ClassificationRules
        output_dir: Output directory (will create stage1/ subfolder)
        force_overwrite: If True, overwrite even if MANUALLY_EDITED

    Returns:
        Path to exported file
    """
    from .base import format_categories_section, format_stage1_examples_section

    output_dir = Path(output_dir)
    stage1_dir = output_dir / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    filepath = stage1_dir / "category_detection.py"

    # Check for manual edits
    if not force_overwrite and is_manually_edited(filepath):
        print(f"  ⏭ Skipping {filepath} (MANUALLY_EDITED: True)")
        return filepath

    # Build prompt components
    categories_section = format_categories_section(condensed)

    example_list = examples.examples if isinstance(examples, ExampleSet) else examples
    examples_section = format_stage1_examples_section(example_list)

    # Format rules
    rules_list = rules.get_stage1_rules()
    rules_section = "\n".join(f"{i}. {r}" for i, r in enumerate(rules_list, 1))

    # Generate header
    header = generate_header(
        title="Stage 1: Category Detection Prompt",
        description="Detects which categories are present in conference feedback comments.",
        source_files=["condensed_taxonomy.json", "examples.json", "rules.json"],
    )

    # Build the module content
    module_content = f'''{header}

def stage1_category_detection_prompt(comment: str) -> str:
    """
    Generate Stage 1 category detection prompt.

    Args:
        comment: The conference feedback comment to analyze

    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in the following comment.

COMMENT TO ANALYZE:
{{comment}}

---

CATEGORIES TO CONSIDER:

{categories_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Analyze the comment and return ONLY valid JSON with:
- categories_present: list of category names that apply
- has_classifiable_content: true/false
- reasoning: brief explanation"""


# Convenience alias
STAGE1_PROMPT = stage1_category_detection_prompt
'''

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(module_content)

    return filepath


# =============================================================================
# Stage 2 Export
# =============================================================================


def export_stage2_prompts(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: ClassificationRules,
    output_dir: str | Path,
    force_overwrite: bool = False,
) -> List[Path]:
    """
    Export Stage 2 element extraction prompts (one per category).

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        rules: ClassificationRules
        output_dir: Output directory (will create stage2/ subfolder)
        force_overwrite: If True, overwrite even if MANUALLY_EDITED

    Returns:
        List of paths to exported files
    """
    from ..prompts.stage2 import (
        format_elements_section,
        format_stage2_examples_section,
        get_stage2_examples_for_category,
    )

    output_dir = Path(output_dir)
    stage2_dir = output_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []

    for category in condensed.categories:
        # Sanitize filename
        filename = sanitize_name(category.name) + ".py"
        filepath = stage2_dir / filename

        # Check for manual edits
        if not force_overwrite and is_manually_edited(filepath):
            print(f"  ⏭ Skipping {filepath} (MANUALLY_EDITED: True)")
            exported_files.append(filepath)
            continue

        # Build prompt components
        elements_section = format_elements_section(category)

        category_examples = get_stage2_examples_for_category(examples, category.name)
        examples_section = format_stage2_examples_section(category_examples, category.name)

        # Build rules section
        all_rules = rules.get_all_stage2_rules(category.name)
        rules_section = "\n".join(f"{i}. {r}" for i, r in enumerate(all_rules, 1))

        # Generate header
        header = generate_header(
            title=f"Stage 2: {category.name}",
            description=f"Extracts feedback elements related to {category.name}.",
            source_files=["condensed_taxonomy.json", "examples.json", "rules.json"],
            additional_notes=[
                f"Category: {category.name}",
                f"Elements: {', '.join(e.name for e in category.elements)}",
            ],
        )

        # Build function name
        func_name = f"stage2_{sanitize_name(category.name)}_prompt"

        # Build module content
        module_content = f'''{header}

CATEGORY_NAME = "{category.name}"

ELEMENTS = [
    {", ".join(f'"{e.name}"' for e in category.elements)}
]


def {func_name}(comment: str) -> str:
    """
    Generate Stage 2 element extraction prompt for {category.name}.

    Args:
        comment: The conference feedback comment to analyze

    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Extract specific feedback related to {category.name} from this comment.

COMMENT TO ANALYZE:
{{comment}}

---

ELEMENTS TO IDENTIFY:

{elements_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to {category.name}, return {{"classifications": []}}."""


# Convenience alias
PROMPT = {func_name}
'''

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(module_content)

        exported_files.append(filepath)

    return exported_files


# =============================================================================
# Init File Generation
# =============================================================================


def _generate_stage1_init(stage1_dir: Path) -> None:
    """Generate __init__.py for stage1 folder."""
    content = '''"""
Stage 1: Category Detection Prompts

Auto-generated prompt modules for category detection.

Usage:
    from prompts.stage1 import stage1_category_detection_prompt

    prompt = stage1_category_detection_prompt("Your comment here")
"""

from .category_detection import stage1_category_detection_prompt, STAGE1_PROMPT

__all__ = ["stage1_category_detection_prompt", "STAGE1_PROMPT"]
'''

    with open(stage1_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


def _generate_stage2_init(stage2_dir: Path, categories: List[str]) -> None:
    """Generate __init__.py for stage2 folder."""
    imports = []
    exports = []
    dict_entries = []

    for cat_name in categories:
        sanitized = sanitize_name(cat_name)
        func_name = f"stage2_{sanitized}_prompt"
        imports.append(f"from .{sanitized} import {func_name}")
        exports.append(func_name)
        dict_entries.append(f'    "{cat_name}": {func_name},')

    imports_str = "\n".join(imports)
    exports_str = ",\n    ".join(f'"{e}"' for e in exports)
    dict_str = "\n".join(dict_entries)

    content = f'''"""
Stage 2: Element Extraction Prompts

Auto-generated prompt modules for element extraction.

Usage:
    from prompts.stage2 import STAGE2_PROMPTS
    from prompts.stage2 import stage2_people_prompt

    # Use the dict
    prompt = STAGE2_PROMPTS["People"]("Your comment here")

    # Or import directly
    prompt = stage2_people_prompt("Your comment here")
"""

from typing import Callable, Dict

{imports_str}

STAGE2_PROMPTS: Dict[str, Callable[[str], str]] = {{
{dict_str}
}}

__all__ = [
    "STAGE2_PROMPTS",
    {exports_str}
]
'''

    with open(stage2_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


def _generate_main_init(output_dir: Path) -> None:
    """Generate main __init__.py for prompts folder."""
    content = '''"""
Classification Prompts

Auto-generated prompt modules for the classification pipeline.

Structure:
    prompts/
    ├── stage1/           # Category detection
    │   └── category_detection.py
    └── stage2/           # Element extraction (one file per category)
        ├── people.py
        └── ...

Usage:
    # Stage 1
    from prompts.stage1 import stage1_category_detection_prompt

    # Stage 2
    from prompts.stage2 import STAGE2_PROMPTS
    from prompts.stage2 import stage2_people_prompt
"""

from .stage1 import stage1_category_detection_prompt, STAGE1_PROMPT
from .stage2 import STAGE2_PROMPTS

__all__ = [
    "stage1_category_detection_prompt",
    "STAGE1_PROMPT",
    "STAGE2_PROMPTS",
]
'''

    with open(output_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# Main Export Function
# =============================================================================


def export_prompts_hierarchical(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: ClassificationRules,
    output_dir: str | Path,
    force_overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Export all prompts to a hierarchical folder structure.

    Creates:
        prompts/
        ├── __init__.py
        ├── stage1/
        │   ├── __init__.py
        │   └── category_detection.py
        └── stage2/
            ├── __init__.py
            ├── attendee_engagement_and_interaction.py
            └── ...

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        rules: ClassificationRules
        output_dir: Root output directory
        force_overwrite: If True, overwrite files even if MANUALLY_EDITED
        verbose: Print progress

    Returns:
        Dict with export results:
            - stage1_file: Path to Stage 1 prompt
            - stage2_files: List of paths to Stage 2 prompts
            - total_files: Total number of files created
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("EXPORTING PROMPTS (Hierarchical)")
        print("=" * 70)
        print(f"Output: {output_dir}")

    # Export Stage 1
    if verbose:
        print("\nStage 1:")
    stage1_file = export_stage1_prompt(condensed, examples, rules, output_dir, force_overwrite)
    if verbose:
        print(f"  ✓ {stage1_file}")

    # Export Stage 2
    if verbose:
        print(f"\nStage 2 ({len(condensed.categories)} categories):")
    stage2_files = export_stage2_prompts(
        condensed, examples, rules, output_dir, force_overwrite
    )
    for f in stage2_files:
        if verbose:
            print(f"  ✓ {f}")

    # Generate __init__.py files
    if verbose:
        print("\nGenerating __init__.py files...")

    _generate_stage1_init(output_dir / "stage1")
    _generate_stage2_init(output_dir / "stage2", [c.name for c in condensed.categories])
    _generate_main_init(output_dir)

    if verbose:
        print(f"\n✓ Exported {1 + len(stage2_files)} prompt files to {output_dir}")

    return {
        "stage1_file": stage1_file,
        "stage2_files": stage2_files,
        "total_files": 1 + len(stage2_files),
    }


# =============================================================================
# Stage 3 Export
# =============================================================================


def export_stage3_prompts(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: ClassificationRules,
    taxonomy: dict,
    output_dir: str | Path,
    force_overwrite: bool = False,
) -> List[Path]:
    """
    Export Stage 3 attribute extraction prompts (nested: category/element.py).

    Creates:
        stage3/
        ├── __init__.py
        ├── attendee_engagement_and_interaction/
        │   ├── __init__.py
        │   ├── community.py
        │   └── knowledge_exchange.py
        └── people/
            ├── __init__.py
            ├── speakers_presenters.py
            └── ...

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        rules: ClassificationRules
        taxonomy: Raw taxonomy dict (for attribute info)
        output_dir: Output directory (will create stage3/ subfolder)
        force_overwrite: If True, overwrite even if MANUALLY_EDITED

    Returns:
        List of paths to exported files
    """
    from .stage3 import (
        format_attributes_section,
        format_stage3_examples_section,
        format_stage3_rules,
        get_attributes_for_element,
        get_stage3_examples_for_element,
    )

    output_dir = Path(output_dir)
    stage3_dir = output_dir / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []

    for category in condensed.categories:
        # Create category subfolder
        cat_sanitized = sanitize_name(category.name)
        cat_dir = stage3_dir / cat_sanitized
        cat_dir.mkdir(parents=True, exist_ok=True)

        element_funcs = []  # Track for __init__.py

        for element in category.elements:
            # Get attributes to check if element has any
            attributes = get_attributes_for_element(taxonomy, category.name, element.name)
            if not attributes:
                continue  # Skip elements without attributes

            # Sanitize filename
            elem_sanitized = sanitize_name(element.name)
            filename = f"{elem_sanitized}.py"
            filepath = cat_dir / filename

            # Check for manual edits
            if not force_overwrite and is_manually_edited(filepath):
                print(f"  ⏭ Skipping {filepath} (MANUALLY_EDITED: True)")
                exported_files.append(filepath)
                element_funcs.append((element.name, elem_sanitized))
                continue

            # Build prompt components
            attributes_section = format_attributes_section(element, taxonomy, category.name)

            element_examples = get_stage3_examples_for_element(
                examples, category.name, element.name
            )
            examples_section = format_stage3_examples_section(
                element_examples, category.name, element.name, taxonomy
            )

            # Build rules section
            all_rules = rules.get_all_stage3_rules(category.name, element.name)
            rules_section = "\n".join(f"{i}. {r}" for i, r in enumerate(all_rules, 1))

            # Generate header
            header = generate_header(
                title=f"Stage 3: {category.name} > {element.name}",
                description=f"Extracts attribute-level feedback for {element.name}.",
                source_files=["condensed_taxonomy.json", "examples.json", "rules.json"],
                additional_notes=[
                    f"Category: {category.name}",
                    f"Element: {element.name}",
                    f"Attributes: {', '.join(a['name'] for a in attributes)}",
                ],
            )

            # Build function name
            func_name = f"stage3_{cat_sanitized}_{elem_sanitized}_prompt"

            # Build module content
            module_content = f'''{header}

CATEGORY_NAME = "{category.name}"
ELEMENT_NAME = "{element.name}"

ATTRIBUTES = [
    {", ".join(f'"{a["name"]}"' for a in attributes)}
]


def {func_name}(comment: str) -> str:
    """
    Generate Stage 3 attribute extraction prompt for {category.name} > {element.name}.

    Args:
        comment: The conference feedback comment to analyze

    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Extract specific feedback related to attributes of {element.name} (within {category.name}) from this comment.

COMMENT TO ANALYZE:
{{comment}}

---

CONTEXT:
This comment has been identified as discussing "{element.name}" within the "{category.name}" category.
Your task is to identify which specific ATTRIBUTES of {element.name} are being discussed.

---

ATTRIBUTES TO IDENTIFY:

{attributes_section}

---

CLASSIFICATION RULES:

{rules_section}

---

EXAMPLES:

{examples_section}

---

Extract all relevant excerpts and return ONLY valid JSON. If no content relates to specific attributes of {element.name}, return {{"classifications": []}}."""


# Convenience alias
PROMPT = {func_name}
'''

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(module_content)

            exported_files.append(filepath)
            element_funcs.append((element.name, elem_sanitized))

        # Generate category __init__.py
        if element_funcs:
            _generate_stage3_category_init(cat_dir, category.name, cat_sanitized, element_funcs)

    # Generate stage3 __init__.py
    _generate_stage3_init(stage3_dir, condensed)

    return exported_files


def _generate_stage3_category_init(
    cat_dir: Path,
    category_name: str,
    cat_sanitized: str,
    element_funcs: List[tuple],
) -> None:
    """Generate __init__.py for a Stage 3 category folder."""
    imports = []
    exports = []
    dict_entries = []

    for elem_name, elem_sanitized in element_funcs:
        func_name = f"stage3_{cat_sanitized}_{elem_sanitized}_prompt"
        imports.append(f"from .{elem_sanitized} import {func_name}")
        exports.append(func_name)
        dict_entries.append(f'    "{elem_name}": {func_name},')

    imports_str = "\n".join(imports)
    exports_str = ",\n    ".join(f'"{e}"' for e in exports)
    dict_str = "\n".join(dict_entries)

    content = f'''"""
Stage 3: {category_name}

Attribute extraction prompts for elements in {category_name}.

Usage:
    from stage3.{cat_sanitized} import STAGE3_PROMPTS
    prompt = STAGE3_PROMPTS["Community"]("Your comment here")
"""

from typing import Callable, Dict

{imports_str}

STAGE3_PROMPTS: Dict[str, Callable[[str], str]] = {{
{dict_str}
}}

CATEGORY_NAME = "{category_name}"

__all__ = [
    "STAGE3_PROMPTS",
    "CATEGORY_NAME",
    {exports_str}
]
'''

    with open(cat_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


def _generate_stage3_init(stage3_dir: Path, condensed: CondensedTaxonomy) -> None:
    """Generate main __init__.py for stage3 folder."""
    imports = []
    dict_entries = []

    for category in condensed.categories:
        cat_sanitized = sanitize_name(category.name)
        imports.append(f"from . import {cat_sanitized}")
        dict_entries.append(f'    "{category.name}": {cat_sanitized}.STAGE3_PROMPTS,')

    imports_str = "\n".join(imports)
    dict_str = "\n".join(dict_entries)

    content = f'''"""
Stage 3: Attribute Extraction Prompts

Nested structure: category -> element -> prompt function

Usage:
    from prompts.stage3 import STAGE3_PROMPTS

    # Access by category then element
    prompt_fn = STAGE3_PROMPTS["Attendee Engagement & Interaction"]["Community"]
    prompt = prompt_fn("Your comment here")
"""

from typing import Callable, Dict

{imports_str}

# Nested dict: category -> element -> prompt_fn
STAGE3_PROMPTS: Dict[str, Dict[str, Callable[[str], str]]] = {{
{dict_str}
}}

__all__ = [
    "STAGE3_PROMPTS",
]
'''

    with open(stage3_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


def export_prompts_hierarchical_with_stage3(
    condensed: CondensedTaxonomy,
    examples: ExampleSet | List[ClassificationExample],
    rules: ClassificationRules,
    taxonomy: dict,
    output_dir: str | Path,
    force_overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Export all prompts including Stage 3 to a hierarchical folder structure.

    Creates:
        prompts/
        ├── __init__.py
        ├── stage1/
        │   └── category_detection.py
        ├── stage2/
        │   ├── people.py
        │   └── ...
        └── stage3/
            ├── attendee_engagement_and_interaction/
            │   ├── community.py
            │   └── ...
            └── people/
                └── ...

    Args:
        condensed: CondensedTaxonomy
        examples: ExampleSet or list
        rules: ClassificationRules
        taxonomy: Raw taxonomy dict (for attribute info)
        output_dir: Root output directory
        force_overwrite: If True, overwrite files even if MANUALLY_EDITED
        verbose: Print progress

    Returns:
        Dict with export results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("EXPORTING PROMPTS (Hierarchical with Stage 3)")
        print("=" * 70)
        print(f"Output: {output_dir}")

    # Export Stage 1
    if verbose:
        print("\nStage 1:")
    stage1_file = export_stage1_prompt(condensed, examples, rules, output_dir, force_overwrite)
    if verbose:
        print(f"  ✓ {stage1_file}")

    # Export Stage 2
    if verbose:
        print(f"\nStage 2 ({len(condensed.categories)} categories):")
    stage2_files = export_stage2_prompts(
        condensed, examples, rules, output_dir, force_overwrite
    )
    for f in stage2_files:
        if verbose:
            print(f"  ✓ {f}")

    # Export Stage 3
    total_elements = sum(len(c.elements) for c in condensed.categories)
    if verbose:
        print(
            f"\nStage 3 ({total_elements} elements across {len(condensed.categories)} categories):"
        )
    stage3_files = export_stage3_prompts(
        condensed, examples, rules, taxonomy, output_dir, force_overwrite
    )
    for f in stage3_files:
        if verbose:
            print(f"  ✓ {f}")

    # Generate __init__.py files
    if verbose:
        print("\nGenerating __init__.py files...")

    _generate_stage1_init(output_dir / "stage1")
    _generate_stage2_init(output_dir / "stage2", [c.name for c in condensed.categories])
    _generate_main_init_with_stage3(output_dir)

    total_files = 1 + len(stage2_files) + len(stage3_files)
    if verbose:
        print(f"\n✓ Exported {total_files} prompt files to {output_dir}")

    return {
        "stage1_file": stage1_file,
        "stage2_files": stage2_files,
        "stage3_files": stage3_files,
        "total_files": total_files,
    }


def _generate_main_init_with_stage3(output_dir: Path) -> None:
    """Generate main __init__.py for prompts folder including Stage 3."""
    content = '''"""
Classification Prompts

Auto-generated prompt modules for the classification pipeline.

Structure:
    prompts/
    ├── stage1/           # Category detection
    │   └── category_detection.py
    ├── stage2/           # Element extraction (one file per category)
    │   ├── people.py
    │   └── ...
    └── stage3/           # Attribute extraction (nested: category/element.py)
        ├── people/
        │   └── speakers_presenters.py
        └── ...

Usage:
    # Stage 1
    from prompts.stage1 import stage1_category_detection_prompt

    # Stage 2
    from prompts.stage2 import STAGE2_PROMPTS

    # Stage 3
    from prompts.stage3 import STAGE3_PROMPTS
"""

from .stage1 import stage1_category_detection_prompt, STAGE1_PROMPT
from .stage2 import STAGE2_PROMPTS
from .stage3 import STAGE3_PROMPTS

__all__ = [
    "stage1_category_detection_prompt",
    "STAGE1_PROMPT",
    "STAGE2_PROMPTS",
    "STAGE3_PROMPTS",
]
'''

    with open(output_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)
