"""
Dynamic Pydantic Model Builder

Generates Pydantic models dynamically from a JSON taxonomy structure.
Supports both 2-level (element + sentiment) and 3-level (element + attribute + sentiment)
classification schemas.

Usage:
    from classifier.taxonomy import build_models_from_taxonomy, load_taxonomy_models

    # From a taxonomy dict
    models = build_models_from_taxonomy(taxonomy, include_attributes=True)

    # From a JSON file
    models = load_taxonomy_models("path/to/taxonomy.json", include_attributes=True)

    # Access generated models
    category_to_schema = models["category_to_schema"]
    FinalOutput = models["final_output_model"]
    ClassificationSpan = models["classification_span_model"]
"""

from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, create_model, model_validator

from .schemas import SentimentType
from .utils import sanitize_model_name

# =============================================================================
# Internal Helpers
# =============================================================================


def _create_attribute_validator(element_to_attrs: Dict[str, List[str]]):
    """
    Factory function to create a Pydantic validator that ensures
    the attribute is valid for the given element.

    Args:
        element_to_attrs: Mapping of element names to their valid attributes

    Returns:
        A validator function for use with Pydantic's model_validator
    """

    def validate_attribute_for_element(self):
        element = self.element
        attribute = self.attribute

        if attribute is None:
            return self

        valid_attrs = element_to_attrs.get(element, [])

        if not valid_attrs:
            raise ValueError(
                f"Element '{element}' does not have any attributes, but got '{attribute}'"
            )

        if attribute not in valid_attrs:
            raise ValueError(
                f"Attribute '{attribute}' is not valid for element '{element}'. "
                f"Valid attributes: {valid_attrs}"
            )

        return self

    return validate_attribute_for_element


# =============================================================================
# Main Builder Function
# =============================================================================


def build_models_from_taxonomy(
    taxonomy: dict,
    include_attributes: bool = False,
) -> Dict[str, Any]:
    """
    Build Pydantic models dynamically from a JSON taxonomy.

    The taxonomy is expected to have the structure:
        Root
        └── Categories (level 1)
            └── Elements (level 2)
                └── Attributes (level 3, optional)

    Args:
        taxonomy: The parsed JSON taxonomy dictionary
        include_attributes: If True, generates 3-level models (element + attribute + sentiment).
                          If False, generates 2-level models (element + sentiment).

    Returns:
        Dictionary containing:
            - category_to_schema: Maps category name to its Output Pydantic model
            - category_to_elements: Maps category name to list of element names
            - element_to_attributes: Maps element name to list of attribute names
            - category_type: Literal type of all valid category names
            - classification_span_model: The unified ClassificationSpan model
            - final_output_model: The FinalClassificationOutput model
            - include_attributes: Echo of the input parameter

    Example:
        >>> models = build_models_from_taxonomy(taxonomy, include_attributes=True)
        >>> PeopleOutput = models["category_to_schema"]["People"]
        >>> FinalOutput = models["final_output_model"]
    """
    category_to_schema: Dict[str, Type[BaseModel]] = {}
    category_to_elements: Dict[str, List[str]] = {}
    element_to_attributes: Dict[str, List[str]] = {}

    categories = taxonomy.get("children", [])
    category_names = [cat["name"] for cat in categories]

    # Create CategoryType Literal from taxonomy
    category_type = Literal.__getitem__(tuple(category_names))

    # Process each category
    for category in categories:
        category_name = category["name"]
        elements = category.get("children", [])

        element_names = [el["name"] for el in elements]
        category_to_elements[category_name] = element_names

        if not element_names:
            continue

        # Build element-to-attributes mapping
        for element in elements:
            element_name = element["name"]
            attributes = [attr["name"] for attr in element.get("children", [])]
            element_to_attributes[element_name] = attributes

        # Create element Literal for this category
        element_literal = Literal.__getitem__(tuple(element_names))

        # Build the Span model for this category
        if include_attributes:
            # Collect all attributes for this category (for Literal type)
            all_attributes = []
            for element in elements:
                all_attributes.extend(attr["name"] for attr in element.get("children", []))

            if all_attributes:
                # Deduplicate while preserving order
                seen = set()
                unique_attributes = []
                for attr in all_attributes:
                    if attr not in seen:
                        seen.add(attr)
                        unique_attributes.append(attr)

                attribute_literal = Literal.__getitem__(tuple(unique_attributes))

                # Build element->attributes mapping for validation
                category_element_to_attrs = {
                    el["name"]: [attr["name"] for attr in el.get("children", [])]
                    for el in elements
                }

                span_model = create_model(
                    f"{sanitize_model_name(category_name)}Span",
                    excerpt=(str, ...),
                    reasoning=(str, ...),
                    element=(element_literal, ...),
                    attribute=(Optional[attribute_literal], None),
                    sentiment=(SentimentType, ...),
                    __validators__={
                        "validate_attribute_for_element": model_validator(mode="after")(
                            _create_attribute_validator(category_element_to_attrs)
                        )
                    },
                )
            else:
                # Category has elements but no attributes
                span_model = create_model(
                    f"{sanitize_model_name(category_name)}Span",
                    excerpt=(str, ...),
                    reasoning=(str, ...),
                    element=(element_literal, ...),
                    attribute=(Optional[str], None),
                    sentiment=(SentimentType, ...),
                )
        else:
            # 2-level: no attributes
            span_model = create_model(
                f"{sanitize_model_name(category_name)}Span",
                excerpt=(str, ...),
                reasoning=(str, ...),
                element=(element_literal, ...),
                sentiment=(SentimentType, ...),
            )

        # Create the Output model wrapping a list of spans
        output_model = create_model(
            f"{sanitize_model_name(category_name)}Output",
            classifications=(List[span_model], ...),
        )

        category_to_schema[category_name] = output_model

    # Build the unified ClassificationSpan model (for final output)
    all_elements = []
    for elements_list in category_to_elements.values():
        all_elements.extend(elements_list)

    if include_attributes:
        all_attributes = []
        for attrs_list in element_to_attributes.values():
            all_attributes.extend(attrs_list)

        # Deduplicate
        seen = set()
        unique_all_attributes = []
        for attr in all_attributes:
            if attr not in seen:
                seen.add(attr)
                unique_all_attributes.append(attr)

        if unique_all_attributes:
            all_attributes_literal = Literal.__getitem__(tuple(unique_all_attributes))
            classification_span_model = create_model(
                "ClassificationSpan",
                excerpt=(str, ...),
                reasoning=(str, ...),
                category=(category_type, ...),
                element=(str, ...),
                attribute=(Optional[all_attributes_literal], None),
                sentiment=(SentimentType, ...),
            )
        else:
            classification_span_model = create_model(
                "ClassificationSpan",
                excerpt=(str, ...),
                reasoning=(str, ...),
                category=(category_type, ...),
                element=(str, ...),
                attribute=(Optional[str], None),
                sentiment=(SentimentType, ...),
            )
    else:
        classification_span_model = create_model(
            "ClassificationSpan",
            excerpt=(str, ...),
            reasoning=(str, ...),
            category=(category_type, ...),
            element=(str, ...),
            sentiment=(SentimentType, ...),
        )

    # Build the FinalClassificationOutput model
    final_output_model = create_model(
        "FinalClassificationOutput",
        original_comment=(str, ...),
        has_classifiable_content=(bool, ...),
        category_reasoning=(str, ...),
        classifications=(List[classification_span_model], ...),
    )

    return {
        "category_to_schema": category_to_schema,
        "category_to_elements": category_to_elements,
        "element_to_attributes": element_to_attributes,
        "category_type": category_type,
        "classification_span_model": classification_span_model,
        "final_output_model": final_output_model,
        "include_attributes": include_attributes,
    }


# =============================================================================
# Convenience Loader
# =============================================================================


def load_taxonomy_models(
    taxonomy_path: str,
    include_attributes: bool = False,
) -> Dict[str, Any]:
    """
    Load a taxonomy from a JSON file and build models.

    Args:
        taxonomy_path: Path to the JSON taxonomy file
        include_attributes: If True, generates 3-level models

    Returns:
        Dictionary of generated models (same as build_models_from_taxonomy)
    """
    from utils.data_io import load_json

    taxonomy = load_json(taxonomy_path)
    return build_models_from_taxonomy(taxonomy, include_attributes=include_attributes)
