"""
Static Classification Schemas

Contains Pydantic models that are constant across all taxonomies:
    - SentimentType: Literal type for sentiment values
    - CategoryDetectionOutput: Stage 1 output schema
    - ElementExtractionSpan: Stage 2 span schema (generic)
    - ElementExtractionOutput: Stage 2 output schema (generic)
    - ClassificationSpan: Final unified span model
    - FinalClassificationOutput: Combined output after both stages
"""

from typing import List, Literal, Optional

from pydantic import BaseModel

# =============================================================================
# Shared Types
# =============================================================================

SentimentType = Literal["positive", "negative", "neutral", "mixed"]
"""Standard sentiment values used across all classification spans."""


# =============================================================================
# Stage 1: Category Detection
# =============================================================================


class CategoryDetectionOutput(BaseModel):
    """
    Stage 1 output: Identifies which categories are present in a comment.

    Note: The `categories_present` field uses `List[str]` rather than a Literal
    because the valid categories are determined dynamically from the taxonomy.
    Validation against valid categories should happen at the orchestrator level.

    Attributes:
        categories_present: List of category names detected in the comment
        has_classifiable_content: Whether the comment contains classifiable feedback
        reasoning: Explanation of why these categories were selected
    """

    categories_present: List[str]
    has_classifiable_content: bool
    reasoning: str


# =============================================================================
# Stage 2: Element Extraction
# =============================================================================


class ElementExtractionSpan(BaseModel):
    """
    Stage 2 span: A single element extracted from the comment.

    This is a generic schema - for category-specific schemas with Literal
    element types, use build_models_from_taxonomy().

    Attributes:
        excerpt: The exact text excerpt from the comment
        element: The element name (should match taxonomy)
        sentiment: The sentiment of this excerpt
        reasoning: Why this excerpt maps to this element
    """

    excerpt: str
    element: str
    sentiment: SentimentType
    reasoning: str


class ElementExtractionOutput(BaseModel):
    """
    Stage 2 output: List of element extractions for a single category.

    This is a generic schema - for category-specific schemas with Literal
    element types, use build_models_from_taxonomy().

    Attributes:
        classifications: List of extracted element spans
    """

    classifications: List[ElementExtractionSpan]


# =============================================================================
# Final Combined Output
# =============================================================================


class ClassificationSpan(BaseModel):
    """
    Unified span model for final output.

    Combines category (from Stage 1) with element details (from Stage 2).

    Attributes:
        excerpt: The exact text excerpt from the comment
        category: The top-level category name
        element: The element name within the category
        sentiment: The sentiment of this excerpt
        reasoning: Why this excerpt maps to this element
    """

    excerpt: str
    category: str
    element: str
    sentiment: SentimentType
    reasoning: str


class FinalClassificationOutput(BaseModel):
    """
    Combined output after both stages.

    This is the final result of classifying a single comment through
    Stage 1 (category detection) and Stage 2 (element extraction).

    Attributes:
        original_comment: The original comment text
        has_classifiable_content: Whether the comment contains classifiable feedback
        category_reasoning: Stage 1 reasoning for category detection
        classifications: List of all extracted classifications
    """

    original_comment: str
    has_classifiable_content: bool
    category_reasoning: str
    classifications: List[ClassificationSpan]
