"""
Static Classification Schemas

Contains Pydantic models that are constant across all taxonomies:
    - SentimentType: Literal type for sentiment values
    - CategoryDetectionOutput: Stage 1 output schema
    - ElementExtractionSpan: Stage 2 span schema (generic)
    - ElementExtractionOutput: Stage 2 output schema (generic)
    - AttributeExtractionSpan: Stage 3 span schema (generic)
    - AttributeExtractionOutput: Stage 3 output schema (generic)
    - ClassificationSpan: Final unified span model (2-stage)
    - ClassificationSpanWithAttribute: Final unified span model (3-stage)
    - FinalClassificationOutput: Combined output after all stages
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
        sentiment: The sentiment of this excerpt (element-level)
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
# Stage 3: Attribute Extraction
# =============================================================================


class AttributeExtractionSpan(BaseModel):
    """
    Stage 3 span: A single attribute extracted from the comment.

    Attributes:
        excerpt: The exact text excerpt from the comment
        attribute: The attribute name (should match taxonomy)
        sentiment: The sentiment of this excerpt (attribute-level)
        reasoning: Why this excerpt maps to this attribute
    """

    excerpt: str
    attribute: str
    sentiment: SentimentType
    reasoning: str


class AttributeExtractionOutput(BaseModel):
    """
    Stage 3 output: List of attribute extractions for a single element.

    Attributes:
        classifications: List of extracted attribute spans
    """

    classifications: List[AttributeExtractionSpan]


# =============================================================================
# Final Combined Output (2-Stage)
# =============================================================================


class ClassificationSpan(BaseModel):
    """
    Unified span model for 2-stage output.

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
    Combined output after 2 stages (category + element).

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


# =============================================================================
# Final Combined Output (3-Stage)
# =============================================================================


class ClassificationSpanWithAttribute(BaseModel):
    """
    Unified span model for 3-stage output.

    Combines category (Stage 1) + element (Stage 2) + attribute (Stage 3).
    Includes sentiment at both element and attribute levels for consensus checking.
    Preserves reasoning and excerpts from each stage for interpretability.

    Attributes:
        category: The top-level category name
        element: The element name within the category
        element_excerpt: The excerpt identified in Stage 2
        element_sentiment: Sentiment at the element level (from Stage 2)
        element_reasoning: Why this excerpt maps to this element (Stage 2)
        attribute: The attribute name within the element
        attribute_excerpt: The excerpt identified in Stage 3 (may be same or subset)
        attribute_sentiment: Sentiment at the attribute level (from Stage 3)
        attribute_reasoning: Why this excerpt maps to this attribute (Stage 3)
        sentiment_consensus: Whether element and attribute sentiments match
    """

    category: str
    element: str
    element_excerpt: str
    element_sentiment: SentimentType
    element_reasoning: str
    attribute: str
    attribute_excerpt: str
    attribute_sentiment: SentimentType
    attribute_reasoning: str

    @property
    def sentiment_consensus(self) -> bool:
        """Check if element and attribute sentiments match."""
        return self.element_sentiment == self.attribute_sentiment

    # Convenience aliases for backward compatibility
    @property
    def excerpt(self) -> str:
        """Primary excerpt (attribute-level for most specific)."""
        return self.attribute_excerpt

    @property
    def reasoning(self) -> str:
        """Primary reasoning (attribute-level for most specific)."""
        return self.attribute_reasoning


class FinalClassificationOutputWithAttributes(BaseModel):
    """
    Combined output after 3 stages (category + element + attribute).

    Includes sentiment at both element and attribute levels for validation.

    Attributes:
        original_comment: The original comment text
        has_classifiable_content: Whether the comment contains classifiable feedback
        category_reasoning: Stage 1 reasoning for category detection
        classifications: List of all extracted classifications with attributes
    """

    original_comment: str
    has_classifiable_content: bool
    category_reasoning: str
    classifications: List[ClassificationSpanWithAttribute]

    def get_consensus_rate(self) -> float:
        """Calculate the rate of sentiment consensus across classifications."""
        if not self.classifications:
            return 1.0
        matches = sum(1 for c in self.classifications if c.sentiment_consensus)
        return matches / len(self.classifications)

    def get_mismatches(self) -> List[ClassificationSpanWithAttribute]:
        """Get classifications where element and attribute sentiments differ."""
        return [c for c in self.classifications if not c.sentiment_consensus]
