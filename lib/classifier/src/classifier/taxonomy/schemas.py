"""
Static Classification Schemas

Contains Pydantic models that are constant across all taxonomies:
    - SentimentType: Literal type for sentiment values
    - CategoryDetectionOutput: Stage 1 output schema
"""

from typing import List, Literal

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
