"""
YAML Artifact Schemas

Pydantic models for validating YAML artifact files.
Each file type has its own schema for validation.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Sentiment Type (shared)
# =============================================================================

SentimentType = Literal["positive", "negative", "neutral", "mixed"]


# =============================================================================
# Example Schemas (for different stages)
# =============================================================================


class CategoryExample(BaseModel):
    """Example for Stage 1 (category detection)."""

    comment: str = Field(description="The full comment text")
    reasoning: str = Field(description="Why this comment belongs to this category")


class ElementExample(BaseModel):
    """Example for Stage 2 (element extraction)."""

    comment: str = Field(description="The full comment text")
    excerpt: str = Field(description="The specific text that maps to this element")
    sentiment: SentimentType = Field(description="Sentiment of the excerpt")
    reasoning: str = Field(description="Why this excerpt maps to this element")


class AttributeExample(BaseModel):
    """Example for Stage 3 (attribute extraction)."""

    excerpt: str = Field(description="The specific text that maps to this attribute")
    sentiment: SentimentType = Field(description="Sentiment of the excerpt")
    reasoning: str = Field(description="Why this excerpt maps to this attribute")


# =============================================================================
# File Schemas
# =============================================================================


class CategoryFile(BaseModel):
    """
    Schema for _category.yaml files.

    Contains Stage 1 configuration for a category.
    """

    name: str = Field(description="Category name (must match taxonomy)")
    description: str = Field(description="Concise description for classification prompts")
    rules: List[str] = Field(
        default_factory=list, description="Disambiguation rules for detecting this category"
    )
    examples: List[CategoryExample] = Field(
        default_factory=list, description="Example comments that belong to this category"
    )


class ElementFile(BaseModel):
    """
    Schema for _element.yaml files.

    Contains Stage 2 configuration for an element.
    """

    name: str = Field(description="Element name (must match taxonomy)")
    category: str = Field(description="Parent category name")
    description: str = Field(description="Concise description for classification prompts")
    rules: List[str] = Field(
        default_factory=list, description="Disambiguation rules for extracting this element"
    )
    examples: List[ElementExample] = Field(
        default_factory=list, description="Example excerpts that map to this element"
    )


class AttributeFile(BaseModel):
    """
    Schema for attribute YAML files (e.g., comfort_level.yaml).

    Contains Stage 3 configuration for an attribute.
    """

    name: str = Field(description="Attribute name (must match taxonomy)")
    category: str = Field(description="Parent category name")
    element: str = Field(description="Parent element name")
    description: str = Field(description="Concise description for classification prompts")
    rules: List[str] = Field(
        default_factory=list, description="Disambiguation rules for extracting this attribute"
    )
    examples: List[AttributeExample] = Field(
        default_factory=list, description="Example excerpts that map to this attribute"
    )


class SchemaRefFile(BaseModel):
    """
    Schema for _schema_ref.yaml file.

    Reference to the source schema for validation.
    """

    schema_source: str = Field(description="Path to source schema.json")
    created_at: str = Field(description="ISO timestamp of creation")
    structure: dict = Field(description="Summary of taxonomy structure")
    migrated_from: Optional[dict] = Field(
        default=None, description="Source files if migrated from JSON"
    )


# =============================================================================
# Loaded Artifact Collections
# =============================================================================


class LoadedCategory(BaseModel):
    """A fully loaded category with all its elements."""

    name: str
    description: str
    rules: List[str]
    examples: List[CategoryExample]
    elements: dict  # element_name -> LoadedElement


class LoadedElement(BaseModel):
    """A fully loaded element with all its attributes."""

    name: str
    category: str
    description: str
    rules: List[str]
    examples: List[ElementExample]
    attributes: dict  # attribute_name -> LoadedAttribute


class LoadedAttribute(BaseModel):
    """A fully loaded attribute."""

    name: str
    category: str
    element: str
    description: str
    rules: List[str]
    examples: List[AttributeExample]


class LoadedArtifacts(BaseModel):
    """Complete loaded artifacts from folder structure."""

    schema_ref: SchemaRefFile
    categories: dict  # category_name -> LoadedCategory

    def get_category(self, name: str) -> Optional[LoadedCategory]:
        """Get a category by name."""
        return self.categories.get(name)

    def get_element(self, category: str, element: str) -> Optional[LoadedElement]:
        """Get an element by category and element name."""
        cat = self.categories.get(category)
        if cat:
            return cat.elements.get(element)
        return None

    def get_attribute(
        self, category: str, element: str, attribute: str
    ) -> Optional[LoadedAttribute]:
        """Get an attribute by full path."""
        elem = self.get_element(category, element)
        if elem:
            return elem.attributes.get(attribute)
        return None

    def get_all_categories(self) -> List[str]:
        """Get list of all category names."""
        return list(self.categories.keys())

    def get_elements_for_category(self, category: str) -> List[str]:
        """Get element names for a category."""
        cat = self.categories.get(category)
        return list(cat.elements.keys()) if cat else []

    def get_attributes_for_element(self, category: str, element: str) -> List[str]:
        """Get attribute names for an element."""
        elem = self.get_element(category, element)
        return list(elem.attributes.keys()) if elem else []
