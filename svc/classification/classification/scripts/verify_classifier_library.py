"""
Quick Validation Script

Tests that the classifier library is properly structured and all imports work.
No LLM required - just validates the code structure.

Usage:
    python validate_classifier_library.py
"""


def test_imports():
    """Test all imports work correctly."""
    print("Testing imports...")

    # Core schemas
    from classifier import CategoryDetectionOutput, SentimentType

    print("  ✓ Core schemas")

    # Taxonomy types
    from classifier import (
        ClassificationExample,
        CondensedCategory,
        CondensedElement,
        CondensedTaxonomy,
        ElementDetail,
        ExampleSet,
    )

    print("  ✓ Taxonomy types")

    # Model building
    from classifier import build_models_from_taxonomy, load_taxonomy_models

    print("  ✓ Model building")

    # Condensation
    from classifier import condense_taxonomy, load_condensed, save_condensed

    print("  ✓ Condensation functions")

    # Examples
    from classifier import (
        combine_examples,
        generate_all_examples,
        generate_complex_examples,
        generate_simple_examples,
        load_examples,
        save_examples,
    )

    print("  ✓ Example generation functions")

    # Prompts
    from classifier import (
        DEFAULT_STAGE1_RULES,
        build_stage1_prompt_function,
        build_stage1_prompt_string,
        export_stage1_prompt_module,
    )

    print("  ✓ Prompt building functions")

    # Utilities
    from classifier import (
        get_taxonomy_stats,
        print_taxonomy_hierarchy,
        print_valid_combinations,
    )

    print("  ✓ Utilities")

    print("\n✓ All imports successful!")


def test_schemas():
    """Test that schemas can be instantiated."""
    print("\nTesting schema instantiation...")

    from classifier import (
        CategoryDetectionOutput,
        ClassificationExample,
        CondensedCategory,
        CondensedElement,
        CondensedTaxonomy,
        ElementDetail,
        ExampleSet,
    )

    # CategoryDetectionOutput
    result = CategoryDetectionOutput(
        categories_present=["People", "Learning & Content Delivery"],
        has_classifiable_content=True,
        reasoning="Test reasoning",
    )
    assert len(result.categories_present) == 2
    print("  ✓ CategoryDetectionOutput")

    # CondensedElement
    elem = CondensedElement(
        name="Networking",
        short_description="Meeting new people, professional connections",
    )
    assert elem.name == "Networking"
    print("  ✓ CondensedElement")

    # CondensedCategory
    cat = CondensedCategory(
        name="Attendee Engagement & Interaction",
        short_description="Feedback about connecting with others",
        elements=[elem],
    )
    assert len(cat.elements) == 1
    print("  ✓ CondensedCategory")

    # CondensedTaxonomy
    taxonomy = CondensedTaxonomy(categories=[cat])
    assert taxonomy.get_category_names() == ["Attendee Engagement & Interaction"]
    print("  ✓ CondensedTaxonomy")

    # ElementDetail
    detail = ElementDetail(
        category="People",
        element="Speakers/Presenters",
        excerpt="The speaker was great",
        sentiment="positive",
        reasoning="Praises speaker",
    )
    assert detail.sentiment == "positive"
    print("  ✓ ElementDetail")

    # ClassificationExample
    example = ClassificationExample(
        comment="The WiFi was terrible.",
        categories_present=["Event Logistics & Infrastructure"],
        has_classifiable_content=True,
        stage1_reasoning="Discusses WiFi issues",
        element_details=[],
        example_type="simple",
    )
    assert example.has_classifiable_content is True
    print("  ✓ ClassificationExample")

    # ExampleSet
    example_set = ExampleSet(examples=[example])
    assert example_set.get_stats()["total"] == 1
    print("  ✓ ExampleSet")

    print("\n✓ All schemas work correctly!")


def test_prompt_building():
    """Test prompt building with mock data."""
    print("\nTesting prompt building...")

    from classifier import (
        ClassificationExample,
        CondensedCategory,
        CondensedElement,
        CondensedTaxonomy,
        ExampleSet,
        build_stage1_prompt_function,
        build_stage1_prompt_string,
    )

    # Create mock condensed taxonomy
    condensed = CondensedTaxonomy(
        categories=[
            CondensedCategory(
                name="Test Category",
                short_description="Test description",
                elements=[
                    CondensedElement(
                        name="Test Element",
                        short_description="Test element description",
                    )
                ],
            )
        ]
    )

    # Create mock examples
    examples = ExampleSet(
        examples=[
            ClassificationExample(
                comment="Test comment",
                categories_present=["Test Category"],
                has_classifiable_content=True,
                stage1_reasoning="Test reasoning",
                element_details=[],
            )
        ]
    )

    # Build prompt string
    prompt = build_stage1_prompt_string(
        "The WiFi was terrible.",
        condensed,
        examples.examples,
    )
    assert "WiFi was terrible" in prompt
    assert "Test Category" in prompt
    print("  ✓ build_stage1_prompt_string")

    # Build prompt function
    prompt_fn = build_stage1_prompt_function(condensed, examples)
    prompt2 = prompt_fn("Another test comment")
    assert "Another test comment" in prompt2
    print("  ✓ build_stage1_prompt_function")

    print("\n✓ Prompt building works correctly!")


def test_save_load_cycle():
    """Test save/load functions with temporary files."""
    print("\nTesting save/load cycle...")

    from pathlib import Path
    import tempfile

    from classifier import (
        ClassificationExample,
        CondensedCategory,
        CondensedElement,
        CondensedTaxonomy,
        ExampleSet,
        load_condensed,
        load_examples,
        save_condensed,
        save_examples,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test condensed taxonomy save/load
        condensed = CondensedTaxonomy(
            categories=[
                CondensedCategory(
                    name="Test Category",
                    short_description="Test description",
                    elements=[
                        CondensedElement(
                            name="Test Element",
                            short_description="Test element desc",
                        )
                    ],
                )
            ]
        )

        condensed_path = tmpdir / "condensed.json"
        save_condensed(condensed, condensed_path)
        loaded_condensed = load_condensed(condensed_path)
        assert loaded_condensed.categories[0].name == "Test Category"
        print("  ✓ save_condensed / load_condensed")

        # Test examples save/load
        examples = ExampleSet(
            examples=[
                ClassificationExample(
                    comment="Test comment",
                    categories_present=["Test Category"],
                    has_classifiable_content=True,
                    stage1_reasoning="Test",
                    element_details=[],
                )
            ]
        )

        examples_path = tmpdir / "examples.json"
        save_examples(examples, examples_path)
        loaded_examples = load_examples(examples_path)
        assert loaded_examples.examples[0].comment == "Test comment"
        print("  ✓ save_examples / load_examples")

    print("\n✓ Save/load cycle works correctly!")


def main():
    print("=" * 70)
    print("CLASSIFIER LIBRARY VALIDATION")
    print("=" * 70)

    try:
        test_imports()
        test_schemas()
        test_prompt_building()
        test_save_load_cycle()

        print("\n" + "=" * 70)
        print("🎉 ALL VALIDATION TESTS PASSED! 🎉")
        print("=" * 70)
        print("\nThe classifier library is ready to use!")
        print("\nNext steps:")
        print("  1. Run test_stage1_pipeline.py with your taxonomy")
        print("  2. Generate and save artifacts")
        print("  3. Use for production classification")

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
