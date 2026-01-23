"""
Test Stage 1 Classification with Dynamically Generated Prompt

Uses the prompt we assembled from condensed taxonomy and curated examples!

Usage:
    python 7_test_stage1_classification.py
    python 7_test_stage1_classification.py --real-data  # Use real conference data
"""

import argparse

# Import the dynamically generated prompt
import importlib.util
from pathlib import Path

from classifier import CategoryDetectionOutput
from llm_parallelization.new_processor import NewProcessor
import pandas as pd

GENERATED_PROMPT_PATH = "./stage1_prompt_generated.py"

# Load the generated module
spec = importlib.util.spec_from_file_location("stage1_prompt_generated", GENERATED_PROMPT_PATH)
stage1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage1_module)

# Get the prompt function
stage1_category_detection_prompt = stage1_module.stage1_category_detection_prompt

# =============================================================================
# Configuration
# =============================================================================

MODEL = "casperhansen/mistral-nemo-instruct-2407-awq"
REAL_DATA_PATH = "/data-fast/data3/clyde/projects/world/documents/annotator_files/conference_comments_annotated.xlsx"

# Test comments
TEST_COMMENTS = [
    "The Bluenotes GLOBAL Conference continues to be the best professional development opportunity each year. The community is so supportive and open to sharing their knowledge.",
    "The networking sessions were fantastic and I made great connections with peers from other institutions.",
    "The WiFi kept dropping during sessions and the room was too cold.",
    "The keynote speaker was brilliant and the presentation on machine learning was very insightful.",
    "The conference was well organized but I wish there were more hands-on workshops. The Explorance team was very helpful.",
    "Change the networking session. Nobody from Explorance showed up at my table.",
    "The presentations were better than the panel discussion. Better panelists would be preferred.",
    "Wish there was hand sanitizer more available around the conference.",
    "I loved connecting with the Blue community and sharing knowledge with other users.",
    "Data integrity never goes out of style.",
]


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test Stage 1 classification")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="GPU(s) to use")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real conference data instead of test comments",
    )
    parser.add_argument(
        "--num-comments",
        type=int,
        default=10,
        help="Number of real comments to process",
    )

    args = parser.parse_args()

    # Select comments
    if args.real_data:
        print("=" * 70)
        print("LOADING REAL CONFERENCE DATA")
        print("=" * 70)
        df = pd.read_excel(REAL_DATA_PATH)
        comments = df["comment"].dropna().tolist()[: args.num_comments]
        print(f"Loaded {len(comments)} comments from real data")
    else:
        comments = TEST_COMMENTS
        print("=" * 70)
        print(f"USING {len(TEST_COMMENTS)} TEST COMMENTS")
        print("=" * 70)

    # Show preview
    print("\nComments to classify:")
    for i, comment in enumerate(comments[:5], 1):
        preview = comment[:80] + "..." if len(comment) > 80 else comment
        print(f"  {i}. {preview}")
    if len(comments) > 5:
        print(f"  ... and {len(comments) - 5} more")

    # Generate prompts using our dynamically generated function
    print("\n" + "=" * 70)
    print("GENERATING STAGE 1 PROMPTS (from dynamic template)")
    print("=" * 70)

    prompts = [stage1_category_detection_prompt(c) for c in comments]

    print(f"✓ Generated {len(prompts)} prompts")
    print(f"Average prompt length: {sum(len(p) for p in prompts) / len(prompts):,.0f} chars")

    # Run classification
    print("\n" + "=" * 70)
    print("RUNNING STAGE 1: CATEGORY DETECTION")
    print("=" * 70)

    with NewProcessor(
        gpu_list=args.gpu,
        llm=MODEL,
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.9,
    ) as processor:
        responses = processor.process_with_schema(
            prompts=prompts,
            schema=CategoryDetectionOutput,
            batch_size=len(prompts),
            guided_config={
                "temperature": 0.1,
                "top_k": 50,
                "top_p": 0.95,
                "max_tokens": 500,
            },
        )

        results = processor.parse_results_with_schema(
            schema=CategoryDetectionOutput,
            responses=responses,
            validate=True,
        )

    # Display results
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    for i, (comment, result) in enumerate(zip(comments, results), 1):
        print(f"\n{i}. Comment: {comment[:100]}...")

        if result is None:
            print("   ❌ FAILED TO PARSE")
            continue

        if not result.has_classifiable_content:
            print("   ⚠️  No classifiable content")
            print(f"   Reasoning: {result.reasoning}")
        else:
            print(f"   ✓ Categories: {result.categories_present}")
            print(f"   Reasoning: {result.reasoning}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    successful = sum(1 for r in results if r is not None)
    classifiable = sum(1 for r in results if r and r.has_classifiable_content)
    failed = len(results) - successful

    print(f"Total comments: {len(results)}")
    print(f"Successfully parsed: {successful}")
    print(f"Classifiable content: {classifiable}")
    print(f"Failed to parse: {failed}")

    # Category distribution
    if classifiable > 0:
        print("\nCategory distribution:")
        from collections import Counter

        category_counts = Counter()
        for result in results:
            if result and result.has_classifiable_content:
                for cat in result.categories_present:
                    category_counts[cat] += 1

        for cat, count in category_counts.most_common():
            print(f"  {cat}: {count}")

    # Create DataFrame
    print("\n" + "=" * 70)
    print("CREATING RESULTS DATAFRAME")
    print("=" * 70)

    rows = []
    for comment, result in zip(comments, results):
        if result is None:
            rows.append(
                {
                    "comment": comment,
                    "has_classifiable_content": None,
                    "categories_present": None,
                    "reasoning": "FAILED TO PARSE",
                }
            )
        else:
            rows.append(
                {
                    "comment": comment,
                    "has_classifiable_content": result.has_classifiable_content,
                    "categories_present": (
                        ", ".join(result.categories_present)
                        if result.categories_present
                        else ""
                    ),
                    "reasoning": result.reasoning,
                }
            )

    df = pd.DataFrame(rows)

    # Save results
    output_path = "./stage1_classification_results.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved results to: {output_path}")

    # Show sample
    print("\nSample results (first 5):")
    print(df.head().to_string())

    print("\n" + "=" * 70)
    print("🔥 CLASSIFICATION COMPLETE! 🔥")
    print("=" * 70)
    print("\nYour dynamically generated Stage 1 prompt is working!")
    print("Next step: Build Stage 2 prompts for element extraction! 🚀")


if __name__ == "__main__":
    main()
