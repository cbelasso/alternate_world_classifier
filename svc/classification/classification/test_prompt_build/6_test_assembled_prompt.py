"""
Test the Assembled Stage 1 Prompt

Quick test to verify the assembled prompt works correctly.

Usage:
    python 6_test_assembled_prompt.py
"""

from utils.data_io import load_json

# Path to generated prompt module
GENERATED_PROMPT_PATH = "./stage1_prompt_generated.py"

# Load the generated module
import importlib.util

spec = importlib.util.spec_from_file_location("stage1_prompt_generated", GENERATED_PROMPT_PATH)
stage1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage1_module)

# Get the prompt function
stage1_category_detection_prompt = stage1_module.stage1_category_detection_prompt

# Test comments
test_comments = [
    "The networking sessions were fantastic and I made great connections!",
    "WiFi was terrible but the venue was beautiful.",
    "The keynote speaker was brilliant.",
    "Everything was perfect!",
    "Data integrity never goes out of style.",
]

print("=" * 70)
print("TESTING ASSEMBLED STAGE 1 PROMPT")
print("=" * 70)

for i, comment in enumerate(test_comments, 1):
    print(f'\n{i}. Testing: "{comment}"')

    prompt = stage1_category_detection_prompt(comment)

    print(f"   Prompt length: {len(prompt):,} chars (~{len(prompt) // 4} tokens)")
    print("   ✓ Generated successfully")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nThe prompt is ready to use with your LLM processor!")
print("\nNext steps:")
print("  1. Import the generated function")
print("  2. Use with your NewProcessor")
print("  3. Get category classifications!")
