"""
Quick test to verify the LLM parsing fix works.
Run this after applying the fix to servqual_llm_model.py
"""

from src.absa.servqual_llm_model import servqual_llm

# Test cases that should work
test_cases = [
    {
        "content": "The app crashes constantly and doesn't work as advertised. Very unreliable.",
        "expected": ["reliability"],
        "rating": 1
    },
    {
        "content": "Customer service was amazing! The support team helped me quickly and professionally.",
        "expected": ["assurance"],
        "rating": 5
    },
    {
        "content": "The interface is confusing and hard to navigate. Poor design.",
        "expected": ["tangibles"],
        "rating": 2
    },
    {
        "content": "Slow delivery and takes forever to get help. Very unresponsive.",
        "expected": ["responsiveness"],
        "rating": 2
    },
    {
        "content": "They have flexible return policies and really care about customers.",
        "expected": ["empathy"],
        "rating": 5
    }
]

print("🧪 Testing LLM SERVQUAL Fix")
print("=" * 40)

for i, test in enumerate(test_cases):
    print(f"\n--- Test {i + 1} ---")
    print(f"Content: {test['content']}")
    print(f"Expected: {test['expected']}")
    print(f"Rating: {test['rating']}")

    try:
        result = servqual_llm.analyze_review_servqual(
            review_text=test['content'],
            app_id="test_app",
            rating=test['rating'],
            review_id=f"test_{i + 1}"
        )

        if result.success:
            detected = []
            for dim, scores in result.servqual_dimensions.items():
                if scores['relevant']:
                    detected.append(dim)

            print(f"✅ Detected: {detected}")
            print(f"Processing time: {result.processing_time_ms}ms")

            # Check if expected dimensions were found
            expected_set = set(test['expected'])
            detected_set = set(detected)

            if expected_set.issubset(detected_set):
                print(f"✅ SUCCESS: All expected dimensions found")
            else:
                missing = expected_set - detected_set
                print(f"⚠️  Missing: {list(missing)}")

            if detected_set - expected_set:
                extra = detected_set - expected_set
                print(f"ℹ️  Extra detections: {list(extra)}")

        else:
            print(f"❌ LLM Failed: {result.error_message}")

    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 40)
print("Expected Results:")
print("✅ All tests should detect at least their expected dimensions")
print("✅ Processing should be under 10 seconds per review")
print("✅ No parsing errors in logs")