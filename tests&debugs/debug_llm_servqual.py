"""
Debug LLM SERVQUAL processing to see why only 2 dimensions are detected.
This will test the LLM directly and show what's happening.
"""

from src.absa.servqual_llm_model import servqual_llm
from src.data.storage import storage
import json

print("🔍 LLM SERVQUAL Debugging Analysis")
print("=" * 50)

# 1. Check LLM model status
print("\n1. 🤖 LLM Model Status:")
try:
    model_info = servqual_llm.get_model_info()
    print(f"   Model: {model_info.get('model_name', 'unknown')}")
    print(f"   Available: {model_info.get('model_available', False)}")
    print(f"   Performance: {model_info.get('validated_performance', {})}")
except Exception as e:
    print(f"   ❌ Error getting model info: {e}")

# 2. Get sample processed reviews from database
print("\n2. 📊 Sample SERVQUAL Results from Database:")
try:
    query = """
    SELECT app_id, dimension, COUNT(*) as count, AVG(sentiment_score) as avg_sentiment
    FROM servqual_scores 
    WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
    GROUP BY app_id, dimension
    ORDER BY app_id, dimension
    LIMIT 20
    """

    df = storage.db.execute_query(query)
    if not df.empty:
        print("   Current SERVQUAL dimensions in database:")
        for _, row in df.iterrows():
            print(f"     {row['app_id']}: {row['dimension']} (count: {row['count']}, avg: {row['avg_sentiment']:.2f})")
    else:
        print("   ❌ No recent SERVQUAL scores found")

except Exception as e:
    print(f"   ❌ Error querying database: {e}")

# 3. Test LLM with sample reviews manually
print("\n3. 🧪 Manual LLM Testing:")

# Get some actual reviews to test
try:
    sample_query = """
    SELECT review_id, app_id, content, rating
    FROM reviews 
    WHERE servqual_processed = TRUE 
    AND content IS NOT NULL
    AND LENGTH(content) > 20
    ORDER BY RANDOM()
    LIMIT 5
    """

    sample_df = storage.db.execute_query(sample_query)

    if not sample_df.empty:
        print("   Testing LLM on actual processed reviews:")

        for i, row in sample_df.iterrows():
            print(f"\n   --- Test Review {i + 1} ---")
            content = row['content'][:100] + "..." if len(row['content']) > 100 else row['content']
            print(f"   Content: {content}")
            print(f"   Rating: {row['rating']}")

            try:
                # Test direct LLM call
                result = servqual_llm.analyze_review_servqual(
                    review_text=row['content'],
                    app_id=row['app_id'],
                    rating=row['rating'],
                    review_id=row['review_id']
                )

                if result.success:
                    print(f"   ✅ LLM Success:")
                    for dim, scores in result.servqual_dimensions.items():
                        if isinstance(scores, dict):
                            relevant = scores.get('relevant', False)
                            sentiment = scores.get('sentiment', 0)
                            confidence = scores.get('confidence', 0)
                        else:
                            # Legacy format
                            relevant = True if scores != 0 else False
                            sentiment = scores
                            confidence = 0.9

                        if relevant:
                            print(f"     {dim}: sentiment={sentiment:.2f}, confidence={confidence:.2f}")

                    print(f"   Platform: {result.platform_context}")
                    print(f"   Processing time: {result.processing_time_ms}ms")
                else:
                    print(f"   ❌ LLM Failed: {result.error_message}")

            except Exception as e:
                print(f"   ❌ Error testing LLM: {e}")

    else:
        print("   ❌ No sample reviews found")

except Exception as e:
    print(f"   ❌ Error getting sample reviews: {e}")

# 4. Test LLM with synthetic reviews for each dimension
print("\n4. 🎯 Testing LLM with Synthetic Reviews:")

synthetic_tests = [
    {
        "content": "The app crashes constantly and doesn't work as advertised. Very unreliable product quality.",
        "expected": ["reliability", "tangibles"],
        "rating": 1
    },
    {
        "content": "Customer service was amazing! The support team helped me quickly and professionally. Great assistance.",
        "expected": ["assurance", "empathy"],
        "rating": 5
    },
    {
        "content": "The user interface is confusing and hard to navigate. Poor design and layout makes it frustrating to use.",
        "expected": ["tangibles"],
        "rating": 2
    },
    {
        "content": "Slow delivery and poor response times. Takes forever to get help or updates. Very unresponsive service.",
        "expected": ["responsiveness"],
        "rating": 2
    },
    {
        "content": "They really care about customers and have flexible return policies. Very understanding and accommodating.",
        "expected": ["empathy"],
        "rating": 5
    }
]

for i, test in enumerate(synthetic_tests):
    print(f"\n   --- Synthetic Test {i + 1} ---")
    print(f"   Content: {test['content']}")
    print(f"   Expected dimensions: {test['expected']}")

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
                if isinstance(scores, dict):
                    relevant = scores.get('relevant', False)
                else:
                    relevant = scores != 0

                if relevant:
                    detected.append(dim)

            print(f"   ✅ Detected: {detected}")

            missing = set(test['expected']) - set(detected)
            unexpected = set(detected) - set(test['expected'])

            if missing:
                print(f"   ⚠️  Missing expected: {list(missing)}")
            if unexpected:
                print(f"   ⚠️  Unexpected detections: {list(unexpected)}")

        else:
            print(f"   ❌ LLM Failed: {result.error_message}")

    except Exception as e:
        print(f"   ❌ Error in synthetic test: {e}")

print("\n" + "=" * 50)
print("📋 Diagnosis Summary:")
print("If you see:")
print("• Only reliability + responsiveness detected → Prompt engineering issue")
print("• Missing expected dimensions → Model bias or keyword issues")
print("• LLM failures → Model/API connection issues")
print("• Unexpected detections → Overly sensitive prompts")

print("\n💡 Next Steps:")
print("1. Check if prompt templates are biased toward certain dimensions")
print("2. Verify all 5 SERVQUAL dimensions are in prompt")
print("3. Test prompt engineering with more diverse examples")
print("4. Check if platform detection is affecting results")