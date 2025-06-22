"""
Simple SERVQUAL-Only Processing Script with Correct Signature
Save as: run_servqual_simple.py
"""

import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🤖 SERVQUAL-Only Processing (Fixed)")
    print("=" * 40)

    try:
        from src.data.storage import storage
        from src.absa.servqual_llm_model import ServqualLLM
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return

    # Get reviews that need SERVQUAL processing
    print("\n📋 Getting reviews for SERVQUAL processing...")
    try:
        reviews_query = """
        SELECT review_id, app_id, content, rating, review_date
        FROM reviews 
        WHERE servqual_processed = FALSE
        AND NOT COALESCE(is_spam, FALSE)
        AND LENGTH(content) > 10
        ORDER BY review_date DESC
        LIMIT 50
        """

        reviews_df = storage.db.execute_query(reviews_query)

        if reviews_df.empty:
            print("  ❌ No reviews found for SERVQUAL processing")
            return

        print(f"  ✅ Found {len(reviews_df)} reviews to process")

        # Sample of what we're processing
        print("  Sample reviews:")
        for _, review in reviews_df.head(3).iterrows():
            content_preview = review['content'][:60] + "..." if len(review['content']) > 60 else review['content']
            print(f"    Rating {review['rating']}: {content_preview}")

    except Exception as e:
        print(f"❌ Failed to get reviews: {e}")
        return

    # Initialize LLM
    print("\n🤖 Initializing SERVQUAL LLM...")
    try:
        llm_model = ServqualLLM()
        print("  ✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        return

    # Process reviews with correct signature
    print(f"\n⚙️  Processing {len(reviews_df)} reviews...")

    try:
        processed_count = 0
        failed_count = 0

        for _, review in reviews_df.iterrows():
            try:
                # Use correct signature: review_text, app_id, rating, review_id
                result = llm_model.analyze_review_servqual(
                    review['content'],      # review_text
                    review['app_id'],       # app_id
                    review['rating'],       # rating
                    review['review_id']     # review_id
                )

                if result and result.success:
                    # Mark review as processed
                    update_query = """
                    UPDATE reviews 
                    SET servqual_processed = TRUE, 
                        servqual_processed_at = CURRENT_TIMESTAMP
                    WHERE review_id = :review_id
                    """
                    storage.db.execute_non_query(update_query, {'review_id': review['review_id']})
                    processed_count += 1

                    if processed_count % 10 == 0:
                        print(f"    ✅ Processed {processed_count} reviews...")
                else:
                    failed_count += 1
                    print(f"    ❌ LLM returned unsuccessful result")

            except Exception as e:
                print(f"    ❌ Review failed: {str(e)[:50]}...")
                failed_count += 1
                continue

        print(f"\n🎯 Processing Complete!")
        print(f"  ✅ Successfully processed: {processed_count} reviews")
        print(f"  ❌ Failed: {failed_count} reviews")

        if processed_count > 0:
            success_rate = (processed_count/(processed_count+failed_count)*100)
            print(f"  📊 Success rate: {success_rate:.1f}%")
            print(f"\n💡 Check dashboard for updated SERVQUAL insights!")
            print(f"🔄 Expected ~{int(processed_count * 0.25)} contributing reviews (25% detection rate)")
        else:
            print(f"  ⚠️  No reviews processed successfully")
            print(f"     Check Ollama status: curl http://localhost:11434/api/tags")

    except Exception as e:
        print(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    main()