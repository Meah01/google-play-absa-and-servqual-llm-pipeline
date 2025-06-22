"""
Diagnose LLM to SERVQUAL Scores Conversion Issue
Save as: diagnose_llm_conversion.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("🔍 LLM SERVQUAL Conversion Diagnostic")
    print("=" * 50)

    try:
        from src.data.storage import storage
        print("✅ Storage imported")
    except Exception as e:
        print(f"❌ Storage import failed: {e}")
        return

    # Check if there's a table storing individual LLM results
    print("\n📋 Checking for LLM SERVQUAL Results Storage...")
    try:
        # Check what tables exist that might store individual LLM results
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE '%servqual%'
        ORDER BY table_name
        """

        tables = storage.db.execute_query(tables_query)
        print("SERVQUAL-related tables:")
        for _, table in tables.iterrows():
            table_name = table['table_name']
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = storage.db.execute_query(count_query)
            count = count_result.iloc[0]['count']
            print(f"  {table_name}: {count:,} records")

    except Exception as e:
        print(f"❌ Table check failed: {e}")

    # Check individual SERVQUAL reviews vs aggregated scores
    print("\n🔍 Checking Review Processing vs Score Generation...")
    try:
        # Count reviews marked as SERVQUAL processed
        processed_query = """
        SELECT 
            COUNT(*) as total_servqual_processed,
            COUNT(*) FILTER (WHERE servqual_processed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours') as processed_last_24h,
            MIN(servqual_processed_at) as earliest_processed,
            MAX(servqual_processed_at) as latest_processed
        FROM reviews 
        WHERE servqual_processed = TRUE
        AND NOT COALESCE(is_spam, FALSE)
        """

        processed_result = storage.db.execute_query(processed_query)
        row = processed_result.iloc[0]

        print(f"Reviews marked as SERVQUAL processed: {row['total_servqual_processed']:,}")
        print(f"Processed in last 24h: {row['processed_last_24h']:,}")
        print(f"Processing period: {row['earliest_processed']} to {row['latest_processed']}")

        # Count actual SERVQUAL scores generated
        scores_query = """
        SELECT 
            COUNT(*) as total_scores,
            SUM(review_count) as total_reviews_in_scores,
            COUNT(DISTINCT app_id) as apps_with_scores,
            COUNT(DISTINCT dimension) as dimensions_with_scores,
            MIN(date) as earliest_score_date,
            MAX(date) as latest_score_date
        FROM servqual_scores
        """

        scores_result = storage.db.execute_query(scores_query)
        scores_row = scores_result.iloc[0]

        print(f"\nSERVQUAL scores generated: {scores_row['total_scores']:,}")
        print(f"Total reviews contributing to scores: {scores_row['total_reviews_in_scores']:,}")
        print(f"Apps with scores: {scores_row['apps_with_scores']}")
        print(f"Dimensions with scores: {scores_row['dimensions_with_scores']}")
        print(f"Score period: {scores_row['earliest_score_date']} to {scores_row['latest_score_date']}")

        # Calculate conversion rate
        total_processed = row['total_servqual_processed']
        total_contributing = scores_row['total_reviews_in_scores']
        conversion_rate = (total_contributing / total_processed * 100) if total_processed > 0 else 0

        print(f"\n📊 CONVERSION RATE: {conversion_rate:.1f}%")
        print(f"   ({total_contributing:,} contributing / {total_processed:,} processed)")

        if conversion_rate < 10:
            print("   ❌ CRITICAL: Conversion rate is extremely low!")
        elif conversion_rate < 30:
            print("   ⚠️  WARNING: Conversion rate is low")
        else:
            print("   ✅ Conversion rate is acceptable")

    except Exception as e:
        print(f"❌ Processing vs scores check failed: {e}")

    # Check if there's a missing intermediate table for individual LLM results
    print("\n🔍 Checking for Individual LLM Results...")
    try:
        # Look for any table that might store individual review LLM results
        possible_tables = ['servqual_llm_results', 'llm_servqual_results', 'review_servqual_analysis']

        found_individual_results = False
        for table_name in possible_tables:
            try:
                test_query = f"SELECT COUNT(*) as count FROM {table_name} LIMIT 1"
                result = storage.db.execute_query(test_query)
                count = result.iloc[0]['count']
                print(f"  ✅ Found {table_name}: {count:,} records")
                found_individual_results = True
            except:
                continue

        if not found_individual_results:
            print("  ❌ No individual LLM results table found!")
            print("     This might explain the low conversion rate.")
            print("     Individual LLM results might not be stored, only aggregated.")

    except Exception as e:
        print(f"❌ Individual results check failed: {e}")

    # Check SERVQUAL scores by app and date to understand aggregation
    print("\n📊 SERVQUAL Scores Breakdown by App...")
    try:
        breakdown_query = """
        SELECT 
            app_id,
            dimension,
            COUNT(*) as score_records,
            SUM(review_count) as total_reviews,
            AVG(quality_score) as avg_quality,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM servqual_scores
        GROUP BY app_id, dimension
        ORDER BY total_reviews DESC
        """

        breakdown = storage.db.execute_query(breakdown_query)

        app_names = {
            'com.amazon.mShop.android.shopping': 'Amazon',
            'com.ebay.mobile': 'eBay',
            'com.einnovation.temu': 'Temu',
            'com.etsy.android': 'Etsy',
            'com.zzkko': 'SHEIN'
        }

        print("App-Dimension breakdown:")
        for _, row in breakdown.iterrows():
            app_name = app_names.get(row['app_id'], row['app_id'])
            dimension = row['dimension']
            reviews = row['total_reviews']
            records = row['score_records']
            avg_quality = row['avg_quality']

            print(f"  {app_name} - {dimension}: {reviews} reviews, {records} score records, avg {avg_quality:.1f}/5")

    except Exception as e:
        print(f"❌ Breakdown check failed: {e}")

    # Check recent reviews to see LLM processing pattern
    print("\n🔎 Sample Recent SERVQUAL Processed Reviews...")
    try:
        sample_query = """
        SELECT 
            review_id,
            app_id,
            content,
            rating,
            servqual_processed_at
        FROM reviews 
        WHERE servqual_processed = TRUE
        AND servqual_processed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
        ORDER BY servqual_processed_at DESC
        LIMIT 5
        """

        samples = storage.db.execute_query(sample_query)

        if samples.empty:
            print("  ❌ No recent SERVQUAL processed reviews found")
        else:
            print(f"  Sample of {len(samples)} recent processed reviews:")
            for _, review in samples.iterrows():
                app_name = app_names.get(review['app_id'], review['app_id'])
                content_preview = review['content'][:50] + "..." if len(review['content']) > 50 else review['content']
                print(f"    {app_name} - Rating {review['rating']}: {content_preview}")

    except Exception as e:
        print(f"❌ Sample reviews check failed: {e}")

    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS COMPLETE")
    print("\nPossible issues causing low conversion:")
    print("1. ❌ Individual LLM results not stored - only final aggregates")
    print("2. ❌ LLM detection rate very low (most reviews don't mention SERVQUAL dimensions)")
    print("3. ❌ Aggregation logic too strict (filtering out most results)")
    print("4. ❌ Date/time grouping issues preventing proper aggregation")
    print("5. ❌ Pipeline bug: LLM runs but results don't get properly stored")
    print("\nRecommended investigation:")
    print("- Check src/pipeline/batch.py SERVQUAL processing logic")
    print("- Check src/absa/servqual_llm_model.py LLM detection rates")
    print("- Check src/data/servqual_storage.py aggregation logic")


if __name__ == "__main__":
    main()