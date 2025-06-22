"""
Check Processing Results - Diagnose SERVQUAL processing issue
Save as: check_processing.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("🔍 Processing Results Diagnostic")
    print("=" * 50)

    try:
        from src.data.storage import storage
        print("✅ Storage imported")
    except Exception as e:
        print(f"❌ Storage import failed: {e}")
        return

    # Check recent processing activity
    print("\n📊 Recent Processing Activity (Last 24 Hours):")
    try:
        recent_query = """
        SELECT 
            COUNT(*) as total_reviews,
            COUNT(*) FILTER (WHERE absa_processed = TRUE) as absa_processed,
            COUNT(*) FILTER (WHERE servqual_processed = TRUE) as servqual_processed,
            COUNT(*) FILTER (WHERE absa_processed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours') as absa_last_24h,
            COUNT(*) FILTER (WHERE servqual_processed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours') as servqual_last_24h
        FROM reviews 
        WHERE NOT COALESCE(is_spam, FALSE)
        """

        result = storage.db.execute_query(recent_query)
        row = result.iloc[0]

        print(f"  Total reviews: {row['total_reviews']:,}")
        print(f"  ABSA processed (total): {row['absa_processed']:,}")
        print(f"  SERVQUAL processed (total): {row['servqual_processed']:,}")
        print(f"  ABSA processed (last 24h): {row['absa_last_24h']:,}")
        print(f"  SERVQUAL processed (last 24h): {row['servqual_last_24h']:,}")

        if row['absa_last_24h'] > 0 and row['servqual_last_24h'] == 0:
            print("  ⚠️  WARNING: ABSA ran but SERVQUAL didn't process any reviews!")

    except Exception as e:
        print(f"❌ Recent activity check failed: {e}")

    # Check SERVQUAL scores table
    print("\n🤖 SERVQUAL Scores Analysis:")
    try:
        servqual_query = """
        SELECT 
            app_id,
            dimension,
            COUNT(*) as records,
            MAX(date) as latest_date,
            AVG(quality_score) as avg_quality,
            SUM(review_count) as total_reviews_used
        FROM servqual_scores 
        GROUP BY app_id, dimension
        ORDER BY app_id, dimension
        """

        servqual_df = storage.db.execute_query(servqual_query)

        if servqual_df.empty:
            print("  ❌ No SERVQUAL scores found!")
        else:
            print(f"  📋 SERVQUAL scores found: {len(servqual_df)} records")

            # Group by app
            app_summary = servqual_df.groupby('app_id').agg({
                'records': 'sum',
                'latest_date': 'max',
                'total_reviews_used': 'sum'
            }).reset_index()

            for _, app_row in app_summary.iterrows():
                print(
                    f"    {app_row['app_id']}: {app_row['records']} scores, {app_row['total_reviews_used']} reviews, latest: {app_row['latest_date']}")

    except Exception as e:
        print(f"❌ SERVQUAL scores check failed: {e}")

    # Check processing jobs
    print("\n⚙️  Processing Jobs History:")
    try:
        jobs_query = """
        SELECT 
            job_id,
            job_type,
            status,
            current_phase,
            start_time,
            end_time,
            records_processed,
            error_message
        FROM processing_jobs 
        WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL '48 hours'
        ORDER BY start_time DESC
        LIMIT 10
        """

        jobs_df = storage.db.execute_query(jobs_query)

        if jobs_df.empty:
            print("  ⚠️  No recent processing jobs found")
        else:
            print(f"  📋 Recent jobs: {len(jobs_df)}")
            for _, job in jobs_df.iterrows():
                status = job['status']
                phase = job['current_phase']
                processed = job['records_processed'] or 0
                error = job['error_message']

                print(f"    {job['job_type']}: {status} (phase: {phase})")
                print(f"      Processed: {processed}, Started: {job['start_time']}")
                if error:
                    print(f"      Error: {error}")
                print()

    except Exception as e:
        print(f"❌ Processing jobs check failed: {e}")

    # Check if LLM SERVQUAL model is available
    print("\n🔧 LLM SERVQUAL Availability Check:")
    try:
        # Try to import LLM model
        from src.absa.servqual_llm_model import ServqualLLM
        print("  ✅ LLM SERVQUAL model can be imported")

        # Try to initialize (but don't actually call LLM)
        llm_model = ServqualLLM()
        print("  ✅ LLM SERVQUAL model can be initialized")

    except Exception as e:
        print(f"  ❌ LLM SERVQUAL model error: {e}")
        print("    This might explain why SERVQUAL processing failed!")

    # Check deep_absa vs servqual processing mismatch
    print("\n🔍 Processing Mismatch Analysis:")
    try:
        mismatch_query = """
        SELECT 
            COUNT(*) as reviews_with_absa_no_servqual,
            MIN(absa_processed_at) as earliest_absa,
            MAX(absa_processed_at) as latest_absa
        FROM reviews 
        WHERE absa_processed = TRUE 
        AND servqual_processed = FALSE
        AND NOT COALESCE(is_spam, FALSE)
        """

        mismatch_df = storage.db.execute_query(mismatch_query)
        row = mismatch_df.iloc[0]

        unprocessed_count = row['reviews_with_absa_no_servqual']
        print(f"  Reviews with ABSA but no SERVQUAL: {unprocessed_count:,}")

        if unprocessed_count > 0:
            print(f"  Earliest ABSA processing: {row['earliest_absa']}")
            print(f"  Latest ABSA processing: {row['latest_absa']}")
            print("  💡 These reviews should be processed by SERVQUAL LLM!")

    except Exception as e:
        print(f"❌ Mismatch analysis failed: {e}")

    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS COMPLETE")
    print("\nPossible issues:")
    print("1. Sequential processing only ran ABSA, not SERVQUAL")
    print("2. LLM SERVQUAL model failed to load/run")
    print("3. SERVQUAL processing was skipped due to errors")
    print("4. Configuration issue with LLM integration")


if __name__ == "__main__":
    main()