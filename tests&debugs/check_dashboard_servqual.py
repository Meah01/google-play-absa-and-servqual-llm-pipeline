"""
Check Dashboard SERVQUAL Display
Save as: check_dashboard_servqual.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("🎛️ Dashboard SERVQUAL Display Diagnostic")
    print("=" * 50)

    try:
        from src.data.storage import storage
        from src.data.servqual_storage import servqual_storage
        print("✅ Storage modules imported")
    except Exception as e:
        print(f"❌ Storage import failed: {e}")
        return

    # Check what Amazon SERVQUAL data the dashboard sees
    print("\n🛒 Amazon SERVQUAL Dashboard Data:")
    try:
        # Test different time periods
        for days in [7, 14, 30]:
            amazon_data = servqual_storage.get_amazon_focus_data(days=days)

            print(f"\n  📅 Last {days} days:")

            current_profile = amazon_data.get('current_profile', {})
            if current_profile:
                overall_quality = current_profile.get('overall_quality', 0)
                dimensions = current_profile.get('dimensions', {})
                total_reviews = current_profile.get('total_reviews', 0)

                print(f"    Overall Quality: {overall_quality:.2f}/5")
                print(f"    Total Reviews: {total_reviews}")
                print(f"    Dimensions: {len(dimensions)}")

                for dim_name, dim_data in dimensions.items():
                    quality = dim_data.get('quality_score', 0)
                    review_count = dim_data.get('review_count', 0)
                    print(f"      {dim_name}: {quality}/5 ({review_count} reviews)")
            else:
                print(f"    ❌ No Amazon profile data found")

            # Check trends
            trends = amazon_data.get('trends', [])
            print(f"    Trend data points: {len(trends)}")

            # Check competitive ranking
            ranking = amazon_data.get('competitive_ranking', {})
            print(f"    Competitive rankings: {len(ranking)} dimensions")

    except Exception as e:
        print(f"❌ Amazon SERVQUAL check failed: {e}")

    # Check comparative analysis across apps
    print("\n🏪 Comparative Analysis Across Apps:")
    try:
        dimensions = ['reliability', 'assurance', 'tangibles', 'empathy', 'responsiveness']

        for dimension in dimensions:
            comp_data = servqual_storage.get_comparative_analysis(dimension, days=30)

            print(f"\n  📊 {dimension.title()} (last 30 days):")
            if comp_data.empty:
                print(f"    ❌ No data found for {dimension}")
            else:
                print(f"    ✅ {len(comp_data)} apps have {dimension} data")
                for _, row in comp_data.iterrows():
                    app_name = row['app_name']
                    avg_quality = row['avg_quality']
                    total_reviews = row['total_reviews']
                    print(f"      {app_name}: {avg_quality:.2f}/5 ({total_reviews} reviews)")

    except Exception as e:
        print(f"❌ Comparative analysis check failed: {e}")

    # Check raw SERVQUAL scores table
    print("\n📋 Raw SERVQUAL Scores (Recent):")
    try:
        recent_scores_query = """
        SELECT 
            app_id,
            dimension,
            quality_score,
            review_count,
            date,
            created_at
        FROM servqual_scores 
        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY created_at DESC
        LIMIT 20
        """

        recent_scores = storage.db.execute_query(recent_scores_query)

        if recent_scores.empty:
            print("  ❌ No recent SERVQUAL scores found!")
        else:
            print(f"  ✅ {len(recent_scores)} recent SERVQUAL scores:")

            app_names = {
                'com.amazon.mShop.android.shopping': 'Amazon',
                'com.ebay.mobile': 'eBay',
                'com.einnovation.temu': 'Temu',
                'com.etsy.android': 'Etsy',
                'com.zzkko': 'SHEIN'
            }

            for _, score in recent_scores.iterrows():
                app_short = app_names.get(score['app_id'], score['app_id'])
                print(
                    f"    {app_short} - {score['dimension']}: {score['quality_score']}/5 ({score['review_count']} reviews) - {score['date']}")

    except Exception as e:
        print(f"❌ Recent scores check failed: {e}")

    # Check if dashboard components work
    print("\n🎨 Dashboard Components Test:")
    try:
        from dashboard.servqual_components import servqual_dashboard
        print("  ✅ SERVQUAL dashboard components can be imported")

        # Test dashboard data loader
        from dashboard.data_loader import dashboard_data_loader

        # Check processing status
        status_data = dashboard_data_loader.load_processing_status()
        active_jobs = status_data.get('active_jobs', [])
        queue_metrics = status_data.get('queue_metrics', {})

        print(f"  Active jobs: {len(active_jobs)}")
        print(f"  SERVQUAL queue: {queue_metrics.get('total_pending_servqual', 0)}")

    except Exception as e:
        print(f"❌ Dashboard components test failed: {e}")

    print("\n" + "=" * 50)
    print("🎯 DASHBOARD DIAGNOSTIC COMPLETE")
    print("\nPossible reasons for 'only 2 scores' display:")
    print("1. Dashboard filtering by recent dates (last 7 days)")
    print("2. Dashboard aggregation issues")
    print("3. SERVQUAL scores not being grouped properly")
    print("4. Cache issues in dashboard display")
    print("\nSolution: Check SERVQUAL tab in dashboard with different time periods")


if __name__ == "__main__":
    main()