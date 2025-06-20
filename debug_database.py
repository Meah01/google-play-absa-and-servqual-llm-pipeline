"""
Database Diagnostic Script for ABSA Dashboard
Run this to check what data exists and identify dashboard issues.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("🔍 ABSA Database Diagnostic Tool")
    print("=" * 50)

    try:
        from src.data.storage import storage
        print("✅ Storage module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import storage: {e}")
        return

    # Test 1: Database Connection
    print("\n📡 Testing Database Connection...")
    try:
        health = storage.health_check()
        print(f"Database health: {health}")

        if health.get('overall') != 'healthy':
            print("❌ Database connection issues detected!")
            return
        else:
            print("✅ Database connection OK")

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    # Test 2: Check Table Structure
    print("\n📋 Checking Table Structure...")
    try:
        tables_query = """
        SELECT table_name, 
               (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
        FROM information_schema.tables t 
        WHERE table_schema = 'public' 
        ORDER BY table_name
        """

        tables = storage.db.execute_query(tables_query)
        print("=== EXISTING TABLES ===")
        for _, row in tables.iterrows():
            print(f"  {row['table_name']}: {row['column_count']} columns")

    except Exception as e:
        print(f"❌ Failed to check tables: {e}")
        return

    # Test 3: Check Critical Tables
    print("\n🔎 Checking Critical Tables...")
    critical_tables = ['reviews', 'apps', 'deep_absa', 'servqual_scores']

    for table in critical_tables:
        try:
            # Check if table exists and has data
            count_query = f"SELECT COUNT(*) as count FROM {table}"
            result = storage.db.execute_query(count_query)
            count = result.iloc[0]['count']

            # Check table structure
            columns_query = f"""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = '{table}' 
            ORDER BY ordinal_position
            """
            columns = storage.db.execute_query(columns_query)

            print(f"\n=== {table.upper()} TABLE ===")
            print(f"  Records: {count:,}")
            print(f"  Columns: {', '.join(columns['column_name'].tolist())}")

            if count == 0:
                print(f"  ⚠️  WARNING: {table} table is empty!")

        except Exception as e:
            print(f"  ❌ {table}: {e}")

    # Test 4: Check Processing Status
    print("\n⚙️  Checking Processing Status...")
    try:
        # Check reviews table structure first
        sample_reviews = storage.db.execute_query("SELECT * FROM reviews LIMIT 1")
        review_columns = sample_reviews.columns.tolist()
        print(f"Reviews table columns: {review_columns}")

        # Check if processing flags exist
        if 'absa_processed' in review_columns and 'servqual_processed' in review_columns:
            processing_check = storage.db.execute_query("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE absa_processed = TRUE) as absa_done,
                    COUNT(*) FILTER (WHERE servqual_processed = TRUE) as servqual_done
                FROM reviews 
                WHERE NOT COALESCE(is_spam, FALSE)
            """)

            result = processing_check.iloc[0]
            print(f"  Total reviews: {result['total']:,}")
            print(f"  ABSA processed: {result['absa_done']:,}")
            print(f"  SERVQUAL processed: {result['servqual_done']:,}")

            if result['absa_done'] == 0:
                print("  ⚠️  WARNING: No ABSA processing completed!")
            if result['servqual_done'] == 0:
                print("  ⚠️  WARNING: No SERVQUAL processing completed!")
        else:
            print("  ❌ Processing flag columns missing from reviews table!")
            print("     Expected: absa_processed, servqual_processed")

    except Exception as e:
        print(f"  ❌ Processing status check failed: {e}")

    # Test 5: Check Data for Dashboard
    print("\n📊 Checking Dashboard Data Requirements...")
    try:
        # Check ABSA data
        absa_count = storage.db.execute_query("SELECT COUNT(*) as count FROM deep_absa")
        absa_records = absa_count.iloc[0]['count']
        print(f"  ABSA records: {absa_records:,}")

        if absa_records == 0:
            print("  ⚠️  WARNING: No ABSA data found! Dashboard ABSA section will be empty.")

        # Check SERVQUAL data
        servqual_count = storage.db.execute_query("SELECT COUNT(*) as count FROM servqual_scores")
        servqual_records = servqual_count.iloc[0]['count']
        print(f"  SERVQUAL records: {servqual_records:,}")

        if servqual_records == 0:
            print("  ⚠️  WARNING: No SERVQUAL data found! Dashboard SERVQUAL section will be empty.")

        # Check apps data
        apps_count = storage.db.execute_query("SELECT COUNT(*) as count FROM apps")
        apps_records = apps_count.iloc[0]['count']
        print(f"  Apps records: {apps_records:,}")

        if apps_records == 0:
            print("  ❌ ERROR: No apps data found! Dashboard will not work.")

    except Exception as e:
        print(f"  ❌ Dashboard data check failed: {e}")

    # Test 6: Test Dashboard Data Loader
    print("\n🎛️  Testing Dashboard Data Loader...")
    try:
        from dashboard.data_loader import dashboard_data_loader
        print("  ✅ Data loader imported successfully")

        # Test specific dashboard functions
        print("  Testing Amazon fixed ABSA data...")
        amazon_data = dashboard_data_loader.load_amazon_fixed_absa(days=30)
        print(f"    Amazon ABSA aspects found: {len(amazon_data.get('aspects', []))}")

        print("  Testing processing status...")
        status_data = dashboard_data_loader.load_processing_status()
        print(f"    Active jobs: {len(status_data.get('active_jobs', []))}")
        print(f"    ABSA queue: {status_data.get('queue_metrics', {}).get('total_pending_absa', 0)}")

    except Exception as e:
        print(f"  ❌ Dashboard data loader test failed: {e}")

    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("\nNext steps based on results:")
    print("1. If tables are missing → Run schema.sql")
    print("2. If data is missing → Run scraping + processing")
    print("3. If columns are missing → Update database schema")
    print("4. If processing flags missing → Run schema updates")


if __name__ == "__main__":
    main()