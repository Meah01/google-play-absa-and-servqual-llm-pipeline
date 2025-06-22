"""
Cleanup Stuck Processing Jobs
Save as: cleanup_jobs.py and run once
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("🧹 Cleaning Up Stuck Processing Jobs")
    print("=" * 40)

    try:
        from src.data.storage import storage
        print("✅ Storage imported")
    except Exception as e:
        print(f"❌ Storage import failed: {e}")
        return

    # Check current stuck jobs
    print("\n📋 Checking stuck jobs...")
    try:
        stuck_query = """
        SELECT job_id, job_type, status, current_phase, start_time
        FROM processing_jobs 
        WHERE status = 'running' 
        AND start_time < CURRENT_TIMESTAMP - INTERVAL '2 hours'
        """

        stuck_jobs = storage.db.execute_query(stuck_query)

        if stuck_jobs.empty:
            print("  ✅ No stuck jobs found!")
            return

        print(f"  ⚠️  Found {len(stuck_jobs)} stuck jobs:")
        for _, job in stuck_jobs.iterrows():
            print(f"    {job['job_id']}: {job['job_type']} ({job['current_phase']}) - started {job['start_time']}")

        # Clean them up
        print("\n🔧 Cleaning up stuck jobs...")
        cleanup_query = """
        UPDATE processing_jobs 
        SET status = 'failed', 
            end_time = CURRENT_TIMESTAMP,
            error_message = 'Cleaned up stuck job - likely interrupted or zombie process'
        WHERE status = 'running' 
        AND start_time < CURRENT_TIMESTAMP - INTERVAL '2 hours'
        """

        cleaned_count = storage.db.execute_non_query(cleanup_query)
        print(f"  ✅ Cleaned up {cleaned_count} stuck jobs")

        # Also clean up old checkpoints
        print("\n🧹 Cleaning up old checkpoints...")
        checkpoint_cleanup = """
        UPDATE processing_checkpoints 
        SET status = 'failed'
        WHERE status = 'active' 
        AND checkpoint_time < CURRENT_TIMESTAMP - INTERVAL '2 hours'
        """

        checkpoint_cleaned = storage.db.execute_non_query(checkpoint_cleanup)
        print(f"  ✅ Cleaned up {checkpoint_cleaned} old checkpoints")

        print(f"\n🎯 Cleanup complete! System ready for new processing jobs.")

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")


if __name__ == "__main__":
    main()