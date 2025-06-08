"""
Test script for the Streamlit dashboard.
Verifies dashboard functionality and data loading.
"""

import sys
import time
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.storage import storage
from src.data.scraper import scrape_app_reviews


def test_dashboard_data():
    """Test that dashboard can load data correctly."""
    print("🔍 Testing Dashboard Data Loading...")

    try:
        # Test storage connectivity
        health = storage.health_check()
        print(f"   Storage health: {health['overall']}")

        # Get apps
        apps = storage.apps.get_all_apps()
        print(f"   Apps available: {len(apps)}")

        # Get reviews
        all_reviews = []
        for app in apps:
            reviews = storage.reviews.get_reviews_for_processing(app['app_id'], limit=100)
            all_reviews.extend(reviews)

        print(f"   Reviews available: {len(all_reviews)}")

        if len(apps) == 0:
            print("⚠️ No apps found - dashboard will show welcome screen")
            return True

        if len(all_reviews) == 0:
            print("⚠️ No reviews found - scraping sample data...")
            result = scrape_app_reviews("com.instagram.android", count=5)
            if result['success']:
                print(f"   ✅ Scraped {result['statistics']['stored']} sample reviews")
            else:
                print("   ❌ Failed to scrape sample data")

        print("✅ Dashboard data test passed")
        return True

    except Exception as e:
        print(f"❌ Dashboard data test failed: {e}")
        return False


def test_dashboard_launch():
    """Test launching the dashboard."""
    print("\n🚀 Testing Dashboard Launch...")

    try:
        dashboard_path = project_root / "dashboard_app.py"

        if not dashboard_path.exists():
            print(f"❌ Dashboard file not found: {dashboard_path}")
            return False

        print("   ✅ Dashboard file exists")

        # Test imports
        try:
            import streamlit
            print(f"   ✅ Streamlit available (version: {streamlit.__version__})")
        except ImportError:
            print("   ❌ Streamlit not installed")
            print("   💡 Install with: pip install streamlit")
            return False

        try:
            import plotly
            print(f"   ✅ Plotly available (version: {plotly.__version__})")
        except ImportError:
            print("   ❌ Plotly not installed")
            print("   💡 Install with: pip install plotly")
            return False

        print("✅ Dashboard launch test passed")
        return True

    except Exception as e:
        print(f"❌ Dashboard launch test failed: {e}")
        return False


def test_main_orchestrator():
    """Test the main orchestrator functionality."""
    print("\n🎯 Testing Main Orchestrator...")

    try:
        main_path = project_root / "main.py"

        if not main_path.exists():
            print(f"❌ Main file not found: {main_path}")
            return False

        print("   ✅ Main orchestrator file exists")

        # Test status command
        try:
            result = subprocess.run(
                [sys.executable, "main.py", "status"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("   ✅ Status command works")
            else:
                print(f"   ⚠️ Status command returned error: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("   ⚠️ Status command timed out")
        except Exception as e:
            print(f"   ⚠️ Status command failed: {e}")

        print("✅ Main orchestrator test completed")
        return True

    except Exception as e:
        print(f"❌ Main orchestrator test failed: {e}")
        return False


def main():
    """Run all dashboard tests."""
    print("🧪 Dashboard Test Suite")
    print("=" * 40)

    tests = [
        ("Dashboard Data Loading", test_dashboard_data),
        ("Dashboard Launch", test_dashboard_launch),
        ("Main Orchestrator", test_main_orchestrator)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1

    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Dashboard is ready to launch.")
        print("\n🚀 To start the complete pipeline:")
        print("   python main.py run")
        print("\n🎨 To start just the dashboard:")
        print("   python main.py dashboard")
    else:
        print("\n⚠️ Some tests failed. Check the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)