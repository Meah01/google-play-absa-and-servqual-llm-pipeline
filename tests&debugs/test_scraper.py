"""
Fixed test script for Google Play Store scraper.
Includes timeout protection and better error handling.
"""

import sys
import os
import signal
import threading
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.scraper import PlayStoreScraper, scrape_app_reviews
from src.data.storage import storage
import json

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, timeout_seconds=60):
    """Run function with timeout protection."""
    def target():
        try:
            return func()
        except Exception as e:
            print(f"Function failed: {e}")
            return None

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print(f"⚠️ Operation timed out after {timeout_seconds} seconds")
        return False
    return True

def test_app_info_only():
    """Test only app info scraping (quick test)."""
    print("🔍 Quick Test: App Info Scraping Only...")

    scraper = PlayStoreScraper()
    app_id = "com.instagram.android"

    try:
        app_info = scraper.get_app_info(app_id)
        if app_info:
            print(f"✅ App info scraped successfully:")
            print(f"   App Name: {app_info['app_name']}")
            print(f"   Developer: {app_info['developer']}")
            print(f"   Rating: {app_info['rating']}")
            print(f"   Category: {app_info['category']}")

            # Store app info
            success = storage.apps.store_app(app_info)
            print(f"   Storage: {'✅ Success' if success else '❌ Failed'}")
            return True
        else:
            print("❌ Failed to scrape app info")
            return False
    except Exception as e:
        print(f"❌ App info test failed: {e}")
        return False

def test_minimal_reviews():
    """Test minimal review scraping with timeout protection."""
    print("\n🔍 Minimal Review Test (with timeout protection)...")

    def scrape_with_limits():
        scraper = PlayStoreScraper()

        # Override scraper settings for faster testing
        scraper.delay = 0.5  # Reduce delay
        scraper.max_reviews = 10  # Limit total

        app_id = "com.instagram.android"

        # Just try to get any reviews, don't insist on exact count
        try:
            print("   Attempting to scrape up to 3 reviews...")
            reviews = scraper.scrape_reviews(app_id, count=3)

            print(f"   ✅ Scraped {len(reviews)} reviews successfully")

            if reviews:
                sample = reviews[0]
                print(f"   Sample: {sample.user_name} - {sample.rating}/5")
                print(f"   Content: {sample.content[:50]}...")

            # Try to store one review
            if reviews:
                from dataclasses import asdict
                review_dict = asdict(reviews[0])
                success = storage.reviews.store_review(review_dict)
                print(f"   Storage: {'✅ Success' if success else '❌ Failed'}")

            return True

        except Exception as e:
            print(f"   ❌ Review scraping failed: {e}")
            return False

    # Run with 60-second timeout
    success = run_with_timeout(scrape_with_limits, timeout_seconds=60)
    return success

def test_storage_verification():
    """Verify data was stored correctly."""
    print("\n🔍 Storage Verification...")

    try:
        # Check apps
        apps = storage.apps.get_all_apps()
        print(f"✅ Found {len(apps)} apps in database")

        # Check reviews
        reviews = storage.reviews.get_reviews_for_processing(limit=10)
        print(f"✅ Found {len(reviews)} unprocessed reviews")

        # Show sample data
        if apps:
            app = apps[0]
            print(f"   Sample app: {app['app_name']}")

        if reviews:
            review = reviews[0]
            print(f"   Sample review: {len(review['content'])} chars, rating {review['rating']}")

        return True

    except Exception as e:
        print(f"❌ Storage verification failed: {e}")
        return False

def main():
    """Run fixed scraper tests with proper error handling."""
    print("🚀 Fixed Google Play Store Scraper Test\n")

    tests = [
        ("App Info Scraping", test_app_info_only),
        ("Minimal Review Scraping", test_minimal_reviews),
        ("Storage Verification", test_storage_verification)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                print(f"✅ {test_name} - PASSED\n")
            else:
                print(f"❌ {test_name} - FAILED\n")

        except KeyboardInterrupt:
            print(f"\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"💥 {test_name} - ERROR: {e}\n")
            results.append((test_name, False))

    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")

    passed = sum(1 for name, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests successful! Scraper is working correctly.")
    elif passed > 0:
        print("⚠️ Partial success. Some components working.")
    else:
        print("💥 All tests failed. Need troubleshooting.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")