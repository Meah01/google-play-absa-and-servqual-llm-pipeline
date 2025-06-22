"""
Quick test script for storage layer functionality.
Run this to verify database and Redis connections work.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.storage import storage
import json

def test_storage_connections():
    """Test basic storage connectivity."""
    print("Testing Storage Layer...")

    # Health check
    health = storage.health_check()
    print(f"Health Status: {json.dumps(health, indent=2)}")

    # Test database operations
    print("\n--- Testing Database ---")

    # Get existing apps
    apps = storage.apps.get_all_apps()
    print(f"Found {len(apps)} apps in database")

    if apps:
        app = apps[0]
        print(f"Sample app: {app['app_name']} ({app['app_id']})")

        # Get app details
        app_details = storage.apps.get_app(app['app_id'])
        print(f"App rating: {app_details.get('rating', 'N/A')}")

    # Test Redis operations
    print("\n--- Testing Redis Cache ---")

    # Test cache operations
    test_data = {"test": "data", "timestamp": "2024-01-15"}
    cache_result = storage.cache.cache_dashboard_data("test_key", test_data)
    print(f"Cache store result: {cache_result}")

    cached_data = storage.cache.get_dashboard_data("test_key")
    print(f"Retrieved from cache: {cached_data}")

    # Test review storage
    print("\n--- Testing Review Operations ---")

    sample_review = {
        "app_id": "com.instagram.android",
        "user_name": "TestUser",
        "content": "This is a test review for storage testing",
        "rating": 4,
        "review_date": "2024-01-15",
        "language": "en",
        "is_spam": False
    }

    success = storage.reviews.store_review(sample_review)
    print(f"Review storage result: {success}")

    # Get reviews for processing
    unprocessed = storage.reviews.get_reviews_for_processing(limit=5)
    print(f"Unprocessed reviews: {len(unprocessed)}")

    print("\n✅ Storage layer test completed!")

if __name__ == "__main__":
    test_storage_connections()