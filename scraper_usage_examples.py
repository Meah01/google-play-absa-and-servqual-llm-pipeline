"""
Usage examples for the Google Play Store scraper.
Demonstrates different ways to use the scraper for various scenarios.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.scraper import PlayStoreScraper, scrape_app_reviews, scrape_multiple_apps


# Example 1: Basic single app scraping
def example_single_app():
    """Scrape reviews for a single app."""
    print("Example 1: Single App Scraping")

    app_id = "com.netflix.mediaclient"  # Netflix

    # Simple one-line usage
    result = scrape_app_reviews(app_id, count=50)

    print(f"Scraped {result['statistics']['stored']} reviews for Netflix")
    print(f"Execution time: {result['execution_time']} seconds")


# Example 2: Batch scraping multiple apps
def example_multiple_apps():
    """Scrape reviews for multiple apps."""
    print("\nExample 2: Multiple App Scraping")

    app_ids = [
        "com.instagram.android",
        "com.whatsapp",
        "com.spotify.music",
        "com.netflix.mediaclient",
        "com.google.android.youtube"
    ]

    results = scrape_multiple_apps(app_ids)

    print("Batch scraping results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['app_id']}: {result['statistics']['stored']} reviews")


# Example 3: Advanced scraper usage with custom settings
def example_advanced_scraper():
    """Advanced scraper usage with customization."""
    print("\nExample 3: Advanced Scraper Usage")

    # Create scraper instance for more control
    scraper = PlayStoreScraper()

    app_id = "com.instagram.android"

    # 1. Get app info only
    app_info = scraper.get_app_info(app_id)
    print(f"App: {app_info['app_name']} by {app_info['developer']}")

    # 2. Scrape reviews with specific settings
    reviews = scraper.scrape_reviews(app_id, count=20)
    print(f"Scraped {len(reviews)} reviews")

    # 3. Get scraping statistics
    stats = scraper.get_scraping_stats()
    print(f"Stats: {stats}")

    # 4. Process reviews manually if needed
    for review in reviews[:3]:  # Show first 3 reviews
        print(f"- {review.user_name}: {review.rating}/5 stars")
        print(f"  {review.content[:100]}...")


# Example 4: Daily scraping routine
def example_daily_scraping():
    """Example of a daily scraping routine."""
    print("\nExample 4: Daily Scraping Routine")

    # Apps to monitor daily
    monitored_apps = [
        "com.instagram.android",
        "com.whatsapp",
        "com.tiktok.android",
        "com.spotify.music"
    ]

    scraper = PlayStoreScraper()
    daily_results = []

    for app_id in monitored_apps:
        print(f"Processing {app_id}...")

        # Scrape recent reviews (smaller batch for daily updates)
        result = scraper.scrape_and_store_app(app_id)
        daily_results.append(result)

        print(f"  Stored {result['statistics']['stored']} new reviews")

    # Summary
    total_reviews = sum(r['statistics']['stored'] for r in daily_results)
    print(f"\nDaily scraping complete: {total_reviews} total new reviews")


# Example 5: Error handling and monitoring
def example_error_handling():
    """Demonstrate error handling in scraping."""
    print("\nExample 5: Error Handling")

    # Mix of valid and invalid app IDs
    app_ids = [
        "com.instagram.android",  # Valid
        "invalid.app.id",  # Invalid
        "com.whatsapp",  # Valid
        "another.invalid.app"  # Invalid
    ]

    scraper = PlayStoreScraper()

    for app_id in app_ids:
        try:
            result = scraper.scrape_and_store_app(app_id)

            if result['success']:
                print(f"✅ {app_id}: {result['statistics']['stored']} reviews")
            else:
                print(f"❌ {app_id}: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"💥 {app_id}: Unexpected error - {e}")


if __name__ == "__main__":
    print("🚀 Google Play Store Scraper Usage Examples\n")

    # Run all examples
    try:
        example_single_app()
        example_multiple_apps()
        example_advanced_scraper()
        example_daily_scraping()
        example_error_handling()

        print("\n🎉 All examples completed!")

    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Example execution failed: {e}")