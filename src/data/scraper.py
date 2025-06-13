"""
Google Play Store review scraper with validation and deduplication.
Handles both batch and real-time scraping with rate limiting and error handling.
Enhanced with historical date range scraping functionality.
"""

import time
import logging
import hashlib
import uuid
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
from google_play_scraper import app, reviews, Sort
from google_play_scraper.exceptions import NotFoundError, ExtraHTTPError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.config import config
from src.data.storage import storage


@dataclass
class ReviewData:
    """Structured review data model."""
    review_id: str
    app_id: str
    user_name: str
    content: str
    rating: int
    thumbs_up_count: int
    review_created_version: Optional[str]
    review_date: Any  # Changed from str to Any to handle date objects
    reply_content: Optional[str]
    reply_date: Optional[Any]  # Changed from Optional[str] to Optional[Any]
    language: str = 'en'
    is_spam: bool = False
    content_length: Optional[int] = None

    def __post_init__(self):
        """Calculate content length and generate review_id if missing."""
        if self.content:
            self.content_length = len(self.content)

        if not self.review_id:
            # Generate consistent review_id based on content + user + date
            hash_input = f"{self.app_id}:{self.user_name}:{self.content}:{self.review_date}"
            self.review_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_input))


class ReviewValidator:
    """Validates and cleans scraped review data."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.scraper.validator")
        self.spam_keywords = {
            'download', 'click here', 'free money', 'visit site', 'hack',
            'cheat', 'generator', 'unlimited', 'fake', 'bot'
        }
        self.min_content_length = 5  # Reduced from 10 for testing
        self.max_content_length = 5000

    def is_valid_review(self, review_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate review data quality and authenticity."""

        # Check required fields (using actual Google Play API field names)
        required_fields = ['content', 'score', 'userName']  # Changed 'rating' to 'score'
        for field in required_fields:
            if not review_data.get(field):
                return False, f"Missing required field: {field}"

        content = review_data['content'].strip()

        # Content length validation
        if len(content) < self.min_content_length:
            return False, "Content too short"

        if len(content) > self.max_content_length:
            return False, "Content too long"

        # Rating validation (using 'score' field from Google Play API)
        rating = review_data.get('score', 0)
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return False, "Invalid rating"

        # Spam detection
        if self.is_spam(content):
            return False, "Detected as spam"

        # Language detection (basic)
        if self.is_non_english(content):
            return False, "Non-English content"

        return True, "Valid"

    def is_spam(self, content: str) -> bool:
        """Basic spam detection based on keywords and patterns."""
        content_lower = content.lower()

        # Check for spam keywords
        spam_count = sum(1 for keyword in self.spam_keywords if keyword in content_lower)
        if spam_count >= 2:
            return True

        # Check for excessive repetition
        words = content_lower.split()
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                return True

        # Check for excessive caps
        if len(content) > 20:
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if caps_ratio > 0.6:  # More than 60% caps
                return True

        return False

    def is_non_english(self, content: str) -> bool:
        """Basic English language detection."""
        # Simple heuristic: check for English alphabet ratio
        english_chars = sum(1 for c in content if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in content if c.isalpha())

        if total_chars == 0:
            return True

        english_ratio = english_chars / total_chars
        return english_ratio < 0.5  # Reduced from 0.7 to be less strict

    def clean_content(self, content: str) -> str:
        """Clean and normalize review content."""
        # Remove excessive whitespace
        content = ' '.join(content.split())

        # Remove special characters but keep punctuation
        # This is a basic implementation - can be enhanced
        return content.strip()

    def normalize_review_data(self, raw_review: Dict[str, Any], app_id: str) -> Optional[ReviewData]:
        """Convert raw scraped data to normalized ReviewData object."""
        try:
            # Handle review date conversion
            review_date = raw_review.get('at')
            if review_date:
                if isinstance(review_date, datetime):
                    review_date = review_date.date()
                elif isinstance(review_date, str):
                    from datetime import datetime as dt
                    review_date = dt.strptime(review_date, '%Y-%m-%d').date()
            else:
                review_date = datetime.now().date()

            # Handle reply date conversion
            reply_date = raw_review.get('repliedAt')
            if reply_date:
                if isinstance(reply_date, datetime):
                    reply_date = reply_date.date()
                elif isinstance(reply_date, str):
                    from datetime import datetime as dt
                    try:
                        reply_date = dt.strptime(reply_date, '%Y-%m-%d').date()
                    except:
                        reply_date = None
            else:
                reply_date = None

            # Map Google Play Scraper fields to our schema
            review_data = ReviewData(
                review_id="",  # Will be generated in __post_init__
                app_id=app_id,
                user_name=raw_review.get('userName', 'Anonymous'),
                content=self.clean_content(raw_review.get('content', '')),
                rating=raw_review.get('score', 0),
                thumbs_up_count=raw_review.get('thumbsUpCount', 0),
                review_created_version=raw_review.get('reviewCreatedVersion'),
                review_date=review_date,  # Now using date object
                reply_content=raw_review.get('replyContent'),
                reply_date=reply_date,  # Now using date object or None
                language='en',
                is_spam=False
            )

            return review_data

        except Exception as e:
            self.logger.error(f"Error normalizing review data: {e}")
            return None


class ReviewDeduplicator:
    """Handles review deduplication using multiple strategies."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.scraper.deduplicator")
        self.seen_hashes: Set[str] = set()
        self.seen_review_ids: Set[str] = set()

    def generate_content_hash(self, content: str, user_name: str, rating: int) -> str:
        """Generate hash for content-based deduplication."""
        hash_input = f"{content.lower().strip()}:{user_name.lower()}:{rating}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def is_duplicate(self, review: ReviewData) -> bool:
        """Check if review is a duplicate using multiple criteria."""

        # Check by review_id
        if review.review_id in self.seen_review_ids:
            return True

        # Check by content hash
        content_hash = self.generate_content_hash(
            review.content, review.user_name, review.rating
        )

        if content_hash in self.seen_hashes:
            return True

        # Add to seen sets
        self.seen_review_ids.add(review.review_id)
        self.seen_hashes.add(content_hash)

        return False

    def load_existing_review_ids(self, app_id: str, days_back: int = 30):
        """Load existing review IDs and content hashes from database."""
        try:
            # Get recent reviews with content for hash-based deduplication
            query = """
            SELECT review_id, user_name, content, rating FROM reviews 
            WHERE app_id = :app_id 
            AND scraped_at >= CURRENT_DATE - INTERVAL '%s days'
            """ % days_back

            result = storage.db.execute_query(query, {"app_id": app_id})

            if not result.empty:
                # Add review IDs
                existing_ids = set(result['review_id'].tolist())
                self.seen_review_ids.update(existing_ids)

                # Add content hashes
                for _, row in result.iterrows():
                    content_hash = self.generate_content_hash(
                        row['content'], row['user_name'], row['rating']
                    )
                    self.seen_hashes.add(content_hash)

            self.logger.info(f"Loaded {len(existing_ids)} existing review IDs for {app_id}")

        except Exception as e:
            self.logger.error(f"Error loading existing review IDs: {e}")


class PlayStoreScraper:
    """Main Google Play Store scraper with rate limiting and error handling."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.scraper")
        self.validator = ReviewValidator()
        self.deduplicator = ReviewDeduplicator()

        # Configuration (must be set before creating session)
        self.max_reviews = config.scraping.max_reviews_per_app
        self.delay = config.scraping.scraping_delay
        self.timeout = config.scraping.request_timeout
        self.max_retries = config.scraping.max_retries

        # Create session after configuration is set
        self.session = self._create_session()

        # Statistics
        self.stats = {
            'scraped': 0,
            'valid': 0,
            'invalid': 0,
            'duplicates': 0,
            'stored': 0,
            'errors': 0
        }

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()

        try:
            # Try with new parameter name first
            retry_strategy = Retry(
                total=self.max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )
        except TypeError:
            # Fallback to old parameter name for older urllib3 versions
            retry_strategy = Retry(
                total=self.max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            'User-Agent': config.scraping.user_agent
        })

        return session

    def get_app_info(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Scrape app metadata from Google Play Store."""
        try:
            self.logger.info(f"Scraping app info for: {app_id}")

            app_info = app(app_id, lang='en', country='us')

            # Convert timestamp to date if needed
            last_updated = app_info.get('updated')
            if isinstance(last_updated, (int, float)):
                # Convert Unix timestamp to date
                last_updated = datetime.fromtimestamp(last_updated).date()
            elif isinstance(last_updated, datetime):
                last_updated = last_updated.date()
            elif not last_updated:
                last_updated = datetime.now().date()

            # Normalize app data
            normalized_app = {
                'app_id': app_id,
                'app_name': app_info.get('title', ''),
                'category': app_info.get('genre', ''),
                'developer': app_info.get('developer', ''),
                'rating': float(app_info.get('score', 0.0)),
                'installs': app_info.get('installs', ''),
                'price': app_info.get('price', 'Free'),
                'content_rating': app_info.get('contentRating', ''),
                'last_updated': last_updated
            }

            self.logger.info(f"Successfully scraped app info: {normalized_app['app_name']}")
            return normalized_app

        except NotFoundError:
            self.logger.error(f"App not found: {app_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error scraping app info for {app_id}: {e}")
            return None

    def scrape_reviews(self, app_id: str, count: Optional[int] = None,
                      sort_order: Sort = Sort.NEWEST) -> List[ReviewData]:
        """Scrape reviews for a specific app."""
        if count is None:
            count = self.max_reviews

        self.logger.info(f"Starting review scraping for {app_id}, target: {count} reviews")

        # Load existing reviews to avoid duplicates
        self.deduplicator.load_existing_review_ids(app_id)

        valid_reviews = []
        continuation_token = None
        total_attempts = 0
        max_attempts = 20  # Safety limit to prevent infinite loops

        try:
            while len(valid_reviews) < count and total_attempts < max_attempts:
                total_attempts += 1

                # Calculate batch size (max 200 per API call)
                batch_size = min(200, count - len(valid_reviews))

                self.logger.info(f"Scraping batch {total_attempts} of {batch_size} reviews...")

                # Scrape batch of reviews
                result, continuation_token = reviews(
                    app_id,
                    lang='en',
                    country='us',
                    sort=sort_order,
                    count=batch_size,
                    continuation_token=continuation_token
                )

                if not result:
                    self.logger.info("No more reviews available from API")
                    break

                self.logger.info(f"API returned {len(result)} raw reviews")

                # Process batch
                batch_valid = self._process_review_batch(result, app_id)
                valid_reviews.extend(batch_valid)

                self.stats['scraped'] += len(result)

                self.logger.info(f"Batch {total_attempts}: {len(batch_valid)} valid reviews added. Total valid: {len(valid_reviews)}")

                # If we got some valid reviews or no continuation token, we can be satisfied
                if len(batch_valid) > 0 or not continuation_token:
                    if len(valid_reviews) > 0:
                        self.logger.info(f"Breaking early - found {len(valid_reviews)} valid reviews")
                        break

                # Rate limiting
                time.sleep(self.delay)

                # Break if no continuation token (reached end)
                if not continuation_token:
                    self.logger.info("Reached end of available reviews")
                    break

            if total_attempts >= max_attempts:
                self.logger.warning(f"Stopped scraping after {max_attempts} attempts to prevent infinite loop")

            self.logger.info(f"Scraping completed. Valid reviews: {len(valid_reviews)}")
            return valid_reviews

        except ExtraHTTPError as e:
            self.logger.error(f"HTTP error while scraping reviews: {e}")
            self.stats['errors'] += 1
            return valid_reviews
        except Exception as e:
            self.logger.error(f"Unexpected error while scraping reviews: {e}")
            self.stats['errors'] += 1
            return valid_reviews

    def _process_review_batch(self, raw_reviews: List[Dict], app_id: str) -> List[ReviewData]:
        """Process a batch of raw reviews through validation and deduplication."""
        valid_reviews = []

        self.logger.info(f"Processing batch of {len(raw_reviews)} raw reviews")

        for i, raw_review in enumerate(raw_reviews):
            try:
                # Validate review
                is_valid, reason = self.validator.is_valid_review(raw_review)
                if not is_valid:
                    self.logger.debug(f"Review {i+1} invalid: {reason}")
                    self.stats['invalid'] += 1
                    continue

                # Normalize review data
                review_data = self.validator.normalize_review_data(raw_review, app_id)
                if not review_data:
                    self.logger.debug(f"Review {i+1} failed normalization")
                    self.stats['invalid'] += 1
                    continue

                # Check for duplicates
                if self.deduplicator.is_duplicate(review_data):
                    self.logger.debug(f"Review {i+1} is duplicate")
                    self.stats['duplicates'] += 1
                    continue

                # Mark as spam if detected
                if self.validator.is_spam(review_data.content):
                    self.logger.debug(f"Review {i+1} marked as spam")
                    review_data.is_spam = True

                valid_reviews.append(review_data)
                self.stats['valid'] += 1
                self.logger.debug(f"Review {i+1} added - {review_data.user_name}: {review_data.rating}/5")

            except Exception as e:
                self.logger.error(f"Error processing review {i+1}: {e}")
                self.stats['errors'] += 1

        self.logger.info(f"Batch processing complete: {len(valid_reviews)} valid out of {len(raw_reviews)} raw reviews")
        return valid_reviews

    def scrape_and_store_app(self, app_id: str, update_app_info: bool = True) -> Dict[str, Any]:
        """Complete workflow: scrape app info and reviews, then store in database."""
        start_time = time.time()

        self.logger.info(f"Starting complete scraping workflow for {app_id}")

        # Reset stats for this app
        self.stats = {k: 0 for k in self.stats}

        try:
            # Scrape and store app info
            if update_app_info:
                app_info = self.get_app_info(app_id)
                if app_info:
                    success = storage.apps.store_app(app_info)
                    if success:
                        self.logger.info(f"Stored app info for {app_info['app_name']}")
                    else:
                        self.logger.warning(f"Failed to store app info for {app_id}")

            # Scrape reviews
            reviews_data = self.scrape_reviews(app_id)

            # Store reviews in database
            if reviews_data:
                review_dicts = [asdict(review) for review in reviews_data]
                stored_count, error_count = storage.reviews.store_reviews(review_dicts)
                self.stats['stored'] = stored_count
                self.stats['errors'] += error_count

            # Calculate execution time
            execution_time = time.time() - start_time

            # Prepare summary
            summary = {
                'app_id': app_id,
                'execution_time': round(execution_time, 2),
                'statistics': self.stats.copy(),
                'success': True
            }

            self.logger.info(f"Scraping completed for {app_id}: {summary}")
            return summary

        except Exception as e:
            execution_time = time.time() - start_time
            error_summary = {
                'app_id': app_id,
                'execution_time': round(execution_time, 2),
                'error': str(e),
                'statistics': self.stats.copy(),
                'success': False
            }

            self.logger.error(f"Scraping failed for {app_id}: {error_summary}")
            return error_summary

    def scrape_multiple_apps(self, app_ids: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple apps sequentially with progress tracking."""
        results = []

        self.logger.info(f"Starting batch scraping for {len(app_ids)} apps")

        for i, app_id in enumerate(app_ids, 1):
            self.logger.info(f"Processing app {i}/{len(app_ids)}: {app_id}")

            result = self.scrape_and_store_app(app_id)
            results.append(result)

            # Delay between apps to avoid rate limiting
            if i < len(app_ids):
                time.sleep(self.delay * 2)

        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        total_reviews = sum(r['statistics']['stored'] for r in results)

        self.logger.info(f"Batch scraping completed: {successful}/{len(app_ids)} apps successful, {total_reviews} total reviews stored")

        return results

    def scrape_historical_app(self, app_id: str, start_date: date, end_date: date,
                            max_reviews: int = 500) -> Dict[str, Any]:
        """
        Scrape historical reviews for a specific app within date range.

        Args:
            app_id: Application identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            max_reviews: Maximum reviews to attempt

        Returns:
            Dictionary with scraping results and statistics
        """
        start_time = time.time()

        self.logger.info(f"Historical scraping {app_id}: {start_date} to {end_date}")

        # Reset stats
        self.stats = {k: 0 for k in self.stats}
        historical_stats = {
            'target_reviews': max_reviews,
            'date_filtered': 0,
            'historical_stored': 0,
            'outside_range': 0
        }

        try:
            # Load existing reviews to avoid duplicates
            self.deduplicator.load_existing_review_ids(app_id, days_back=365)

            historical_reviews = []
            continuation_token = None
            total_attempts = 0
            max_attempts = 50  # Higher limit for historical scraping
            reviews_outside_range_count = 0
            consecutive_old_reviews = 0

            while (len(historical_reviews) < max_reviews and
                   total_attempts < max_attempts and
                   consecutive_old_reviews < 10):  # Stop if 10 consecutive reviews are too old

                total_attempts += 1
                batch_size = min(200, max_reviews - len(historical_reviews))

                self.logger.info(f"Historical batch {total_attempts}: requesting {batch_size} reviews")

                # Scrape batch
                result, continuation_token = reviews(
                    app_id,
                    lang='en',
                    country='us',
                    sort=Sort.NEWEST,  # Start with newest
                    count=batch_size,
                    continuation_token=continuation_token
                )

                if not result:
                    self.logger.info("No more reviews available")
                    break

                self.stats['scraped'] += len(result)

                # Process and filter batch by date
                batch_historical = self._process_historical_batch(
                    result, app_id, start_date, end_date, historical_stats
                )

                # Check if we're getting reviews outside our date range
                batch_dates = []
                for raw_review in result:
                    review_date = self._extract_review_date(raw_review)
                    if review_date:
                        batch_dates.append(review_date)

                # Count consecutive old reviews
                if batch_dates and max(batch_dates) < start_date:
                    consecutive_old_reviews += 1
                    self.logger.info(f"Batch {total_attempts}: All reviews before {start_date}, "
                                   f"consecutive old batches: {consecutive_old_reviews}")
                else:
                    consecutive_old_reviews = 0

                historical_reviews.extend(batch_historical)

                self.logger.info(f"Historical batch {total_attempts}: {len(batch_historical)} valid historical reviews. "
                                f"Total: {len(historical_reviews)}")

                # Rate limiting
                time.sleep(self.delay)

                if not continuation_token:
                    self.logger.info("Reached end of available reviews")
                    break

            # Store historical reviews
            if historical_reviews:
                review_dicts = [asdict(review) for review in historical_reviews]
                stored_count, error_count = storage.reviews.store_reviews(review_dicts)
                historical_stats['historical_stored'] = stored_count
                self.stats['stored'] = stored_count
                self.stats['errors'] += error_count

            execution_time = time.time() - start_time

            # Prepare summary
            summary = {
                'app_id': app_id,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'execution_time': round(execution_time, 2),
                'statistics': self.stats.copy(),
                'historical_statistics': historical_stats,
                'success': True
            }

            self.logger.info(f"Historical scraping completed for {app_id}: {summary}")
            return summary

        except Exception as e:
            execution_time = time.time() - start_time
            error_summary = {
                'app_id': app_id,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'execution_time': round(execution_time, 2),
                'error': str(e),
                'statistics': self.stats.copy(),
                'historical_statistics': historical_stats,
                'success': False
            }

            self.logger.error(f"Historical scraping failed for {app_id}: {error_summary}")
            return error_summary

    def _extract_review_date(self, raw_review: Dict[str, Any]) -> Optional[date]:
        """Extract review date from raw review data with debugging."""
        try:
            review_date = raw_review.get('at')

            # Debug: Print raw date information
            self.logger.info(f"Debug: Raw review date field 'at': {review_date} (type: {type(review_date)})")

            if review_date:
                if isinstance(review_date, datetime):
                    extracted_date = review_date.date()
                    self.logger.info(f"Debug: Extracted date from datetime: {extracted_date}")
                    return extracted_date
                elif isinstance(review_date, str):
                    try:
                        extracted_date = datetime.strptime(review_date, '%Y-%m-%d').date()
                        self.logger.info(f"Debug: Extracted date from string: {extracted_date}")
                        return extracted_date
                    except ValueError:
                        # Try alternative date format
                        try:
                            extracted_date = datetime.strptime(review_date, '%Y-%m-%d %H:%M:%S').date()
                            self.logger.info(f"Debug: Extracted date from datetime string: {extracted_date}")
                            return extracted_date
                        except ValueError:
                            self.logger.warning(f"Debug: Could not parse date string: {review_date}")
                            return None
                else:
                    self.logger.warning(f"Debug: Unexpected date type: {type(review_date)}")
                    return None
            else:
                self.logger.warning("Debug: No 'at' field found in review")
                # Check for alternative date fields
                for field_name in ['date', 'reviewDate', 'created', 'timestamp']:
                    if field_name in raw_review:
                        self.logger.info(
                            f"Debug: Found alternative date field '{field_name}': {raw_review[field_name]}")
                return None

        except Exception as e:
            self.logger.error(f"Debug: Error extracting review date: {e}")
            return None

    def _process_historical_batch(self, raw_reviews: List[Dict], app_id: str,
                                  start_date: date, end_date: date,
                                  historical_stats: Dict) -> List[ReviewData]:
        """Process batch with historical date filtering and enhanced debugging."""
        historical_reviews = []

        self.logger.info(f"Debug: Processing {len(raw_reviews)} raw reviews for date range {start_date} to {end_date}")

        # Debug: Show structure of first review
        if raw_reviews:
            first_review = raw_reviews[0]
            self.logger.info(f"Debug: First review keys: {list(first_review.keys())}")
            self.logger.info(f"Debug: First review sample: {dict(list(first_review.items())[:5])}")

        dates_found = []
        dates_in_range = []

        for i, raw_review in enumerate(raw_reviews):
            try:
                # Extract review date first for filtering
                review_date = self._extract_review_date(raw_review)

                if review_date:
                    dates_found.append(review_date)
                    self.logger.info(f"Debug: Review {i + 1} date: {review_date}")
                else:
                    self.logger.warning(f"Debug: Review {i + 1} - no date extracted")
                    continue

                # Date range filtering
                if review_date < start_date or review_date > end_date:
                    historical_stats['outside_range'] += 1
                    self.logger.info(f"Debug: Review {i + 1} date {review_date} outside range {start_date}-{end_date}")
                    continue

                dates_in_range.append(review_date)
                historical_stats['date_filtered'] += 1
                self.logger.info(f"Debug: Review {i + 1} date {review_date} WITHIN range - processing")

                # Standard validation and processing
                is_valid, reason = self.validator.is_valid_review(raw_review)
                if not is_valid:
                    self.logger.info(f"Debug: Review {i + 1} invalid: {reason}")
                    self.stats['invalid'] += 1
                    continue

                review_data = self.validator.normalize_review_data(raw_review, app_id)
                if not review_data:
                    self.logger.info(f"Debug: Review {i + 1} failed normalization")
                    self.stats['invalid'] += 1
                    continue

                if self.deduplicator.is_duplicate(review_data):
                    self.logger.info(f"Debug: Review {i + 1} is duplicate")
                    self.stats['duplicates'] += 1
                    continue

                if self.validator.is_spam(review_data.content):
                    review_data.is_spam = True
                    self.logger.info(f"Debug: Review {i + 1} marked as spam")

                historical_reviews.append(review_data)
                self.stats['valid'] += 1
                self.logger.info(f"Debug: Review {i + 1} ACCEPTED - Content: {review_data.content[:50]}...")

            except Exception as e:
                self.logger.error(f"Debug: Error processing historical review {i + 1}: {e}")
                self.stats['errors'] += 1

        # Summary debug info
        self.logger.info(f"Debug: Batch summary:")
        self.logger.info(f"  - Total reviews processed: {len(raw_reviews)}")
        self.logger.info(f"  - Dates found: {len(dates_found)}")
        self.logger.info(f"  - Dates in range: {len(dates_in_range)}")
        self.logger.info(f"  - Valid historical reviews: {len(historical_reviews)}")

        if dates_found:
            self.logger.info(f"  - Date range in batch: {min(dates_found)} to {max(dates_found)}")

        return historical_reviews

    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get current scraping statistics."""
        return self.stats.copy()


# Convenience functions for easy usage
def scrape_app_reviews(app_id: str, count: int = 1000) -> Dict[str, Any]:
    """Convenience function to scrape reviews for a single app."""
    scraper = PlayStoreScraper()
    return scraper.scrape_and_store_app(app_id)


def scrape_multiple_apps(app_ids: List[str]) -> List[Dict[str, Any]]:
    """Convenience function to scrape reviews for multiple apps."""
    scraper = PlayStoreScraper()
    return scraper.scrape_multiple_apps(app_ids)


def update_app_info(app_id: str) -> bool:
    """Convenience function to update app metadata only."""
    scraper = PlayStoreScraper()
    app_info = scraper.get_app_info(app_id)
    if app_info:
        return storage.apps.store_app(app_info)
    return False


def scrape_historical_range(app_ids: List[str], start_date: date, end_date: date,
                          max_reviews_per_app: int = 500) -> List[Dict[str, Any]]:
    """
    Scrape historical reviews within specific date range for multiple apps.

    Args:
        app_ids: List of app IDs to scrape
        start_date: Start date for historical scraping (inclusive)
        end_date: End date for historical scraping (inclusive)
        max_reviews_per_app: Maximum reviews to attempt per app

    Returns:
        List of results for each app
    """
    scraper = PlayStoreScraper()
    results = []

    scraper.logger.info(f"Starting historical scraping: {start_date} to {end_date}")
    scraper.logger.info(f"Apps: {len(app_ids)}, Max reviews per app: {max_reviews_per_app}")

    for i, app_id in enumerate(app_ids, 1):
        scraper.logger.info(f"Historical scraping app {i}/{len(app_ids)}: {app_id}")

        result = scraper.scrape_historical_app(
            app_id=app_id,
            start_date=start_date,
            end_date=end_date,
            max_reviews=max_reviews_per_app
        )
        results.append(result)

        # Delay between apps
        if i < len(app_ids):
            time.sleep(scraper.delay * 2)

    # Summary
    total_historical = sum(r.get('historical_stored', 0) for r in results)
    successful_apps = sum(1 for r in results if r.get('success', False))

    scraper.logger.info(f"Historical scraping completed: {successful_apps}/{len(app_ids)} apps, "
                       f"{total_historical} historical reviews stored")

    return results