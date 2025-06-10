"""
Storage operations for ABSA Pipeline.
Handles PostgreSQL database operations and Redis caching.
Provides unified interface for all data storage and retrieval.
"""

import json
import logging
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager
import pandas as pd
from sqlalchemy import create_engine, text, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import redis
from redis.exceptions import RedisError

from src.utils.config import config


class DatabaseConnection:
    """Manages PostgreSQL database connections and sessions."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.storage.db")
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

    def _initialize_connection(self):
        """Initialize database engine with connection pooling."""
        if self._initialized:
            return

        try:
            self.engine = create_engine(
                config.database.connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=config.debug
            )

            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self.logger.info("Database connection initialized successfully")
            self._initialized = True

        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            raise

    @contextmanager
    def get_session(self):
        """Context manager for database sessions with automatic cleanup."""
        if not self._initialized:
            self._initialize_connection()

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        if not self._initialized:
            self._initialize_connection()

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise

    def execute_non_query(self, query: str, params: Dict = None) -> int:
        """Execute non-query SQL statement (INSERT, UPDATE, DELETE)."""
        if not self._initialized:
            self._initialize_connection()

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result.rowcount
        except Exception as e:
            self.logger.error(f"Non-query execution failed: {e}")
            raise


class RedisConnection:
    """Manages Redis connections and operations."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.storage.redis")
        self.client = None
        self._initialized = False

    def _initialize_connection(self):
        """Initialize Redis connection."""
        if self._initialized:
            return

        try:
            self.client = redis.Redis(**config.redis.connection_params)

            # Test connection
            self.client.ping()

            self.logger.info("Redis connection initialized successfully")
            self._initialized = True

        except Exception as e:
            self.logger.error(f"Failed to initialize Redis connection: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with JSON deserialization."""
        if not self._initialized:
            self._initialize_connection()

        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            self.logger.error(f"Redis GET error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with JSON serialization."""
        if not self._initialized:
            self._initialize_connection()

        try:
            json_value = json.dumps(value, default=str)
            if ttl:
                return self.client.setex(key, ttl, json_value)
            else:
                return self.client.set(key, json_value)
        except (RedisError, json.JSONEncodeError) as e:
            self.logger.error(f"Redis SET error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self._initialized:
            self._initialize_connection()

        try:
            return bool(self.client.delete(key))
        except RedisError as e:
            self.logger.error(f"Redis DELETE error for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self._initialized:
            self._initialize_connection()

        try:
            return bool(self.client.exists(key))
        except RedisError as e:
            self.logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False


class AppStorage:
    """Handles app-related database operations."""

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.logger = logging.getLogger("absa_pipeline.storage.apps")

    def store_app(self, app_data: Dict[str, Any]) -> bool:
        """Store or update app information."""
        query = """
        INSERT INTO apps (app_id, app_name, category, developer, rating, installs, 
                         price, content_rating, last_updated)
        VALUES (:app_id, :app_name, :category, :developer, :rating, :installs,
                :price, :content_rating, :last_updated)
        ON CONFLICT (app_id) 
        DO UPDATE SET 
            app_name = EXCLUDED.app_name,
            category = EXCLUDED.category,
            developer = EXCLUDED.developer,
            rating = EXCLUDED.rating,
            installs = EXCLUDED.installs,
            price = EXCLUDED.price,
            content_rating = EXCLUDED.content_rating,
            last_updated = EXCLUDED.last_updated,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            self.db.execute_non_query(query, app_data)
            self.logger.info(f"Stored app: {app_data.get('app_id')}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store app {app_data.get('app_id')}: {e}")
            return False

    def get_app(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve app information by app_id."""
        query = "SELECT * FROM apps WHERE app_id = :app_id"

        try:
            df = self.db.execute_query(query, {"app_id": app_id})
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve app {app_id}: {e}")
            return None

    def get_all_apps(self) -> List[Dict[str, Any]]:
        """Retrieve all apps with basic statistics."""
        query = """
        SELECT a.*, rs.total_reviews, rs.avg_rating as review_avg_rating,
               rs.processed_reviews, rs.latest_review_date
        FROM apps a
        LEFT JOIN review_summary rs ON a.app_id = rs.app_id
        ORDER BY a.app_name
        """

        try:
            df = self.db.execute_query(query)
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Failed to retrieve all apps: {e}")
            return []


class ReviewStorage:
    """Handles review-related database operations."""

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.logger = logging.getLogger("absa_pipeline.storage.reviews")

    def store_reviews(self, reviews: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Store multiple reviews. Returns (inserted, errors)."""
        inserted_count = 0
        error_count = 0

        for review in reviews:
            if self.store_review(review):
                inserted_count += 1
            else:
                error_count += 1

        self.logger.info(f"Stored {inserted_count} reviews, {error_count} errors")
        return inserted_count, error_count

    def store_review(self, review_data: Dict[str, Any]) -> bool:
        """Store single review."""
        # Generate UUID if not provided
        if 'review_id' not in review_data:
            review_data['review_id'] = str(uuid.uuid4())

        # Ensure all required fields have default values
        review_data.setdefault('thumbs_up_count', 0)
        review_data.setdefault('review_created_version', None)
        review_data.setdefault('reply_content', None)
        review_data.setdefault('reply_date', None)
        review_data.setdefault('language', 'en')
        review_data.setdefault('is_spam', False)

        query = """
        INSERT INTO reviews (review_id, app_id, user_name, content, rating,
                           thumbs_up_count, review_created_version, review_date,
                           reply_content, reply_date, language, is_spam)
        VALUES (:review_id, :app_id, :user_name, :content, :rating,
                :thumbs_up_count, :review_created_version, :review_date,
                :reply_content, :reply_date, :language, :is_spam)
        ON CONFLICT (review_id) DO NOTHING
        """

        try:
            rows_affected = self.db.execute_non_query(query, review_data)
            return rows_affected > 0
        except Exception as e:
            self.logger.error(f"Failed to store review {review_data.get('review_id')}: {e}")
            return False

    def get_reviews_for_processing(self, app_id: Optional[str] = None,
                                 limit: int = 1000) -> List[Dict[str, Any]]:
        """Get unprocessed reviews for ABSA analysis."""
        base_query = """
        SELECT review_id, app_id, content, rating, review_date
        FROM reviews 
        WHERE processed = FALSE AND is_spam = FALSE
        """

        params = {"limit": limit}

        if app_id:
            base_query += " AND app_id = :app_id"
            params["app_id"] = app_id

        base_query += " ORDER BY review_date DESC LIMIT :limit"

        try:
            df = self.db.execute_query(base_query, params)
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Failed to get reviews for processing: {e}")
            return []

    def mark_reviews_processed(self, review_ids: List[str]) -> int:
        """Mark reviews as processed."""
        if not review_ids:
            return 0

        placeholders = ','.join([f"'{rid}'" for rid in review_ids])
        query = f"""
        UPDATE reviews 
        SET processed = TRUE
        WHERE review_id IN ({placeholders})
        """

        try:
            return self.db.execute_non_query(query)
        except Exception as e:
            self.logger.error(f"Failed to mark reviews as processed: {e}")
            return 0

    def get_review_stats(self, app_id: str) -> Dict[str, Any]:
        """Get review statistics for an app."""
        query = """
        SELECT 
            COUNT(*) as total_reviews,
            AVG(rating) as avg_rating,
            COUNT(*) FILTER (WHERE processed = TRUE) as processed_reviews,
            COUNT(*) FILTER (WHERE is_spam = TRUE) as spam_reviews,
            MIN(review_date) as oldest_review,
            MAX(review_date) as newest_review
        FROM reviews 
        WHERE app_id = :app_id
        """

        try:
            df = self.db.execute_query(query, {"app_id": app_id})
            if not df.empty:
                return df.iloc[0].to_dict()
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get review stats for {app_id}: {e}")
            return {}


class ABSAStorage:
    """Handles ABSA results storage and retrieval."""

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.logger = logging.getLogger("absa_pipeline.storage.absa")

    def store_deep_absa_results(self, results: List[Dict[str, Any]]) -> int:
        """Store deep ABSA analysis results."""
        if not results:
            return 0

        query = """
        INSERT INTO deep_absa (review_id, app_id, aspect, sentiment_score,
                              confidence_score, opinion_text, opinion_start_pos,
                              opinion_end_pos, processing_model, processing_version)
        VALUES (:review_id, :app_id, :aspect, :sentiment_score,
                :confidence_score, :opinion_text, :opinion_start_pos,
                :opinion_end_pos, :processing_model, :processing_version)
        """

        inserted_count = 0
        for result in results:
            try:
                rows = self.db.execute_non_query(query, result)
                inserted_count += rows
            except Exception as e:
                self.logger.error(f"Failed to store deep ABSA result: {e}")

        self.logger.info(f"Stored {inserted_count} deep ABSA results")
        return inserted_count

    def store_quick_absa_result(self, review_id: str, app_id: str,
                               aspects_sentiment: Dict[str, float],
                               overall_sentiment: float,
                               processing_time_ms: int) -> bool:
        """Store quick ABSA analysis result."""
        query = """
        INSERT INTO quick_absa (review_id, app_id, aspects_sentiment,
                               overall_sentiment, processing_time_ms)
        VALUES (:review_id, :app_id, :aspects_sentiment,
                :overall_sentiment, :processing_time_ms)
        ON CONFLICT (review_id) 
        DO UPDATE SET 
            aspects_sentiment = EXCLUDED.aspects_sentiment,
            overall_sentiment = EXCLUDED.overall_sentiment,
            processing_time_ms = EXCLUDED.processing_time_ms,
            created_at = CURRENT_TIMESTAMP
        """

        params = {
            "review_id": review_id,
            "app_id": app_id,
            "aspects_sentiment": json.dumps(aspects_sentiment),
            "overall_sentiment": overall_sentiment,
            "processing_time_ms": processing_time_ms
        }

        try:
            self.db.execute_non_query(query, params)
            return True
        except Exception as e:
            self.logger.error(f"Failed to store quick ABSA result: {e}")
            return False

    def get_aspect_sentiment_trends(self, app_id: str,
                                  days: int = 30) -> pd.DataFrame:
        """Get aspect sentiment trends for dashboard visualization."""
        query = """
        SELECT date, aspect, avg_sentiment, sentiment_count,
               positive_count, negative_count, neutral_count,
               trend_direction, trend_magnitude
        FROM daily_aspect_sentiment
        WHERE app_id = :app_id 
        AND date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY date DESC, aspect
        """ % days

        try:
            return self.db.execute_query(query, {"app_id": app_id})
        except Exception as e:
            self.logger.error(f"Failed to get aspect sentiment trends: {e}")
            return pd.DataFrame()

    def get_app_sentiment_summary(self, app_id: str) -> Dict[str, Any]:
        """Get overall sentiment summary for an app."""
        query = """
        SELECT 
            aspect,
            AVG(avg_sentiment) as avg_sentiment,
            SUM(sentiment_count) as total_reviews,
            SUM(positive_count) as positive_count,
            SUM(negative_count) as negative_count,
            SUM(neutral_count) as neutral_count
        FROM daily_aspect_sentiment
        WHERE app_id = :app_id
        AND date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY aspect
        ORDER BY total_reviews DESC
        """

        try:
            df = self.db.execute_query(query, {"app_id": app_id})
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Failed to get sentiment summary for {app_id}: {e}")
            return []


class CacheStorage:
    """Handles Redis caching operations for real-time data."""

    def __init__(self, redis_conn: RedisConnection):
        self.redis = redis_conn
        self.logger = logging.getLogger("absa_pipeline.storage.cache")
        self.default_ttl = config.dashboard.cache_ttl

    def cache_absa_result(self, review_id: str, absa_data: Dict[str, Any],
                         ttl: Optional[int] = None) -> bool:
        """Cache ABSA result for real-time access."""
        key = f"absa:{review_id}"
        return self.redis.set(key, absa_data, ttl or self.default_ttl)

    def get_cached_absa(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached ABSA result."""
        key = f"absa:{review_id}"
        return self.redis.get(key)

    def cache_realtime_trends(self, app_id: str, trends_data: Dict[str, Any],
                            ttl: Optional[int] = None) -> bool:
        """Cache real-time trend data for an app."""
        key = f"trends:{app_id}"
        return self.redis.set(key, trends_data, ttl or self.default_ttl)

    def get_realtime_trends(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get cached real-time trends."""
        key = f"trends:{app_id}"
        return self.redis.get(key)

    def cache_dashboard_data(self, data_key: str, data: Any,
                           ttl: Optional[int] = None) -> bool:
        """Cache dashboard data for performance."""
        key = f"dashboard:{data_key}"
        return self.redis.set(key, data, ttl or self.default_ttl)

    def get_dashboard_data(self, data_key: str) -> Optional[Any]:
        """Get cached dashboard data."""
        key = f"dashboard:{data_key}"
        return self.redis.get(key)

    def invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate multiple cache keys matching pattern."""
        try:
            keys = self.redis.client.keys(pattern)
            if keys:
                return self.redis.client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"Failed to invalidate cache pattern {pattern}: {e}")
            return 0


class Storage:
    """Unified storage interface combining database and cache operations."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.storage")

        # Initialize connections (but don't connect yet)
        self.db = DatabaseConnection()
        self.redis = RedisConnection()

        # Initialize storage handlers
        self.apps = AppStorage(self.db)
        self.reviews = ReviewStorage(self.db)
        self.absa = ABSAStorage(self.db)
        self.cache = CacheStorage(self.redis)

        self.logger.info("Storage layer initialized (connections will be made on first use)")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all storage components."""
        health_status = {
            "database": "unknown",
            "redis": "unknown",
            "overall": "unknown"
        }

        # Check database
        try:
            if not self.db._initialized:
                self.db._initialize_connection()
            self.db.execute_query("SELECT 1")
            health_status["database"] = "healthy"
        except Exception as e:
            health_status["database"] = f"unhealthy: {str(e)}"

        # Check Redis
        try:
            if not self.redis._initialized:
                self.redis._initialize_connection()
            self.redis.client.ping()
            health_status["redis"] = "healthy"
        except Exception as e:
            health_status["redis"] = f"unhealthy: {str(e)}"

        # Overall status
        if all(status == "healthy" for status in [health_status["database"], health_status["redis"]]):
            health_status["overall"] = "healthy"
        else:
            health_status["overall"] = "unhealthy"

        return health_status

    def close_connections(self):
        """Close all database and cache connections."""
        try:
            if self.db.engine:
                self.db.engine.dispose()
            if self.redis.client:
                self.redis.client.close()
            self.logger.info("All storage connections closed")
        except Exception as e:
            self.logger.error(f"Error closing storage connections: {e}")


# Global storage instance
storage = Storage()