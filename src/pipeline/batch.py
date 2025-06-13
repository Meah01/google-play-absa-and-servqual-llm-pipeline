"""
Batch processing pipeline for ABSA Pipeline.
Coordinates deep ABSA analysis, SERVQUAL mapping, and data aggregation.
Integrates the ABSA engine with existing SERVQUAL infrastructure for business intelligence.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
import pandas as pd

from src.utils.config import config
from src.data.storage import storage
from src.absa.engine import absa_engine, AnalysisMode, BatchProcessingResult
from src.absa.servqual_mapper import ServqualMapper, process_app_servqual
from src.data.servqual_storage import servqual_storage


@dataclass
class BatchJobResult:
    """Result of a complete batch processing job."""
    job_id: str
    app_id: Optional[str]
    start_time: datetime
    end_time: datetime
    reviews_processed: int
    aspects_extracted: int
    servqual_dimensions_updated: int
    processing_time_seconds: float
    success: bool
    error_message: Optional[str]
    statistics: Dict[str, Any]


@dataclass
class BatchConfiguration:
    """Configuration for batch processing operations."""
    batch_size: int
    max_reviews_per_job: int
    confidence_threshold: float
    enable_servqual_processing: bool
    cleanup_old_data: bool
    retention_days: int


class ProcessingJobTracker:
    """Tracks batch processing jobs in the database."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.batch.tracker")

    def create_job(self, job_type: str, app_id: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> str:
        """Create a new processing job record."""
        job_id = str(uuid.uuid4())

        try:
            query = """
            INSERT INTO processing_jobs (job_id, job_type, app_id, status, start_time, metadata)
            VALUES (:job_id, :job_type, :app_id, 'running', :start_time, :metadata)
            """

            # Convert metadata dict to JSON string for JSONB column
            metadata_json = None
            if metadata:
                import json
                metadata_json = json.dumps(metadata)

            params = {
                'job_id': job_id,
                'job_type': job_type,
                'app_id': app_id,
                'start_time': datetime.now(),
                'metadata': metadata_json  # Now properly converted to JSON string
            }

            storage.db.execute_non_query(query, params)
            self.logger.info(f"Created processing job: {job_id}")
            return job_id

        except Exception as e:
            self.logger.error(f"Error creating processing job: {e}")
            raise

    def update_job_progress(self, job_id: str, records_processed: int):
        """Update job progress."""
        try:
            query = """
            UPDATE processing_jobs 
            SET records_processed = :records_processed
            WHERE job_id = :job_id
            """

            storage.db.execute_non_query(query, {
                'job_id': job_id,
                'records_processed': records_processed
            })

        except Exception as e:
            self.logger.error(f"Error updating job progress: {e}")

        except Exception as e:
            self.logger.error(f"Error updating job progress: {e}")

    def complete_job(self, job_id: str, success: bool, records_processed: int,
                     error_message: Optional[str] = None):
        """Mark job as completed."""
        try:
            status = 'completed' if success else 'failed'

            query = """
            UPDATE processing_jobs 
            SET status = :status, end_time = :end_time, records_processed = :records_processed,
                error_message = :error_message
            WHERE job_id = :job_id
            """

            storage.db.execute_non_query(query, {
                'job_id': job_id,
                'status': status,
                'end_time': datetime.now(),
                'records_processed': records_processed,
                'error_message': error_message
            })

            self.logger.info(f"Job {job_id} completed with status: {status}")

        except Exception as e:
            self.logger.error(f"Error completing job: {e}")


class DataLifecycleManager:
    """Manages data retention and cleanup operations."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.batch.lifecycle")

    def cleanup_old_processing_jobs(self, retention_days: int = 30) -> int:
        """Clean up old processing job records."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            query = """
            DELETE FROM processing_jobs 
            WHERE created_at < :cutoff_date AND status IN ('completed', 'failed')
            """

            deleted_count = storage.db.execute_non_query(query, {'cutoff_date': cutoff_date})

            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old processing job records")

            return deleted_count

        except Exception as e:
            self.logger.error(f"Error cleaning up processing jobs: {e}")
            return 0

    def cleanup_old_absa_results(self, retention_days: int = 90) -> int:
        """Clean up old ABSA results based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Clean up old deep_absa records
            query = """
            DELETE FROM deep_absa 
            WHERE created_at < :cutoff_date
            """

            deleted_count = storage.db.execute_non_query(query, {'cutoff_date': cutoff_date})

            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old ABSA result records")

            return deleted_count

        except Exception as e:
            self.logger.error(f"Error cleaning up ABSA results: {e}")
            return 0

    def aggregate_daily_sentiment(self, target_date: date = None) -> int:
        """Aggregate daily sentiment data for reporting."""
        if target_date is None:
            target_date = datetime.now().date()

        try:
            # Aggregate aspect sentiment by app and date
            query = """
            INSERT INTO daily_aspect_sentiment 
            (app_id, date, aspect, avg_sentiment, sentiment_count, 
             positive_count, negative_count, neutral_count)
            SELECT 
                da.app_id,
                :target_date as date,
                da.aspect,
                AVG(da.sentiment_score) as avg_sentiment,
                COUNT(*) as sentiment_count,
                COUNT(*) FILTER (WHERE da.sentiment_score > 0.1) as positive_count,
                COUNT(*) FILTER (WHERE da.sentiment_score < -0.1) as negative_count,
                COUNT(*) FILTER (WHERE da.sentiment_score BETWEEN -0.1 AND 0.1) as neutral_count
            FROM deep_absa da
            INNER JOIN reviews r ON da.review_id = r.review_id
            WHERE DATE(r.review_date) = :target_date
            GROUP BY da.app_id, da.aspect
            ON CONFLICT (app_id, date, aspect) 
            DO UPDATE SET
                avg_sentiment = EXCLUDED.avg_sentiment,
                sentiment_count = EXCLUDED.sentiment_count,
                positive_count = EXCLUDED.positive_count,
                negative_count = EXCLUDED.negative_count,
                neutral_count = EXCLUDED.neutral_count
            """

            aggregated_count = storage.db.execute_non_query(query, {'target_date': target_date})

            if aggregated_count > 0:
                self.logger.info(f"Aggregated {aggregated_count} daily sentiment records for {target_date}")

            return aggregated_count

        except Exception as e:
            self.logger.error(f"Error aggregating daily sentiment: {e}")
            return 0


class BatchProcessor:
    """Main batch processing coordinator for ABSA and SERVQUAL analysis."""

    def __init__(self, config_override: Optional[BatchConfiguration] = None):
        self.logger = logging.getLogger("absa_pipeline.batch.processor")

        # Configuration
        self.config = config_override or BatchConfiguration(
            batch_size=config.absa.batch_size,
            max_reviews_per_job=config.absa.batch_size * 20,  # Process up to 1000 reviews per job
            confidence_threshold=config.absa.confidence_threshold,
            enable_servqual_processing=True,
            cleanup_old_data=True,
            retention_days=90
        )

        # Components
        self.job_tracker = ProcessingJobTracker()
        self.lifecycle_manager = DataLifecycleManager()
        self.servqual_mapper = ServqualMapper()

        self.logger.info(f"Batch processor initialized with batch size: {self.config.batch_size}")

    def get_unprocessed_reviews(self, app_id: Optional[str] = None,
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get unprocessed reviews for ABSA analysis."""
        try:
            batch_limit = limit or self.config.max_reviews_per_job

            base_query = """
            SELECT review_id, app_id, content, rating, review_date
            FROM reviews 
            WHERE processed = FALSE AND is_spam = FALSE AND content IS NOT NULL
            """

            params = {"limit": batch_limit}

            if app_id:
                base_query += " AND app_id = :app_id"
                params["app_id"] = app_id

            base_query += " ORDER BY review_date DESC LIMIT :limit"

            df = storage.db.execute_query(base_query, params)
            reviews = df.to_dict('records')

            self.logger.info(f"Found {len(reviews)} unprocessed reviews"
                           f"{f' for app {app_id}' if app_id else ''}")

            return reviews

        except Exception as e:
            self.logger.error(f"Error getting unprocessed reviews: {e}")
            return []

    def process_review_batch(self, reviews: List[Dict[str, Any]],
                           job_id: str) -> BatchProcessingResult:
        """Process a batch of reviews through the complete ABSA pipeline."""
        self.logger.info(f"Processing batch of {len(reviews)} reviews for job {job_id}")

        try:
            # Process through ABSA engine
            batch_result = absa_engine.analyze_batch(reviews, mode=AnalysisMode.DEEP)

            # Store ABSA results in database
            if batch_result.database_records:
                stored_count = storage.absa.store_deep_absa_results(batch_result.database_records)
                self.logger.info(f"Stored {stored_count} ABSA results in database")

            # Mark reviews as processed
            review_ids = [r['review_id'] for r in reviews]
            processed_count = storage.reviews.mark_reviews_processed(review_ids)
            self.logger.info(f"Marked {processed_count} reviews as processed")

            # Update job progress
            self.job_tracker.update_job_progress(job_id, len(reviews))

            return batch_result

        except Exception as e:
            self.logger.error(f"Error processing review batch: {e}")
            raise

    def process_servqual_analysis(self, app_id: str, target_date: date = None) -> int:
        """Process SERVQUAL analysis for an app."""
        try:
            self.logger.info(f"Processing SERVQUAL analysis for {app_id}")

            # Get the most recent review date instead of today's date
            recent_date_query = """
            SELECT MAX(DATE(review_date)) as latest_date 
            FROM reviews 
            WHERE app_id = :app_id AND processed = TRUE
            """

            date_result = storage.db.execute_query(recent_date_query, {'app_id': app_id})
            if not date_result.empty and date_result.iloc[0]['latest_date']:
                target_date = date_result.iloc[0]['latest_date']
            else:
                target_date = datetime.now().date()

            self.logger.info(f"Using target date: {target_date}")

            # Use the existing SERVQUAL mapper
            servqual_results = self.servqual_mapper.process_daily_servqual(app_id, target_date)

            if servqual_results:
                # Store SERVQUAL scores
                stored_count, error_count = servqual_storage.store_servqual_scores(servqual_results)

                self.logger.info(f"Stored {stored_count} SERVQUAL scores for {app_id}")
                return stored_count
            else:
                self.logger.info(f"No SERVQUAL data generated for {app_id} on {target_date}")
                return 0

        except Exception as e:
            self.logger.error(f"Error processing SERVQUAL for {app_id}: {e}")
            return 0

    def run_daily_processing(self, app_id: Optional[str] = None, progress_callback=None) -> BatchJobResult:
        """Run the complete daily processing workflow."""
        start_time = datetime.now()
        job_id = self.job_tracker.create_job(
            job_type="daily_absa_processing",
            app_id=app_id,
            metadata={"batch_size": self.config.batch_size}
        )

        self.logger.info(f"Starting daily processing job {job_id}"
                        f"{f' for app {app_id}' if app_id else ' for all apps'}")

        total_reviews_processed = 0
        total_aspects_extracted = 0
        total_servqual_updated = 0
        error_message = None

        try:
            # Step 1: Get unprocessed reviews
            reviews = self.get_unprocessed_reviews(app_id)

            if not reviews:
                self.logger.info("No unprocessed reviews found")
                self.job_tracker.complete_job(job_id, True, 0)

                return BatchJobResult(
                    job_id=job_id,
                    app_id=app_id,
                    start_time=start_time,
                    end_time=datetime.now(),
                    reviews_processed=0,
                    aspects_extracted=0,
                    servqual_dimensions_updated=0,
                    processing_time_seconds=0,
                    success=True,
                    error_message=None,
                    statistics={}
                )

            # Step 2: Process reviews in batches
            total_batches = (len(reviews) + self.config.batch_size - 1) // self.config.batch_size

            for batch_num in range(total_batches):
                start_idx = batch_num * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(reviews))
                batch_reviews = reviews[start_idx:end_idx]

                # Progress callback
                if progress_callback:
                    progress = ((batch_num + 1) / total_batches) * 100
                    progress_callback(progress, f"Processing batch {batch_num + 1}/{total_batches}")

                self.logger.info(f"Processing batch {batch_num + 1}/{total_batches} "
                               f"({len(batch_reviews)} reviews)")

                # Process ABSA for this batch
                batch_result = self.process_review_batch(batch_reviews, job_id)

                total_reviews_processed += batch_result.total_reviews
                total_aspects_extracted += batch_result.total_aspects_extracted

            # Step 3: Process SERVQUAL analysis if enabled
            if self.config.enable_servqual_processing:
                target_date = datetime.now().date()

                if app_id:
                    # Process SERVQUAL for specific app
                    servqual_count = self.process_servqual_analysis(app_id, target_date)
                    total_servqual_updated += servqual_count
                else:
                    # Process SERVQUAL for all apps with new data
                    processed_apps = set(r['app_id'] for r in reviews)

                    for processed_app_id in processed_apps:
                        servqual_count = self.process_servqual_analysis(processed_app_id, target_date)
                        total_servqual_updated += servqual_count

            # Step 4: Aggregate daily sentiment data
            aggregated_count = self.lifecycle_manager.aggregate_daily_sentiment()

            # Step 5: Cleanup old data if enabled
            if self.config.cleanup_old_data:
                self.lifecycle_manager.cleanup_old_processing_jobs(30)
                self.lifecycle_manager.cleanup_old_absa_results(self.config.retention_days)

            # Complete the job
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            self.job_tracker.complete_job(job_id, True, total_reviews_processed)

            # Generate statistics
            statistics = {
                'total_batches_processed': total_batches,
                'batch_size': self.config.batch_size,
                'avg_aspects_per_review': total_aspects_extracted / total_reviews_processed if total_reviews_processed > 0 else 0,
                'aggregated_daily_records': aggregated_count,
                'servqual_processing_enabled': self.config.enable_servqual_processing
            }

            self.logger.info(f"Daily processing completed successfully: "
                           f"{total_reviews_processed} reviews, {total_aspects_extracted} aspects, "
                           f"{total_servqual_updated} SERVQUAL dimensions")

            return BatchJobResult(
                job_id=job_id,
                app_id=app_id,
                start_time=start_time,
                end_time=end_time,
                reviews_processed=total_reviews_processed,
                aspects_extracted=total_aspects_extracted,
                servqual_dimensions_updated=total_servqual_updated,
                processing_time_seconds=processing_time,
                success=True,
                error_message=None,
                statistics=statistics
            )

        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Error in daily processing: {error_message}")

            # Mark job as failed
            self.job_tracker.complete_job(job_id, False, total_reviews_processed, error_message)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return BatchJobResult(
                job_id=job_id,
                app_id=app_id,
                start_time=start_time,
                end_time=end_time,
                reviews_processed=total_reviews_processed,
                aspects_extracted=total_aspects_extracted,
                servqual_dimensions_updated=total_servqual_updated,
                processing_time_seconds=processing_time,
                success=False,
                error_message=error_message,
                statistics={}
            )

    def get_processing_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get processing statistics for the last N days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Get job statistics
            job_query = """
            SELECT 
                COUNT(*) as total_jobs,
                COUNT(*) FILTER (WHERE status = 'completed') as successful_jobs,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
                SUM(records_processed) as total_records_processed,
                AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_processing_time_seconds
            FROM processing_jobs 
            WHERE created_at >= :cutoff_date
            """

            job_stats_df = storage.db.execute_query(job_query, {'cutoff_date': cutoff_date})
            job_stats = job_stats_df.iloc[0].to_dict() if not job_stats_df.empty else {}

            # Get ABSA statistics
            absa_query = """
            SELECT 
                COUNT(*) as total_absa_records,
                COUNT(DISTINCT app_id) as apps_analyzed,
                COUNT(DISTINCT aspect) as unique_aspects,
                AVG(sentiment_score) as avg_sentiment,
                AVG(confidence_score) as avg_confidence
            FROM deep_absa 
            WHERE created_at >= :cutoff_date
            """

            absa_stats_df = storage.db.execute_query(absa_query, {'cutoff_date': cutoff_date})
            absa_stats = absa_stats_df.iloc[0].to_dict() if not absa_stats_df.empty else {}

            # Get SERVQUAL statistics
            servqual_query = """
            SELECT 
                COUNT(*) as total_servqual_records,
                COUNT(DISTINCT app_id) as apps_with_servqual,
                COUNT(DISTINCT dimension) as dimensions_analyzed,
                AVG(quality_score) as avg_quality_score
            FROM servqual_scores 
            WHERE created_at >= :cutoff_date
            """

            servqual_stats_df = storage.db.execute_query(servqual_query, {'cutoff_date': cutoff_date})
            servqual_stats = servqual_stats_df.iloc[0].to_dict() if not servqual_stats_df.empty else {}

            return {
                'period_days': days,
                'job_statistics': job_stats,
                'absa_statistics': absa_stats,
                'servqual_statistics': servqual_stats,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting processing statistics: {e}")
            return {}


# Global batch processor instance
batch_processor = BatchProcessor()


# Convenience functions for easy usage
def run_daily_batch_processing(app_id: Optional[str] = None) -> BatchJobResult:
    """Convenience function to run daily batch processing."""
    return batch_processor.run_daily_processing(app_id)


def process_app_reviews(app_id: str) -> BatchJobResult:
    """Convenience function to process reviews for a specific app."""
    return batch_processor.run_daily_processing(app_id)


def get_batch_processing_stats(days: int = 7) -> Dict[str, Any]:
    """Convenience function to get processing statistics."""
    return batch_processor.get_processing_statistics(days)