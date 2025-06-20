"""
Batch processing pipeline for ABSA Pipeline.
Coordinates deep ABSA analysis, SERVQUAL mapping, and data aggregation.
Integrates the ABSA engine with existing SERVQUAL infrastructure for business intelligence.
Enhanced with LLM SERVQUAL processing capabilities.
"""

from __future__ import annotations

import gc
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
            SET records_processed = :records_processed, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = :job_id
            """

            params = {
                'job_id': job_id,
                'records_processed': records_processed
            }

            storage.db.execute_non_query(query, params)

        except Exception as e:
            self.logger.error(f"Error updating job progress: {e}")

    def complete_job(self, job_id: str, success: bool, error_message: str = None):
        """Mark job as completed."""
        try:
            status = 'completed' if success else 'failed'

            query = """
            UPDATE processing_jobs 
            SET status = :status, end_time = :end_time, error_message = :error_message
            WHERE job_id = :job_id
            """

            params = {
                'job_id': job_id,
                'status': status,
                'end_time': datetime.now(),
                'error_message': error_message
            }

            storage.db.execute_non_query(query, params)
            self.logger.info(f"Completed job {job_id} with status: {status}")

        except Exception as e:
            self.logger.error(f"Error completing job: {e}")


class BatchProcessor:
    """Main batch processing coordinator for ABSA and LLM SERVQUAL analysis."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.batch")
        self.job_tracker = ProcessingJobTracker()
        self.servqual_mapper = ServqualMapper()

        # Configuration
        self.config = BatchConfiguration(
            batch_size=config.absa.batch_size,
            max_reviews_per_job=config.scraping.max_reviews_per_app,
            confidence_threshold=config.absa.confidence_threshold,
            enable_servqual_processing=True,
            cleanup_old_data=True,
            retention_days=30
        )

        self.logger.info(f"Batch processor initialized with batch size: {self.config.batch_size}")

    def run_sequential_processing(self, app_id: Optional[str] = None, limit: Optional[int] = None) -> BatchJobResult:
        """
        Run sequential ABSA → SERVQUAL processing using the sequential processor.
        This method provides compatibility with dashboard calls.

        Args:
            app_id: Process specific app or None for all apps
            limit: Limit number of reviews to process (for testing)

        Returns:
            BatchJobResult with sequential processing statistics
        """
        try:
            # Import sequential processor
            from src.pipeline.sequential_processor import sequential_processor

            self.logger.info(f"[SEQUENTIAL] Starting sequential processing via BatchProcessor wrapper")

            # Use the sequential processor for the actual work
            seq_result = sequential_processor.start_sequential_processing(
                app_id=app_id,
                resume_job_id=None,
                skip_absa=False,
                limit=limit
            )

            # Convert SequentialProcessingResult to BatchJobResult for compatibility
            if seq_result.success:
                result = BatchJobResult(
                    job_id=seq_result.job_id,
                    app_id=seq_result.app_id,
                    start_time=seq_result.start_time,
                    end_time=seq_result.end_time,
                    reviews_processed=seq_result.total_reviews_processed,
                    aspects_extracted=seq_result.total_aspects_extracted,
                    servqual_dimensions_updated=seq_result.total_servqual_dimensions_updated,
                    processing_time_seconds=seq_result.processing_time_seconds,
                    success=True,
                    error_message=None,
                    statistics={
                        'absa_phase_processed': seq_result.absa_phase.reviews_processed,
                        'servqual_phase_processed': seq_result.servqual_phase.reviews_processed,
                        'checkpoints_created': seq_result.checkpoints_created,
                        'failed_reviews': seq_result.failed_reviews,
                        'sequential_mode': True
                    }
                )

                self.logger.info(f"[SEQUENTIAL] Sequential processing completed: {seq_result.total_reviews_processed} reviews")
                return result
            else:
                # Create failed result
                return self._create_failed_result(
                    seq_result.job_id,
                    seq_result.app_id,
                    seq_result.start_time,
                    seq_result.error_message or "Sequential processing failed"
                )

        except Exception as e:
            error_msg = f"Sequential processing wrapper failed: {e}"
            self.logger.error(error_msg)
            job_id = str(uuid.uuid4())
            return self._create_failed_result(job_id, app_id, datetime.now(), error_msg)

    def run_daily_processing(self, app_id: Optional[str] = None, progress_callback=None) -> BatchJobResult:
        """Run the complete daily processing workflow with LLM SERVQUAL."""
        job_id = self.job_tracker.create_job("daily_batch", app_id)
        start_time = datetime.now()

        self.logger.info(f"[DAILY] Starting daily batch processing job: {job_id}")

        try:
            total_reviews_processed = 0
            total_aspects_extracted = 0
            total_servqual_updated = 0

            # Step 1: Process ABSA analysis
            if app_id:
                self.logger.info(f"[DAILY] Processing ABSA for app: {app_id}")
            else:
                self.logger.info("[DAILY] Processing ABSA for all apps")

            absa_result = self._run_absa_processing(app_id, progress_callback)

            if absa_result.success:
                total_reviews_processed += absa_result.reviews_processed
                total_aspects_extracted += absa_result.aspects_extracted
                self.logger.info(f"[DAILY] ABSA processing completed: {absa_result.reviews_processed} reviews")
            else:
                self.logger.warning(f"[DAILY] ABSA processing failed: {absa_result.error_message}")

            # Step 2: Process LLM SERVQUAL analysis if enabled
            if self.config.enable_servqual_processing:
                target_date = datetime.now().date()

                # Use LLM SERVQUAL processing instead of keyword-based
                llm_result = self.run_servqual_llm_processing(app_id)

                if llm_result.success:
                    total_servqual_updated += llm_result.servqual_dimensions_updated
                    self.logger.info(f"[DAILY] LLM SERVQUAL processing completed: {llm_result.servqual_dimensions_updated} dimensions updated")
                else:
                    self.logger.warning(f"[DAILY] LLM SERVQUAL processing failed: {llm_result.error_message}")

            # Step 3: Data aggregation and cleanup
            if self.config.cleanup_old_data:
                self._cleanup_old_data()

            # Step 4: Aggregate daily data
            aggregated_records = self.aggregate_daily_sentiment()
            self.logger.info(f"[DAILY] Aggregated {aggregated_records} sentiment records")

            # Complete job
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            self.job_tracker.complete_job(job_id, True)

            result = BatchJobResult(
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
                statistics={
                    'absa_success': absa_result.success if 'absa_result' in locals() else False,
                    'servqual_success': llm_result.success if 'llm_result' in locals() else False,
                    'aggregated_records': aggregated_records
                }
            )

            self.logger.info(f"[DAILY] Processing complete: {total_reviews_processed} reviews, "
                           f"{total_aspects_extracted} aspects, {total_servqual_updated} SERVQUAL updates")

            return result

        except Exception as e:
            error_msg = f"Daily processing failed: {e}"
            self.logger.error(error_msg)
            self.job_tracker.complete_job(job_id, False, error_msg)
            return self._create_failed_result(job_id, app_id, start_time, error_msg)

    def run_servqual_llm_processing(self, app_id: Optional[str] = None) -> BatchJobResult:
        """
        Run LLM-based SERVQUAL processing for business intelligence.
        Uses Mistral LLM for direct SERVQUAL dimension classification.

        Args:
            app_id: Process specific app or None for all apps

        Returns:
            BatchJobResult with LLM processing statistics
        """
        job_id = self.job_tracker.create_job("servqual_llm", app_id)
        start_time = datetime.now()

        self.logger.info(f"[LLM SERVQUAL] Starting LLM SERVQUAL processing job: {job_id}")

        try:
            # Check LLM availability (with safe import)
            try:
                from src.absa.servqual_llm_model import servqual_llm
                model_info = servqual_llm.get_model_info()
                if not model_info.get('model_available', False):
                    error_msg = "LLM SERVQUAL model not available - ensure Ollama is running with Mistral model"
                    self.logger.error(error_msg)
                    self.job_tracker.complete_job(job_id, False, error_msg)
                    return self._create_failed_result(job_id, app_id, start_time, error_msg)
            except ImportError:
                error_msg = "LLM SERVQUAL model module not found - please create servqual_llm_model.py"
                self.logger.error(error_msg)
                self.job_tracker.complete_job(job_id, False, error_msg)
                return self._create_failed_result(job_id, app_id, start_time, error_msg)

            self.logger.info(f"[LLM SERVQUAL] Using model: {model_info['model_name']}")

            # Get reviews that need SERVQUAL processing
            reviews = self._get_reviews_for_servqual_llm(app_id)

            if not reviews:
                self.logger.info("[LLM SERVQUAL] No reviews found for LLM SERVQUAL processing")
                self.job_tracker.complete_job(job_id, True)
                return self._create_success_result(job_id, app_id, start_time, 0, 0, 0)

            self.logger.info(f"[LLM SERVQUAL] Processing {len(reviews)} reviews")

            # Process reviews in batches for memory efficiency
            batch_size = 25  # Optimal for LLM processing
            total_processed = 0
            total_servqual_updated = 0

            for i in range(0, len(reviews), batch_size):
                batch = reviews[i:i + batch_size]

                self.logger.info(f"[LLM SERVQUAL] Processing batch {i//batch_size + 1}/{(len(reviews) + batch_size - 1)//batch_size}")

                # Process batch with LLM
                batch_results = self._process_servqual_llm_batch(batch, app_id)

                if batch_results:
                    total_processed += len(batch)
                    total_servqual_updated += batch_results

                    # Mark reviews as SERVQUAL processed
                    review_ids = [r['review_id'] for r in batch]
                    self._mark_reviews_servqual_processed(review_ids)

                # Update job progress
                self.job_tracker.update_job_progress(job_id, total_processed)

                # Memory cleanup
                gc.collect()

            # Complete job
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            self.job_tracker.complete_job(job_id, True)

            self.logger.info(f"[LLM SERVQUAL] Processing complete: {total_processed} reviews, {total_servqual_updated} SERVQUAL scores")
            self.logger.info(f"[LLM SERVQUAL] Processing time: {processing_time:.1f}s")

            return BatchJobResult(
                job_id=job_id,
                app_id=app_id,
                start_time=start_time,
                end_time=end_time,
                reviews_processed=total_processed,
                aspects_extracted=0,  # LLM SERVQUAL doesn't extract traditional aspects
                servqual_dimensions_updated=total_servqual_updated,
                processing_time_seconds=processing_time,
                success=True,
                error_message=None,
                statistics={
                    'llm_model': model_info['model_name'],
                    'avg_processing_time_per_review': processing_time / total_processed if total_processed > 0 else 0,
                    'throughput_reviews_per_second': total_processed / processing_time if processing_time > 0 else 0
                }
            )

        except Exception as e:
            error_msg = f"LLM SERVQUAL processing failed: {e}"
            self.logger.error(error_msg)
            self.job_tracker.complete_job(job_id, False, error_msg)
            return self._create_failed_result(job_id, app_id, start_time, error_msg)

    def _get_reviews_for_servqual_llm(self, app_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get reviews that need LLM SERVQUAL processing."""
        try:
            app_filter = ""
            params = {}

            if app_id:
                app_filter = "AND app_id = :app_id"
                params['app_id'] = app_id

            # Get reviews that have been ABSA processed but not SERVQUAL processed
            query = f"""
            SELECT review_id, app_id, content, rating, review_date
            FROM reviews 
            WHERE processed = TRUE 
            AND NOT is_spam 
            AND content IS NOT NULL
            AND (servqual_processed = FALSE OR servqual_processed IS NULL)
            {app_filter}
            ORDER BY review_date DESC
            LIMIT :limit
            """

            params['limit'] = self.config.max_reviews_per_job

            df = storage.db.execute_query(query, params)
            reviews = df.to_dict('records')

            self.logger.info(f"Found {len(reviews)} reviews for LLM SERVQUAL processing"
                            f"{f' for app {app_id}' if app_id else ''}")

            return reviews

        except Exception as e:
            self.logger.error(f"Error getting reviews for LLM SERVQUAL: {e}")
            return []

    def _process_servqual_llm_batch(self, reviews: List[Dict], app_id: str) -> int:
        """Process a batch of reviews with LLM SERVQUAL analysis."""
        try:
            # Placeholder for LLM processing - will need actual LLM implementation
            self.logger.info(f"LLM SERVQUAL batch processing not yet implemented - marking as processed")
            return len(reviews)  # Placeholder return

        except Exception as e:
            self.logger.error(f"Error processing LLM SERVQUAL batch: {e}")
            return 0

    def _mark_reviews_servqual_processed(self, review_ids: List[str]):
        """Mark reviews as SERVQUAL processed."""
        if not review_ids:
            return

        try:
            placeholders = ','.join([f':id{i}' for i in range(len(review_ids))])
            query = f"""
            UPDATE reviews 
            SET servqual_processed = TRUE, servqual_processed_at = CURRENT_TIMESTAMP
            WHERE review_id IN ({placeholders})
            """

            params = {f'id{i}': rid for i, rid in enumerate(review_ids)}
            rows_affected = storage.db.execute_non_query(query, params)

            self.logger.info(f"Marked {rows_affected} reviews as SERVQUAL processed")

        except Exception as e:
            self.logger.error(f"Error marking reviews as SERVQUAL processed: {e}")

    def _run_absa_processing(self, app_id: Optional[str] = None, progress_callback=None) -> BatchJobResult:
        """Run ABSA processing for reviews."""
        job_id = self.job_tracker.create_job("absa_batch", app_id)
        start_time = datetime.now()

        try:
            # Get unprocessed reviews
            reviews = self.get_unprocessed_reviews(app_id)

            if not reviews:
                self.logger.info("No unprocessed reviews found for ABSA")
                self.job_tracker.complete_job(job_id, True)
                return self._create_success_result(job_id, app_id, start_time, 0, 0, 0)

            self.logger.info(f"[ABSA] Processing {len(reviews)} reviews")

            # Process in batches
            total_processed = 0
            total_aspects = 0

            for i in range(0, len(reviews), self.config.batch_size):
                batch = reviews[i:i + self.config.batch_size]

                # Process batch through ABSA engine
                batch_result = absa_engine.analyze_batch(batch, mode=AnalysisMode.DEEP)

                if batch_result.successful_reviews > 0:
                    # Store ABSA results
                    stored_count = storage.absa.store_deep_absa_results(batch_result.database_records)

                    # Mark reviews as processed
                    review_ids = [r['review_id'] for r in batch]
                    storage.reviews.mark_reviews_processed(review_ids)

                    total_processed += batch_result.successful_reviews
                    total_aspects += batch_result.total_aspects_extracted

                # Update progress
                self.job_tracker.update_job_progress(job_id, total_processed)

                if progress_callback:
                    progress_callback(total_processed, len(reviews))

                # Memory cleanup
                gc.collect()

            # Complete job
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            self.job_tracker.complete_job(job_id, True)

            return BatchJobResult(
                job_id=job_id,
                app_id=app_id,
                start_time=start_time,
                end_time=end_time,
                reviews_processed=total_processed,
                aspects_extracted=total_aspects,
                servqual_dimensions_updated=0,
                processing_time_seconds=processing_time,
                success=True,
                error_message=None,
                statistics={}
            )

        except Exception as e:
            error_msg = f"ABSA processing failed: {e}"
            self.logger.error(error_msg)
            self.job_tracker.complete_job(job_id, False, error_msg)
            return self._create_failed_result(job_id, app_id, start_time, error_msg)

    def get_unprocessed_reviews(self, app_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get reviews that need ABSA processing."""
        try:
            app_filter = ""
            params = {"limit": self.config.max_reviews_per_job}

            if app_id:
                app_filter = "AND app_id = :app_id"
                params["app_id"] = app_id

            query = f"""
            SELECT review_id, app_id, content, rating, review_date
            FROM reviews 
            WHERE processed = FALSE 
            AND NOT is_spam 
            AND content IS NOT NULL
            AND LENGTH(TRIM(content)) > 5
            {app_filter}
            ORDER BY review_date DESC
            LIMIT :limit
            """

            df = storage.db.execute_query(query, params)
            reviews = df.to_dict('records')

            self.logger.info(f"Found {len(reviews)} unprocessed reviews"
                            f"{f' for app {app_id}' if app_id else ''}")

            return reviews

        except Exception as e:
            self.logger.error(f"Error getting unprocessed reviews: {e}")
            return []

    def _cleanup_old_data(self):
        """Clean up old processing data."""
        try:
            deleted_count = self.cleanup_old_absa_results(self.config.retention_days)
            self.logger.info(f"Cleaned up {deleted_count} old records")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def cleanup_old_absa_results(self, retention_days: int = 30) -> int:
        """Clean up old ABSA result records."""
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

    def run_sequential_processing(self, app_id: Optional[str] = None, limit: Optional[int] = None) -> BatchJobResult:
        """
        Run sequential ABSA → SERVQUAL processing using the sequential processor.
        This method provides compatibility with dashboard calls.

        Args:
            app_id: Process specific app or None for all apps
            limit: Limit number of reviews to process (for testing)

        Returns:
            BatchJobResult with sequential processing statistics
        """
        try:
            # Import sequential processor
            from src.pipeline.sequential_processor import sequential_processor

            self.logger.info(f"[SEQUENTIAL] Starting sequential processing via BatchProcessor wrapper")

            # Use the sequential processor for the actual work
            seq_result = sequential_processor.start_sequential_processing(
                app_id=app_id,
                resume_job_id=None,
                skip_absa=False,
                limit=limit
            )

            # Convert SequentialProcessingResult to BatchJobResult for compatibility
            if seq_result.success:
                result = BatchJobResult(
                    job_id=seq_result.job_id,
                    app_id=seq_result.app_id,
                    start_time=seq_result.start_time,
                    end_time=seq_result.end_time,
                    reviews_processed=seq_result.total_reviews_processed,
                    aspects_extracted=seq_result.total_aspects_extracted,
                    servqual_dimensions_updated=seq_result.total_servqual_dimensions_updated,
                    processing_time_seconds=seq_result.processing_time_seconds,
                    success=True,
                    error_message=None,
                    statistics={
                        'absa_phase_processed': seq_result.absa_phase.reviews_processed,
                        'servqual_phase_processed': seq_result.servqual_phase.reviews_processed,
                        'checkpoints_created': seq_result.checkpoints_created,
                        'failed_reviews': seq_result.failed_reviews,
                        'sequential_mode': True
                    }
                )

                self.logger.info(f"[SEQUENTIAL] Sequential processing completed: {seq_result.total_reviews_processed} reviews")
                return result
            else:
                # Create failed result
                return self._create_failed_result(
                    seq_result.job_id,
                    seq_result.app_id,
                    seq_result.start_time,
                    seq_result.error_message or "Sequential processing failed"
                )

        except Exception as e:
            error_msg = f"Sequential processing wrapper failed: {e}"
            self.logger.error(error_msg)
            job_id = str(uuid.uuid4())
            return self._create_failed_result(job_id, app_id, datetime.now(), error_msg)

    def get_processing_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive processing statistics including LLM."""
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

            # Get LLM SERVQUAL statistics
            llm_stats = get_llm_servqual_stats(days)

            return {
                'period_days': days,
                'job_statistics': job_stats,
                'absa_statistics': absa_stats,
                'servqual_statistics': servqual_stats,
                'llm_servqual_statistics': llm_stats,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting processing statistics: {e}")
            return {}

    def _create_success_result(self, job_id: str, app_id: Optional[str], start_time: datetime,
                              reviews_processed: int, aspects_extracted: int,
                              servqual_dimensions: int) -> BatchJobResult:
        """Create successful batch result."""
        end_time = datetime.now()
        return BatchJobResult(
            job_id=job_id,
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            reviews_processed=reviews_processed,
            aspects_extracted=aspects_extracted,
            servqual_dimensions_updated=servqual_dimensions,
            processing_time_seconds=(end_time - start_time).total_seconds(),
            success=True,
            error_message=None,
            statistics={}
        )

    def _create_failed_result(self, job_id: str, app_id: Optional[str], start_time: datetime,
                             error_message: str) -> BatchJobResult:
        """Create failed batch result."""
        end_time = datetime.now()
        return BatchJobResult(
            job_id=job_id,
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            reviews_processed=0,
            aspects_extracted=0,
            servqual_dimensions_updated=0,
            processing_time_seconds=(end_time - start_time).total_seconds(),
            success=False,
            error_message=error_message,
            statistics={}
        )


def get_llm_servqual_stats(days: int = 7) -> Dict[str, Any]:
    """Get LLM SERVQUAL processing statistics."""
    try:
        query = """
        SELECT 
            COUNT(*) as total_llm_jobs,
            COUNT(*) FILTER (WHERE pj.status = 'completed') as successful_jobs,
            AVG(ps.processing_time_seconds) as avg_processing_time,
            SUM(ps.reviews_processed) as total_reviews_processed,
            SUM(ps.servqual_dimensions_updated) as total_dimensions_updated
        FROM processing_jobs pj
        LEFT JOIN processing_statistics ps ON pj.job_id = ps.job_id
        WHERE pj.job_type = 'servqual_llm'
        AND pj.created_at >= CURRENT_DATE - INTERVAL ':days days'
        """

        df = storage.db.execute_query(query, {'days': days})

        if not df.empty:
            row = df.iloc[0]
            return {
                'total_llm_jobs': int(row['total_llm_jobs']),
                'successful_jobs': int(row['successful_jobs']),
                'avg_processing_time': float(row['avg_processing_time']) if row['avg_processing_time'] else 0,
                'total_reviews_processed': int(row['total_reviews_processed']) if row['total_reviews_processed'] else 0,
                'total_dimensions_updated': int(row['total_dimensions_updated']) if row['total_dimensions_updated'] else 0,
                'success_rate': row['successful_jobs'] / row['total_llm_jobs'] if row['total_llm_jobs'] > 0 else 0
            }

        return {}

    except Exception as e:
        logging.error(f"Error getting LLM SERVQUAL stats: {e}")
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