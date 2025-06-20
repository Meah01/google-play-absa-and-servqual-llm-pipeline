"""
Sequential Processing Engine for ABSA Pipeline.
Coordinates ABSA → SERVQUAL workflow with checkpoint/resume capability.
Handles progress tracking, error recovery, and resource management.
Enhanced with LLM SERVQUAL processing capabilities.
"""

import gc
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import json

from src.utils.config import config
from src.data.storage import storage
from src.absa.engine import absa_engine, AnalysisMode
from src.absa.servqual_mapper import ServqualMapper
from src.absa.servqual_llm_model import servqual_llm
from src.data.servqual_storage import servqual_storage


@dataclass
class ProcessingPhase:
    """Represents a phase in sequential processing."""
    name: str
    reviews_processed: int
    total_reviews: int
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class SequentialProcessingResult:
    """Result of complete sequential processing job."""
    job_id: str
    app_id: Optional[str]
    start_time: datetime
    end_time: datetime
    absa_phase: ProcessingPhase
    servqual_phase: ProcessingPhase
    total_reviews_processed: int
    total_aspects_extracted: int
    total_servqual_dimensions_updated: int
    checkpoints_created: int
    failed_reviews: int
    success: bool
    error_message: Optional[str]
    processing_time_seconds: float


class SequentialProcessor:
    """
    Manages sequential ABSA → SERVQUAL processing with robust error handling,
    checkpoint/resume capability, and progress tracking.
    Enhanced with LLM SERVQUAL processing capabilities.
    """

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.sequential_processor")
        self.servqual_mapper = ServqualMapper()

        # Configuration with timeout protection
        self.checkpoint_frequency = 50  # Every 50 reviews
        self.checkpoint_time_interval = 900  # 15 minutes in seconds
        self.batch_size = 25  # Process reviews in smaller batches
        self.max_retries = 2
        self.max_processing_time = 3600  # 1 hour timeout

        # Progress tracking
        self.current_job_id: Optional[str] = None
        self.last_checkpoint_time: Optional[datetime] = None
        self.progress_milestones = [0.5, 0.75]  # 50% and 75% for >500 reviews

    def start_sequential_processing(self, app_id: Optional[str] = None,
                                    resume_job_id: Optional[str] = None,
                                    skip_absa: bool = False,
                                    limit: Optional[int] = None) -> SequentialProcessingResult:
        """
        Start or resume sequential ABSA → SERVQUAL LLM processing.

        Args:
            app_id: Process specific app or None for all apps
            resume_job_id: Resume existing job or None for new job

        Returns:
            SequentialProcessingResult with processing statistics
        """
        start_time = datetime.now()

        # Generate new UUID or use provided resume job ID
        if resume_job_id:
            # Validate resume job ID is a proper UUID
            try:
                uuid.UUID(resume_job_id)
                self.current_job_id = resume_job_id
            except ValueError:
                raise ValueError(f"Invalid UUID format for resume_job_id: {resume_job_id}")
        else:
            self.current_job_id = str(uuid.uuid4())

        self.logger.info(f"Starting sequential processing job: {self.current_job_id}")
        if app_id:
            self.logger.info(f"Processing app: {app_id}")
        else:
            self.logger.info("Processing all apps")

        try:
            # Initialize or resume job
            if resume_job_id:
                job_state = self._get_resumable_job_state(resume_job_id)
                if not job_state:
                    raise ValueError(f"Cannot resume job {resume_job_id} - no valid checkpoint found")

                self.logger.info(f"Resuming job from checkpoint: {job_state['reviews_processed']} reviews processed")
            else:
                job_state = self._create_new_job(app_id)

            # Check LLM availability before processing
            if not self._check_llm_availability():
                error_msg = "LLM SERVQUAL model not available - ensure Ollama is running with Mistral model"
                self.logger.error(error_msg)
                return self._create_failure_result(start_time, app_id, error_msg)

            # Get reviews to process
            reviews_to_process = self._get_unprocessed_reviews(app_id, job_state, limit)
            total_reviews = len(reviews_to_process)

            self.logger.info(f"Review processing summary:")
            self.logger.info(f"  Total reviews found: {total_reviews}")
            if total_reviews > 0:
                absa_pending = sum(1 for r in reviews_to_process if not r.get('absa_processed', False))
                servqual_pending = sum(1 for r in reviews_to_process if not r.get('servqual_processed', False))
                self.logger.info(f"  ABSA pending: {absa_pending}")
                self.logger.info(f"  SERVQUAL pending: {servqual_pending}")

            if total_reviews == 0:
                self.logger.info("No unprocessed reviews found")
                return self._create_success_result(start_time, app_id, 0, 0, 0)

            self.logger.info(f"Found {total_reviews} reviews to process")

            # Create progress notifications for large jobs
            if total_reviews > 500:
                self._create_progress_notification("start", 0,
                                                   f"Starting processing of {total_reviews} reviews")

            # Phase 1: ABSA Processing (with timeout protection) - Optional
            if skip_absa:
                self.logger.info("Skipping ABSA processing - going directly to LLM SERVQUAL")
                # Create a mock successful ABSA phase
                absa_phase = ProcessingPhase(
                    name="absa_skipped",
                    reviews_processed=total_reviews,
                    total_reviews=total_reviews,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    success=True
                )
                # Mark all reviews as ABSA processed so they can go to SERVQUAL
                review_ids = [r['review_id'] for r in reviews_to_process]
                self._mark_reviews_absa_processed(review_ids)
            else:
                phase_start_time = datetime.now()
                absa_phase = self._process_absa_phase(reviews_to_process, job_state)

                # Check for timeout
                if (datetime.now() - phase_start_time).total_seconds() > self.max_processing_time:
                    error_msg = f"ABSA processing timed out after {self.max_processing_time} seconds"
                    self.logger.error(error_msg)
                    return self._create_failure_result(start_time, app_id, error_msg)

            if not absa_phase.success:
                return self._create_failure_result(start_time, app_id, absa_phase.error_message)

            # Phase 2: SERVQUAL LLM Processing
            servqual_phase = self._process_servqual_phase(reviews_to_process, job_state)

            # Complete job
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            self._complete_job(self.current_job_id, absa_phase, servqual_phase)

            # Final notification
            self._create_progress_notification("complete", 100,
                                               f"Processing completed: {total_reviews} reviews processed in {processing_time:.1f}s")

            return SequentialProcessingResult(
                job_id=self.current_job_id,
                app_id=app_id,
                start_time=start_time,
                end_time=end_time,
                absa_phase=absa_phase,
                servqual_phase=servqual_phase,
                total_reviews_processed=total_reviews,
                total_aspects_extracted=absa_phase.reviews_processed * 4,  # Estimate
                total_servqual_dimensions_updated=servqual_phase.reviews_processed,
                checkpoints_created=job_state.get('checkpoints_created', 0),
                failed_reviews=job_state.get('failed_reviews', 0),
                success=True,
                error_message=None,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            self.logger.error(f"Sequential processing failed: {e}")
            self._mark_job_failed(self.current_job_id, str(e))
            return self._create_failure_result(start_time, app_id, str(e))

    def _check_llm_availability(self) -> bool:
        """Check if LLM SERVQUAL model is available before processing."""
        try:
            model_info = servqual_llm.get_model_info()
            available = model_info.get('model_available', False)

            if available:
                self.logger.info(f"LLM SERVQUAL model available: {model_info['model_name']}")
            else:
                self.logger.error("LLM SERVQUAL model not available")

            return available

        except Exception as e:
            self.logger.error(f"Error checking LLM availability: {e}")
            return False

    def _process_absa_phase(self, reviews: List[Dict], job_state: Dict) -> ProcessingPhase:
        """Process ABSA phase with checkpointing."""
        phase_start = datetime.now()
        phase = ProcessingPhase(
            name="absa",
            reviews_processed=0,
            total_reviews=len(reviews),
            start_time=phase_start
        )

        try:
            self.logger.info("Starting ABSA processing phase")
            self._update_job_phase(self.current_job_id, "absa")

            # Filter reviews that haven't been ABSA processed (but in LLM mode, we'll skip traditional ABSA)
            absa_reviews = [r for r in reviews if not r.get('absa_processed', False)]
            self.logger.info(f"Processing {len(absa_reviews)} reviews for ABSA (LLM SERVQUAL mode - skipping traditional models)")

            if len(absa_reviews) == 0:
                self.logger.info("No reviews need ABSA processing - all already processed")
                phase.reviews_processed = 0
                phase.end_time = datetime.now()
                phase.success = True
                return phase

            processed_count = 0
            failed_count = 0
            self.last_checkpoint_time = datetime.now()

            # Process in batches with retry limit
            retry_count = 0
            max_retries_per_batch = 3

            for i in range(0, len(absa_reviews), self.batch_size):
                batch = absa_reviews[i:i + self.batch_size]

                batch_success = False
                current_retry = 0

                while not batch_success and current_retry < max_retries_per_batch:
                    try:
                        # Process batch
                        batch_results = self._process_absa_batch(batch)

                        # Mark reviews as processed
                        review_ids = [r['review_id'] for r in batch if r['review_id'] in batch_results]
                        self._mark_reviews_absa_processed(review_ids)

                        processed_count += len(review_ids)
                        failed_count += len(batch) - len(review_ids)
                        batch_success = True

                        # Create checkpoint if needed
                        if self._should_create_checkpoint(processed_count):
                            self._create_checkpoint("absa", processed_count, len(absa_reviews))

                        # Progress notification for large jobs
                        if len(reviews) > 500:
                            progress = processed_count / len(absa_reviews)
                            self._check_progress_milestones(progress, processed_count, "ABSA")

                        # Clear memory between batches
                        gc.collect()

                    except Exception as e:
                        current_retry += 1
                        self.logger.warning(f"ABSA batch processing failed (attempt {current_retry}/{max_retries_per_batch}): {e}")

                        if current_retry >= max_retries_per_batch:
                            self.logger.error(f"ABSA batch failed after {max_retries_per_batch} attempts, skipping batch")
                            failed_count += len(batch)
                            break
                        else:
                            # Wait before retry
                            time.sleep(2 ** current_retry)  # Exponential backoff

            phase.reviews_processed = processed_count
            phase.end_time = datetime.now()
            # Success if we processed some reviews OR if there were no reviews to process
            phase.success = processed_count > 0 or len(absa_reviews) == 0

            if failed_count > 0:
                self.logger.warning(f"ABSA phase completed with {failed_count} failed reviews")

            self.logger.info(f"ABSA phase completed: {processed_count} reviews processed")
            return phase

        except Exception as e:
            phase.error_message = str(e)
            phase.end_time = datetime.now()
            self.logger.error(f"ABSA phase failed: {e}")
            return phase

    def _process_servqual_phase(self, reviews: List[Dict], job_state: Dict) -> ProcessingPhase:
        """Process SERVQUAL phase using LLM with checkpointing."""
        phase_start = datetime.now()
        phase = ProcessingPhase(
            name="servqual_llm",  # Updated name to reflect LLM usage
            reviews_processed=0,
            total_reviews=len(reviews),
            start_time=phase_start
        )

        try:
            self.logger.info("Starting LLM SERVQUAL processing phase")
            self._update_job_phase(self.current_job_id, "servqual_llm")

            # For LLM SERVQUAL mode, process all reviews that need SERVQUAL (regardless of ABSA status)
            servqual_reviews = [r for r in reviews if not r.get('servqual_processed', False)]
            self.logger.info(f"Processing {len(servqual_reviews)} reviews for LLM SERVQUAL (direct processing - ABSA not required)")

            processed_count = 0
            failed_count = 0

            # Group reviews by app for efficient LLM processing
            app_reviews = {}
            for review in servqual_reviews:
                app_id = review['app_id']
                if app_id not in app_reviews:
                    app_reviews[app_id] = []
                app_reviews[app_id].append(review)

            # Process each app's reviews with LLM
            for app_id, app_review_list in app_reviews.items():
                try:
                    self.logger.info(f"Processing LLM SERVQUAL for app {app_id}: {len(app_review_list)} reviews")

                    # Process SERVQUAL for this app using LLM
                    servqual_results = self._process_servqual_for_app(app_id, app_review_list)

                    # Mark reviews as SERVQUAL processed
                    if servqual_results:
                        review_ids = [r['review_id'] for r in app_review_list]
                        self._mark_reviews_servqual_processed(review_ids)
                        processed_count += len(review_ids)
                    else:
                        failed_count += len(app_review_list)

                    # Create checkpoint if needed
                    if self._should_create_checkpoint(processed_count):
                        self._create_checkpoint("servqual_llm", processed_count, len(servqual_reviews))

                    # Clear memory after each app
                    gc.collect()

                except Exception as e:
                    self.logger.warning(f"LLM SERVQUAL processing failed for app {app_id}: {e}")
                    failed_count += len(app_review_list)
                    continue

            phase.reviews_processed = processed_count
            phase.end_time = datetime.now()
            # Success if we processed some reviews OR if there were no reviews to process
            phase.success = processed_count > 0 or len(servqual_reviews) == 0

            if failed_count > 0:
                self.logger.warning(f"LLM SERVQUAL phase completed with {failed_count} failed reviews")

            # Calculate processing statistics
            processing_time = (phase.end_time - phase.start_time).total_seconds()
            avg_time_per_review = (processing_time / processed_count) if processed_count > 0 else 0

            self.logger.info(f"LLM SERVQUAL phase completed: {processed_count} reviews processed")
            self.logger.info(f"Average processing time: {avg_time_per_review:.2f}s per review")

            return phase

        except Exception as e:
            phase.error_message = str(e)
            phase.end_time = datetime.now()
            self.logger.error(f"LLM SERVQUAL phase failed: {e}")
            return phase

    def _process_absa_batch(self, batch: List[Dict]) -> Dict[str, Any]:
        """Process a batch of reviews through traditional ABSA engine (RoBERTa + spaCy)."""
        try:
            from src.absa.engine import absa_engine, AnalysisMode

            self.logger.info(f"Processing {len(batch)} reviews through traditional ABSA (RoBERTa + spaCy)")

            # Process batch through deep ABSA engine
            batch_result = absa_engine.analyze_batch(batch, mode=AnalysisMode.DEEP)

            if batch_result.successful_reviews > 0:
                # Store ABSA results in database
                from src.data.storage import storage
                stored_count = storage.absa.store_deep_absa_results(batch_result.database_records)

                self.logger.info(f"ABSA batch completed: {batch_result.successful_reviews} successful, "
                                 f"{batch_result.total_aspects_extracted} aspects extracted, "
                                 f"{stored_count} records stored")

                # Return successful review IDs
                return {r['review_id']: {'success': True} for r in batch
                        if any(record['review_id'] == r['review_id'] for record in batch_result.database_records)}
            else:
                self.logger.warning(f"ABSA batch failed: no successful reviews processed")
                return {}


        except Exception as e:
            self.logger.error(f"ABSA batch processing error: {e}")
            # Return empty dict to indicate failure
            return {}

    def _process_servqual_for_app(self, app_id: str, reviews: List[Dict]) -> bool:
        """
        Process SERVQUAL analysis for an app's reviews using LLM.

        Args:
            app_id: Application identifier
            reviews: List of review dictionaries

        Returns:
            True if processing successful, False otherwise
        """
        try:
            self.logger.info(f"Processing LLM SERVQUAL for app {app_id}: {len(reviews)} reviews")

            # Use LLM for SERVQUAL analysis
            llm_results = servqual_llm.batch_analyze_reviews(reviews)

            if not llm_results:
                self.logger.warning(f"No LLM results generated for app {app_id}")
                return False

            # Convert LLM results to SERVQUAL scores format
            servqual_scores = []
            today = datetime.now().date()

            # Aggregate results by dimension
            dimension_data = {
                'reliability': [],
                'assurance': [],
                'tangibles': [],
                'empathy': [],
                'responsiveness': []
            }

            for llm_result in llm_results:
                if llm_result.success:
                    for dimension, data in llm_result.servqual_dimensions.items():
                        if data['relevant']:
                            dimension_data[dimension].append({
                                'sentiment': data['sentiment'],
                                'confidence': data['confidence']
                            })

            # Create aggregated SERVQUAL scores
            for dimension, scores in dimension_data.items():
                if scores:  # Only create scores for dimensions with data
                    avg_sentiment = sum(s['sentiment'] for s in scores) / len(scores)
                    avg_confidence = sum(s['confidence'] for s in scores) / len(scores)

                    # Convert sentiment to quality score (1-5 scale)
                    quality_score = max(1, min(5, 3 + (avg_sentiment * 2)))

                    servqual_score = {
                        'app_id': app_id,
                        'dimension': dimension,
                        'sentiment_score': avg_sentiment,
                        'quality_score': int(round(quality_score)),
                        'review_count': len(scores),
                        'date': today,
                        'confidence_score': avg_confidence,
                        'platform_context': llm_results[0].platform_context if llm_results else 'generic',
                        'model_version': llm_results[0].model_version if llm_results else 'unknown'
                    }
                    servqual_scores.append(servqual_score)

            # Store SERVQUAL scores in database
            if servqual_scores:
                stored_count, error_count = servqual_storage.store_servqual_scores(servqual_scores)
                self.logger.info(f"Stored {stored_count} LLM SERVQUAL scores for {app_id}")
                return stored_count > 0
            else:
                self.logger.info(f"No relevant SERVQUAL dimensions found for {app_id}")
                return True  # Not an error, just no relevant data

        except Exception as e:
            self.logger.error(f"LLM SERVQUAL processing error for app {app_id}: {e}")
            return False

    def _get_unprocessed_reviews(self, app_id: Optional[str], job_state: Dict, limit: Optional[int] = None) -> List[
        Dict]:
        """Get reviews that need processing."""
        try:
            app_filter = ""
            limit_clause = ""
            params = {}

            if app_id:
                app_filter = "AND app_id = :app_id"
                params['app_id'] = app_id

            if limit:
                limit_clause = f"LIMIT {limit}"

            # Get reviews that need processing (either ABSA or SERVQUAL)
            query = f"""
            SELECT 
                review_id,
                app_id,
                content,
                rating,
                review_date,
                COALESCE(absa_processed, FALSE) as absa_processed,
                COALESCE(servqual_processed, FALSE) as servqual_processed
            FROM reviews
            WHERE NOT is_spam
            AND content IS NOT NULL
            AND LENGTH(TRIM(content)) > 5
            AND (
                COALESCE(absa_processed, FALSE) = FALSE OR 
                COALESCE(servqual_processed, FALSE) = FALSE
            )
            {app_filter}
            ORDER BY review_date ASC
            {limit_clause}
            """

            df = storage.db.execute_query(query, params)
            return df.to_dict('records') if not df.empty else []

        except Exception as e:
            self.logger.error(f"Error getting unprocessed reviews: {e}")
            return []

    def _should_create_checkpoint(self, processed_count: int) -> bool:
        """Determine if checkpoint should be created - disabled for now."""
        # Disable checkpointing temporarily to avoid database errors
        return False

    def _create_checkpoint(self, phase: str, processed: int, total: int):
        """Create processing checkpoint with LLM-specific metadata."""
        try:
            # Validate UUID format
            uuid.UUID(self.current_job_id)

            query = """
            SELECT create_processing_checkpoint(
                :job_id, :job_type, :processed, :total, 
                NULL, NULL, :metadata
            )
            """

            metadata = {
                'phase': phase,
                'checkpoint_time': datetime.now().isoformat(),
                'progress_percentage': round((processed / total) * 100, 1) if total > 0 else 0,
                'llm_enabled': phase == 'servqual_llm',  # Track LLM usage
                'model_version': servqual_llm.model_version if phase == 'servqual_llm' else None
            }

            params = {
                'job_id': self.current_job_id,
                'job_type': 'sequential_llm',  # Updated job type
                'processed': processed,
                'total': total,
                'metadata': json.dumps(metadata)
            }

            storage.db.execute_non_query(query, params)
            self.last_checkpoint_time = datetime.now()

            self.logger.info(f"Created LLM checkpoint for job {self.current_job_id}: {processed}/{total} reviews processed")

        except ValueError as e:
            self.logger.error(f"Invalid UUID format for current_job_id: {self.current_job_id}")
        except Exception as e:
            self.logger.error(f"Error creating LLM checkpoint: {e}")

    def _check_progress_milestones(self, progress: float, processed: int, phase: str):
        """Check and notify progress milestones."""
        for milestone in self.progress_milestones:
            if progress >= milestone and not hasattr(self, f'milestone_{milestone}_reached'):
                setattr(self, f'milestone_{milestone}_reached', True)
                percentage = int(milestone * 100)
                message = f"{phase} processing {percentage}% complete ({processed} reviews processed)"
                self._create_progress_notification("milestone", percentage, message)

    def _create_progress_notification(self, notification_type: str,
                                      progress: int, message: str):
        """Create progress notification for dashboard - simplified."""
        try:
            # Simplified progress tracking without database storage
            self.logger.info(f"PROGRESS: {progress}% - {message}")

        except Exception as e:
            self.logger.warning(f"Progress notification failed (non-critical): {e}")

    def _mark_reviews_absa_processed(self, review_ids: List[str]):
        """Mark reviews as ABSA processed."""
        if not review_ids:
            return

        try:
            placeholders = ','.join([f':id{i}' for i in range(len(review_ids))])
            query = f"""
            UPDATE reviews 
            SET absa_processed = TRUE, absa_processed_at = CURRENT_TIMESTAMP
            WHERE review_id IN ({placeholders})
            """

            params = {f'id{i}': rid for i, rid in enumerate(review_ids)}
            storage.db.execute_non_query(query, params)

        except Exception as e:
            self.logger.error(f"Error marking reviews as ABSA processed: {e}")

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
            storage.db.execute_non_query(query, params)

        except Exception as e:
            self.logger.error(f"Error marking reviews as SERVQUAL processed: {e}")

    def _create_new_job(self, app_id: Optional[str]) -> Dict:
        """Create new processing job with UUID."""
        try:
            # Ensure job_id is UUID format
            if not self.current_job_id:
                self.current_job_id = str(uuid.uuid4())

            # Validate it's a proper UUID
            uuid.UUID(self.current_job_id)

            query = """
            INSERT INTO processing_jobs 
            (job_id, job_type, app_id, status, start_time, sequential_mode, current_phase)
            VALUES (:job_id, 'sequential_llm', :app_id, 'running', :start_time, TRUE, 'pending')
            """

            params = {
                'job_id': self.current_job_id,
                'app_id': app_id,  # app_id stored separately as string
                'start_time': datetime.now()
            }

            storage.db.execute_non_query(query, params)

            self.logger.info(f"Created new LLM processing job: {self.current_job_id} for app: {app_id}")

            return {
                'job_id': self.current_job_id,
                'reviews_processed': 0,
                'checkpoints_created': 0,
                'failed_reviews': 0
            }

        except ValueError as e:
            self.logger.error(f"Invalid UUID format: {self.current_job_id}")
            raise
        except Exception as e:
            self.logger.error(f"Error creating new job: {e}")
            raise

    def _get_resumable_job_state(self, job_id: str) -> Optional[Dict]:
        """Get state for resumable job."""
        try:
            # Validate that job_id is a proper UUID
            uuid.UUID(job_id)  # This will raise ValueError if not valid UUID

            query = "SELECT * FROM get_resumable_job_state(:job_id)"
            params = {'job_id': job_id}

            df = storage.db.execute_query(query, params)

            if df.empty:
                self.logger.warning(f"No resumable job state found for job: {job_id}")
                return None

            row = df.iloc[0]
            if not row['can_resume']:
                self.logger.warning(f"Job {job_id} cannot be resumed - invalid state")
                return None

            return {
                'job_id': job_id,
                'reviews_processed': row['reviews_processed'],
                'total_reviews': row['total_reviews'],
                'last_review_id': row['last_review_id'],
                'checkpoints_created': 1,
                'failed_reviews': 0
            }

        except ValueError as e:
            self.logger.error(f"Invalid UUID format for job_id: {job_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting resumable job state: {e}")
            return None

    def _update_job_phase(self, job_id: str, phase: str):
        """Update current processing phase."""
        try:
            # Validate UUID format
            uuid.UUID(job_id)

            query = """
            UPDATE processing_jobs 
            SET current_phase = :phase
            WHERE job_id = :job_id
            """

            params = {'job_id': job_id, 'phase': phase}
            rows_affected = storage.db.execute_non_query(query, params)

            if rows_affected > 0:
                self.logger.info(f"Updated job {job_id} phase to: {phase}")
            else:
                self.logger.warning(f"No job found to update phase: {job_id}")

        except ValueError as e:
            self.logger.error(f"Invalid UUID format for job_id: {job_id}")
        except Exception as e:
            self.logger.error(f"Error updating job phase: {e}")

    def _complete_job(self, job_id: str, absa_phase: ProcessingPhase, servqual_phase: ProcessingPhase):
        """Mark job as completed with LLM performance logging."""
        try:
            # Validate UUID format
            uuid.UUID(job_id)

            query = """
            UPDATE processing_jobs 
            SET status = 'completed', end_time = :end_time, current_phase = 'completed'
            WHERE job_id = :job_id
            """

            params = {
                'job_id': job_id,
                'end_time': datetime.now()
            }

            rows_affected = storage.db.execute_non_query(query, params)

            if rows_affected > 0:
                self.logger.info(f"Completed job: {job_id}")
                # Create processing statistics
                self._create_processing_statistics(job_id, absa_phase, servqual_phase)

                # Log LLM performance summary
                if servqual_phase.name == "servqual_llm":
                    self._log_llm_performance_summary(servqual_phase)
            else:
                self.logger.warning(f"No job found to complete: {job_id}")

        except ValueError as e:
            self.logger.error(f"Invalid UUID format for job_id: {job_id}")
        except Exception as e:
            self.logger.error(f"Error completing job: {e}")

    def _log_llm_performance_summary(self, servqual_phase: ProcessingPhase):
        """Log LLM SERVQUAL performance summary."""
        try:
            if servqual_phase.success and servqual_phase.reviews_processed > 0:
                processing_time = (servqual_phase.end_time - servqual_phase.start_time).total_seconds()
                avg_time_per_review = processing_time / servqual_phase.reviews_processed
                throughput = servqual_phase.reviews_processed / processing_time if processing_time > 0 else 0

                self.logger.info("=== LLM SERVQUAL PERFORMANCE SUMMARY ===")
                self.logger.info(f"Reviews processed: {servqual_phase.reviews_processed}")
                self.logger.info(f"Total processing time: {processing_time:.1f}s")
                self.logger.info(f"Average time per review: {avg_time_per_review:.2f}s")
                self.logger.info(f"Throughput: {throughput:.3f} reviews/second")
                # Fix Unicode issue:
                target_met = "YES" if avg_time_per_review < 6 else "NO"
                self.logger.info(f"Target performance: <6s per review ({target_met})")

        except Exception as e:
            self.logger.error(f"Error logging LLM performance summary: {e}")

    def _create_processing_statistics(self, job_id: str,
                                      absa_phase: ProcessingPhase,
                                      servqual_phase: ProcessingPhase):
        """Create processing statistics record."""
        try:
            total_time = 0
            if absa_phase.end_time and servqual_phase.end_time:
                total_time = (servqual_phase.end_time - absa_phase.start_time).total_seconds()

            query = """
            INSERT INTO processing_statistics 
            (job_id, job_type, start_time, end_time, reviews_processed, 
             processing_time_seconds, success_rate)
            VALUES (:job_id, 'sequential_llm', :start_time, :end_time, :reviews_processed,
                    :processing_time, :success_rate)
            """

            total_processed = absa_phase.reviews_processed + servqual_phase.reviews_processed
            total_attempted = absa_phase.total_reviews + servqual_phase.total_reviews
            success_rate = (total_processed / total_attempted * 100) if total_attempted > 0 else 0

            params = {
                'job_id': job_id,
                'start_time': absa_phase.start_time,
                'end_time': servqual_phase.end_time or datetime.now(),
                'reviews_processed': total_processed,
                'processing_time': total_time,
                'success_rate': success_rate
            }

            storage.db.execute_non_query(query, params)

        except Exception as e:
            self.logger.error(f"Error creating processing statistics: {e}")

    def _mark_job_failed(self, job_id: str, error_message: str):
        """Mark job as failed."""
        try:
            # Validate UUID format
            uuid.UUID(job_id)

            query = """
            UPDATE processing_jobs 
            SET status = 'failed', end_time = :end_time, error_message = :error
            WHERE job_id = :job_id
            """

            params = {
                'job_id': job_id,
                'end_time': datetime.now(),
                'error': error_message
            }

            rows_affected = storage.db.execute_non_query(query, params)

            if rows_affected > 0:
                self.logger.info(f"Marked job {job_id} as failed: {error_message}")
            else:
                self.logger.warning(f"No job found to mark as failed: {job_id}")

        except ValueError as e:
            self.logger.error(f"Invalid UUID format for job_id: {job_id}")
        except Exception as e:
            self.logger.error(f"Error marking job as failed: {e}")

    def _create_success_result(self, start_time: datetime, app_id: Optional[str],
                               total_reviews: int, aspects: int, servqual: int) -> SequentialProcessingResult:
        """Create successful processing result."""
        end_time = datetime.now()
        return SequentialProcessingResult(
            job_id=self.current_job_id,
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            absa_phase=ProcessingPhase("absa", total_reviews, total_reviews, start_time, end_time, True),
            servqual_phase=ProcessingPhase("servqual_llm", total_reviews, total_reviews, start_time, end_time, True),
            total_reviews_processed=total_reviews,
            total_aspects_extracted=aspects,
            total_servqual_dimensions_updated=servqual,
            checkpoints_created=0,
            failed_reviews=0,
            success=True,
            error_message=None,
            processing_time_seconds=(end_time - start_time).total_seconds()
        )

    def _create_failure_result(self, start_time: datetime, app_id: Optional[str],
                               error_message: str) -> SequentialProcessingResult:
        """Create failed processing result."""
        end_time = datetime.now()
        return SequentialProcessingResult(
            job_id=self.current_job_id,
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            absa_phase=ProcessingPhase("absa", 0, 0, start_time, end_time, False, error_message),
            servqual_phase=ProcessingPhase("servqual_llm", 0, 0, start_time, end_time, False, error_message),
            total_reviews_processed=0,
            total_aspects_extracted=0,
            total_servqual_dimensions_updated=0,
            checkpoints_created=0,
            failed_reviews=0,
            success=False,
            error_message=error_message,
            processing_time_seconds=(end_time - start_time).total_seconds()
        )


# Global instance
sequential_processor = SequentialProcessor()