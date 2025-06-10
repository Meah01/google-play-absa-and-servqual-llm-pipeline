"""
Unified ABSA Engine Interface for ABSA Pipeline.
Provides a single interface for both deep and quick ABSA analysis.
Orchestrates deep engine (Phase 2) and quick engine (Phase 3) with consistent API.
Integrates with batch processing pipeline and SERVQUAL mapping.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.utils.config import config
from src.absa.deep_engine import (
    DeepABSAEngine,
    ReviewAnalysisResult,
    AspectResult,
    analyze_review_batch as deep_analyze_batch
)


class AnalysisMode(Enum):
    """ABSA analysis mode selection."""
    DEEP = "deep"        # High accuracy, slower processing for business intelligence
    QUICK = "quick"      # Fast processing, real-time analysis (Phase 3)
    AUTO = "auto"        # Automatically select based on context


@dataclass
class EngineStatus:
    """Status information for the ABSA engine."""
    deep_engine_loaded: bool
    quick_engine_loaded: bool
    current_mode: AnalysisMode
    models_memory_usage_mb: float
    total_reviews_processed: int
    last_batch_processing_time_ms: int
    engine_version: str


@dataclass
class BatchProcessingResult:
    """Result of batch processing operation."""
    total_reviews: int
    successful_reviews: int
    failed_reviews: int
    total_aspects_extracted: int
    processing_time_seconds: float
    database_records: List[Dict[str, Any]]
    analysis_summary: Dict[str, Any]
    engine_mode: AnalysisMode


class ABSAEngine:
    """
    Unified ABSA Engine that orchestrates deep and quick analysis.
    Provides consistent interface for all ABSA operations in the pipeline.
    """

    def __init__(self, default_mode: AnalysisMode = AnalysisMode.DEEP):
        self.logger = logging.getLogger("absa_pipeline.engine")

        # Configuration
        self.default_mode = default_mode
        self.batch_size = config.absa.batch_size
        self.engine_version = "2.0.0"

        # Engine instances
        self.deep_engine = None
        self.quick_engine = None  # Phase 3 implementation

        # Statistics
        self.total_reviews_processed = 0
        self.last_batch_time = 0

        # Initialize based on default mode
        self._initialize_engines()

        self.logger.info(f"Unified ABSA Engine initialized with mode: {default_mode.value}")

    def _initialize_engines(self):
        """Initialize required engines based on configuration."""
        try:
            # Always initialize deep engine for Phase 2
            if self.default_mode in [AnalysisMode.DEEP, AnalysisMode.AUTO]:
                self.deep_engine = DeepABSAEngine()
                self.logger.info("Deep ABSA engine initialized")

            # Quick engine initialization (Phase 3)
            if self.default_mode == AnalysisMode.QUICK:
                self.logger.info("Quick ABSA engine initialization - Phase 3 implementation pending")
                # TODO: Initialize QuickABSAEngine when implemented

        except Exception as e:
            self.logger.error(f"Error initializing ABSA engines: {e}")
            raise

    def analyze_review(self, review_id: str, app_id: str, review_text: str,
                      mode: Optional[AnalysisMode] = None) -> ReviewAnalysisResult:
        """
        Analyze a single review for aspects and sentiment.

        Args:
            review_id: Unique review identifier
            app_id: Application identifier
            review_text: Review content to analyze
            mode: Analysis mode override (defaults to engine default)

        Returns:
            Complete analysis result
        """
        analysis_mode = mode or self.default_mode

        try:
            if analysis_mode == AnalysisMode.DEEP:
                if not self.deep_engine:
                    self.deep_engine = DeepABSAEngine()

                result = self.deep_engine.analyze_single_review(review_id, app_id, review_text)
                self.total_reviews_processed += 1
                return result

            elif analysis_mode == AnalysisMode.QUICK:
                # Phase 3: Quick analysis implementation
                self.logger.warning("Quick analysis mode not yet implemented (Phase 3)")
                # Fallback to deep analysis for now
                if not self.deep_engine:
                    self.deep_engine = DeepABSAEngine()
                return self.deep_engine.analyze_single_review(review_id, app_id, review_text)

            elif analysis_mode == AnalysisMode.AUTO:
                # Auto mode: Use deep analysis for batch processing, quick for real-time
                # For now, default to deep analysis
                return self.analyze_review(review_id, app_id, review_text, AnalysisMode.DEEP)

            else:
                raise ValueError(f"Unknown analysis mode: {analysis_mode}")

        except Exception as e:
            self.logger.error(f"Error analyzing review {review_id}: {e}")
            raise

    def analyze_batch(self, reviews: List[Dict[str, str]],
                     mode: Optional[AnalysisMode] = None) -> BatchProcessingResult:
        """
        Analyze a batch of reviews efficiently.

        Args:
            reviews: List of review dictionaries with keys: review_id, app_id, content
            mode: Analysis mode override

        Returns:
            Batch processing result with statistics and database records
        """
        analysis_mode = mode or self.default_mode
        start_time = time.time()

        self.logger.info(f"Starting batch analysis of {len(reviews)} reviews in {analysis_mode.value} mode")

        try:
            # Route to appropriate engine
            if analysis_mode == AnalysisMode.DEEP:
                results = self._analyze_batch_deep(reviews)
            elif analysis_mode == AnalysisMode.QUICK:
                results = self._analyze_batch_quick(reviews)
            elif analysis_mode == AnalysisMode.AUTO:
                # Auto mode: prefer deep for batch processing
                results = self._analyze_batch_deep(reviews)
            else:
                raise ValueError(f"Unknown analysis mode: {analysis_mode}")

            # Calculate processing statistics
            processing_time = time.time() - start_time
            self.last_batch_time = int(processing_time * 1000)

            # Generate analysis summary
            if self.deep_engine and analysis_mode == AnalysisMode.DEEP:
                analysis_summary = self.deep_engine.get_analysis_summary(results)
            else:
                analysis_summary = self._generate_basic_summary(results)

            # Convert to database format
            database_records = self._convert_to_database_format(results, analysis_mode)

            # Count statistics
            successful_reviews = len([r for r in results if r.aspects])
            failed_reviews = len(results) - successful_reviews
            total_aspects = sum(len(r.aspects) for r in results)

            # Update global statistics
            self.total_reviews_processed += len(reviews)

            batch_result = BatchProcessingResult(
                total_reviews=len(reviews),
                successful_reviews=successful_reviews,
                failed_reviews=failed_reviews,
                total_aspects_extracted=total_aspects,
                processing_time_seconds=processing_time,
                database_records=database_records,
                analysis_summary=analysis_summary,
                engine_mode=analysis_mode
            )

            self.logger.info(f"Batch analysis complete: {successful_reviews}/{len(reviews)} successful, "
                           f"{total_aspects} aspects extracted, {processing_time:.2f}s")

            return batch_result

        except Exception as e:
            self.logger.error(f"Error in batch analysis: {e}")
            raise

    def _analyze_batch_deep(self, reviews: List[Dict[str, str]]) -> List[ReviewAnalysisResult]:
        """Analyze batch using deep ABSA engine."""
        if not self.deep_engine:
            self.deep_engine = DeepABSAEngine()

        return self.deep_engine.analyze_batch(reviews)

    def _analyze_batch_quick(self, reviews: List[Dict[str, str]]) -> List[ReviewAnalysisResult]:
        """Analyze batch using quick ABSA engine (Phase 3)."""
        self.logger.warning("Quick batch analysis not implemented - using deep analysis")
        return self._analyze_batch_deep(reviews)

    def _convert_to_database_format(self, results: List[ReviewAnalysisResult],
                                  mode: AnalysisMode) -> List[Dict[str, Any]]:
        """Convert results to database format based on analysis mode."""
        if mode == AnalysisMode.DEEP:
            if self.deep_engine:
                return self.deep_engine.convert_to_database_format(results)

        # Fallback conversion
        database_records = []
        for result in results:
            for aspect in result.aspects:
                record = {
                    'review_id': result.review_id,
                    'app_id': result.app_id,
                    'aspect': aspect.aspect,
                    'sentiment_score': aspect.sentiment_score,
                    'confidence_score': aspect.confidence_score,
                    'opinion_text': aspect.opinion_text,
                    'opinion_start_pos': aspect.opinion_start_pos if hasattr(aspect, 'opinion_start_pos') else 0,
                    'opinion_end_pos': aspect.opinion_end_pos if hasattr(aspect, 'opinion_end_pos') else 0,
                    'processing_model': result.processing_model,
                    'processing_version': result.processing_version
                }
                database_records.append(record)

        return database_records

    def _generate_basic_summary(self, results: List[ReviewAnalysisResult]) -> Dict[str, Any]:
        """Generate basic summary statistics for results."""
        if not results:
            return {}

        total_reviews = len(results)
        successful_reviews = len([r for r in results if r.aspects])
        total_aspects = sum(len(r.aspects) for r in results)

        all_sentiments = []
        all_confidences = []
        aspect_counts = {}

        for result in results:
            for aspect in result.aspects:
                all_sentiments.append(aspect.sentiment_score)
                all_confidences.append(aspect.confidence_score)
                aspect_counts[aspect.aspect] = aspect_counts.get(aspect.aspect, 0) + 1

        return {
            'total_reviews': total_reviews,
            'successful_reviews': successful_reviews,
            'total_aspects_extracted': total_aspects,
            'avg_sentiment_score': sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0,
            'avg_confidence_score': sum(all_confidences) / len(all_confidences) if all_confidences else 0,
            'most_common_aspects': sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def get_engine_status(self) -> EngineStatus:
        """Get current engine status and statistics."""
        try:
            # Get memory usage from model manager
            from src.absa.models import model_manager
            memory_status = model_manager.get_memory_status()
            memory_usage = memory_status.get('cache', {}).get('cache_size_mb', 0)

        except Exception as e:
            self.logger.warning(f"Could not get memory status: {e}")
            memory_usage = 0

        return EngineStatus(
            deep_engine_loaded=self.deep_engine is not None,
            quick_engine_loaded=self.quick_engine is not None,
            current_mode=self.default_mode,
            models_memory_usage_mb=memory_usage,
            total_reviews_processed=self.total_reviews_processed,
            last_batch_processing_time_ms=self.last_batch_time,
            engine_version=self.engine_version
        )

    def process_reviews_for_servqual(self, reviews: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process reviews specifically for SERVQUAL integration.
        Optimized workflow for business intelligence pipeline.

        Args:
            reviews: List of review dictionaries

        Returns:
            Database records ready for SERVQUAL mapping
        """
        self.logger.info(f"Processing {len(reviews)} reviews for SERVQUAL analysis")

        try:
            # Use deep analysis for SERVQUAL (requires high accuracy)
            batch_result = self.analyze_batch(reviews, mode=AnalysisMode.DEEP)

            self.logger.info(f"SERVQUAL processing complete: {batch_result.total_aspects_extracted} "
                           f"aspects extracted from {batch_result.successful_reviews} reviews")

            return batch_result.database_records

        except Exception as e:
            self.logger.error(f"Error processing reviews for SERVQUAL: {e}")
            raise

    def get_supported_aspects(self) -> List[Dict[str, Any]]:
        """Get list of supported aspects for analysis."""
        try:
            from src.data.storage import storage

            query = """
            SELECT aspect_name, servqual_dimension, weight, description
            FROM aspects 
            WHERE is_active = TRUE
            ORDER BY weight DESC, aspect_name
            """

            df = storage.db.execute_query(query)
            return df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Error getting supported aspects: {e}")
            return []

    def validate_review_data(self, reviews: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Validate review data before processing.

        Args:
            reviews: List of review dictionaries

        Returns:
            Tuple of (valid_reviews, validation_errors)
        """
        valid_reviews = []
        errors = []

        required_fields = ['review_id', 'app_id', 'content']

        for i, review in enumerate(reviews):
            # Check required fields
            missing_fields = [field for field in required_fields if not review.get(field)]
            if missing_fields:
                errors.append(f"Review {i}: missing fields {missing_fields}")
                continue

            # Check content length
            content = review['content'].strip()
            if len(content) < 5:
                errors.append(f"Review {i}: content too short")
                continue

            if len(content) > 10000:  # Reasonable upper limit
                errors.append(f"Review {i}: content too long")
                continue

            valid_reviews.append(review)

        if errors:
            self.logger.warning(f"Validation found {len(errors)} issues in {len(reviews)} reviews")

        return valid_reviews, errors

    def clear_models(self):
        """Clear all loaded models to free memory."""
        try:
            from src.absa.models import model_manager
            model_manager.clear_models()

            self.deep_engine = None
            self.quick_engine = None

            self.logger.info("All ABSA models cleared from memory")

        except Exception as e:
            self.logger.error(f"Error clearing models: {e}")


# Global engine instance for convenience
absa_engine = ABSAEngine(default_mode=AnalysisMode.DEEP)


# Convenience functions for easy usage
def analyze_review_for_servqual(review_id: str, app_id: str, review_text: str) -> ReviewAnalysisResult:
    """Convenience function for single review SERVQUAL analysis."""
    return absa_engine.analyze_review(review_id, app_id, review_text, mode=AnalysisMode.DEEP)


def process_batch_for_servqual(reviews: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convenience function for batch SERVQUAL processing."""
    return absa_engine.process_reviews_for_servqual(reviews)


def get_absa_engine_status() -> EngineStatus:
    """Convenience function to get engine status."""
    return absa_engine.get_engine_status()


def validate_and_process_reviews(reviews: List[Dict[str, str]]) -> BatchProcessingResult:
    """Convenience function with validation and processing."""
    valid_reviews, errors = absa_engine.validate_review_data(reviews)

    if errors:
        logging.getLogger("absa_pipeline.engine").warning(f"Found {len(errors)} validation errors")

    if valid_reviews:
        return absa_engine.analyze_batch(valid_reviews)
    else:
        raise ValueError("No valid reviews to process")