"""
SERVQUAL Mapper for ABSA Pipeline.
Maps aspect-level sentiment analysis to SERVQUAL service quality dimensions.
Converts sentiment scores to business-actionable quality ratings (1-5 scale).
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
from dataclasses import dataclass
import pandas as pd

from src.utils.config import config
from src.data.storage import storage


@dataclass
class ServqualResult:
    """Structured SERVQUAL dimension result."""
    app_id: str
    dimension: str
    sentiment_score: float
    quality_score: int
    review_count: int
    date: date
    contributing_aspects: List[str]


class ServqualMapper:
    """Maps ABSA results to SERVQUAL service quality dimensions."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.servqual.mapper")

        # SERVQUAL dimensions
        self.dimensions = {
            'reliability': 'Platform consistency and order processing accuracy',
            'assurance': 'Security, trust, and professional competence',
            'tangibles': 'Physical appearance and interface design',
            'empathy': 'Personal attention and customer understanding',
            'responsiveness': 'Speed of service and communication'
        }

        # Load aspect-to-dimension mapping from database
        self._load_aspect_mappings()

    def _load_aspect_mappings(self) -> None:
        """Load aspect-to-SERVQUAL dimension mappings from database."""
        try:
            query = """
            SELECT aspect_name, servqual_dimension, weight
            FROM aspects 
            WHERE servqual_dimension IS NOT NULL
            """

            df = storage.db.execute_query(query)

            if not df.empty:
                self.aspect_mappings = dict(zip(df['aspect_name'], df['servqual_dimension']))
                self.aspect_weights = dict(zip(df['aspect_name'], df['weight']))
                self.logger.info(f"Loaded {len(self.aspect_mappings)} aspect-to-SERVQUAL mappings")
            else:
                self.logger.warning("No aspect-to-SERVQUAL mappings found in database")
                self.aspect_mappings = {}
                self.aspect_weights = {}

        except Exception as e:
            self.logger.error(f"Error loading aspect mappings: {e}")
            self.aspect_mappings = {}
            self.aspect_weights = {}

    def sentiment_to_quality_score(self, sentiment_score: float) -> int:
        """
        Convert sentiment score (-1 to +1) to SERVQUAL quality scale (1-5).

        Args:
            sentiment_score: Sentiment score from ABSA analysis (-1 to +1)

        Returns:
            Quality score on SERVQUAL scale (1-5)
        """
        if sentiment_score >= 0.8:
            return 5  # Exceeds Expectations
        elif sentiment_score >= 0.3:
            return 4  # Meets Expectations
        elif sentiment_score >= -0.2:
            return 3  # Neutral/Mixed
        elif sentiment_score >= -0.7:
            return 2  # Below Expectations
        else:
            return 1  # Far Below Expectations

    def map_aspects_to_servqual(self, aspect_results: List[Dict[str, Any]]) -> List[ServqualResult]:
        """
        Map aspect-level sentiment results to SERVQUAL dimensions.

        Args:
            aspect_results: List of aspect analysis results with sentiment scores

        Returns:
            List of SERVQUAL dimension results
        """
        if not aspect_results:
            return []

        try:
            # Group aspects by SERVQUAL dimension
            dimension_groups = {}

            for aspect_result in aspect_results:
                aspect_name = aspect_result.get('aspect')
                dimension = self.aspect_mappings.get(aspect_name)

                if not dimension:
                    self.logger.debug(f"No SERVQUAL mapping for aspect: {aspect_name}")
                    continue

                if dimension not in dimension_groups:
                    dimension_groups[dimension] = []

                dimension_groups[dimension].append(aspect_result)

            # Calculate dimension-level scores
            servqual_results = []

            for dimension, aspects in dimension_groups.items():
                result = self._aggregate_dimension_sentiment(dimension, aspects)
                if result:
                    servqual_results.append(result)

            self.logger.info(f"Mapped {len(aspect_results)} aspects to {len(servqual_results)} SERVQUAL dimensions")
            return servqual_results

        except Exception as e:
            self.logger.error(f"Error mapping aspects to SERVQUAL: {e}")
            return []

    def _aggregate_dimension_sentiment(self, dimension: str, aspects: List[Dict[str, Any]]) -> Optional[ServqualResult]:
        """
        Aggregate sentiment from multiple aspects into single dimension score.

        Args:
            dimension: SERVQUAL dimension name
            aspects: List of aspect results for this dimension

        Returns:
            Aggregated SERVQUAL result for the dimension
        """
        try:
            if not aspects:
                return None

            # Calculate weighted average sentiment
            total_weighted_sentiment = 0.0
            total_weight = 0.0
            review_count = 0
            contributing_aspects = []
            app_id = None
            analysis_date = None

            for aspect in aspects:
                aspect_name = aspect.get('aspect', '')
                sentiment_score = aspect.get('sentiment_score', 0.0)
                weight = self.aspect_weights.get(aspect_name, 1.0)
                count = aspect.get('review_count', 1)

                total_weighted_sentiment += sentiment_score * weight * count
                total_weight += weight * count
                review_count += count
                contributing_aspects.append(aspect_name)

                # Get common fields
                if not app_id:
                    app_id = aspect.get('app_id')
                if not analysis_date:
                    analysis_date = aspect.get('date', datetime.now().date())

            # Calculate final scores
            if total_weight > 0:
                avg_sentiment = total_weighted_sentiment / total_weight
            else:
                avg_sentiment = 0.0

            quality_score = self.sentiment_to_quality_score(avg_sentiment)

            return ServqualResult(
                app_id=app_id,
                dimension=dimension,
                sentiment_score=round(avg_sentiment, 3),
                quality_score=quality_score,
                review_count=review_count,
                date=analysis_date,
                contributing_aspects=contributing_aspects
            )

        except Exception as e:
            self.logger.error(f"Error aggregating dimension sentiment for {dimension}: {e}")
            return None

    def process_daily_servqual(self, app_id: str, target_date: date = None) -> List[ServqualResult]:
        """
        Process daily SERVQUAL scores for an app.

        Args:
            app_id: Application identifier
            target_date: Date to process (defaults to today)

        Returns:
            List of SERVQUAL results for all dimensions
        """
        if target_date is None:
            target_date = datetime.now().date()

        try:
            # Get aspect-level results for the date
            aspect_query = """
            SELECT 
                da.app_id,
                da.aspect,
                AVG(da.sentiment_score) as sentiment_score,
                AVG(da.confidence_score) as confidence_score,
                COUNT(*) as review_count,
                :target_date as date
            FROM deep_absa da
            INNER JOIN reviews r ON da.review_id = r.review_id
            WHERE da.app_id = :app_id 
            AND DATE(r.review_date) = :target_date
            GROUP BY da.app_id, da.aspect
            """

            params = {
                'app_id': app_id,
                'target_date': target_date
            }

            df = storage.db.execute_query(aspect_query, params)

            if df.empty:
                self.logger.info(f"No aspect data found for {app_id} on {target_date}")
                return []

            # Convert to list of dictionaries
            aspect_results = df.to_dict('records')

            # Map to SERVQUAL dimensions
            servqual_results = self.map_aspects_to_servqual(aspect_results)

            self.logger.info(f"Processed SERVQUAL for {app_id} on {target_date}: {len(servqual_results)} dimensions")
            return servqual_results

        except Exception as e:
            self.logger.error(f"Error processing daily SERVQUAL for {app_id}: {e}")
            return []

    def get_dimension_description(self, dimension: str) -> str:
        """Get description for a SERVQUAL dimension."""
        return self.dimensions.get(dimension, f"Unknown dimension: {dimension}")

    def get_quality_score_label(self, quality_score: int) -> str:
        """Get descriptive label for quality score."""
        labels = {
            5: "Exceeds Expectations",
            4: "Meets Expectations",
            3: "Neutral/Mixed",
            2: "Below Expectations",
            1: "Far Below Expectations"
        }
        return labels.get(quality_score, "Unknown Score")

    def validate_servqual_result(self, result: ServqualResult) -> bool:
        """Validate SERVQUAL result before storage."""
        try:
            # Check required fields
            if not all([result.app_id, result.dimension, result.date]):
                return False

            # Check score ranges
            if not (-1.0 <= result.sentiment_score <= 1.0):
                return False

            if not (1 <= result.quality_score <= 5):
                return False

            # Check dimension validity
            if result.dimension not in self.dimensions:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating SERVQUAL result: {e}")
            return False


# Convenience functions
def process_app_servqual(app_id: str, target_date: date = None) -> List[ServqualResult]:
    """Convenience function to process SERVQUAL for an app."""
    mapper = ServqualMapper()
    return mapper.process_daily_servqual(app_id, target_date)


def sentiment_to_quality(sentiment_score: float) -> int:
    """Convenience function for sentiment-to-quality conversion."""
    mapper = ServqualMapper()
    return mapper.sentiment_to_quality_score(sentiment_score)