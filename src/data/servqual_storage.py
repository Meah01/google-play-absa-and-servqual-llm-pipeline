"""
SERVQUAL Storage Operations for ABSA Pipeline.
Handles database operations specific to SERVQUAL service quality scores.
Provides data access for SERVQUAL dashboard and analytics.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any, Tuple

from src.data.storage import storage
from src.absa.servqual_mapper import ServqualResult


class ServqualStorage:
    """Handles SERVQUAL-specific database operations."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.servqual.storage")
        self.db = storage.db

    def store_servqual_scores(self, servqual_results: List[ServqualResult]) -> Tuple[int, int]:
        """
        Store SERVQUAL dimension scores in database.

        Args:
            servqual_results: List of SERVQUAL results to store

        Returns:
            Tuple of (inserted_count, error_count)
        """
        if not servqual_results:
            return 0, 0

        inserted_count = 0
        error_count = 0

        insert_query = """
        INSERT INTO servqual_scores 
        (app_id, dimension, sentiment_score, quality_score, review_count, date)
        VALUES (:app_id, :dimension, :sentiment_score, :quality_score, :review_count, :date)
        ON CONFLICT (app_id, dimension, date)
        DO UPDATE SET
            sentiment_score = EXCLUDED.sentiment_score,
            quality_score = EXCLUDED.quality_score,
            review_count = EXCLUDED.review_count,
            created_at = CURRENT_TIMESTAMP
        """

        for result in servqual_results:
            try:
                params = {
                    'app_id': result.app_id,
                    'dimension': result.dimension,
                    'sentiment_score': result.sentiment_score,
                    'quality_score': result.quality_score,
                    'review_count': result.review_count,
                    'date': result.date
                }

                rows_affected = self.db.execute_non_query(insert_query, params)
                if rows_affected > 0:
                    inserted_count += 1

            except Exception as e:
                self.logger.error(f"Error storing SERVQUAL result: {e}")
                error_count += 1

        self.logger.info(f"Stored SERVQUAL scores: {inserted_count} inserted, {error_count} errors")
        return inserted_count, error_count

    def get_app_servqual_profile(self, app_id: str, target_date: date = None) -> Dict[str, Any]:
        """
        Get current SERVQUAL profile for an app.

        Args:
            app_id: Application identifier
            target_date: Specific date (defaults to latest available)

        Returns:
            Dictionary with dimension scores and metadata
        """
        try:
            if target_date:
                date_filter = "AND date = :target_date"
                params = {'app_id': app_id, 'target_date': target_date}
            else:
                date_filter = "AND date = (SELECT MAX(date) FROM servqual_scores WHERE app_id = :app_id)"
                params = {'app_id': app_id}

            query = f"""
            SELECT 
                dimension,
                sentiment_score,
                quality_score,
                review_count,
                date,
                created_at
            FROM servqual_scores
            WHERE app_id = :app_id
            {date_filter}
            ORDER BY dimension
            """

            df = self.db.execute_query(query, params)

            if df.empty:
                return {}

            # Structure the profile
            profile = {
                'app_id': app_id,
                'date': df.iloc[0]['date'],
                'dimensions': {},
                'overall_quality': 0.0,
                'total_reviews': 0
            }

            total_score = 0
            dimension_count = 0

            for _, row in df.iterrows():
                dimension = row['dimension']
                profile['dimensions'][dimension] = {
                    'sentiment_score': float(row['sentiment_score']),
                    'quality_score': int(row['quality_score']),
                    'review_count': int(row['review_count']),
                    'label': self._get_quality_label(row['quality_score'])
                }

                total_score += row['quality_score']
                dimension_count += 1
                profile['total_reviews'] += row['review_count']

            # Calculate overall quality score
            if dimension_count > 0:
                profile['overall_quality'] = round(total_score / dimension_count, 2)

            return profile

        except Exception as e:
            self.logger.error(f"Error getting SERVQUAL profile for {app_id}: {e}")
            return {}

    def get_dimension_trends(self, app_id: Optional[str] = None, dimension: str = None,
                             days: int = 30) -> pd.DataFrame:
        """
        Get SERVQUAL dimension trends over time.

        Args:
            app_id: Application identifier (None for all apps)
            dimension: Specific dimension (None for all dimensions)
            days: Number of days to include

        Returns:
            DataFrame with time series data
        """
        try:
            # Build query conditions
            conditions = []
            params = {'days': days}

            if app_id:
                conditions.append("app_id = :app_id")
                params['app_id'] = app_id

            if dimension:
                conditions.append("dimension = :dimension")
                params['dimension'] = dimension

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions) + " AND "
            else:
                where_clause = "WHERE "

            query = f"""
            SELECT 
                app_id,
                dimension,
                date,
                sentiment_score,
                quality_score,
                review_count
            FROM servqual_scores
            {where_clause} date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY app_id, dimension, date
            """

            df = self.db.execute_query(query, params)
            return df

        except Exception as e:
            self.logger.error(f"Error getting dimension trends: {e}")
            return pd.DataFrame()

    def get_comparative_analysis(self, dimension: str, days: int = 30) -> pd.DataFrame:
        """
        Get comparative analysis across apps for a specific dimension.

        Args:
            dimension: SERVQUAL dimension to analyze
            days: Number of days to include

        Returns:
            DataFrame with comparative data
        """
        try:
            query = """
            SELECT 
                s.app_id,
                a.app_name,
                s.dimension,
                AVG(s.sentiment_score) as avg_sentiment,
                AVG(s.quality_score) as avg_quality,
                SUM(s.review_count) as total_reviews,
                COUNT(*) as data_points
            FROM servqual_scores s
            INNER JOIN apps a ON s.app_id = a.app_id
            WHERE s.dimension = :dimension
            AND s.date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY s.app_id, a.app_name, s.dimension
            ORDER BY avg_quality DESC, total_reviews DESC
            """ % days

            params = {'dimension': dimension}
            df = self.db.execute_query(query, params)
            return df

        except Exception as e:
            self.logger.error(f"Error getting comparative analysis for {dimension}: {e}")
            return pd.DataFrame()

    def get_dimension_summary(self, app_id: str = None, days: int = 7) -> Dict[str, Any]:
        """
        Get summary statistics for SERVQUAL dimensions.

        Args:
            app_id: Application identifier (None for all apps)
            days: Number of days to include

        Returns:
            Dictionary with summary statistics
        """
        try:
            app_filter = ""
            params = {'days': days}

            if app_id:
                app_filter = "AND app_id = :app_id"
                params['app_id'] = app_id

            query = f"""
            SELECT 
                dimension,
                AVG(sentiment_score) as avg_sentiment,
                AVG(quality_score) as avg_quality,
                MIN(quality_score) as min_quality,
                MAX(quality_score) as max_quality,
                SUM(review_count) as total_reviews,
                COUNT(DISTINCT app_id) as app_count,
                COUNT(DISTINCT date) as day_count
            FROM servqual_scores
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
            {app_filter}
            GROUP BY dimension
            ORDER BY avg_quality DESC
            """

            df = self.db.execute_query(query, params)

            if df.empty:
                return {}

            summary = {
                'period_days': days,
                'app_id': app_id,
                'dimensions': {},
                'overall_stats': {}
            }

            total_reviews = df['total_reviews'].sum()
            avg_quality = df['avg_quality'].mean()

            for _, row in df.iterrows():
                dimension = row['dimension']
                summary['dimensions'][dimension] = {
                    'avg_sentiment': round(float(row['avg_sentiment']), 3),
                    'avg_quality': round(float(row['avg_quality']), 2),
                    'min_quality': int(row['min_quality']),
                    'max_quality': int(row['max_quality']),
                    'total_reviews': int(row['total_reviews']),
                    'quality_label': self._get_quality_label(row['avg_quality'])
                }

            summary['overall_stats'] = {
                'total_reviews': int(total_reviews),
                'avg_quality_score': round(float(avg_quality), 2),
                'dimension_count': len(df),
                'quality_label': self._get_quality_label(avg_quality)
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting dimension summary: {e}")
            return {}

    def get_amazon_focus_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Get specialized data for Amazon SERVQUAL analysis.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with Amazon-focused SERVQUAL data
        """
        try:
            amazon_app_id = "com.amazon.mShop.android.shopping"

            # Get current profile
            current_profile = self.get_app_servqual_profile(amazon_app_id)

            # Get trend data
            trends_df = self.get_dimension_trends(amazon_app_id, days=days)

            # Get comparative ranking
            ranking_data = {}
            for dimension in ['reliability', 'assurance', 'tangibles', 'empathy', 'responsiveness']:
                comp_df = self.get_comparative_analysis(dimension, days=days)
                if not comp_df.empty:
                    amazon_rank = comp_df[comp_df['app_id'] == amazon_app_id]
                    if not amazon_rank.empty:
                        total_apps = len(comp_df)
                        amazon_position = comp_df.index[comp_df['app_id'] == amazon_app_id].tolist()[0] + 1
                        ranking_data[dimension] = {
                            'rank': amazon_position,
                            'total_apps': total_apps,
                            'percentile': round((total_apps - amazon_position + 1) / total_apps * 100, 1)
                        }

            return {
                'current_profile': current_profile,
                'trends': trends_df.to_dict('records') if not trends_df.empty else [],
                'competitive_ranking': ranking_data,
                'analysis_period': days
            }

        except Exception as e:
            self.logger.error(f"Error getting Amazon focus data: {e}")
            return {}

    def _get_quality_label(self, quality_score: float) -> str:
        """Get descriptive label for quality score."""
        if quality_score >= 4.5:
            return "Exceeds Expectations"
        elif quality_score >= 3.5:
            return "Meets Expectations"
        elif quality_score >= 2.5:
            return "Neutral/Mixed"
        elif quality_score >= 1.5:
            return "Below Expectations"
        else:
            return "Far Below Expectations"

    def delete_servqual_scores(self, app_id: str, start_date: date = None,
                               end_date: date = None) -> int:
        """
        Delete SERVQUAL scores for an app within date range.

        Args:
            app_id: Application identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Number of records deleted
        """
        try:
            conditions = ["app_id = :app_id"]
            params = {'app_id': app_id}

            if start_date:
                conditions.append("date >= :start_date")
                params['start_date'] = start_date

            if end_date:
                conditions.append("date <= :end_date")
                params['end_date'] = end_date

            where_clause = " AND ".join(conditions)

            query = f"DELETE FROM servqual_scores WHERE {where_clause}"
            deleted_count = self.db.execute_non_query(query, params)

            self.logger.info(f"Deleted {deleted_count} SERVQUAL scores for {app_id}")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error deleting SERVQUAL scores: {e}")
            return 0


# Global instance
servqual_storage = ServqualStorage()