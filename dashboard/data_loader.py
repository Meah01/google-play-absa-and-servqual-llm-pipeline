"""
Enhanced Data Loader for ABSA Dashboard Components.
Handles data aggregation and preparation for enhanced dashboard visualizations.
Supports Amazon-focused ABSA analysis and customizable aspect analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import logging

from src.data.storage import storage


class DashboardDataLoader:
    """Handles data loading and aggregation for enhanced dashboard components."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.dashboard.data_loader")

        # Aspect categories for UX grouping
        self.aspect_categories = {
            'Product': ['product_quality', 'product_description', 'product_variety', 'pricing_value'],
            'Service': ['customer_service', 'return_refund', 'seller_trust'],
            'Technical': ['app_performance', 'app_usability', 'search_navigation', 'checkout_process'],
            'Logistics': ['shipping_delivery', 'shipping_cost', 'order_tracking'],
            'Security': ['payment_security']
        }

        # User-friendly aspect names
        self.aspect_display_names = {
            'product_quality': 'Product Quality',
            'product_description': 'Product Description',
            'product_variety': 'Product Variety',
            'pricing_value': 'Pricing & Value',
            'customer_service': 'Customer Service',
            'return_refund': 'Return & Refund',
            'seller_trust': 'Seller Trust',
            'app_performance': 'App Performance',
            'app_usability': 'App Usability',
            'search_navigation': 'Search & Navigation',
            'checkout_process': 'Checkout Process',
            'shipping_delivery': 'Shipping & Delivery',
            'shipping_cost': 'Shipping Cost',
            'order_tracking': 'Order Tracking',
            'payment_security': 'Payment Security'
        }

    def load_amazon_fixed_absa(self, days: int = 30) -> Dict[str, Any]:
        """
        Load Amazon fixed ABSA table with top business-critical aspects.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with Amazon ABSA percentage data
        """
        try:
            amazon_app_id = "com.amazon.mShop.android.shopping"

            # Top business-critical aspects for Amazon
            critical_aspects = [
                'product_quality', 'customer_service', 'app_performance',
                'shipping_delivery', 'return_refund', 'pricing_value',
                'payment_security', 'app_usability'
            ]

            aspects_placeholder = ','.join([f"'{aspect}'" for aspect in critical_aspects])

            query = f"""
            SELECT 
                da.aspect,
                COUNT(*) as total_mentions,
                AVG(da.sentiment_score) as avg_sentiment,
                AVG(da.confidence_score) as avg_confidence,
                COUNT(*) FILTER (WHERE da.sentiment_score > 0.1) as positive_count,
                COUNT(*) FILTER (WHERE da.sentiment_score BETWEEN -0.1 AND 0.1) as neutral_count,
                COUNT(*) FILTER (WHERE da.sentiment_score < -0.1) as negative_count
            FROM deep_absa da
            INNER JOIN reviews r ON da.review_id = r.review_id
            WHERE r.app_id = :app_id
            AND r.review_date >= CURRENT_DATE - INTERVAL '{days} days'
            AND da.aspect IN ({aspects_placeholder})
            GROUP BY da.aspect
            ORDER BY total_mentions DESC
            """

            params = {'app_id': amazon_app_id}
            df = storage.db.execute_query(query, params)

            if df.empty:
                return {"aspects": [], "summary": {}, "app_name": "Amazon Shopping"}

            # Calculate percentages with safe division
            aspects_data = []
            for _, row in df.iterrows():
                total = row['total_mentions']
                if total is None or total == 0:
                    continue  # Skip if no mentions

                positive_count = row['positive_count'] or 0
                neutral_count = row['neutral_count'] or 0
                negative_count = row['negative_count'] or 0

                aspect_name = self.aspect_display_names.get(row['aspect'], row['aspect'].replace('_', ' ').title())

                aspects_data.append({
                    'aspect': aspect_name,
                    'total_mentions': int(total),
                    'positive_pct': round((positive_count / total) * 100, 1),
                    'neutral_pct': round((neutral_count / total) * 100, 1),
                    'negative_pct': round((negative_count / total) * 100, 1),
                    'avg_sentiment': round(float(row['avg_sentiment'] or 0), 3),
                    'avg_confidence': round(float(row['avg_confidence'] or 0), 3)
                })

            # Overall summary
            total_mentions = df['total_mentions'].sum()
            total_positive = df['positive_count'].sum()
            total_neutral = df['neutral_count'].sum()
            total_negative = df['negative_count'].sum()

            summary = {
                'total_mentions': int(total_mentions),
                'overall_positive_pct': round((total_positive / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'overall_neutral_pct': round((total_neutral / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'overall_negative_pct': round((total_negative / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'avg_sentiment': round(float(df['avg_sentiment'].fillna(0).mean()), 3) if not df.empty else 0,
                'aspect_count': len(df)
            }

            return {
                'aspects': aspects_data,
                'summary': summary,
                'period_days': days,
                'app_name': 'Amazon Shopping'
            }

        except Exception as e:
            self.logger.error(f"Error loading Amazon fixed ABSA data: {e}")
            return {"aspects": [], "summary": {}, "app_name": "Amazon Shopping"}

    def load_amazon_category_absa(self, category: str, days: int = 30) -> Dict[str, Any]:
        """
        Load Amazon deep-dive ABSA data by category.

        Args:
            category: Aspect category (Product, Service, Technical, etc.)
            days: Number of days to include

        Returns:
            Dictionary with category-specific ABSA data
        """
        try:
            amazon_app_id = "com.amazon.mShop.android.shopping"

            if category not in self.aspect_categories:
                return {"aspects": [], "summary": {}, "category": category}

            category_aspects = self.aspect_categories[category]
            aspects_placeholder = ','.join([f"'{aspect}'" for aspect in category_aspects])

            query = f"""
            SELECT 
                da.aspect,
                COUNT(*) as total_mentions,
                AVG(da.sentiment_score) as avg_sentiment,
                AVG(da.confidence_score) as avg_confidence,
                COUNT(*) FILTER (WHERE da.sentiment_score > 0.1) as positive_count,
                COUNT(*) FILTER (WHERE da.sentiment_score BETWEEN -0.1 AND 0.1) as neutral_count,
                COUNT(*) FILTER (WHERE da.sentiment_score < -0.1) as negative_count
            FROM deep_absa da
            INNER JOIN reviews r ON da.review_id = r.review_id
            WHERE r.app_id = :app_id
            AND r.review_date >= CURRENT_DATE - INTERVAL '{days} days'
            AND da.aspect IN ({aspects_placeholder})
            GROUP BY da.aspect
            ORDER BY total_mentions DESC
            """

            params = {'app_id': amazon_app_id}
            df = storage.db.execute_query(query, params)

            if df.empty:
                return {"aspects": [], "summary": {}, "category": category}

            # Calculate percentages with safe division
            aspects_data = []
            for _, row in df.iterrows():
                total = row['total_mentions']
                if total is None or total == 0:
                    continue  # Skip if no mentions

                positive_count = row['positive_count'] or 0
                neutral_count = row['neutral_count'] or 0
                negative_count = row['negative_count'] or 0

                aspect_name = self.aspect_display_names.get(row['aspect'], row['aspect'].replace('_', ' ').title())

                aspects_data.append({
                    'aspect': aspect_name,
                    'total_mentions': int(total),
                    'positive_pct': round((positive_count / total) * 100, 1),
                    'neutral_pct': round((neutral_count / total) * 100, 1),
                    'negative_pct': round((negative_count / total) * 100, 1),
                    'avg_sentiment': round(float(row['avg_sentiment'] or 0), 3),
                    'avg_confidence': round(float(row['avg_confidence'] or 0), 3)
                })

            # Category summary
            total_mentions = df['total_mentions'].sum()
            total_positive = df['positive_count'].sum()
            total_neutral = df['neutral_count'].sum()
            total_negative = df['negative_count'].sum()

            summary = {
                'total_mentions': int(total_mentions),
                'overall_positive_pct': round((total_positive / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'overall_neutral_pct': round((total_neutral / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'overall_negative_pct': round((total_negative / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'avg_sentiment': round(float(df['avg_sentiment'].mean()), 3),
                'aspect_count': len(df)
            }

            return {
                'aspects': aspects_data,
                'summary': summary,
                'period_days': days,
                'category': category,
                'app_name': 'Amazon Shopping'
            }

        except Exception as e:
            self.logger.error(f"Error loading Amazon category ABSA data: {e}")
            return {"aspects": [], "summary": {}, "category": category}

    def load_customizable_absa(self, app_id: str, selected_aspects: List[str],
                               days: int = 30) -> Dict[str, Any]:
        """
        Load customizable ABSA data for selected app and aspects.

        Args:
            app_id: Application identifier
            selected_aspects: List of aspect names to include
            days: Number of days to include

        Returns:
            Dictionary with customizable ABSA data
        """
        try:
            if not selected_aspects:
                return {"aspects": [], "summary": {}, "app_id": app_id}

            # Convert display names back to database names
            db_aspects = []
            for aspect in selected_aspects:
                for db_name, display_name in self.aspect_display_names.items():
                    if display_name == aspect:
                        db_aspects.append(db_name)
                        break
                else:
                    # If not found in display names, use as-is (fallback)
                    db_aspects.append(aspect.lower().replace(' ', '_'))

            aspects_placeholder = ','.join([f"'{aspect}'" for aspect in db_aspects])

            query = f"""
            SELECT 
                da.aspect,
                COUNT(*) as total_mentions,
                AVG(da.sentiment_score) as avg_sentiment,
                AVG(da.confidence_score) as avg_confidence,
                COUNT(*) FILTER (WHERE da.sentiment_score > 0.1) as positive_count,
                COUNT(*) FILTER (WHERE da.sentiment_score BETWEEN -0.1 AND 0.1) as neutral_count,
                COUNT(*) FILTER (WHERE da.sentiment_score < -0.1) as negative_count
            FROM deep_absa da
            INNER JOIN reviews r ON da.review_id = r.review_id
            WHERE r.app_id = :app_id
            AND r.review_date >= CURRENT_DATE - INTERVAL '{days} days'
            AND da.aspect IN ({aspects_placeholder})
            GROUP BY da.aspect
            ORDER BY total_mentions DESC
            """

            params = {'app_id': app_id}
            df = storage.db.execute_query(query, params)

            if df.empty:
                return {"aspects": [], "summary": {}, "app_id": app_id}

            # Calculate percentages with safe division
            aspects_data = []
            for _, row in df.iterrows():
                total = row['total_mentions']
                if total is None or total == 0:
                    continue  # Skip if no mentions

                positive_count = row['positive_count'] or 0
                neutral_count = row['neutral_count'] or 0
                negative_count = row['negative_count'] or 0

                aspect_name = self.aspect_display_names.get(row['aspect'], row['aspect'].replace('_', ' ').title())

                aspects_data.append({
                    'aspect': aspect_name,
                    'total_mentions': int(total),
                    'positive_pct': round((positive_count / total) * 100, 1),
                    'neutral_pct': round((neutral_count / total) * 100, 1),
                    'negative_pct': round((negative_count / total) * 100, 1),
                    'avg_sentiment': round(float(row['avg_sentiment'] or 0), 3),
                    'avg_confidence': round(float(row['avg_confidence'] or 0), 3)
                })

            # Overall summary with safe calculations
            if df.empty:
                total_mentions = 0
                total_positive = 0
                total_neutral = 0
                total_negative = 0
            else:
                total_mentions = df['total_mentions'].fillna(0).sum()
                total_positive = df['positive_count'].fillna(0).sum()
                total_neutral = df['neutral_count'].fillna(0).sum()
                total_negative = df['negative_count'].fillna(0).sum()

            summary = {
                'total_mentions': int(total_mentions),
                'overall_positive_pct': round((total_positive / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'overall_neutral_pct': round((total_neutral / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'overall_negative_pct': round((total_negative / total_mentions) * 100, 1) if total_mentions > 0 else 0,
                'avg_sentiment': round(float(df['avg_sentiment'].mean()), 3),
                'aspect_count': len(df)
            }

            # Get app name
            app_name_query = "SELECT app_name FROM apps WHERE app_id = :app_id"
            app_df = storage.db.execute_query(app_name_query, {'app_id': app_id})
            app_name = app_df.iloc[0]['app_name'] if not app_df.empty else app_id

            return {
                'aspects': aspects_data,
                'summary': summary,
                'period_days': days,
                'app_id': app_id,
                'app_name': app_name
            }

        except Exception as e:
            self.logger.error(f"Error loading customizable ABSA data: {e}")
            return {"aspects": [], "summary": {}, "app_id": app_id}

    def get_top_mentioned_aspects(self, limit: int = 5) -> List[str]:
        """
        Get top mentioned aspects across all apps for default selection.

        Args:
            limit: Number of top aspects to return

        Returns:
            List of top aspect display names
        """
        try:
            query = """
            SELECT 
                da.aspect,
                COUNT(*) as total_mentions
            FROM deep_absa da
            INNER JOIN reviews r ON da.review_id = r.review_id
            WHERE r.review_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY da.aspect
            ORDER BY total_mentions DESC
            LIMIT :limit
            """

            params = {'limit': limit}
            df = storage.db.execute_query(query, params)

            if df.empty:
                # Fallback to default aspects
                return ['Product Quality', 'Customer Service', 'App Performance', 'Shipping & Delivery', 'Pricing & Value']

            top_aspects = []
            for _, row in df.iterrows():
                display_name = self.aspect_display_names.get(row['aspect'], row['aspect'].replace('_', ' ').title())
                top_aspects.append(display_name)

            return top_aspects

        except Exception as e:
            self.logger.error(f"Error getting top mentioned aspects: {e}")
            return ['Product Quality', 'Customer Service', 'App Performance', 'Shipping & Delivery', 'Pricing & Value']

    def load_processing_status(self) -> Dict[str, Any]:
        """
        Load enhanced processing status with LLM performance metrics.
        Handles all None values safely and missing data gracefully.
        """
        # Safe default structure
        default_response = {
            'active_jobs': [],
            'unprocessed_counts': [],
            'queue_metrics': {'total_pending_absa': 0, 'total_pending_servqual': 0, 'ready_for_servqual': 0},
            'llm_performance': {'avg_processing_time': 0, 'success_rate': 0, 'total_jobs': 0, 'total_reviews_processed': 0, 'last_job_time': None},
            'last_updated': datetime.now().isoformat()
        }

        try:
            # Check basic database connectivity first
            health_check = storage.health_check()
            if health_check.get('overall') != 'healthy':
                self.logger.warning("Database not healthy - returning safe defaults")
                return default_response

            # Check if processing tables exist
            try:
                table_check = storage.db.execute_query("SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = 'processing_jobs'")
                if table_check.empty or table_check.iloc[0]['count'] == 0:
                    self.logger.warning("Processing tables not found - returning defaults")
                    return default_response
            except Exception as e:
                self.logger.warning(f"Could not check table existence: {e}")
                return default_response

            # Get simple review counts (most important for dashboard)
            try:
                basic_counts_query = """
                SELECT 
                    COUNT(*) as total_reviews,
                    COUNT(*) FILTER (WHERE COALESCE(absa_processed, FALSE) = FALSE) as absa_pending,
                    COUNT(*) FILTER (WHERE COALESCE(servqual_processed, FALSE) = FALSE) as servqual_pending
                FROM reviews 
                WHERE COALESCE(is_spam, FALSE) = FALSE
                """

                basic_df = storage.db.execute_query(basic_counts_query)

                if not basic_df.empty:
                    row = basic_df.iloc[0]
                    default_response['queue_metrics'] = {
                        'total_pending_absa': int(row.get('absa_pending', 0) or 0),
                        'total_pending_servqual': int(row.get('servqual_pending', 0) or 0),
                        'ready_for_servqual': 0  # Simple fallback
                    }

            except Exception as e:
                self.logger.warning(f"Could not get basic counts: {e}")

            # Try to get active jobs (optional)
            try:
                jobs_query = """
                SELECT 
                    job_id,
                    job_type,
                    app_id,
                    status,
                    COALESCE(current_phase, 'pending') as current_phase,
                    start_time
                FROM processing_jobs 
                WHERE status IN ('running', 'paused', 'pending')
                ORDER BY start_time DESC
                LIMIT 3
                """

                jobs_df = storage.db.execute_query(jobs_query)
                if not jobs_df.empty:
                    default_response['active_jobs'] = jobs_df.to_dict('records')

            except Exception as e:
                self.logger.debug(f"Could not get active jobs (non-critical): {e}")

            # Try to get app breakdown (optional)
            try:
                app_breakdown_query = """
                SELECT 
                    r.app_id,
                    a.app_name,
                    COUNT(*) as total_reviews,
                    COUNT(*) FILTER (WHERE COALESCE(r.absa_processed, FALSE) = FALSE) as absa_pending
                FROM reviews r
                INNER JOIN apps a ON r.app_id = a.app_id
                WHERE COALESCE(r.is_spam, FALSE) = FALSE
                GROUP BY r.app_id, a.app_name
                ORDER BY absa_pending DESC
                LIMIT 5
                """

                app_df = storage.db.execute_query(app_breakdown_query)
                if not app_df.empty:
                    default_response['unprocessed_counts'] = app_df.to_dict('records')

            except Exception as e:
                self.logger.debug(f"Could not get app breakdown (non-critical): {e}")

            return default_response

        except Exception as e:
            self.logger.error(f"Error loading processing status: {e}")
            return default_response

    def get_app_list_with_data(self) -> List[Dict[str, str]]:
        """
        Get list of apps with ABSA data for dashboard filtering.

        Returns:
            List of app dictionaries with processing status
        """
        try:
            query = """
            SELECT 
                a.app_id,
                a.app_name,
                a.category,
                COUNT(r.review_id) as total_reviews,
                COUNT(*) FILTER (WHERE r.absa_processed) as absa_processed_count,
                MAX(r.review_date) as latest_review_date
            FROM apps a
            INNER JOIN reviews r ON a.app_id = r.app_id
            WHERE NOT r.is_spam
            GROUP BY a.app_id, a.app_name, a.category
            HAVING COUNT(*) FILTER (WHERE r.absa_processed) > 0
            ORDER BY absa_processed_count DESC
            """

            df = storage.db.execute_query(query)
            return df.to_dict('records') if not df.empty else []

        except Exception as e:
            self.logger.error(f"Error loading app list with data: {e}")
            return []

    def get_aspect_categories_for_selection(self) -> Dict[str, List[str]]:
        """
        Get aspect categories with display names for UI selection.

        Returns:
            Dictionary of categories with display names
        """
        categories_with_display = {}
        for category, aspects in self.aspect_categories.items():
            display_aspects = [self.aspect_display_names.get(aspect, aspect.replace('_', ' ').title())
                             for aspect in aspects]
            categories_with_display[category] = display_aspects

        return categories_with_display


# Global instance for dashboard use
dashboard_data_loader = DashboardDataLoader()