"""
SERVQUAL Implementation Script for ABSA Pipeline.
Applies all necessary database schema changes and initial setup for SERVQUAL integration.
Run this script once to upgrade your existing ABSA pipeline with SERVQUAL capabilities.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.storage import storage
from src.utils.config import logger


class ServqualImplementation:
    """Handles SERVQUAL implementation and database migration."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.servqual.implementation")
        self.db = storage.db

    def run_implementation(self):
        """Run complete SERVQUAL implementation."""
        self.logger.info("Starting SERVQUAL implementation...")

        try:
            # Step 1: Add SERVQUAL column to aspects table
            self.add_servqual_column()

            # Step 2: Create SERVQUAL scores table
            self.create_servqual_tables()

            # Step 3: Map existing aspects to SERVQUAL dimensions
            self.map_aspects_to_servqual()

            # Step 4: Create indexes for performance
            self.create_indexes()

            # Step 5: Create views for easy querying
            self.create_views()

            # Step 6: Verify implementation
            self.verify_implementation()

            self.logger.info("SERVQUAL implementation completed successfully!")
            self.logger.info("You can now use the SERVQUAL dashboard section")

            return True

        except Exception as e:
            self.logger.error(f"SERVQUAL implementation failed: {e}")
            return False

    def add_servqual_column(self):
        """Add SERVQUAL dimension column to aspects table."""
        self.logger.info("Adding SERVQUAL dimension column to aspects table...")

        query = """
        ALTER TABLE aspects ADD COLUMN IF NOT EXISTS servqual_dimension VARCHAR(50);
        """

        try:
            self.db.execute_non_query(query)
            self.logger.info("SERVQUAL column added to aspects table")
        except Exception as e:
            self.logger.error(f"Error adding SERVQUAL column: {e}")
            raise

    def create_servqual_tables(self):
        """Create SERVQUAL scores table."""
        self.logger.info("Creating SERVQUAL scores table...")

        query = """
        CREATE TABLE IF NOT EXISTS servqual_scores (
            score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            app_id VARCHAR(255) NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
            dimension VARCHAR(50) NOT NULL,
            sentiment_score DECIMAL(4,3),
            quality_score INTEGER CHECK (quality_score >= 1 AND quality_score <= 5),
            review_count INTEGER DEFAULT 0,
            date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(app_id, dimension, date)
        );
        """

        try:
            self.db.execute_non_query(query)
            self.logger.info("SERVQUAL scores table created")
        except Exception as e:
            self.logger.error(f"Error creating SERVQUAL tables: {e}")
            raise

    def map_aspects_to_servqual(self):
        """Map existing ecommerce aspects to SERVQUAL dimensions."""
        self.logger.info("Mapping aspects to SERVQUAL dimensions...")

        mappings = [
            # Reliability dimension
            (
                "UPDATE aspects SET servqual_dimension = 'reliability' WHERE aspect_name IN ('product_quality', 'product_description', 'app_performance');"),

            # Assurance dimension
            (
                "UPDATE aspects SET servqual_dimension = 'assurance' WHERE aspect_name IN ('customer_service', 'payment_security', 'seller_trust');"),

            # Tangibles dimension
            (
                "UPDATE aspects SET servqual_dimension = 'tangibles' WHERE aspect_name IN ('search_navigation', 'app_usability', 'product_variety');"),

            # Empathy dimension
            ("UPDATE aspects SET servqual_dimension = 'empathy' WHERE aspect_name IN ('return_refund');"),

            # Responsiveness dimension
            (
                "UPDATE aspects SET servqual_dimension = 'responsiveness' WHERE aspect_name IN ('shipping_delivery', 'shipping_cost', 'order_tracking');")
        ]

        try:
            for query in mappings:
                rows_updated = self.db.execute_non_query(query)
                self.logger.info(f"Updated {rows_updated} aspects")

            self.logger.info("Aspect-to-SERVQUAL mapping completed")
        except Exception as e:
            self.logger.error(f"Error mapping aspects: {e}")
            raise

    def create_indexes(self):
        """Create indexes for SERVQUAL performance."""
        self.logger.info("Creating performance indexes...")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_servqual_scores_app_date ON servqual_scores(app_id, date);",
            "CREATE INDEX IF NOT EXISTS idx_servqual_scores_dimension ON servqual_scores(dimension);",
            "CREATE INDEX IF NOT EXISTS idx_aspects_servqual_dimension ON aspects(servqual_dimension);"
        ]

        try:
            for query in indexes:
                self.db.execute_non_query(query)

            self.logger.info("Performance indexes created")
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
            raise

    def create_views(self):
        """Create SERVQUAL analysis views."""
        self.logger.info("Creating SERVQUAL analysis views...")

        views = [
            # SERVQUAL dimension summary view
            """
            CREATE OR REPLACE VIEW servqual_dimension_summary AS
            SELECT
                app_id,
                dimension,
                DATE_TRUNC('week', date) as week,
                AVG(quality_score) as avg_quality_score,
                AVG(sentiment_score) as avg_sentiment_score,
                SUM(review_count) as total_reviews,
                MAX(date) as latest_date
            FROM servqual_scores
            GROUP BY app_id, dimension, DATE_TRUNC('week', date)
            ORDER BY app_id, dimension, week DESC;
            """,

            # App SERVQUAL profile view
            """
            CREATE OR REPLACE VIEW app_servqual_profile AS
            SELECT
                app_id,
                MAX(CASE WHEN dimension = 'reliability' THEN quality_score END) as reliability_score,
                MAX(CASE WHEN dimension = 'assurance' THEN quality_score END) as assurance_score,
                MAX(CASE WHEN dimension = 'tangibles' THEN quality_score END) as tangibles_score,
                MAX(CASE WHEN dimension = 'empathy' THEN quality_score END) as empathy_score,
                MAX(CASE WHEN dimension = 'responsiveness' THEN quality_score END) as responsiveness_score,
                date
            FROM servqual_scores
            GROUP BY app_id, date
            ORDER BY app_id, date DESC;
            """
        ]

        try:
            for query in views:
                self.db.execute_non_query(query)

            self.logger.info("SERVQUAL analysis views created")
        except Exception as e:
            self.logger.error(f"Error creating views: {e}")
            raise

    def verify_implementation(self):
        """Verify SERVQUAL implementation."""
        self.logger.info("Verifying SERVQUAL implementation...")

        try:
            # Check if servqual_dimension column exists
            check_column_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'aspects' AND column_name = 'servqual_dimension';
            """

            result = self.db.execute_query(check_column_query)
            if result.empty:
                raise Exception("SERVQUAL dimension column not found in aspects table")

            # Check if servqual_scores table exists
            check_table_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'servqual_scores';
            """

            result = self.db.execute_query(check_table_query)
            if result.empty:
                raise Exception("SERVQUAL scores table not found")

            # Check aspect mappings
            mapping_query = """
            SELECT servqual_dimension, COUNT(*) as aspect_count
            FROM aspects 
            WHERE servqual_dimension IS NOT NULL
            GROUP BY servqual_dimension;
            """

            mappings_df = self.db.execute_query(mapping_query)
            if mappings_df.empty:
                raise Exception("No aspect-to-SERVQUAL mappings found")

            self.logger.info("SERVQUAL Implementation Verification Results:")
            self.logger.info(f"SERVQUAL dimension column: EXISTS")
            self.logger.info(f"SERVQUAL scores table: EXISTS")
            self.logger.info(f"Aspect mappings: {len(mappings_df)} dimensions mapped")

            for _, row in mappings_df.iterrows():
                self.logger.info(f"   - {row['servqual_dimension']}: {row['aspect_count']} aspects")

            self.logger.info("SERVQUAL implementation verification completed successfully")

        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            raise

    def show_next_steps(self):
        """Show next steps after implementation."""
        print("\n" + "=" * 60)
        print("SERVQUAL IMPLEMENTATION COMPLETE!")
        print("=" * 60)
        print()
        print("What's New:")
        print("* SERVQUAL dimension scoring (1-5 scale)")
        print("* Amazon-focused service quality analysis")
        print("* Competitive app comparisons")
        print("* Business intelligence dashboards")
        print()
        print("Next Steps:")
        print("1. Run your dashboard: streamlit run dashboard_app.py")
        print("2. Navigate to 'SERVQUAL' section")
        print("3. Scrape some review data to see SERVQUAL analysis")
        print("4. Explore Amazon Focus and App Comparisons tabs")
        print()
        print("Phase 2 Development:")
        print("* Real-time SERVQUAL processing")
        print("* Advanced sentiment-to-quality algorithms")
        print("* Automated SERVQUAL reporting")
        print("* Machine learning enhancements")
        print()
        print("Framework Details:")
        print("* Reliability: Platform consistency, order accuracy")
        print("* Assurance: Security, trust, competence")
        print("* Tangibles: UI/UX design, packaging")
        print("* Empathy: Customer care, returns handling")
        print("* Responsiveness: Speed, communication, tracking")
        print()


def main():
    """Main implementation function."""
    print("ABSA Pipeline - SERVQUAL Implementation")
    print("==========================================")
    print()

    # Check if user wants to proceed
    response = input("This will modify your database schema. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Implementation cancelled.")
        return

    # Run implementation
    implementation = ServqualImplementation()

    success = implementation.run_implementation()

    if success:
        implementation.show_next_steps()
    else:
        print("\nImplementation failed. Check logs for details.")
        print("Try running the script again or check database connectivity.")


if __name__ == "__main__":
    main()