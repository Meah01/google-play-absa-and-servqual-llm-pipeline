"""
Enhanced Streamlit Dashboard for ABSA Sentiment Pipeline with LLM Integration.
Displays enhanced ABSA analysis with Amazon-focused and customizable tables.
Includes real-time processing monitoring and LLM performance metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.storage import storage
from src.data.scraper import scrape_app_reviews, scrape_multiple_apps
from dashboard.servqual_components import servqual_dashboard
from dashboard.data_loader import dashboard_data_loader

# Import heatmap utilities with graceful fallback
try:
    from dashboard.heatmap_utils import apply_heatmap_styling
except ImportError:
    # Graceful fallback if heatmap utils not available
    apply_heatmap_styling = None

# Page configuration
st.set_page_config(
    page_title="ABSA Sentiment Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling with proper contrast
st.markdown("""
<style>
.metric-card {
    background-color: #2c3e50;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 4px solid #3498db;
}

.success-metric {
    background-color: #27ae60;
    color: white;
    border-left: 4px solid #2ecc71;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.warning-metric {
    background-color: #e67e22;
    color: white;
    border-left: 4px solid #f39c12;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.info-metric {
    background-color: #3498db;
    color: white;
    border-left: 4px solid #2980b9;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.processing-card {
    background-color: #34495e;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #e74c3c;
    margin: 0.5rem 0;
}

.app-info-card {
    background-color: #2c3e50;
    color: white;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #e74c3c;
}

.sidebar-info {
    background-color: #34495e;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #3498db;
}

.sidebar-info strong {
    color: #f39c12;
}

.absa-table {
    background-color: #ecf0f1;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border: 1px solid #bdc3c7;
}

.amazon-section {
    background-color: #fff8e1;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #ff9800;
}

.custom-section {
    background-color: #f3e5f5;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #9c27b0;
}

.progress-container {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)


# @st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dashboard_data():
    """Load and cache dashboard data."""
    try:
        # Get all apps
        apps = storage.apps.get_all_apps()
        apps_df = pd.DataFrame(apps) if apps else pd.DataFrame()

        # Get all reviews for each app
        all_reviews = []
        for app in apps:
            reviews = storage.reviews.get_reviews_for_processing(app['app_id'], limit=1000)
            all_reviews.extend(reviews)

        reviews_df = pd.DataFrame(all_reviews) if all_reviews else pd.DataFrame()

        # System health
        health = storage.health_check()

        return apps_df, reviews_df, health

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), {"overall": "unhealthy"}


def render_processing_status():
    """Render enhanced processing status in sidebar."""
    st.sidebar.markdown("### 🔄 Processing Status")

    try:
        # Load processing status with error handling
        status_data = dashboard_data_loader.load_processing_status()

        # Active jobs section
        active_jobs = status_data.get('active_jobs', [])

        if active_jobs:
            st.sidebar.markdown("**Active Jobs:**")

            for job in active_jobs:
                job_type = job.get('job_type', 'unknown')
                progress = job.get('progress_percentage', 0)
                processed = job.get('reviews_processed', 0)
                total = job.get('total_reviews', 0)
                phase = job.get('current_phase', 'pending')

                # Ensure all values are safe
                processed = processed if processed is not None else 0
                total = total if total is not None else 0
                progress = progress if progress is not None else 0

                # Simple text display without any custom CSS
                st.sidebar.write(f"🔄 **{job_type.replace('_', ' ').title()}**")
                st.sidebar.write(f"Phase: {phase.title()}")
                st.sidebar.write(f"Progress: {processed}/{total} reviews")

                # Progress bar
                if total > 0:
                    st.sidebar.progress(min(processed / total, 1.0))
                else:
                    st.sidebar.progress(0)

                st.sidebar.write("---")  # Separator
        else:
            st.sidebar.success("✅ No active processing jobs")

        # Queue metrics
        queue_metrics = status_data.get('queue_metrics', {})

        col1, col2 = st.sidebar.columns(2)
        with col1:
            absa_queue = queue_metrics.get('total_pending_absa', 0)
            absa_queue = absa_queue if absa_queue is not None else 0
            st.metric("ABSA Queue", f"{absa_queue:,}")
        with col2:
            servqual_queue = queue_metrics.get('total_pending_servqual', 0)
            servqual_queue = servqual_queue if servqual_queue is not None else 0
            st.metric("SERVQUAL Queue", f"{servqual_queue:,}")

        # LLM Performance metrics
        st.sidebar.markdown("**🤖 LLM Performance:**")
        llm_perf = status_data.get('llm_performance', {})

        col1, col2 = st.sidebar.columns(2)
        with col1:
            processing_time = llm_perf.get('avg_processing_time', 0)
            processing_time = processing_time if processing_time is not None else 0
            st.metric("Avg Speed", f"{processing_time:.1f}s/review")
        with col2:
            success_rate = llm_perf.get('success_rate', 0)
            success_rate = success_rate if success_rate is not None else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        total_jobs = llm_perf.get('total_jobs', 0)
        total_jobs = total_jobs if total_jobs is not None else 0
        st.sidebar.metric("Total Jobs", f"{total_jobs:,}")

    except Exception as e:
        st.sidebar.error(f"Error loading processing status: {str(e)}")
        st.sidebar.info("ABSA Queue: 0")
        st.sidebar.info("SERVQUAL Queue: 0")
        st.sidebar.info("Processing metrics unavailable")


def render_sidebar():
    """Render enhanced sidebar with processing status."""
    st.sidebar.title("Dashboard Controls")

    # Navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "ABSA Analysis", "SERVQUAL", "System Health"],
        key="page_selection"
    )

    # App selection for non-ABSA pages
    apps_df, _, _ = load_dashboard_data()

    if page not in ["ABSA Analysis"]:  # ABSA Analysis has its own app selection
        if not apps_df.empty:
            app_options = ["All Apps"] + apps_df['app_name'].tolist()
            selected_app = st.sidebar.selectbox("Select App", app_options)

            # Show selected app info
            if selected_app != "All Apps":
                app_info = apps_df[apps_df['app_name'] == selected_app].iloc[0]

                st.sidebar.markdown(f"""
                <div class="metric-card">
                <strong>Name:</strong> {app_info['app_name']}<br>
                <strong>Developer:</strong> {app_info['developer']}<br>
                <strong>Rating:</strong> {app_info['rating']:.1f}/5.0<br>
                <strong>Category:</strong> {app_info['category']}<br>
                <strong>Reviews:</strong> {app_info.get('total_reviews', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
        else:
            selected_app = "All Apps"
    else:
        selected_app = "All Apps"  # Default for ABSA Analysis

    # Processing Status Section
    render_processing_status()

    # Data scraping controls
    st.sidebar.markdown("### 📥 Data Collection")

    if st.sidebar.button("Quick Scrape Amazon", type="primary"):
        with st.spinner("Scraping Amazon Shopping reviews..."):
            result = scrape_app_reviews("com.amazon.mShop.android.shopping", count=10)
            if result['success']:
                st.sidebar.success(f"✅ Scraped {result['statistics']['stored']} reviews!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.error("❌ Scraping failed")

    if st.sidebar.button("Scrape All Apps"):
        with st.spinner("Scraping multiple ecommerce apps..."):
            app_ids = [
                "com.amazon.mShop.android.shopping",
                "com.einnovation.temu",
                "com.zzkko",
                "com.ebay.mobile",
                "com.etsy.android"
            ]
            results = scrape_multiple_apps(app_ids)
            total_reviews = sum(r['statistics']['stored'] for r in results)
            st.sidebar.success(f"✅ Scraped {total_reviews} total reviews!")
            st.cache_data.clear()
            st.rerun()

    # Analysis controls
    st.sidebar.markdown("### 🔬 Analysis")

    if st.sidebar.button("Run Sequential Processing", type="secondary"):
        try:
            from src.pipeline.batch import BatchProcessor
            batch_processor = BatchProcessor()
            unprocessed_reviews = batch_processor.get_unprocessed_reviews()

            if len(unprocessed_reviews) == 0:
                st.sidebar.warning("⚠️ No unprocessed reviews found")
                return page, selected_app

            st.sidebar.info(f"📊 Found {len(unprocessed_reviews)} unprocessed reviews")

            # Progress containers
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()

            with st.spinner("Running sequential ABSA + SERVQUAL processing..."):
                try:
                    status_text.text("Starting sequential processing...")
                    result = batch_processor.run_sequential_processing()

                    progress_bar.progress(100)

                    if result and result.success:
                        status_text.empty()
                        st.sidebar.success(f"✅ Sequential Processing Complete!")
                        st.sidebar.info(f"📊 Processed: {result.reviews_processed} reviews")
                        st.sidebar.info(f"⏱️ Time: {result.processing_time_seconds:.1f} seconds")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        error_msg = result.error_message if result else "Unknown error"
                        st.sidebar.error(f"❌ Processing failed: {error_msg}")

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.sidebar.error(f"❌ Error: {str(e)}")

        except Exception as e:
            st.sidebar.error(f"❌ Setup Error: {str(e)}")

    # Refresh controls
    st.sidebar.markdown("### 🔄 Refresh")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Auto-refresh during processing
    status_data = dashboard_data_loader.load_processing_status()
    active_jobs = status_data.get('active_jobs', [])

   # if active_jobs:
    #    st.sidebar.markdown("**Auto-refreshing every 1 minute during processing**")
    #   time.sleep(60)
    #  st.rerun()

    return page, selected_app


def render_absa_percentage_table(data: Dict, title: str, help_text: str = None):
    """Render ABSA percentage table with improved intuitive heatmap colors."""
    if not data.get('aspects'):
        st.info("No data available for this analysis.")
        return

    st.markdown(f"#### {title}")
    if help_text:
        st.info(help_text)

    # Create percentage table
    table_data = []
    for aspect in data['aspects']:
        table_data.append({
            'Aspect': aspect['aspect'],
            'Positive (%)': aspect['positive_pct'],
            'Neutral (%)': aspect['neutral_pct'],
            'Negative (%)': aspect['negative_pct'],
            'Total Mentions': aspect['total_mentions']
        })

    df = pd.DataFrame(table_data)

    # Apply improved heatmap styling if available
    if apply_heatmap_styling is not None:
        try:
            # Define sentiment columns for styling
            sentiment_columns = {
                'positive': 'Positive (%)',
                'neutral': 'Neutral (%)',
                'negative': 'Negative (%)'
            }

            # Apply improved heatmap styling
            styled_df = apply_heatmap_styling(df, sentiment_columns)

            # Additional formatting for non-sentiment columns
            styled_df = styled_df.format({
                'Positive (%)': '{:.1f}',
                'Neutral (%)': '{:.1f}',
                'Negative (%)': '{:.1f}',
                'Total Mentions': '{:,}'
            })

            st.markdown("""
            **Color Logic:** 
            🟢 **Positive**: Red (0%) → Green (100%) - *More positive = greener*  
            🟡 **Neutral**: Light → Deep yellow - *Higher % = deeper yellow*  
            🔴 **Negative**: Green (0%) → Red (100%) - *More negative = redder*
            """)

        except Exception as e:
            st.warning(f"Using fallback styling - error with heatmap utilities: {str(e)}")
            styled_df = df.style.format({
                'Positive (%)': '{:.1f}%',
                'Neutral (%)': '{:.1f}%',
                'Negative (%)': '{:.1f}%',
                'Total Mentions': '{:,}'
            }).background_gradient(subset=['Positive (%)', 'Neutral (%)', 'Negative (%)'], cmap='RdYlGn_r')
    else:
        # Fallback to basic styling if heatmap utils not available
        st.warning("Using fallback styling - heatmap utilities not found")
        styled_df = df.style.format({
            'Positive (%)': '{:.1f}%',
            'Neutral (%)': '{:.1f}%',
            'Negative (%)': '{:.1f}%',
            'Total Mentions': '{:,}'
        }).background_gradient(subset=['Positive (%)', 'Neutral (%)', 'Negative (%)'], cmap='RdYlGn_r')

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Summary metrics
    summary = data.get('summary', {})
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Mentions", f"{summary.get('total_mentions', 0):,}")
        with col2:
            st.metric("Overall Positive", f"{summary.get('overall_positive_pct', 0):.1f}%")
        with col3:
            st.metric("Overall Neutral", f"{summary.get('overall_neutral_pct', 0):.1f}%")
        with col4:
            st.metric("Overall Negative", f"{summary.get('overall_negative_pct', 0):.1f}%")


def render_absa_analysis():
    """Render enhanced ABSA Analysis section."""
    st.markdown("##ABSA Analysis")
    st.markdown("**Enhanced aspect-based sentiment analysis with business intelligence focus**")

    # Time period selector
    col1, col2 = st.columns([3, 1])
    with col2:
        days = st.selectbox(
            "Analysis Period",
            [7, 14, 30, 60, 90],
            index=2,
            key="absa_period"
        )

    # Debug section
    try:
        # Check database connectivity and data availability
        test_query = "SELECT COUNT(*) as count FROM deep_absa"
        test_result = storage.db.execute_query(test_query)
        absa_count = test_result.iloc[0]['count'] if not test_result.empty else 0

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Debug: Found {absa_count} ABSA records")
        with col2:
            if absa_count == 0:
                st.warning("No ABSA data found. Run processing first!")
            else:
                st.success("ABSA data available")

    except Exception as e:
        st.error(f"Database connectivity issue: {str(e)}")
        st.stop()

    # Amazon Fixed Table Section
    st.markdown("### 🛒 Amazon Shopping - Business Critical Aspects")

    try:
        amazon_fixed_data = dashboard_data_loader.load_amazon_fixed_absa(days=days)
        render_absa_percentage_table(
            amazon_fixed_data,
            "Amazon Key Performance Indicators",
            f"Top business-critical aspects for Amazon Shopping (last {days} days)"
        )
    except Exception as e:
        st.error(f"Error loading Amazon fixed data: {str(e)}")
        st.info("This might be normal if you haven't run ABSA processing yet.")

    # Amazon Deep-Dive Table Section
    st.markdown("### 🔍 Amazon Deep-Dive by Category")

    col1, col2 = st.columns([2, 1])
    with col2:
        categories = list(dashboard_data_loader.aspect_categories.keys())
        selected_category = st.selectbox(
            "Select Category",
            categories,
            key="amazon_category"
        )

    try:
        amazon_category_data = dashboard_data_loader.load_amazon_category_absa(
            category=selected_category, days=days
        )
        render_absa_percentage_table(
            amazon_category_data,
            f"Amazon {selected_category} Analysis",
            f"Detailed analysis of {selected_category} aspects for Amazon Shopping"
        )
    except Exception as e:
        st.error(f"Error loading Amazon category data: {str(e)}")

    # Customizable Table Section
    st.markdown("### ⚙️ Customizable Analysis")

    # App and aspect selection
    col1, col2 = st.columns(2)

    with col1:
        try:
            apps_with_data = dashboard_data_loader.get_app_list_with_data()
            if apps_with_data:
                app_options = [(app['app_name'], app['app_id']) for app in apps_with_data]
                selected_app_display = st.selectbox(
                    "Select App",
                    [option[0] for option in app_options],
                    key="custom_app"
                )
                selected_app_id = next(app[1] for app in app_options if app[0] == selected_app_display)
            else:
                st.warning("No apps with ABSA data available")
                selected_app_id = None
        except Exception as e:
            st.error(f"Error loading apps: {str(e)}")
            selected_app_id = None

    with col2:
        if selected_app_id:
            # Get aspect categories for selection
            aspect_categories = dashboard_data_loader.get_aspect_categories_for_selection()

            # Get top mentioned aspects as default
            top_aspects = dashboard_data_loader.get_top_mentioned_aspects(5)

            st.markdown("**Select Aspects by Category:**")
            selected_aspects = []

            for category, aspects in aspect_categories.items():
                st.markdown(f"**{category}:**")
                for aspect in aspects:
                    default_checked = aspect in top_aspects
                    if st.checkbox(aspect, value=default_checked, key=f"aspect_{aspect}"):
                        selected_aspects.append(aspect)

    # Load and display customizable data
    if selected_app_id and selected_aspects:
        try:
            custom_data = dashboard_data_loader.load_customizable_absa(
                app_id=selected_app_id,
                selected_aspects=selected_aspects,
                days=days
            )
            render_absa_percentage_table(
                custom_data,
                f"{selected_app_display} - Selected Aspects",
                f"Custom analysis for {selected_app_display} with selected aspects"
            )
        except Exception as e:
            st.error(f"Error loading custom data: {str(e)}")
    elif selected_app_id:
        st.info("Please select at least one aspect to analyze.")


# Replace the render_overview_metrics function in dashboard_app.py with this safer version:

def render_overview_metrics(apps_df, reviews_df):
    """Render overview metrics cards with comprehensive error handling."""
    st.markdown("### 📊 System Overview")

    # Database status check
    try:
        health_check = storage.health_check()
        db_status = health_check.get('overall', 'unknown')

        if db_status == 'healthy':
            st.success("✅ Database: Connected")
        else:
            st.error("❌ Database: Connection issues")

    except Exception as e:
        st.error(f"❌ Database: Error checking status ({str(e)[:50]})")

    # Get basic metrics with safe queries
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        try:
            # Count apps safely
            apps_count = len(apps_df) if not apps_df.empty else 0
            st.metric(
                label="📱 Total Apps",
                value=f"{apps_count}",
                help="Apps currently tracked in the system"
            )
        except Exception as e:
            st.metric(label="📱 Total Apps", value="Error", help=f"Error: {e}")

    with col2:
        try:
            # Get total reviews with direct query (safer than dataframe)
            review_query = "SELECT COUNT(*) as total FROM reviews WHERE NOT COALESCE(is_spam, FALSE)"
            review_df = storage.db.execute_query(review_query)
            total_reviews = int(review_df.iloc[0]['total']) if not review_df.empty else 0

            st.metric(
                label="💬 Total Reviews",
                value=f"{total_reviews:,}",
                help="Total reviews scraped (excluding spam)"
            )
        except Exception as e:
            st.metric(label="💬 Total Reviews", value="Error", help=f"Database error: {str(e)[:50]}")

    with col3:
        try:
            # Get processing status
            processed_query = """
            SELECT 
                COUNT(*) FILTER (WHERE COALESCE(absa_processed, FALSE) = TRUE) as absa_done,
                COUNT(*) FILTER (WHERE COALESCE(servqual_processed, FALSE) = TRUE) as servqual_done,
                COUNT(*) as total
            FROM reviews WHERE NOT COALESCE(is_spam, FALSE)
            """
            proc_df = storage.db.execute_query(processed_query)

            if not proc_df.empty:
                absa_done = int(proc_df.iloc[0]['absa_done'] or 0)
                total = int(proc_df.iloc[0]['total'] or 0)
                percentage = (absa_done / total * 100) if total > 0 else 0

                st.metric(
                    label="🧠 ABSA Processed",
                    value=f"{absa_done:,}",
                    delta=f"{percentage:.1f}% complete",
                    help="Reviews processed through ABSA analysis"
                )
            else:
                st.metric(label="🧠 ABSA Processed", value="0", help="No processing data available")

        except Exception as e:
            st.metric(label="🧠 ABSA Processed", value="Error", help=f"Query error: {str(e)[:50]}")

    with col4:
        try:
            # Get SERVQUAL status - re-run query for safety
            servqual_query = """
            SELECT 
                COUNT(*) FILTER (WHERE COALESCE(servqual_processed, FALSE) = TRUE) as servqual_done,
                COUNT(*) as total
            FROM reviews WHERE NOT COALESCE(is_spam, FALSE)
            """
            servqual_df = storage.db.execute_query(servqual_query)

            if not servqual_df.empty:
                servqual_done = int(servqual_df.iloc[0]['servqual_done'] or 0)
                total = int(servqual_df.iloc[0]['total'] or 0)
                percentage = (servqual_done / total * 100) if total > 0 else 0

                st.metric(
                    label="🤖 SERVQUAL LLM",
                    value=f"{servqual_done:,}",
                    delta=f"{percentage:.1f}% complete",
                    help="Reviews processed through LLM SERVQUAL analysis"
                )
            else:
                st.metric(label="🤖 SERVQUAL LLM", value="0", help="No LLM processing data available")

        except Exception as e:
            st.metric(label="🤖 SERVQUAL LLM", value="Error", help=f"Query error: {str(e)[:50]}")

    # Show data availability status
    st.markdown("#### 📋 Data Availability Status")

    try:
        # Check table status
        status_checks = {
            "Reviews Table": "SELECT COUNT(*) FROM reviews",
            "Apps Table": "SELECT COUNT(*) FROM apps",
            "ABSA Results": "SELECT COUNT(*) FROM deep_absa",
            "SERVQUAL Results": "SELECT COUNT(*) FROM servqual_scores",
            "Processing Jobs": "SELECT COUNT(*) FROM processing_jobs"
        }

        col1, col2, col3 = st.columns(3)

        for i, (check_name, query) in enumerate(status_checks.items()):
            try:
                result_df = storage.db.execute_query(query)
                count = int(result_df.iloc[0]['count']) if not result_df.empty else 0

                # Determine which column to use
                col = [col1, col2, col3][i % 3]

                with col:
                    if count > 0:
                        st.success(f"✅ {check_name}: {count:,} records")
                    else:
                        st.warning(f"⚠️ {check_name}: No data")

            except Exception as e:
                with [col1, col2, col3][i % 3]:
                    st.error(f"❌ {check_name}: Error")

    except Exception as e:
        st.error(f"Unable to check data status: {e}")

    # Recent activity check
    try:
        recent_query = """
        SELECT 
            MAX(review_date) as latest_review,
            MAX(created_at) as latest_processing
        FROM reviews 
        WHERE NOT COALESCE(is_spam, FALSE)
        """

        recent_df = storage.db.execute_query(recent_query)

        if not recent_df.empty and recent_df.iloc[0]['latest_review'] is not None:
            latest_review = recent_df.iloc[0]['latest_review']
            latest_processing = recent_df.iloc[0]['latest_processing']

            st.markdown("#### 📅 Recent Activity")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"**Latest Review:** {latest_review}")
            with col2:
                if latest_processing:
                    st.info(f"**Latest Processing:** {latest_processing}")
                else:
                    st.warning("**Latest Processing:** No processing data")
        else:
            st.warning("📅 No recent activity data available")

    except Exception as e:
        st.warning(f"📅 Unable to check recent activity: {str(e)[:100]}")


def render_app_overview(apps_df):
    """Render enhanced apps overview with detailed metrics table."""
    st.markdown("### 📱 Apps Overview & Metrics")

    if apps_df.empty:
        st.info("No apps found. Use the sidebar to scrape some app data!")
        return

    # Get detailed metrics for each app
    try:
        detailed_metrics = []

        for _, app in apps_df.iterrows():
            app_id = app['app_id']

            # Get review counts and processing status
            metrics_query = """
            SELECT 
                COUNT(*) as total_reviews,
                COUNT(*) FILTER (WHERE absa_processed = TRUE) as absa_processed,
                COUNT(*) FILTER (WHERE servqual_processed = TRUE) as servqual_processed,
                COUNT(*) FILTER (WHERE is_spam = TRUE) as spam_reviews,
                AVG(rating) as avg_rating,
                MIN(review_date) as earliest_review,
                MAX(review_date) as latest_review,
                COUNT(*) FILTER (WHERE review_date >= CURRENT_DATE - INTERVAL '7 days') as recent_reviews
            FROM reviews 
            WHERE app_id = :app_id
            """

            result = storage.db.execute_query(metrics_query, {'app_id': app_id})

            if not result.empty:
                row = result.iloc[0]
                detailed_metrics.append({
                    'App Name': app['app_name'],
                    'Developer': app['developer'],
                    'Category': app['category'],
                    'Store Rating': f"{app['rating']:.1f}/5.0",
                    'Total Reviews': f"{int(row['total_reviews']):,}",
                    'ABSA Processed': f"{int(row['absa_processed']):,}",
                    'SERVQUAL Processed': f"{int(row['servqual_processed']):,}",
                    'Spam Reviews': f"{int(row['spam_reviews']):,}",
                    'Avg Rating': f"{float(row['avg_rating']):.1f}" if row['avg_rating'] else "N/A",
                    'Recent (7d)': f"{int(row['recent_reviews']):,}",
                    'Latest Review': str(row['latest_review'])[:10] if row['latest_review'] else "N/A"
                })
            else:
                # No reviews for this app
                detailed_metrics.append({
                    'App Name': app['app_name'],
                    'Developer': app['developer'],
                    'Category': app['category'],
                    'Store Rating': f"{app['rating']:.1f}/5.0",
                    'Total Reviews': "0",
                    'ABSA Processed': "0",
                    'SERVQUAL Processed': "0",
                    'Spam Reviews': "0",
                    'Avg Rating': "N/A",
                    'Recent (7d)': "0",
                    'Latest Review': "N/A"
                })

        # Create detailed metrics DataFrame
        metrics_df = pd.DataFrame(detailed_metrics)

        # Display enhanced table
        st.dataframe(
            metrics_df,
            use_container_width=True,
            column_config={
                "Store Rating": st.column_config.TextColumn(
                    "Store Rating",
                    help="App store rating"
                ),
                "Total Reviews": st.column_config.TextColumn(
                    "Total Reviews",
                    help="Total reviews scraped"
                ),
                "ABSA Processed": st.column_config.TextColumn(
                    "ABSA Processed",
                    help="Reviews processed through ABSA engine"
                ),
                "SERVQUAL Processed": st.column_config.TextColumn(
                    "SERVQUAL Processed",
                    help="Reviews processed through LLM SERVQUAL"
                ),
                "Recent (7d)": st.column_config.TextColumn(
                    "Recent (7d)",
                    help="Reviews from last 7 days"
                )
            },
            hide_index=True
        )

        # Summary statistics
        st.markdown("#### 📊 Platform Summary")

        col1, col2, col3, col4 = st.columns(4)

        total_apps = len(apps_df)
        total_reviews = sum(int(m['Total Reviews'].replace(',', '')) for m in detailed_metrics)
        total_absa = sum(int(m['ABSA Processed'].replace(',', '')) for m in detailed_metrics)
        total_servqual = sum(int(m['SERVQUAL Processed'].replace(',', '')) for m in detailed_metrics)

        with col1:
            st.metric("📱 Total Apps", f"{total_apps}")

        with col2:
            st.metric("💬 Total Reviews", f"{total_reviews:,}")

        with col3:
            st.metric("🧠 ABSA Processed", f"{total_absa:,}")

        with col4:
            st.metric("🤖 SERVQUAL Processed", f"{total_servqual:,}")

        # Processing progress
        if total_reviews > 0:
            absa_progress = (total_absa / total_reviews) * 100
            servqual_progress = (total_servqual / total_reviews) * 100

            st.markdown("#### 🔄 Processing Progress")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**ABSA Processing Progress**")
                st.progress(absa_progress / 100)
                st.write(f"{absa_progress:.1f}% complete ({total_absa:,}/{total_reviews:,})")

            with col2:
                st.write("**SERVQUAL Processing Progress**")
                st.progress(servqual_progress / 100)
                st.write(f"{servqual_progress:.1f}% complete ({total_servqual:,}/{total_reviews:,})")

    except Exception as e:
        st.error(f"Error loading detailed metrics: {e}")

        # Fallback to simple table
        st.markdown("**Showing basic app information:**")
        display_df = apps_df[['app_name', 'developer', 'category', 'rating']].copy()
        display_df.columns = ['App Name', 'Developer', 'Category', 'Rating']
        display_df['Rating'] = display_df['Rating'].fillna(0).round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_health_status():
    """Render system health status."""
    st.markdown("### System Health")

    _, _, health = load_dashboard_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        db_status = health.get('database', 'unknown')
        if db_status == 'healthy':
            st.success("Database: Healthy")
        else:
            st.error(f"Database: {db_status}")

    with col2:
        redis_status = health.get('redis', 'unknown')
        if redis_status == 'healthy':
            st.success("⚡ Redis: Healthy")
        else:
            st.error(f"⚡ Redis: {redis_status}")

    with col3:
        overall_status = health.get('overall', 'unknown')
        if overall_status == 'healthy':
            st.success("Overall: Healthy")
        else:
            st.error(f"Overall: {overall_status}")


def main():
    """Main dashboard function."""
    # Header
    st.title("📊 ABSA Sentiment Pipeline Dashboard")
    st.markdown("**Enhanced with LLM-powered SERVQUAL Analysis & Real-time Processing**")

    # Sidebar
    page, selected_app = render_sidebar()

    # Load data
    apps_df, reviews_df, health = load_dashboard_data()

    # Main content based on page selection
    if page == "Overview":
        # Overview metrics
        render_overview_metrics(apps_df, reviews_df)

        # Apps overview
        render_app_overview(apps_df)

        if apps_df.empty and reviews_df.empty:
            st.info("🚀 **Welcome to Enhanced ABSA Pipeline!** Use the sidebar to get started.")
            st.markdown("""
            ### 🎯 Getting Started:
            1. Click **"Quick Scrape Amazon"** to get sample data
            2. Or click **"Scrape All Apps"** for multiple apps
            3. Use **"Run Sequential Processing"** for ABSA + LLM SERVQUAL analysis
            4. Explore enhanced dashboards with business intelligence

            ### 📋 Enhanced Capabilities:
            - ✅ Enhanced ABSA Analysis with Amazon focus
            - ✅ LLM-powered SERVQUAL business intelligence
            - ✅ Real-time processing monitoring
            - ✅ Executive summary insights
            - ✅ Customizable aspect analysis
            """)

    elif page == "ABSA Analysis":
        # Enhanced ABSA Analysis
        render_absa_analysis()

    elif page == "SERVQUAL":
        # SERVQUAL Section with debugging
        if apps_df.empty:
            st.warning("📊 No apps available. Please scrape some app data first!")
            st.info("Use the sidebar controls to scrape Amazon or all apps.")
        else:
            # Add debug information
            st.info(f"Debug: Found {len(apps_df)} apps in database")

            # Check if we have any ABSA or SERVQUAL data
            try:
                # Quick database check
                query_result = storage.db.execute_query("SELECT COUNT(*) as count FROM reviews WHERE absa_processed = TRUE")
                absa_count = query_result.iloc[0]['count'] if not query_result.empty else 0

                query_result = storage.db.execute_query("SELECT COUNT(*) as count FROM servqual_scores")
                servqual_count = query_result.iloc[0]['count'] if not query_result.empty else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"ABSA Processed Reviews: {absa_count}")
                with col2:
                    st.info(f"SERVQUAL Scores: {servqual_count}")

                if servqual_count == 0:
                    st.warning("No SERVQUAL data found. Please run LLM SERVQUAL processing first.")
                    st.info("💡 Use 'Run Sequential Processing' from the sidebar to generate data.")

                    # Show sample of what's available
                    sample_query = "SELECT app_id, COUNT(*) as review_count FROM reviews GROUP BY app_id LIMIT 5"
                    sample_df = storage.db.execute_query(sample_query)
                    if not sample_df.empty:
                        st.markdown("**Available Apps:**")
                        st.dataframe(sample_df)
                else:
                    # Render SERVQUAL dashboard
                    servqual_dashboard.render_servqual_section()

            except Exception as e:
                st.error(f"Database error: {str(e)}")
                st.info("Please check your database connection and schema.")

    elif page == "System Health":
        # Health status
        render_health_status()

        # Additional system info
        st.markdown("### System Information")

        col1, col2 = st.columns(2)

        with col1:
            st.info("**Enhanced Features:**\n- Amazon-focused ABSA\n- LLM SERVQUAL analysis\n- Real-time processing\n- Executive insights")

        with col2:
            st.success("**Processing Capabilities:**\n- Sequential processing\n- Progress monitoring\n- Error recovery\n- Performance metrics")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    Enhanced ABSA Pipeline v2.0 - LLM-Powered Business Intelligence
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()