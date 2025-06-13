"""
Enhanced Streamlit Dashboard for ABSA Sentiment Pipeline with SERVQUAL Integration.
Displays scraped reviews, app information, basic analytics, and SERVQUAL service quality analysis.
Phase 1.5 implementation with SERVQUAL business intelligence.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.storage import storage
from src.data.scraper import scrape_app_reviews, scrape_multiple_apps
from dashboard.servqual_components import servqual_dashboard

# Page configuration
st.set_page_config(
    page_title="ABSA Sentiment Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.success-metric {
    background-color: #28a745;
    color: white;
    border-left: 4px solid #1e7e34;
    font-weight: bold;
    text-align: center;
}

.warning-metric {
    background-color: #ffc107;
    color: #212529;
    border-left: 4px solid #e0a800;
    font-weight: bold;
    text-align: center;
}

.info-metric {
    background-color: #17a2b8;
    color: white;
    border-left: 4px solid #138496;
}

.sidebar-info {
    background-color: #495057;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #007bff;
}

.sidebar-info strong {
    color: #ffc107;
}

.servqual-section {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #007bff;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
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


def render_sidebar():
    """Render sidebar with controls and information."""
    st.sidebar.title("🎛️ Controls")

    # Navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Reviews Analysis", "SERVQUAL", "System Health"],
        key="page_selection"
    )

    # App selection
    apps_df, _, _ = load_dashboard_data()

    if not apps_df.empty:
        app_options = ["All Apps"] + apps_df['app_name'].tolist()
        selected_app = st.sidebar.selectbox("Select App", app_options)

        # Show selected app info
        if selected_app != "All Apps":
            app_info = apps_df[apps_df['app_name'] == selected_app].iloc[0]

            st.sidebar.markdown("### App Info")
            st.sidebar.markdown(f"""
            <div class="sidebar-info">
            <strong>Name:</strong> {app_info['app_name']}<br>
            <strong>Developer:</strong> {app_info['developer']}<br>
            <strong>Rating:</strong> {app_info['rating']:.1f}/5.0<br>
            <strong>Category:</strong> {app_info['category']}<br>
            <strong>Reviews:</strong> {app_info.get('total_reviews', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
    else:
        selected_app = "All Apps"
        st.sidebar.warning("No apps found in database")
        page = "Overview"  # Default to overview if no apps

    # Data scraping controls
    st.sidebar.markdown("### Data Collection")

    if st.sidebar.button("Quick Scrape Amazon", type="primary"):
        with st.spinner("Scraping Amazon Shopping reviews..."):
            result = scrape_app_reviews("com.amazon.mShop.android.shopping", count=10)
            if result['success']:
                st.sidebar.success(f"✅ Scraped {result['statistics']['stored']} reviews!")
                st.experimental_rerun()
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
            st.experimental_rerun()

    # Historical scraping controls
    st.sidebar.markdown("### Historical Data Collection")

    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            max_value=datetime.now().date(),
            key="hist_start_date"
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            key="hist_end_date"
        )

    # App selection for historical scraping
    if not apps_df.empty:
        hist_app_options = apps_df['app_name'].tolist()
        selected_hist_apps = st.sidebar.multiselect(
            "Select Apps for Historical Scraping",
            hist_app_options,
            default=hist_app_options[:2] if len(hist_app_options) >= 2 else hist_app_options,
            key="hist_apps"
        )

        # Max reviews per app for historical scraping
        max_historical_reviews = st.sidebar.slider(
            "Max Reviews per App",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            key="hist_max_reviews"
        )

        if st.sidebar.button("Historical Scrape", type="secondary"):
            if start_date >= end_date:
                st.sidebar.error("❌ Start date must be before end date")
            elif not selected_hist_apps:
                st.sidebar.error("❌ Please select at least one app")
            else:
                # Convert app names to app_ids
                selected_app_ids = []
                for app_name in selected_hist_apps:
                    app_data = apps_df[apps_df['app_name'] == app_name]
                    if not app_data.empty:
                        selected_app_ids.append(app_data.iloc[0]['app_id'])

                with st.spinner(f"Historical scraping: {start_date} to {end_date}..."):
                    try:
                        # Import historical scraper
                        from src.data.scraper import scrape_historical_range

                        # Run historical scraping
                        results = scrape_historical_range(
                            app_ids=selected_app_ids,
                            start_date=start_date,
                            end_date=end_date,
                            max_reviews_per_app=max_historical_reviews
                        )

                        # Display results
                        total_scraped = sum(r.get('historical_stored', 0) for r in results)
                        successful_apps = sum(1 for r in results if r.get('success', False))

                        if total_scraped > 0:
                            st.sidebar.success(f"✅ Historical scraping complete!")
                            st.sidebar.info(f"📊 {total_scraped} reviews from {successful_apps} apps")
                            st.cache_data.clear()
                            st.experimental_rerun()
                        else:
                            st.sidebar.warning("⚠️ No historical reviews found in date range")

                    except Exception as e:
                        st.sidebar.error(f"❌ Historical scraping error: {str(e)}")
    else:
        st.sidebar.info("Add apps first for historical scraping")


    # Analysis controls
    st.sidebar.markdown("### Analysis")

    if st.sidebar.button("Run ABSA Analysis", type="secondary"):
        # Get unprocessed reviews first to show count
        try:
            from src.pipeline.batch import BatchProcessor
            batch_processor = BatchProcessor()
            unprocessed_reviews = batch_processor.get_unprocessed_reviews()

            if len(unprocessed_reviews) == 0:
                st.sidebar.warning("⚠️ No unprocessed reviews found")
                return

            st.sidebar.info(f"📊 Found {len(unprocessed_reviews)} unprocessed reviews")

            # Ask user if they want to process all or limit
            if len(unprocessed_reviews) > 100:
                process_all = st.sidebar.checkbox(
                    f"Process all {len(unprocessed_reviews)} reviews? (Unchecked = process 50 for testing)",
                    value=False
                )
                limit = None if process_all else 50
            else:
                limit = None

            # Create progress containers
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()

            with st.spinner("Running ABSA sentiment analysis..."):
                try:
                    # Process with progress updates
                    if limit:
                        status_text.text(f"Processing {limit} reviews for testing...")
                        result = batch_processor.run_daily_processing()

                        # Limit the processing by setting a smaller batch size temporarily
                        original_config = batch_processor.config.max_reviews_per_job
                        batch_processor.config.max_reviews_per_job = limit

                        result = batch_processor.run_daily_processing()

                        # Restore original config
                        batch_processor.config.max_reviews_per_job = original_config
                    else:
                        status_text.text(f"Processing all {len(unprocessed_reviews)} reviews...")
                        result = batch_processor.run_daily_processing()

                    progress_bar.progress(100)

                    if result and result.success:
                        processed_count = result.reviews_processed
                        aspects_count = result.aspects_extracted
                        servqual_count = result.servqual_dimensions_updated
                        processing_time = result.processing_time_seconds

                        status_text.empty()
                        st.sidebar.success(f"✅ ABSA Analysis Complete!")
                        st.sidebar.info(f"📊 Processed: {processed_count} reviews")
                        st.sidebar.info(f"🔍 Extracted: {aspects_count} aspects")
                        st.sidebar.info(f"🎯 SERVQUAL: {servqual_count} dimensions")
                        st.sidebar.info(f"⏱️ Time: {processing_time:.1f} seconds")

                        st.cache_data.clear()
                        st.experimental_rerun()
                    else:
                        error_msg = result.error_message if result else "Unknown error"
                        st.sidebar.error(f"❌ ABSA analysis failed: {error_msg}")

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.sidebar.error(f"❌ Error: {str(e)}")

        except Exception as e:
            st.sidebar.error(f"❌ Setup Error: {str(e)}")

    if st.sidebar.button("Run SERVQUAL Analysis", type="secondary"):
        with st.spinner("Running SERVQUAL service quality analysis..."):
            try:
                # Import SERVQUAL modules
                from src.absa.servqual_mapper import ServqualMapper
                from src.data.servqual_storage import servqual_storage

                # Initialize SERVQUAL processor
                servqual_mapper = ServqualMapper()

                # Process SERVQUAL for all apps
                apps = storage.apps.get_all_apps()
                total_processed = 0

                for app in apps:
                    app_id = app['app_id']
                    result = servqual_mapper.process_daily_servqual(app_id)

                    if result:
                        servqual_storage.store_servqual_scores(result)
                        total_processed += len(result)

                if total_processed > 0:
                    st.sidebar.success(f"✅ Generated {total_processed} SERVQUAL scores!")
                    st.cache_data.clear()  # Clear cache to show new data
                    st.experimental_rerun()
                else:
                    st.sidebar.warning("⚠️ No SERVQUAL data generated - run ABSA analysis first")

            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

    # System status
    st.sidebar.markdown("### System Status")
    _, _, health = load_dashboard_data()

    if health['overall'] == 'healthy':
        st.sidebar.markdown('<div class="metric-card success-metric">🟢 All Systems Healthy</div>',
                            unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="metric-card warning-metric">🟡 System Issues Detected</div>',
                            unsafe_allow_html=True)

    # Refresh controls
    st.sidebar.markdown("### Refresh")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    with col2:
        if auto_refresh:
            st.write("⏱️ 30s")

    if auto_refresh:
        import time
        time.sleep(30)
        st.experimental_rerun()

    return page, selected_app


def render_overview_metrics(apps_df, reviews_df):
    """Render overview metrics cards."""
    st.markdown("### Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📱 Total Apps",
            value=len(apps_df),
            delta=f"{len(apps_df)} tracked"
        )

    with col2:
        total_reviews = len(reviews_df)
        st.metric(
            label="💬 Total Reviews",
            value=f"{total_reviews:,}",
            delta="Across all apps"
        )

    with col3:
        if not reviews_df.empty:
            avg_rating = reviews_df['rating'].mean()
            st.metric(
                label="⭐ Average Rating",
                value=f"{avg_rating:.1f}",
                delta="Out of 5.0"
            )
        else:
            st.metric(label="⭐ Average Rating", value="N/A")

    with col4:
        if not reviews_df.empty:
            # Count reviews from last 7 days
            reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
            recent_reviews = reviews_df[
                reviews_df['review_date'] >= datetime.now() - timedelta(days=7)
                ]
            st.metric(
                label="🔥 Recent Activity",
                value=len(recent_reviews),
                delta="Last 7 days"
            )
        else:
            st.metric(label="🔥 Recent Activity", value="0")


def render_app_overview(apps_df):
    """Render apps overview table."""
    st.markdown("### 📱 Apps Overview")

    if apps_df.empty:
        st.info("No apps found. Use the sidebar to scrape some app data!")
        return

    # Prepare display data
    display_df = apps_df[['app_name', 'developer', 'category', 'rating', 'total_reviews']].copy()
    display_df.columns = ['App Name', 'Developer', 'Category', 'Rating', 'Reviews']

    # Fill NaN values
    display_df['Reviews'] = display_df['Reviews'].fillna(0).astype(int)
    display_df['Rating'] = display_df['Rating'].fillna(0).round(1)

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Rating": st.column_config.NumberColumn(
                "Rating",
                help="App store rating",
                min_value=0,
                max_value=5,
                format="%.1f ⭐"
            ),
            "Reviews": st.column_config.NumberColumn(
                "Reviews",
                help="Number of reviews scraped",
                format="%d"
            )
        }
    )


def render_reviews_analysis(reviews_df, selected_app):
    """Render reviews analysis section."""
    st.markdown("### Reviews Analysis")

    if reviews_df.empty:
        st.info("No reviews found. Scrape some reviews using the sidebar controls!")
        return

    # Filter by selected app if specified
    if selected_app != "All Apps":
        app_id = storage.apps.get_all_apps()
        app_id = [app['app_id'] for app in app_id if app['app_name'] == selected_app]
        if app_id:
            reviews_df = reviews_df[reviews_df['app_id'] == app_id[0]]

    if reviews_df.empty:
        st.info(f"No reviews found for {selected_app}")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # Rating distribution
        st.markdown("#### Rating Distribution")

        rating_counts = reviews_df['rating'].value_counts().sort_index()
        fig_ratings = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title="Review Ratings Distribution",
            color=rating_counts.values,
            color_continuous_scale="RdYlGn"
        )
        fig_ratings.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_ratings, use_container_width=True)

    with col2:
        # Review statistics
        st.markdown("#### Statistics")

        stats_data = {
            "Total Reviews": len(reviews_df),
            "Average Rating": f"{reviews_df['rating'].mean():.1f}",
            "5-Star Reviews": len(reviews_df[reviews_df['rating'] == 5]),
            "1-Star Reviews": len(reviews_df[reviews_df['rating'] == 1]),
            "Avg Content Length": f"{reviews_df['content'].str.len().mean():.0f} chars"
        }

        for metric, value in stats_data.items():
            st.metric(label=metric, value=value)


def render_recent_reviews(reviews_df, selected_app):
    """Render recent reviews section."""
    st.markdown("### Recent Reviews")

    if reviews_df.empty:
        st.info("No reviews to display")
        return

    # Filter by selected app if specified
    if selected_app != "All Apps":
        app_id = storage.apps.get_all_apps()
        app_id = [app['app_id'] for app in app_id if app['app_name'] == selected_app]
        if app_id:
            reviews_df = reviews_df[reviews_df['app_id'] == app_id[0]]

    # Sort by most recent
    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
    recent_reviews = reviews_df.sort_values('review_date', ascending=False).head(10)

    for idx, review in recent_reviews.iterrows():
        with st.expander(f"⭐ {review['rating']}/5 - {review['review_date'].strftime('%Y-%m-%d')}"):
            st.write(f"**Content:** {review['content']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**App:** {review['app_id'].split('.')[-1].title()}")
            with col2:
                st.write(f"**Length:** {len(review['content'])} chars")
            with col3:
                if review.get('processed', False):
                    st.success("✅ Processed")
                else:
                    st.warning("⏳ Pending")


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
    st.markdown("**Enhanced with SERVQUAL Service Quality Analysis**")

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
            st.info("🚀 **Welcome to ABSA Pipeline!** Use the sidebar to scrape your first app reviews.")
            st.markdown("""
            ### 🎯 Getting Started:
            1. Click **"Quick Scrape Amazon"** to get sample data
            2. Or click **"Scrape All Apps"** for multiple apps
            3. Explore the dashboard as data loads
            4. Use **SERVQUAL section** for business intelligence

            ### 📋 Current Capabilities:
            - ✅ Google Play Store review scraping
            - ✅ App metadata collection
            - ✅ Basic review analytics
            - ✅ SERVQUAL service quality analysis
            - ✅ System health monitoring
            - ⏳ ABSA sentiment analysis (Phase 2)
            """)

    elif page == "Reviews Analysis":
        # Reviews analysis
        render_reviews_analysis(reviews_df, selected_app)

        # Recent reviews
        render_recent_reviews(reviews_df, selected_app)

    elif page == "SERVQUAL":
        # SERVQUAL Section
        if apps_df.empty:
            st.warning("📊 No apps available. Please scrape some app data first!")
            st.info("Use the sidebar controls to scrape Amazon or all apps.")
        else:
            # Render SERVQUAL dashboard
            servqual_dashboard.render_servqual_section()

    elif page == "System Health":
        # Health status
        render_health_status()

        # Additional system info
        st.markdown("### 📊 System Information")

        col1, col2 = st.columns(2)

        with col1:
            st.info("**Phase 1 Features:**\n- Data collection\n- Basic analytics\n- SERVQUAL analysis\n- Health monitoring")

        with col2:
            st.warning("**Phase 2 Coming:**\n- Real-time ABSA\n- Advanced ML models\n- Predictive analytics\n- API endpoints")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    ABSA Pipeline v1.6
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()