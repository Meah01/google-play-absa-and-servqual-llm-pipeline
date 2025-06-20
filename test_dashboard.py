"""
Simple Dashboard Test - Minimal version to test rendering
Save as: test_simple_dashboard.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_rendering():
    """Test basic Streamlit rendering"""
    st.title("🔧 Dashboard Test - Basic Rendering")
    st.success("✅ Basic Streamlit rendering works!")

    st.write("Testing basic components...")

    # Test metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Metric 1", "1,234")
    with col2:
        st.metric("Test Metric 2", "5,678")
    with col3:
        st.metric("Test Metric 3", "9,999")

    st.success("✅ Basic components work!")

def test_data_loading():
    """Test data loading components"""
    st.markdown("## 📊 Testing Data Loading")

    try:
        from src.data.storage import storage
        st.success("✅ Storage import works")

        # Test simple query
        result = storage.db.execute_query("SELECT COUNT(*) as count FROM apps")
        apps_count = result.iloc[0]['count']
        st.success(f"✅ Database query works: {apps_count} apps found")

        # Test dashboard data loader
        from dashboard.data_loader import dashboard_data_loader
        st.success("✅ Dashboard data loader import works")

        # Test Amazon data
        amazon_data = dashboard_data_loader.load_amazon_fixed_absa(days=30)
        aspects_count = len(amazon_data.get('aspects', []))
        st.success(f"✅ Amazon ABSA data loads: {aspects_count} aspects found")

        if aspects_count > 0:
            st.write("**Sample Amazon aspects:**")
            for aspect in amazon_data['aspects'][:3]:  # Show first 3
                st.write(f"- {aspect['aspect']}: {aspect['positive_pct']:.1f}% positive")

    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        st.write("**Error details:**", str(e))

def test_dashboard_components():
    """Test individual dashboard components"""
    st.markdown("## 🎛️ Testing Dashboard Components")

    try:
        # Test processing status
        from dashboard.data_loader import dashboard_data_loader
        status_data = dashboard_data_loader.load_processing_status()

        st.success("✅ Processing status loads")
        st.write(f"Active jobs: {len(status_data.get('active_jobs', []))}")
        st.write(f"ABSA queue: {status_data.get('queue_metrics', {}).get('total_pending_absa', 0)}")

        # Test apps loading
        apps_data = dashboard_data_loader.get_app_list_with_data()
        st.success(f"✅ Apps with data: {len(apps_data)} found")

        if apps_data:
            st.write("**Apps with ABSA data:**")
            for app in apps_data[:3]:  # Show first 3
                st.write(f"- {app['app_name']}: {app['absa_processed_count']} processed")

    except Exception as e:
        st.error(f"❌ Component testing error: {e}")
        st.write("**Error details:**", str(e))

def test_servqual_components():
    """Test SERVQUAL components"""
    st.markdown("## 🤖 Testing SERVQUAL Components")

    try:
        from dashboard.servqual_components import servqual_dashboard
        st.success("✅ SERVQUAL components import works")

        from src.data.servqual_storage import servqual_storage
        st.success("✅ SERVQUAL storage import works")

        # Test Amazon SERVQUAL data
        amazon_data = servqual_storage.get_amazon_focus_data(days=30)
        if amazon_data.get('current_profile'):
            st.success("✅ Amazon SERVQUAL data loads")
            profile = amazon_data['current_profile']
            st.write(f"Overall quality: {profile.get('overall_quality', 0):.2f}/5")
            st.write(f"Dimensions: {len(profile.get('dimensions', {}))}")
        else:
            st.warning("⚠️ No Amazon SERVQUAL profile found")

    except Exception as e:
        st.error(f"❌ SERVQUAL testing error: {e}")
        st.write("**Error details:**", str(e))

def main():
    st.set_page_config(
        page_title="Dashboard Test",
        page_icon="🔧",
        layout="wide"
    )

    # Basic rendering test
    test_basic_rendering()

    # Data loading test
    test_data_loading()

    # Dashboard components test
    test_dashboard_components()

    # SERVQUAL components test
    test_servqual_components()

    st.markdown("---")
    st.success("🎯 **All tests completed!** If you see this, basic rendering works.")
    st.info("**Next step:** Check main dashboard for specific rendering issues.")

if __name__ == "__main__":
    main()