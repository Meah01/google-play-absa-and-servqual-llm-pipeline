"""
SERVQUAL Dashboard Components for ABSA Pipeline.
Provides specialized dashboard sections for SERVQUAL service quality analysis.
Includes Amazon-focused analytics and competitive comparisons.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

from src.data.servqual_storage import servqual_storage


class ServqualDashboard:
    """SERVQUAL-specific dashboard components."""

    def __init__(self):
        self.servqual_storage = servqual_storage

        # SERVQUAL dimension colors for consistent visualization
        self.dimension_colors = {
            'reliability': '#1f77b4',  # Blue
            'assurance': '#ff7f0e',  # Orange
            'tangibles': '#2ca02c',  # Green
            'empathy': '#d62728',  # Red
            'responsiveness': '#9467bd'  # Purple
        }

    def render_servqual_section(self):
        """Main SERVQUAL section with tabs for different analyses."""
        st.markdown("## SERVQUAL Service Quality Analysis")
        st.markdown("**Strategic business insights using validated service quality framework**")

        # Create tabs
        tab1, tab2 = st.tabs(["Amazon Focus", "App Comparisons"])

        with tab1:
            self.render_amazon_focus()

        with tab2:
            self.render_app_comparisons()

    def render_amazon_focus(self):
        """Amazon-focused SERVQUAL analysis."""
        st.markdown("### Amazon SERVQUAL Analysis")

        # Time period selection
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("**Amazon Shopping Service Quality Dashboard**")

        with col2:
            days = st.selectbox(
                "Analysis Period",
                [7, 14, 30, 60, 90],
                index=2,
                key="amazon_period"
            )

        with col3:
            auto_refresh = st.checkbox("Auto-refresh", key="amazon_refresh")

        # Load Amazon data
        try:
            amazon_data = self.servqual_storage.get_amazon_focus_data(days=days)

            if not amazon_data or not amazon_data.get('current_profile'):
                st.warning("📊 No SERVQUAL data available for Amazon. Process some reviews first!")
                return

            # Current SERVQUAL Profile
            self.render_amazon_radar_chart(amazon_data['current_profile'])

            # Amazon Trends
            if amazon_data.get('trends'):
                self.render_amazon_trends(amazon_data['trends'], days)

            # Competitive Ranking
            if amazon_data.get('competitive_ranking'):
                self.render_amazon_ranking(amazon_data['competitive_ranking'])

        except Exception as e:
            st.error(f"Error loading Amazon SERVQUAL data: {e}")

    def render_amazon_radar_chart(self, profile_data: dict):
        """Render Amazon SERVQUAL radar chart."""
        st.markdown("#### Current SERVQUAL Profile")

        try:
            dimensions = profile_data.get('dimensions', {})

            if not dimensions:
                st.info("No dimension data available")
                return

            # Prepare radar chart data
            dimension_names = list(dimensions.keys())
            quality_scores = [dimensions[dim]['quality_score'] for dim in dimension_names]

            # Create radar chart
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=quality_scores,
                theta=dimension_names,
                fill='toself',
                name='Amazon Shopping',
                line_color='rgb(31, 119, 180)',
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[1, 5],
                        tickmode='linear',
                        tick0=1,
                        dtick=1,
                        ticktext=['1 - Far Below', '2 - Below', '3 - Neutral',
                                  '4 - Meets', '5 - Exceeds'],
                        tickvals=[1, 2, 3, 4, 5]
                    )
                ),
                showlegend=True,
                title=f"Amazon SERVQUAL Profile - Overall Score: {profile_data.get('overall_quality', 0)}/5",
                height=500
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**📋 Dimension Scores**")

                for dim, data in dimensions.items():
                    quality_score = data['quality_score']
                    sentiment_score = data['sentiment_score']
                    review_count = data['review_count']

                    # Color code based on performance
                    if quality_score >= 4:
                        color = "🟢"
                    elif quality_score >= 3:
                        color = "🟡"
                    else:
                        color = "🔴"

                    st.metric(
                        label=f"{color} {dim.title()}",
                        value=f"{quality_score}/5",
                        delta=f"{sentiment_score:+.2f} sentiment",
                        help=f"Based on {review_count} reviews"
                    )

        except Exception as e:
            st.error(f"Error rendering radar chart: {e}")

    def render_amazon_trends(self, trends_data: list, days: int):
        """Render Amazon dimension trends over time."""
        st.markdown("#### 📈 Amazon Trends - All Dimensions")

        try:
            if not trends_data:
                st.info("No trend data available")
                return

            # Convert to DataFrame
            df = pd.DataFrame(trends_data)
            df['date'] = pd.to_datetime(df['date'])

            # Create time series chart
            fig = go.Figure()

            for dimension in df['dimension'].unique():
                dim_data = df[df['dimension'] == dimension].sort_values('date')

                fig.add_trace(go.Scatter(
                    x=dim_data['date'],
                    y=dim_data['quality_score'],
                    mode='lines+markers',
                    name=dimension.title(),
                    line=dict(color=self.dimension_colors.get(dimension, '#000000')),
                    hovertemplate=f'<b>{dimension.title()}</b><br>' +
                                  'Date: %{x}<br>' +
                                  'Quality Score: %{y}/5<br>' +
                                  '<extra></extra>'
                ))

            fig.update_layout(
                title=f"Amazon SERVQUAL Trends - Last {days} Days",
                xaxis_title="Date",
                yaxis_title="Quality Score (1-5)",
                yaxis=dict(range=[1, 5]),
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_quality = df['quality_score'].mean()
                st.metric("Average Quality", f"{avg_quality:.2f}/5")

            with col2:
                trend_direction = "📈" if df['quality_score'].iloc[-1] > df['quality_score'].iloc[0] else "📉"
                st.metric("Trend", trend_direction)

            with col3:
                total_reviews = df['review_count'].sum()
                st.metric("Total Reviews", f"{total_reviews:,}")

        except Exception as e:
            st.error(f"Error rendering trends: {e}")

    def render_amazon_ranking(self, ranking_data: dict):
        """Render Amazon competitive ranking."""
        st.markdown("#### Competitive Ranking")

        try:
            if not ranking_data:
                st.info("No ranking data available")
                return

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Amazon vs Competitors**")

                for dimension, rank_info in ranking_data.items():
                    rank = rank_info['rank']
                    total = rank_info['total_apps']
                    percentile = rank_info['percentile']

                    # Rank indicator
                    if rank == 1:
                        indicator = "🥇"
                    elif rank == 2:
                        indicator = "🥈"
                    elif rank == 3:
                        indicator = "🥉"
                    else:
                        indicator = "📍"

                    st.metric(
                        label=f"{indicator} {dimension.title()}",
                        value=f"#{rank} of {total}",
                        delta=f"{percentile}th percentile"
                    )

            with col2:
                # Ranking chart
                dimensions = list(ranking_data.keys())
                ranks = [ranking_data[dim]['rank'] for dim in dimensions]
                totals = [ranking_data[dim]['total_apps'] for dim in dimensions]

                fig = go.Figure(data=[
                    go.Bar(
                        x=dimensions,
                        y=ranks,
                        name='Amazon Rank',
                        marker_color='lightblue'
                    )
                ])

                fig.update_layout(
                    title="Amazon Ranking by Dimension",
                    xaxis_title="SERVQUAL Dimension",
                    yaxis_title="Rank (Lower is Better)",
                    yaxis=dict(autorange='reversed'),
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering ranking: {e}")

    def render_app_comparisons(self):
        """Multi-app SERVQUAL comparisons."""
        st.markdown("### App Comparisons")

        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            selected_dimension = st.selectbox(
                "Select SERVQUAL Dimension",
                ['reliability', 'assurance', 'tangibles', 'empathy', 'responsiveness'],
                key="comparison_dimension"
            )

        with col2:
            days = st.selectbox(
                "Time Period",
                [7, 14, 30, 60],
                index=2,
                key="comparison_period"
            )

        with col3:
            chart_type = st.selectbox(
                "Chart Type",
                ["Time Series", "Heatmap", "Rankings"],
                key="comparison_chart"
            )

        # Load comparison data
        try:
            if chart_type == "Time Series":
                self.render_dimension_time_series(selected_dimension, days)
            elif chart_type == "Heatmap":
                self.render_servqual_heatmap(days)
            elif chart_type == "Rankings":
                self.render_dimension_rankings(selected_dimension, days)

        except Exception as e:
            st.error(f"Error loading comparison data: {e}")

    def render_dimension_time_series(self, dimension: str, days: int):
        """Render time series comparison for a dimension across apps."""
        st.markdown(f"#### 📊 {dimension.title()} Trends Across Apps")

        try:
            # Get trends data for all apps
            trends_df = self.servqual_storage.get_dimension_trends(
                app_id=None,
                dimension=dimension,
                days=days
            )

            if trends_df.empty:
                st.info(f"No data available for {dimension} dimension")
                return

            # Get app names
            app_names = {
                'com.amazon.mShop.android.shopping': 'Amazon',
                'com.einnovation.temu': 'Temu',
                'com.zzkko': 'SHEIN',
                'com.ebay.mobile': 'eBay',
                'com.etsy.android': 'Etsy'
            }

            trends_df['app_name'] = trends_df['app_id'].map(app_names)
            trends_df['date'] = pd.to_datetime(trends_df['date'])

            # App selection
            available_apps = trends_df['app_name'].unique()
            selected_apps = st.multiselect(
                "Select Apps to Compare",
                available_apps,
                default=available_apps,
                key=f"apps_{dimension}"
            )

            filtered_df = trends_df[trends_df['app_name'].isin(selected_apps)]

            if filtered_df.empty:
                st.warning("No data for selected apps")
                return

            # Create time series chart
            fig = px.line(
                filtered_df,
                x='date',
                y='quality_score',
                color='app_name',
                title=f"{dimension.title()} Quality Scores Over Time",
                labels={
                    'quality_score': 'Quality Score (1-5)',
                    'date': 'Date',
                    'app_name': 'App'
                },
                markers = True
            )

            fig.update_traces(
                mode='lines+markers',
                marker=dict(size=8),
                line=dict(width=3)
            )

            fig.update_layout(
                yaxis=dict(range=[1, 5]),
                hovermode='x unified',
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            summary = filtered_df.groupby('app_name').agg({
                'quality_score': ['mean', 'std'],
                'sentiment_score': 'mean',
                'review_count': 'sum'
            }).round(2)

            summary.columns = ['Avg Quality', 'Quality StdDev', 'Avg Sentiment', 'Total Reviews']
            summary = summary.sort_values('Avg Quality', ascending=False)

            st.markdown("**📋 Summary Statistics**")
            st.dataframe(summary, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering time series: {e}")

        #Debug section
        """
        st.write(f"Debug: Retrieved {len(trends_df)} records")
        if not trends_df.empty:
            st.write(f"Debug: Apps in data: {trends_df['app_id'].unique().tolist()}")

            # Add detailed debug info
            st.write("Debug: Data points per app:")
            app_counts = trends_df.groupby('app_id').size()
            for app_id, count in app_counts.items():
                app_name = app_names.get(app_id, app_id)
                st.write(f"  - {app_name}: {count} data points")

            st.write("Debug: Date range:")
            st.write(f"  - Min date: {trends_df['date'].min()}")
            st.write(f"  - Max date: {trends_df['date'].max()}")

            # Show actual data
            st.write("Debug: Raw data sample:")
            st.dataframe(trends_df[['app_id', 'date', 'quality_score']].head(10))
        """
    def render_servqual_heatmap(self, days: int):
        """Render SERVQUAL heatmap across apps and dimensions."""
        st.markdown("#### SERVQUAL Performance Heatmap")

        try:
            # Get data for all dimensions and apps
            all_data = []
            dimensions = ['reliability', 'assurance', 'tangibles', 'empathy', 'responsiveness']

            for dimension in dimensions:
                comp_data = self.servqual_storage.get_comparative_analysis(dimension, days)
                if not comp_data.empty:
                    comp_data['dimension'] = dimension
                    all_data.append(comp_data)

            if not all_data:
                st.info("No data available for heatmap")
                return

            combined_df = pd.concat(all_data, ignore_index=True)

            # Pivot for heatmap
            heatmap_data = combined_df.pivot(
                index='app_name',
                columns='dimension',
                values='avg_quality'
            )

            # Create heatmap
            fig = px.imshow(
                heatmap_data.values,
                labels=dict(x="SERVQUAL Dimension", y="App", color="Quality Score"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                zmin=1,
                zmax=5
            )

            fig.update_layout(
                title=f"SERVQUAL Performance Heatmap - Last {days} Days",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Performance insights
            st.markdown("**Key Insights**")

            # Best performing app overall
            overall_scores = heatmap_data.mean(axis=1).sort_values(ascending=False)
            best_app = overall_scores.index[0]
            best_score = overall_scores.iloc[0]

            # Best dimension overall
            dimension_scores = heatmap_data.mean(axis=0).sort_values(ascending=False)
            best_dimension = dimension_scores.index[0]
            best_dim_score = dimension_scores.iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"🏆 **Best Overall App:** {best_app} ({best_score:.2f}/5)")

            with col2:
                st.info(f"⭐ **Strongest Dimension:** {best_dimension.title()} ({best_dim_score:.2f}/5)")

        except Exception as e:
            st.error(f"Error rendering heatmap: {e}")

    def render_dimension_rankings(self, dimension: str, days: int):
        """Render competitive rankings for a dimension."""
        st.markdown(f"#### 🏅 {dimension.title()} Rankings")

        try:
            comp_df = self.servqual_storage.get_comparative_analysis(dimension, days)

            if comp_df.empty:
                st.info(f"No ranking data available for {dimension}")
                return

            # Create ranking chart
            fig = go.Figure(data=[
                go.Bar(
                    x=comp_df['app_name'],
                    y=comp_df['avg_quality'],
                    text=[f"{score:.2f}" for score in comp_df['avg_quality']],
                    textposition='auto',
                    marker_color=['gold' if i == 0 else 'silver' if i == 1 else 'brown' if i == 2 else 'lightblue'
                                  for i in range(len(comp_df))]
                )
            ])

            fig.update_layout(
                title=f"{dimension.title()} Quality Rankings",
                xaxis_title="App",
                yaxis_title="Average Quality Score",
                yaxis=dict(range=[1, 5]),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed rankings table
            st.markdown("**📊 Detailed Rankings**")

            ranking_display = comp_df.copy()
            ranking_display['rank'] = range(1, len(ranking_display) + 1)
            ranking_display = ranking_display[['rank', 'app_name', 'avg_quality', 'avg_sentiment', 'total_reviews']]
            ranking_display.columns = ['Rank', 'App', 'Avg Quality', 'Avg Sentiment', 'Total Reviews']

            st.dataframe(ranking_display, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error rendering rankings: {e}")


# Global instance
servqual_dashboard = ServqualDashboard()