#!/usr/bin/env python3
"""
FPL AI Assistant - Streamlit Web Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
import time

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data.fpl_api import FPLApi
from src.models.fpl_predictor import FPLPlayerPredictor, predict_next_gameweek
from src.optimization.team_optimizer import FPLTeamOptimizer, optimize_fpl_team

# Page config
st.set_page_config(
    page_title="FPL AI Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #00FF87, #60EFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00FF87;
    }
    .player-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_fpl_data():
    """Load FPL data with caching."""
    api = FPLApi()
    return api.get_players_df()


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_predictions():
    """Load latest predictions with caching."""
    predictions_dir = project_root / "data" / "predictions"
    
    if not predictions_dir.exists():
        return pd.DataFrame()
    
    pred_files = list(predictions_dir.glob("fpl_predictions_*.csv"))
    if not pred_files:
        return pd.DataFrame()
    
    latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest_file)


def generate_new_predictions(players_df):
    """Generate new predictions using the trained model."""
    
    with st.spinner("🤖 Generating AI predictions..."):
        # Filter active players
        active_players = players_df[players_df['minutes'] > 0].copy()
        
        if len(active_players) < 10:
            st.error("Not enough active players to generate predictions")
            return pd.DataFrame()
        
        # Train and predict
        predictor = FPLPlayerPredictor("xgboost")
        metrics = predictor.train(active_players)
        
        # Make predictions for all players
        predictions = predictor.predict(players_df)
        players_df['predicted_points'] = predictions
        players_df['predicted_points'] = players_df['predicted_points'].clip(lower=0)
        
        # Get top predictions
        top_predictions = players_df[players_df['minutes'] > 0].nlargest(50, 'predicted_points')
        
        st.success(f"✅ Generated predictions! Model R² = {metrics['test_r2']:.3f}")
        
        return top_predictions


def create_player_comparison_chart(predictions_df):
    """Create interactive player comparison chart."""
    
    if predictions_df.empty:
        st.warning("No prediction data available")
        return
    
    fig = px.scatter(
        predictions_df.head(20),
        x="price",
        y="predicted_points",
        size="ownership",
        color="position",
        hover_name="web_name",
        hover_data=["team_short", "total_points"],
        title="Player Performance vs Price (Top 20 Predicted)",
        labels={
            "price": "Price (£m)",
            "predicted_points": "Predicted Points",
            "ownership": "Ownership %"
        }
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def create_value_analysis_chart(predictions_df):
    """Create value analysis chart."""
    
    if predictions_df.empty or 'predicted_points' not in predictions_df.columns:
        return
    
    predictions_df['value_ratio'] = predictions_df['predicted_points'] / predictions_df['price']
    top_value = predictions_df.nlargest(15, 'value_ratio')
    
    fig = px.bar(
        top_value,
        x="web_name",
        y="value_ratio",
        color="position",
        title="Best Value Players (Predicted Points per £m)",
        labels={"value_ratio": "Points per £m", "web_name": "Player"}
    )
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def display_optimized_team(optimization_result):
    """Display optimized team in formatted cards."""
    
    if not optimization_result or optimization_result.get('squad_optimization', {}).get('status') != 'optimal':
        st.error("Team optimization failed")
        return
    
    squad = optimization_result['squad_optimization']['squad']
    xi_result = optimization_result.get('starting_xi_optimization', {})
    
    # Team overview metrics
    results = optimization_result['squad_optimization']['results']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"£{results['total_cost']:.1f}m")
    
    with col2:
        st.metric("Budget Remaining", f"£{results['budget_remaining']:.1f}m")
    
    with col3:
        st.metric("Predicted Points", f"{results['total_objective']:.1f}")
    
    with col4:
        st.metric("Value Ratio", f"{results['average_value_ratio']:.2f} pts/£m")
    
    # Starting XI
    if xi_result:
        st.subheader(f"⚽ Starting XI ({xi_result['formation']})")
        
        starting_xi = xi_result['starting_xi']
        captain = xi_result['captain_suggestion']
        
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        for pos_id in [1, 2, 3, 4]:
            pos_players = starting_xi[starting_xi['element_type'] == pos_id]
            
            if not pos_players.empty:
                st.markdown(f"**{positions[pos_id]} ({len(pos_players)})**")
                
                for _, player in pos_players.iterrows():
                    is_captain = player['web_name'] == captain
                    captain_emoji = "👑 " if is_captain else ""
                    
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"{captain_emoji}{player['web_name']} ({player.get('team_short', 'N/A')})")
                    with col2:
                        st.write(f"£{player['price']:.1f}m")
                    with col3:
                        st.write(f"{player.get('predicted_points', 0):.1f} pts")
                    with col4:
                        st.write(f"{player.get('ownership', 0):.1f}%")
        
        # Bench
        bench = xi_result['bench']
        st.markdown("**Bench**")
        
        for _, player in bench.iterrows():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"{player['web_name']} ({positions[player['element_type']]}, {player.get('team_short', 'N/A')})")
            with col2:
                st.write(f"£{player['price']:.1f}m")
            with col3:
                st.write(f"{player.get('predicted_points', 0):.1f} pts")
            with col4:
                st.write(f"{player.get('ownership', 0):.1f}%")


def captain_analysis_page():
    """Captain analysis and recommendation page."""
    
    st.header("👑 Captain Analysis")
    
    players_df = load_fpl_data()
    predictions_df = load_predictions()
    
    if players_df.empty:
        st.error("Failed to load player data")
        return
    
    # Merge predictions if available
    if not predictions_df.empty:
        analysis_df = players_df.merge(
            predictions_df[['web_name', 'predicted_points']], 
            on='web_name', 
            how='left'
        )
        objective = 'predicted_points'
    else:
        analysis_df = players_df.copy()
        objective = 'total_points'
    
    # Captain criteria
    st.subheader("Captain Selection Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_ownership = st.slider("Minimum Ownership %", 0.0, 50.0, 10.0, 0.5)
        max_ownership = st.slider("Maximum Ownership %", 10.0, 100.0, 40.0, 0.5)
    
    with col2:
        min_price = st.slider("Minimum Price £m", 4.0, 15.0, 6.0, 0.1)
        min_predicted = st.slider("Minimum Predicted Points", 5.0, 20.0, 8.0, 0.5)
    
    # Filter potential captains
    captain_candidates = analysis_df[
        (analysis_df['ownership'].between(min_ownership, max_ownership)) &
        (analysis_df['price'] >= min_price) &
        (analysis_df[objective] >= min_predicted) &
        (analysis_df['minutes'] > 0)
    ].copy()
    
    if captain_candidates.empty:
        st.warning("No players match the selected criteria")
        return
    
    # Calculate captain metrics
    captain_candidates['captain_points'] = captain_candidates[objective] * 2  # Double points for captain
    captain_candidates['risk_score'] = 100 - captain_candidates['ownership']  # Higher ownership = lower risk
    captain_candidates['value_score'] = captain_candidates[objective] / captain_candidates['price']
    
    # Sort by predicted captain points
    top_captains = captain_candidates.nlargest(15, 'captain_points')
    
    st.subheader("🎯 Top Captain Picks")
    
    # Display captain options
    for i, (_, player) in enumerate(top_captains.iterrows(), 1):
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{i}**")
            with col2:
                st.write(f"**{player['web_name']}** ({player.get('position', 'N/A')}, {player.get('team_short', 'N/A')})")
            with col3:
                st.write(f"{player['captain_points']:.1f} pts")
            with col4:
                st.write(f"£{player['price']:.1f}m")
            with col5:
                st.write(f"{player['ownership']:.1f}%")
            with col6:
                if player['ownership'] < 15:
                    st.write("🔥 Differential")
                elif player['ownership'] > 30:
                    st.write("✅ Safe")
                else:
                    st.write("⚖️ Balanced")
    
    # Captain recommendation chart
    fig = px.scatter(
        top_captains,
        x="ownership",
        y="captain_points",
        size="price",
        color="position",
        hover_name="web_name",
        title="Captain Points vs Ownership (Risk vs Reward)",
        labels={
            "ownership": "Ownership %",
            "captain_points": "Captain Points (2x)",
            "price": "Price £m"
        }
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ FPL AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Player Predictions", "Team Optimizer", "Captain Analysis", "Data Overview"]
    )
    
    # Load data
    with st.spinner("📡 Loading FPL data..."):
        players_df = load_fpl_data()
    
    if players_df.empty:
        st.error("❌ Failed to load FPL data. Please check your connection.")
        return
    
    st.sidebar.success(f"✅ Loaded {len(players_df)} players")
    
    # Page routing
    if page == "Player Predictions":
        st.header("🔮 Player Performance Predictions")
        
        # Load or generate predictions
        predictions_df = load_predictions()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if not predictions_df.empty:
                last_update = predictions_df.attrs.get('last_update', 'Unknown')
                st.info(f"📊 Using cached predictions ({len(predictions_df)} players)")
            else:
                st.warning("No cached predictions found")
        
        with col2:
            if st.button("🔄 Generate New Predictions"):
                predictions_df = generate_new_predictions(players_df)
        
        if not predictions_df.empty:
            # Display predictions
            st.subheader("🌟 Top Predicted Performers")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                position_filter = st.selectbox(
                    "Position", 
                    ["All"] + list(predictions_df['position'].unique()) if 'position' in predictions_df.columns else ["All"]
                )
            
            with col2:
                max_price = st.slider("Max Price £m", 4.0, 15.0, 12.0, 0.5)
            
            with col3:
                min_ownership = st.slider("Min Ownership %", 0.0, 50.0, 0.0, 1.0)
            
            # Apply filters
            filtered_df = predictions_df.copy()
            
            if position_filter != "All":
                filtered_df = filtered_df[filtered_df['position'] == position_filter]
            
            filtered_df = filtered_df[
                (filtered_df['price'] <= max_price) &
                (filtered_df['ownership'] >= min_ownership)
            ]
            
            # Display filtered results
            display_cols = ['web_name', 'position', 'team_short', 'price', 'predicted_points', 'ownership', 'total_points']
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # Charts
            create_player_comparison_chart(filtered_df)
            create_value_analysis_chart(filtered_df)
    
    elif page == "Team Optimizer":
        st.header("🎯 FPL Team Optimization")
        
        # Optimization settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            formation = st.selectbox("Formation", ["3-5-2", "4-4-2", "4-3-3", "3-4-3", "5-4-1"])
        
        with col2:
            budget = st.slider("Budget £m", 80.0, 100.0, 100.0, 0.5)
        
        with col3:
            use_predictions = st.checkbox("Use AI Predictions", True)
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                min_price = st.slider("Min Player Price £m", 4.0, 10.0, 4.0, 0.1)
                max_price = st.slider("Max Player Price £m", 10.0, 15.0, 15.0, 0.1)
            
            with col2:
                must_include = st.text_input("Must Include Players (comma-separated)")
                exclude_players = st.text_input("Exclude Players (comma-separated)")
        
        # Run optimization
        if st.button("🚀 Optimize Team"):
            with st.spinner("🎯 Optimizing your FPL team..."):
                
                # Prepare data
                optimization_data = players_df.copy()
                
                if use_predictions:
                    predictions_df = load_predictions()
                    if not predictions_df.empty:
                        optimization_data = optimization_data.merge(
                            predictions_df[['web_name', 'predicted_points']], 
                            on='web_name', 
                            how='left'
                        )
                
                # Parse inputs
                must_include_list = [name.strip() for name in must_include.split(",")] if must_include else None
                exclude_list = [name.strip() for name in exclude_players.split(",")] if exclude_players else None
                
                # Run optimization
                optimizer = FPLTeamOptimizer()
                
                objective = 'predicted_points' if use_predictions and 'predicted_points' in optimization_data.columns else 'total_points'
                
                squad_result = optimizer.optimize_squad(
                    optimization_data,
                    objective=objective,
                    min_price=min_price,
                    max_price=max_price,
                    exclude_players=exclude_list,
                    must_include=must_include_list
                )
                
                if squad_result['status'] == 'optimal':
                    xi_result = optimizer.optimize_starting_xi(
                        squad_result['squad'],
                        formation=formation,
                        objective=objective
                    )
                    
                    optimization_result = {
                        'squad_optimization': squad_result,
                        'starting_xi_optimization': xi_result
                    }
                    
                    display_optimized_team(optimization_result)
                else:
                    st.error("❌ Optimization failed. Try adjusting your constraints.")
    
    elif page == "Captain Analysis":
        captain_analysis_page()
    
    elif page == "Data Overview":
        st.header("📊 FPL Data Overview")
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", len(players_df))
        
        with col2:
            active_players = len(players_df[players_df['minutes'] > 0])
            st.metric("Active Players", active_players)
        
        with col3:
            avg_price = players_df['price'].mean()
            st.metric("Average Price", f"£{avg_price:.1f}m")
        
        with col4:
            total_points = players_df['total_points'].sum()
            st.metric("Total Points", f"{total_points:,}")
        
        # Position breakdown
        st.subheader("📍 Players by Position")
        if 'position' in players_df.columns:
            position_counts = players_df['position'].value_counts()
            
            fig = px.pie(
                values=position_counts.values,
                names=position_counts.index,
                title="Player Distribution by Position"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Price distribution
        st.subheader("💰 Price Distribution")
        fig = px.histogram(
            players_df,
            x="price",
            nbins=30,
            title="Player Price Distribution",
            labels={"price": "Price (£m)", "count": "Number of Players"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top performers
        st.subheader("🌟 Top Performers This Season")
        top_performers = players_df[players_df['minutes'] > 0].nlargest(10, 'total_points')
        
        display_cols = ['web_name', 'position', 'team_short', 'price', 'total_points', 'ownership']
        available_cols = [col for col in display_cols if col in top_performers.columns]
        
        st.dataframe(
            top_performers[available_cols],
            use_container_width=True,
            hide_index=True
        )


if __name__ == "__main__":
    main()