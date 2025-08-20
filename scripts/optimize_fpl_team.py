#!/usr/bin/env python3
"""
FPL Team Optimization Script
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.fpl_api import FPLApi
from src.optimization.team_optimizer import FPLTeamOptimizer, optimize_fpl_team


def load_latest_predictions() -> pd.DataFrame:
    """Load the most recent predictions file."""
    
    predictions_dir = project_root / "data" / "predictions"
    
    if not predictions_dir.exists():
        return pd.DataFrame()
    
    # Find latest predictions file
    pred_files = list(predictions_dir.glob("fpl_predictions_*.csv"))
    if not pred_files:
        return pd.DataFrame()
    
    latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Loading predictions from: {latest_file.name}")
    
    return pd.read_csv(latest_file)


def display_optimized_squad(result: dict):
    """Display the optimized squad in a formatted way."""
    
    if result.get('status') != 'optimal':
        print(f"❌ Optimization failed: {result.get('status', 'unknown error')}")
        return
    
    squad = result['squad']
    squad_results = result['results']
    
    print(f"\n🎯 OPTIMIZED FPL SQUAD")
    print("="*80)
    print(f"💰 Total Cost: £{squad_results['total_cost']:.1f}m")
    print(f"💸 Budget Remaining: £{squad_results['budget_remaining']:.1f}m")
    print(f"📊 Total Predicted Points: {squad_results['total_objective']:.1f}")
    print(f"⚡ Average Value Ratio: {squad_results['average_value_ratio']:.2f} pts/£m")
    print(f"🌟 Best Value Player: {squad_results['best_value_player']}")
    
    # Position breakdown
    print(f"\n📍 SQUAD BY POSITION:")
    print("-"*80)
    
    positions = {1: 'GOALKEEPERS', 2: 'DEFENDERS', 3: 'MIDFIELDERS', 4: 'FORWARDS'}
    
    for pos_id, pos_name in positions.items():
        pos_players = squad[squad['element_type'] == pos_id].sort_values('predicted_points', ascending=False)
        
        if not pos_players.empty:
            pos_total = squad_results['position_breakdown'][pos_name.title()[:-1]]
            print(f"\n{pos_name} ({len(pos_players)}) - £{pos_total['cost']:.1f}m - {pos_total['objective_total']:.1f} pts")
            print("-" * 60)
            
            for _, player in pos_players.iterrows():
                ownership = player.get('ownership', 0)
                team = player.get('team_short', 'N/A')
                predicted = player.get('predicted_points', 0)
                actual = player.get('total_points', 0)
                
                print(f"  {player['web_name']:<15} ({team}) - "
                      f"£{player['price']:4.1f}m - "
                      f"{predicted:4.1f} pred - "
                      f"{actual:2.0f} actual - "
                      f"{ownership:4.1f}% owned")


def display_starting_xi(xi_result: dict):
    """Display the optimized starting XI."""
    
    if not xi_result:
        print("❌ Starting XI optimization failed")
        return
    
    starting_xi = xi_result['starting_xi']
    formation = xi_result['formation']
    
    print(f"\n⚽ STARTING XI ({formation})")
    print("="*60)
    print(f"🎯 Total Predicted Points: {xi_result['total_predicted_points']:.1f}")
    print(f"👑 Captain Suggestion: {xi_result['captain_suggestion']}")
    
    positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    print(f"\nStarting XI:")
    print("-"*60)
    
    for pos_id in [1, 2, 3, 4]:  # GK, DEF, MID, FWD
        pos_players = starting_xi[starting_xi['element_type'] == pos_id]
        
        if not pos_players.empty:
            pos_name = positions[pos_id]
            print(f"\n{pos_name} ({len(pos_players)}):")
            
            for _, player in pos_players.sort_values('predicted_points', ascending=False).iterrows():
                predicted = player.get('predicted_points', 0)
                team = player.get('team_short', 'N/A')
                is_captain = player['web_name'] == xi_result['captain_suggestion']
                captain_marker = " (C)" if is_captain else ""
                
                print(f"  {player['web_name']:<15} ({team}) - {predicted:4.1f} pts{captain_marker}")
    
    # Bench
    bench = xi_result['bench']
    print(f"\nBench ({len(bench)}):")
    print("-"*30)
    for _, player in bench.sort_values(['element_type', 'predicted_points'], ascending=[True, False]).iterrows():
        predicted = player.get('predicted_points', 0)
        team = player.get('team_short', 'N/A')
        pos = positions[player['element_type']]
        print(f"  {player['web_name']:<15} ({pos}, {team}) - {predicted:4.1f} pts")


def analyze_transfers(current_squad: pd.DataFrame, optimized_squad: pd.DataFrame):
    """Analyze potential transfers between current and optimized squads."""
    
    print(f"\n🔄 TRANSFER ANALYSIS")
    print("="*50)
    
    if current_squad.empty:
        print("No current squad provided for comparison")
        return
    
    current_players = set(current_squad['web_name'].tolist())
    optimized_players = set(optimized_squad['web_name'].tolist())
    
    transfers_out = current_players - optimized_players
    transfers_in = optimized_players - current_players
    
    print(f"📤 Players to transfer OUT ({len(transfers_out)}):")
    for player in transfers_out:
        player_data = current_squad[current_squad['web_name'] == player].iloc[0]
        print(f"  {player} - £{player_data.get('price', 0):.1f}m")
    
    print(f"\n📥 Players to transfer IN ({len(transfers_in)}):")
    for player in transfers_in:
        player_data = optimized_squad[optimized_squad['web_name'] == player].iloc[0]
        predicted = player_data.get('predicted_points', 0)
        print(f"  {player} - £{player_data.get('price', 0):.1f}m - {predicted:.1f} pts")
    
    if len(transfers_out) <= 1:
        print(f"\n✅ Only {len(transfers_out)} transfer needed - within free transfer allowance!")
    else:
        extra_transfers = len(transfers_out) - 1
        cost = extra_transfers * 4  # 4 points per extra transfer
        print(f"\n💰 {extra_transfers} extra transfers needed (cost: {cost} points)")


def main():
    parser = argparse.ArgumentParser(description="FPL Team Optimization")
    parser.add_argument("--formation", default="3-5-2", help="Formation for starting XI")
    parser.add_argument("--budget", type=float, default=100.0, help="Budget in millions")
    parser.add_argument("--min-price", type=float, default=4.0, help="Minimum player price")
    parser.add_argument("--max-price", type=float, default=15.0, help="Maximum player price")
    parser.add_argument("--must-include", nargs="+", help="Players that must be included")
    parser.add_argument("--exclude", nargs="+", help="Players to exclude")
    parser.add_argument("--current-squad", help="CSV file with current squad for transfer analysis")
    parser.add_argument("--use-predictions", action="store_true", default=True, help="Use prediction models")
    
    args = parser.parse_args()
    
    print("🎯 FPL Team Optimization")
    print("="*50)
    
    # Get current FPL data
    print("📡 Fetching current FPL data...")
    api = FPLApi()
    players_df = api.get_players_df()
    
    if players_df.empty:
        print("❌ Failed to load player data")
        return
    
    print(f"📊 Loaded {len(players_df)} players")
    
    # Load predictions if available and requested
    predictions_df = None
    if args.use_predictions:
        predictions_df = load_latest_predictions()
        if not predictions_df.empty:
            print(f"🔮 Loaded predictions for {len(predictions_df)} players")
        else:
            print("⚠️  No predictions found, using actual points")
    
    # Load current squad for comparison
    current_squad = pd.DataFrame()
    if args.current_squad and Path(args.current_squad).exists():
        current_squad = pd.read_csv(args.current_squad)
        print(f"📋 Loaded current squad with {len(current_squad)} players")
    
    # Run optimization
    print(f"\n🚀 Running optimization...")
    print(f"   Formation: {args.formation}")
    print(f"   Budget: £{args.budget}m")
    print(f"   Price range: £{args.min_price}m - £{args.max_price}m")
    
    if args.must_include:
        print(f"   Must include: {', '.join(args.must_include)}")
    if args.exclude:
        print(f"   Excluding: {', '.join(args.exclude)}")
    
    # Set up optimizer
    optimizer = FPLTeamOptimizer()
    
    # Merge predictions if available
    if predictions_df is not None and not predictions_df.empty:
        players_df = players_df.merge(
            predictions_df[['web_name', 'predicted_points']], 
            on='web_name', 
            how='left'
        )
        objective = 'predicted_points'
    else:
        objective = 'total_points'
    
    # Optimize squad
    squad_result = optimizer.optimize_squad(
        players_df,
        objective=objective,
        min_price=args.min_price,
        max_price=args.max_price,
        exclude_players=args.exclude,
        must_include=args.must_include
    )
    
    # Display results
    display_optimized_squad(squad_result)
    
    if squad_result.get('status') == 'optimal':
        # Optimize starting XI
        xi_result = optimizer.optimize_starting_xi(
            squad_result['squad'],
            formation=args.formation,
            objective=objective
        )
        
        display_starting_xi(xi_result)
        
        # Transfer analysis
        if not current_squad.empty:
            analyze_transfers(current_squad, squad_result['squad'])
        
        # Save optimized squad
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "data" / "optimized_squads"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        squad_file = output_dir / f"optimized_squad_{timestamp}.csv"
        xi_file = output_dir / f"starting_xi_{timestamp}.csv"
        
        # Save files
        squad_result['squad'].to_csv(squad_file, index=False)
        print(f"\n💾 Optimized squad saved to: {squad_file}")
        
        if xi_result:
            xi_result['starting_xi'].to_csv(xi_file, index=False)
            print(f"💾 Starting XI saved to: {xi_file}")
    
    print(f"\n✅ FPL optimization completed!")


if __name__ == "__main__":
    main()