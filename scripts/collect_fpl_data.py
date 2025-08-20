#!/usr/bin/env python3
"""
FPL Data Collection Script
Collects current player data, fixtures, and saves to local files.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.fpl_api import FPLApi


def collect_all_fpl_data():
    """Collect all FPL data and save to files."""
    
    print("🏆 FPL Data Collection Starting...")
    
    # Initialize API
    api = FPLApi()
    
    # Create data directories
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Collect Players Data
        print("\n👤 Collecting player data...")
        players_df = api.get_players_df()
        
        if not players_df.empty:
            players_file = data_dir / f"fpl_players_{datetime.now().strftime('%Y%m%d')}.csv"
            players_df.to_csv(players_file, index=False)
            print(f"✅ Saved {len(players_df)} players to {players_file}")
            
            # Show top players by position
            positions = players_df['position_short'].unique()
            print("\n🌟 Top players by position:")
            for pos in positions:
                top_player = players_df[players_df['position_short'] == pos].iloc[0]
                print(f"  {pos}: {top_player['web_name']} ({top_player['team_short']}) - {top_player['total_points']} pts, £{top_player['price']:.1f}m")
        
        # 2. Collect Fixtures Data
        print("\n📅 Collecting fixtures data...")
        fixtures_df = api.get_fixtures()
        
        if not fixtures_df.empty:
            fixtures_file = data_dir / f"fpl_fixtures_{datetime.now().strftime('%Y%m%d')}.csv"
            fixtures_df.to_csv(fixtures_file, index=False)
            print(f"✅ Saved {len(fixtures_df)} fixtures to {fixtures_file}")
        
        # 3. Get Current Gameweek Info
        print("\n🗓️ Getting current gameweek...")
        current_gw = api.get_current_gameweek()
        print(f"✅ Current gameweek: {current_gw}")
        
        # 4. Collect sample player histories for top players
        print("\n📈 Collecting top player histories...")
        top_players = players_df.nlargest(10, 'total_points')
        
        all_histories = []
        for _, player in top_players.iterrows():
            player_id = player['id']
            history = api.get_player_history(player_id)
            
            if not history.empty:
                all_histories.append(history)
                print(f"  ✅ {player['web_name']}: {len(history)} gameweeks")
        
        if all_histories:
            combined_history = pd.concat(all_histories, ignore_index=True)
            history_file = data_dir / f"fpl_player_histories_{datetime.now().strftime('%Y%m%d')}.csv"
            combined_history.to_csv(history_file, index=False)
            print(f"✅ Saved combined history data to {history_file}")
        
        # 5. Generate Summary Report
        print("\n📊 Generating summary report...")
        
        summary = {
            'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_gameweek': current_gw,
            'total_players': len(players_df),
            'total_fixtures': len(fixtures_df),
            'player_histories_collected': len(all_histories)
        }
        
        # Most expensive team
        if not players_df.empty:
            most_expensive = players_df.nlargest(5, 'price')[['web_name', 'team_short', 'position', 'price', 'total_points']]
            
            print("\n💰 Most expensive players:")
            for _, player in most_expensive.iterrows():
                print(f"  £{player['price']:.1f}m - {player['web_name']} ({player['position']}, {player['team_short']}) - {player['total_points']} pts")
        
        # Best value players (points per million)
        if not players_df.empty:
            players_df['value'] = players_df['total_points'] / players_df['price']
            best_value = players_df[players_df['total_points'] > 20].nlargest(5, 'value')[['web_name', 'team_short', 'position', 'price', 'total_points', 'value']]
            
            print("\n💎 Best value players (min 20 points):")
            for _, player in best_value.iterrows():
                print(f"  {player['value']:.1f} pts/£m - {player['web_name']} ({player['position']}, {player['team_short']}) - {player['total_points']} pts @ £{player['price']:.1f}m")
        
        print(f"\n🎉 FPL data collection completed successfully!")
        print(f"📁 Data saved to: {data_dir}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_player_lookup(player_name: str):
    """Quick lookup of a specific player."""
    api = FPLApi()
    
    player_info = api.get_player_summary(player_name)
    
    if player_info:
        print(f"\n👤 {player_info['name']} ({player_info['position']}, {player_info['team']})")
        print(f"💰 Price: £{player_info['price']:.1f}m")
        print(f"🏆 Total Points: {player_info['total_points']}")
        print(f"📊 Ownership: {player_info['ownership']:.1f}%")
        print(f"📈 Form: {player_info['form']}")
    else:
        print(f"❌ Player '{player_name}' not found")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FPL Data Collection")
    parser.add_argument("--player", help="Look up specific player")
    parser.add_argument("--collect", action="store_true", help="Collect all FPL data")
    
    args = parser.parse_args()
    
    if args.player:
        quick_player_lookup(args.player)
    elif args.collect or len(sys.argv) == 1:
        collect_all_fpl_data()
    else:
        parser.print_help()