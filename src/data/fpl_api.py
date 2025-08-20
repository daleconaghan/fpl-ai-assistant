"""
FPL Official API client for data collection.
"""

import requests
import pandas as pd
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FPLApi:
    """Client for interacting with the official FPL API."""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Cache for bootstrap data
        self._bootstrap_cache = None
        self._cache_time = None
    
    def get_bootstrap_data(self, force_refresh: bool = False) -> Dict:
        """
        Get the main bootstrap data containing all players, teams, and game settings.
        This is the most important endpoint - contains current player prices, stats, etc.
        """
        # Use cache if available and not expired (1 hour)
        if (not force_refresh and self._bootstrap_cache and self._cache_time and 
            (datetime.now() - self._cache_time).seconds < 3600):
            return self._bootstrap_cache
        
        try:
            response = self.session.get(f"{self.base_url}/bootstrap-static/")
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the data
            self._bootstrap_cache = data
            self._cache_time = datetime.now()
            
            logger.info("Successfully fetched bootstrap data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching bootstrap data: {e}")
            return {}
    
    def get_players_df(self) -> pd.DataFrame:
        """Get all players as a pandas DataFrame with cleaned column names."""
        bootstrap = self.get_bootstrap_data()
        
        if 'elements' not in bootstrap:
            logger.error("No elements (players) found in bootstrap data")
            return pd.DataFrame()
        
        players_df = pd.DataFrame(bootstrap['elements'])
        
        # Add team and position names
        teams_df = pd.DataFrame(bootstrap['element_types'])
        positions_df = pd.DataFrame(bootstrap['element_stats'])
        
        # Merge team names
        if 'teams' in bootstrap:
            teams_lookup = pd.DataFrame(bootstrap['teams'])
            players_df = players_df.merge(
                teams_lookup[['id', 'name', 'short_name']], 
                left_on='team', 
                right_on='id', 
                suffixes=('', '_team')
            ).drop('id_team', axis=1)
            players_df.rename(columns={'name_team': 'team_name', 'short_name': 'team_short'}, inplace=True)
        
        # Merge position names
        if 'element_types' in bootstrap:
            position_lookup = pd.DataFrame(bootstrap['element_types'])
            players_df = players_df.merge(
                position_lookup[['id', 'singular_name', 'singular_name_short']], 
                left_on='element_type', 
                right_on='id', 
                suffixes=('', '_pos')
            ).drop('id_pos', axis=1)
            players_df.rename(columns={
                'singular_name': 'position', 
                'singular_name_short': 'position_short'
            }, inplace=True)
        
        # Clean up key columns
        if 'now_cost' in players_df.columns:
            players_df['price'] = players_df['now_cost'] / 10.0  # Convert to £m
        
        if 'selected_by_percent' in players_df.columns:
            players_df['ownership'] = pd.to_numeric(players_df['selected_by_percent'], errors='coerce')
        
        # Sort by total points
        if 'total_points' in players_df.columns:
            players_df = players_df.sort_values('total_points', ascending=False)
        
        logger.info(f"Processed {len(players_df)} players")
        return players_df
    
    def get_fixtures(self) -> pd.DataFrame:
        """Get all fixtures data."""
        try:
            response = self.session.get(f"{self.base_url}/fixtures/")
            response.raise_for_status()
            
            fixtures_df = pd.DataFrame(response.json())
            
            # Convert dates
            if 'kickoff_time' in fixtures_df.columns:
                fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'])
            
            logger.info(f"Fetched {len(fixtures_df)} fixtures")
            return fixtures_df
            
        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")
            return pd.DataFrame()
    
    def get_player_history(self, player_id: int) -> pd.DataFrame:
        """Get detailed history for a specific player."""
        try:
            response = self.session.get(f"{self.base_url}/element-summary/{player_id}/")
            response.raise_for_status()
            
            data = response.json()
            
            # Get this season's history
            if 'history' in data:
                history_df = pd.DataFrame(data['history'])
                
                # Add player ID for reference
                history_df['player_id'] = player_id
                
                logger.info(f"Fetched history for player {player_id}: {len(history_df)} gameweeks")
                return history_df
            else:
                logger.warning(f"No history found for player {player_id}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching player {player_id} history: {e}")
            return pd.DataFrame()
    
    def get_gameweek_live(self, gameweek: int) -> Dict:
        """Get live data for a specific gameweek."""
        try:
            response = self.session.get(f"{self.base_url}/event/{gameweek}/live/")
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched live data for gameweek {gameweek}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching gameweek {gameweek} live data: {e}")
            return {}
    
    def get_current_gameweek(self) -> int:
        """Get the current gameweek number."""
        bootstrap = self.get_bootstrap_data()
        
        if 'events' in bootstrap:
            events = bootstrap['events']
            current_gw = None
            
            for event in events:
                if event.get('is_current', False):
                    current_gw = event['id']
                    break
            
            if current_gw:
                logger.info(f"Current gameweek: {current_gw}")
                return current_gw
            else:
                # If no current gameweek, find the next one
                for event in events:
                    if event.get('is_next', False):
                        logger.info(f"Next gameweek: {event['id']}")
                        return event['id']
        
        logger.warning("Could not determine current gameweek, defaulting to 1")
        return 1
    
    def get_top_players_by_position(self, position: str, limit: int = 10) -> pd.DataFrame:
        """Get top players by position based on total points."""
        players_df = self.get_players_df()
        
        if players_df.empty:
            return pd.DataFrame()
        
        # Filter by position
        position_players = players_df[players_df['position_short'] == position.upper()]
        
        # Get top players by total points
        top_players = position_players.nlargest(limit, 'total_points')
        
        # Select key columns
        columns = [
            'web_name', 'team_short', 'position', 'price', 'total_points', 
            'ownership', 'points_per_game', 'form'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in columns if col in top_players.columns]
        
        return top_players[available_columns].reset_index(drop=True)
    
    def get_player_summary(self, player_name: str) -> Optional[Dict]:
        """Get summary info for a player by name."""
        players_df = self.get_players_df()
        
        if players_df.empty:
            return None
        
        # Try to find player by web name
        player_match = players_df[players_df['web_name'].str.contains(player_name, case=False, na=False)]
        
        if player_match.empty:
            # Try by full name
            if 'first_name' in players_df.columns and 'second_name' in players_df.columns:
                player_match = players_df[
                    (players_df['first_name'].str.contains(player_name, case=False, na=False)) |
                    (players_df['second_name'].str.contains(player_name, case=False, na=False))
                ]
        
        if not player_match.empty:
            player = player_match.iloc[0]
            return {
                'id': player.get('id'),
                'name': player.get('web_name'),
                'team': player.get('team_short'),
                'position': player.get('position'),
                'price': player.get('price'),
                'total_points': player.get('total_points'),
                'ownership': player.get('ownership'),
                'form': player.get('form')
            }
        
        return None


# Utility functions
def save_players_data(filepath: str = "data/raw/fpl_players.csv"):
    """Save current players data to CSV."""
    api = FPLApi()
    players_df = api.get_players_df()
    
    if not players_df.empty:
        players_df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(players_df)} players to {filepath}")
    else:
        logger.error("No players data to save")


def save_fixtures_data(filepath: str = "data/raw/fpl_fixtures.csv"):
    """Save fixtures data to CSV."""
    api = FPLApi()
    fixtures_df = api.get_fixtures()
    
    if not fixtures_df.empty:
        fixtures_df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(fixtures_df)} fixtures to {filepath}")
    else:
        logger.error("No fixtures data to save")


if __name__ == "__main__":
    # Test the API
    api = FPLApi()
    
    print("Testing FPL API...")
    
    # Test players data
    players = api.get_players_df()
    print(f"Found {len(players)} players")
    
    if not players.empty:
        print("\nTop 5 players by total points:")
        print(players[['web_name', 'team_short', 'position', 'total_points', 'price']].head())
    
    # Test current gameweek
    current_gw = api.get_current_gameweek()
    print(f"\nCurrent gameweek: {current_gw}")
    
    # Test player search
    haaland = api.get_player_summary("Haaland")
    if haaland:
        print(f"\nHaaland info: {haaland}")
    
    print("\nFPL API test completed!")