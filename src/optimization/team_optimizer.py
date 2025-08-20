"""
FPL Team Optimization using Linear Programming
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import yaml

try:
    from pulp import *
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp"])
    from pulp import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FPLTeamOptimizer:
    """Optimize FPL team selection using linear programming."""
    
    def __init__(self, config_path: str = None):
        """Initialize the team optimizer with FPL rules."""
        
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "fpl_config.yaml"
        
        # Load FPL configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # FPL constraints
        self.budget = self.config['fpl_rules']['budget']  # £100m
        self.squad_size = self.config['fpl_rules']['squad_size']  # 15 players
        self.max_per_team = self.config['fpl_rules']['team_constraints']['max_players_per_team']  # 3
        
        # Position requirements
        positions = self.config['fpl_rules']['positions']
        self.position_limits = {
            1: positions['goalkeeper'],  # GK: min 2, max 2
            2: positions['defender'],    # DEF: min 5, max 5  
            3: positions['midfielder'],  # MID: min 5, max 5
            4: positions['forward']      # FWD: min 3, max 3
        }
        
        logger.info("FPL Team Optimizer initialized")
        logger.info(f"Budget: £{self.budget}m, Squad size: {self.squad_size}")
    
    def optimize_squad(self, 
                      players_df: pd.DataFrame, 
                      objective: str = "predicted_points",
                      min_price: float = 4.0,
                      max_price: float = 15.0,
                      exclude_players: List[str] = None,
                      must_include: List[str] = None) -> Dict:
        """
        Optimize FPL squad selection.
        
        Args:
            players_df: DataFrame with player data including predictions
            objective: Column to optimize (e.g., 'predicted_points', 'total_points')
            min_price: Minimum player price to consider
            max_price: Maximum player price to consider  
            exclude_players: List of player names to exclude
            must_include: List of player names that must be included
        
        Returns:
            Dict with optimized squad and optimization results
        """
        
        logger.info(f"Optimizing FPL squad with objective: {objective}")
        
        # Prepare data
        df = self._prepare_optimization_data(
            players_df, objective, min_price, max_price, exclude_players
        )
        
        if df.empty:
            raise ValueError("No players available for optimization")
        
        # Create optimization problem
        prob = LpProblem("FPL_Squad_Optimization", LpMaximize)
        
        # Decision variables - binary selection for each player
        player_vars = {}
        for idx, player in df.iterrows():
            player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')
        
        # Objective function - maximize predicted points
        prob += lpSum([df.loc[idx, objective] * player_vars[idx] for idx in df.index])
        
        # Constraints
        self._add_constraints(prob, df, player_vars, must_include)
        
        # Solve optimization
        logger.info("Solving optimization problem...")
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract results
        if prob.status == 1:  # Optimal solution found
            selected_players = self._extract_solution(df, player_vars)
            results = self._analyze_solution(selected_players, objective)
            
            logger.info(f"✅ Optimization successful!")
            logger.info(f"Total {objective}: {results['total_objective']:.1f}")
            logger.info(f"Total cost: £{results['total_cost']:.1f}m")
            logger.info(f"Budget remaining: £{results['budget_remaining']:.1f}m")
            
            return {
                'squad': selected_players,
                'results': results,
                'status': 'optimal',
                'objective_value': value(prob.objective)
            }
        else:
            logger.error(f"Optimization failed with status: {LpStatus[prob.status]}")
            return {
                'squad': pd.DataFrame(),
                'results': {},
                'status': 'failed',
                'objective_value': 0
            }
    
    def _prepare_optimization_data(self, 
                                 players_df: pd.DataFrame,
                                 objective: str,
                                 min_price: float,
                                 max_price: float,
                                 exclude_players: List[str] = None) -> pd.DataFrame:
        """Prepare and filter data for optimization."""
        
        df = players_df.copy()
        
        # Ensure required columns exist
        required_cols = ['web_name', 'element_type', 'team', 'price', objective]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Try alternative column names
            if 'now_cost' in df.columns and 'price' not in df.columns:
                df['price'] = df['now_cost'] / 10.0
            
            # Check again
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter by price range
        df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
        
        # Filter out players with missing objective values
        df = df[df[objective].notna()]
        
        # Exclude specific players
        if exclude_players:
            df = df[~df['web_name'].isin(exclude_players)]
        
        # Only include players with some game time or high predicted value
        if 'minutes' in df.columns:
            df = df[(df['minutes'] > 0) | (df[objective] > df[objective].quantile(0.8))]
        
        logger.info(f"Prepared {len(df)} players for optimization")
        return df.reset_index(drop=True)
    
    def _add_constraints(self, 
                        prob: LpProblem, 
                        df: pd.DataFrame, 
                        player_vars: Dict,
                        must_include: List[str] = None):
        """Add FPL constraints to the optimization problem."""
        
        # Budget constraint
        prob += lpSum([df.loc[idx, 'price'] * player_vars[idx] for idx in df.index]) <= self.budget
        
        # Squad size constraint
        prob += lpSum([player_vars[idx] for idx in df.index]) == self.squad_size
        
        # Position constraints
        for position_id, limits in self.position_limits.items():
            position_players = df[df['element_type'] == position_id].index
            if len(position_players) > 0:
                prob += lpSum([player_vars[idx] for idx in position_players]) >= limits['min']
                prob += lpSum([player_vars[idx] for idx in position_players]) <= limits['max']
        
        # Team constraints (max 3 players per team)
        for team_id in df['team'].unique():
            team_players = df[df['team'] == team_id].index
            if len(team_players) > 0:
                prob += lpSum([player_vars[idx] for idx in team_players]) <= self.max_per_team
        
        # Must include constraints
        if must_include:
            for player_name in must_include:
                player_indices = df[df['web_name'] == player_name].index
                if len(player_indices) > 0:
                    prob += lpSum([player_vars[idx] for idx in player_indices]) >= 1
                else:
                    logger.warning(f"Must-include player '{player_name}' not found in available players")
    
    def _extract_solution(self, df: pd.DataFrame, player_vars: Dict) -> pd.DataFrame:
        """Extract the selected players from the optimization solution."""
        
        selected_indices = []
        for idx, var in player_vars.items():
            if var.varValue == 1:
                selected_indices.append(idx)
        
        selected_players = df.loc[selected_indices].copy()
        selected_players = selected_players.sort_values(['element_type', 'price'], ascending=[True, False])
        
        return selected_players.reset_index(drop=True)
    
    def _analyze_solution(self, squad_df: pd.DataFrame, objective: str) -> Dict:
        """Analyze the optimized squad solution."""
        
        results = {
            'total_cost': squad_df['price'].sum(),
            'budget_remaining': self.budget - squad_df['price'].sum(),
            'total_objective': squad_df[objective].sum(),
            'squad_size': len(squad_df)
        }
        
        # Position breakdown
        position_names = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}
        results['position_breakdown'] = {}
        
        for pos_id, pos_name in position_names.items():
            pos_players = squad_df[squad_df['element_type'] == pos_id]
            results['position_breakdown'][pos_name] = {
                'count': len(pos_players),
                'cost': pos_players['price'].sum(),
                'objective_total': pos_players[objective].sum()
            }
        
        # Team distribution
        team_counts = squad_df['team'].value_counts()
        results['team_distribution'] = team_counts.to_dict()
        
        # Value metrics
        squad_df['value_ratio'] = squad_df[objective] / squad_df['price']
        results['average_value_ratio'] = squad_df['value_ratio'].mean()
        results['best_value_player'] = squad_df.loc[squad_df['value_ratio'].idxmax(), 'web_name']
        
        return results
    
    def optimize_starting_xi(self, 
                           squad_df: pd.DataFrame,
                           formation: str = "3-5-2",
                           objective: str = "predicted_points") -> Dict:
        """
        Optimize starting XI selection from the squad.
        
        Args:
            squad_df: DataFrame with the 15-player squad
            formation: Formation string (e.g., "3-5-2", "4-4-2", "4-3-3")
            objective: Column to optimize
        
        Returns:
            Dict with starting XI and bench players
        """
        
        logger.info(f"Optimizing starting XI with formation: {formation}")
        
        # Parse formation
        formation_parts = formation.split('-')
        if len(formation_parts) != 3:
            raise ValueError("Formation must be in format 'DEF-MID-FWD' (e.g., '3-5-2')")
        
        required_positions = {
            1: 1,  # Always 1 GK
            2: int(formation_parts[0]),  # Defenders
            3: int(formation_parts[1]),  # Midfielders
            4: int(formation_parts[2])   # Forwards
        }
        
        # Create optimization problem for starting XI
        prob = LpProblem("FPL_Starting_XI", LpMaximize)
        
        # Decision variables
        xi_vars = {}
        for idx, player in squad_df.iterrows():
            xi_vars[idx] = LpVariable(f"xi_{idx}", cat='Binary')
        
        # Objective function
        prob += lpSum([squad_df.loc[idx, objective] * xi_vars[idx] for idx in squad_df.index])
        
        # Constraints
        # Exactly 11 players in starting XI
        prob += lpSum([xi_vars[idx] for idx in squad_df.index]) == 11
        
        # Position requirements for formation
        for position_id, required_count in required_positions.items():
            position_players = squad_df[squad_df['element_type'] == position_id].index
            if len(position_players) > 0:
                prob += lpSum([xi_vars[idx] for idx in position_players]) == required_count
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            # Extract starting XI
            starting_xi_indices = [idx for idx, var in xi_vars.items() if var.varValue == 1]
            starting_xi = squad_df.loc[starting_xi_indices].copy()
            bench = squad_df[~squad_df.index.isin(starting_xi_indices)].copy()
            
            # Sort by position for display
            starting_xi = starting_xi.sort_values(['element_type', objective], ascending=[True, False])
            bench = bench.sort_values(['element_type', objective], ascending=[True, False])
            
            results = {
                'starting_xi': starting_xi.reset_index(drop=True),
                'bench': bench.reset_index(drop=True),
                'formation': formation,
                'total_predicted_points': starting_xi[objective].sum(),
                'captain_suggestion': starting_xi.loc[starting_xi[objective].idxmax(), 'web_name']
            }
            
            logger.info(f"✅ Starting XI optimized!")
            logger.info(f"Total {objective}: {results['total_predicted_points']:.1f}")
            logger.info(f"Captain suggestion: {results['captain_suggestion']}")
            
            return results
        else:
            logger.error("Starting XI optimization failed")
            return {}


def optimize_fpl_team(players_df: pd.DataFrame, 
                     predictions_df: pd.DataFrame = None,
                     formation: str = "3-5-2") -> Dict:
    """
    Complete FPL team optimization pipeline.
    
    Args:
        players_df: All player data
        predictions_df: Optional predictions data to merge
        formation: Formation for starting XI
    
    Returns:
        Complete optimization results
    """
    
    # Merge predictions if provided
    if predictions_df is not None:
        # Ensure we have predicted_points column
        if 'predicted_points' not in players_df.columns:
            players_df = players_df.merge(
                predictions_df[['web_name', 'predicted_points']], 
                on='web_name', 
                how='left'
            )
    
    # Use actual points if no predictions available
    objective = 'predicted_points' if 'predicted_points' in players_df.columns else 'total_points'
    
    # Initialize optimizer
    optimizer = FPLTeamOptimizer()
    
    # Optimize squad
    squad_result = optimizer.optimize_squad(players_df, objective=objective)
    
    if squad_result['status'] == 'optimal':
        # Optimize starting XI
        xi_result = optimizer.optimize_starting_xi(
            squad_result['squad'], 
            formation=formation,
            objective=objective
        )
        
        return {
            'squad_optimization': squad_result,
            'starting_xi_optimization': xi_result,
            'formation': formation,
            'objective_used': objective
        }
    else:
        return squad_result


if __name__ == "__main__":
    # Test the optimizer
    print("🎯 Testing FPL Team Optimizer...")
    
    # Load test data
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.data.fpl_api import FPLApi
    
    api = FPLApi()
    players_df = api.get_players_df()
    
    if not players_df.empty:
        print(f"📊 Loaded {len(players_df)} players")
        
        # Test squad optimization
        optimizer = FPLTeamOptimizer()
        result = optimizer.optimize_squad(players_df, objective='total_points')
        
        if result['status'] == 'optimal':
            squad = result['squad']
            print(f"\n✅ Optimized Squad (£{result['results']['total_cost']:.1f}m):")
            print("="*80)
            
            # Group by position
            positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            for pos_id, pos_name in positions.items():
                pos_players = squad[squad['element_type'] == pos_id]
                if not pos_players.empty:
                    print(f"\n{pos_name} ({len(pos_players)}):")
                    for _, player in pos_players.iterrows():
                        print(f"  {player['web_name']} - £{player['price']:.1f}m - {player.get('total_points', 0)} pts")
            
            # Test starting XI optimization
            xi_result = optimizer.optimize_starting_xi(squad, formation="3-5-2")
            if xi_result:
                print(f"\n🌟 Starting XI ({xi_result['formation']}):")
                print("="*50)
                for _, player in xi_result['starting_xi'].iterrows():
                    print(f"  {player['web_name']} ({positions[player['element_type']]}) - {player.get('total_points', 0)} pts")
                print(f"\nCaptain suggestion: {xi_result['captain_suggestion']}")
        
        else:
            print("❌ Optimization failed")
    
    print("\n✅ FPL Team Optimizer test completed!")