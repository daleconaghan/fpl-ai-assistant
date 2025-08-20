"""
FPL Player Performance Prediction Models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FPLPlayerPredictor:
    """Machine Learning models to predict FPL player performance."""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for FPL prediction."""
        
        feature_df = df.copy()
        
        # Basic features
        basic_features = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'own_goals', 'penalties_saved',
            'penalties_missed', 'yellow_cards', 'red_cards',
            'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat'
        ]
        
        # Ensure basic features exist
        for feature in basic_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0
        
        # Derived features
        feature_df['points_per_minute'] = feature_df['total_points'] / (feature_df['minutes'] + 1)
        feature_df['goals_per_game'] = feature_df['goals_scored'] / (feature_df['minutes'] / 90 + 0.1)
        feature_df['assists_per_game'] = feature_df['assists'] / (feature_df['minutes'] / 90 + 0.1)
        feature_df['saves_per_game'] = feature_df['saves'] / (feature_df['minutes'] / 90 + 0.1)
        
        # Position encoding
        if 'element_type' in feature_df.columns:
            feature_df['is_goalkeeper'] = (feature_df['element_type'] == 1).astype(int)
            feature_df['is_defender'] = (feature_df['element_type'] == 2).astype(int)
            feature_df['is_midfielder'] = (feature_df['element_type'] == 3).astype(int)
            feature_df['is_forward'] = (feature_df['element_type'] == 4).astype(int)
        
        # Price and ownership features
        if 'now_cost' in feature_df.columns:
            feature_df['price'] = feature_df['now_cost'] / 10.0
        if 'selected_by_percent' in feature_df.columns:
            feature_df['ownership'] = pd.to_numeric(feature_df['selected_by_percent'], errors='coerce').fillna(0)
        
        # Value metrics
        if 'price' in feature_df.columns and 'total_points' in feature_df.columns:
            feature_df['value'] = feature_df['total_points'] / (feature_df['price'] + 0.1)
        
        # Form metrics and advanced stats - convert to numeric
        numeric_columns = ['form', 'influence', 'creativity', 'threat', 'ict_index']
        for col in numeric_columns:
            if col in feature_df.columns:
                feature_df[f'{col}_numeric'] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
        
        # Team strength proxy (average team performance)
        if 'team' in feature_df.columns:
            team_stats = feature_df.groupby('team')['total_points'].mean()
            feature_df['team_strength'] = feature_df['team'].map(team_stats)
        
        return feature_df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for prediction."""
        
        potential_features = [
            # Basic performance
            'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'saves', 'bonus', 'bps',
            
            # Advanced metrics (converted to numeric)
            'influence_numeric', 'creativity_numeric', 'threat_numeric',
            'points_per_minute', 'goals_per_game', 'assists_per_game',
            'saves_per_game', 'value',
            
            # Position indicators
            'is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward',
            
            # Market metrics
            'price', 'ownership', 'form_numeric', 'team_strength',
            
            # Penalties and cards
            'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards'
        ]
        
        # Only use features that exist in the dataframe
        available_features = [f for f in potential_features if f in df.columns]
        
        # Remove features with too many missing values
        feature_df = df[available_features]
        missing_ratios = feature_df.isnull().sum() / len(feature_df)
        valid_features = missing_ratios[missing_ratios < 0.5].index.tolist()
        
        logger.info(f"Selected {len(valid_features)} features for prediction")
        return valid_features
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'total_points') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        
        # Engineer features
        feature_df = self.engineer_features(df)
        
        # Select features
        self.feature_columns = self.select_features(feature_df)
        
        # Prepare X and y
        X = feature_df[self.feature_columns].fillna(0)
        y = feature_df[target_col]
        
        # Remove rows where target is missing
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Prepared data: {len(X)} samples, {len(self.feature_columns)} features")
        return X, y
    
    def train(self, df: pd.DataFrame, target_col: str = 'total_points', test_size: float = 0.2) -> Dict:
        """Train the FPL prediction model."""
        
        X, y = self.prepare_data(df, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'n_samples': len(X),
            'n_features': len(self.feature_columns)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        self.is_trained = True
        
        logger.info(f"Model trained successfully!")
        logger.info(f"Test R²: {metrics['test_r2']:.3f}")
        logger.info(f"Test MAE: {metrics['test_mae']:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions for new data."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Engineer features
        feature_df = self.engineer_features(df)
        
        # Use same features as training
        X = feature_df[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")


class FPLEnsemblePredictor:
    """Ensemble of multiple FPL prediction models."""
    
    def __init__(self, model_types: List[str] = None):
        if model_types is None:
            model_types = ["xgboost", "random_forest"]
        
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
        for model_type in model_types:
            self.models[model_type] = FPLPlayerPredictor(model_type)
            self.weights[model_type] = 1.0 / len(model_types)  # Equal weights initially
    
    def train(self, df: pd.DataFrame, target_col: str = 'total_points') -> Dict:
        """Train all models in the ensemble."""
        
        all_metrics = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            metrics = model.train(df, target_col)
            all_metrics[model_name] = metrics
            
            # Weight models based on their performance
            self.weights[model_name] = metrics['test_r2']
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.is_trained = True
        
        logger.info("Ensemble training completed!")
        logger.info(f"Model weights: {self.weights}")
        
        return all_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = np.zeros(len(df))
        
        for model_name, model in self.models.items():
            model_pred = model.predict(df)
            predictions += self.weights[model_name] * model_pred
        
        return predictions
    
    def save_ensemble(self, filepath: str):
        """Save the entire ensemble."""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
        
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")


def predict_next_gameweek(players_df: pd.DataFrame, model_path: str = None) -> pd.DataFrame:
    """Predict points for the next gameweek."""
    
    if model_path and Path(model_path).exists():
        # Load existing model
        predictor = FPLPlayerPredictor()
        predictor.load_model(model_path)
    else:
        # Train new model on current data
        predictor = FPLPlayerPredictor("xgboost")
        predictor.train(players_df)
    
    # Make predictions
    predictions = predictor.predict(players_df)
    
    # Create results dataframe
    results_df = players_df.copy()
    results_df['predicted_points'] = predictions
    results_df['predicted_points'] = results_df['predicted_points'].clip(lower=0)  # No negative points
    
    # Sort by predicted points
    results_df = results_df.sort_values('predicted_points', ascending=False)
    
    return results_df


if __name__ == "__main__":
    # Test the predictor with current FPL data
    print("🤖 Testing FPL Player Predictor...")
    
    # Load current players data
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.data.fpl_api import FPLApi
    
    api = FPLApi()
    players_df = api.get_players_df()
    
    if not players_df.empty:
        print(f"📊 Loaded {len(players_df)} players")
        
        # Filter players with some minutes played
        active_players = players_df[players_df['minutes'] > 0].copy()
        print(f"🏃 {len(active_players)} active players (with minutes)")
        
        if len(active_players) > 10:
            # Train predictor
            predictor = FPLPlayerPredictor("xgboost")
            metrics = predictor.train(active_players)
            
            print(f"✅ Model trained with R² = {metrics['test_r2']:.3f}")
            
            # Get predictions for top players
            predictions = predictor.predict(active_players)
            active_players['predicted_points'] = predictions
            
            # Show top predicted performers
            top_predicted = active_players.nlargest(10, 'predicted_points')
            print("\n🌟 Top 10 Predicted Performers:")
            for _, player in top_predicted.iterrows():
                print(f"  {player['web_name']} ({player.get('position', 'N/A')}) - "
                      f"Predicted: {player['predicted_points']:.1f}, "
                      f"Actual: {player['total_points']:.1f}")
            
            # Feature importance
            importance = predictor.get_feature_importance()
            print(f"\n🔍 Top 5 Important Features:")
            for _, row in importance.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
    
    print("\n✅ FPL Predictor test completed!")