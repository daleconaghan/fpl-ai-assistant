#!/usr/bin/env python3
"""
Train FPL prediction models and save for future use.
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
from src.models.fpl_predictor import FPLPlayerPredictor, FPLEnsemblePredictor


def train_single_model(model_type: str = "xgboost", save_model: bool = True):
    """Train a single FPL prediction model."""
    
    print(f"🤖 Training {model_type} FPL prediction model...")
    
    # Get current FPL data
    api = FPLApi()
    players_df = api.get_players_df()
    
    if players_df.empty:
        print("❌ No player data available")
        return None
    
    # Filter active players
    active_players = players_df[players_df['minutes'] > 0].copy()
    print(f"📊 Training on {len(active_players)} active players")
    
    if len(active_players) < 10:
        print("❌ Not enough active players for training")
        return None
    
    # Train model
    predictor = FPLPlayerPredictor(model_type)
    metrics = predictor.train(active_players)
    
    print(f"✅ Model trained!")
    print(f"   R² Score: {metrics['test_r2']:.3f}")
    print(f"   MAE: {metrics['test_mae']:.3f}")
    print(f"   Features: {metrics['n_features']}")
    print(f"   CV Score: {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    if not importance.empty:
        print(f"\n🔍 Top 5 Important Features:")
        for _, row in importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    if save_model:
        models_dir = project_root / "models" / "saved"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = models_dir / f"fpl_{model_type}_{timestamp}.pkl"
        
        predictor.save_model(str(model_file))
        print(f"💾 Model saved to: {model_file}")
        
        return str(model_file)
    
    return predictor


def train_ensemble_model(save_model: bool = True):
    """Train an ensemble of FPL prediction models."""
    
    print("🎯 Training FPL ensemble model...")
    
    # Get current FPL data
    api = FPLApi()
    players_df = api.get_players_df()
    
    if players_df.empty:
        print("❌ No player data available")
        return None
    
    # Filter active players
    active_players = players_df[players_df['minutes'] > 0].copy()
    print(f"📊 Training ensemble on {len(active_players)} active players")
    
    if len(active_players) < 10:
        print("❌ Not enough active players for training")
        return None
    
    # Train ensemble
    ensemble = FPLEnsemblePredictor(["xgboost", "random_forest"])
    all_metrics = ensemble.train(active_players)
    
    print(f"✅ Ensemble trained!")
    
    for model_name, metrics in all_metrics.items():
        print(f"   {model_name}: R² = {metrics['test_r2']:.3f}, MAE = {metrics['test_mae']:.3f}")
    
    print(f"   Final weights: {ensemble.weights}")
    
    # Save ensemble
    if save_model:
        models_dir = project_root / "models" / "saved"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_file = models_dir / f"fpl_ensemble_{timestamp}.pkl"
        
        ensemble.save_ensemble(str(ensemble_file))
        print(f"💾 Ensemble saved to: {ensemble_file}")
        
        return str(ensemble_file)
    
    return ensemble


def generate_predictions(model_path: str = None, top_n: int = 50):
    """Generate predictions for next gameweek."""
    
    print("🔮 Generating FPL predictions...")
    
    # Get current FPL data
    api = FPLApi()
    players_df = api.get_players_df()
    
    if players_df.empty:
        print("❌ No player data available")
        return None
    
    # Load or train model
    if model_path and Path(model_path).exists():
        print(f"📂 Loading model from {model_path}")
        predictor = FPLPlayerPredictor()
        predictor.load_model(model_path)
    else:
        print("🎯 Training new model for predictions...")
        predictor = train_single_model("xgboost", save_model=False)
        if predictor is None:
            return None
    
    # Make predictions
    predictions = predictor.predict(players_df)
    players_df['predicted_points'] = predictions
    players_df['predicted_points'] = players_df['predicted_points'].clip(lower=0)
    
    # Filter and sort
    active_predictions = players_df[players_df['minutes'] > 0].copy()
    top_predictions = active_predictions.nlargest(top_n, 'predicted_points')
    
    print(f"\n🌟 Top {min(top_n, len(top_predictions))} Predicted Performers:")
    print("="*80)
    
    for i, (_, player) in enumerate(top_predictions.iterrows(), 1):
        position = player.get('position', 'N/A')
        team = player.get('team_short', 'N/A')
        price = player.get('price', 0)
        predicted = player['predicted_points']
        actual = player.get('total_points', 0)
        ownership = player.get('ownership', 0)
        
        print(f"{i:2d}. {player['web_name']:<15} ({position}, {team}) - "
              f"Predicted: {predicted:4.1f} pts, "
              f"Actual: {actual:2.0f} pts, "
              f"£{price:4.1f}m, "
              f"{ownership:4.1f}% owned")
    
    # Save predictions
    predictions_dir = project_root / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file = predictions_dir / f"fpl_predictions_{timestamp}.csv"
    
    # Save with key columns
    save_columns = [
        'web_name', 'position', 'team_short', 'price', 'total_points', 
        'predicted_points', 'ownership', 'form_numeric', 'minutes'
    ]
    available_columns = [col for col in save_columns if col in top_predictions.columns]
    
    top_predictions[available_columns].to_csv(pred_file, index=False)
    print(f"\n💾 Predictions saved to: {pred_file}")
    
    return top_predictions


def analyze_predictions(predictions_df: pd.DataFrame):
    """Analyze prediction quality and insights."""
    
    print("\n📊 Prediction Analysis:")
    print("="*50)
    
    # Position breakdown
    if 'position' in predictions_df.columns:
        position_counts = predictions_df['position'].value_counts()
        print(f"🏃 Top performers by position:")
        for pos, count in position_counts.items():
            avg_pred = predictions_df[predictions_df['position'] == pos]['predicted_points'].mean()
            print(f"   {pos}: {count} players, avg {avg_pred:.1f} predicted pts")
    
    # Price vs Performance
    if 'price' in predictions_df.columns and 'predicted_points' in predictions_df.columns:
        predictions_df['value_prediction'] = predictions_df['predicted_points'] / predictions_df['price']
        best_value = predictions_df.nlargest(5, 'value_prediction')
        
        print(f"\n💎 Best Value Predictions (points per £m):")
        for _, player in best_value.iterrows():
            print(f"   {player['web_name']} - {player['value_prediction']:.2f} pts/£m "
                  f"({player['predicted_points']:.1f} pts @ £{player['price']:.1f}m)")
    
    # Ownership insights
    if 'ownership' in predictions_df.columns:
        low_owned = predictions_df[predictions_df['ownership'] < 5.0].nlargest(5, 'predicted_points')
        
        if not low_owned.empty:
            print(f"\n🔥 Differential Picks (<5% ownership):")
            for _, player in low_owned.iterrows():
                print(f"   {player['web_name']} - {player['predicted_points']:.1f} pts, "
                      f"{player['ownership']:.1f}% owned")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPL Model Training and Predictions")
    parser.add_argument("--model", choices=["xgboost", "random_forest", "ensemble"], 
                       default="xgboost", help="Model type to train")
    parser.add_argument("--predict", action="store_true", help="Generate predictions")
    parser.add_argument("--top", type=int, default=50, help="Number of top predictions to show")
    parser.add_argument("--model-path", help="Path to saved model for predictions")
    
    args = parser.parse_args()
    
    if args.predict:
        # Generate predictions
        predictions = generate_predictions(args.model_path, args.top)
        if predictions is not None:
            analyze_predictions(predictions)
    
    elif args.model == "ensemble":
        # Train ensemble
        train_ensemble_model()
    
    else:
        # Train single model
        train_single_model(args.model)
    
    print("\n✅ FPL model training completed!")