import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Uses AutoGluon to automatically train and compare multiple models
    to find the best one for our churn prediction task.
    """
    print("--- Starting Automated Model Comparison with AutoGluon ---")
    
    # Load the processed data
    try:
        df = pd.read_csv('data/processed_user_features.csv')
    except FileNotFoundError:
        print("Error: 'data/processed_user_features.csv' not found. Please run 'make featurize' first.")
        return

    # AutoGluon works directly with pandas DataFrames
    train_data = TabularDataset(df)
    
    # Define the output directory for AutoGluon models and artifacts
    save_path = 'ml_artifacts/autogluon'

    # Initialize the TabularPredictor
    # We specify the label, evaluation metric, and where to save models.
    predictor = TabularPredictor(
        label='churn',
        eval_metric='roc_auc',  # Perfect metric for imbalanced churn problem
        path=save_path
    ).fit(
        train_data,
        presets='best_quality', # A preset that emphasizes model quality
        time_limit=600,         # Run for 10 minutes; increase for better results
        excluded_model_types=['KNN'] # Exclude simple models if desired
    )

    # Display the leaderboard of all trained models
    print("\n--- AutoGluon Model Leaderboard (Sorted by roc_auc) ---")
    leaderboard = predictor.leaderboard(df, silent=True)
    print(leaderboard)

    # Get the best model name
    best_model_name = leaderboard.iloc[0]['model']
    print(f"\n--- Best Performing Model: {best_model_name} ---")
    print("Models and artifacts are saved in:", save_path)


if __name__ == '__main__':
    main()