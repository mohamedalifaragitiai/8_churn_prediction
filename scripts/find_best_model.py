import os
import sys

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    """
    Uses AutoGluon to automatically train and compare multiple models
    to find the best one for our churn prediction task.
    """
    print("--- Starting Automated Model Comparison with AutoGluon ---")

    try:
        df = pd.read_csv("data/processed_user_features.csv")
    except FileNotFoundError:
        print(
            "Error: 'data/processed_user_features.csv' not found. "
            "Please run 'make featurize' first."
        )
        return

    train_data = TabularDataset(df)
    save_path = "ml_artifacts/autogluon"

    predictor = TabularPredictor(
        label="churn", eval_metric="roc_auc", path=save_path
    ).fit(
        train_data, presets="best_quality", time_limit=600, excluded_model_types=["KNN"]
    )

    print("\n--- AutoGluon Model Leaderboard (Sorted by roc_auc) ---")
    leaderboard = predictor.leaderboard(df, silent=True)
    print(leaderboard)

    best_model_name = leaderboard.iloc[0]["model"]
    print(f"\n--- Best Performing Model: {best_model_name} ---")
    print("Models and artifacts are saved in:", save_path)


if __name__ == "__main__":
    main()
