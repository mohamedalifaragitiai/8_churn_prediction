import os
import sys

import pandas as pd

from src.churn_predictor.feature_engineering import FeatureEngineer

# Add the project root to the Python path BEFORE any other imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Define file paths
INPUT_PATH = "data/customer_churn_mini.json"
OUTPUT_PATH = "data/processed_user_features.csv"


def main():
    """Main function to run the feature engineering pipeline."""
    print("Starting feature engineering...")

    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input data not found at {INPUT_PATH}. Please add it.")
        return

    df = pd.read_json(INPUT_PATH, lines=True)

    feature_engineer = FeatureEngineer(df)
    processed_df = feature_engineer.process()

    if processed_df.empty:
        print("Feature engineering failed. Exiting.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    processed_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature engineering complete. Data saved to {OUTPUT_PATH}")
    print("Shape of processed data:", processed_df.shape)
    print("Churn distribution:\n", processed_df["churn"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
