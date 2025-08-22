import pandas as pd
import os
# Add the project root directory to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.churn_predictor.feature_engineering import FeatureEngineer

# Define file paths
INPUT_PATH = 'data/customer_churn_mini.json'
OUTPUT_PATH = 'data/processed_user_features.csv'

def main():
    """Main function to run the feature engineering pipeline."""
    print("Starting feature engineering...")

    # Load data
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input data not found at {INPUT_PATH}. Please add it.")
        return
        
    df = pd.read_json(INPUT_PATH, lines=True)

    # Instantiate the engineer and process data
    feature_engineer = FeatureEngineer(df)
    processed_df = feature_engineer.process()

    # Save the processed data
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    processed_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Feature engineering complete. Processed data saved to {OUTPUT_PATH}")
    print("Shape of processed data:", processed_df.shape)
    print("Churn distribution:\n", processed_df['churn'].value_counts(normalize=True))

if __name__ == '__main__':
    main()
