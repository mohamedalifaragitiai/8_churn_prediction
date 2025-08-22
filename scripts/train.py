import os
import sys

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define paths for the new model
DATA_PATH = "data/processed_user_features.csv"
ARTIFACTS_DIR = "ml_artifacts"
MODEL_PATH = os.path.join(
    ARTIFACTS_DIR, "random_forest_churn_model.pkl"
)  # <-- Updated model name
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_list.joblib")


def main():
    """
    Trains the final RandomForest model, which was identified as a top performer
    by the AutoML step.
    """
    print("Starting model training with RandomForestClassifier...")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # Define features and target
    y = df["churn"]
    X = df.drop(columns=["userId", "churn"])

    # Save feature list for the API
    joblib.dump(list(X.columns), FEATURES_PATH)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- START OF CHANGE ---
    # Initialize RandomForestClassifier
    # Use class_weight='balanced' to handle class imbalance automatically.
    model = RandomForestClassifier(
        n_estimators=100,  # A good starting point
        random_state=42,
        class_weight="balanced",  # This is the key parameter for imbalance
        n_jobs=-1,  # Use all available CPU cores
    )
    # --- END OF CHANGE ---

    mlflow.set_experiment("Churn_Prediction")
    with mlflow.start_run(run_name="RandomForest_Final_Model") as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_params(model.get_params())

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate model performance
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        auc_roc = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        metrics = {
            "AUC_ROC": auc_roc,
            "F1_Score": f1,
            "Precision": precision,
            "Recall": recall,
        }
        mlflow.log_metrics(metrics)
        print("Evaluation Metrics:", metrics)

        # Log Precision-Recall curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals)
        mlflow.log_metric("PR_AUC", pr_auc)

        fig, ax = plt.subplots()
        ax.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.2f}")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        plt.savefig("pr_curve.png")
        mlflow.log_artifact("pr_curve.png")
        plt.close()

        # Log and save the trained model
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, MODEL_PATH)

    print(f"Model training complete. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
