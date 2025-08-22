# Technical Report: Customer Churn Prediction System

## 1. Problem Definition & Data Overview

**Objective**: Develop a machine learning model to predict customer churn for a music streaming service using user activity logs. The primary business goal is to proactively identify at-risk users to enable targeted retention campaigns.

**Data**: The input is a JSON file (`customer_churn_mini.json`) containing event logs. Each record represents a single action taken by a user. Key columns include `userId`, `ts` (timestamp), `page` (event type), `level` (paid/free), and song details. A major challenge is the class imbalance, with churned users representing a small fraction of the total user base.

## 2. Feature Engineering

The raw event-level data was aggregated to create a feature set for each unique `userId`.

**Data Cleaning**:
-   Records for logged-out users (`auth != 'Logged In'`) were dropped.
-   Timestamps (`ts`, `registration`) were converted to datetime objects.
-   Missing user-level data (`gender`, `location`) was imputed using a forward/backward fill strategy grouped by user.

**Churn Definition**:
-   A user was labeled as **churned (1)** if they visited the `'Cancellation Confirmation'` page at any point. All other users were labeled as **active (0)**.

**Key Features Created**:
-   **Tenure**: Days since registration.
-   **Engagement**: `total_songs_played`, `total_listen_time`, `num_thumbs_up`, `num_thumbs_down`.
-   **Session Metrics**: `num_sessions`, `avg_songs_per_session`.
-   **Social/Account**: `num_friends_added`, `num_downgrades`, `last_subscription_level`.
-   **Technical**: `OS` and `browser` extracted from the `userAgent` string.
-   Categorical features were one-hot encoded for model consumption.

## 3. Model Selection and Justification

**Model**: **LightGBM (Light Gradient Boosting Machine)** was chosen as the primary model.

**Justification**:
-   **Performance**: Tree-based models like LightGBM excel on tabular data, effectively capturing non-linear relationships and feature interactions.
-   **Efficiency**: LightGBM is known for its high training speed and low memory usage compared to alternatives like XGBoost or RandomForest.
-   **Imbalance Handling**: It has a built-in `scale_pos_weight` parameter, providing a simple and effective way to handle the class imbalance without requiring complex data resampling techniques like SMOTE.
-   A **Logistic Regression** model served as a baseline, and LightGBM demonstrated significantly better performance, particularly in terms of AUC and F1-score.

## 4. Performance Evaluation and Error Analysis

The model was evaluated on a validation set (20% of users), split at the `userId` level to prevent data leakage.

**Evaluation Metrics**:
-   **AUC-ROC**: The primary metric, suitable for imbalanced classification, measuring the model's ability to distinguish between classes.
-   **F1-Score**: The harmonic mean of Precision and Recall, providing a balanced measure of performance.
-   **Precision-Recall Curve**: Visualizes the trade-off between precision and recall, crucial for business decisions.

**Error Analysis**:
-   **False Positives (Predicted Churn, Did Not Churn)**: These users might be flagged for unnecessary retention offers, leading to minor costs. They might exhibit some churn-like behavior (e.g., visiting the 'Settings' page often) but ultimately decide to stay.
-   **False Negatives (Predicted Stay, Did Churn)**: **This is the more critical error**. We fail to identify an at-risk user, leading to a lost customer. These users might not exhibit obvious pre-churn signals, making them harder to detect. Future work should focus on engineering features that better capture subtle changes in their behavior to reduce this error rate.

## 5. Productionization and MLOps Architecture

The system is designed for production deployment with a robust MLOps foundation.
-   **Containerization**: A `Dockerfile` packages the FastAPI application, ensuring a consistent and reproducible runtime environment.
-   **API**: A `FastAPI` service exposes a `/predict` endpoint for real-time inference. It accepts user features and returns a churn probability.
-   **Automation**: A `Makefile` automates common tasks like installation, testing, and training. `pre-commit` hooks enforce code quality with `ruff` and `black`.
-   **Experiment Tracking**: `MLflow` is integrated into the training script to log parameters, metrics, and model artifacts for every run, ensuring full traceability and reproducibility of experiments.

## 6. Monitoring and Retraining Strategy

A hybrid strategy is proposed to maintain model performance over time.

**Monitoring**:
-   **Data Drift**: Key input feature distributions (e.g., `tenure`, `avg_songs_per_session`) will be monitored using the Kolmogorov-Smirnov (K-S) test. A significant drift from the training distribution will trigger an alert.
-   **Concept Drift**: The model's F1-score and AUC will be tracked on new, labeled data. A sustained performance drop below a predefined threshold will signal concept drift.

**Retraining**:
-   **Schedule-Based**: A weekly retraining pipeline will run automatically to capture evolving user behaviors.
-   **Trigger-Based**: Retraining will also be triggered immediately if the monitoring system detects significant data/concept drift. This ensures the model adapts quickly to changes in the data landscape.

## 7. Technical Challenges Faced

-   **Class Imbalance**: Mitigated using the `scale_pos_weight` parameter in LightGBM.
-   **Data Leakage**: Prevented by splitting the data into train/validation sets at the `userId` level.
-   **Feature Definition**: Defining robust behavioral features from raw event logs required careful aggregation and domain knowledge.

## 8. Future Improvements

-   **Advanced Feature Engineering**: Explore more complex features, such as time-series analysis of user activity in the weeks leading up to churn.
-   **Hyperparameter Tuning**: Implement an automated hyperparameter optimization pipeline (e.g., using Optuna) to find the best model configuration.
-   **Explainability**: Integrate SHAP (SHapley Additive exPlanations) to provide local, instance-level explanations for why a specific user is predicted to churn, empowering business teams with actionable insights.
