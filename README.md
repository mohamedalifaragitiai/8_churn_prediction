# End-to-End Customer Churn Prediction System

This repository contains a production-grade machine learning system designed to predict customer churn for a music streaming service. The project demonstrates a full-cycle MLOps workflow, beginning with raw data analysis and culminating in a containerized API that serves live predictions.

The core objective is to leverage user activity logs to proactively identify customers who are likely to cancel their subscriptions. This enables the business to deploy targeted retention strategies, thereby minimizing customer attrition and maximizing revenue.

## 1. Project Objective & Approach

The goal is to build a robust machine learning model that identifies users likely to churn based on their platform activity. My approach addresses the entire ML lifecycle, from data analysis and model development to deployment and monitoring, with a strong emphasis on automation and reproducibility.

This solution was engineered to overcome several significant real-world challenges outlined in the project description, including the absence of an explicit churn label, severe class imbalance in the data, and the need for a scalable, production-ready system.

## 2. Data Analysis & Feature Engineering

The initial step involved a thorough analysis of the raw `customer_churn_mini.json` data to understand user behavior patterns.

### Inferred Churn Definition

A primary challenge was the lack of an explicit `is_churn` flag. To solve this, I engineered a logical definition of churn based on user behavior:

> A user is flagged as **churned** if they perform a "trigger" action (e.g., *Submit Downgrade* or *Thumbs Down*) and subsequently show **no platform activity for 30 or more days**.

This data-driven definition creates a realistic and reliable target variable for the model. After processing, the dataset showed a **16.4% churn rate**, highlighting the class imbalance problem.

### Feature Creation

The event-level logs were aggregated into a user-level summary, creating a rich feature set for modeling. Key features include:
*   **Tenure:** Days since user registration.
*   **Usage Volume:** `total_songs_played`, `total_listen_time`.
*   **Engagement Metrics:** `num_thumbs_up`, `num_thumbs_down`, `num_friends_added`.
*   **Session Behavior:** `num_sessions`, `avg_session_duration`.
*   **Account Information:** `last_level_paid`, `gender`, `os`.

The entire feature engineering pipeline is automated via the `make featurize` command, which processes the raw data and saves the final feature set.

## 3. Model Development and Evaluation

### a. Model Selection

For this classification task, a **RandomForestClassifier** was chosen as the final model. This choice was justified through a data-driven approach using the **AutoGluon** AutoML framework (`make find_best_model`). The AutoML process systematically tested a wide range of architectures and revealed that tree-based ensembles, particularly RandomForest, were top performers for this dataset, achieving a validation AUC score of approximately **0.85**.

**Why RandomForestClassifier is a suitable choice:**
*   **Performance:** It demonstrated top-tier performance on the leaderboard.
*   **Robustness:** It is less sensitive to outliers and can handle non-linear relationships between features effectively.
*   **Imbalance Handling:** It includes a built-in `class_weight='balanced'` parameter, which internally adjusts for the severe class imbalance without needing manual data resampling techniques like SMOTE, which can introduce artificial data points.

### b. Performance Metrics

Given the imbalanced nature of the dataset (only 16.4% churners), **accuracy is a misleading metric**. Instead, the model's performance was evaluated using more appropriate metrics:
*   **AUC ROC (0.82):** Measures the model's ability to distinguish between churners and non-churners. An AUC of 0.82 indicates a good level of separability.
*   **F1-Score (0.44):** The harmonic mean of Precision and Recall, providing a balanced measure for imbalanced classes.
*   **Precision (1.0):** Of all the users predicted to churn, 100% of them actually did. This is a critical result for the business, as it ensures that retention resources are not wasted on customers who were never at risk.
*   **Recall (0.29):** The model successfully identified 29% of all actual churners.

### c. Error Analysis

The evaluation metrics reveal a clear trade-off in the model's behavior:
*   **Zero False Positives:** With a **Precision of 1.0**, the model is extremely conservative. Every user it flags as a churn risk is a correct prediction. This is highly valuable as it prevents the business from offering unnecessary discounts or incentives to happy customers.
*   **Presence of False Negatives:** The **Recall of 0.29** indicates that the model fails to identify 71% of the users who will eventually churn. This means there is a missed opportunity to retain a significant portion of at-risk customers.

**Conclusion:** The model is optimized to be a high-confidence "churn detector." While it misses some at-risk users, it ensures that every alert it raises is actionable and correct, making it a reliable tool for targeted intervention campaigns.

## 4. Proposed Model Retraining System

To handle data drift and ensure the model remains accurate over time, a periodic retraining strategy is essential. The proposed system is designed for automation using a workflow orchestrator like **Airflow** or **Prefect**.

The retraining pipeline would consist of the following automated steps:
1.  **Data Ingestion:** A recurring job (e.g., weekly) pulls the latest user activity logs from the production database.
2.  **Feature Engineering:** The `featurize.py` script runs on the new data to generate an updated feature set. The churn labels are calculated based on the 30-day inactivity window.
3.  **Model Retraining:** The `train.py` script retrains the `RandomForestClassifier` on the newly expanded dataset.
4.  **Model Validation:** The newly trained model is evaluated against a hold-out test set. Its performance (AUC, F1-Score) is compared to the currently deployed model, logged in MLflow.
5.  **Conditional Deployment:** If the new model shows a statistically significant performance improvement (e.g., >5% increase in F1-score), it is automatically promoted and deployed to the API. Otherwise, the existing model is retained.

This automated workflow ensures the model adapts to evolving user behaviors without manual intervention.

## 5. Proposed Monitoring System

A simple yet effective monitoring system is proposed to detect data drift, concept drift, and performance degradation.

1.  **Data Drift Detection:**
    *   **Mechanism:** Implement statistical tests to compare the distribution of key input features between the training data and live prediction requests. The **Kolmogorov-Smirnov (K-S) test** is a good candidate for this.
    *   **Implementation:** A scheduled script would run daily, pulling feature data from live requests and comparing its distribution to the training set's feature distributions. If the p-value from the K-S test drops below a threshold (e.g., 0.05) for critical features like `num_thumbs_down` or `avg_session_duration`, an alert is triggered.

2.  **Concept Drift & Performance Tracking:**
    *   **Mechanism:** Since real-world churn labels are delayed by 30 days, we will track proxy metrics. The primary metric to monitor is the model's **prediction distribution**. A sudden shift in the proportion of users being flagged as churn risks (e.g., from 5% to 20%) could indicate concept drift.
    *   **Implementation:** Log the output probabilities of the model for all prediction requests. A monitoring dashboard (e.g., in Grafana) would visualize the distribution of these probabilities over time. When true labels become available, the **AUC and F1-score** will be re-calculated and tracked to monitor the model's true performance.

## 6. Project Packaging and Automation

The project is structured following software engineering best practices to ensure scalability, reproducibility, and ease of use.

### Project Structure
The repository is organized into a modular structure:
.
├── Dockerfile # Defines the container for the API
├── Makefile # Automates project setup, training, and deployment
├── README.md # This project documentation
├── api/
│ └── main.py # FastAPI application for serving predictions
├── data/ # (Git-ignored) For raw and processed data
├── notebooks/ # Jupyter notebooks for EDA and analysis
├── pyproject.toml # Manages all project dependencies via uv
├── scripts/
│ ├── featurize.py # Feature engineering pipeline
│ ├── find_best_model.py # AutoML script for model selection
│ └── train.py # Final model training and MLflow logging
└── src/
└── churn_predictor/ # Core source code for the project

text

### Key Technical Features
*   **API Service:** A **FastAPI** application serves predictions via a `/predict` endpoint, packaged with **Docker** for a consistent and isolated runtime environment.
*   **Dependency Management:** Project dependencies are managed with **uv** and a `pyproject.toml` file for fast and reliable installation.
*   **Automation:** A `Makefile` automates the entire MLOps workflow (`make install`, `make all`, `make run-api`), ensuring the process is fully reproducible.
*   **Code Quality:** **black** and **ruff** are used for automated code formatting and linting, enforced via **pre-commit** hooks to maintain high code standards.
*   **Experiment Tracking:** **MLflow** is integrated into the training pipeline (`make train`) to log all parameters, metrics, and model artifacts, providing full traceability of experiments.

## 7. How to Run This Project

### Prerequisites
*   Python 3.12+
*   Docker
*   `make` utility

### Step-by-Step Instructions
1.  **Clone the Repository and Install Dependencies**
    ```
    git clone https://github.com/mohamedalifaragitiai/8_churn_prediction.git
    cd 8_churn_prediction

    # Create venv and install all packages
    make install

    # Activate the virtual environment
    source churn/bin/activate
    ```
    *Note: Place the `customer_churn_mini.json` dataset in the `data/` directory.*

2.  **Run the Full ML Pipeline**
    This single command executes feature engineering, model selection, and final training.
    ```
    make all
    ```

3.  **Serve the Model via API**
    This command builds the Docker image and deploys the containerized FastAPI service.
    ```
    make run-api
    ```
    The API will be accessible at `http://localhost:8000`.

4.  **Test the Live API**
    Use `curl` to send a sample request to the `/predict` endpoint:
    ```
    curl -X 'POST' \
      'http://localhost:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "tenure": 150, "total_songs": 80, "total_listen_time": 20000.0,
        "num_artists": 50, "num_thumbs_up": 5, "num_thumbs_down": 10,
        "num_sessions": 15, "num_friends_added": 0, "num_downgrades": 1,
        "avg_songs_per_session": 5.33, "gender_Male": false,
        "last_level_paid": false, "os_Windows": true
      }'
    ```

## 8. Future Improvements & Ideas

*   **Enhanced Feature Engineering:** Incorporate time-series analysis to capture trends in user behavior (e.g., declining song plays over the last 14 days) for more predictive power.
*   **Deeper Model Explainability:** Integrate **SHAP** (SHapley Additive exPlanations) to explain individual predictions. This would provide the business team with granular insights into *why* a specific user is flagged as a churn risk, allowing for more personalized retention efforts.
*   **Full CI/CD Implementation:** Build out the proposed retraining and monitoring strategies using a dedicated orchestrator like Airflow to create a fully automated, production-grade ML pipeline.
