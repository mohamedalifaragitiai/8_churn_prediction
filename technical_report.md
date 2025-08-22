# Customer Churn Prediction System

## 1. Problem Definition & Project Objective
**Objective:** This project implements a robust, end-to-end machine learning system to predict customer churn for a music streaming service.
The primary business goal is to proactively identify users who are at high risk of canceling their subscription, enabling the business to deploy targeted retention strategies and reduce revenue loss.

**Data Overview:**
The system uses a JSON log file (`customer_churn_mini.json`) containing user activity events.
A significant challenge is the absence of an explicit churn label in the data, requiring a robust, inferred definition.
Furthermore, the dataset exhibits a severe class imbalance, which must be addressed to build a meaningful predictive model.

---

## 2. Feature Engineering & Data Processing
A critical first step was to transform the raw, event-level data into a user-level feature set suitable for modeling.

### Data Cleaning
- Filtered out all events from logged-out users (`auth != 'Logged In'`).
- Converted all timestamp columns (`ts`, `registration`) to datetime objects.
- Handled missing user metadata (gender, location, etc.) by propagating the last known valid value for each user (forward-fill and backward-fill).

### Inferred Churn Definition
The most significant data challenge was the lack of an explicit `is_churned` flag.
To solve this, churn was inferred based on a two-part behavioral pattern:

1. **Trigger Event:** The user performs an action indicating dissatisfaction or intent to leave, specifically visiting the *Submit Downgrade* or *Thumbs Down* pages.
2. **Subsequent Inactivity:** After a trigger event, the user shows no further activity for a significant period (defined as 30+ days).

This definition is more robust than relying on a single event, as it captures both intent and action, providing a more reliable and actionable churn signal.

### Key Features Created
- **Tenure:** Days between user registration and their last seen activity.
- **Engagement Metrics:** Aggregates of user actions, including `total_songs_played`, `total_listen_time`, `num_thumbs_up`, and `num_thumbs_down`.
- **Session Behavior:** `num_sessions` and `avg_songs_per_session` to measure usage intensity.
- **Social and Account Interactions:** `num_friends_added` and `num_downgrades`.
- **Technical Footprint:** User's OS and browser, extracted from the `userAgent` string.

All categorical features were one-hot encoded to be used in the model.

---

## 3. Model Selection and Justification
A systematic, data-driven approach was used to select the best model for this problem.

### Model Selection Process
- **Automated Evaluation with AutoGluon:** Instead of manually testing different models, we used AutoGluon, a state-of-the-art AutoML framework.
It automatically trained, tuned, and compared a wide range of models, including LightGBM, XGBoost, CatBoost, RandomForest, and stacked ensembles.
- **Best Performer Identification:** AutoGluon's leaderboard showed that a `RandomForestClassifier` (as part of a bagged ensemble) consistently ranked as a top-performing model, outperforming a standalone LightGBM on the `roc_auc` metric.
- **Final Model Choice:** `RandomForestClassifier`.

### Justification
- **Proven Performance:** Demonstrated superior results by AutoML evaluation.
- **Robustness:** RandomForest is robust to outliers, noise, and captures complex non-linear relationships effectively.
- **Built-in Imbalance Handling:** With `class_weight='balanced'`, RandomForest automatically adjusts for class imbalance during training.

---

## 4. Performance Evaluation and Error Analysis
The final model was evaluated on a held-out validation set (20% of users), stratified by the churn label.

### Final Model Metrics
- **AUC-ROC:** `0.821`
- **Precision:** `1.0` (every predicted churn user actually churned)
- **Recall:** `0.286`
- **F1-Score:** `0.444`

### Error Analysis
- **False Positives:** Zero — the model never targeted non-churning users.
- **False Negatives:** High (~71%). These are churners not captured by current features, requiring more advanced feature engineering.

---

## 5. Productionization and MLOps Architecture
The project follows a production-first design with MLOps best practices:

- **Containerization:** `Dockerfile` for environment consistency.
- **API for Inference:** `FastAPI` service exposing `/predict` endpoint.
- **Automation:** `Makefile` for setup (`make install`), pipeline (`make all`), and API deployment (`make run-api`).
- **Code Quality & CI:** Pre-commit hooks with `ruff` and `black`.
- **Experiment Tracking:** MLflow logs parameters, metrics, and models for full traceability.

---

## 6. Monitoring and Retraining Strategy
Hybrid monitoring + retraining strategy:

### Monitoring
- **Data Drift:** Monitor feature distributions using statistical tests (e.g., K-S test).
- **Concept Drift:** Track performance metrics (AUC, Precision, Recall) on new data.

### Retraining
- **Schedule-Based:** Weekly/bi-weekly retraining to capture gradual shifts.
- **Trigger-Based:** Immediate retraining if drift or performance drop is detected.

---

## 7. Technical Challenges and Solutions
- **Defining Churn:** Solved via robust inferred churn definition.
- **Model Selection:** Simplified via AutoGluon AutoML.
- **Environment & Dependencies:** Resolved by replacing PyCaret with AutoGluon (less restrictive).

---

## 8. Future Improvements
- **Advanced Feature Engineering:** Use time-series trends (e.g., decline in "Thumbs Up").
- **Model Explainability:** Apply SHAP for interpretability at both global and local levels.
- **Automated Retraining Orchestration:** Implement with Airflow or Prefect.

---
