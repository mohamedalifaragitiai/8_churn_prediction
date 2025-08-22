import lightgbm as lgb
import joblib
import yaml
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os

class ChurnModel:
    """
    A wrapper class for the LightGBM churn prediction model.
    It handles loading configuration, training, evaluation, and saving the model.
    """
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.model_params = self.config['params']
        self.model_path = self.config['model']['save_path']

    def _prepare_data(self, df: pd.DataFrame):
        """Prepares data for training and validation."""
        X = df.drop(columns=['userId', 'churn'])
        y = df['churn']
        
        # Save feature list
        features_dir = os.path.dirname(self.model_path)
        os.makedirs(features_dir, exist_ok=True)
        joblib.dump(list(X.columns), os.path.join(features_dir, 'feature_list.joblib'))
        
        # Handle class imbalance
        scale_pos_weight = y.value_counts()[0] / y.value_counts()
        self.model_params['scale_pos_weight'] = scale_pos_weight

        return train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )

    def train(self, df: pd.DataFrame):
        """
        Trains the LightGBM model.
        
        Args:
            df (pd.DataFrame): The processed user features dataframe.
        """
        X_train, X_val, y_train, y_val = self._prepare_data(df)
        
        self.model = lgb.LGBMClassifier(**self.model_params)
        
        print("Starting model training...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(
                    self.config['training']['early_stopping_rounds'],
                    verbose=True
                )
            ]
        )
        print("Model training complete.")
        self.save_model()

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """
        Evaluates the model on the validation set.
        
        Returns:
            dict: A dictionary of evaluation metrics.
        """
        if not self.model:
            raise ValueError("Model has not been trained yet.")
            
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            "AUC_ROC": roc_auc_score(y_val, y_pred_proba),
            "F1_Score": f1_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred),
            "Recall": recall_score(y_val, y_pred)
        }
        return metrics

    def predict(self, input_data: pd.DataFrame) -> tuple[int, float]:
        """
        Makes a prediction on new data.

        Args:
            input_data (pd.DataFrame): A DataFrame with a single row of features.

        Returns:
            tuple: A tuple containing the prediction (0 or 1) and probability.
        """
        if not self.model:
            self.load_model()
            
        probability = self.model.predict_proba(input_data)[:, 1][0]
        prediction = int(probability > 0.5)
        return prediction, probability

    def save_model(self):
        """Saves the trained model to the path specified in the config."""
        print(f"Saving model to {self.model_path}")
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        """Loads the model from the path specified in the config."""
        print(f"Loading model from {self.model_path}")
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

