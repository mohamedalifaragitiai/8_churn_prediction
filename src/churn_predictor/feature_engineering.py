# import pandas as pd
# from user_agents import parse

# class FeatureEngineer:
#     """
#     Handles data cleaning, preprocessing, and feature engineering.
#     """
#     def __init__(self, df: pd.DataFrame):
#         self.df = df.copy()

#     def clean_data(self):
#         """Cleans and preprocesses the raw event data."""
#         self.df = self.df[self.df['auth'] == 'Logged In'].copy()
#         self.df['ts'] = pd.to_datetime(self.df['ts'], unit='ms')
#         self.df['registration'] = pd.to_datetime(self.df['registration'], unit='ms')

#         user_cols = ['location', 'userAgent', 'lastName', 'firstName', 'gender', 'registration']
#         self.df[user_cols] = self.df.groupby('userId')[user_cols].transform(lambda x: x.ffill().bfill())
#         self.df.dropna(subset=user_cols, inplace=True)
#         return self

#     def create_churn_label(self):
#         """Creates the churn label based on the actual event from the data."""
        
#         # !!! IMPORTANT: REPLACE THIS WITH THE REAL EVENT NAME YOU FOUND !!!
#         CHURN_EVENT_NAME = 'Cancellation Confirmation' 
        
#         churn_users = self.df[self.df['page'] == CHURN_EVENT_NAME]['userId'].unique()

#         print(f"Found {len(churn_users)} users who churned based on the event: '{CHURN_EVENT_NAME}'")

#         if len(churn_users) == 0:
#             print(f"\nCRITICAL ERROR: No users found with the page event '{CHURN_EVENT_NAME}'.")
#             print("Model training cannot proceed. Please verify the event name in 'inspect_data.py' and update it in the script.")
#             return None # Signal failure

#         self.df['churn'] = self.df['userId'].isin(churn_users).astype(int)
#         return self
    
#     def create_user_level_features(self) -> pd.DataFrame:
#         """Aggregates event-level data to create a user-level feature set."""
#         # This check is crucial to prevent running on empty data
#         if 'churn' not in self.df.columns or self.df['churn'].nunique() < 2:
#             return pd.DataFrame() # Return an empty DataFrame to signal failure

#         user_df = self.df.groupby('userId').agg(
#             churn=('churn', 'max'),
#             gender=('gender', 'first'),
#             registration_ts=('registration', 'first'),
#             last_session_ts=('ts', 'max'),
#             total_songs=('song', 'count'),
#             total_listen_time=('length', 'sum'),
#             num_artists=('artist', 'nunique'),
#             num_thumbs_up=('page', lambda x: (x == 'Thumbs Up').sum()),
#             num_thumbs_down=('page', lambda x: (x == 'Thumbs Down').sum()),
#             num_sessions=('sessionId', 'nunique'),
#             num_friends_added=('page', lambda x: (x == 'Add Friend').sum()),
#             num_downgrades=('page', lambda x: (x == 'Submit Downgrade').sum()),
#             last_level=('level', 'last')
#         ).reset_index()

#         user_df['tenure'] = (user_df['last_session_ts'] - user_df['registration_ts']).dt.days
#         user_df['avg_songs_per_session'] = (user_df['total_songs'] / user_df['num_sessions']).fillna(0)

#         user_agents = self.df[['userId', 'userAgent']].drop_duplicates()
#         user_df = pd.merge(user_df, user_agents, on='userId', how='left')
#         user_df['os'] = user_df['userAgent'].apply(lambda x: parse(x).os.family if x else 'Unknown')
#         user_df['browser'] = user_df['userAgent'].apply(lambda x: parse(x).browser.family if x else 'Unknown')

#         features = user_df.drop(columns=['registration_ts', 'last_session_ts', 'userAgent'])
#         categorical_cols = ['gender', 'last_level', 'os', 'browser']
#         features = pd.get_dummies(features, columns=categorical_cols, drop_first=True, dummy_na=True)
        
#         return features

#     def process(self) -> pd.DataFrame:
#         """Executes the full feature engineering pipeline."""
#         self.clean_data()
#         if self.create_churn_label() is None:
#             # If churn label creation failed, return an empty DataFrame
#             return pd.DataFrame()
#         user_features_df = self.create_user_level_features()
#         return user_features_df

import pandas as pd
from user_agents import parse
import numpy as np

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_data(self):
        self.df = self.df[self.df['auth'] == 'Logged In'].copy()
        self.df['ts'] = pd.to_datetime(self.df['ts'], unit='ms')
        self.df['registration'] = pd.to_datetime(self.df['registration'], unit='ms')

        user_cols = ['location', 'userAgent', 'lastName', 'firstName', 'gender', 'registration']
        self.df[user_cols] = self.df.groupby('userId')[user_cols].transform(lambda x: x.ffill().bfill())
        self.df.dropna(subset=user_cols, inplace=True)
        return self

    def create_churn_label(self):
        """
        Defines churn by identifying users who performed a churn-trigger action
        ('Submit Downgrade' or 'Thumbs Down') and then became inactive.
        """
        # 1. Define the observation window and inactivity threshold
        max_date = self.df['ts'].max()
        INACTIVITY_THRESHOLD = pd.Timedelta(days=30)
        cutoff_date = max_date - INACTIVITY_THRESHOLD

        # 2. Identify users who performed a trigger action
        downgrade_users = self.df[self.df['page'] == 'Submit Downgrade']['userId'].unique()
        thumbs_down_users = self.df[self.df['page'] == 'Thumbs Down']['userId'].unique()
        potential_churners = np.union1d(downgrade_users, thumbs_down_users)
        
        # 3. Get the last interaction for each of these potential churners
        last_interaction = self.df[self.df['userId'].isin(potential_churners)].groupby('userId')['ts'].max()
        
        # 4. Final churn definition: a potential churner whose last action was before the cutoff
        churned_user_ids = last_interaction[last_interaction < cutoff_date].index

        print(f"Found {len(churned_user_ids)} churned users based on trigger events and inactivity.")

        if len(churned_user_ids) == 0:
            print("\nCRITICAL WARNING: No churners were identified with the new logic. The data may be too sparse or short in duration.")
            return None

        self.df['churn'] = self.df['userId'].isin(churned_user_ids).astype(int)
        return self

    def create_user_level_features(self):
        # ... (This method remains the same as before)
        user_df = self.df.groupby('userId').agg(
            churn=('churn', 'max'),
            gender=('gender', 'first'),
            registration_ts=('registration', 'first'),
            last_session_ts=('ts', 'max'),
            total_songs=('song', 'count'),
            total_listen_time=('length', 'sum'),
            num_artists=('artist', 'nunique'),
            num_thumbs_up=('page', lambda x: (x == 'Thumbs Up').sum()),
            num_thumbs_down=('page', lambda x: (x == 'Thumbs Down').sum()),
            num_sessions=('sessionId', 'nunique'),
            num_friends_added=('page', lambda x: (x == 'Add Friend').sum()),
            num_downgrades=('page', lambda x: (x == 'Submit Downgrade').sum()),
            last_level=('level', 'last')
        ).reset_index()

        user_df['tenure'] = (user_df['last_session_ts'] - user_df['registration_ts']).dt.days
        user_df['avg_songs_per_session'] = (user_df['total_songs'] / user_df['num_sessions']).fillna(0)

        user_agents = self.df[['userId', 'userAgent']].drop_duplicates()
        user_df = pd.merge(user_df, user_agents, on='userId', how='left')
        user_df['os'] = user_df['userAgent'].apply(lambda x: parse(x).os.family if x else 'Unknown')
        user_df['browser'] = user_df['userAgent'].apply(lambda x: parse(x).browser.family if x else 'Unknown')

        features = user_df.drop(columns=['registration_ts', 'last_session_ts', 'userAgent'])
        categorical_cols = ['gender', 'last_level', 'os', 'browser']
        features = pd.get_dummies(features, columns=categorical_cols, drop_first=True, dummy_na=True)
        return features

    def process(self):
        self.clean_data()
        if self.create_churn_label() is None:
            return pd.DataFrame()
        return self.create_user_level_features()

