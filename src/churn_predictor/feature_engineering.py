import numpy as np
import pandas as pd
from user_agents import parse


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_data(self):
        self.df = self.df[self.df["auth"] == "Logged In"].copy()
        self.df["ts"] = pd.to_datetime(self.df["ts"], unit="ms")
        self.df["registration"] = pd.to_datetime(self.df["registration"], unit="ms")

        user_cols = [
            "location",
            "userAgent",
            "lastName",
            "firstName",
            "gender",
            "registration",
        ]
        self.df[user_cols] = self.df.groupby("userId")[user_cols].transform(
            lambda x: x.ffill().bfill()
        )
        self.df.dropna(subset=user_cols, inplace=True)
        return self

    def create_churn_label(self):
        """
        Defines churn by identifying users who performed a churn-trigger action
        ('Submit Downgrade' or 'Thumbs Down') and then became inactive.
        """
        max_date = self.df["ts"].max()
        INACTIVITY_THRESHOLD = pd.Timedelta(days=30)
        cutoff_date = max_date - INACTIVITY_THRESHOLD

        downgrade_users = self.df[self.df["page"] == "Submit Downgrade"][
            "userId"
        ].unique()
        thumbs_down_users = self.df[self.df["page"] == "Thumbs Down"]["userId"].unique()
        potential_churners = np.union1d(downgrade_users, thumbs_down_users)

        last_interaction = (
            self.df[self.df["userId"].isin(potential_churners)]
            .groupby("userId")["ts"]
            .max()
        )

        churned_user_ids = last_interaction[last_interaction < cutoff_date].index

        print(
            f"Found {len(churned_user_ids)} churned users based on "
            "trigger events and inactivity."
        )

        if len(churned_user_ids) == 0:
            print(
                "\nCRITICAL WARNING: No churners were identified. "
                "The data may be too sparse."
            )
            return None

        self.df["churn"] = self.df["userId"].isin(churned_user_ids).astype(int)
        return self

    def create_user_level_features(self):
        user_df = (
            self.df.groupby("userId")
            .agg(
                churn=("churn", "max"),
                gender=("gender", "first"),
                registration_ts=("registration", "first"),
                last_session_ts=("ts", "max"),
                total_songs=("song", "count"),
                total_listen_time=("length", "sum"),
                num_artists=("artist", "nunique"),
                num_thumbs_up=("page", lambda x: (x == "Thumbs Up").sum()),
                num_thumbs_down=("page", lambda x: (x == "Thumbs Down").sum()),
                num_sessions=("sessionId", "nunique"),
                num_friends_added=("page", lambda x: (x == "Add Friend").sum()),
                num_downgrades=("page", lambda x: (x == "Submit Downgrade").sum()),
                last_level=("level", "last"),
            )
            .reset_index()
        )

        user_df["tenure"] = (
            user_df["last_session_ts"] - user_df["registration_ts"]
        ).dt.days
        user_df["avg_songs_per_session"] = (
            user_df["total_songs"] / user_df["num_sessions"]
        ).fillna(0)

        user_agents = self.df[["userId", "userAgent"]].drop_duplicates()
        user_df = pd.merge(user_df, user_agents, on="userId", how="left")
        user_df["os"] = user_df["userAgent"].apply(
            lambda x: parse(x).os.family if x else "Unknown"
        )
        user_df["browser"] = user_df["userAgent"].apply(
            lambda x: parse(x).browser.family if x else "Unknown"
        )

        features = user_df.drop(
            columns=["registration_ts", "last_session_ts", "userAgent"]
        )
        categorical_cols = ["gender", "last_level", "os", "browser"]
        features = pd.get_dummies(
            features, columns=categorical_cols, drop_first=True, dummy_na=True
        )
        return features

    def process(self):
        self.clean_data()
        if self.create_churn_label() is None:
            return pd.DataFrame()
        return self.create_user_level_features()
