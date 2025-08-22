from typing import Optional

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """
    Pydantic model for the input features for a single prediction.
    This should match the user-level features created in Phase 1.
    """

    tenure: int
    total_songs: int
    total_listen_time: float
    num_artists: int
    num_thumbs_up: int
    num_thumbs_down: int
    num_sessions: int
    num_friends_added: int
    num_downgrades: int
    avg_songs_per_session: float
    gender_Male: bool = False
    last_level_paid: bool = True
    os_Linux: bool = False
    os_Mac_OS_X: bool = False
    os_Windows: bool = False
    # Add other OS/browser dummy columns as needed

    class Config:
        schema_extra = {
            "example": {
                "tenure": 120,
                "total_songs": 500,
                "total_listen_time": 120000.0,
                "num_artists": 150,
                "num_thumbs_up": 20,
                "num_thumbs_down": 2,
                "num_sessions": 30,
                "num_friends_added": 5,
                "num_downgrades": 0,
                "avg_songs_per_session": 16.67,
                "gender_Male": True,
                "last_level_paid": True,
                "os_Windows": True,
            }
        }


class PredictionResponse(BaseModel):
    """
    Pydantic model for the prediction output.
    """

    churn_prediction: Optional[int] = None
    churn_probability: Optional[float] = None
    error: Optional[str] = None
