from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
import pandas as pd

class UserPredictor:
    def __init__(self):
        custom_trans = make_column_transformer(
            (PolynomialFeatures(), ["popularity", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]),
            (OneHotEncoder(), ["key", "mode", "music_genre"]),
            )
        self.pipe = Pipeline([
        ("ct", custom_trans),
        ("std2", StandardScaler()),
        ("lr", LogisticRegression()),
        
    ])
    
    def fit(self, train_x_df, train_y_df):
        self.pipe.fit(train_x_df[["popularity", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence", "key", "mode", "music_genre"]], train_y_df["y"])

        
    def predict(self, test_x_df):
        return self.pipe.predict(test_x_df[["popularity", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence", "key", "mode", "music_genre"]])
    def predict_proba(self, test_x_df):
        return self.pipe.predict_proba(test_x_df[["popularity", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence", "key", "mode", "music_genre"]])