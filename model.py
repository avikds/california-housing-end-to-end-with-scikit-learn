"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
import os
import tarfile
import tempfile
import urllib.request
import pandas as pd

def load_housing():
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    tgz_path = os.path.join(tempfile.gettempdir(), "housing.tgz")

    # Download the dataset only if it is not already present.
    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(url, tgz_path)

    # Read housing/housing.csv directly from the tarball.
    with tarfile.open(tgz_path, mode="r:gz") as housing_tgz:
        with housing_tgz.extractfile("housing/housing.csv") as csv_file:
            housing = pd.read_csv(csv_file)

    return housing

# Step 2 - income_categories
import numpy as np

def income_categories(df):
    bins = [0, 1.5, 3.0, 4.5, 6.0, np.inf]
    labels = [1, 2, 3, 4, 5]

    categories = pd.cut(
        df["median_income"],
        bins=bins,
        labels=labels
    )

    return categories.astype(int)

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split

def stratified_split(df, test_size=0.2, random_state=42):
    income_cat = income_categories(df)

    train_set, test_set = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=income_cat
    )

    return train_set, test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    corr_matrix = df.corr(numeric_only=True)
    correlations = corr_matrix["median_house_value"].drop("median_house_value")
    return correlations.sort_values(ascending=False)

# Step 5 - add_ratio_features
def add_ratio_features(df):
    result = df.copy()

    result["rooms_per_house"] = result["total_rooms"] / result["households"]
    result["bedrooms_ratio"] = result["total_bedrooms"] / result["total_rooms"]
    result["people_per_house"] = result["population"] / result["households"]

    return result

# Step 6 - split_features_labels
def split_features_labels(df):
    X = df.drop(columns=["median_house_value"]).copy()
    y = df["median_house_value"].copy()

    return X, y

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, names=None):
        return [
            f"Cluster {i} similarity"
            for i in range(self.n_clusters)
        ]

# Step 8 - numeric_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

# Step 9 - categorical_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

def categorical_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

# Step 10 - build_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )

    return ColumnTransformer(
        transformers=[
            (
                "log",
                log_pipeline,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ]
            ),
            (
                "geo",
                ClusterSimilarity(
                    n_clusters=n_clusters,
                    gamma=gamma,
                    random_state=random_state
                ),
                ["latitude", "longitude"]
            ),
            (
                "cat",
                categorical_pipeline(),
                ["ocean_proximity"]
            ),
        ],
        remainder=numeric_pipeline()
    )

# Step 11 - rmse
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Step 12 - dummy_baseline_rmse
from sklearn.dummy import DummyRegressor

def dummy_baseline_rmse(X, y):
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)

    y_pred = model.predict(X)

    return rmse(y, y_pred)

# Step 13 - cross_val_rmse
from sklearn.model_selection import cross_val_score

def cross_val_rmse(model, X, y, cv=3):
    scores = -cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    scores = [float(score) for score in scores]

    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }

# Step 14 - linear_model
from sklearn.linear_model import LinearRegression

def linear_model(preprocessing):
    return make_pipeline(
        preprocessing,
        LinearRegression()
    )

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

