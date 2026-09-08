"""
California Housing, End to End with Scikit-Learn scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""California Housing, end to end with scikit-learn (Hands-On ML, chapter 2).

Story: get the data, split it honestly, look at it, build a preprocessing pipeline
that cannot leak, beat a dummy baseline with a linear model and then a random
forest under cross-validation, tune the whole pipeline, report the test RMSE with
a bootstrap interval, inspect what the model relies on and where it fails, then
save it, reload it and predict on raw new districts. A 4,000-district subsample
keeps the run under the time budget; the numbers track the book's.
"""
import os
import tarfile
import tempfile
import urllib.request
import numpy as np
import pandas as pd


def main() -> None:
    housing = load_housing()
    print(f"loaded {len(housing):,} districts, {housing.shape[1]} columns; "
          f"missing total_bedrooms: {int(housing['total_bedrooms'].isna().sum())}")

    # ---- 1. Split first, then look ----
    train_set, test_set = stratified_split(housing)
    print(f"train {len(train_set):,} / test {len(test_set):,} (stratified on income category)")
    corr = explore_correlations(add_ratio_features(train_set))
    print("top correlations with value:", ", ".join(f"{k} {v:+.2f}" for k, v in corr.head(3).items()))

    sample = train_set.sample(4000, random_state=42)
    X, y = split_features_labels(add_ratio_features(sample))

    # ---- 2. Baseline, then models under cross-validation ----
    dummy = dummy_baseline_rmse(X, y)
    print(f"\ndummy (predict the mean)   RMSE {dummy:>10,.0f}")
    lin = cross_val_rmse(linear_model(build_preprocessing()), X, y)
    print(f"linear regression   CV   RMSE {lin['mean']:>10,.0f}  (+/- {lin['std']:,.0f})")
    forest = forest_model(build_preprocessing(), n_estimators=30)
    fr = cross_val_rmse(forest, X, y)
    train_r = rmse(y, forest.fit(X, y).predict(X))
    print(f"random forest       CV   RMSE {fr['mean']:>10,.0f}  (+/- {fr['std']:,.0f}); "
          f"on its own training data {train_r:,.0f} -> it overfits, trust the CV number")

    # ---- 3. Tune the whole pipeline ----
    search = random_search(forest_model(build_preprocessing(), n_estimators=30), X, y, n_iter=3)
    print(f"\nrandom search best CV RMSE {-search.best_score_:,.0f} with {search.best_params_}")

    # ---- 4. The test set, once ----
    final_model = search.best_estimator_
    X_test, y_test = split_features_labels(add_ratio_features(test_set))
    pred = final_model.predict(X_test)
    low, high = bootstrap_rmse_ci(y_test, pred, n_boot=200)
    print(f"TEST RMSE {test_rmse(final_model, test_set):,.0f}   95% bootstrap CI [{low:,.0f}, {high:,.0f}]")
    print("what it relies on:", ", ".join(f"{n} {i:.3f}" for i, n in feature_importances(search, k=4)))
    worst = worst_errors(final_model, test_set, k=3)
    print("worst misses (actual / predicted):",
          ", ".join(f"{a:,.0f} / {p:,.0f}" for a, p in zip(worst["actual"], worst["predicted"])))

    # ---- 5. Ship ----
    path = os.path.join(tempfile.gettempdir(), "california_housing_model.pkl")
    served = save_and_reload(final_model, path)
    districts = [
        {"longitude": -122.2, "latitude": 37.8, "housing_median_age": 30.0, "total_rooms": 2000.0,
         "total_bedrooms": 400.0, "population": 1000.0, "households": 380.0, "median_income": 5.5,
         "ocean_proximity": "NEAR BAY"},
        {"longitude": -119.5, "latitude": 36.5, "housing_median_age": 20.0, "total_rooms": 1500.0,
         "total_bedrooms": None, "population": 900.0, "households": 300.0, "median_income": 2.1,
         "ocean_proximity": "INLAND"},
        {"longitude": -118.4, "latitude": 34.0, "housing_median_age": 40.0, "total_rooms": 2500.0,
         "total_bedrooms": 450.0, "population": 1100.0, "households": 420.0, "median_income": 9.0,
         "ocean_proximity": "<1H OCEAN"},
    ]
    preds = predict_new(served, districts)
    for d, p in zip(districts, preds):
        print(f"  {d['ocean_proximity']:<10} income {d['median_income']:>4}  ->  ${p:,.0f}")
    print(f"\nsaved to {os.path.basename(path)}; reloaded model reproduces the test score: "
          f"{abs(test_rmse(served, test_set) - test_rmse(final_model, test_set)) < 1e-6}")


if __name__ == "__main__":
    main()

