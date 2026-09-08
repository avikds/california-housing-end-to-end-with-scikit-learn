# California Housing, End to End with Scikit-Learn

Build the classic end-to-end regression project from chapter 2 of Hands-On Machine Learning the way it is done in practice: download the California housing data, split it honestly with stratified sampling, explore it, build a leak-proof preprocessing pipeline with a custom cluster-similarity transformer, beat a dummy baseline with linear and random-forest models under cross-validation, tune with RandomizedSearchCV, report the test RMSE with a bootstrap confidence interval, inspect feature importances and the worst errors, then save the fitted pipeline, reload it and predict on raw new districts.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_housing
- [x] **2.** income_categories
- [x] **3.** stratified_split
- [x] **4.** explore_correlations
- [x] **5.** add_ratio_features
- [x] **6.** split_features_labels
- [x] **7.** ClusterSimilarity
- [x] **8.** numeric_pipeline
- [x] **9.** categorical_pipeline
- [x] **10.** build_preprocessing
- [x] **11.** rmse
- [x] **12.** dummy_baseline_rmse
- [x] **13.** cross_val_rmse
- [x] **14.** linear_model
- [x] **15.** forest_model
- [x] **16.** random_search
- [x] **17.** test_rmse
- [x] **18.** bootstrap_rmse_ci
- [x] **19.** feature_importances
- [x] **20.** worst_errors
- [x] **21.** save_and_reload
- [ ] **22.** predict_new

---

Built on Deep-ML.
