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
- [x] **22.** predict_new

## Results

```
loaded 20,640 districts, 10 columns; missing total_bedrooms: 207
train 16,512 / test 4,128 (stratified on income category)
top correlations with value: median_income +0.69, rooms_per_house +0.14, total_rooms +0.14

dummy (predict the mean)   RMSE    114,284
linear regression   CV   RMSE     68,866  (+/- 2,193)
random forest       CV   RMSE     54,438  (+/- 1,341); on its own training data 20,695 -> it overfits, trust the CV number

random search best CV RMSE 53,329 with {'randomforestregressor__max_features': 7, 'columntransformer__geo__n_clusters': 7}
TEST RMSE 51,887   95% bootstrap CI [49,911, 54,181]
what it relies on: log__median_income 0.284, cat__ocean_proximity_INLAND 0.130, remainder__bedrooms_ratio 0.108, remainder__people_per_house 0.092
worst misses (actual / predicted): 500,001 / 131,780, 500,001 / 167,447, 500,001 / 170,580
  NEAR BAY   income  5.5  ->  $228,063
  INLAND     income  2.1  ->  $65,877
  <1H OCEAN  income  9.0  ->  $411,150

saved to california_housing_model.pkl; reloaded model reproduces the test score: True
```
