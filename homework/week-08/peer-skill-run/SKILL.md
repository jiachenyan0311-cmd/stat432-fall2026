---
name: LDA/QDA Classification Analysis
description: Build and evaluate reproducible classification models using ordinary LDA and QDA, and choose a model by training-only cross-validation.
---

# LDA/QDA Classification Analysis

1. Check response labels, predictor types, missing values, class counts, and whether covariance estimates are feasible. Explain any excluded rows or features. Never use outcome information to preprocess predictors.
2. Set and report a random seed. Make a stratified train/test split before model comparison. Hold the test set untouched until the final model has been selected. Use an approximately 70/30 split unless the user specifies otherwise.
3. Compare ordinary LDA and QDA as mandatory candidates. If QDA has singular within-class covariance matrices, document this and use an appropriate training-only feature reduction or PCA-QDA candidate; do not silently omit QDA. Additional candidates are optional.
4. Use identical stratified cross-validation folds for all candidates within the training data (default: 5 folds). Refit every preprocessing step, including imputation, scaling, feature selection, and PCA, within each training fold. Apply those fitted steps to that fold's validation data. Never select settings using test data.
5. Calculate fold-level misclassification rates; compare candidates by mean validation error, breaking ties in favor of the simpler model. Report CV mean errors and the selected settings. If a model fails on a fold, record the failure rather than replacing it with an invented score.
6. Refit the selected pipeline on the entire training set. Predict the reserved test set exactly once. Report test misclassification rate, test sample size, and confusion matrix. Distinguish CV selection results from final held-out evaluation.
7. Check for class imbalance, small class sample sizes, covariance instability, and uncertainty from one split. Avoid claims of universal superiority.

Create a legible statistical report of **at most one printed page**. Include: a short data/split description and random seed; a compact table of candidate models, CV mean misclassification rates, and selected settings; one clearly labeled figure showing comparative CV performance; the selected model; held-out test misclassification rate and numerical model-fitting results; and a brief interpretation with limitations. A small confusion matrix may be included if space permits. Keep tables and plots readable when printed. Supply reproducible code separately from the one-page report when requested.

Invocation: Only apply this skill when the user explicitly requests it. Do not automatically apply it to other tasks or analyses.