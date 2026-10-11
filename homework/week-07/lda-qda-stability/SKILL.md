---
name: lda-qda-stability
description: Explicitly invoked classification analysis using repeated cross-validation and a shuffled-label diagnostic for LDA and QDA.
---

# LDA/QDA stability analysis

Load this file only when the user explicitly supplies its path and asks to use it. Keep it outside automatically discovered skill folders.

Accept the dataset, response, predictors, coding, seed, and output location separately. Seek the lowest cross-validated misclassification rate among ordinary LDA and QDA; do not claim optimality beyond these candidates. Use empirical class priors estimated from each fitting sample. Report covariance rank failures rather than silently regularizing or dropping predictors.

## Analysis

1. Check predictor coding, finite values, class counts, and matrix dimensions. If data require imputation or other preprocessing, define a reproducible procedure fitted only on the fitting observations within each fold. Do not change the user's predictor coding.
2. Set and report a seed. Reserve approximately 20% of each class as a test set before model comparison. Keep the test observations out of all diagnostics, revisions, and selection until the final evaluation.
3. On the remaining training data, run stratified five-fold cross-validation repeated ten times. Use identical folds for both candidates. Fit each model from scratch in each fold, including any preprocessing and empirical priors. For each repetition, calculate the error over all out-of-fold predictions. Report each model's mean error and the standard deviation across repetitions; the latter describes split sensitivity, not a confidence interval.
4. Select the candidate with the lowest mean repeated-CV error. Resolve exact ties in favor of LDA. Report how often each model wins a repetition, counting ties separately. Do not select the seed or folds to obtain a preferred winner.
5. As an auxiliary placebo diagnostic, shuffle only the training labels 50 times, preserving class counts. For each shuffle, repeat the entire same CV procedure for both models and record the lower mean error. Generate stratified folds from those shuffled labels and share them between candidates. Compare the observed minimum CV error with this distribution and the training majority-class error. Do not describe chance error as necessarily 50%, use placebo results for tuning, or infer causality. Treat this as a descriptive diagnostic, not proof of predictive validity.
6. Review code and training/CV results, finalize instructions, and record the selected candidate before accessing test outcomes. Refit it on all training data and evaluate once on the reserved test set. Report test sample size, error count, misclassification rate, and a binomial Wilson 95% interval. Do not compare other models on the test set or revise after seeing its error. Subsequent report rendering must reuse saved results.

## Report

Produce a readable English report of at most one printed letter/A4 page, and verify the exported page count. Include the target, sample sizes, seed, candidate models, repeated-CV and placebo procedure, a small table of both candidates' CV mean errors and split standard deviations, and one figure showing the placebo minimum-error distribution with the observed minimum marked. Give the selected model, numerical final-test results, and a short interpretation. Explain that repeated folds overlap, permutation results are diagnostic, and a small reserved test set limits precision. Include numerical fitting details such as predictor count and pooled or class covariance dimensions. Avoid causal or clinical claims. Put the complete skill, byte count, invocation prompt, and reproducible code outside this one-page report.
