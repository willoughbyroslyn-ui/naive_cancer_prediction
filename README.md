# Cervical Cancer Risk Prediction (Naive Bayes)

A Gaussian Naive Bayes classifier predicting cervical cancer diagnosis (`Dx:Cancer`) from the
[UCI Cervical Cancer Risk Factors dataset](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)
(858 patients, 36 features). Completed as a student research project in summer 2025.

## What the notebook covers

- Cleaning `?` placeholders, median imputation, one-hot encoding, and robust scaling
- 70/30 train/test split and a Gaussian Naive Bayes model
- Evaluation with a confusion matrix, precision, recall, specificity, ROC AUC, and 10-fold cross-validation

## Key finding

The dataset is highly imbalanced: only 18 of 858 patients (2.1%) have a cancer diagnosis, and 7 of the 258 test cases.

| Metric (cancer = positive class) | Value |
|---|---|
| Accuracy | 77.5% (below the 97.3% "always predict no cancer" baseline) |
| Sensitivity | 14% (1 of 7 cancer cases detected) |
| Specificity | 79% |
| Precision | 2% |
| ROC AUC | 0.47 test, 0.61 cross-validated |

The model performs close to chance at identifying cancer cases. The main lesson is that overall accuracy is
misleading on imbalanced medical data, and that sensitivity, precision, and precision-recall metrics matter more.
Several features (Dx, Dx:HPV, Dx:CIN, Hinselmann, Schiller, Citology, Biopsy) are diagnostic test results rather
than risk factors and should be excluded from a true risk model.

## Next steps

Stratified splitting, class weights or resampling (e.g., SMOTE), threshold tuning, comparison with logistic
regression and random forests, and removing diagnostic-result features.

## Run it

Download `risk_factors_cervical_cancer.csv` from the UCI link above, update the path in the first cell, and run
`Naive_Bayes.ipynb` (pandas, scikit-learn, category_encoders, seaborn).

The implementation follows the structure of
[this Kaggle Naive Bayes tutorial](https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python).
