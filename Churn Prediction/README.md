# Customer Churn Prediction: Random Forest vs XGBoost

Predict whether a telecom customer will leave (churn), and compare two tree ensembles on the same data.

## Dataset
IBM Telco Customer Churn: 7,043 customers, 20 input columns (contract, tenure, charges, services), target `Churn` (26.5% Yes).

## Approach
1. Clean: convert `TotalCharges` to numeric (11 blanks for brand-new customers set to 0), drop `customerID`.
2. One-hot encode categorical columns (30 features).
3. Stratified 80/20 train/test split (seed 42).
4. Tune each model with `RandomizedSearchCV` (20 trials, 5-fold stratified CV, ROC-AUC).
5. Handle imbalance: `class_weight="balanced"` for RF, `scale_pos_weight` for XGBoost.
6. Evaluate on the held-out test set.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV AUC | Train time |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.754 | 0.524 | 0.805 | 0.635 | 0.844 | 0.848 | 81 s |
| XGBoost | 0.749 | 0.517 | 0.802 | 0.629 | **0.846** | **0.849** | **16 s** |

XGBoost has slightly better ROC-AUC (both on CV and test) and trained about 5x faster during tuning. Random Forest is marginally ahead on accuracy/F1 at the default 0.5 threshold. The gap is small; this dataset is mostly explained by a few strong features (contract type, tenure, charges), so both models hit a similar ceiling.

