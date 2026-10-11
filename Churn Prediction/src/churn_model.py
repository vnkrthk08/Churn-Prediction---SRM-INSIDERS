"""
Customer churn prediction: Random Forest vs XGBoost
Dataset: IBM Telco Customer Churn (7,043 customers)

Run:  python src/churn_model.py
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

SEED = 42
DATA_PATH = "data/telco.csv"
OUT = "outputs"


def load_and_clean(path):
    df = pd.read_csv(path)

    # TotalCharges is stored as text; new customers (tenure 0) have a blank
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # one-hot encode every text column
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df


def evaluate(name, model, X_test, y_test, train_time):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "train_time_s": train_time,
    }, proba, pred


def main():
    df = load_and_clean(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    print(f"Rows: {len(df)}, features: {X.shape[1]}, churn rate: {y.mean():.1%}")

    # stratify keeps the same churn ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    imbalance = (y_train == 0).sum() / (y_train == 1).sum()

    # ---- Random Forest ----
    rf_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", 0.3, 0.5],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        rf_params, n_iter=20, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1)
    t0 = time.time()
    rf_search.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf = rf_search.best_estimator_

    # ---- XGBoost ----
    xgb_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1, 3, 5],
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(scale_pos_weight=imbalance, eval_metric="logloss",
                      random_state=SEED, n_jobs=-1),
        xgb_params, n_iter=20, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1)
    t0 = time.time()
    xgb_search.fit(X_train, y_train)
    xgb_time = time.time() - t0
    xgb = xgb_search.best_estimator_

    rf_res, rf_proba, rf_pred = evaluate("Random Forest", rf, X_test, y_test, rf_time)
    xgb_res, xgb_proba, xgb_pred = evaluate("XGBoost", xgb, X_test, y_test, xgb_time)
    rf_res["cv_auc"] = rf_search.best_score_
    xgb_res["cv_auc"] = xgb_search.best_score_

    results = pd.DataFrame([rf_res, xgb_res]).set_index("model")
    print("\n", results.round(4))
    results.round(4).to_csv(f"{OUT}/results.csv")
    with open(f"{OUT}/best_params.json", "w") as f:
        json.dump({"random_forest": rf_search.best_params_,
                   "xgboost": {k: (v.item() if hasattr(v, "item") else v)
                               for k, v in xgb_search.best_params_.items()}}, f, indent=2, default=str)

    # ---- plots ----
    plt.figure(figsize=(6, 5))
    for name, proba in [("Random Forest", rf_proba), ("XGBoost", xgb_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/roc_curve.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name, pred in [(axes[0], "Random Forest", rf_pred), (axes[1], "XGBoost", xgb_pred)]:
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                               display_labels=["Stay", "Churn"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, model in [(axes[0], "Random Forest", rf), (axes[1], "XGBoost", xgb)]:
        imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(10).sort_values()
        imp.plot.barh(ax=ax, color="#4C72B0")
        ax.set_title(f"{name} - top 10 features")
    plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance.png", dpi=150); plt.close()

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results[metrics].T.plot.bar(figsize=(8, 4.5), rot=0, color=["#55A868", "#C44E52"])
    plt.ylim(0, 1); plt.title("Test set metrics"); plt.tight_layout()
    plt.savefig(f"{OUT}/metrics_comparison.png", dpi=150); plt.close()

    # churn rate by contract (EDA)
    raw = pd.read_csv(DATA_PATH)
    (raw.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean())
        .plot.bar(rot=0, color="#8172B2", figsize=(6, 4), title="Churn rate by contract type"))
    plt.tight_layout(); plt.savefig(f"{OUT}/eda_contract.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
