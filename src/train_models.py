"""
Full model comparison: Logistic Regression -> Random Forest -> XGBoost.
Fixes applied (per red-flag review):
  1. 5-fold stratified CV instead of single split -> mean +/- std reported
  2. Rare categories (<1%) grouped into "Other", fit on TRAIN ONLY
  3. Consistent imbalance handling across all 3 models
Test set touched exactly once, at the very end, after tuning is locked.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, recall_score
)

from data_prep import load_and_clean, split_features, fit_rare_category_map, group_rare_categories, HIGH_CARD_COLS

RANDOM_STATE = 42

# ---------- 1. Load, split (70/15/15, stratified) ----------
df = load_and_clean("../data/Leads.csv")
X, y, num_cols, cat_cols = split_features(df)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

# ---------- 2. Rare-category grouping fit on TRAIN ONLY (flag #2) ----------
freq_map = fit_rare_category_map(X_train, cols=HIGH_CARD_COLS, threshold=0.01)
X_train = group_rare_categories(X_train, freq_map, cols=HIGH_CARD_COLS)
X_val = group_rare_categories(X_val, freq_map, cols=HIGH_CARD_COLS)
X_test = group_rare_categories(X_test, freq_map, cols=HIGH_CARD_COLS)

# ---------- 3. Preprocessing pipeline ----------
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Not Provided")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # for XGBoost scale_pos_weight

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, scale_pos_weight=pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False,
    ),
}

# ---------- 4. 5-fold CV comparison (flag #1 fix) — TRAIN SET ONLY ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"recall": "recall", "pr_auc": "average_precision", "roc_auc": "roc_auc"}

print("\n=== 5-fold CV on TRAIN set (mean +/- std) ===")
cv_results = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = scores
    print(f"\n{name}")
    for metric in scoring:
        vals = scores[f"test_{metric}"]
        print(f"  {metric:8s}: {vals.mean():.3f} +/- {vals.std():.3f}")

joblib.dump(
    {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
     "X_test": X_test, "y_test": y_test, "num_cols": num_cols, "cat_cols": cat_cols,
     "preprocess": preprocess, "pos_weight": pos_weight, "freq_map": freq_map},
    "artifacts.joblib"
)
print("\nSaved split + preprocessing artifacts to artifacts.joblib")

# ---------- 5. Hyperparameter tuning — RF & XGBoost (LogReg already near-optimal, few params) ----------
print("\n=== Hyperparameter tuning (RandomizedSearchCV, scoring=average_precision) ===")

rf_param_dist = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [8, 12, 16, None],
    "clf__min_samples_leaf": [1, 2, 5, 10],
    "clf__max_features": ["sqrt", "log2"],
}
xgb_param_dist = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [3, 4, 6, 8],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
}

tuned = {}
for name, clf, dist in [
    ("Random Forest", models["Random Forest"], rf_param_dist),
    ("XGBoost", models["XGBoost"], xgb_param_dist),
]:
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    search = RandomizedSearchCV(
        pipe, dist, n_iter=20, scoring="average_precision", cv=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned[name] = search.best_estimator_
    print(f"\n{name} best CV PR-AUC: {search.best_score_:.3f}")
    print(f"{name} best params: {search.best_params_}")

# Re-run CV for tuned models + untuned LogReg baseline for a fair final table
print("\n=== FINAL CV comparison (tuned RF/XGB vs LogReg) ===")
final_candidates = {
    "Logistic Regression": Pipeline([("prep", preprocess), ("clf", models["Logistic Regression"])]),
    "Random Forest (tuned)": tuned["Random Forest"],
    "XGBoost (tuned)": tuned["XGBoost"],
}
for name, pipe in final_candidates.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\n{name}")
    for metric in scoring:
        vals = scores[f"test_{metric}"]
        print(f"  {metric:8s}: {vals.mean():.3f} +/- {vals.std():.3f}")

joblib.dump(tuned, "tuned_models.joblib")
joblib.dump(final_candidates["Logistic Regression"], "logreg_pipeline.joblib")
print("\nSaved tuned models.")
