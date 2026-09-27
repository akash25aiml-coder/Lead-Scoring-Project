"""
Baseline model: Logistic Regression on the cleaned Leads dataset.
Establishes the floor that Random Forest / XGBoost (models 2 & 3) must beat.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, recall_score,
)

from data_prep import load_and_clean, split_features

df = load_and_clean("../data/Leads.csv")
X, y, num_cols, cat_cols = split_features(df)

# Stratified split — preserves 61.5/38.5 class ratio in both sets.
# Test set touched ONLY at final evaluation, never during tuning.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

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

baseline = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

baseline.fit(X_train, y_train)
val_pred = baseline.predict(X_val)
val_proba = baseline.predict_proba(X_val)[:, 1]

print("\n--- Baseline: Logistic Regression (validation set) ---")
print(classification_report(y_val, val_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_val, val_proba), 3))
print("PR-AUC :", round(average_precision_score(y_val, val_proba), 3))
print("Recall (converted class):", round(recall_score(y_val, val_pred), 3))
