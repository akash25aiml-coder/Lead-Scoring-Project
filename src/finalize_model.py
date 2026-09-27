"""
Final model selection, ONE-TIME test set evaluation, SHAP explainability,
and artifact export for the Streamlit app.
"""
import joblib
import numpy as np
import shap
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, recall_score

artifacts = joblib.load("artifacts.joblib")
tuned = joblib.load("tuned_models.joblib")

X_train, y_train = artifacts["X_train"], artifacts["y_train"]
X_test, y_test = artifacts["X_test"], artifacts["y_test"]

final_model = tuned["XGBoost"]  # winner: best recall, PR-AUC, ROC-AUC after tuning

# ---------- ONE-TIME test evaluation ----------
test_pred = final_model.predict(X_test)
test_proba = final_model.predict_proba(X_test)[:, 1]

print("=== FINAL MODEL: XGBoost (tuned) — TEST SET (touched once) ===")
print(classification_report(y_test, test_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, test_proba), 3))
print("PR-AUC :", round(average_precision_score(y_test, test_proba), 3))
print("Recall (converted):", round(recall_score(y_test, test_pred), 3))

# ---------- Business KPI: top-decile conversion lift ----------
order = np.argsort(test_proba)[::-1]
top_decile_n = int(0.1 * len(test_proba))
top_decile_idx = order[:top_decile_n]
top_decile_conv_rate = y_test.values[top_decile_idx].mean()
baseline_conv_rate = y_test.mean()
lift = top_decile_conv_rate / baseline_conv_rate
print(f"\nTop-decile conversion rate: {top_decile_conv_rate:.3f}")
print(f"Baseline conversion rate  : {baseline_conv_rate:.3f}")
print(f"Conversion lift           : {lift:.2f}x  (target: >=2x)")

# ---------- SHAP feature importance ----------
prep = final_model.named_steps["prep"]
clf = final_model.named_steps["clf"]
X_train_transformed = prep.transform(X_train)
feat_names = prep.get_feature_names_out()

explainer = shap.TreeExplainer(clf)
sample = X_train_transformed[np.random.RandomState(42).choice(X_train_transformed.shape[0], 500, replace=False)]
shap_values = explainer.shap_values(sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:10]
print("\n=== Top 10 features by mean |SHAP value| ===")
for i in top_idx:
    print(f"{feat_names[i]:50s} {mean_abs_shap[i]:.4f}")

# ---------- Export for Streamlit ----------
joblib.dump(final_model, "final_model.joblib")
print("\nSaved final_model.joblib for the Streamlit app.")

# ---------- Bundle everything the app needs for consistent inference ----------
# (single-lead AND batch scoring must apply IDENTICAL preprocessing to what
# training used — bundling freq_map + schema prevents silent train/serve skew)
NUM_BOUNDS = {c: (float(X_train[c].min()), float(X_train[c].max())) for c in artifacts["num_cols"]}
CAT_VALUES = {c: sorted(X_train[c].dropna().unique().tolist()) for c in artifacts["cat_cols"]}

bundle = {
    "model": final_model,
    "freq_map": artifacts["freq_map"],
    "num_cols": artifacts["num_cols"],
    "cat_cols": artifacts["cat_cols"],
    "num_bounds": NUM_BOUNDS,
    "cat_values": CAT_VALUES,
    "test_metrics": {
        "recall": round(recall_score(y_test, test_pred), 3),
        "pr_auc": round(average_precision_score(y_test, test_proba), 3),
        "roc_auc": round(roc_auc_score(y_test, test_proba), 3),
        "top_decile_lift": round(lift, 2),
    },
}
joblib.dump(bundle, "model_bundle.joblib")
print("\nSaved model_bundle.joblib (model + freq_map + schema + bounds + metrics)")
