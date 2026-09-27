"""Generate saved visual evidence (confusion matrix, SHAP summary) as PNGs
for the README — a results table is a claim, a chart is evidence."""
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

artifacts = joblib.load("artifacts.joblib")
bundle = joblib.load("model_bundle.joblib")
model = bundle["model"]
X_test, y_test = artifacts["X_test"], artifacts["y_test"]
X_train = artifacts["X_train"]

# ---- Confusion matrix ----
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, display_labels=["Not Converted", "Converted"],
    cmap="Blues", ax=ax,
)
ax.set_title("XGBoost (tuned) — Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig("../reports/confusion_matrix.png", dpi=150)
plt.close()
print("Saved reports/confusion_matrix.png")

# ---- SHAP summary plot ----
prep = model.named_steps["prep"]
clf = model.named_steps["clf"]
X_sample = X_train.sample(500, random_state=42)
X_transformed = prep.transform(X_sample)
feat_names = prep.get_feature_names_out()

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(
    shap_values, X_transformed, feature_names=feat_names,
    max_display=12, show=False, plot_size=(10, 7),
)
fig = plt.gcf()
fig.suptitle("Top Features by SHAP Impact", fontsize=13, y=1.02)
ax = plt.gca()
ax.set_xlabel("SHAP value (impact on conversion probability)", fontsize=10, labelpad=10)
plt.subplots_adjust(bottom=0.12)
plt.savefig("../reports/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved reports/shap_summary.png")
