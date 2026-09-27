"""
Lead Scoring Dashboard - Streamlit product interface.

Two modes:
  1. Score a single lead via a form (sales rep checking one prospect)
  2. Batch-score a CSV of leads (the realistic use case: hundreds of leads
     dumped from a CRM export, scored and ranked in one pass)

Loads model_bundle.joblib: the tuned XGBoost pipeline + the exact
preprocessing schema (rare-category map, valid values, numeric bounds) it
was trained with, so single-lead and batch scoring can never silently drift
apart from each other or from training.
"""
import io
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

import os

st.set_page_config(page_title="Lead Scoring", page_icon="🎯", layout="centered")

# Absolute path relative to this script's own location — works regardless
# of whether Streamlit Cloud's working directory is repo root or app/.
# (Previously a bare relative path — worked locally, silent risk on deploy.)
BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bundle.joblib")


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_resource
def load_explainer(_clf):
    return shap.TreeExplainer(_clf)


bundle = load_bundle()
model = bundle["model"]
freq_map = bundle["freq_map"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]
num_bounds = bundle["num_bounds"]
cat_values = bundle["cat_values"]
metrics = bundle["test_metrics"]
explainer = load_explainer(model.named_steps["clf"])

ALL_COLS = num_cols + cat_cols


def apply_rare_grouping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, rare_set in freq_map.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "Other" if x in rare_set else x)
    return df


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_rare_grouping(df)
    proba = model.predict_proba(df[ALL_COLS])[:, 1]
    out = df.copy()
    out["conversion_probability"] = proba
    out["tier"] = pd.cut(
        proba, bins=[-0.01, 0.40, 0.70, 1.01], labels=["Cold", "Warm", "Hot"]
    )
    return out.sort_values("conversion_probability", ascending=False)


st.title("🎯 Lead Conversion Scorer")
st.caption(
    "Predicts conversion probability so reps prioritize high-fit leads first - "
    "quantifies the same ICP-fit judgment made manually in account-based marketing."
)
st.info(
    "Trained on a public EdTech lead dataset (Kaggle), not B2B SaaS/CRM data - "
    "treat predictions as a methodology demo, not a calibrated score for a "
    "different industry. No input data is stored; each session is stateless.",
    icon="ℹ️",
)

tab_single, tab_batch = st.tabs(["Score one lead", "Batch score (CSV)"])

# ============================================================
# TAB 1 - single lead
# ============================================================
with tab_single:
    with st.form("lead_form"):
        st.subheader("Lead details")

        col1, col2 = st.columns(2)
        with col1:
            total_visits = st.number_input(
                "Total visits", 0, int(num_bounds["TotalVisits"][1]), 3,
                help=f"Training data range: {int(num_bounds['TotalVisits'][0])}-{int(num_bounds['TotalVisits'][1])}",
            )
            time_on_site = st.number_input(
                "Total time on website (seconds)", 0, int(num_bounds["Total Time Spent on Website"][1]), 250,
            )
            page_views = st.number_input(
                "Page views per visit", 0.0, float(num_bounds["Page Views Per Visit"][1]), 2.0, step=0.5,
            )
        with col2:
            activity_score = st.slider(
                "Activity score (internal engagement index)",
                int(num_bounds["Asymmetrique Activity Score"][0]),
                int(num_bounds["Asymmetrique Activity Score"][1]), 14,
            )
            profile_score = st.slider(
                "Profile score (internal fit index)",
                int(num_bounds["Asymmetrique Profile Score"][0]),
                int(num_bounds["Asymmetrique Profile Score"][1]), 16,
            )

        lead_origin = st.selectbox("Lead origin", cat_values["Lead Origin"])
        lead_source = st.selectbox("Lead source", cat_values["Lead Source"])
        last_activity = st.selectbox("Last activity", cat_values["Last Activity"])
        last_notable_activity = st.selectbox("Last notable activity", cat_values["Last Notable Activity"])
        occupation = st.selectbox("Current occupation", cat_values["What is your current occupation"])
        specialization = st.selectbox("Specialization", cat_values["Specialization"])
        what_matters = st.selectbox("What matters most in choosing a course", cat_values["What matters most to you in choosing a course"])
        city = st.selectbox("City", cat_values["City"])
        country = st.selectbox("Country", cat_values["Country"])
        activity_index = st.selectbox("Activity index (tier)", cat_values["Asymmetrique Activity Index"])
        profile_index = st.selectbox("Profile index (tier)", cat_values["Asymmetrique Profile Index"])
        do_not_email = st.selectbox("Opted out of email?", cat_values["Do Not Email"])
        free_copy = st.selectbox("Given free copy of 'Mastering The Interview'?", cat_values["A free copy of Mastering The Interview"])

        submitted = st.form_submit_button("Score this lead", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "TotalVisits": total_visits, "Total Time Spent on Website": time_on_site,
            "Page Views Per Visit": page_views, "Asymmetrique Activity Score": activity_score,
            "Asymmetrique Profile Score": profile_score, "Lead Origin": lead_origin,
            "Lead Source": lead_source, "Do Not Email": do_not_email, "Last Activity": last_activity,
            "Country": country, "Specialization": specialization,
            "What is your current occupation": occupation,
            "What matters most to you in choosing a course": what_matters, "City": city,
            "Asymmetrique Activity Index": activity_index, "Asymmetrique Profile Index": profile_index,
            "A free copy of Mastering The Interview": free_copy,
            "Last Notable Activity": last_notable_activity,
        }])

        scored = score_dataframe(row)
        proba = scored["conversion_probability"].iloc[0]
        tier = scored["tier"].iloc[0]
        color = {"Hot": "green", "Warm": "orange", "Cold": "red"}[tier]
        icon = {"Hot": "🔥", "Warm": "🌤️", "Cold": "❄️"}[tier]

        st.divider()
        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Conversion probability", f"{proba:.1%}")
        c2.markdown(f"### :{color}[{icon} {tier}]")
        st.progress(float(proba))

        row_transformed = model.named_steps["prep"].transform(apply_rare_grouping(row)[ALL_COLS])
        feat_names = model.named_steps["prep"].get_feature_names_out()
        shap_vals = explainer.shap_values(row_transformed)[0]
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]

        st.subheader("Top factors for this prediction")
        for i in top_idx:
            name = feat_names[i].replace("num__", "").replace("cat__", "")
            direction = "pushes conversion UP" if shap_vals[i] > 0 else "pushes conversion DOWN"
            st.write(f"**{name}** - {direction} (impact: {shap_vals[i]:+.3f})")

# ============================================================
# TAB 2 - batch scoring
# ============================================================
with tab_batch:
    st.subheader("Score a CSV of leads")
    st.write(
        "Upload a CSV with the same columns as the training schema. "
        "Every row gets a conversion probability and a Hot/Warm/Cold tier, "
        "sorted highest-probability first - ready for a sales team to work "
        "top-down."
    )

    template = pd.DataFrame([{c: "" for c in ALL_COLS}])
    st.download_button(
        "Download empty CSV template",
        template.to_csv(index=False).encode(),
        file_name="lead_scoring_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload leads CSV", type="csv")

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read the CSV: {e}")
            st.stop()

        missing_cols = [c for c in ALL_COLS if c not in raw.columns]
        if missing_cols:
            st.error(
                f"Missing required column(s): {', '.join(missing_cols)}. "
                f"Download the template above to see the exact schema."
            )
            st.stop()

        raw = raw.replace("Select", pd.NA)
        n_input_rows = len(raw)

        issues = []
        valid_mask = pd.Series(True, index=raw.index)
        for c in num_cols:
            lo, hi = num_bounds[c]
            col_num = pd.to_numeric(raw[c], errors="coerce")
            raw[c] = col_num
            bad = col_num.isna() | (col_num < lo * 0.5) | (col_num > hi * 1.5)
            if bad.sum():
                issues.append(f"{int(bad.sum())} row(s) have out-of-range or non-numeric `{c}`")
            valid_mask &= ~bad

        clean = raw[valid_mask].copy()
        dropped = n_input_rows - len(clean)

        if issues:
            with st.expander(f"{dropped} row(s) skipped - validation details", expanded=dropped > 0):
                for msg in issues:
                    st.write("- " + msg)

        if clean.empty:
            st.error("No valid rows to score after validation.")
            st.stop()

        for c in cat_cols:
            clean[c] = clean[c].fillna("Not Provided")

        results = score_dataframe(clean)

        st.success(f"Scored {len(results)} of {n_input_rows} uploaded leads.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Hot leads", int((results["tier"] == "Hot").sum()))
        c2.metric("Warm leads", int((results["tier"] == "Warm").sum()))
        c3.metric("Cold leads", int((results["tier"] == "Cold").sum()))

        display_cols = ["conversion_probability", "tier"] + [c for c in ALL_COLS if c in results.columns][:6]
        st.dataframe(
            results[display_cols].style.format({"conversion_probability": "{:.1%}"}),
            use_container_width=True,
        )

        csv_buf = io.StringIO()
        results.to_csv(csv_buf, index=False)
        st.download_button(
            "Download scored results (CSV)",
            csv_buf.getvalue().encode(),
            file_name="scored_leads.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
with st.expander("Model details"):
    st.markdown(
        f"- **Model:** XGBoost (tuned via RandomizedSearchCV, 5-fold CV)\n"
        f"- **Test-set performance:** Recall {metrics['recall']} · "
        f"PR-AUC {metrics['pr_auc']} · ROC-AUC {metrics['roc_auc']}\n"
        f"- **Business KPI:** top-decile scored leads convert at "
        f"{metrics['top_decile_lift']}x the baseline rate\n"
        f"- **Leakage-audited:** 3 outcome-correlated columns removed before training\n"
        f"- Trained on the public Kaggle Lead Scoring dataset (X Education)\n"
        f"- No uploaded data is stored - each session is stateless"
    )
