import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Shelter Return Risk Predictor", page_icon="🐾", layout="centered")

# ---- Load model ----
@st.cache_resource
def load_model():
    return joblib.load("recidivism_model.pkl")

model = load_model()

FEATURE_ORDER = [
    "los_days", "age_at_first_visit",
    "spp_k9", "spp_other", "spp_wildlife",
    "akc_group_hound", "akc_group_mixed_breed", "akc_group_non_sporting",
    "akc_group_sporting", "akc_group_terrier", "akc_group_toy", "akc_group_working",
    "first_reason_medical", "first_reason_other", "first_reason_routine",
]

# ---- Header ----
st.title("🐾 Shelter Return Risk Predictor")
st.markdown(
    "Predicts the likelihood that an adopted animal will be **returned to the shelter "
    "within 30 days**, based on a Random Forest model trained on Austin Animal Center "
    "intake/outcome records. Built as part of the "
    "[Animal Shelter Recidivism Prediction](https://github.com/MLuftig/animal-shelter-recidivism-prediction) project."
)
st.divider()

# ---- Inputs ----
col1, col2 = st.columns(2)

with col1:
    los_days = st.slider("Length of stay before adoption (days)", 0, 120, 14)
    age_at_first_visit = st.slider("Age at first shelter visit (years)", 0.0, 20.0, 2.0, step=0.5)
    species = st.selectbox("Species", ["Feline", "Canine", "Wildlife", "Other"])

with col2:
    breed_group = st.selectbox(
        "Breed group (dogs only — ignored for other species)",
        ["Herding", "Hound", "Mixed Breed", "Non-Sporting", "Sporting", "Terrier", "Toy", "Working"],
    )
    intake_reason = st.selectbox("Reason for original intake", ["Stray", "Medical", "Routine", "Other"])

# ---- Build feature vector ----
row = {f: 0 for f in FEATURE_ORDER}
row["los_days"] = los_days
row["age_at_first_visit"] = age_at_first_visit

species_map = {"Canine": "spp_k9", "Other": "spp_other", "Wildlife": "spp_wildlife"}
if species in species_map:
    row[species_map[species]] = 1

breed_map = {
    "Hound": "akc_group_hound", "Mixed Breed": "akc_group_mixed_breed",
    "Non-Sporting": "akc_group_non_sporting", "Sporting": "akc_group_sporting",
    "Terrier": "akc_group_terrier", "Toy": "akc_group_toy", "Working": "akc_group_working",
}
if breed_group in breed_map:
    row[breed_map[breed_group]] = 1

reason_map = {"Medical": "first_reason_medical", "Other": "first_reason_other", "Routine": "first_reason_routine"}
if intake_reason in reason_map:
    row[reason_map[intake_reason]] = 1

X = pd.DataFrame([row])[FEATURE_ORDER]

# ---- Predict ----
proba = model.predict_proba(X)[0][1]
risk_pct = proba * 100

st.divider()
st.subheader("Prediction")

if risk_pct >= 50:
    st.error(f"⚠️ **High Risk of Return — {risk_pct:.1f}%**")
    st.markdown("Consider a proactive follow-up check-in during the first month post-adoption.")
elif risk_pct >= 25:
    st.warning(f"🟡 **Moderate Risk of Return — {risk_pct:.1f}%**")
    st.markdown("Worth a routine check-in call within the first two weeks.")
else:
    st.success(f"✅ **Low Risk of Return — {risk_pct:.1f}%**")
    st.markdown("No additional intervention likely needed.")

st.progress(min(int(risk_pct), 100))

with st.expander("How this model works"):
    st.markdown(
        """
        This is a **Random Forest classifier** trained on 4,916 historical Austin Animal Center
        records. It predicts whether an animal will be returned to the shelter within 30 days
        of adoption.

        **Feature importance in this model:**
        - Length of stay: **76.5%** — by far the strongest predictor
        - Age at first visit: **19.0%**
        - Species, breed group, and intake reason: remaining ~4.5% combined

        The model was optimized for **recall over accuracy** — it prioritizes catching true
        at-risk cases (61% recall) even at the cost of some false alarms, since missing a
        genuine risk case is more costly than an unnecessary check-in call.
        """
    )

st.caption("Model and analysis: [github.com/MLuftig/animal-shelter-recidivism-prediction](https://github.com/MLuftig/animal-shelter-recidivism-prediction)")
