import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Shelter Return Risk Predictor", page_icon="🐾", layout="wide")

# ---- Load models ----
@st.cache_resource
def load_models():
    austin_model = joblib.load("recidivism_model.pkl")
    bloomington_model = joblib.load("bloomington_recidivism_model.pkl")
    return austin_model, bloomington_model

austin_model, bloomington_model = load_models()

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
    "within 30 days**, comparing two independently-trained Random Forest models side "
    "by side -- one trained on real Austin Animal Center data, one on real Bloomington "
    "Animal Care & Control data. Both use the exact same animal profile, so any "
    "difference in predicted risk reflects a genuine difference in what each shelter's "
    "own historical data says drives returns."
)
st.info(
    "⚠️ At Austin, **age and species** are the dominant predictors (79% combined "
    "importance). At Bloomington, **length of stay** dominates instead (62% importance) "
    "-- the two models learned meaningfully different patterns from their own shelters' "
    "history. See the "
    "[Animal Shelter Recidivism Prediction](https://github.com/MLuftig/animal-shelter-recidivism-prediction) "
    "repo for the full cross-shelter analysis."
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

# ---- Predict, both models ----
st.divider()
st.subheader("Prediction")

pred_col1, pred_col2 = st.columns(2)

for col, city_name, model in [(pred_col1, "Austin, TX", austin_model), (pred_col2, "Bloomington, IN", bloomington_model)]:
    with col:
        st.markdown(f"### {city_name}")
        proba = model.predict_proba(X)[0][1]
        risk_pct = proba * 100

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

st.divider()
with st.expander("How this model works"):
    st.markdown(
        """
        These are two independently-trained **Random Forest classifiers**, one per
        shelter, each predicting whether an animal will be returned within 30 days
        of adoption.

        **Austin** — trained on 54,408 historical adoption records.
        Feature importance: age at first visit (**58.7%**), species (**20.7%**),
        length of stay and other features (remaining ~20.6%). Recall: **76%**.

        **Bloomington** — trained on 7,199 historical adoption records.
        Feature importance: length of stay (**61.9%**), age at first visit
        (**17.3%**), species (**9.4%**), other features (remaining ~11.4%).
        Recall: **74%**.

        Both models were tested for cross-shelter transfer: applying Austin's
        model directly to Bloomington's data only partially worked (AUC dropped
        from 0.71 to 0.64, recall fell sharply) — the two shelters' own trained
        models above perform far better on their own data than either model does
        on the other's. That's the whole reason this tool shows both separately
        rather than picking one as "the" model.

        An earlier version of the Austin model had a flawed target definition
        (it only measured return *speed* among animals that had already
        returned, not whether a return happened at all), which had produced a
        misleading result pointing to length of stay as Austin's dominant
        driver. The corrected model above reflects a properly rebuilt binary
        classification target — used consistently for both cities' models.

        Both models were optimized for **recall over accuracy** — prioritizing
        catching true at-risk cases even at the cost of some false alarms, since
        missing a genuine risk case is more costly than an unnecessary check-in
        call. Both models' output probabilities are calibrated (isotonic
        regression) so the displayed risk percentages reflect real-world
        frequencies rather than an inflated relative score.
        """
    )

st.caption("Model and analysis: [github.com/MLuftig/animal-shelter-recidivism-prediction](https://github.com/MLuftig/animal-shelter-recidivism-prediction)")
