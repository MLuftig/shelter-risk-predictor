# Shelter Return Risk Predictor

An interactive Streamlit app predicting the likelihood that an adopted animal will be returned to the shelter within 30 days, comparing two independently-trained Random Forest models side by side — one trained on real Austin Animal Center data, one on real Bloomington Animal Care & Control data. Both models evaluate the exact same animal profile, so any difference in predicted risk reflects a genuine difference in what each shelter's own historical data says actually drives returns.

**Live App:** [shelter-risk-predictor.streamlit.app](https://shelter-risk-predictor.streamlit.app/)

**Full methodology, model correction notes, and cross-shelter analysis:** [Animal Shelter Recidivism Prediction](https://github.com/MLuftig/animal-shelter-recidivism-prediction) — this repo is the deployed app only; the underlying research (the target-leakage correction, calibration, and the full cross-shelter comparison) lives there.

## What It Shows
Enter an animal's length of stay, age, species, breed group, and intake reason, and the app runs that same profile through both cities' independently-trained models, showing:
- Austin's predicted return risk
- Bloomington's predicted return risk
- A clear side-by-side comparison, making the cross-shelter divergence tangible for a specific, concrete case rather than an abstract chart

## Why the Two Models Disagree
The two cities' models learned meaningfully different patterns from their own shelters' history, not just different numbers from the same underlying pattern:

- **At Austin, age and species are the dominant predictors** — 79% combined feature importance.
- **At Bloomington, length of stay dominates instead** — 62% importance, with age and species playing a much smaller role.

This means the same animal profile can genuinely receive very different risk assessments from the two models — not due to a bug or inconsistency, but because each model faithfully reflects what actually predicted returns at its own shelter. See the [full analysis](https://github.com/MLuftig/animal-shelter-recidivism-prediction) for the complete cross-shelter generalization test (AUC dropped from 0.71 to 0.64 when Austin's model was tested directly on Bloomington data) that motivated training a second, independent model rather than assuming one shelter's model would transfer.

## Running Locally
```bash
git clone https://github.com/MLuftig/shelter-risk-predictor.git
cd shelter-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
`Python`, `Streamlit`, `Scikit-Learn`, `Pandas`, `Joblib`

## Repository Structure
```text
├── app.py                          # Streamlit application
├── recidivism_model.pkl            # Calibrated Austin Random Forest classifier
├── bloomington_recidivism_model.pkl # Calibrated Bloomington Random Forest classifier
├── requirements.txt
└── README.md
```

Both models are pre-trained and validated in the companion [analysis repository](https://github.com/MLuftig/animal-shelter-recidivism-prediction) — see that repo for the target-leakage correction, probability calibration fix, and the full feature-importance breakdown for both cities.
