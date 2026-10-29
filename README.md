# ⚽ Bundesliga Match Outcome Predictor

A machine learning project that predicts 1X2 outcomes (Home Win / Draw / Away Win) 
for Bundesliga matches using historical team statistics and Expected Goals (xG).

**[👉 Try the live dashboard here](LINK_A_STREAMLIT_AQUI)** *(cuando lo despliegues)*

## 📊 Results

- **Accuracy**: 53.6% on the 2025/26 test season (baseline: 44%)
- **Log Loss**: 0.98
- **Note**: Predicting draws is inherently difficult; the model reflects state-of-the-art limitations in football prediction.

## 🚀 Features

- **Data Pipeline**: Automated data fetching from Understat via `soccerdata`.
- **Feature Engineering**: Rolling averages of xG (for/against) over the last 5 matches, separated by home/away context. Implements league-average imputation for newly promoted teams.
- **Dual-Model Approach**: Compares a Multinomial Logistic Regression (direct 1X2 classification) against a Poisson Regression (modeling expected goals to derive scoreline matrices).
- **Interactive Dashboard**: Built with Streamlit, featuring:
  - Individual match predictions with scoreline probability heatmaps.
  - Historical season simulation (Monte Carlo) validated against actual standings.
  - Pre-season forecast for the upcoming 2026/27 season.

## 🛠️ Tech Stack

- **Data**: `pandas`, `numpy`, `soccerdata` (Understat)
- **Machine Learning**: `scikit-learn` (Logistic Regression, Poisson Regressor), `scipy`
- **Frontend**: `streamlit`, `matplotlib`, `seaborn`
- **Testing**: `pytest`

## 📂 Project Structure

```text
bundesliga-predictor/
├── app/
│   └── dashboard.py          # Streamlit application
├── data/
│   ├── raw/                  # Downloaded Understat data
│   └── processed/            # Engineered features
├── models/                   # Trained .pkl models
├── src/
│   ├── data_pipeline.py      # Download raw data
│   ├── features.py           # Feature engineering
│   ├── train.py              # Training & evaluation (holdout)
│   ├── train_production.py   # Training on all data
│   └── evaluate.py           # Model comparison script
├── tests/                    # Unit tests
└── requirements.txt