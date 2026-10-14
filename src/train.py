"""
Training module for Bundesliga match prediction.

Trains two types of models:
1. Logistic Regression: predicts 1X2 (home win / draw / away win)
2. Poisson Regression: two models predicting home_goals and away_goals

Usage:
    python src/train.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


# --- Feature columns used by the models ---
FEATURE_COLS = [
    "home_xg_for_last_5",
    "home_xg_against_last_5",
    "home_xg_for_last_5_at_home",
    "home_xg_against_last_5_at_home",
    "away_xg_for_last_5",
    "away_xg_against_last_5",
    "away_xg_for_last_5_away",
    "away_xg_against_last_5_away",
]


# --- Temporal split ---
# Train on seasons 2021/22 - 2024/25, test on 2025/26
TRAIN_SEASONS = ["2122", "2223", "2324", "2425"]
TEST_SEASONS = ["2526"]


def load_features_data(filename: str = "features.csv") -> pd.DataFrame:
    """Load the processed feature data from CSV."""
    path = PROCESSED_DATA_DIR / filename
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'result' column with values:
      0 = home win
      1 = draw
      2 = away win
    """
    df = df.copy()
    conditions = [
        df["home_goals"] > df["away_goals"],    # home win
        df["home_goals"] == df["away_goals"],   # draw
        df["home_goals"] < df["away_goals"],    # away win
    ]
    choices = [0, 1, 2]
    df["result"] = np.select(conditions, choices)
    return df


def temporal_split(
    df: pd.DataFrame,
    train_seasons: list,
    test_seasons: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train and test sets by season.
    This is a temporal split, which avoids data leakage.
    """
    train = df[df["season"].astype(str).isin(train_seasons)].copy()
    test = df[df["season"].astype(str).isin(test_seasons)].copy()
    return train, test


def train_logistic_regression(X_train, y_train) -> Pipeline:
    """
    Train a multinomial Logistic Regression model with feature scaling.

    Returns a Pipeline containing both the scaler and the classifier,
    so scaling is applied automatically at prediction time.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_poisson_models(X_train, y_train_home, y_train_away):
    """
    Train two Poisson regressors:
      - One to predict home team goals (lambda_home)
      - One to predict away team goals (lambda_away)
    """
    home_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", PoissonRegressor(alpha=0.1, max_iter=1000)),
    ])
    home_model.fit(X_train, y_train_home)

    away_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", PoissonRegressor(alpha=0.1, max_iter=1000)),
    ])
    away_model.fit(X_train, y_train_away)

    return home_model, away_model


def evaluate_logistic(model, X_test, y_test, label: str = "Logistic Regression"):
    """Print standard classification metrics for a 1X2 model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"\n=== {label} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Home Win", "Draw", "Away Win"],
        zero_division=0,
    ))

    return acc, ll


def save_models(models: dict, folder: Path = MODELS_DIR) -> None:
    """Save trained models as .pkl files in the given folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = folder / f"{name}.pkl"
        joblib.dump(model, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    print("1. Loading features...")
    df = load_features_data()
    print(f"   Loaded {len(df)} matches.")

    print("\n2. Adding target column...")
    df = add_target_column(df)
    print("   Result distribution:")
    print(df["result"].value_counts().sort_index().to_string())

    print("\n3. Splitting temporally...")
    train_df, test_df = temporal_split(df, TRAIN_SEASONS, TEST_SEASONS)
    print(f"   Train: {len(train_df)} matches (seasons {TRAIN_SEASONS})")
    print(f"   Test:  {len(test_df)} matches (seasons {TEST_SEASONS})")

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    y_train = train_df["result"]
    y_test = test_df["result"]

    print("\n4. Training Logistic Regression...")
    logreg = train_logistic_regression(X_train, y_train)
    evaluate_logistic(logreg, X_test, y_test, label="Logistic Regression (test set)")

    print("\n5. Training Poisson models...")
    poisson_home, poisson_away = train_poisson_models(
        X_train,
        y_train_home=train_df["home_goals"],
        y_train_away=train_df["away_goals"],
    )
    print("   Both Poisson models trained.")

    print("\n6. Saving models...")
    save_models({
        "logistic_regression": logreg,
        "poisson_home": poisson_home,
        "poisson_away": poisson_away,
    })

    print("\n✅ Done.")