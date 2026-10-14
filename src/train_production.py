"""
Train the "production" models using ALL available seasons.

Unlike src/train.py, which holds out 2025/26 for evaluation,
this script uses every season to produce the best possible model
for predicting the current (2026/27) season.

Usage:
    python src/train_production.py
"""

from train import (
    FEATURE_COLS,
    load_features_data,
    add_target_column,
    train_logistic_regression,
    train_poisson_models,
    save_models,
)


# All seasons available in our dataset
ALL_SEASONS = ["2122", "2223", "2324", "2425", "2526"]


if __name__ == "__main__":
    print("1. Loading features...")
    df = load_features_data()
    print(f"   Loaded {len(df)} matches.")

    print("2. Adding target column...")
    df = add_target_column(df)

    # Filter to only the seasons we want to train on
    df = df[df["season"].astype(str).isin(ALL_SEASONS)].copy()
    print(f"   Training on {len(df)} matches from seasons {ALL_SEASONS}")

    X = df[FEATURE_COLS]
    y = df["result"]

    print("\n3. Training Logistic Regression...")
    logreg = train_logistic_regression(X, y)

    print("4. Training Poisson models...")
    poisson_home, poisson_away = train_poisson_models(
        X,
        y_train_home=df["home_goals"],
        y_train_away=df["away_goals"],
    )

    print("\n5. Saving production models...")
    save_models({
        "logistic_regression_prod": logreg,
        "poisson_home_prod": poisson_home,
        "poisson_away_prod": poisson_away,
    })

    print("\n✅ Production models saved.")