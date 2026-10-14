"""
Evaluation module: compare Logistic Regression vs Poisson on the test set.

For the Poisson approach:
1. Predict lambda_home and lambda_away for each match.
2. Compute score probability matrix (P(home=i) * P(away=j)).
3. Sum to get P(home win), P(draw), P(away win).
4. Evaluate with the same metrics as Logistic Regression.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson as poisson_dist
from sklearn.metrics import accuracy_score, log_loss, classification_report

# Reuse from train.py
from train import (
    FEATURE_COLS,
    TRAIN_SEASONS,
    TEST_SEASONS,
    load_features_data,
    add_target_column,
    temporal_split,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MAX_GOALS = 10  # Upper bound for the score grid


def load_trained_models() -> dict:
    """Load the three trained models from disk."""
    return {
        "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        "poisson_home": joblib.load(MODELS_DIR / "poisson_home.pkl"),
        "poisson_away": joblib.load(MODELS_DIR / "poisson_away.pkl"),
    }


def poisson_1x2_probabilities(lambda_home: float, lambda_away: float) -> tuple:
    """
    Compute (P(home win), P(draw), P(away win)) from two Poisson lambdas.

    Uses a score grid from 0 to MAX_GOALS for each team.
    """
    # Probability of each goal count: P(X = k) for k = 0..MAX_GOALS
    home_goal_probs = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lambda_home)
    away_goal_probs = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lambda_away)

    # Score matrix: P(home_goals=i, away_goals=j) = P_home(i) * P_away(j)
    score_matrix = np.outer(home_goal_probs, away_goal_probs)

    # Sum over regions of the matrix
    # Home win: strictly above diagonal (i > j)
    prob_home_win = np.tril(score_matrix, -1).sum()  # tril with -1 = below diagonal
    # Wait, let's be careful: 
    #   score_matrix[i, j] = P(home=i, away=j)
    #   Home win means i > j, i.e., row > column, i.e., BELOW the diagonal in standard indexing
    #   np.tril(matrix, -1) keeps entries where row > col
    # Draw: i == j, diagonal
    prob_draw = np.trace(score_matrix)
    # Away win: i < j, above diagonal
    prob_away_win = np.triu(score_matrix, 1).sum()

    # Normalize (in case of truncation at MAX_GOALS)
    total = prob_home_win + prob_draw + prob_away_win
    return (
        prob_home_win / total,
        prob_draw / total,
        prob_away_win / total,
    )


def evaluate_poisson(model_home, model_away, X_test: pd.DataFrame) -> np.ndarray:
    """
    Apply the Poisson pipeline to the test set.

    Returns an (N, 3) array with columns [P(home), P(draw), P(away)].
    """
    lambda_home = model_home.predict(X_test)
    lambda_away = model_away.predict(X_test)

    probs = np.zeros((len(X_test), 3))
    for i, (lh, la) in enumerate(zip(lambda_home, lambda_away)):
        probs[i] = poisson_1x2_probabilities(lh, la)
    return probs


def compare_models(y_test, logreg_probs, poisson_probs):
    """Print a side-by-side comparison of the two approaches."""
    # 1X2 from probability arrays: argmax
    logreg_pred = logreg_probs.argmax(axis=1)
    poisson_pred = poisson_probs.argmax(axis=1)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON ON TEST SET")
    print("=" * 60)

    for name, pred, probs in [
        ("Logistic Regression", logreg_pred, logreg_probs),
        ("Poisson", poisson_pred, poisson_probs),
    ]:
        acc = accuracy_score(y_test, pred)
        ll = log_loss(y_test, probs)
        print(f"\n--- {name} ---")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Log Loss: {ll:.4f}")
        print("  Classification report:")
        print(classification_report(
            y_test, pred,
            target_names=["Home Win", "Draw", "Away Win"],
            zero_division=0,
            digits=3,
        ))

    # How often each model predicts each class
    print("\n--- Prediction distribution ---")
    for name, pred in [("LogReg", logreg_pred), ("Poisson", poisson_pred)]:
        counts = pd.Series(pred).value_counts().sort_index()
        print(f"{name}: " + ", ".join(
            f"{['Home','Draw','Away'][k]}: {v}" for k, v in counts.items()
        ))

    # Actual distribution
    counts = pd.Series(y_test).value_counts().sort_index()
    print("Actual: " + ", ".join(
        f"{['Home','Draw','Away'][k]}: {v}" for k, v in counts.items()
    ))


if __name__ == "__main__":
    print("1. Loading data and models...")
    df = load_features_data()
    df = add_target_column(df)
    train_df, test_df = temporal_split(df, TRAIN_SEASONS, TEST_SEASONS)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["result"].values

    models = load_trained_models()

    print("\n2. Predicting with Logistic Regression...")
    logreg_probs = models["logistic_regression"].predict_proba(X_test)

    print("3. Predicting with Poisson...")
    poisson_probs = evaluate_poisson(
        models["poisson_home"],
        models["poisson_away"],
        X_test,
    )

    print("\n4. Comparing models...")
    compare_models(y_test, logreg_probs, poisson_probs)

    # Show a couple of examples
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (first 5 test matches)")
    print("=" * 60)
    sample = test_df.head(5).copy()
    sample["logreg_home"] = logreg_probs[:5, 0]
    sample["logreg_draw"] = logreg_probs[:5, 1]
    sample["logreg_away"] = logreg_probs[:5, 2]
    sample["poisson_home"] = poisson_probs[:5, 0]
    sample["poisson_draw"] = poisson_probs[:5, 1]
    sample["poisson_away"] = poisson_probs[:5, 2]
    cols = [
        "home_team", "away_team", "home_goals", "away_goals",
        "logreg_home", "logreg_draw", "logreg_away",
        "poisson_home", "poisson_draw", "poisson_away",
    ]
    print(sample[cols].to_string(index=False))