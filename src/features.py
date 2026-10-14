"""
Feature engineering for Bundesliga match prediction.

Transforms the raw match-level data into a DataFrame ready for modeling
by adding rolling statistics (recent form) for each team.

Usage:
    python src/features.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW = 5  # Number of recent matches to average over


def load_raw_data(filename: str = "bundesliga_matches.csv") -> pd.DataFrame:
    """Load the raw Bundesliga match data from CSV."""
    path = RAW_DATA_DIR / filename
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_team_match_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the match-level DataFrame into a team-level view.

    Each match generates two rows: one from the home team's perspective
    and one from the away team's perspective. This makes it possible to
    compute rolling statistics per team (since teams alternate between
    home and away columns in the raw data).

    Args:
        df: Raw match-level DataFrame.

    Returns:
        Long-format DataFrame with one row per (team, match).
    """
    home_view = pd.DataFrame({
        "match_id": df["game"],
        "date": df["date"],
        "season": df["season"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "is_home": True,
        "xg_for": df["home_xg"],
        "xg_against": df["away_xg"],
        "goals_for": df["home_goals"],
        "goals_against": df["away_goals"],
    })

    away_view = pd.DataFrame({
        "match_id": df["game"],
        "date": df["date"],
        "season": df["season"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "is_home": False,
        "xg_for": df["away_xg"],
        "xg_against": df["home_xg"],
        "goals_for": df["away_goals"],
        "goals_against": df["home_goals"],
    })

    team_df = pd.concat([home_view, away_view], ignore_index=True)
    team_df = team_df.sort_values(["team", "date"]).reset_index(drop=True)

    return team_df


def _rolling_mean(df, group_col, value_col, window):
    """
    Compute rolling mean of `value_col` per group, using only PAST matches.

    The shift(1) ensures we exclude the current match (avoids data leakage).
    """
    return (
        df.groupby(group_col)[value_col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )


def add_team_features(team_df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Add rolling features to the team-level DataFrame.

    Features added:
    - xg_for_last_{window}, xg_against_last_{window}
        Rolling average of xG over the last N matches (home + away mixed).
    - xg_for_last_{window}_at_home, xg_against_last_{window}_at_home
        Rolling average over the last N matches PLAYED AT HOME.
    - xg_for_last_{window}_at_away, xg_against_last_{window}_at_away
        Rolling average over the last N matches PLAYED AWAY.
    - match_number_in_season
        How many matches the team has played in the current season (1-indexed).
    """
    team_df = team_df.sort_values(["team", "date"]).reset_index(drop=True).copy()

    # --- General rolling features (all matches) ---
    for col in ["xg_for", "xg_against"]:
        team_df[f"{col}_last_{window}"] = _rolling_mean(team_df, "team", col, window)

    # --- Home-specific rolling features ---
    # We compute the rolling average only using matches where the team played at home.
    # Initialized with NaN, then filled only for home matches.
    is_home = team_df["is_home"]
    for col in ["xg_for", "xg_against"]:
        team_df[f"{col}_last_{window}_at_home"] = np.nan
        home_slice = team_df[is_home]
        team_df.loc[is_home, f"{col}_last_{window}_at_home"] = _rolling_mean(
            home_slice, "team", col, window
        )

    # --- Away-specific rolling features ---
    is_away = ~team_df["is_home"]
    for col in ["xg_for", "xg_against"]:
        team_df[f"{col}_last_{window}_at_away"] = np.nan
        away_slice = team_df[is_away]
        team_df.loc[is_away, f"{col}_last_{window}_at_away"] = _rolling_mean(
            away_slice, "team", col, window
        )

    # --- Match count within the season ---
    # cumcount() returns 0-indexed count within each (team, season) group, so we add 1.
    team_df["match_number_in_season"] = (
        team_df.groupby(["team", "season"]).cumcount() + 1
    )

    return team_df


def merge_features_back(
    matches_df: pd.DataFrame, team_df: pd.DataFrame, window: int = WINDOW
) -> pd.DataFrame:
    """
    Merge team-level rolling features back into the match-level DataFrame.

    Each match gets:
    - home_* features from the home team's row
    - away_* features from the away team's row
    """
    matches_df = matches_df.copy()
    matches_df["match_id"] = matches_df["game"]

    # --- Home team features ---
    home_feats = team_df[team_df["is_home"]][[
        "match_id",
        f"xg_for_last_{window}",
        f"xg_against_last_{window}",
        f"xg_for_last_{window}_at_home",
        f"xg_against_last_{window}_at_home",
        "match_number_in_season",
    ]].rename(columns={
        f"xg_for_last_{window}": "home_xg_for_last_5",
        f"xg_against_last_{window}": "home_xg_against_last_5",
        f"xg_for_last_{window}_at_home": "home_xg_for_last_5_at_home",
        f"xg_against_last_{window}_at_home": "home_xg_against_last_5_at_home",
        "match_number_in_season": "matches_home_in_season",
    })

    # --- Away team features ---
    away_feats = team_df[~team_df["is_home"]][[
        "match_id",
        f"xg_for_last_{window}",
        f"xg_against_last_{window}",
        f"xg_for_last_{window}_at_away",
        f"xg_against_last_{window}_at_away",
        "match_number_in_season",
    ]].rename(columns={
        f"xg_for_last_{window}": "away_xg_for_last_5",
        f"xg_against_last_{window}": "away_xg_against_last_5",
        f"xg_for_last_{window}_at_away": "away_xg_for_last_5_away",
        f"xg_against_last_{window}_at_away": "away_xg_against_last_5_away",
        "match_number_in_season": "matches_away_in_season",
    })

    matches_df = matches_df.merge(home_feats, on="match_id", how="left")
    matches_df = matches_df.merge(away_feats, on="match_id", how="left")
    matches_df = matches_df.drop(columns=["match_id"])

    return matches_df


def fill_missing_with_league_average(
    df: pd.DataFrame, feature_cols: list
) -> pd.DataFrame:
    """
    Fill NaN values in rolling features with the global mean.

    This handles:
    - First match of the season (no prior matches to compute a rolling mean)
    - Newly promoted teams with no top-flight history in the dataset

    Note: a more sophisticated approach could use a team-specific prior
    (e.g., last known form from a previous season). We use the global mean
    for simplicity.
    """
    df = df.copy()
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].mean())
    return df


def save_processed_data(df: pd.DataFrame, filename: str = "features.csv") -> Path:
    """Save the feature-enriched DataFrame to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def build_features() -> pd.DataFrame:
    """Full pipeline: raw data -> feature-enriched DataFrame."""
    print("1. Loading raw data...")
    matches_df = load_raw_data()
    print(f"   Loaded {len(matches_df)} matches.")

    print("2. Building team-level view...")
    team_df = build_team_match_view(matches_df)
    print(f"   Built {len(team_df)} team-match rows.")

    print("3. Computing rolling features...")
    team_df = add_team_features(team_df, window=WINDOW)

    print("4. Merging features back to match level...")
    features_df = merge_features_back(matches_df, team_df, window=WINDOW)

    print("5. Filling missing values with league average...")
    feature_cols = [
        "home_xg_for_last_5", "home_xg_against_last_5",
        "home_xg_for_last_5_at_home", "home_xg_against_last_5_at_home",
        "away_xg_for_last_5", "away_xg_against_last_5",
        "away_xg_for_last_5_away", "away_xg_against_last_5_away",
    ]
    features_df = fill_missing_with_league_average(features_df, feature_cols)

    print(f"   Final shape: {features_df.shape}")
    return features_df


if __name__ == "__main__":
    features_df = build_features()
    path = save_processed_data(features_df)
    print(f"\n✅ Saved to {path}")
    print("\nNew feature columns:")
    for col in features_df.columns:
        if "last_5" in col or "matches_" in col:
            print(f"  - {col}")