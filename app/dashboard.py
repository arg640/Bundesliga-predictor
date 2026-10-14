"""
Streamlit dashboard for Bundesliga match outcome prediction.

Three tabs:
1. Individual Prediction: pick a historical match, see model predictions.
2. Season Simulation 2025/26: Monte Carlo simulation using evaluation models,
   compared against the actual final standings.
3. 2026/27 Forecast: Monte Carlo simulation using production models,
   with priors for promoted teams.

Usage:
    streamlit run app/dashboard.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
from scipy.stats import poisson as poisson_dist


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --- Constants ---
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
MAX_GOALS = 6
RESULT_LABELS = ["Home Win", "Draw", "Away Win"]

# ============================================================
# EDIT THIS: Promoted teams for 2026/27
# ============================================================
PROMOTED_2026_27 = []  # e.g. ["FC Schalke 04", "Hamburger SV", "1. FC Köln"]
RELEGATED_2026_27 = []  # e.g. ["Holstein Kiel", "VfL Bochum", "FC St. Pauli"]


# ============================================================
# Resource loading
# ============================================================
@st.cache_resource
def load_models():
    """Load both evaluation and production models."""
    return {
        # Trained on 2021/22 - 2024/25 (never saw 2025/26)
        "eval_logreg": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        "eval_poisson_home": joblib.load(MODELS_DIR / "poisson_home.pkl"),
        "eval_poisson_away": joblib.load(MODELS_DIR / "poisson_away.pkl"),
        # Trained on 2021/22 - 2025/26 (all data)
        "prod_logreg": joblib.load(MODELS_DIR / "logistic_regression_prod.pkl"),
        "prod_poisson_home": joblib.load(MODELS_DIR / "poisson_home_prod.pkl"),
        "prod_poisson_away": joblib.load(MODELS_DIR / "poisson_away_prod.pkl"),
    }


@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["result"] = np.select(
        [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] == df["away_goals"],
            df["home_goals"] < df["away_goals"],
        ],
        [0, 1, 2],
        default=-1,
    )
    return df


# ============================================================
# Prediction helpers
# ============================================================
def poisson_1x2_probs(lambda_home: float, lambda_away: float):
    """Return (P(home), P(draw), P(away)) from two Poisson lambdas."""
    home_probs = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lambda_home)
    away_probs = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lambda_away)
    score_matrix = np.outer(home_probs, away_probs)
    p_home = np.tril(score_matrix, -1).sum()
    p_draw = np.trace(score_matrix)
    p_away = np.triu(score_matrix, 1).sum()
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def predict_match(logreg, poisson_home, poisson_away, features_row):
    X = features_row[FEATURE_COLS].to_frame().T.astype(float)
    logreg_probs = logreg.predict_proba(X)[0]
    lh = poisson_home.predict(X)[0]
    la = poisson_away.predict(X)[0]
    poisson_probs = np.array(poisson_1x2_probs(lh, la))
    return {
        "logreg_probs": logreg_probs,
        "poisson_probs": poisson_probs,
        "lambda_home": lh,
        "lambda_away": la,
    }


# ============================================================
# Simulation helpers
# ============================================================
def extract_team_features(df: pd.DataFrame, season: str, position: str = "last") -> pd.DataFrame:
    """
    Extract ONE feature vector per team from a given season.

    position="first" -> form going INTO the season (pre-season prior)
    position="last"  -> form at the END of the season
    """
    season_df = df[df["season"].astype(str) == season].sort_values("date").copy()
    teams = sorted(set(season_df["home_team"]) | set(season_df["away_team"]))
    idx = 0 if position == "first" else -1

    rows = {}
    for team in teams:
        home_matches = season_df[season_df["home_team"] == team]
        away_matches = season_df[season_df["away_team"] == team]
        all_matches = pd.concat([home_matches, away_matches]).sort_values("date")

        if all_matches.empty:
            continue

        general = all_matches.iloc[idx]
        is_home = general["home_team"] == team

        xg_for = general["home_xg_for_last_5"] if is_home else general["away_xg_for_last_5"]
        xg_against = general["home_xg_against_last_5"] if is_home else general["away_xg_against_last_5"]

        if not home_matches.empty:
            hr = home_matches.sort_values("date").iloc[idx]
            xg_for_h = hr["home_xg_for_last_5_at_home"]
            xg_ag_h = hr["home_xg_against_last_5_at_home"]
        else:
            xg_for_h, xg_ag_h = xg_for, xg_against

        if not away_matches.empty:
            ar = away_matches.sort_values("date").iloc[idx]
            xg_for_a = ar["away_xg_for_last_5_away"]
            xg_ag_a = ar["away_xg_against_last_5_away"]
        else:
            xg_for_a, xg_ag_a = xg_for, xg_against

        rows[team] = {
            "xg_for_last_5": xg_for,
            "xg_against_last_5": xg_against,
            "xg_for_last_5_at_home": xg_for_h,
            "xg_against_last_5_at_home": xg_ag_h,
            "xg_for_last_5_at_away": xg_for_a,
            "xg_against_last_5_at_away": xg_ag_a,
        }

    return pd.DataFrame(rows).T


def build_fixtures(teams: list) -> pd.DataFrame:
    """Round-robin fixtures: each pair plays home and away."""
    rows = []
    for home in teams:
        for away in teams:
            if home != away:
                rows.append({"home_team": home, "away_team": away})
    return pd.DataFrame(rows)


def fixtures_to_features(fixtures: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, fix in fixtures.iterrows():
        h, a = fix["home_team"], fix["away_team"]
        rows.append({
            "home_xg_for_last_5": team_features.loc[h, "xg_for_last_5"],
            "home_xg_against_last_5": team_features.loc[h, "xg_against_last_5"],
            "home_xg_for_last_5_at_home": team_features.loc[h, "xg_for_last_5_at_home"],
            "home_xg_against_last_5_at_home": team_features.loc[h, "xg_against_last_5_at_home"],
            "away_xg_for_last_5": team_features.loc[a, "xg_for_last_5"],
            "away_xg_against_last_5": team_features.loc[a, "xg_against_last_5"],
            "away_xg_for_last_5_away": team_features.loc[a, "xg_for_last_5_at_away"],
            "away_xg_against_last_5_away": team_features.loc[a, "xg_against_last_5_at_away"],
        })
    return pd.DataFrame(rows)[FEATURE_COLS]


def simulate_season(fixtures, fixture_features, poisson_home, poisson_away, n_sims=1000, seed=42):
    """Monte Carlo season simulation."""
    np.random.seed(seed)

    lambda_home = poisson_home.predict(fixture_features)
    lambda_away = poisson_away.predict(fixture_features)

    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    n_teams = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    home_idx = np.array([team_to_idx[h] for h in fixtures["home_team"]])
    away_idx = np.array([team_to_idx[a] for a in fixtures["away_team"]])

    points_matrix = np.zeros((n_sims, n_teams))
    positions_matrix = np.zeros((n_sims, n_teams))

    for sim in range(n_sims):
        hg = np.random.poisson(lambda_home)
        ag = np.random.poisson(lambda_away)

        pts = np.zeros(n_teams)
        gd = np.zeros(n_teams)

        home_wins = hg > ag
        draws = hg == ag
        away_wins = ag > hg

        np.add.at(pts, home_idx[home_wins], 3)
        np.add.at(pts, away_idx[away_wins], 3)
        np.add.at(pts, home_idx[draws], 1)
        np.add.at(pts, away_idx[draws], 1)

        goal_diff = hg - ag
        np.add.at(gd, home_idx, goal_diff)
        np.add.at(gd, away_idx, -goal_diff)

        order = np.lexsort((gd, pts))[::-1]
        positions = np.empty(n_teams, dtype=int)
        positions[order] = np.arange(1, n_teams + 1)

        points_matrix[sim] = pts
        positions_matrix[sim] = positions

    results = pd.DataFrame({
        "team": teams,
        "expected_points": points_matrix.mean(axis=0),
        "expected_position": positions_matrix.mean(axis=0),
        "p_champion": (positions_matrix == 1).mean(axis=0),
        "p_top4": (positions_matrix <= 4).mean(axis=0),
        "p_relegation": (positions_matrix >= 17).mean(axis=0),
    })
    return results.sort_values("expected_position").reset_index(drop=True)


def compute_real_standings(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Compute the actual final standings for a season."""
    s = df[df["season"].astype(str) == season].copy()
    teams = sorted(set(s["home_team"]) | set(s["away_team"]))
    points = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}

    for _, row in s.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        gd[h] += hg - ag
        gd[a] += ag - hg
        if hg > ag:
            points[h] += 3
        elif hg == ag:
            points[h] += 1
            points[a] += 1
        else:
            points[a] += 3

    standings = pd.DataFrame({
        "team": teams,
        "actual_points": [points[t] for t in teams],
        "actual_gd": [gd[t] for t in teams],
    }).sort_values(["actual_points", "actual_gd"], ascending=[False, False]).reset_index(drop=True)

    standings["actual_position"] = standings.index + 1
    return standings[["team", "actual_points", "actual_position"]]


# ============================================================
# Plot helpers
# ============================================================
def plot_1x2_bars(logreg_probs, poisson_probs):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, logreg_probs, w, label="Logistic Regression", color="#3b82f6")
    ax.bar(x + w / 2, poisson_probs, w, label="Poisson", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(RESULT_LABELS)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    for i, (l, p) in enumerate(zip(logreg_probs, poisson_probs)):
        ax.text(i - w / 2, l + 0.02, f"{l:.1%}", ha="center", fontsize=9)
        ax.text(i + w / 2, p + 0.02, f"{p:.1%}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_score_heatmap(lambda_home, lambda_away):
    home_probs = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lambda_home)
    away_probs = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lambda_away)
    matrix = np.outer(home_probs, away_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt=".1%", cmap="YlOrRd",
        xticklabels=range(MAX_GOALS + 1),
        yticklabels=range(MAX_GOALS + 1),
        cbar_kws={"label": "Probability"},
        ax=ax, annot_kws={"size": 9},
    )
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    return fig


def plot_standings_comparison(results: pd.DataFrame, has_actual: bool = True):
    """Bar chart: expected points (and actual points if available)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    teams_sorted = results["team"].tolist()
    y = np.arange(len(teams_sorted))

    if has_actual and "actual_points" in results.columns:
        ax.barh(y + 0.2, results["actual_points"], 0.4, label="Actual", color="#10b981")
        ax.barh(y - 0.2, results["expected_points"], 0.4, label="Expected", color="#3b82f6")
    else:
        ax.barh(y, results["expected_points"], 0.6, color="#3b82f6")

    ax.set_yticks(y)
    ax.set_yticklabels(teams_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Points")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    return fig


# ============================================================
# App layout
# ============================================================
st.set_page_config(page_title="Bundesliga Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Bundesliga Match Outcome Predictor")
st.caption(
    "Predictions based on Expected Goals (xG) and recent team form. "
    "Trained on Bundesliga seasons 2021/22 – 2025/26."
)

models = load_models()
df = load_data()

tab1, tab2, tab3 = st.tabs([
    "🎯 Individual Prediction",
    "🎲 Season Simulation 2025/26",
    "🔮 2026/27 Forecast",
])


# ============================================================
# TAB 1 — Individual prediction
# ============================================================
with tab1:
    st.subheader("Pick a match to predict")

    col1, col2, col3 = st.columns(3)
    seasons = sorted(df["season"].astype(str).unique())

    with col1:
        season = st.selectbox("Season", seasons, index=len(seasons) - 1)

    season_df = df[df["season"].astype(str) == season]
    teams = sorted(season_df["home_team"].unique())

    with col2:
        home_team = st.selectbox("Home team", teams)
    with col3:
        away_team = st.selectbox("Away team", [t for t in teams if t != home_team])

    match = season_df[
        (season_df["home_team"] == home_team) & (season_df["away_team"] == away_team)
    ]

    if match.empty:
        st.warning(f"No match found: {home_team} vs {away_team} in {season}.")
    else:
        row = match.iloc[0]
        st.markdown(
            f"**Match:** {home_team} vs {away_team} — {row['date'].strftime('%Y-%m-%d')}"
        )
        st.markdown(
            f"**Actual result:** {int(row['home_goals'])} – {int(row['away_goals'])} "
            f"({RESULT_LABELS[int(row['result'])]})"
        )
        st.divider()

        preds = predict_match(
            models["prod_logreg"], models["prod_poisson_home"], models["prod_poisson_away"], row
        )

        cA, cB = st.columns(2)
        with cA:
            st.markdown("### Logistic Regression")
            for label, p in zip(RESULT_LABELS, preds["logreg_probs"]):
                st.metric(label, f"{p:.1%}")
        with cB:
            st.markdown("### Poisson")
            for label, p in zip(RESULT_LABELS, preds["poisson_probs"]):
                st.metric(label, f"{p:.1%}")
            st.caption(
                f"Expected goals — Home: **{preds['lambda_home']:.2f}** | "
                f"Away: **{preds['lambda_away']:.2f}**"
            )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1X2 probability comparison")
            st.pyplot(plot_1x2_bars(preds["logreg_probs"], preds["poisson_probs"]))
        with c2:
            st.markdown("#### Score probability matrix (Poisson)")
            st.pyplot(plot_score_heatmap(preds["lambda_home"], preds["lambda_away"]))


# ============================================================
# TAB 2 — Season Simulation 2025/26
# ============================================================
with tab2:
    st.subheader("Season Simulation — 2025/26")
    st.caption(
        "Monte Carlo simulation using the **evaluation models** "
        "(trained on 2021/22–2024/25, never saw 2025/26). "
        "Compares simulated table vs actual final standings."
    )

    if st.button("🎲 Simulate 1000 seasons", key="btn_sim_2526"):
        with st.spinner("Simulating 1000 seasons..."):
            team_features = extract_team_features(df, "2526", position="first")
            real_fixtures = df[df["season"].astype(str) == "2526"][
                ["home_team", "away_team"]
            ].copy()
            fix_feats = fixtures_to_features(real_fixtures, team_features)

            results = simulate_season(
                real_fixtures, fix_feats,
                models["eval_poisson_home"], models["eval_poisson_away"],
                n_sims=1000,
            )
            real = compute_real_standings(df, "2526")
            results = results.merge(real, on="team", how="left")
            st.session_state["data_sim_2526"] = results

    if "data_sim_2526" in st.session_state:
        res = st.session_state["data_sim_2526"]
        display = res[[
            "team", "expected_points", "actual_points",
            "expected_position", "actual_position",
            "p_champion", "p_top4", "p_relegation",
        ]].copy()
        display["expected_points"] = display["expected_points"].round(1)
        display["expected_position"] = display["expected_position"].round(1)
        for c in ["p_champion", "p_top4", "p_relegation"]:
            display[c] = (display[c] * 100).round(1).astype(str) + "%"
        display.columns = [
            "Team", "Exp Pts", "Real Pts", "Exp Pos", "Real Pos",
            "P(Champion)", "P(Top 4)", "P(Relegation)",
        ]

        st.markdown("### Simulated vs Actual Standings")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("### Points comparison")
        st.pyplot(plot_standings_comparison(res, has_actual=True))


# ============================================================
# TAB 3 — 2026/27 Forecast
# ============================================================
with tab3:
    st.subheader("2026/27 Season Forecast")

    st.warning(
        "**⚠️ Pre-season forecast based on 2021/22 – 2025/26 data**\n\n"
        "This simulation uses **end-of-2025/26 form as a proxy** for 2026/27 team strength. "
        "It does **NOT** account for: summer transfers, managerial changes, "
        "actual 2026/27 form (only 5 matchdays played), or promoted team signings. "
        "Promoted teams receive priors based on the **average of the bottom 3 teams** "
        "from 2025/26. **Expect high inaccuracy.** Treat this as a pre-season forecast, "
        "not a live prediction."
    )

    if st.button("🔮 Simulate 1000 seasons", key="btn_sim_2627"):
        with st.spinner("Simulating 1000 seasons..."):
            # Teams for 2026/27
            teams_2025_26 = sorted(
                set(df[df["season"].astype(str) == "2526"]["home_team"])
                | set(df[df["season"].astype(str) == "2526"]["away_team"])
            )

            if RELEGATED_2026_27 or PROMOTED_2026_27:
                teams_2026_27 = [
                    t for t in teams_2025_26 if t not in RELEGATED_2026_27
                ] + PROMOTED_2026_27
            else:
                teams_2026_27 = teams_2025_26

            # Team features: end-of-2025/26 form
            team_features = extract_team_features(df, "2526", position="last")

            # Prior for promoted teams: bottom-3 average from 2025/26
            real_standings = compute_real_standings(df, "2526")
            bottom3 = real_standings.tail(3)["team"].tolist()
            bottom3_features = team_features.loc[bottom3].mean()

            for team in teams_2026_27:
                if team not in team_features.index:
                    team_features.loc[team] = bottom3_features

            # Keep only 2026/27 teams
            team_features = team_features.loc[teams_2026_27]

            # Round-robin fixtures
            fixtures = build_fixtures(teams_2026_27)
            fix_feats = fixtures_to_features(fixtures, team_features)

            results = simulate_season(
                fixtures, fix_feats,
                models["prod_poisson_home"], models["prod_poisson_away"],
                n_sims=1000,
            )
            st.session_state["data_sim_2627"] = results

    if "data_sim_2627" in st.session_state:
        res = st.session_state["data_sim_2627"]
        display = res[[
            "team", "expected_points", "expected_position",
            "p_champion", "p_top4", "p_relegation",
        ]].copy()
        display["expected_points"] = display["expected_points"].round(1)
        display["expected_position"] = display["expected_position"].round(1)
        for c in ["p_champion", "p_top4", "p_relegation"]:
            display[c] = (display[c] * 100).round(1).astype(str) + "%"
        display.columns = [
            "Team", "Expected Pts", "Expected Pos",
            "P(Champion)", "P(Top 4)", "P(Relegation)",
        ]

        st.markdown("### Forecast Standings")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("### Expected points by team")
        st.pyplot(plot_standings_comparison(res, has_actual=False))